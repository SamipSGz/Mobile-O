import Foundation
import CoreML
import MLX
import MLXLMCommon
import MLXVLM
import Tokenizers
import Metal

import UIKit

/// Observable wrapper for `MobileOGenerator` that manages model loading, inference lifecycle,
/// and MLMultiArray → UIImage conversion via Metal.
@Observable
@MainActor
class MobileOModel {

    public var running = false
    public var modelInfo = ""
    public var generationTime: String = ""

    public var inferenceTime: TimeInterval = 0
    public var lastTiming: MobileOGenerator.TimingInfo? = nil
    public var currentStep: Int = 0
    public var totalSteps: Int = 0
    public let modelVariant: MobileOGenerator.ModelVariant = .fp32

    public var schedulerType: MobileOGenerator.SchedulerType = .dpmSolver {
        didSet {
            if oldValue != schedulerType {
                UserDefaults.standard.set(schedulerType.rawValue, forKey: "schedulerType")
            }
        }
    }

    enum EvaluationState: String, CaseIterable {
        case idle = "Idle"
        case processingPrompt = "Processing Prompt"
        case generatingImage = "Generating Image"
    }

    public var evaluationState = EvaluationState.idle

    private var generator: MobileOGenerator?
    private var currentTask: Task<PlatformImage?, Never>?
    private weak var sharedFastVLM: FastVLM?
    private var sharedTokenizer: Tokenizer?
    private let modelDirectory: URL?

    nonisolated(unsafe) private let metalDevice: MTLDevice?
    nonisolated(unsafe) private let metalCommandQueue: MTLCommandQueue?
    nonisolated(unsafe) private var floatToRGBAPipeline: MTLComputePipelineState?

    public init(sharedFastVLM: FastVLM? = nil, sharedTokenizer: Tokenizer? = nil, modelDirectory: URL? = nil) {
        self.sharedFastVLM = sharedFastVLM
        self.sharedTokenizer = sharedTokenizer
        self.modelDirectory = modelDirectory

        if let savedSchedulerRaw = UserDefaults.standard.string(forKey: "schedulerType"),
           let savedScheduler = MobileOGenerator.SchedulerType(rawValue: savedSchedulerRaw) {
            self.schedulerType = savedScheduler
        }

        self.metalDevice = MTLCreateSystemDefaultDevice()
        self.metalCommandQueue = metalDevice?.makeCommandQueue()

        if let device = metalDevice {
            do {
                guard let library = device.makeDefaultLibrary() else {
                    print("MobileOModel WARNING: Could not load Metal library")
                    return
                }

                if let f = library.makeFunction(name: "floatToRGBA") {
                    floatToRGBAPipeline = try device.makeComputePipelineState(function: f)
                }
            } catch {
                print("MobileOModel WARNING: Failed to create Metal pipeline: \(error)")
            }
        }
    }

    /// Load all models using the current variant and scheduler type.
    public func load() async {
        await loadWithVariant(modelVariant, schedulerType: schedulerType)
    }

    /// Load (or hot-swap) a specific model variant, optionally changing the scheduler.
    public func loadWithVariant(_ variant: MobileOGenerator.ModelVariant, schedulerType: MobileOGenerator.SchedulerType? = nil) async {
        if let schedulerType = schedulerType {
            self.schedulerType = schedulerType
        }

        if let sana = generator, sharedFastVLM != nil, sharedTokenizer != nil {
            modelInfo = "Loading \(modelVariant.rawValue)..."
            do {
                try await sana.loadDiT(variant: modelVariant)
                modelInfo = "Ready to generate (\(modelVariant.rawValue))"
                } catch {
                modelInfo = "Error: \(error.localizedDescription)"
            }
        } else {
            await loadModels(variant: modelVariant, schedulerType: self.schedulerType)
        }
    }

    /// Hot-swap the scheduler without reloading the transformer or VAE.
    public func loadWithScheduler(_ schedulerType: MobileOGenerator.SchedulerType) async {
        self.schedulerType = schedulerType

        if let sana = generator {
            modelInfo = "Loading \(schedulerType.rawValue)..."
            do {
                try await sana.loadScheduler(schedulerType: schedulerType)
                modelInfo = "Ready to generate (\(modelVariant.rawValue), \(schedulerType.rawValue))"
            } catch {
                modelInfo = "Error: \(error.localizedDescription)"
            }
        } else {
            await loadModels(variant: modelVariant, schedulerType: schedulerType)
        }
    }

    private func loadModels(variant: MobileOGenerator.ModelVariant, schedulerType: MobileOGenerator.SchedulerType) async {
        do {
            modelInfo = "Loading models..."

            guard let fastVLM = sharedFastVLM, let tokenizer = sharedTokenizer else {
                throw NSError(domain: "MobileOModel", code: -1,
                             userInfo: [NSLocalizedDescriptionKey: "Shared FastVLM not available. Must initialize with shared instance."])
            }

            modelInfo = "Loading SANA models (\(variant.rawValue), \(schedulerType.rawValue))..."

            guard let modelDir = modelDirectory else {
                throw NSError(domain: "MobileOModel", code: -1,
                             userInfo: [NSLocalizedDescriptionKey: "Model directory not set."])
            }
            let sana = MobileOGenerator(fastVLM: fastVLM, tokenizer: tokenizer, modelDirectory: modelDir)
            try await sana.loadModels(variant: variant, schedulerType: schedulerType)

            self.generator = sana
            modelInfo = "Ready to generate (\(variant.rawValue))"

        } catch {
            modelInfo = "Error loading models: \(error.localizedDescription)"
        }
    }

    /// Run text-to-image generation and return the resulting UIImage, or nil if cancelled.
    nonisolated public func generate(_ params: MobileOGenerator.GenerationParameters) async -> PlatformImage? {
        await MainActor.run { currentTask?.cancel() }

        await MainActor.run {
            running = true
            evaluationState = .processingPrompt
            currentStep = 0
            totalSteps = params.numSteps
        }

        let task = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return nil as PlatformImage? }
            do {
                let startTime = Date()

                let sana = await self.generator
                guard let sana = sana else {
                    throw NSError(domain: "MobileOModel", code: -1,
                                 userInfo: [NSLocalizedDescriptionKey: "MobileOGenerator not loaded"])
                }

                await MainActor.run { self.evaluationState = .generatingImage }

                if Task.isCancelled { return nil as PlatformImage? }

                let progressCallback: ((Int, Int) -> Void)? = { [weak self] step, total in
                    guard let self = self else { return }
                    Task { @MainActor [weak self] in
                        guard let self = self else { return }
                        self.currentStep = step
                        self.totalSteps = total
                    }
                }

                let paramsWithCallback = MobileOGenerator.GenerationParameters(
                    prompt: params.prompt,
                    numSteps: params.numSteps,
                    guidanceScale: params.guidanceScale,
                    enableCFG: params.enableCFG,
                    seed: params.seed,
                    progressCallback: progressCallback
                )

                let (imageArray, timing) = try await sana.generate(paramsWithCallback)

                if Task.isCancelled { return nil as PlatformImage? }

                let image = try multiArrayToImage(imageArray)

                let elapsed = Date().timeIntervalSince(startTime)

                await MainActor.run {
                    self.generationTime = String(format: "%.1f seconds", elapsed)
                    self.inferenceTime = timing.totalTime
                    self.lastTiming = timing
                    self.resetState()
                }

                return image

            } catch {
                if !(error is CancellationError || Task.isCancelled) {
                    await MainActor.run {
                        self.modelInfo = "Error: \(error.localizedDescription)"
                    }
                }

                await MainActor.run { self.resetState() }
                return nil
            }
        }

        await MainActor.run { currentTask = task }
        return await task.value
    }

    public func cancel() {
        currentTask?.cancel()
        currentTask = nil
        resetState()
    }

    /// Reset observable state to idle
    private func resetState() {
        running = false
        evaluationState = .idle
        currentStep = 0
        totalSteps = 0
    }

    public func releaseModels() {
        generator?.releaseModels()
        generator = nil
    }

    // MARK: - Helper Functions

    /// Convert an MLMultiArray [1, C, H, W] to a UIImage using Metal GPU
    nonisolated private func multiArrayToImage(_ array: MLMultiArray) throws -> PlatformImage {
        let shape = array.shape.map { $0.intValue }

        guard shape.count == 4 else {
            throw NSError(domain: "MobileOModel", code: -2,
                         userInfo: [NSLocalizedDescriptionKey: "Invalid array shape"])
        }

        let channels = shape[1]
        let height = shape[2]
        let width = shape[3]
        let totalElements = channels * height * width

        var pixelData = [UInt8](repeating: 0, count: height * width * 4)

        guard let device = metalDevice,
              let commandQueue = metalCommandQueue,
              let pipeline = floatToRGBAPipeline,
              array.dataType == .float32 else {
            throw NSError(domain: "MobileOModel", code: -3,
                         userInfo: [NSLocalizedDescriptionKey: "Metal GPU not available for image conversion"])
        }

        guard let inputBuffer = device.makeBuffer(
            bytes: array.dataPointer,
            length: totalElements * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: height * width * 4 * MemoryLayout<UInt8>.stride,
            options: .storageModeShared
        ) else {
            throw NSError(domain: "MobileOModel", code: -3,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create Metal buffers"])
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw NSError(domain: "MobileOModel", code: -3,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create Metal encoder"])
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)

        var channelsVar = UInt32(channels)
        var heightVar = UInt32(height)
        var widthVar = UInt32(width)
        encoder.setBytes(&channelsVar, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&heightVar, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&widthVar, length: MemoryLayout<UInt32>.stride, index: 4)

        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (width + 15) / 16,
            height: (height + 15) / 16,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        memcpy(&pixelData, outputBuffer.contents(), height * width * 4)

        let provider = CGDataProvider(data: Data(pixelData) as CFData)!
        let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )!

        return UIImage(cgImage: cgImage)
    }
}
