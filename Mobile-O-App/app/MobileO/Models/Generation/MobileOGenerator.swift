import CoreML
import Foundation
import MLX
import MLXLMCommon
import Accelerate
import Tokenizers
import Metal
import OSLog

// MARK: - Signposter for Instruments profiling
//
// Filter in Instruments by subsystem: "com.samip.mobileo"
// Categories: "Generation" (whole pipeline), "Stage" (per stage), "Step" (per DiT step)
//
// To profile:
//   1. Plug in iPhone, open Xcode
//   2. Product → Profile (⌘I) → choose "Logging" or "Time Profiler" template
//   3. In Instruments, add an "os_signpost" track filtered to com.samip.mobileo
//   4. Run a generation; each stage shows up as a colored interval
let mobileoSignposter = OSSignposter(
    subsystem: "com.samip.mobileo",
    category: "Generation"
)
let stageSignposter = OSSignposter(
    subsystem: "com.samip.mobileo",
    category: "Stage"
)
let stepSignposter = OSSignposter(
    subsystem: "com.samip.mobileo",
    category: "Step"
)

/// Common interface for diffusion schedulers (DPM-Solver++, Custom AB-3, etc.).
protocol DiffusionScheduler {
    associatedtype ConfigType
    var config: ConfigType { get }
    var timesteps: [Float] { get }
    func setTimesteps(numInferenceSteps: Int)
    func step(modelOutput: MLMultiArray, timestep: Float, sample: MLMultiArray) throws -> MLMultiArray
}

/// Seeded random number generator for reproducible generation (xorshift64)
struct RandomNumberGeneratorWithSeed: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}

/// SANA text-to-image generation using FastVLM text encoding + CoreML diffusion.
///
/// Pipeline: text → FastVLM encoder → conditioning connector → DiT denoising → VAE decode → image
public class MobileOGenerator {

    // MARK: - Model Variant

    public enum ModelVariant: String, CaseIterable {
        case fp32 = "FP32"

        var fileName: String { ModelConfiguration.variants[self]!.transformerFileName }
        var vaeFileName: String { ModelConfiguration.variants[self]!.vaeFileName }
    }

    // MARK: - Scheduler Type

    public enum SchedulerType: String, CaseIterable {
        case dpmSolver = "DPM-Solver++"
        case customScheduler = "Custom Scheduler"

        var description: String {
            switch self {
            case .dpmSolver: "DPM-Solver++ - Fast and high quality, 12-20 steps"
            case .customScheduler: "Custom Scheduler - AB-3 with update clamping, 8-12 steps"
            }
        }
    }

    // MARK: - CoreML Models

    private var connector: ConditioningConnector?
    private var transformer: TransformerModel?
    private var vae: VAEModel?
    public private(set) var currentModelVariant: ModelVariant = .fp32

    private let vaeScalingFactor: Float = 0.41407

    // MARK: - Scheduler

    private var scheduler: DiffusionScheduler?
    public private(set) var currentSchedulerType: SchedulerType = .dpmSolver

    // MARK: - Memory Pool

    private var arrayPool: [String: MLMultiArray] = [:]
    private let poolLock = NSLock()
    private var cachedTimestepArray: MLMultiArray?
    private var cachedLatentArray: MLMultiArray?
    private var cachedVAEInputArray: MLMultiArray?
    private var cachedUncondEmbeddings: (MLMultiArray, MLMultiArray)?
    private var cachedUncondSeqLen: Int = 0
    private var cachedFloat32ConversionArray: MLMultiArray?

    // MARK: - Metal Compute Pipelines

    private let metalDevice: MTLDevice?
    private let metalCommandQueue: MTLCommandQueue?
    private var normalizeVAEPipeline: MTLComputePipelineState?
    private var normalizeVAEFloat16Pipeline: MTLComputePipelineState?

    // MARK: - Text Encoding

    private let fastVLM: FastVLM
    private let tokenizer: Tokenizer
    private let cacheLock = NSLock()
    private var tokenCache: [String: [Int]] = [:]

    // MARK: - Model Directory

    /// Directory containing downloaded model weights (`.mlmodelc` files).
    private let modelDirectory: URL

    // MARK: - Initialization

    public init(fastVLM: FastVLM, tokenizer: Tokenizer, modelDirectory: URL) {
        self.fastVLM = fastVLM
        self.tokenizer = tokenizer
        self.modelDirectory = modelDirectory
        self.metalDevice = MTLCreateSystemDefaultDevice()
        self.metalCommandQueue = metalDevice?.makeCommandQueue()

        if let device = metalDevice {
            do {
                guard let library = device.makeDefaultLibrary() else { return }
                if let f = library.makeFunction(name: "normalizeVAEOutput") {
                    normalizeVAEPipeline = try device.makeComputePipelineState(function: f)
                }
                if let f = library.makeFunction(name: "normalizeVAEOutputFloat16") {
                    normalizeVAEFloat16Pipeline = try device.makeComputePipelineState(function: f)
                }
            } catch {}
        }
    }

    /// Run a block with a system activity guard to prevent sleep during generation.
    private func withHighPerformanceActivity<T>(
        reason: String,
        _ work: () async throws -> T
    ) async throws -> T {
        let activity = ProcessInfo.processInfo.beginActivity(
            options: [.suddenTerminationDisabled, .automaticTerminationDisabled,
                      .idleSystemSleepDisabled, .userInitiated],
            reason: reason
        )
        defer { ProcessInfo.processInfo.endActivity(activity) }
        return try await work()
    }

    // MARK: - Model Loading

    /// Load all CoreML models (connector, transformer, VAE) and scheduler
    public func loadModels(variant: ModelVariant = .fp32, schedulerType: SchedulerType = .dpmSolver) async throws {
        self.currentModelVariant = variant
        self.currentSchedulerType = schedulerType

        if connector == nil { try loadConnector() }

        let config = makeModelConfiguration(variant: variant)

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                let transformer = try ModelFactory.createTransformer(
                    variant: variant, configuration: config, modelDirectory: self.modelDirectory)
                await MainActor.run { self.transformer = transformer }
            }
            if self.vae == nil {
                group.addTask {
                    let vae = try ModelFactory.createVAE(
                        variant: variant, configuration: config, modelDirectory: self.modelDirectory)
                    await MainActor.run { self.vae = vae }
                }
            }
            try await group.waitForAll()
        }

        try await loadScheduler(schedulerType: schedulerType)
        prewarmMemoryPool()
    }

    /// Hot-swap only the DiT transformer (and VAE if the variant requires a different one).
    public func loadDiT(variant: ModelVariant) async throws {
        let needsVAESwitch = variant.vaeFileName != currentModelVariant.vaeFileName

        await MainActor.run {
            self.transformer = nil
            if needsVAESwitch { self.vae = nil }
        }

        let config = makeModelConfiguration(variant: variant)

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                let transformer = try ModelFactory.createTransformer(
                    variant: variant, configuration: config, modelDirectory: self.modelDirectory)
                await MainActor.run { self.transformer = transformer }
            }
            if needsVAESwitch {
                group.addTask {
                    let vae = try ModelFactory.createVAE(
                        variant: variant, configuration: config, modelDirectory: self.modelDirectory)
                    await MainActor.run { self.vae = vae }
                }
            }
            try await group.waitForAll()
        }

        self.currentModelVariant = variant
    }

    /// Hot-swap the diffusion scheduler without reloading transformer or VAE.
    public func loadScheduler(schedulerType: SchedulerType) async throws {
        // Look for scheduler_config.json in model directory first, then bundle
        let modelDirConfig = modelDirectory.appendingPathComponent("scheduler_config.json").path
        let bundleConfig   = Bundle.main.path(forResource: "scheduler_config", ofType: "json")

        let configPath: String
        if FileManager.default.fileExists(atPath: modelDirConfig) {
            configPath = modelDirConfig
        } else if let bundlePath = bundleConfig {
            configPath = bundlePath
        } else {
            throw NSError(domain: "MobileOGenerator", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "scheduler_config.json not found"])
        }

        switch schedulerType {
        case .dpmSolver:
            self.scheduler = try DPMSolverScheduler.fromConfigFile(path: configPath)
        case .customScheduler:
            self.scheduler = try CustomScheduler.fromConfigFile(path: configPath)
        }

        self.currentSchedulerType = schedulerType
    }

    private func makeModelConfiguration(variant: ModelVariant) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        // .all lets CoreML route each layer to ANE, GPU, or CPU automatically.
        // DiT transformer blocks are conv-heavy — ANE gives significant speedup vs GPU-only.
        config.computeUnits = .all
        config.allowLowPrecisionAccumulationOnGPU = true
        if let device = metalDevice {
            config.preferredMetalDevice = device
        }
        if #available(iOS 17.0, *) {
            config.modelDisplayName = "SANA_\(variant.rawValue)"
        }
        return config
    }

    private func loadConnector() throws {
        let connectorURL = modelDirectory.appendingPathComponent("connector.mlmodelc")
        self.connector = try MobileConditioningConnector(
            modelURL: connectorURL,
            numLayers: ConditioningConnectorConfiguration.numLayers,
            inputDim: ConditioningConnectorConfiguration.inputDim,
            outputDim: ConditioningConnectorConfiguration.outputDim,
            minSeqLen: ConditioningConnectorConfiguration.minSeqLen,
            maxSeqLen: ConditioningConnectorConfiguration.maxSeqLen
        )
    }

    // MARK: - Text Encoding

    /// Encode text prompt via FastVLM, returning all layer hidden states
    private func encodeTextInternal(_ prompt: String) async throws -> [MLXArray] {
        let messages = [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "Please generate image based on the following caption: \(prompt)"]
        ]

        let chatText = formatChatTemplate(messages: messages)
        let inputText = chatText + "<im_start>"

        let truncatedTokens: [Int]
        cacheLock.lock()
        if let cachedTokens = tokenCache[inputText] {
            truncatedTokens = cachedTokens
            cacheLock.unlock()
        } else {
            cacheLock.unlock()
            let tokens = tokenizer.encode(text: inputText)
            let tokensToCache = tokens.count > 512 ? Array(tokens.prefix(512)) : tokens
            cacheLock.lock()
            tokenCache[inputText] = tokensToCache
            cacheLock.unlock()
            truncatedTokens = tokensToCache
        }

        let inputIds = MLXArray(truncatedTokens)[.newAxis, 0...]
        let allHiddenStates = fastVLM.getAllHiddenStates(tokens: inputIds)
        MLX.eval(allHiddenStates)
        return allHiddenStates
    }

    private func formatChatTemplate(messages: [[String: String]]) -> String {
        var result = ""
        for message in messages {
            guard let role = message["role"], let content = message["content"] else { continue }
            result += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
        }
        result += "<|im_start|>assistant\n"
        return result
    }

    // MARK: - SANA Pipeline

    /// Transform hidden states to DiT conditioning via the mobile connector
    private func transformToSanaConditioning(_ allHiddenStates: [MLXArray]) throws -> (MLMultiArray, MLMultiArray) {
        guard let connector = connector else {
            throw NSError(domain: "MobileOGenerator", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Conditioning Connector not loaded"])
        }
        return try connector.forward(allHiddenStates)
    }

    /// Build unconditional (zero) embeddings for classifier-free guidance.
    /// Cached by sequence length — zeros of the same shape always produce the same output.
    private func buildCFGEmbeddings(_ allHiddenStates: [MLXArray]) throws -> (MLMultiArray, MLMultiArray) {
        let seqLen = allHiddenStates[0].dim(1)
        if let cached = cachedUncondEmbeddings, cachedUncondSeqLen == seqLen {
            return cached
        }
        let zeros = allHiddenStates.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
        let result = try connector!.forward(zeros)
        cachedUncondEmbeddings = result
        cachedUncondSeqLen = seqLen
        return result
    }

    // MARK: - Denoising

    /// Run the diffusion denoising loop
    private func denoise(
        encoderHiddenStates: MLMultiArray,
        attentionMask: MLMultiArray,
        unconditionalEncoderHiddenStates: MLMultiArray? = nil,
        unconditionalAttentionMask: MLMultiArray? = nil,
        numSteps: Int = 30,
        guidanceScale: Float = 1.3,
        enableCFG: Bool = false,
        seed: UInt64? = nil,
        progressCallback: (@MainActor (Int, Int) -> Void)? = nil
    ) async throws -> (result: MLMultiArray, vaeTime: TimeInterval) {
        guard let transformer = transformer else {
            throw NSError(domain: "MobileOGenerator", code: -3,
                         userInfo: [NSLocalizedDescriptionKey: "SANA DiT transformer not loaded"])
        }

        // DC-AE latent space: 32 channels, 16x16 spatial for 512px model
        let latentShape = [1, 32, 16, 16]
        if cachedLatentArray == nil || cachedLatentArray!.shape.map({ $0.intValue }) != latentShape {
            cachedLatentArray = try MLMultiArray(shape: latentShape.map { NSNumber(value: $0) }, dataType: .float32)
        }
        let latents = cachedLatentArray!

        if let seed = seed {
            var rng = RandomNumberGeneratorWithSeed(seed: seed)
            fillRandomNormal(latents, using: &rng)
        } else {
            var rng = SystemRandomNumberGenerator()
            fillRandomNormal(latents, using: &rng)
        }

        guard let scheduler = self.scheduler else {
            throw NSError(domain: "MobileOGenerator", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Scheduler not loaded"])
        }
        scheduler.setTimesteps(numInferenceSteps: numSteps)
        let timesteps = scheduler.timesteps

        var currentLatents = latents
        var currentTimestepArray: MLMultiArray? = nil

        for (index, t) in timesteps.enumerated() {
            if Task.isCancelled { throw CancellationError() }

            if let callback = progressCallback {
                Task { @MainActor in callback(index + 1, numSteps) }
            }

            if currentTimestepArray == nil {
                currentTimestepArray = try createTimestepArray(t: Int(t))
            }

            // Per-step signpost — see each DiT iteration in Instruments
            let stepState = stepSignposter.beginInterval("DiT_Step",
                id: stepSignposter.makeSignpostID(),
                "step=\(index + 1)/\(numSteps), t=\(Int(t))")

            var condNoisePred = try await transformer.predict(
                latent: currentLatents, timestep: currentTimestepArray!,
                encoderHiddenStates: encoderHiddenStates, encoderAttentionMask: attentionMask
            )
            if condNoisePred.dataType == .float16 {
                condNoisePred = try convertFloat16ToFloat32(condNoisePred)
            }

            let noisePred: MLMultiArray
            if enableCFG, let uncondHiddenStates = unconditionalEncoderHiddenStates,
               let uncondMask = unconditionalAttentionMask {
                var uncondNoisePred = try await transformer.predict(
                    latent: currentLatents, timestep: currentTimestepArray!,
                    encoderHiddenStates: uncondHiddenStates, encoderAttentionMask: uncondMask
                )
                if uncondNoisePred.dataType == .float16 {
                    uncondNoisePred = try convertFloat16ToFloat32(uncondNoisePred)
                }
                noisePred = try applyCFG(unconditional: uncondNoisePred, conditional: condNoisePred, guidanceScale: guidanceScale)
            } else {
                noisePred = condNoisePred
            }

            currentLatents = try scheduler.step(modelOutput: noisePred, timestep: t, sample: currentLatents)

            if index + 1 < timesteps.count {
                currentTimestepArray = try createTimestepArray(t: Int(timesteps[index + 1]))
            }

            stepSignposter.endInterval("DiT_Step", stepState)
        }

        let vaeMarker = stageSignposter.beginInterval("VAE_Decode",
            id: stageSignposter.makeSignpostID(), "ANE-split p1+p2")
        let vaeStart = Date()
        let finalImage = try await decodeLatents(currentLatents)
        let vaeTime = Date().timeIntervalSince(vaeStart)
        stageSignposter.endInterval("VAE_Decode", vaeMarker,
            "took=\(Int(vaeTime * 1000))ms")
        return (result: finalImage, vaeTime: vaeTime)
    }

    /// Apply CFG: noise_pred = uncond + guidance_scale * (cond - uncond)
    /// Uses two vDSP passes with zero temporary allocations (result buffer is pooled).
    private func applyCFG(unconditional: MLMultiArray, conditional: MLMultiArray, guidanceScale: Float) throws -> MLMultiArray {
        let shape = conditional.shape.map { $0.intValue }
        let count = shape.reduce(1, *)
        let result = try getPooledArray(shape: shape, key: "cfg_result")

        unconditional.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { uncondPtr in
            conditional.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { condPtr in
                result.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { resultPtr in
                    // result = cond - uncond
                    vDSP_vsub(uncondPtr, 1, condPtr, 1, resultPtr, 1, vDSP_Length(count))
                    // result = result * scale + uncond
                    var scale = guidanceScale
                    vDSP_vsma(resultPtr, 1, &scale, uncondPtr, 1, resultPtr, 1, vDSP_Length(count))
                }
            }
        }
        return result
    }

    /// Decode latents → scale + convert to float16 → VAE decode → normalize to [0,1].
    private func decodeLatents(_ latents: MLMultiArray) async throws -> MLMultiArray {
        guard let vae = vae else {
            throw NSError(domain: "MobileOGenerator", code: -5,
                         userInfo: [NSLocalizedDescriptionKey: "SANA VAE decoder not loaded"])
        }

        let float16Latents = try scaleAndConvertToFloat16(latents, scale: 1.0 / vaeScalingFactor)
        let image = try await vae.decode(latent: float16Latents)
        return try normalizeVAEOutput(image)
    }

    /// Normalize VAE output from [-1, 1] to [0, 1] using Metal GPU.
    /// Uses fused half→float+normalize kernel when input is FP16, avoiding a CPU conversion pass.
    private func normalizeVAEOutput(_ array: MLMultiArray) throws -> MLMultiArray {
        let shape = array.shape.map { $0.intValue }
        let result = try getPooledArray(shape: shape, key: "vae_normalize")
        let count = shape.reduce(1, *)

        guard let device = metalDevice, let commandQueue = metalCommandQueue else {
            throw NSError(domain: "MobileOGenerator", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Metal not available"])
        }

        let isFloat16 = array.dataType == .float16
        let pipeline: MTLComputePipelineState

        if isFloat16 {
            guard let p = normalizeVAEFloat16Pipeline else {
                throw NSError(domain: "MobileOGenerator", code: -1,
                             userInfo: [NSLocalizedDescriptionKey: "Metal float16 normalize pipeline not loaded"])
            }
            pipeline = p
        } else if array.dataType == .float32 {
            guard let p = normalizeVAEPipeline else {
                throw NSError(domain: "MobileOGenerator", code: -1,
                             userInfo: [NSLocalizedDescriptionKey: "Metal normalize pipeline not loaded"])
            }
            pipeline = p
        } else {
            throw NSError(domain: "MobileOGenerator", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Expected Float32 or Float16 from VAE, got \(array.dataType.rawValue)"])
        }

        let inputSize = isFloat16 ? count * MemoryLayout<UInt16>.stride : count * MemoryLayout<Float>.stride
        let outputSize = count * MemoryLayout<Float>.stride

        guard let inputBuffer = device.makeBuffer(bytes: array.dataPointer, length: inputSize, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw NSError(domain: "MobileOGenerator", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create Metal buffers"])
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw NSError(domain: "MobileOGenerator", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create Metal command encoder"])
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        var countVar = UInt32(count)
        encoder.setBytes(&countVar, length: MemoryLayout<UInt32>.stride, index: 2)

        let vectorizedCount = (count + 3) / 4
        let threadsPerThreadgroup = MTLSize(width: min(pipeline.threadExecutionWidth, vectorizedCount), height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(width: (vectorizedCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        memcpy(result.dataPointer, outputBuffer.contents(), outputSize)
        return result
    }

    // MARK: - Public Generation API

    public struct GenerationParameters {
        let prompt: String
        let numSteps: Int
        let guidanceScale: Float
        let enableCFG: Bool
        let seed: UInt64?
        let progressCallback: (@MainActor (Int, Int) -> Void)?

        public init(
            prompt: String,
            numSteps: Int = 20,
            guidanceScale: Float = 1.3,
            enableCFG: Bool = true,
            seed: UInt64? = nil,
            progressCallback: (@MainActor (Int, Int) -> Void)? = nil
        ) {
            self.prompt = prompt
            self.numSteps = numSteps
            self.guidanceScale = guidanceScale
            self.enableCFG = enableCFG
            self.seed = seed
            self.progressCallback = progressCallback
        }
    }

    /// Timing breakdown for a generation run
    public struct TimingInfo {
        public let tokenizationTime: TimeInterval
        public let llmTime: TimeInterval
        public let connectorTime: TimeInterval
        public let diffusionTime: TimeInterval
        public let vaeTime: TimeInterval
        public let totalTime: TimeInterval
    }

    public func generate(_ params: GenerationParameters) async throws -> (image: MLMultiArray, timing: TimingInfo) {
        try await withHighPerformanceActivity(reason: "ML Image Generation") {
            try await self.performPipeline(
                encodeHiddenStates: { try await self.encodeTextInternal(params.prompt) },
                params: (numSteps: params.numSteps, guidanceScale: params.guidanceScale,
                         enableCFG: params.enableCFG, seed: params.seed, progressCallback: params.progressCallback)
            )
        }
    }

    // MARK: - Shared Pipeline

    private typealias PipelineParams = (
        numSteps: Int, guidanceScale: Float, enableCFG: Bool,
        seed: UInt64?, progressCallback: (@MainActor (Int, Int) -> Void)?
    )

    /// Shared encode → connector → CFG → denoise → timing pipeline
    private func performPipeline(
        encodeHiddenStates: () async throws -> [MLXArray],
        params: PipelineParams
    ) async throws -> (image: MLMultiArray, timing: TimingInfo) {
        // ── Whole pipeline signpost ──────────────────────────────────────
        let pipelineState = mobileoSignposter.beginInterval(
            "Generate", id: mobileoSignposter.makeSignpostID(),
            "steps=\(params.numSteps), cfg=\(params.enableCFG)"
        )
        defer { mobileoSignposter.endInterval("Generate", pipelineState) }

        let totalStart = Date()

        // ── Stage 1: LLM (FastVLM hidden states) ─────────────────────────
        let llmState = stageSignposter.beginInterval("LLM_Forward",
            id: stageSignposter.makeSignpostID(), "MLX 4-bit on ANE")
        let encodingStart = Date()
        let allHiddenStates = try await encodeHiddenStates()
        let encodingTime = Date().timeIntervalSince(encodingStart)
        stageSignposter.endInterval("LLM_Forward", llmState,
            "took=\(Int(encodingTime * 1000))ms")

        // ── Stage 2: Connector (MCP projector) ───────────────────────────
        let connState = stageSignposter.beginInterval("Connector",
            id: stageSignposter.makeSignpostID(), "CoreML FP32")
        let connectorStart = Date()
        let (encoderHiddenStates, attentionMask) = try transformToSanaConditioning(allHiddenStates)
        let connectorTime = Date().timeIntervalSince(connectorStart)
        stageSignposter.endInterval("Connector", connState,
            "took=\(Int(connectorTime * 1000))ms")

        var uncondEncoderHiddenStates: MLMultiArray? = nil
        var uncondAttentionMask: MLMultiArray? = nil

        if params.enableCFG {
            let cfgState = stageSignposter.beginInterval("CFG_Embeddings",
                id: stageSignposter.makeSignpostID())
            let (uncondEncoded, uncondMask) = try buildCFGEmbeddings(allHiddenStates)
            uncondEncoderHiddenStates = uncondEncoded
            uncondAttentionMask = uncondMask
            stageSignposter.endInterval("CFG_Embeddings", cfgState)
        }

        // ── Stage 3 + 4: DiT denoise (+ VAE inside denoise) ──────────────
        let ditState = stageSignposter.beginInterval("DiT_Denoise",
            id: stageSignposter.makeSignpostID(),
            "ANE-split p1+p2, \(params.numSteps) steps")
        let diffusionStart = Date()
        let (decodedImage, vaeTime) = try await denoise(
            encoderHiddenStates: encoderHiddenStates,
            attentionMask: attentionMask,
            unconditionalEncoderHiddenStates: uncondEncoderHiddenStates,
            unconditionalAttentionMask: uncondAttentionMask,
            numSteps: params.numSteps,
            guidanceScale: params.guidanceScale,
            enableCFG: params.enableCFG,
            seed: params.seed,
            progressCallback: params.progressCallback
        )
        let diffusionTime = Date().timeIntervalSince(diffusionStart) - vaeTime
        stageSignposter.endInterval("DiT_Denoise", ditState,
            "took=\(Int(diffusionTime * 1000))ms")

        // VAE timing was already captured inside denoise(); emit its own marker
        let vaeState = stageSignposter.beginInterval("VAE_Decode_Marker",
            id: stageSignposter.makeSignpostID(),
            "VAE took=\(Int(vaeTime * 1000))ms")
        stageSignposter.endInterval("VAE_Decode_Marker", vaeState)

        let timing = TimingInfo(
            tokenizationTime: encodingTime * 0.1,
            llmTime: encodingTime * 0.9,
            connectorTime: connectorTime,
            diffusionTime: diffusionTime,
            vaeTime: vaeTime,
            totalTime: Date().timeIntervalSince(totalStart)
        )

        // Console log for easy verification without Instruments
        let logger = Logger(subsystem: "com.samip.mobileo", category: "Timing")
        logger.notice("""
        Mobile-O timing (iPhone 17 A19):
          LLM forward:        \(Int(timing.llmTime * 1000))ms
          Connector:          \(Int(timing.connectorTime * 1000))ms
          DiT (\(params.numSteps) steps):    \(Int(timing.diffusionTime * 1000))ms
          VAE decode:         \(Int(timing.vaeTime * 1000))ms
          TOTAL:              \(Int(timing.totalTime * 1000))ms
        """)

        return (image: decodedImage, timing: timing)
    }
}

// MARK: - Helper Functions

extension MobileOGenerator {

    private func getPooledArray(shape: [Int], key: String) throws -> MLMultiArray {
        poolLock.lock()
        defer { poolLock.unlock() }

        if let cached = arrayPool[key],
           cached.shape.map({ $0.intValue }) == shape,
           cached.dataType == .float32 {
            return cached
        }

        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        arrayPool[key] = array
        return array
    }

    private func prewarmMemoryPool() {
        do {
            // Latent space: [1, 32, 16, 16] for 512px model
            _ = try getPooledArray(shape: [1, 32, 16, 16], key: "cfg_result")
            _ = try getPooledArray(shape: [1, 3, 512, 512], key: "vae_normalize")
        } catch {}
    }

    public func clearMemoryPool() {
        poolLock.lock()
        defer { poolLock.unlock() }
        arrayPool.removeAll(keepingCapacity: false)
        tokenCache.removeAll(keepingCapacity: false)
        cachedUncondEmbeddings = nil
        cachedUncondSeqLen = 0
    }

    public func releaseModels() {
        transformer = nil
        vae = nil
        clearMemoryPool()
    }

    // MARK: - Numeric Helpers

    /// Convert FP16 to FP32 in-place using a cached output buffer (avoids allocation in hot loop).
    private func convertFloat16ToFloat32(_ array: MLMultiArray) throws -> MLMultiArray {
        let shape = array.shape
        let shapeInts = shape.map { $0.intValue }
        let count = shapeInts.reduce(1, *)

        if cachedFloat32ConversionArray == nil || cachedFloat32ConversionArray!.shape.map({ $0.intValue }) != shapeInts {
            cachedFloat32ConversionArray = try MLMultiArray(shape: shape, dataType: .float32)
        }
        let float32Array = cachedFloat32ConversionArray!
        let srcPtr = array.dataPointer.assumingMemoryBound(to: Float16.self)
        let dstPtr = float32Array.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<count { dstPtr[i] = Float(srcPtr[i]) }
        return float32Array
    }

    private func createTimestepArray(t: Int) throws -> MLMultiArray {
        if cachedTimestepArray == nil {
            cachedTimestepArray = try MLMultiArray(shape: [1], dataType: .float32)
        }
        cachedTimestepArray!.dataPointer.assumingMemoryBound(to: Float.self).pointee = Float(t)
        return cachedTimestepArray!
    }

    /// Fill array with normally-distributed random values (Box-Muller transform)
    private func fillRandomNormal<T: RandomNumberGenerator>(_ array: MLMultiArray, using rng: inout T) {
        let count = array.shape.map { $0.intValue }.reduce(1, *)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<count {
            let u1 = Float.random(in: 0..<1, using: &rng)
            let u2 = Float.random(in: 0..<1, using: &rng)
            ptr[i] = sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
        }
    }

    /// Fused scale + Float32→Float16 conversion in a single pass.
    /// Eliminates the intermediate scaled Float32 array and the separate Metal convert dispatch.
    private func scaleAndConvertToFloat16(_ array: MLMultiArray, scale: Float) throws -> MLMultiArray {
        let shape = array.shape
        let shapeInts = shape.map { $0.intValue }
        let count = shapeInts.reduce(1, *)

        if cachedVAEInputArray == nil || cachedVAEInputArray!.shape.map({ $0.intValue }) != shapeInts {
            cachedVAEInputArray = try MLMultiArray(shape: shape, dataType: .float16)
        }
        let result = cachedVAEInputArray!

        let srcPtr = array.dataPointer.assumingMemoryBound(to: Float.self)
        let dstPtr = result.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<count {
            dstPtr[i] = Float16(srcPtr[i] * scale)
        }
        return result
    }
}

// MARK: - Model Configuration

extension MobileOGenerator {

    struct ModelConfiguration {
        let transformerFileName: String
        let vaeFileName: String

        static let variants: [ModelVariant: ModelConfiguration] = [
            .fp32: ModelConfiguration(
                transformerFileName: "transformer_ane_p1_0.5.mlmodelc",
                vaeFileName: "vae_ane_p1_0.5.mlmodelc"
            )
        ]
    }

    struct ConditioningConnectorConfiguration {
        static let numLayers = 4
        static let inputDim = 896
        static let outputDim = 2304
        static let minSeqLen = 77
        static let maxSeqLen = 512
    }
}
