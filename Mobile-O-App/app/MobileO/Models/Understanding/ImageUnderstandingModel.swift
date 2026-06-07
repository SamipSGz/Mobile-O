import Foundation
import MLX
import MLXNN
import MLXLMCommon
import Tokenizers
import MLXVLM
import OSLog

import UIKit
import CoreImage.CIFilterBuiltins

/// Signposters for understanding pipeline — visible in Instruments
private let understandingSignposter = OSSignposter(
    subsystem: "com.samip.mobileo", category: "Understanding")
private let understandingStageSignposter = OSSignposter(
    subsystem: "com.samip.mobileo", category: "UnderstandingStage")

/// Observable wrapper for FastVLM image understanding inference
@Observable
@MainActor
class ImageUnderstandingModel {

    public var running = false
    public var modelInfo = ""
    public var response: String = ""
    public var timeToFirstToken: TimeInterval = 0
    public var totalTime: TimeInterval = 0
    public var tokensGenerated: Int = 0
    public var visionEncoderTime: TimeInterval = 0
    public var prepareTime: TimeInterval = 0
    public var llmGenerateTime: TimeInterval = 0

    private var container: ModelContainer?
    private var currentTask: Task<String, Never>?
    private var startTime: Date?
    private var firstTokenTime: Date?

    public init(container: ModelContainer? = nil) {
        self.container = container
    }

    public func load() async {
        modelInfo = container != nil ? "Ready to understand images" : "Error: Container not provided"
    }

    nonisolated public func understand(image: PlatformImage, prompt: String) async -> String {
        await MainActor.run {
            currentTask?.cancel()
            running = true
            response = ""
            timeToFirstToken = 0
            totalTime = 0
            tokensGenerated = 0
            visionEncoderTime = 0
            startTime = Date()
            firstTokenTime = nil
        }

        let task = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return "" }
            let activity = ProcessInfo.processInfo.beginActivity(
                options: [.suddenTerminationDisabled, .userInitiated],
                reason: "ML Image Understanding"
            )
            defer { ProcessInfo.processInfo.endActivity(activity) }

            // ── Whole understanding pipeline signpost ─────────────────
            let pipelineState = understandingSignposter.beginInterval(
                "Understand", id: understandingSignposter.makeSignpostID())
            defer { understandingSignposter.endInterval("Understand", pipelineState) }

            do {
                let container = await self.container
                guard let container = container else {
                    throw NSError(domain: "ImageUnderstandingModel", code: -1,
                                 userInfo: [NSLocalizedDescriptionKey: "Model not loaded"])
                }

                guard let cgImage = image.cgImage else {
                    throw NSError(domain: "ImageUnderstandingModel", code: -2,
                                 userInfo: [NSLocalizedDescriptionKey: "Failed to convert image"])
                }

                let userInput = UserInput(
                    prompt: .text(prompt),
                    images: [.ciImage(CIImage(cgImage: cgImage))]
                )

                var fullResponse = ""
                var processedTokenCount = 0
                var prepareDuration: TimeInterval = 0
                var llmGenDuration: TimeInterval = 0

                try await container.perform { context in
                    guard let processor = context.processor as? UserInputProcessor else {
                        throw NSError(domain: "ImageUnderstandingModel", code: -3,
                                     userInfo: [NSLocalizedDescriptionKey: "Invalid processor type"])
                    }

                    // ── Stage 1: Input prep (includes image encoding via FastVLM CoreML) ──
                    let prepState = understandingStageSignposter.beginInterval(
                        "Prepare_Input",
                        id: understandingStageSignposter.makeSignpostID(),
                        "vision encode + tokenize")
                    let prepStart = Date()
                    let preparedInput = try await processor.prepare(input: userInput)
                    prepareDuration = Date().timeIntervalSince(prepStart)
                    understandingStageSignposter.endInterval("Prepare_Input", prepState,
                        "took=\(Int(prepareDuration * 1000))ms")

                    // ── Stage 2: LLM token generation (MLX on ANE) ──
                    let llmState = understandingStageSignposter.beginInterval(
                        "LLM_Generate",
                        id: understandingStageSignposter.makeSignpostID(),
                        "MLX 4-bit autoregressive")
                    let llmStart = Date()
                    let result = try MLXLMCommon.generate(
                        input: preparedInput,
                        parameters: GenerateParameters(temperature: 0.6),
                        context: context
                    ) { tokens in
                        if Task.isCancelled { return .stop }

                        if tokens.count > processedTokenCount {
                            let newTokens = Array(tokens[processedTokenCount...])
                            let tokenString = context.tokenizer.decode(tokens: newTokens)
                            let currentTokenCount = tokens.count

                            Task { @MainActor in
                                if self.firstTokenTime == nil {
                                    self.firstTokenTime = Date()
                                    if let start = self.startTime {
                                        self.timeToFirstToken = self.firstTokenTime!.timeIntervalSince(start)
                                    }
                                }
                                self.response += tokenString
                                self.tokensGenerated = currentTokenCount
                            }

                            processedTokenCount = tokens.count
                        }
                        return .more
                    }

                    fullResponse = result.output
                    llmGenDuration = Date().timeIntervalSince(llmStart)
                    understandingStageSignposter.endInterval("LLM_Generate", llmState,
                        "took=\(Int(llmGenDuration * 1000))ms, tokens=\(processedTokenCount)")
                }

                // Capture vision encoder timing (separately tracked inside FastVLM)
                var visionTime: TimeInterval = 0
                try await container.perform { context in
                    if let fastVLM = context.model as? FastVLM {
                        visionTime = fastVLM.getVisionEncoderTime()
                        await MainActor.run { self.visionEncoderTime = visionTime }
                    }
                }

                let capturedStart: Date? = await MainActor.run { self.startTime }
                let totalDuration: TimeInterval = (capturedStart.map { Date().timeIntervalSince($0) })
                    ?? (prepareDuration + llmGenDuration)

                // Console log for paper measurements
                let logger = Logger(subsystem: "com.samip.mobileo", category: "Timing")
                logger.notice("""
                Mobile-O UNDERSTANDING timing (iPhone 17 A19):
                  Input prepare (incl. vision encode):  \(Int(prepareDuration * 1000))ms
                    └─ Vision encoder (CoreML/ANE):     \(Int(visionTime * 1000))ms
                  LLM generate (\(processedTokenCount) tokens):   \(Int(llmGenDuration * 1000))ms
                  TOTAL:                                \(Int(totalDuration * 1000))ms
                """)

                await MainActor.run {
                    if !Task.isCancelled { self.response = fullResponse }
                    self.totalTime = totalDuration
                    self.prepareTime = prepareDuration
                    self.llmGenerateTime = llmGenDuration
                    self.running = false
                }

                return Task.isCancelled ? await MainActor.run { self.response } : fullResponse

            } catch {
                if !Task.isCancelled {
                    await MainActor.run { self.modelInfo = "Error: \(error.localizedDescription)" }
                }
                await MainActor.run { self.running = false }
                return ""
            }
        }

        await MainActor.run { currentTask = task }
        return await task.value
    }

    public func cancel() {
        currentTask?.cancel()
        currentTask = nil
        running = false
        timeToFirstToken = 0
        totalTime = 0
        tokensGenerated = 0
        visionEncoderTime = 0
        startTime = nil
        firstTokenTime = nil
    }

    public func releaseModels() {
        container = nil
    }
}
