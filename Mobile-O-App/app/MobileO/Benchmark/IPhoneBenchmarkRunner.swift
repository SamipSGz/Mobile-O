import CoreImage
import CoreML
import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import OSLog
import SwiftUI
import Tokenizers
import UIKit

struct IPhoneBenchmarkView: View {
    let modelDirectory: URL
    @State private var status = "Preparing benchmark..."

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Mobile-O iPhone Benchmark")
                .font(.headline)
            Text(status)
                .font(.system(.body, design: .monospaced))
        }
        .padding()
        .task {
            do {
                let runner = IPhoneBenchmarkRunner(modelDirectory: modelDirectory)
                let config = IPhoneBenchmarkConfig.fromLaunchArguments()
                status = "Running \(config.rawValue)..."
                let output = try await runner.run(config: config)
                status = "Saved \(output.lastPathComponent)"
                try? await Task.sleep(nanoseconds: 800_000_000)
                Darwin.exit(0)
            } catch {
                status = "Benchmark failed: \(error.localizedDescription)"
                IPhoneBenchmarkRunner.writeLaunchError(error)
                Logger(subsystem: "com.samip.mobileo", category: "IPhoneBenchmark")
                    .error("Benchmark failed: \(error.localizedDescription, privacy: .public)")
                try? await Task.sleep(nanoseconds: 3_000_000_000)
                Darwin.exit(2)
            }
        }
    }
}

enum IPhoneBenchmarkConfig: String, Codable, CaseIterable {
    case beforeCPUGPU = "A_iPhone_CPU_GPU_Vision_MLX4bit"
    case afterANE = "B_iPhone_ANE_Vision_MLX4bit"
    case afterANETextOnly = "C_iPhone_ANE_MLX4bit_TextOnly"

    var usesImage: Bool {
        self != .afterANETextOnly
    }

    var visionComputeUnits: MLComputeUnits {
        switch self {
        case .beforeCPUGPU:
            return .cpuAndGPU
        case .afterANE, .afterANETextOnly:
            return .all
        }
    }

    var description: String {
        switch self {
        case .beforeCPUGPU:
            return "CoreML vision with CPU+GPU only, MLX 4-bit LLM"
        case .afterANE:
            return "CoreML vision with computeUnits=.all, MLX 4-bit LLM"
        case .afterANETextOnly:
            return "Text-only MLX 4-bit LLM control; image features omitted"
        }
    }

    static func fromLaunchArguments() -> IPhoneBenchmarkConfig {
        let args = ProcessInfo.processInfo.arguments
        guard let idx = args.firstIndex(of: "--iphone-benchmark-config"),
              idx + 1 < args.count,
              let config = IPhoneBenchmarkConfig(rawValue: args[idx + 1]) else {
            return .afterANE
        }
        return config
    }
}

private struct FrameSpec: Codable {
    let index: Int
    let fileName: String
    let expectedKeywords: [String]
}

private struct POPEProbe: Codable {
    let prompt: String
    let expectedYN: String
}

private struct InferenceRecord: Codable {
    let frame: Int
    let image: String
    let prompt: String
    let expectedYN: String?
    let expectedKeywords: [String]?
    let tokens: Int
    let seconds: Double
    let prepareSeconds: Double
    let visionSeconds: Double
    let llmSeconds: Double
    let answer: String
    let predYN: String?
    let correct: Bool?
    let keywordHit: Bool?
}

private struct OpenSummary: Codable {
    let mode: String                          // "sequential" or "pipelined"
    let measuredFrames: Int
    let wallSeconds: Double
    let tokens: Int
    let pipelineTokensPerSecond: Double       // tokens / wallSeconds (aggregate)
    let medianTokensPerSecond: Double         // median(tokens / llmSeconds) per query
    let framesPerSecond: Double
    let visionMedianMs: Double                // median per-frame vision time
    let llmMedianMs: Double                   // median per-frame LLM decode time
    let keywordHits: Int
    let keywordTotal: Int
    let keywordHitRate: Double
}

private struct POPESummary: Codable {
    let accuracy: Double
    let correct: Int
    let total: Int
    let yesRecall: Double
    let yesCorrect: Int
    let yesTotal: Int
    let noRecall: Double
    let noCorrect: Int
    let noTotal: Int
    let unknownRate: Double
}

private struct BenchmarkOutput: Codable {
    let device: [String: String]
    let config: IPhoneBenchmarkConfig
    let configDescription: String
    let warmupFrames: Int
    let maxOpenTokens: Int
    let maxPopeTokens: Int
    let openPrompt: String
    let openTemperature: Float
    let frames: [FrameSpec]
    let openSummary: OpenSummary              // sequential
    let pipelinedSummary: OpenSummary?        // pipelined (vision N+1 overlaps LLM N)
    let popeSummary: POPESummary
    let openResults: [InferenceRecord]
    let pipelinedResults: [InferenceRecord]?
    let popeResults: [InferenceRecord]
}

private final class IPhoneBenchmarkRunner {
    private let modelDirectory: URL
    private let logger = Logger(subsystem: "com.samip.mobileo", category: "IPhoneBenchmark")
    private let warmupFrames = 2
    private let openPrompt = "What is in the image?"
    private let maxOpenTokens = 64
    private let maxPopeTokens = 16
    /// Greedy decode (temperature 0) for reproducible per-query token counts.
    /// This makes pipelineTokensPerSecond fair across A/B/C reruns (no sampling variance).
    private let openTemperature: Float = 0.0

    private let frames: [FrameSpec] = [
        FrameSpec(index: 0, fileName: "cute_cat.png", expectedKeywords: ["cat", "kitten", "feline", "whisker"]),
        FrameSpec(index: 1, fileName: "funny_image.jpeg", expectedKeywords: ["dog", "shiba", "meme"]),
        FrameSpec(index: 2, fileName: "mobile-o-teaser.jpg", expectedKeywords: ["text-to-image", "generation", "diagram", "prompt", "mobile", "rainforest", "parrot", "visual", "comparison"]),
        FrameSpec(index: 3, fileName: "mobile-o-qualitative.jpg", expectedKeywords: ["prompt", "response", "question", "text", "generated", "qualitative", "comparison", "topic"]),
        FrameSpec(index: 4, fileName: "training_figure.jpg", expectedKeywords: ["diagram", "flowchart", "architecture", "training", "loss", "model", "language", "vae", "autoencoder", "projector", "encoder"]),
        FrameSpec(index: 5, fileName: "cute_cat.png", expectedKeywords: ["cat", "kitten", "feline", "whisker"]),
        FrameSpec(index: 6, fileName: "funny_image.jpeg", expectedKeywords: ["dog", "shiba", "meme"]),
        FrameSpec(index: 7, fileName: "mobile-o-teaser.jpg", expectedKeywords: ["text-to-image", "generation", "diagram", "prompt", "mobile", "rainforest", "parrot", "visual", "comparison"]),
    ]

    private let popeProbes: [String: [POPEProbe]] = [
        "cute_cat.png": [
            POPEProbe(prompt: "Is there a cat in the image? Answer yes or no.", expectedYN: "yes"),
            POPEProbe(prompt: "Are there whiskers in the image? Answer yes or no.", expectedYN: "yes"),
            POPEProbe(prompt: "Is there a dog in the image? Answer yes or no.", expectedYN: "no"),
            POPEProbe(prompt: "Is there a banana in the image? Answer yes or no.", expectedYN: "no"),
        ],
        "funny_image.jpeg": [
            POPEProbe(prompt: "Is there a dog in the image? Answer yes or no.", expectedYN: "yes"),
            POPEProbe(prompt: "Is there text in the image? Answer yes or no.", expectedYN: "yes"),
            POPEProbe(prompt: "Is there a cat in the image? Answer yes or no.", expectedYN: "no"),
            POPEProbe(prompt: "Is there a car in the image? Answer yes or no.", expectedYN: "no"),
        ],
        "mobile-o-teaser.jpg": [
            POPEProbe(prompt: "Is there text in the image? Answer yes or no.", expectedYN: "yes"),
            POPEProbe(prompt: "Is there a chart in the image? Answer yes or no.", expectedYN: "yes"),
            POPEProbe(prompt: "Is there a cat in the image? Answer yes or no.", expectedYN: "no"),
            POPEProbe(prompt: "Is there a horse in the image? Answer yes or no.", expectedYN: "no"),
        ],
        "mobile-o-qualitative.jpg": [
            POPEProbe(prompt: "Is there text in the image? Answer yes or no.", expectedYN: "yes"),
            POPEProbe(prompt: "Is there a flamingo in the image? Answer yes or no.", expectedYN: "yes"),
            POPEProbe(prompt: "Is there a cat in the image? Answer yes or no.", expectedYN: "no"),
            POPEProbe(prompt: "Is there a car in the image? Answer yes or no.", expectedYN: "no"),
        ],
        "training_figure.jpg": [
            POPEProbe(prompt: "Is there text in the image? Answer yes or no.", expectedYN: "yes"),
            POPEProbe(prompt: "Is there a diagram in the image? Answer yes or no.", expectedYN: "yes"),
            POPEProbe(prompt: "Is there a cat in the image? Answer yes or no.", expectedYN: "no"),
            POPEProbe(prompt: "Is there a dog in the image? Answer yes or no.", expectedYN: "no"),
        ],
    ]

    init(modelDirectory: URL) {
        self.modelDirectory = modelDirectory
    }

    static func writeLaunchError(_ error: Error) {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let dir = documents.appendingPathComponent("IPhoneBenchmarkResults", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("error.json")
        let payload: [String: Any] = [
            "error": error.localizedDescription,
            "arguments": ProcessInfo.processInfo.arguments,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: url, options: .atomic)
        }
    }

    func run(config: IPhoneBenchmarkConfig) async throws -> URL {
        FastVLM.benchmarkVisionComputeUnits = config.visionComputeUnits

        let container = try await loadContainer()
        try await warmup(container: container, config: config)

        // ── Sequential pass ────────────────────────────────────────────────
        let (openResults, openSummary) = try await runOpenFrames(
            container: container, config: config, mode: "sequential")
        let (popeResults, popeSummary) = try await runPOPE(container: container, config: config)

        // ── Pipelined pass (vision N+1 overlaps with LLM N) ────────────────
        // Skip for text-only control (no vision = no pipeline benefit)
        var pipelinedResults: [InferenceRecord]? = nil
        var pipelinedSummary: OpenSummary? = nil
        if config.usesImage {
            // Reset vision time counter, then run pipelined open-frames pass
            let (pr, ps) = try await runOpenFramesPipelined(container: container, config: config)
            pipelinedResults = pr
            pipelinedSummary = ps
        }

        let output = BenchmarkOutput(
            device: deviceInfo(),
            config: config,
            configDescription: config.description,
            warmupFrames: warmupFrames,
            maxOpenTokens: maxOpenTokens,
            maxPopeTokens: maxPopeTokens,
            openPrompt: openPrompt,
            openTemperature: openTemperature,
            frames: frames,
            openSummary: openSummary,
            pipelinedSummary: pipelinedSummary,
            popeSummary: popeSummary,
            openResults: openResults,
            pipelinedResults: pipelinedResults,
            popeResults: popeResults
        )

        let outURL = try write(output: output, config: config)
        logger.notice("IPHONE_BENCHMARK_RESULT \(outURL.path, privacy: .public)")
        return outURL
    }

    private func loadContainer() async throws -> ModelContainer {
        let nestedLLMDirectory = modelDirectory.appendingPathComponent("llm")
        let llmDirectory = FileManager.default.fileExists(
            atPath: nestedLLMDirectory.appendingPathComponent("config.json").path
        ) ? nestedLLMDirectory : modelDirectory

        let preprocDest = llmDirectory.appendingPathComponent("preprocessor_config.json")
        if !FileManager.default.fileExists(atPath: preprocDest.path),
           let bundled = Bundle.main.url(forResource: "preprocessor_config", withExtension: "json") {
            try? FileManager.default.copyItem(at: bundled, to: preprocDest)
        }

        FastVLM.customModelDirectory = llmDirectory
        FastVLM.register(modelFactory: VLMModelFactory.shared)
        let config = ModelConfiguration(directory: llmDirectory)

        let container = try await VLMModelFactory.shared.loadContainer(configuration: config) { _ in }
        try await container.perform { context in
            guard context.model is FastVLM else {
                throw NSError(domain: "IPhoneBenchmark", code: -1,
                              userInfo: [NSLocalizedDescriptionKey: "Loaded model is not FastVLM"])
            }
        }
        return container
    }

    private func warmup(container: ModelContainer, config: IPhoneBenchmarkConfig) async throws {
        try await container.perform { context in
            if let fastVLM = context.model as? FastVLM {
                fastVLM.warmup()
            }
        }
        if let firstMeasured = frames.dropFirst(warmupFrames).first {
            _ = try await infer(
                container: container,
                config: config,
                frame: firstMeasured,
                prompt: openPrompt,
                expectedYN: nil,
                expectedKeywords: firstMeasured.expectedKeywords,
                maxTokens: 8,
                temperature: 0.0
            )
        }
    }

    private func runOpenFrames(
        container: ModelContainer,
        config: IPhoneBenchmarkConfig,
        mode: String
    ) async throws -> ([InferenceRecord], OpenSummary) {
        var results: [InferenceRecord] = []

        // Warmup (not measured)
        for frame in frames.prefix(warmupFrames) {
            _ = try await infer(
                container: container,
                config: config,
                frame: frame,
                prompt: openPrompt,
                expectedYN: nil,
                expectedKeywords: frame.expectedKeywords,
                maxTokens: maxOpenTokens,
                temperature: openTemperature
            )
        }

        let wallStart = Date()
        for frame in frames.dropFirst(warmupFrames) {
            let record = try await infer(
                container: container,
                config: config,
                frame: frame,
                prompt: openPrompt,
                expectedYN: nil,
                expectedKeywords: frame.expectedKeywords,
                maxTokens: maxOpenTokens,
                temperature: openTemperature
            )
            results.append(record)
        }
        let wallSeconds = Date().timeIntervalSince(wallStart)

        return (results, makeOpenSummary(results: results, wallSeconds: wallSeconds, mode: mode))
    }

    /// Pipelined pass: vision encode of frame N+1 runs concurrently with LLM decode of frame N.
    /// Uses async tasks on independent ContainerPerform calls so CoreML (ANE) and MLX (Metal)
    /// can overlap on physically separate silicon.
    private func runOpenFramesPipelined(
        container: ModelContainer,
        config: IPhoneBenchmarkConfig
    ) async throws -> ([InferenceRecord], OpenSummary) {
        let measured = Array(frames.dropFirst(warmupFrames))

        // Stage 1 result holder: prepared input (after vision encode) + per-frame timings
        actor PreparedQueue {
            var slots: [Int: (input: LMInput, prepareSeconds: Double,
                              visionSeconds: Double, startTime: Date)] = [:]
            var waiters: [Int: CheckedContinuation<Void, Never>] = [:]

            func push(idx: Int, input: LMInput, prepareSeconds: Double,
                      visionSeconds: Double, startTime: Date) {
                slots[idx] = (input, prepareSeconds, visionSeconds, startTime)
                if let cont = waiters.removeValue(forKey: idx) { cont.resume() }
            }
            func pull(idx: Int) async -> (LMInput, Double, Double, Date) {
                if let s = slots.removeValue(forKey: idx) {
                    return (s.input, s.prepareSeconds, s.visionSeconds, s.startTime)
                }
                await withCheckedContinuation { cont in waiters[idx] = cont }
                let s = slots.removeValue(forKey: idx)!
                return (s.input, s.prepareSeconds, s.visionSeconds, s.startTime)
            }
        }

        let queue = PreparedQueue()
        var results: [InferenceRecord?] = Array(repeating: nil, count: measured.count)

        let wallStart = Date()

        try await withThrowingTaskGroup(of: Void.self) { group in
            // Producer: encode vision for each frame, push prepared input to queue
            group.addTask {
                for (idx, frame) in measured.enumerated() {
                    let frameStart = Date()
                    let visionBefore: Double = try await container.perform { ctx -> Double in
                        if let v = ctx.model as? FastVLM { return v.getVisionEncoderTime() }
                        return 0
                    }
                    var preparedInput: LMInput?
                    var prepareSeconds: Double = 0
                    var visionAfter: Double = 0
                    try await container.perform { ctx in
                        guard let processor = ctx.processor as? UserInputProcessor else {
                            throw NSError(domain: "IPhoneBenchmark", code: -2,
                                          userInfo: [NSLocalizedDescriptionKey: "Invalid processor type"])
                        }
                        let userInput = try self.makeUserInput(config: config, frame: frame, prompt: self.openPrompt)
                        let ps = Date()
                        let p = try await processor.prepare(input: userInput)
                        prepareSeconds = Date().timeIntervalSince(ps)
                        preparedInput = p
                        if let v = ctx.model as? FastVLM {
                            visionAfter = v.getVisionEncoderTime()
                        }
                    }
                    let visionSeconds = max(visionAfter - visionBefore, 0)
                    await queue.push(idx: idx, input: preparedInput!,
                                     prepareSeconds: prepareSeconds,
                                     visionSeconds: visionSeconds,
                                     startTime: frameStart)
                }
            }

            // Consumer: pull prepared inputs in order, run LLM decode
            group.addTask {
                for (idx, frame) in measured.enumerated() {
                    let (input, prepareSeconds, visionSeconds, startTime) =
                        await queue.pull(idx: idx)
                    var tokens = 0
                    var answer = ""
                    let llmStart = Date()
                    try await container.perform { ctx in
                        let res = try MLXLMCommon.generate(
                            input: input,
                            parameters: GenerateParameters(maxTokens: self.maxOpenTokens,
                                                           temperature: self.openTemperature),
                            context: ctx
                        ) { generated in
                            tokens = generated.count
                            return .more
                        }
                        answer = res.output.trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                    let llmSeconds = Date().timeIntervalSince(llmStart)
                    let elapsed = Date().timeIntervalSince(startTime)
                    let kwHit = containsAny(answer, keywords: frame.expectedKeywords)
                    let rec = InferenceRecord(
                        frame: frame.index,
                        image: frame.fileName,
                        prompt: self.openPrompt,
                        expectedYN: nil,
                        expectedKeywords: frame.expectedKeywords,
                        tokens: tokens,
                        seconds: elapsed,
                        prepareSeconds: prepareSeconds,
                        visionSeconds: visionSeconds,
                        llmSeconds: llmSeconds,
                        answer: answer,
                        predYN: nil,
                        correct: nil,
                        keywordHit: kwHit
                    )
                    results[idx] = rec
                }
            }

            try await group.waitForAll()
        }

        let wallSeconds = Date().timeIntervalSince(wallStart)
        let final = results.compactMap { $0 }
        return (final, makeOpenSummary(results: final, wallSeconds: wallSeconds, mode: "pipelined"))
    }

    private func makeOpenSummary(results: [InferenceRecord],
                                  wallSeconds: Double, mode: String) -> OpenSummary {
        let tokens = results.reduce(0) { $0 + $1.tokens }
        let keywordHits = results.reduce(0) { $0 + (($1.keywordHit ?? false) ? 1 : 0) }
        let total = results.count

        let perQueryTps = results.compactMap { r -> Double? in
            r.llmSeconds > 0 && r.tokens > 0 ? Double(r.tokens) / r.llmSeconds : nil
        }
        let visionMs = results.map { $0.visionSeconds * 1000 }.sorted()
        let llmMs = results.map { $0.llmSeconds * 1000 }.sorted()
        let medianTps = perQueryTps.isEmpty ? 0 : perQueryTps.sorted()[perQueryTps.count / 2]
        let medianVision = visionMs.isEmpty ? 0 : visionMs[visionMs.count / 2]
        let medianLLM = llmMs.isEmpty ? 0 : llmMs[llmMs.count / 2]

        return OpenSummary(
            mode: mode,
            measuredFrames: total,
            wallSeconds: wallSeconds,
            tokens: tokens,
            pipelineTokensPerSecond: wallSeconds > 0 ? Double(tokens) / wallSeconds : 0,
            medianTokensPerSecond: medianTps,
            framesPerSecond: wallSeconds > 0 ? Double(total) / wallSeconds : 0,
            visionMedianMs: medianVision,
            llmMedianMs: medianLLM,
            keywordHits: keywordHits,
            keywordTotal: total,
            keywordHitRate: total > 0 ? Double(keywordHits) / Double(total) : 0
        )
    }

    private func runPOPE(
        container: ModelContainer,
        config: IPhoneBenchmarkConfig
    ) async throws -> ([InferenceRecord], POPESummary) {
        var results: [InferenceRecord] = []
        for frame in frames.dropFirst(warmupFrames) {
            for probe in popeProbes[frame.fileName, default: []] {
                let record = try await infer(
                    container: container,
                    config: config,
                    frame: frame,
                    prompt: probe.prompt,
                    expectedYN: probe.expectedYN,
                    expectedKeywords: nil,
                    maxTokens: maxPopeTokens,
                    temperature: 0.0
                )
                results.append(record)
            }
        }

        let correct = results.reduce(0) { $0 + (($1.correct ?? false) ? 1 : 0) }
        let yesRows = results.filter { $0.expectedYN == "yes" }
        let noRows = results.filter { $0.expectedYN == "no" }
        let yesCorrect = yesRows.reduce(0) { $0 + (($1.correct ?? false) ? 1 : 0) }
        let noCorrect = noRows.reduce(0) { $0 + (($1.correct ?? false) ? 1 : 0) }
        let unknown = results.filter { $0.predYN == "unknown" }.count

        let summary = POPESummary(
            accuracy: results.isEmpty ? 0 : Double(correct) / Double(results.count),
            correct: correct,
            total: results.count,
            yesRecall: yesRows.isEmpty ? 0 : Double(yesCorrect) / Double(yesRows.count),
            yesCorrect: yesCorrect,
            yesTotal: yesRows.count,
            noRecall: noRows.isEmpty ? 0 : Double(noCorrect) / Double(noRows.count),
            noCorrect: noCorrect,
            noTotal: noRows.count,
            unknownRate: results.isEmpty ? 0 : Double(unknown) / Double(results.count)
        )
        return (results, summary)
    }

    private func infer(
        container: ModelContainer,
        config: IPhoneBenchmarkConfig,
        frame: FrameSpec,
        prompt: String,
        expectedYN: String?,
        expectedKeywords: [String]?,
        maxTokens: Int,
        temperature: Float
    ) async throws -> InferenceRecord {
        let start = Date()
        var prepareSeconds: Double = 0
        var llmSeconds: Double = 0
        var visionSeconds: Double = 0
        var tokens = 0
        var answer = ""

        try await container.perform { context in
            guard let processor = context.processor as? UserInputProcessor else {
                throw NSError(domain: "IPhoneBenchmark", code: -2,
                              userInfo: [NSLocalizedDescriptionKey: "Invalid processor type"])
            }

            let input = try makeUserInput(config: config, frame: frame, prompt: prompt)

            let prepStart = Date()
            let preparedInput = try await processor.prepare(input: input)
            prepareSeconds = Date().timeIntervalSince(prepStart)

            let llmStart = Date()
            let result = try MLXLMCommon.generate(
                input: preparedInput,
                parameters: GenerateParameters(maxTokens: maxTokens, temperature: temperature),
                context: context
            ) { generatedTokens in
                tokens = generatedTokens.count
                return .more
            }
            llmSeconds = Date().timeIntervalSince(llmStart)
            answer = result.output.trimmingCharacters(in: .whitespacesAndNewlines)

            if let fastVLM = context.model as? FastVLM {
                visionSeconds = fastVLM.getVisionEncoderTime()
            }
        }

        let elapsed = Date().timeIntervalSince(start)
        let predYN = expectedYN == nil ? nil : extractYesNo(answer)
        let keywordHit = expectedKeywords.map { containsAny(answer, keywords: $0) }
        let correct = expectedYN.map { $0 == predYN }

        return InferenceRecord(
            frame: frame.index,
            image: frame.fileName,
            prompt: prompt,
            expectedYN: expectedYN,
            expectedKeywords: expectedKeywords,
            tokens: tokens,
            seconds: elapsed,
            prepareSeconds: prepareSeconds,
            visionSeconds: visionSeconds,
            llmSeconds: llmSeconds,
            answer: answer,
            predYN: predYN,
            correct: correct,
            keywordHit: keywordHit
        )
    }

    private func makeUserInput(
        config: IPhoneBenchmarkConfig,
        frame: FrameSpec,
        prompt: String
    ) throws -> UserInput {
        guard config.usesImage else {
            return UserInput(prompt: .text(prompt), images: [])
        }

        let imageURL = imageDirectory().appendingPathComponent(frame.fileName)
        guard let image = UIImage(contentsOfFile: imageURL.path),
              let cgImage = image.cgImage else {
            throw NSError(domain: "IPhoneBenchmark", code: -3,
                          userInfo: [NSLocalizedDescriptionKey: "Missing benchmark image: \(imageURL.path)"])
        }

        return UserInput(
            prompt: .text(prompt),
            images: [.ciImage(CIImage(cgImage: cgImage))]
        )
    }

    private func imageDirectory() -> URL {
        documentsDirectory().appendingPathComponent("IPhoneBenchmarkImages", isDirectory: true)
    }

    private func documentsDirectory() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    }

    private func resultDirectory() -> URL {
        documentsDirectory().appendingPathComponent("IPhoneBenchmarkResults", isDirectory: true)
    }

    private func write(output: BenchmarkOutput, config: IPhoneBenchmarkConfig) throws -> URL {
        let dir = resultDirectory()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("\(config.rawValue).json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(output)
        try data.write(to: url, options: .atomic)
        return url
    }

    private func deviceInfo() -> [String: String] {
        return [
            "chip": HardwareInfo.chipName,
            "gpu": HardwareInfo.gpuName,
            "ram_gb": String(format: "%.1f", HardwareInfo.ramGB),
            "has_ane": HardwareInfo.hasANE ? "true" : "false",
        ]
    }
}

private func containsAny(_ text: String, keywords: [String]) -> Bool {
    let haystack = text.lowercased()
    return keywords.contains { haystack.contains($0.lowercased()) }
}

private func extractYesNo(_ text: String) -> String {
    let lower = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
    if lower.hasPrefix("yes") { return "yes" }
    if lower.hasPrefix("no") { return "no" }

    let head = String(lower.prefix(80))
    let noSignals = [
        "there is no", "there are no", "does not", "do not", "cannot see",
        "can't see", "not visible", "no ",
    ]
    let yesSignals = [
        "there is", "there are", "i see", "visible", "appears", "contains",
        "shown", "present",
    ]
    if noSignals.contains(where: { head.contains($0) }) { return "no" }
    if yesSignals.contains(where: { head.contains($0) }) { return "yes" }
    return "unknown"
}
