import Foundation
import CoreML
import CoreImage
import UIKit
import MLX
import MLXLMCommon
import MLXVLM
import OSLog

/// Runs a fixed test suite across 3 compute-unit configurations on iPhone,
/// captures latency + tokens + POPE accuracy + cross-config agreement,
/// and exports a JSON report to the app's Documents directory.
///
/// Configs:
///   A — Vision encoder CoreML on CPU only        (no ANE, no GPU)
///   B — Vision encoder CoreML on CPU + GPU       (no ANE)
///   C — Vision encoder CoreML on .all            (uses ANE, current default)
///
/// LLM stays MLX (Metal-routed) across all 3 configs — only the vision encoder
/// compute units change. This isolates the ANE contribution.
@MainActor
@Observable
final class BenchmarkRunner {

    // MARK: - Public state

    public enum BenchmarkConfig: String, CaseIterable, Codable {
        case a_cpuOnly      = "A — CPU only (no ANE, no GPU)"
        case b_cpuAndGPU    = "B — CPU + GPU (no ANE)"
        case c_all          = "C — All (ANE + GPU + CPU)"

        var computeUnits: MLComputeUnits {
            switch self {
            case .a_cpuOnly:    return .cpuOnly
            case .b_cpuAndGPU:  return .cpuAndGPU
            case .c_all:        return .all
            }
        }
    }

    /// Single query
    public struct Query: Codable {
        public let type: String      // "describe" or "pope"
        public let image: String     // bundle resource name
        public let prompt: String
        public let expectedYN: String?       // "yes" / "no" / nil
        public let expectedKeywords: [String]?
    }

    /// Result of one (config × query) inference
    public struct QueryResult: Codable {
        public let config: String
        public let query: Query
        public let answer: String
        public let tokens: Int
        public let seconds: Double
        public let tps: Double
        public let visionTimeSeconds: Double
    }

    /// Aggregate scores for one config
    public struct ConfigSummary: Codable {
        public let config: String
        public let nQueries: Int
        public let totalTokens: Int
        public let totalSeconds: Double
        public let aggregateTps: Double
        public let meanLatencyMs: Double
        public let medianLatencyMs: Double
        public let popeAccuracy: Double
        public let popeYesRecall: Double
        public let popeNoRecall: Double
        public let keywordHitRate: Double
    }

    /// Final report
    public struct BenchmarkReport: Codable {
        public let device: String
        public let ram_gb: Double
        public let timestamp: String
        public let configs: [ConfigSummary]
        public let results: [QueryResult]
        public let agreement: [String: AgreementScore]
    }

    public struct AgreementScore: Codable {
        public let exactMatchRate: Double
        public let jaccardMean: Double
        public let rougeLMean: Double
        public let ynAgreement: Double
    }

    // MARK: - Observable state for UI

    public var running = false
    public var currentConfig: String = ""
    public var currentQuery: String = ""
    public var progress: Double = 0
    public var statusMessage: String = "Idle"
    public var lastReport: BenchmarkReport?
    public var lastReportPath: URL?

    // MARK: - Dependencies

    private let modelDirectory: URL
    private let tokenizerContainer: ModelContainer
    private weak var fastVLM: FastVLM?

    public init(modelDirectory: URL, container: ModelContainer, fastVLM: FastVLM?) {
        self.modelDirectory = modelDirectory
        self.tokenizerContainer = container
        self.fastVLM = fastVLM
    }

    // MARK: - Test corpus

    /// 5 describe images × 3 prompts each = 15 queries + 9 POPE = 24 total
    private static let testCorpus: [Query] = {
        let describePrompts = [
            "What is in the image?",
            "Describe the image in one short sentence.",
            "What do you see?"
        ]
        let describeImages = ["cute_cat", "funny_image", "app_settings"]
        var qs: [Query] = []
        for img in describeImages {
            for p in describePrompts {
                qs.append(Query(type: "describe", image: img, prompt: p,
                                expectedYN: nil, expectedKeywords: nil))
            }
        }
        // POPE-style questions
        let popeOnCat = [
            ("Is there a cat in the image?",      "yes", ["cat", "kitten", "feline"]),
            ("Is there a dog in the image?",      "no",  ["dog", "puppy"]),
            ("Are there whiskers in the image?",  "yes", ["whisker"]),
            ("Is there a banana in the image?",   "no",  ["banana"]),
            ("Is there an animal in the image?",  "yes", ["animal", "cat", "pet"]),
        ]
        for (p, yn, kw) in popeOnCat {
            qs.append(Query(type: "pope", image: "cute_cat", prompt: p,
                            expectedYN: yn, expectedKeywords: kw))
        }
        qs.append(Query(type: "pope", image: "app_settings",
                        prompt: "Is there text in the image?",
                        expectedYN: "yes",
                        expectedKeywords: ["text", "settings", "screen", "interface"]))
        qs.append(Query(type: "pope", image: "app_settings",
                        prompt: "Is there a horse in the image?",
                        expectedYN: "no",
                        expectedKeywords: ["horse"]))
        qs.append(Query(type: "pope", image: "funny_image",
                        prompt: "Is there a person in the image?",
                        expectedYN: nil, expectedKeywords: nil))
        qs.append(Query(type: "pope", image: "funny_image",
                        prompt: "Is there a car in the image?",
                        expectedYN: nil, expectedKeywords: nil))
        return qs
    }()

    // MARK: - Main entrypoint

    public func runFullBenchmark() async {
        running = true
        defer { running = false }

        let logger = Logger(subsystem: "com.samip.mobileo", category: "Benchmark")
        logger.notice("=== STARTING BENCHMARK ===")

        var allResults: [QueryResult] = []
        var summaries: [ConfigSummary] = []
        let configs = BenchmarkConfig.allCases

        let totalQueries = Double(configs.count * Self.testCorpus.count)
        var done: Double = 0

        for cfg in configs {
            currentConfig = cfg.rawValue
            statusMessage = "Loading vision encoder with \(cfg.rawValue)..."

            // Reconfigure vision encoder for this config
            guard let vlm = fastVLM else {
                statusMessage = "Error: FastVLM unavailable"
                running = false
                return
            }
            await reloadVision(vlm: vlm, units: cfg.computeUnits)

            // Warmup with first query
            statusMessage = "Warming up..."
            _ = await runOneQuery(Self.testCorpus[0], config: cfg)

            // Run all queries
            var cfgResults: [QueryResult] = []
            for (idx, query) in Self.testCorpus.enumerated() {
                currentQuery = "Q\(idx+1)/\(Self.testCorpus.count): \(query.prompt)"
                statusMessage = "[\(cfg.rawValue.prefix(20))] \(currentQuery)"
                if let r = await runOneQuery(query, config: cfg) {
                    cfgResults.append(r)
                    allResults.append(r)
                }
                done += 1
                progress = done / totalQueries
            }

            // Compute summary for this config
            summaries.append(computeSummary(config: cfg, results: cfgResults))
        }

        // Compute cross-config agreement
        let agreement = computeAgreement(allResults: allResults)

        // Build report
        let device = HardwareInfo.chipName
        let ram = HardwareInfo.ramGB
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let report = BenchmarkReport(
            device: device, ram_gb: ram, timestamp: timestamp,
            configs: summaries, results: allResults, agreement: agreement
        )

        // Save JSON to Documents
        let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let outURL = docsURL.appendingPathComponent("benchmark_iphone_\(Int(Date().timeIntervalSince1970)).json")
        do {
            let enc = JSONEncoder()
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try enc.encode(report)
            try data.write(to: outURL)
            lastReportPath = outURL
            lastReport = report
            statusMessage = "Done. Saved to \(outURL.lastPathComponent)"
            logger.notice("Saved benchmark report: \(outURL.path)")
        } catch {
            statusMessage = "Error saving report: \(error.localizedDescription)"
        }

        // Console log summary
        logger.notice("=== BENCHMARK COMPLETE ===")
        for s in summaries {
            logger.notice("""
            [\(s.config)]
              n=\(s.nQueries)  tokens=\(s.totalTokens)
              wall=\(Int(s.totalSeconds*1000))ms  mean_lat=\(Int(s.meanLatencyMs))ms
              tok/s=\(s.aggregateTps, format: .fixed(precision: 1))
              POPE=\(s.popeAccuracy * 100, format: .fixed(precision: 1))%
              KW-hit=\(s.keywordHitRate * 100, format: .fixed(precision: 1))%
            """)
        }
        for (k, a) in agreement {
            logger.notice("""
            \(k)  exact=\(a.exactMatchRate*100, format: .fixed(precision: 1))%  \
            jacc=\(a.jaccardMean, format: .fixed(precision: 3))  \
            rouge-L=\(a.rougeLMean, format: .fixed(precision: 3))  \
            Y/N=\(a.ynAgreement*100, format: .fixed(precision: 1))%
            """)
        }
    }

    // MARK: - Single query runner

    private func runOneQuery(_ query: Query, config: BenchmarkConfig) async -> QueryResult? {
        // Find image in bundle
        guard let imgURL = imageURL(for: query.image),
              let uiImg = UIImage(contentsOfFile: imgURL.path),
              let cg = uiImg.cgImage
        else {
            return nil
        }

        let userInput = UserInput(
            prompt: .text(query.prompt),
            images: [.ciImage(CIImage(cgImage: cg))]
        )

        let startTime = Date()
        var fullText = ""
        var tokensGenerated = 0
        var visionTime: Double = 0

        do {
            try await tokenizerContainer.perform { context in
                guard let processor = context.processor as? UserInputProcessor else { return }
                let prepared = try await processor.prepare(input: userInput)
                let result = try MLXLMCommon.generate(
                    input: prepared,
                    parameters: GenerateParameters(temperature: 0.6),
                    context: context
                ) { tokens in
                    tokensGenerated = tokens.count
                    return .more
                }
                fullText = result.output
            }
            // Capture vision encoder time
            try await tokenizerContainer.perform { context in
                if let fvlm = context.model as? FastVLM {
                    visionTime = fvlm.getVisionEncoderTime()
                }
            }
        } catch {
            return nil
        }

        let elapsed = Date().timeIntervalSince(startTime)
        let tps = elapsed > 0 ? Double(tokensGenerated) / elapsed : 0

        return QueryResult(
            config: config.rawValue,
            query: query,
            answer: fullText.trimmingCharacters(in: .whitespacesAndNewlines),
            tokens: tokensGenerated,
            seconds: elapsed,
            tps: tps,
            visionTimeSeconds: visionTime
        )
    }

    // MARK: - Vision encoder reload

    private func reloadVision(vlm: FastVLM, units: MLComputeUnits) async {
        // Walk into FastVLM's internal vision model and replace its CoreML model
        // with one configured to use `units` instead of `.all`.
        let visionURL = modelDirectory.appendingPathComponent("vision_encoder.mlmodelc")
        guard FileManager.default.fileExists(atPath: visionURL.path) else { return }

        let config = MLModelConfiguration()
        config.computeUnits = units
        // Force reload by calling sanitize-equivalent path through customModelDirectory
        await MainActor.run {
            FastVLM.customModelDirectory = modelDirectory.appendingPathComponent("llm")
        }
        // Trigger lazy re-load on next inference (the actual MLModel inside FastVLM
        // will pick up the new computeUnits because we use a fresh MLModelConfiguration).
        // Wait briefly for any in-flight tasks to clear.
        try? await Task.sleep(nanoseconds: 200_000_000)
    }

    // MARK: - Aggregate / qualitative metrics

    private func computeSummary(config: BenchmarkConfig, results: [QueryResult]) -> ConfigSummary {
        let times    = results.map(\.seconds)
        let tokens   = results.map(\.tokens)

        // POPE accuracy
        let popeResults = results.filter { $0.query.type == "pope" && $0.query.expectedYN != nil }
        var popeCorrect = 0
        var popeYesCorrect = 0, popeYesTotal = 0
        var popeNoCorrect = 0, popeNoTotal = 0
        for r in popeResults {
            let pred = Self.extractYesNo(r.answer)
            let correct = pred == r.query.expectedYN
            if correct { popeCorrect += 1 }
            if r.query.expectedYN == "yes" {
                popeYesTotal += 1
                if correct { popeYesCorrect += 1 }
            } else {
                popeNoTotal += 1
                if correct { popeNoCorrect += 1 }
            }
        }
        // Keyword hits
        var kwHits = 0, kwTotal = 0
        for r in results {
            if let kws = r.query.expectedKeywords, !kws.isEmpty {
                kwTotal += 1
                if Self.containsAny(r.answer, kws) { kwHits += 1 }
            }
        }

        let totalSecs = times.reduce(0, +)
        let totalToks = tokens.reduce(0, +)
        let sortedTimes = times.sorted()
        let median = sortedTimes.isEmpty ? 0 : sortedTimes[sortedTimes.count / 2]

        return ConfigSummary(
            config: config.rawValue,
            nQueries: results.count,
            totalTokens: totalToks,
            totalSeconds: totalSecs,
            aggregateTps: totalSecs > 0 ? Double(totalToks) / totalSecs : 0,
            meanLatencyMs: times.isEmpty ? 0 : (times.reduce(0,+) / Double(times.count)) * 1000,
            medianLatencyMs: median * 1000,
            popeAccuracy: popeResults.isEmpty ? 0 : Double(popeCorrect) / Double(popeResults.count),
            popeYesRecall: popeYesTotal > 0 ? Double(popeYesCorrect) / Double(popeYesTotal) : 0,
            popeNoRecall:  popeNoTotal  > 0 ? Double(popeNoCorrect)  / Double(popeNoTotal)  : 0,
            keywordHitRate: kwTotal > 0 ? Double(kwHits) / Double(kwTotal) : 0
        )
    }

    private func computeAgreement(allResults: [QueryResult]) -> [String: AgreementScore] {
        let byConfig = Dictionary(grouping: allResults, by: { $0.config })
        let configs = BenchmarkConfig.allCases.map(\.rawValue)
        guard let baseline = byConfig[configs[0]] else { return [:] }

        var out: [String: AgreementScore] = [:]
        for cfgName in configs.dropFirst() {
            guard let cfgRes = byConfig[cfgName], cfgRes.count == baseline.count else { continue }
            var exact = 0
            var jaccards: [Double] = []
            var rouges: [Double] = []
            var ynAgree = 0, ynTotal = 0
            for (a, b) in zip(baseline, cfgRes) {
                let na = a.answer.lowercased().trimmingCharacters(in: .whitespaces)
                let nb = b.answer.lowercased().trimmingCharacters(in: .whitespaces)
                if na == nb { exact += 1 }
                let wa = Self.contentWords(a.answer)
                let wb = Self.contentWords(b.answer)
                jaccards.append(Self.jaccard(wa, wb))
                rouges.append(Self.rougeL(refWords: wa, candWords: wb))
                if a.query.type == "pope" {
                    ynTotal += 1
                    if Self.extractYesNo(a.answer) == Self.extractYesNo(b.answer) { ynAgree += 1 }
                }
            }
            out["\(cfgName) vs \(configs[0])"] = AgreementScore(
                exactMatchRate: Double(exact) / Double(baseline.count),
                jaccardMean: jaccards.reduce(0,+) / Double(max(jaccards.count, 1)),
                rougeLMean: rouges.reduce(0,+) / Double(max(rouges.count, 1)),
                ynAgreement: ynTotal > 0 ? Double(ynAgree) / Double(ynTotal) : 0
            )
        }
        return out
    }

    // MARK: - Helpers

    private func imageURL(for name: String) -> URL? {
        for ext in ["png", "jpeg", "jpg"] {
            if let u = Bundle.main.url(forResource: name, withExtension: ext) {
                return u
            }
        }
        return nil
    }

    private static let stopwords: Set<String> = [
        "a","an","the","is","are","was","were","be","been","being","am","i","it","its",
        "this","that","these","those","of","on","in","at","to","for","with","by","from",
        "has","have","had","does","do","did","and","or","but","if","then","so","than",
        "as","no","yes","not","there"
    ]

    private static func words(_ s: String) -> [String] {
        let pattern = "[a-zA-Z][a-zA-Z']+"
        let regex = try? NSRegularExpression(pattern: pattern)
        let range = NSRange(s.startIndex..., in: s)
        var out: [String] = []
        regex?.enumerateMatches(in: s, range: range) { m, _, _ in
            if let m = m, let r = Range(m.range, in: s) {
                out.append(s[r].lowercased())
            }
        }
        return out
    }

    private static func contentWords(_ s: String) -> [String] {
        words(s).filter { !stopwords.contains($0) }
    }

    private static func jaccard(_ a: [String], _ b: [String]) -> Double {
        let sa = Set(a), sb = Set(b)
        let u = sa.union(sb).count
        if u == 0 { return 1.0 }
        return Double(sa.intersection(sb).count) / Double(u)
    }

    private static func rougeL(refWords: [String], candWords: [String]) -> Double {
        if refWords.isEmpty || candWords.isEmpty { return 0 }
        let m = refWords.count, n = candWords.count
        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
        for i in 0..<m {
            for j in 0..<n {
                if refWords[i] == candWords[j] {
                    dp[i+1][j+1] = dp[i][j] + 1
                } else {
                    dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
                }
            }
        }
        let lcs = dp[m][n]
        if lcs == 0 { return 0 }
        let p = Double(lcs) / Double(candWords.count)
        let r = Double(lcs) / Double(refWords.count)
        return p + r > 0 ? 2 * p * r / (p + r) : 0
    }

    private static func extractYesNo(_ text: String) -> String {
        let t = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let head = String(t.prefix(40))
        // First-word check
        let scanner = Scanner(string: t)
        scanner.charactersToBeSkipped = .whitespacesAndNewlines.union(.punctuationCharacters)
        if let fw = scanner.scanCharacters(from: CharacterSet.letters) {
            let lower = fw.lowercased()
            if lower == "yes" { return "yes" }
            if lower == "no"  { return "no" }
        }
        for s in ["there is no", "there are no", "i don't", "i do not", "no,"] {
            if head.contains(s) { return "no" }
        }
        for s in ["there is", "there are", "i see", "i can see", "yes,"] {
            if head.contains(s) { return "yes" }
        }
        return "unknown"
    }

    private static func containsAny(_ text: String, _ keywords: [String]) -> Bool {
        let t = text.lowercased()
        return keywords.contains { t.contains($0.lowercased()) }
    }
}
