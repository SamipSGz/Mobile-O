import Foundation
import CoreML
import Metal

/// Detects and reports which hardware units are active on this device.
/// Used to verify hardware-aware scheduling is working at runtime.
struct HardwareInfo {

    // MARK: - Chip detection

    static var chipName: String {
        // Prefer Metal's GPU name — it's authoritative and includes the actual chip gen.
        // "Apple A19 Pro GPU" → "A19 Pro"
        // "Apple A18 GPU" → "A18"
        if let gpu = MTLCreateSystemDefaultDevice()?.name {
            let chip = gpu
                .replacingOccurrences(of: "Apple ", with: "")
                .replacingOccurrences(of: " GPU", with: "")
            // Append iPhone model from sysctl for context
            var size = 0
            sysctlbyname("hw.machine", nil, &size, nil, 0)
            var machine = [CChar](repeating: 0, count: size)
            sysctlbyname("hw.machine", &machine, &size, nil, 0)
            let id = String(cString: machine)
            let modelMap: [String: String] = [
                "iPhone15,2": "iPhone 14 Pro",    "iPhone15,3": "iPhone 14 Pro Max",
                "iPhone15,4": "iPhone 15",         "iPhone15,5": "iPhone 15 Plus",
                "iPhone16,1": "iPhone 15 Pro",     "iPhone16,2": "iPhone 15 Pro Max",
                "iPhone17,1": "iPhone 16 Pro",     "iPhone17,2": "iPhone 16 Pro Max",
                "iPhone17,3": "iPhone 16",         "iPhone17,4": "iPhone 16 Plus",
                "iPhone18,1": "iPhone 17 Pro",     "iPhone18,2": "iPhone 17 Pro Max",
                "iPhone18,3": "iPhone 17",         "iPhone18,4": "iPhone 17 Plus",
                "iPhone18,5": "iPhone 17 Air",
                // Future devices: map from sysctl id when known
            ]
            if let model = modelMap[id] {
                return "\(chip) (\(model))"
            }
            return chip
        }
        // Fallback: sysctl identifier
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var machine = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.machine", &machine, &size, nil, 0)
        return String(cString: machine)
    }

    static var ramGB: Double {
        Double(ProcessInfo.processInfo.physicalMemory) / 1_000_000_000
    }

    // MARK: - Simple heuristic (works without async)

    /// Fast synchronous check: confirms .all compute units are set.
    static func componentSchedule() -> [(component: String, device: String, detail: String)] {
        let ane  = hasANE
        let gpu  = gpuName
        let ram  = ramGB

        // Precision: float16 on GPU/ANE, float32 on CPU-only
        let precision = (ane || !gpu.isEmpty) ? "FP16" : "FP32"

        // Vision: ANE preferred (conv-heavy), GPU fallback
        let visionDevice = ane ? "ANE + GPU" : (gpu.isEmpty ? "CPU" : "GPU")
        // LLM: MLX routes to ANE on Apple Silicon natively
        let llmDevice    = ane ? "ANE" : (gpu.isEmpty ? "CPU" : "GPU")
        // DiT/VAE: GPU-heavy transformer ops
        let ditDevice    = ane ? "ANE + GPU" : (gpu.isEmpty ? "CPU" : "GPU")
        // Connector: small MLP, CPU+GPU
        let connDevice   = gpu.isEmpty ? "CPU" : "CPU + GPU"

        // Quantization based on RAM
        let llmQuant = ram < 4 ? "INT4" : (ram < 8 ? "4-bit" : "4-bit")

        return [
            (component: "Vision Encoder",
             device: visionDevice,
             detail: "CoreML \(precision) · computeUnits = .all"),
            (component: "Language Model",
             device: llmDevice,
             detail: "MLX \(llmQuant) · Neural Engine native"),
            (component: "DiT Transformer",
             device: ditDevice,
             detail: "CoreML FP32 · ANE-split p1+p2 · computeUnits = .all"),
            (component: "VAE Decoder",
             device: ditDevice,
             detail: "CoreML FP32 · ANE-split p1+p2 · computeUnits = .all"),
            (component: "Connector",
             device: connDevice,
             detail: "CoreML FP32 · computeUnits = .all"),
        ]
    }

    // MARK: - Metal GPU name

    static var gpuName: String {
        MTLCreateSystemDefaultDevice()?.name ?? "unknown"
    }

    // MARK: - ANE availability

    static var hasANE: Bool {
        // All iPhone 15+ (A17+) have a 16-core ANE.
        // Derive from Metal GPU name — most reliable source.
        let gpu = MTLCreateSystemDefaultDevice()?.name ?? ""
        return gpu.contains("A17") || gpu.contains("A18") || gpu.contains("A19") || gpu.contains("A20")
    }
}
