import SwiftUI

/// Settings screen for image generation parameters (scheduler, steps, CFG).
struct SettingsView: View {
    @Bindable var model: MobileOModel
    @Bindable var settings: SettingsViewModel
    var benchmarkRunner: BenchmarkRunner?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                // ── Hardware-aware status ─────────────────────────────────
                HardwareStatusSection()

                // ── Benchmark link ───────────────────────────────────────
                if let runner = benchmarkRunner {
                    NavigationLink(destination: BenchmarkView(runner: runner)) {
                        HStack {
                            Image(systemName: "speedometer")
                                .foregroundStyle(.purple)
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Run Hardware Benchmark")
                                    .font(.subheadline.weight(.medium))
                                    .foregroundStyle(.primary)
                                Text("A/B/C compute-unit comparison — 24 queries × 3 configs")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Image(systemName: "chevron.right")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.tertiary)
                        }
                        .padding(14)
                        .background(.regularMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    .buttonStyle(.plain)
                }

                // ── Per-stage timing + Mac comparison ────────────────────
                if let timing = model.lastTiming {
                    TimingComparisonSection(timing: timing)
                }

                SettingsSection(title: "Scheduler", icon: "waveform.path.ecg") {
                    SchedulerPicker(model: model)
                }
                SettingsSection(title: "Generation Parameters", icon: "slider.horizontal.3") {
                    VStack(alignment: .leading, spacing: 14) {
                        StepsSlider(numSteps: $settings.numSteps)
                        Divider()
                        CFGSettings(enableCFG: $settings.enableCFG, guidanceScale: $settings.guidanceScale)
                    }
                }
            }
            .padding(20)
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Settings")
        .navigationBarTitleDisplayMode(.large)
    }
}

// MARK: - Timing Comparison Section

private struct TimingComparisonSection: View {
    let timing: MobileOGenerator.TimingInfo

    // Mac M5 MPS baseline (steady-state, warmed), seconds.
    // From mobileo_hardware_aware_report.md §4.2 G2/G3 measurements.
    // DiT shown per actual step rate (237 ms/step × 15 = 3.555s for 15-step run).
    private let macBaseline: [(stage: String, mac: Double)] = [
        ("LLM Forward",         0.076),
        ("Connector",           0.004),
        ("DiT 20-step",         4.734),
        ("VAE Decode",          0.875),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Stage Timings — iPhone vs Mac M5", systemImage: "timer")
                .font(.title3.weight(.semibold))
                .foregroundStyle(.primary)

            VStack(alignment: .leading, spacing: 0) {
                // Header
                HStack {
                    Text("Stage").font(.caption.weight(.semibold)).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading)
                    Text("iPhone").font(.caption.weight(.semibold)).foregroundStyle(.blue).frame(width: 72, alignment: .trailing)
                    Text("Mac M5").font(.caption.weight(.semibold)).foregroundStyle(.secondary).frame(width: 72, alignment: .trailing)
                    Text("Δ").font(.caption.weight(.semibold)).foregroundStyle(.secondary).frame(width: 44, alignment: .trailing)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(Color(.secondarySystemGroupedBackground))

                Divider()

                // Stage rows
                let iPhoneTimes: [Double] = [
                    timing.llmTime,
                    timing.connectorTime,
                    timing.diffusionTime,
                    timing.vaeTime
                ]

                ForEach(Array(zip(macBaseline, iPhoneTimes).enumerated()), id: \.offset) { _, pair in
                    let (stage, iphone) = pair
                    let mac = stage.mac
                    let ratio = mac / max(iphone, 0.001)
                    let faster = iphone < mac

                    HStack {
                        Text(stage.stage)
                            .font(.subheadline)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Text(formatMs(iphone))
                            .font(.subheadline.monospacedDigit())
                            .foregroundStyle(.blue)
                            .frame(width: 72, alignment: .trailing)
                        Text(formatMs(mac))
                            .font(.subheadline.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .frame(width: 72, alignment: .trailing)
                        Text(faster ? String(format: "%.1f×↑", ratio) : String(format: "%.1f×↓", 1/ratio))
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(faster ? .green : .orange)
                            .frame(width: 44, alignment: .trailing)
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)

                    if stage.stage != macBaseline.last?.stage {
                        Divider().padding(.horizontal, 14)
                    }
                }

                Divider()

                // Total row
                HStack {
                    Text("Total").font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Text(formatMs(timing.totalTime))
                        .font(.subheadline.monospacedDigit().weight(.semibold))
                        .foregroundStyle(.blue)
                        .frame(width: 72, alignment: .trailing)
                    Text(formatMs(macBaseline.map(\.mac).reduce(0,+)))
                        .font(.subheadline.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(width: 72, alignment: .trailing)
                    let totalRatio = macBaseline.map(\.mac).reduce(0,+) / max(timing.totalTime, 0.001)
                    Text(timing.totalTime < macBaseline.map(\.mac).reduce(0,+)
                         ? String(format: "%.1f×↑", totalRatio)
                         : String(format: "%.1f×↓", 1/totalRatio))
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(timing.totalTime < macBaseline.map(\.mac).reduce(0,+) ? .green : .orange)
                        .frame(width: 44, alignment: .trailing)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(Color(.secondarySystemGroupedBackground))
            }
            .background(.regularMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .shadow(color: .black.opacity(0.04), radius: 6, y: 2)

            Text("Mac M5 baseline = MPS fp16, 20 DiT steps. ↑ = iPhone faster.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private func formatMs(_ seconds: Double) -> String {
        if seconds >= 1 { return String(format: "%.1fs", seconds) }
        return String(format: "%dms", Int(seconds * 1000))
    }
}

// MARK: - Hardware Status Section

private struct HardwareStatusSection: View {

    private let schedule = HardwareInfo.componentSchedule()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Hardware-Aware Scheduling", systemImage: "cpu.fill")
                .font(.title3.weight(.semibold))
                .foregroundStyle(.primary)

            VStack(alignment: .leading, spacing: 0) {
                // Device row
                HStack(spacing: 10) {
                    Image(systemName: "iphone")
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                        .frame(width: 20)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(HardwareInfo.chipName)
                            .font(.subheadline.weight(.medium))
                        Text(String(format: "%.1f GB RAM  ·  GPU: %@", HardwareInfo.ramGB, HardwareInfo.gpuName))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    // ANE badge
                    if HardwareInfo.hasANE {
                        Text("ANE ✓")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(Color.green)
                            .clipShape(Capsule())
                    }
                }
                .padding(14)

                Divider().padding(.horizontal, 14)

                // Component rows
                ForEach(schedule, id: \.component) { row in
                    HStack(spacing: 10) {
                        Image(systemName: iconFor(row.device))
                            .font(.system(size: 13))
                            .foregroundStyle(colorFor(row.device))
                            .frame(width: 20)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(row.component)
                                .font(.subheadline.weight(.medium))
                            Text(row.detail)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text(row.device)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(colorFor(row.device))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(colorFor(row.device).opacity(0.12))
                            .clipShape(Capsule())
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)

                    if row.component != schedule.last?.component {
                        Divider().padding(.horizontal, 14)
                    }
                }
            }
            .background(.regularMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .shadow(color: .black.opacity(0.04), radius: 6, y: 2)
        }
    }

    private func iconFor(_ device: String) -> String {
        if device.contains("ANE") { return "bolt.fill" }
        if device.contains("GPU") { return "gpu.amd.fill" }
        return "memorychip"
    }

    private func colorFor(_ device: String) -> Color {
        if device.contains("ANE") { return .green }
        if device.contains("GPU") { return .blue }
        return .orange
    }
}

// MARK: - Section Container

private struct SettingsSection<Content: View>: View {
    let title: String
    let icon: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title, systemImage: icon)
                .font(.title3.weight(.semibold))
                .foregroundStyle(.primary)

            VStack(alignment: .leading, spacing: 14) {
                content()
            }
            .padding(18)
            .background(.regularMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .shadow(color: .black.opacity(0.04), radius: 6, y: 2)
        }
    }
}

// MARK: - Scheduler Picker

private struct SchedulerPicker: View {
    @Bindable var model: MobileOModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(MobileOGenerator.SchedulerType.allCases, id: \.self) { schedulerType in
                Button {
                    guard model.schedulerType != schedulerType else { return }
                    model.schedulerType = schedulerType
                    Task { await model.loadWithScheduler(schedulerType) }
                } label: {
                    HStack(spacing: 12) {
                        Image(systemName: model.schedulerType == schedulerType ? "circle.inset.filled" : "circle")
                            .font(.system(size: 16))
                            .foregroundStyle(model.schedulerType == schedulerType ? .purple : .secondary)

                        VStack(alignment: .leading, spacing: 2) {
                            Text(schedulerType.rawValue)
                                .font(.subheadline.weight(.medium))
                                .foregroundStyle(.primary)
                            Text(schedulerType.description)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }

                        Spacer()
                    }
                    .padding(.vertical, 6)
                    .padding(.horizontal, 10)
                    .background(model.schedulerType == schedulerType ? Color.purple.opacity(0.1) : Color.clear)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .buttonStyle(.plain)
            }
        }
    }
}

// MARK: - Steps Slider

private struct StepsSlider: View {
    @Binding var numSteps: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Inference Steps")
                    .font(.subheadline.weight(.medium))
                Spacer()
                Text("\(Int(numSteps))")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(.thinMaterial)
                    .clipShape(Capsule())
            }
            Slider(value: $numSteps, in: 1...50, step: 1)
                .tint(.blue)
        }
    }
}

// MARK: - CFG Settings

private struct CFGSettings: View {
    @Binding var enableCFG: Bool
    @Binding var guidanceScale: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle(isOn: $enableCFG) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Classifier-Free Guidance (CFG)")
                        .font(.subheadline.weight(.medium))
                    Text("2× slower but significantly better quality and prompt following")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .tint(.orange)

            if enableCFG {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Guidance Scale")
                            .font(.subheadline.weight(.medium))
                        Spacer()
                        Text(String(format: "%.1f", guidanceScale))
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 4)
                            .background(.thinMaterial)
                            .clipShape(Capsule())
                    }
                    Slider(value: $guidanceScale, in: 1.0...2.0, step: 0.1)
                        .tint(.purple)
                    Text("Higher = stronger prompt following")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
}
