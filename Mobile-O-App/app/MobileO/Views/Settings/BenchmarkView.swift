import SwiftUI

/// Triggers a full A/B/C benchmark across compute-unit configurations
/// and shows live progress + final results.
struct BenchmarkView: View {
    @State var runner: BenchmarkRunner

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Label("Hardware-Aware Benchmark", systemImage: "speedometer")
                    .font(.title3.weight(.semibold))

                Text("""
                Runs 24 queries × 3 compute-unit configs = 72 inferences.
                Measures latency, tokens/sec, POPE accuracy, cross-config agreement.
                Saves JSON to the app's Documents folder for export via Xcode.
                """)
                .font(.caption).foregroundStyle(.secondary)

                if runner.running {
                    VStack(alignment: .leading, spacing: 6) {
                        ProgressView(value: runner.progress)
                            .progressViewStyle(.linear).tint(.purple)
                        Text(runner.currentConfig)
                            .font(.subheadline.weight(.medium))
                        Text(runner.currentQuery)
                            .font(.caption).foregroundStyle(.secondary)
                            .lineLimit(2)
                        Text(runner.statusMessage)
                            .font(.caption2).foregroundStyle(.tertiary)
                    }
                    .padding(12)
                    .background(.regularMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                } else {
                    Button(action: { Task { await runner.runFullBenchmark() } }) {
                        Label("Run Full Benchmark", systemImage: "play.fill")
                            .frame(maxWidth: .infinity).frame(height: 48)
                            .background(.purple)
                            .foregroundStyle(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    .buttonStyle(.plain)
                }

                if let report = runner.lastReport {
                    Divider().padding(.vertical, 6)
                    Text("Last Run").font(.subheadline.weight(.semibold))
                    Text(report.timestamp).font(.caption).foregroundStyle(.secondary)
                    Text("Device: \(report.device)  ·  RAM: \(report.ram_gb, specifier: "%.1f") GB")
                        .font(.caption).foregroundStyle(.secondary)
                    if let url = runner.lastReportPath {
                        Text("Saved: \(url.lastPathComponent)")
                            .font(.caption2).foregroundStyle(.tertiary).lineLimit(2)
                    }

                    VStack(alignment: .leading, spacing: 0) {
                        HStack {
                            Text("Config").font(.caption.weight(.semibold))
                                .frame(maxWidth: .infinity, alignment: .leading)
                            Text("Tok/s").font(.caption.weight(.semibold))
                                .frame(width: 64, alignment: .trailing)
                            Text("Lat").font(.caption.weight(.semibold))
                                .frame(width: 64, alignment: .trailing)
                            Text("POPE").font(.caption.weight(.semibold))
                                .frame(width: 60, alignment: .trailing)
                        }
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background(Color(.secondarySystemGroupedBackground))
                        Divider()
                        ForEach(report.configs, id: \.config) { s in
                            HStack {
                                Text(s.config.prefix(28))
                                    .font(.caption2).foregroundStyle(.primary)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                Text(String(format: "%.1f", s.aggregateTps))
                                    .font(.caption2.monospacedDigit())
                                    .frame(width: 64, alignment: .trailing)
                                Text(String(format: "%.0fms", s.medianLatencyMs))
                                    .font(.caption2.monospacedDigit())
                                    .frame(width: 64, alignment: .trailing)
                                Text(String(format: "%.0f%%", s.popeAccuracy * 100))
                                    .font(.caption2.monospacedDigit().weight(.semibold))
                                    .frame(width: 60, alignment: .trailing)
                                    .foregroundStyle(s.popeAccuracy >= 0.8 ? .green :
                                                     s.popeAccuracy >= 0.5 ? .orange : .red)
                            }
                            .padding(.horizontal, 10).padding(.vertical, 6)
                            Divider()
                        }
                    }
                    .background(.regularMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 12))

                    // Agreement table
                    if !report.agreement.isEmpty {
                        Text("Cross-config agreement").font(.subheadline.weight(.semibold)).padding(.top, 8)
                        VStack(alignment: .leading, spacing: 4) {
                            ForEach(report.agreement.sorted(by: { $0.key < $1.key }), id: \.key) { kv in
                                HStack {
                                    Text(kv.key).font(.caption2).lineLimit(1)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                    Text("Jacc \(String(format: "%.2f", kv.value.jaccardMean))")
                                        .font(.caption2.monospacedDigit())
                                    Text("Y/N \(String(format: "%.0f%%", kv.value.ynAgreement*100))")
                                        .font(.caption2.monospacedDigit().weight(.semibold))
                                        .foregroundStyle(.purple)
                                }
                                .padding(.horizontal, 10).padding(.vertical, 4)
                            }
                        }
                        .background(.regularMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                }

                Spacer(minLength: 40)
            }
            .padding(20)
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Benchmark")
    }
}
