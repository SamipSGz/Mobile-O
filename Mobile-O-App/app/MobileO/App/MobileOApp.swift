import SwiftUI

/// App entry point. Gates access behind model download if needed.
@main
struct MobileOApp: App {
    @State private var downloadManager = ModelDownloadManager()

    var body: some Scene {
        WindowGroup {
            if ProcessInfo.processInfo.arguments.contains("--iphone-benchmark") {
                if let bundledModelDirectory = Bundle.main.resourceURL {
                    IPhoneBenchmarkView(
                        modelDirectory: downloadManager.modelsReady
                            ? downloadManager.modelsDirectory
                            : bundledModelDirectory
                    )
                } else {
                    Text("Benchmark failed: bundle resource directory missing")
                }
            } else if downloadManager.modelsReady {
                ContentView(modelDirectory: downloadManager.modelsDirectory)
            } else {
                ModelDownloadGateView(downloadManager: downloadManager)
            }
        }
    }
}
