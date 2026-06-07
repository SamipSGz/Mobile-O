import SwiftUI

/// Root gate view that routes to the appropriate download screen based on `ModelDownloadManager.state`.
struct ModelDownloadGateView: View {
    @Bindable var downloadManager: ModelDownloadManager
    @State private var networkMonitor = NetworkMonitor()

    var body: some View {
        Group {
            switch downloadManager.state {
            case .idle:
                DownloadPermissionView(
                    downloadManager: downloadManager,
                    networkMonitor: networkMonitor
                )
            case .downloading, .paused, .compiling:
                DownloadProgressView(downloadManager: downloadManager)
            case .completed:
                DownloadProgressView(downloadManager: downloadManager)
            case .failed(let message):
                DownloadErrorView(message: message, onRetry: { downloadManager.retry() })
            }
        }
        .animation(.easeInOut(duration: 0.3), value: downloadManager.state)
    }
}

// MARK: - Bundle Compile View (shown when models are bundled, no download needed)

private struct BundleCompileView: View {
    var body: some View {
        VStack(spacing: 28) {
            Spacer()
            Image(systemName: "cpu.fill")
                .font(.system(size: 60))
                .foregroundStyle(.purple)
            Text("Preparing Models")
                .font(.system(size: 26, weight: .bold, design: .rounded))
            Text("Compiling on-device for your chip.\nThis takes a few minutes — only happens once.")
                .font(.system(size: 15))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
            ProgressView()
                .scaleEffect(1.4)
                .tint(.purple)
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemGroupedBackground))
    }
}

// MARK: - Error View

private struct DownloadErrorView: View {
    let message: String
    let onRetry: () -> Void

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 56))
                .foregroundStyle(.orange)

            Text("Download Failed")
                .font(.system(size: 24, weight: .bold, design: .rounded))

            Text(message)
                .font(.system(size: 15))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)

            Button(action: onRetry) {
                Label("Retry Download", systemImage: "arrow.clockwise")
                    .font(.system(size: 17, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .frame(height: 54)
                    .background(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(.purple)
                    )
            }
            .padding(.horizontal, 32)

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemGroupedBackground))
    }
}
