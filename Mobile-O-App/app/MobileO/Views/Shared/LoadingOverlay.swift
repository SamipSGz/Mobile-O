import SwiftUI

/// Full-screen blurred overlay with a spinner and per-stage progress, shown while models are loading.
struct LoadingOverlay: View {
    var loadingStage: String = ""

    var body: some View {
        ZStack {
            Color.black.opacity(0.001)
                .background(.ultraThinMaterial)
                .ignoresSafeArea()
                .transition(.opacity)

            VStack(spacing: 20) {
                ProgressView()
                    .scaleEffect(1.5)
                    .progressViewStyle(.circular)
                    .tint(.white)

                Text("Loading models...")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(.white)

                if !loadingStage.isEmpty {
                    Text(loadingStage)
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.7))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 40)
                }
            }
            .transition(.scale(scale: 0.9).combined(with: .opacity))
        }
    }
}
