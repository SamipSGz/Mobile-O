import SwiftUI
import PhotosUI
import MLXLMCommon
import MLXVLM
import Tokenizers
import UIKit

/// Root view of Mobile-O. Manages the chat interface, model lifecycle, and user input routing.
///
/// Responsibilities:
/// - Loads a single shared FastVLM container on launch and distributes it to three sub-models
///   (generation, understanding, text chat).
/// - Presents either a `WelcomeView` (empty state) or `ConversationView` (active chat).
/// - Routes user prompts to the correct pipeline based on keyword prefix:
///   `"generate …"` → image generation, otherwise → text/understanding.
/// - Hosts the floating `InputBar`, toolbar navigation (camera, settings), and a loading overlay.
struct ContentView: View {

    /// Directory containing downloaded model weights. Passed from the download gate.
    let modelDirectory: URL

    // MARK: - View Models

    @State private var viewModel = ChatViewModel()
    @State private var settingsViewModel = SettingsViewModel()

    // MARK: - Shared Model State (loaded once, reused across sub-models)

    @State private var sharedContainer: ModelContainer?
    @State private var sharedFastVLM: FastVLM?
    @State private var sharedTokenizer: Tokenizer?
    @State private var benchmarkRunner: BenchmarkRunner?

    // MARK: - UI State

    @State private var loadingStage = ""
    @State private var prompt = ""
    @State private var selectedImage: PlatformImage?
    @State private var photoPickerItem: PhotosPickerItem?
    @State private var showImagePicker = false
    @State private var showValidationError = false
    @State private var validationMessage = ""

    /// True while any of the three sub-models are still loading.
    private var modelsLoading: Bool {
        guard let gen = viewModel.generationModel,
              let understand = viewModel.understandingModel,
              let chat = viewModel.textChatModel else { return true }

        func isLoading(_ info: String) -> Bool { info.contains("Loading") || info.isEmpty }
        return isLoading(gen.modelInfo) || isLoading(understand.modelInfo) || isLoading(chat.modelInfo)
    }

    // MARK: - Body

    var body: some View {
        NavigationStack {
            ZStack(alignment: .bottom) {
                chatScrollView
                inputBar
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Mobile-O")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { toolbarItems }
            .overlay { if modelsLoading { LoadingOverlay(loadingStage: loadingStage) } }
            .disabled(modelsLoading)
            .alert("Invalid Prompt", isPresented: $showValidationError) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(validationMessage)
            }
            .task { await loadModels() }
            .onChange(of: prompt) { _, newValue in
                autoClearImageForGenerateMode(newValue)
            }
        }
    }

    // MARK: - Subviews

    /// Scrollable area showing either the welcome screen or the conversation history.
    private var chatScrollView: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 0) {
                if viewModel.conversation.isEmpty {
                    WelcomeView(
                        prompt: $prompt,
                        showImagePicker: $showImagePicker,
                        selectedImage: $selectedImage,
                        hasImage: selectedImage != nil
                    )
                } else {
                    ConversationView(
                        conversation: viewModel.conversation,
                        isGenerating: viewModel.isGenerating,
                        generationModel: viewModel.generationModel
                    )
                    .frame(maxHeight: .infinity)
                }
            }
            .contentShape(Rectangle())
            .onTapGesture { dismissKeyboard() }
            .safeAreaInset(edge: .bottom) {
                Color.clear.frame(height: selectedImage != nil ? 240 : 120)
            }
        }
        .photosPicker(isPresented: $showImagePicker, selection: $photoPickerItem, matching: .images)
        .onChange(of: photoPickerItem) { _, newItem in handlePhotoPick(newItem) }
        .scrollDismissesKeyboard(.interactively)
    }

    /// Floating input bar pinned to the bottom of the screen.
    private var inputBar: some View {
        InputBar(
            prompt: $prompt,
            selectedImage: $selectedImage,
            photoPickerItem: $photoPickerItem,
            conversation: viewModel.conversation,
            isGenerating: viewModel.isGenerating,
            onSend: sendMessage,
            onCancel: { viewModel.cancelGeneration() },
            onClearConversation: clearConversation,
            onReset: clearImage
        )
    }

    /// Toolbar containing camera, clear-conversation, and settings buttons.
    @ToolbarContentBuilder
    private var toolbarItems: some ToolbarContent {
        ToolbarItem(placement: .primaryAction) {
            HStack(spacing: 16) {
                NavigationLink(destination: CameraUnderstandingView(understandingModel: $viewModel.understandingModel)) {
                    Image(systemName: "camera.fill").imageScale(.large)
                }

                if !viewModel.conversation.isEmpty {
                    Button(action: clearConversation) {
                        Image(systemName: "trash").imageScale(.large)
                    }
                }

                if let generationModel = viewModel.generationModel {
                    NavigationLink(destination: SettingsView(model: generationModel, settings: settingsViewModel, benchmarkRunner: benchmarkRunner)) {
                        Image(systemName: "gearshape.fill").imageScale(.large)
                    }
                }
            }
        }
    }

    // MARK: - Model Loading

    /// Loads the shared FastVLM container once and initializes the three sub-models in parallel.
    private func loadModels() async {
        guard sharedContainer == nil || sharedFastVLM == nil || sharedTokenizer == nil ||
              viewModel.generationModel == nil || viewModel.understandingModel == nil || viewModel.textChatModel == nil
        else { return }

        do {
            let llmDirectory = modelDirectory.appendingPathComponent("llm")

            // Copy bundled config files into the downloaded directories if missing
            let preprocDest = llmDirectory.appendingPathComponent("preprocessor_config.json")
            if !FileManager.default.fileExists(atPath: preprocDest.path),
               let bundled = Bundle.main.url(forResource: "preprocessor_config", withExtension: "json") {
                try? FileManager.default.copyItem(at: bundled, to: preprocDest)
            }
            // Copy scheduler_config.json to model root so MobileOGenerator can find it
            let schedulerDest = modelDirectory.appendingPathComponent("scheduler_config.json")
            if !FileManager.default.fileExists(atPath: schedulerDest.path),
               let bundled = Bundle.main.url(forResource: "scheduler_config", withExtension: "json") {
                try? FileManager.default.copyItem(at: bundled, to: schedulerDest)
            }

            let config = ModelConfiguration(directory: llmDirectory)

            // Point FastVLM to the downloaded model directory for config + vision encoder
            FastVLM.customModelDirectory = llmDirectory

            await MainActor.run { loadingStage = "Step 1: Registering model..." }
            FastVLM.register(modelFactory: VLMModelFactory.shared)

            await MainActor.run { loadingStage = "Step 2: Loading config..." }
            // Small delay so the UI updates before the next blocking call
            try? await Task.sleep(nanoseconds: 100_000_000)

            await MainActor.run { loadingStage = "Loading MLX weights — please wait..." }

            // Ticker runs on a completely detached thread independent of model loading
            let startTime = Date()
            let ticker = Task.detached(priority: .background) {
                while !Task.isCancelled {
                    try? await Task.sleep(nanoseconds: 500_000_000)
                    let elapsed = Int(Date().timeIntervalSince(startTime))
                    await MainActor.run {
                        loadingStage = "Loading MLX weights... \(elapsed)s"
                    }
                }
            }

            // Load on detached high-priority thread
            let container: ModelContainer
            do {
                container = try await Task.detached(priority: .userInitiated) {
                    try await VLMModelFactory.shared.loadContainer(configuration: config) { progress in
                        let pct = Int(progress.fractionCompleted * 100)
                        Task { @MainActor in
                            loadingStage = "MLX weights: \(pct)%"
                        }
                    }
                }.value
            } catch {
                ticker.cancel()
                await MainActor.run { loadingStage = "Error: \(error.localizedDescription)" }
                // Wait 3s so user can read the error, then rethrow
                try? await Task.sleep(nanoseconds: 3_000_000_000)
                throw error
            }

            ticker.cancel()

            // Extract the FastVLM instance and tokenizer from the loaded container
            var loadedVLM: FastVLM?
            var loadedTokenizer: Tokenizer?

            try await container.perform { context in
                guard let vlm = context.model as? FastVLM else {
                    throw NSError(domain: "ContentView", code: -1,
                                  userInfo: [NSLocalizedDescriptionKey: "Failed to cast to FastVLM model"])
                }
                loadedVLM = vlm
                loadedTokenizer = context.tokenizer
            }

            guard let fastVLM = loadedVLM, let tokenizer = loadedTokenizer else {
                throw NSError(domain: "ContentView", code: -1,
                              userInfo: [NSLocalizedDescriptionKey: "Failed to extract FastVLM components"])
            }

            sharedContainer = container
            sharedFastVLM = fastVLM
            sharedTokenizer = tokenizer

            // Create benchmark runner that can reach into FastVLM + container
            benchmarkRunner = BenchmarkRunner(
                modelDirectory: modelDirectory,
                container: container,
                fastVLM: fastVLM
            )

            await MainActor.run { loadingStage = "Warming up vision encoder..." }
            // Warmup on a background thread to avoid blocking the UI
            await Task.detached(priority: .userInitiated) { fastVLM.warmup() }.value

            await MainActor.run { loadingStage = "Loading DiT transformer + VAE..." }
            // Create and load the three sub-models sharing a single FastVLM backbone
            let generateModel = MobileOModel(sharedFastVLM: fastVLM, sharedTokenizer: tokenizer, modelDirectory: modelDirectory)
            let understandModel = ImageUnderstandingModel(container: container)
            let chatModel = TextChatModel(container: container)

            async let g: Void = generateModel.load()
            async let u: Void = understandModel.load()
            async let c: Void = chatModel.load()
            _ = await (g, u, c)
            await MainActor.run { loadingStage = "" }

            viewModel.generationModel = generateModel
            viewModel.understandingModel = understandModel
            viewModel.textChatModel = chatModel

        } catch {
            print("ContentView ERROR: Failed to load models: \(error)")
        }
    }

    // MARK: - Actions

    /// Validates the current prompt, clears the input fields, and dispatches the message.
    private func sendMessage() {
        let currentPrompt = prompt
        let currentImage = selectedImage
        let lowercased = currentPrompt.lowercased().trimmingCharacters(in: .whitespaces)

        // Validate "generate" prompts
        if lowercased == "generate" || lowercased.hasPrefix("generate ") {
            let body = lowercased.hasPrefix("generate ") ? String(currentPrompt.dropFirst(9)).trimmingCharacters(in: .whitespaces) : ""
            if body.isEmpty {
                showValidation("Please provide a description after 'generate'")
                return
            }
        }

        // Clear input
        prompt = ""
        selectedImage = nil
        photoPickerItem = nil

        Task.detached(priority: .userInitiated) {
            await viewModel.sendMessage(
                prompt: currentPrompt,
                selectedImage: currentImage,
                numSteps: Int(settingsViewModel.numSteps),
                guidanceScale: settingsViewModel.guidanceScale,
                enableCFG: settingsViewModel.enableCFG,
                seed: settingsViewModel.seed
            )
        }
    }

    private func clearConversation() {
        withAnimation(.spring(response: 0.35, dampingFraction: 0.75)) {
            viewModel.conversation.clear()
        }
    }

    private func clearImage() {
        withAnimation(.spring(response: 0.35, dampingFraction: 0.75)) {
            selectedImage = nil
            photoPickerItem = nil
        }
    }

    // MARK: - Helpers

    /// When the user types "generate …" while an image is attached, auto-clear the image
    /// so the prompt routes to generation rather than understanding.
    private func autoClearImageForGenerateMode(_ newValue: String) {
        let lowercased = newValue.lowercased()
        let isGenerate = lowercased == "generate" || lowercased.hasPrefix("generate ")

        guard isGenerate, selectedImage != nil else { return }
        withAnimation(.spring(response: 0.35, dampingFraction: 0.75)) {
            selectedImage = nil
            photoPickerItem = nil
            viewModel.understandingModel?.response = ""
        }
    }

    /// Loads and downsamples a picked photo on a background thread.
    private func handlePhotoPick(_ item: PhotosPickerItem?) {
        Task.detached(priority: .userInitiated) {
            guard let data = try? await item?.loadTransferable(type: Data.self),
                  let image = UIImage(data: data) else { return }
            let downsampledImage = downsampleImage(image, maxDimension: 2048)
            await MainActor.run {
                withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                    selectedImage = downsampledImage
                }
            }
        }
    }

    private func showValidation(_ message: String) {
        validationMessage = message
        showValidationError = true
    }

    private func dismissKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }

    /// Downsamples an image so its longest edge is at most `maxDimension` points.
    nonisolated private func downsampleImage(_ image: UIImage, maxDimension: CGFloat) -> UIImage {
        let size = image.size
        guard max(size.width, size.height) > maxDimension else { return image }

        let scale = maxDimension / max(size.width, size.height)
        let newSize = CGSize(width: size.width * scale, height: size.height * scale)

        UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
        defer { UIGraphicsEndImageContext() }

        image.draw(in: CGRect(origin: .zero, size: newSize))
        return UIGraphicsGetImageFromCurrentImageContext() ?? image
    }
}

#Preview {
    ContentView(modelDirectory: URL(fileURLWithPath: Bundle.main.resourcePath ?? "/tmp"))
}
