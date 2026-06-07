import Foundation
import SwiftUI

import UIKit

/// View model coordinating chat, image generation, and image understanding
@Observable
@MainActor
class ChatViewModel {
    var generationModel: MobileOModel?
    var understandingModel: ImageUnderstandingModel?
    var textChatModel: TextChatModel?

    var conversation = Conversation()

    var isGenerating: Bool {
        generationModel?.running == true || understandingModel?.running == true || textChatModel?.running == true
    }

    // MARK: - Send Message

    nonisolated func sendMessage(
        prompt: String,
        selectedImage: PlatformImage?,
        numSteps: Int,
        guidanceScale: Double,
        enableCFG: Bool,
        seed: UInt64?
    ) async {
        let userMessage = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let hasAttachedImage = selectedImage != nil
        let lowercased = userMessage.lowercased()
        let isImageGenRequest = lowercased == "generate" || lowercased.hasPrefix("generate ")

        guard !userMessage.isEmpty || hasAttachedImage else { return }

        if isImageGenRequest {
            await handleImageGeneration(
                prompt: userMessage, numSteps: numSteps,
                guidanceScale: guidanceScale, enableCFG: enableCFG, seed: seed
            )
        } else if hasAttachedImage, let image = selectedImage {
            await handleImageUnderstanding(prompt: userMessage, image: image)
        } else {
            await handleTextChat(prompt: userMessage)
        }
    }

    // MARK: - Cancel

    func cancelGeneration() {
        let wasImageGenerating = generationModel?.running == true

        generationModel?.cancel()
        understandingModel?.cancel()
        textChatModel?.cancel()
        conversation.isGenerating = false

        if wasImageGenerating {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                _ = conversation.addMessage(role: .assistant, content: .text("⚠️ Image generation interrupted"))
            }
        }
    }

    // MARK: - Helpers

    /// Strip a command keyword (e.g. "generate") from the start of a prompt
    nonisolated private func stripKeyword(_ keyword: String, from prompt: String) -> String? {
        let trimmed = prompt.trimmingCharacters(in: .whitespaces)
        let lowercased = trimmed.lowercased()

        if lowercased.hasPrefix(keyword + " ") {
            let clean = String(trimmed.dropFirst(keyword.count + 1)).trimmingCharacters(in: .whitespaces)
            return clean.isEmpty ? nil : clean
        } else if lowercased == keyword {
            return nil
        }
        return trimmed
    }

    /// Poll a model's streaming response and update the conversation at ~60fps
    private func streamResponse(
        while isRunning: @escaping () -> Bool,
        currentContent: @escaping () -> String
    ) async {
        var lastContent = ""
        var lastUpdateTime = Date()
        let minUpdateInterval: TimeInterval = 0.016

        while isRunning() {
            let content = currentContent()
            let now = Date()

            if content != lastContent && content.count >= lastContent.count &&
               now.timeIntervalSince(lastUpdateTime) >= minUpdateInterval {
                conversation.updateLastMessage(text: content)
                lastContent = content
                lastUpdateTime = now
            }

            try? await Task.sleep(nanoseconds: 100_000_000)
        }

        let finalContent = currentContent()
        if finalContent != lastContent {
            conversation.updateLastMessage(text: finalContent)
        }
    }

    // MARK: - Image Generation

    nonisolated private func handleImageGeneration(
        prompt: String,
        numSteps: Int,
        guidanceScale: Double,
        enableCFG: Bool,
        seed: UInt64?
    ) async {
        let model = await self.generationModel
        guard let model = model else { return }
        guard let cleanPrompt = stripKeyword("generate", from: prompt) else { return }

        await MainActor.run {
            _ = conversation.addMessage(role: .user, content: .text(prompt))
            conversation.isGenerating = true
        }

        let params = MobileOGenerator.GenerationParameters(
            prompt: cleanPrompt,
            numSteps: numSteps,
            guidanceScale: enableCFG ? Float(guidanceScale) : 0.0,
            seed: seed,
            progressCallback: nil
        )

        let generatedImage = await model.generate(params)

        await MainActor.run {
            if let generatedImage {
                _ = conversation.addMessage(
                    role: .assistant, content: .image(generatedImage),
                    totalTime: model.inferenceTime
                )
            } else {
                // Show the error from modelInfo so user can see what failed
                let errorInfo = model.modelInfo
                _ = conversation.addMessage(
                    role: .assistant,
                    content: .text("Generation failed: \(errorInfo.isEmpty ? "unknown error" : errorInfo)")
                )
            }
            conversation.isGenerating = false
        }
    }

    // MARK: - Image Understanding

    nonisolated private func handleImageUnderstanding(prompt: String, image: PlatformImage) async {
        let understandingModel = await self.understandingModel
        guard let understandingModel = understandingModel else { return }

        let assistantMessageId = await MainActor.run {
            understandingModel.response = ""
            _ = conversation.addMessage(role: .user, content: .textWithImage(prompt, image))
            conversation.isGenerating = true
            return conversation.addMessage(role: .assistant, content: .text(""))
        }

        let streamingTask = Task { @MainActor in
            await streamResponse(
                while: { understandingModel.running },
                currentContent: { understandingModel.response }
            )
        }

        let response = await understandingModel.understand(image: image, prompt: prompt)
        await streamingTask.value

        await MainActor.run {
            if !response.isEmpty { conversation.updateLastMessage(text: response) }

            let tokensPerSecond = understandingModel.totalTime > 0
                ? Double(understandingModel.tokensGenerated) / understandingModel.totalTime : 0

            conversation.updateMessageMetrics(
                id: assistantMessageId,
                timeToFirstToken: understandingModel.timeToFirstToken,
                totalTime: understandingModel.totalTime,
                tokensPerSecond: tokensPerSecond
            )
            conversation.isGenerating = false
        }
    }

    // MARK: - Text Chat

    nonisolated private func handleTextChat(prompt: String) async {
        let textChatModel = await self.textChatModel
        guard let textChatModel = textChatModel else { return }

        let (assistantMessageId, messages) = await MainActor.run { () -> (UUID, [[String: String]]) in
            textChatModel.currentResponse = ""
            _ = conversation.addMessage(role: .user, content: .text(prompt))
            conversation.isGenerating = true

            let id = conversation.addMessage(role: .assistant, content: .text(""))
            var msgs = conversation.getMessagesForModel()
            msgs.removeLast()
            return (id, msgs)
        }

        let streamingTask = Task { @MainActor in
            await streamResponse(
                while: { textChatModel.running },
                currentContent: { textChatModel.currentResponse }
            )
        }

        let response = await textChatModel.chat(messages: messages)
        await streamingTask.value

        await MainActor.run {
            if !response.isEmpty { conversation.updateLastMessage(text: response) }

            let tokensPerSecond = textChatModel.totalTime > 0
                ? Double(textChatModel.tokensGenerated) / textChatModel.totalTime : 0

            conversation.updateMessageMetrics(
                id: assistantMessageId,
                timeToFirstToken: textChatModel.timeToFirstToken,
                totalTime: textChatModel.totalTime,
                tokensPerSecond: tokensPerSecond
            )
            conversation.isGenerating = false
        }
    }
}
