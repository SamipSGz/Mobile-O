import CoreML
import Foundation
import OSLog

/// Signposter for split-model inner stages — shows p1 vs p2 timing in Instruments
private let splitSignposter = OSSignposter(
    subsystem: "com.samip.mobileo",
    category: "Split"
)

// MARK: - Protocol-Based Model Wrappers

extension MobileOGenerator {

    // MARK: - Transformer Protocol

    /// Unified API for SANA DiT transformer noise prediction
    protocol TransformerModel {
        func predict(
            latent: MLMultiArray,
            timestep: MLMultiArray,
            encoderHiddenStates: MLMultiArray,
            encoderAttentionMask: MLMultiArray
        ) async throws -> MLMultiArray
    }

    // MARK: - VAE Protocol

    /// Unified API for SANA VAE latent-to-image decoding
    protocol VAEModel {
        func decode(latent: MLMultiArray) async throws -> MLMultiArray
    }

    // MARK: - FP32 Transformer Wrapper

    /// FP32 SANA transformer with dynamic output key detection.
    /// Loads the model directly from a compiled `.mlmodelc` URL.
    class FP32Transformer: TransformerModel {
        private let model: MLModel
        private let outputKey: String

        init(modelURL: URL, configuration: MLModelConfiguration) throws {
            self.model = try MLModel(contentsOf: modelURL, configuration: configuration)

            let outputs = model.modelDescription.outputDescriptionsByName
            guard let firstOutput = outputs.first else {
                throw NSError(domain: "MobileOGenerator", code: -1,
                            userInfo: [NSLocalizedDescriptionKey: "No output found in transformer model"])
            }
            self.outputKey = firstOutput.key
        }

        func predict(
            latent: MLMultiArray,
            timestep: MLMultiArray,
            encoderHiddenStates: MLMultiArray,
            encoderAttentionMask: MLMultiArray
        ) async throws -> MLMultiArray {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "latent": MLFeatureValue(multiArray: latent),
                "timestep": MLFeatureValue(multiArray: timestep),
                "encoder_hidden_states": MLFeatureValue(multiArray: encoderHiddenStates),
                "encoder_attention_mask": MLFeatureValue(multiArray: encoderAttentionMask)
            ])

            let output = try await model.prediction(from: input)

            guard let result = output.featureValue(for: outputKey)?.multiArrayValue else {
                throw NSError(domain: "MobileOGenerator", code: -2,
                            userInfo: [NSLocalizedDescriptionKey: "Failed to get output '\(outputKey)' from transformer"])
            }

            return result
        }
    }

    // MARK: - FP32 VAE Wrapper

    /// FP32 SANA VAE decoder with dynamic output key detection.
    /// Loads the model directly from a compiled `.mlmodelc` URL.
    class FP32VAE: VAEModel {
        private let model: MLModel
        private let outputKey: String

        init(modelURL: URL, configuration: MLModelConfiguration) throws {
            self.model = try MLModel(contentsOf: modelURL, configuration: configuration)

            let outputs = model.modelDescription.outputDescriptionsByName
            guard let firstOutput = outputs.first else {
                throw NSError(domain: "MobileOGenerator", code: -1,
                            userInfo: [NSLocalizedDescriptionKey: "No output found in vae_decoder model"])
            }
            self.outputKey = firstOutput.key
        }

        func decode(latent: MLMultiArray) async throws -> MLMultiArray {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "latent": MLFeatureValue(multiArray: latent)
            ])
            let output = try await model.prediction(from: input)

            guard let result = output.featureValue(for: outputKey)?.multiArrayValue else {
                throw NSError(domain: "MobileOGenerator", code: -2,
                            userInfo: [NSLocalizedDescriptionKey: "Failed to get output '\(outputKey)' from VAE decoder"])
            }

            return result
        }
    }

    // MARK: - ANE Split Transformer Wrapper

    /// Chains transformer_ane_p1 → transformer_ane_p2.
    ///
    /// P1 inputs:  latent, timestep, encoder_hidden_states, encoder_attention_mask
    /// P1 outputs: h, attn_bias, enc, modulation, embedded_t  (5 intermediate tensors)
    /// P2 inputs:  h, attn_bias, enc, modulation, embedded_t  (exactly P1's outputs)
    /// P2 output:  sample  (final noise prediction)
    class SplitANETransformer: TransformerModel {
        private let p1: MLModel
        private let p2: MLModel

        init(p1URL: URL, p2URL: URL, configuration: MLModelConfiguration) throws {
            self.p1 = try MLModel(contentsOf: p1URL, configuration: configuration)
            self.p2 = try MLModel(contentsOf: p2URL, configuration: configuration)
        }

        func predict(
            latent: MLMultiArray,
            timestep: MLMultiArray,
            encoderHiddenStates: MLMultiArray,
            encoderAttentionMask: MLMultiArray
        ) async throws -> MLMultiArray {
            // Stage 1: patch embedding + first half of DiT blocks
            let p1State = splitSignposter.beginInterval("DiT_p1",
                id: splitSignposter.makeSignpostID(), "first half on ANE")
            let input1 = try MLDictionaryFeatureProvider(dictionary: [
                "latent":                   MLFeatureValue(multiArray: latent),
                "timestep":                 MLFeatureValue(multiArray: timestep),
                "encoder_hidden_states":    MLFeatureValue(multiArray: encoderHiddenStates),
                "encoder_attention_mask":   MLFeatureValue(multiArray: encoderAttentionMask)
            ])
            let out1 = try await p1.prediction(from: input1)
            splitSignposter.endInterval("DiT_p1", p1State)

            // P1 produces 5 intermediate tensors that feed directly into P2
            guard let h          = out1.featureValue(for: "h")?.multiArrayValue,
                  let attnBias   = out1.featureValue(for: "attn_bias")?.multiArrayValue,
                  let enc        = out1.featureValue(for: "enc")?.multiArrayValue,
                  let modulation = out1.featureValue(for: "modulation")?.multiArrayValue,
                  let embeddedT  = out1.featureValue(for: "embedded_t")?.multiArrayValue else {
                // Fallback: try first output key if names differ
                let keys = p1.modelDescription.outputDescriptionsByName.keys.sorted()
                throw NSError(domain: "SplitANETransformer", code: -1,
                              userInfo: [NSLocalizedDescriptionKey:
                                "P1 output keys: \(keys). Expected h, attn_bias, enc, modulation, embedded_t"])
            }

            // Stage 2: second half of DiT blocks → noise prediction
            let p2State = splitSignposter.beginInterval("DiT_p2",
                id: splitSignposter.makeSignpostID(), "second half on ANE")
            let input2 = try MLDictionaryFeatureProvider(dictionary: [
                "h":          MLFeatureValue(multiArray: h),
                "attn_bias":  MLFeatureValue(multiArray: attnBias),
                "enc":        MLFeatureValue(multiArray: enc),
                "modulation": MLFeatureValue(multiArray: modulation),
                "embedded_t": MLFeatureValue(multiArray: embeddedT)
            ])
            let out2 = try await p2.prediction(from: input2)
            splitSignposter.endInterval("DiT_p2", p2State)

            guard let result = out2.featureValue(for: "sample")?.multiArrayValue else {
                let keys = p2.modelDescription.outputDescriptionsByName.keys.sorted()
                throw NSError(domain: "SplitANETransformer", code: -2,
                              userInfo: [NSLocalizedDescriptionKey:
                                "P2 output keys: \(keys). Expected 'sample'"])
            }
            return result
        }
    }

    // MARK: - ANE Split VAE Wrapper

    /// Chains vae_ane_p1 → vae_ane_p2.
    /// P1: latent [1,32,16,16] → hidden [1,1024,32,32]
    /// P2: hidden [1,1024,32,32] → image (pixel output)
    class SplitANEVAE: VAEModel {
        private let p1: MLModel
        private let p2: MLModel

        init(p1URL: URL, p2URL: URL, configuration: MLModelConfiguration) throws {
            self.p1 = try MLModel(contentsOf: p1URL, configuration: configuration)
            self.p2 = try MLModel(contentsOf: p2URL, configuration: configuration)
        }

        func decode(latent: MLMultiArray) async throws -> MLMultiArray {
            // P1: latent → hidden
            let p1State = splitSignposter.beginInterval("VAE_p1",
                id: splitSignposter.makeSignpostID(), "ANE")
            let input1 = try MLDictionaryFeatureProvider(dictionary: [
                "latent": MLFeatureValue(multiArray: latent)
            ])
            let out1 = try await p1.prediction(from: input1)
            splitSignposter.endInterval("VAE_p1", p1State)
            guard let hidden = out1.featureValue(for: "hidden")?.multiArrayValue else {
                let keys = p1.modelDescription.outputDescriptionsByName.keys.sorted()
                throw NSError(domain: "SplitANEVAE", code: -1,
                              userInfo: [NSLocalizedDescriptionKey: "VAE P1 output keys: \(keys). Expected 'hidden'"])
            }

            // P2: hidden → image
            let p2State = splitSignposter.beginInterval("VAE_p2",
                id: splitSignposter.makeSignpostID(), "ANE")
            let input2 = try MLDictionaryFeatureProvider(dictionary: [
                "hidden": MLFeatureValue(multiArray: hidden)
            ])
            let out2 = try await p2.prediction(from: input2)
            splitSignposter.endInterval("VAE_p2", p2State)
            guard let result = out2.featureValue(for: "image")?.multiArrayValue else {
                let keys = p2.modelDescription.outputDescriptionsByName.keys.sorted()
                throw NSError(domain: "SplitANEVAE", code: -2,
                              userInfo: [NSLocalizedDescriptionKey: "VAE P2 output keys: \(keys). Expected 'image'"])
            }
            return result
        }
    }

    // MARK: - Model Factory

    /// Factory for creating model wrappers based on variant
    struct ModelFactory {

        static func createTransformer(
            variant: ModelVariant,
            configuration: MLModelConfiguration,
            modelDirectory: URL
        ) throws -> TransformerModel {
            switch variant {
            case .fp32:
                // Use ANE split transformer — p1 + p2 chain
                let p1URL = modelDirectory.appendingPathComponent("transformer_ane_p1_0.5.mlmodelc")
                let p2URL = modelDirectory.appendingPathComponent("transformer_ane_p2_0.5.mlmodelc")
                if FileManager.default.fileExists(atPath: p1URL.path) &&
                   FileManager.default.fileExists(atPath: p2URL.path) {
                    return try SplitANETransformer(p1URL: p1URL, p2URL: p2URL, configuration: configuration)
                }
                // Fallback to monolithic transformer
                let modelURL = modelDirectory.appendingPathComponent(variant.fileName)
                return try FP32Transformer(modelURL: modelURL, configuration: configuration)
            }
        }

        static func createVAE(
            variant: ModelVariant,
            configuration: MLModelConfiguration,
            modelDirectory: URL
        ) throws -> VAEModel {
            switch variant {
            case .fp32:
                // Use ANE split VAE — p1 + p2 chain
                let p1URL = modelDirectory.appendingPathComponent("vae_ane_p1_0.5.mlmodelc")
                let p2URL = modelDirectory.appendingPathComponent("vae_ane_p2_0.5.mlmodelc")
                if FileManager.default.fileExists(atPath: p1URL.path) &&
                   FileManager.default.fileExists(atPath: p2URL.path) {
                    return try SplitANEVAE(p1URL: p1URL, p2URL: p2URL, configuration: configuration)
                }
                // Fallback to monolithic VAE
                let modelURL = modelDirectory.appendingPathComponent(variant.vaeFileName)
                return try FP32VAE(modelURL: modelURL, configuration: configuration)
            }
        }
    }
}
