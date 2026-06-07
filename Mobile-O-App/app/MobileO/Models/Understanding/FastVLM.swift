import CoreImage
import CoreML
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import MLXVLM
import Tokenizers

// MARK: - Language

/// Qwen2-based language model components (attention, MLP, decoder layers, LM head).
private enum Language {

    fileprivate class Attention: Module {

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        public init(_ args: FastVLMConfiguration.TextConfiguration) {
            let dim = args.hiddenSize
            self.heads = args.attentionHeads
            self.kvHeads = args.kvHeads
            self.headDim = dim / heads
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            self._rotaryEmbedding.wrappedValue = RoPE(
                dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            let offset = cache?.offset ?? 0
            let mask = mask?[0..., 0 ..< keys.dim(-2)]

            queries = rotaryEmbedding(queries, offset: offset)
            keys = rotaryEmbedding(keys, offset: offset)

            if let cache {
                (keys, values) = cache.update(keys: keys, values: values)
            }

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {

        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate class FastVLMDecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        public init(_ args: FastVLMConfiguration.TextConfiguration) {
            self._attention.wrappedValue = Attention(args)
            self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
        ) -> MLXArray {
            var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            r = mlp(postAttentionLayerNorm(h))
            let out = h + r
            return out
        }
    }

    fileprivate class Qwen2Model: Module {

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        fileprivate let layers: [FastVLMDecoderLayer]
        fileprivate let norm: RMSNorm

        public init(_ args: FastVLMConfiguration.TextConfiguration) {
            precondition(args.vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

            self.layers = (0 ..< args.hiddenLayers)
                .map { _ in
                    FastVLMDecoderLayer(args)
                }
            self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        /// Resolve input to embeddings
        private func resolveInput(_ inputs: MLXArray?, inputEmbedding: MLXArray?) -> MLXArray {
            if let inputEmbedding { return inputEmbedding }
            if let inputs { return embedTokens(inputs) }
            fatalError("one of inputs or inputEmbedding must be non-nil")
        }

        /// Extract array-based attention mask for scaled dot product attention
        private func resolveMask(h: MLXArray, cache: [KVCache]?) -> MLXArray? {
            if case .array(let array) = createAttentionMask(h: h, cache: cache, returnArray: true) {
                return array
            }
            return nil
        }

        public func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> MLXArray {
            var h = resolveInput(inputs, inputEmbedding: inputEmbedding)
            let mask = resolveMask(h: h, cache: cache)

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }

            return norm(h)
        }

        /// Forward pass returning hidden states from all layers for multi-layer fusion
        public func callWithHiddenStates(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> [MLXArray] {
            var h = resolveInput(inputs, inputEmbedding: inputEmbedding)
            let mask = resolveMask(h: h, cache: cache)

            var allHiddenStates: [MLXArray] = []

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
                allHiddenStates.append(h)
            }

            allHiddenStates[allHiddenStates.count - 1] = norm(allHiddenStates.last!)
            return allHiddenStates
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: Qwen2Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int]

        public init(_ args: FastVLMConfiguration.TextConfiguration) {
            self.model = Qwen2Model(args)

            if !args.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
            }

            self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        }

        public func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> LMOutput {
            var out = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }
            return LMOutput(logits: out)
        }
    }
}

// MARK: - Vision

/// CoreML-backed vision encoder that produces image feature embeddings.
private enum Vision {

    fileprivate class VisionModelCoreML {

        let lock = NSLock()
        var _model: MLModel?
        private var _modelURL: URL?
        private var outputKey: String = "image_features"
        private(set) var lastEncodingTime: TimeInterval = 0

        init() {
        }

        func setModelURL(_ url: URL) {
            _modelURL = url
        }

        func load() throws -> MLModel {
            try lock.withLock {
                if let model = _model { return model }
                guard let url = _modelURL else {
                    throw NSError(domain: "VisionModelCoreML", code: -1,
                                 userInfo: [NSLocalizedDescriptionKey: "Vision encoder URL not set"])
                }
                let config = MLModelConfiguration()
                config.computeUnits = .all
                let model = try MLModel(contentsOf: url, configuration: config)
                _model = model
                // Detect output key
                if let firstOutput = model.modelDescription.outputDescriptionsByName.first {
                    outputKey = firstOutput.key
                }
                return model
            }
        }

        public func model() -> MLModel? {
            try? load()
        }

        public func encode(_ image: MLXArray) -> MLXArray {
            let encodeStart = Date()

            guard let mlModel = model() else {
                // Vision encoder failed to load (likely memory pressure) — return zeros
                let seqLen = (image.dim(2) / 16) * (image.dim(3) / 16)
                return MLXArray.zeros([1, seqLen, 3072])
            }

            var (data, strides) = {
                let arrayData = image.asType(.float32).asData(access: .noCopyIfContiguous)
                return (arrayData.data, arrayData.strides)
            }()

            precondition(image.ndim == 4)
            precondition(image.dim(0) == 1)
            precondition(image.dim(1) == 3)

            let h = NSNumber(value: image.dim(2))
            let w = NSNumber(value: image.dim(3))

            let result = data.withUnsafeMutableBytes { (ptr: UnsafeMutableRawBufferPointer) in
                guard let array = try? MLMultiArray(
                    dataPointer: ptr.baseAddress!, shape: [1, 3, h, w], dataType: .float32,
                    strides: strides.map { .init(value: $0) }),
                      let input = try? MLDictionaryFeatureProvider(dictionary: [
                          "images": MLFeatureValue(multiArray: array)
                      ]),
                      let output = try? mlModel.prediction(from: input) else {
                    let seqLen = image.dim(2) / 16 * image.dim(3) / 16
                    return MLXArray.zeros([1, seqLen, 3072])
                }

                guard let imageFeatures = output.featureValue(for: outputKey)?.multiArrayValue else {
                    let seqLen = image.dim(2) / 16 * image.dim(3) / 16
                    return MLXArray.zeros([1, seqLen, 3072])
                }

                let shape = imageFeatures.shape.map { $0.intValue }
                guard shape.count == 3 else {
                    let seqLen = image.dim(2) / 16 * image.dim(3) / 16
                    return MLXArray.zeros([1, seqLen, 3072])
                }

                let count = shape.reduce(1, *)

                if imageFeatures.dataType.rawValue == 65552 { // .float16
                    let float16Ptr = imageFeatures.dataPointer.assumingMemoryBound(to: Float16.self)
                    var float32Data = [Float](repeating: 0, count: count)
                    for i in 0..<count {
                        float32Data[i] = Float(float16Ptr[i])
                    }
                    return MLXArray(float32Data, shape)
                } else {
                    // Float32 output - use directly
                    return imageFeatures.withUnsafeBytes { ptr in
                        MLXArray(ptr, shape, type: Float32.self)
                    }
                }
            }

            lastEncodingTime = Date().timeIntervalSince(encodeStart)

            return result
        }
    }

    fileprivate class VisionModel: Module {

        let model = VisionModelCoreML()

        public override init() {}

        public func callAsFunction(_ hiddenStates: MLXArray, gridThw: [THW]) -> MLXArray {
            model.encode(hiddenStates)
        }

        public func getLastEncodingTime() -> TimeInterval {
            model.lastEncodingTime
        }
    }
}

// MARK: - Processor

/// FastVLM `UserInputProcessor`.
///
/// This is meant to be used with ``FastVLM`` and is typically created by ``VLMModelFactory``.
public class FastVLMProcessor: UserInputProcessor {

    private let config: FastVLMProcessorConfiguration
    private let imageProcessingConfig: FastVLMPreProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: FastVLMPreProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = FastVLMProcessorConfiguration()
        self.imageProcessingConfig = config
        self.tokenizer = tokenizer
    }

    public func preprocess(image: CIImage, processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        // first apply the user requested resizing, etc. if any
        var image = MediaProcessingExtensions.apply(image, processing: processing)

        // image_processing_clip.py
        let size = MediaProcessingExtensions.fitIn(
            image.extent.size, shortestEdge: imageProcessingConfig.size.shortestEdge)
        image = MediaProcessingExtensions.resampleBicubic(image, to: size)

        image = MediaProcessingExtensions.centerCrop(
            image, size: imageProcessingConfig.cropSize.size)

        image = MediaProcessing.normalize(
            image, mean: imageProcessingConfig.imageMeanTuple,
            std: imageProcessingConfig.imageStdTuple)

        let array = MediaProcessingExtensions.asPlanarMLXArray(image)
        return (array, .init(0, array.dim(2), array.dim(3)))
    }

    public func prepare(prompt: UserInput.Prompt, imageTHW: THW?) -> String {
        var messages: [Message]
        switch prompt {
        case .text(let text):
            messages = [["role": "user", "content": text]]
        case .messages(let msgs):
            messages = msgs
        case .chat(let chatMsgs):
            messages = chatMsgs.map { ["role": $0.role.rawValue, "content": $0.content] }
        }
        if (messages[0]["role"] as? String) != "system" {
            messages.insert(["role": "system", "content": "You are a helpful assistant."], at: 0)
        }

        let lastIndex = messages.count - 1
        var lastMessage: String = (messages[lastIndex]["content"] as? String) ?? ""

        // processing_llava.py
        if let imageTHW {
            let height = imageTHW.h
            let width = imageTHW.w
            let patchSize = config.patchSize

            var numImageTokens =
                (height / patchSize) * (width / patchSize) + config.numAdditionalImageTokens

            if config.visionFeatureSelectStrategy == .default {
                numImageTokens -= 1
            }

            let imageTokens = String(repeating: config.imageToken, count: numImageTokens)
            // Place image tokens BEFORE text (matches LLaVA training format)
            lastMessage = imageTokens + "\n" + lastMessage
        }

        messages[lastIndex]["content"] = lastMessage

        return
            messages
            .map {
                "<|im_start|>\($0["role"] ?? "user")\n\($0["content"] ?? "")<|im_end|>"
            }
            .joined(separator: "\n")
            + "\n<|im_start|>assistant\n"
    }

    public func prepare(input: UserInput) throws -> LMInput {
        if input.images.isEmpty {
            let prompt = prepare(prompt: input.prompt, imageTHW: nil)
            let promptTokens = tokenizer.encode(text: prompt)
            let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: promptArray).asType(.int8)
            return LMInput(text: .init(tokens: promptArray, mask: mask), image: nil)
        }

        if input.images.count > 1 {
            throw VLMError.singleImageAllowed
        }

        let (pixels, thw) = try preprocess(
            image: input.images[0].asCIImage(), processing: input.processing)
        let image = LMInput.ProcessedImage(pixels: pixels, frames: [thw])

        let prompt = prepare(prompt: input.prompt, imageTHW: thw)
        let promptTokens = tokenizer.encode(text: prompt)

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)

        return LMInput(text: .init(tokens: promptArray, mask: mask), image: image)
    }

}

// MARK: - Model

private class FastVLMMultiModalProjector: Module, UnaryLayer {

    @ModuleInfo(key: "linear_0") var linear0: Linear
    @ModuleInfo(key: "gelu") var gelu: GELU
    @ModuleInfo(key: "linear_2") var linear2: Linear

    public init(_ config: FastVLMConfiguration) {
        self._linear0.wrappedValue = Linear(
            config.visionConfiguration.hiddenSize,
            config.textConfiguration.hiddenSize,
            bias: true)
        self._gelu.wrappedValue = GELU()
        self._linear2.wrappedValue = Linear(
            config.textConfiguration.hiddenSize,
            config.textConfiguration.hiddenSize,
            bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = linear0(x)
        x = gelu(x)
        x = linear2(x)
        return x
    }
}

/// FastVLM: a Qwen2-based VLM with a custom CoreML vision tower.
///
/// Combines a language model, a CoreML vision encoder, and a multi-modal projector.
/// Typically created by ``VLMModelFactory``.
public class FastVLM: Module, VLMModel, KVCacheDimensionProvider {

    /// Custom model directory set externally (e.g. Application Support/Models/llm/).
    /// When set, `modelConfiguration` uses this instead of the app bundle.
    static public var customModelDirectory: URL?

    static public var modelConfiguration: ModelConfiguration {
        if let customDir = customModelDirectory {
            return ModelConfiguration(directory: customDir)
        }
        let bundle = Bundle(for: FastVLM.self)
        let url = bundle.url(forResource: "config", withExtension: "json")!
            .resolvingSymlinksInPath()
            .deletingLastPathComponent()
        return ModelConfiguration(directory: url)
    }

    static public func register(modelFactory: VLMModelFactory) {
        modelFactory.typeRegistry.registerModelType("llava_qwen2") { url in
            let configuration = try JSONDecoder().decode(
                FastVLMConfiguration.self, from: Data(contentsOf: url))
            return FastVLM(configuration)
        }

        modelFactory.processorRegistry.registerProcessorType("LlavaProcessor") { url, tokenizer in
            let configuration = try JSONDecoder().decode(
                FastVLMPreProcessorConfiguration.self, from: Data(contentsOf: url))
            return FastVLMProcessor(configuration, tokenizer: tokenizer)
        }
    }

    @ModuleInfo(key: "vision_tower") private var visionModel: Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector:
        FastVLMMultiModalProjector

    public let config: FastVLMConfiguration

    public var vocabularySize: Int { config.baseConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public func loraLinearLayers() -> MLXLMCommon.LoRALinearLayers {
        languageModel.model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }

    public init(_ config: FastVLMConfiguration) {
        self.config = config
        self._visionModel.wrappedValue = Vision.VisionModel()
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfiguration)
        self._multiModalProjector.wrappedValue = FastVLMMultiModalProjector(config)
    }

    private func inputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?, gridThw: [THW]?)
        -> MLXArray
    {
        guard let pixelValues, let gridThw else {
            return languageModel.model.embedTokens(inputIds)
        }

        let inputEmbeds = languageModel.model.embedTokens(inputIds)
        let imageFeaturesCoreML = self.visionModel(pixelValues, gridThw: gridThw)
        let imageFeatures = multiModalProjector(imageFeaturesCoreML)

        return mergeInputIdsWithImageFeatures(
            inputIds: inputIds, inputEmbeds: inputEmbeds, imageFeatures: imageFeatures)
    }

    private func mergeInputIdsWithImageFeatures(
        inputIds: MLXArray, inputEmbeds: MLXArray, imageFeatures: MLXArray
    ) -> MLXArray {
        let imageTokenIndex = config.baseConfiguration.imageTokenId

        // Find all image token positions
        var imageIndices = [Int]()
        let flatInputIds = inputIds.reshaped([-1])
        for (i, v) in flatInputIds.asArray(Int.self).enumerated() {
            if v == imageTokenIndex {
                imageIndices.append(i)
            }
        }

        if imageIndices.isEmpty {
            return inputEmbeds
        }

        let imgFeatures: MLXArray
        if imageFeatures.ndim == 2 {
            imgFeatures = imageFeatures.expandedDimensions(axis: 0)
        } else {
            imgFeatures = imageFeatures
        }

        let numPatches = imgFeatures.dim(1)
        let patchesPerImage = numPatches / max(imageIndices.count, 1)

        // Replace image tokens with image patches
        let seqLen = inputEmbeds.dim(1)
        var result: [MLXArray] = []
        var prevEnd = 0

        for (imgIdx, imageTokenPos) in imageIndices.enumerated() {
            if imageTokenPos > prevEnd {
                let textBefore = inputEmbeds[0..., prevEnd..<imageTokenPos, 0...]
                result.append(textBefore)
            }

            let startPatch = imgIdx * patchesPerImage
            let endPatch = min(startPatch + patchesPerImage, numPatches)

            if startPatch < endPatch {
                let patches = imgFeatures[0..., startPatch..<endPatch, 0...]
                result.append(patches)
            }

            prevEnd = imageTokenPos + 1
        }

        if prevEnd < seqLen {
            let textAfter = inputEmbeds[0..., prevEnd..<seqLen, 0...]
            result.append(textAfter)
        }

        if result.isEmpty {
            return inputEmbeds
        } else if result.count == 1 {
            return result[0]
        } else {
            return MLX.concatenated(result, axis: 1)
        }
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let gridThw = input.image?.frames

        let dtype = DType.float32
        let pixels = input.image?.pixels.asType(dtype)

        let inputEmbeddings = self.inputEmbeddings(
            inputIds: input.text.tokens, pixelValues: pixels, gridThw: gridThw)

        let result = languageModel(nil, cache: cache, inputEmbedding: inputEmbeddings)

        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    /// Get all hidden states from all layers for multi-layer fusion (last element has RMSNorm applied)
    public func getAllHiddenStates(tokens: MLXArray) -> [MLXArray] {
        languageModel.model.callWithHiddenStates(tokens, cache: nil, inputEmbedding: nil)
    }

    /// Warmup the language model to eliminate cold start latency
    /// This triggers MLX graph compilation and caching for faster first inference
    public func warmup() {
        let dummyTokens = MLXArray([Int32](repeating: 0, count: 77)).expandedDimensions(axis: 0)

        // Run 3 iterations to trigger MLX graph compilation and caching
        for _ in 0..<3 {
            let hiddenStates = languageModel.model(dummyTokens, cache: nil, inputEmbedding: nil)
            MLX.eval(hiddenStates)
        }
    }

    /// Time taken for the most recent vision encoding operation
    public func getVisionEncoderTime() -> TimeInterval {
        visionModel.getLastEncodingTime()
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Set the vision encoder URL from the model directory if available.
        // Do NOT load the CoreML model here — vision_encoder.mlmodelc is 2.2GB
        // and loading it synchronously during weight sanitization blocks iPhone
        // startup. Load it lazily on first image inference instead.
        if let customDir = FastVLM.customModelDirectory {
            let visionURL = customDir.deletingLastPathComponent()
                .appendingPathComponent("vision_encoder.mlmodelc")
            visionModel.model.setModelURL(visionURL)
        }
        return weights
    }
}

// MARK: - Configuration

/// Configuration for ``FastVLM``
public struct FastVLMConfiguration: Codable, Sendable {

    public struct VisionConfiguration: Codable, Sendable {
        public let hiddenSize: Int

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "mm_hidden_size"
        }
    }

    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        private let _rmsNormEps: Float?
        public var rmsNormEps: Float { _rmsNormEps ?? 1e-6 }
        public let vocabularySize: Int
        public let kvHeads: Int
        private let _maxPositionEmbeddings: Int?
        public var maxpPositionEmbeddings: Int { _maxPositionEmbeddings ?? 32768 }
        private let _ropeTheta: Float?
        public var ropeTheta: Float { _ropeTheta ?? 1_000_000 }
        private let _ropeTraditional: Bool?
        public var ropeTraditional: Bool { _ropeTraditional ?? false }
        public let _ropeScaling: [String: StringOrNumber]?
        public var ropeScaling: [String: StringOrNumber]? {
            _ropeScaling ?? ["mrope_section": .ints([2, 1, 1])]
        }
        private let _tieWordEmbeddings: Bool?
        public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? true }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case _rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case kvHeads = "num_key_value_heads"
            case _maxPositionEmbeddings = "max_position_embeddings"
            case _ropeTheta = "rope_theta"
            case _ropeTraditional = "rope_traditional"
            case _ropeScaling = "rope_scaling"
            case _tieWordEmbeddings = "tie_word_embeddings"
        }
    }

    public struct BaseConfiguration: Codable, Sendable {
        public let modelType: String
        public let vocabularySize: Int
        public let imageTokenId: Int
        public let hiddenSize: Int

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case imageTokenId = "image_token_index"
            case hiddenSize = "hidden_size"
        }
    }

    public let visionConfiguration: VisionConfiguration
    public let textConfiguration: TextConfiguration
    public let baseConfiguration: BaseConfiguration

    public init(from decoder: any Swift.Decoder) throws {
        // these are overlaid in the top level
        self.visionConfiguration = try VisionConfiguration(from: decoder)
        self.textConfiguration = try TextConfiguration(from: decoder)
        self.baseConfiguration = try BaseConfiguration(from: decoder)
    }
}

/// Configuration for ``FastVLMProcessor``
public struct FastVLMPreProcessorConfiguration: Codable, Sendable {

    public struct CropSize: Codable, Sendable {
        let width: Int
        let height: Int

        var size: CGSize { .init(width: CGFloat(width), height: CGFloat(height)) }
    }

    public struct Size: Codable, Sendable {
        let shortestEdge: Int

        enum CodingKeys: String, CodingKey {
            case shortestEdge = "shortest_edge"
        }
    }

    public var imageMean: [CGFloat]
    public var imageStd: [CGFloat]
    public var size: Size
    public var cropSize: CropSize

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case size
        case cropSize = "crop_size"
    }
}

public struct FastVLMProcessorConfiguration: Codable, Sendable {

    public enum Strategy: Codable, Sendable {
        case `default`
    }

    public var imageToken = "<image>"
    public var numAdditionalImageTokens = 0
    public var patchSize = 64
    public var visionFeatureSelectStrategy: Strategy?

}
