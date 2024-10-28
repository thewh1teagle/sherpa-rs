use crate::common_config::FeatureConfig;
use sherpa_rs_sys::SherpaOnnxOfflineRecognizer;

/// Configuration for offline/non-streaming transducer.
///
/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html
/// to download pre-trained models
struct OfflineTransducerModelConfig {
    encoder: String, // Path to the encoder model, i.e., encoder.onnx or encoder.int8.onnx
    decoder: String, // Path to the decoder model
    joiner: String,  // Path to the joiner model
}

/// Configuration for offline/non-streaming paraformer.
///
/// please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/index.html
/// to download pre-trained models
struct OfflineParaformerModelConfig {
    model: String, // Path to the model, e.g., model.onnx or model.int8.onnx
}

/// Configuration for offline/non-streaming NeMo CTC models.
///
/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/index.html
/// to download pre-trained models
struct OfflineNemoEncDecCtcModelConfig {
    model: String, // Path to the model, e.g., model.onnx or model.int8.onnx
}

struct OfflineWhisperModelConfig {
    encoder: String,
    decoder: String,
    language: String,
    task: String,
    tail_paddings: i32,
}

struct OfflineTdnnModelConfig {
    model: String,
}

struct OfflineSenseVoiceModelConfig {
    model: String,
    language: String,
    use_inverse_text_normalization: i32,
}

/// Configuration for offline LM.
struct OfflineLMConfig {
    model: String, // Path to the model
    scale: f32,    // scale for LM score
}

struct OfflineModelConfig {
    transducer: OfflineTransducerModelConfig,
    paraformer: OfflineParaformerModelConfig,
    nemo_ctc: OfflineNemoEncDecCtcModelConfig,
    whisper: OfflineWhisperModelConfig,
    tdnn: OfflineTdnnModelConfig,
    sense_voice: OfflineSenseVoiceModelConfig,
    tokens: String, // Path to tokens.txt

    // Number of threads to use for neural network computation
    num_threads: i32,

    // 1 to print model meta information while loading
    debug: i32,

    // Optional. Valid values: cpu, cuda, coreml
    provider: String,

    // Optional. Specify it for faster model initialization.
    model_type: String,

    modeling_unit: String,   // Optional. cjkchar, bpe, cjkchar+bpe
    bpe_vocab: String,       // Optional.
    tele_speech_ctc: String, // Optional.
}

/// Configuration for the offline/non-streaming recognizer.
struct OfflineRecognizerConfig {
    feat_config: FeatureConfig,
    model_config: OfflineModelConfig,
    lm_config: OfflineLMConfig,

    // Valid decoding method: greedy_search, modified_beam_search
    decoding_method: String,

    // Used only when DecodingMethod is modified_beam_search.
    max_active_paths: i32,
    hotwords_file: String,
    hotwords_score: f32,
    blank_penalty: f32,
    rule_fsts: String,
    rule_fars: String,
}

/// It wraps a pointer from C
struct OfflineRecognizer {
    pointer: *const SherpaOnnxOfflineRecognizer,
}

/// It contains recognition result of an offline stream.
struct OfflineRecognizerResult {
    text: String,
    tokens: Vec<String>,
    timestamps: Vec<f32>,
    lang: String,
    emotion: String,
    event: String,
}
