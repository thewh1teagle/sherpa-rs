use crate::common_config::FeatureConfig;
use crate::stream::offline_stream::OfflineStream;
use crate::utils::RawCStr;
use sherpa_rs_sys::{
    SherpaOnnxCreateOfflineRecognizer, SherpaOnnxCreateOfflineStream,
    SherpaOnnxDecodeMultipleOfflineStreams, SherpaOnnxDecodeOfflineStream,
    SherpaOnnxDestroyOfflineRecognizer, SherpaOnnxFeatureConfig, SherpaOnnxOfflineLMConfig,
    SherpaOnnxOfflineModelConfig, SherpaOnnxOfflineNemoEncDecCtcModelConfig,
    SherpaOnnxOfflineParaformerModelConfig, SherpaOnnxOfflineRecognizer,
    SherpaOnnxOfflineRecognizerConfig, SherpaOnnxOfflineSenseVoiceModelConfig,
    SherpaOnnxOfflineStream, SherpaOnnxOfflineTdnnModelConfig,
    SherpaOnnxOfflineTransducerModelConfig, SherpaOnnxOfflineWhisperModelConfig,
};

/// Configuration for offline/non-streaming transducer.
///
/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html
/// to download pre-trained models
pub struct OfflineTransducerModelConfig {
    pub encoder: String, // Path to the encoder model, i.e., encoder.onnx or encoder.int8.onnx
    pub decoder: String, // Path to the decoder model
    pub joiner: String,  // Path to the joiner model
}

/// Configuration for offline/non-streaming paraformer.
///
/// please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/index.html
/// to download pre-trained models
pub struct OfflineParaformerModelConfig {
    pub model: String, // Path to the model, e.g., model.onnx or model.int8.onnx
}

/// Configuration for offline/non-streaming NeMo CTC models.
///
/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/index.html
/// to download pre-trained models
pub struct OfflineNemoEncDecCtcModelConfig {
    pub model: String, // Path to the model, e.g., model.onnx or model.int8.onnx
}

pub struct OfflineWhisperModelConfig {
    pub encoder: String,
    pub decoder: String,
    pub language: String,
    pub task: String,
    pub tail_paddings: i32,
}

pub struct OfflineTdnnModelConfig {
    pub model: String,
}

pub struct OfflineSenseVoiceModelConfig {
    pub model: String,
    pub language: String,
    pub use_inverse_text_normalization: i32,
}

/// Configuration for offline LM.
pub struct OfflineLMConfig {
    pub model: String, // Path to the model
    pub scale: f32,    // scale for LM score
}

pub struct OfflineModelConfig {
    pub transducer: OfflineTransducerModelConfig,
    pub paraformer: OfflineParaformerModelConfig,
    pub nemo_ctc: OfflineNemoEncDecCtcModelConfig,
    pub whisper: OfflineWhisperModelConfig,
    pub tdnn: OfflineTdnnModelConfig,
    pub sense_voice: OfflineSenseVoiceModelConfig,
    pub tokens: String, // Path to tokens.txt

    // Number of threads to use for neural network computation
    pub num_threads: i32,

    // 1 to print model meta information while loading
    pub debug: i32,

    // Optional. Valid values: cpu, cuda, coreml
    pub provider: Option<String>,

    // Optional. Specify it for faster model initialization.
    pub model_type: Option<String>,

    pub modeling_unit: Option<String>, // Optional. cjkchar, bpe, cjkchar+bpe
    pub bpe_vocab: Option<String>,     // Optional.
    pub tele_speech_ctc: Option<String>, // Optional.
}

/// Configuration for the offline/non-streaming recognizer.
pub struct OfflineRecognizerConfig {
    pub feat_config: FeatureConfig,
    pub model_config: OfflineModelConfig,
    pub lm_config: OfflineLMConfig,

    // Valid decoding method: greedy_search, modified_beam_search
    pub decoding_method: String,

    // Used only when DecodingMethod is modified_beam_search.
    pub max_active_paths: i32,
    pub hotwords_file: String,
    pub hotwords_score: f32,
    pub blank_penalty: f32,
    pub rule_fsts: String,
    pub rule_fars: String,
}

/// It wraps a pointer from C
pub struct OfflineRecognizer {
    pub(crate) pointer: *const SherpaOnnxOfflineRecognizer,
}

/// It contains recognition result of an offline stream.
pub struct OfflineRecognizerResult {
    pub text: String,
    pub tokens: Vec<String>,
    pub timestamps: Vec<f32>,
    pub lang: String,
    pub emotion: String,
    pub event: String,
}

impl Drop for OfflineRecognizer {
    fn drop(&mut self) {
        self.delete()
    }
}

impl OfflineRecognizer {
    /// Frees the internal pointer of the recognition to avoid memory leak.
    fn delete(&mut self) {
        unsafe {
            SherpaOnnxDestroyOfflineRecognizer(self.pointer);
        }
    }

    /// The user is responsible to invoke [Self::drop] to free
    /// the returned recognizer to avoid memory leak.
    pub fn new(config: &OfflineRecognizerConfig) -> Self {
        let transducer_encoder = RawCStr::new(&config.model_config.transducer.encoder);
        let transducer_decoder = RawCStr::new(&config.model_config.transducer.decoder);
        let transducer_joiner = RawCStr::new(&config.model_config.transducer.joiner);
        let paraformer_model = RawCStr::new(&config.model_config.paraformer.model);
        let nemo_enc_dec_ctc_mode = RawCStr::new(&config.model_config.nemo_ctc.model);
        let whisper_encoder = RawCStr::new(&config.model_config.whisper.encoder);
        let whisper_decoder = RawCStr::new(&config.model_config.whisper.decoder);
        let whisper_language = RawCStr::new(&config.model_config.whisper.language);
        let whisper_task = RawCStr::new(&config.model_config.whisper.task);
        let tdnn_model = RawCStr::new(&config.model_config.tdnn.model);
        let tokens = RawCStr::new(&config.model_config.tokens);
        let provider = config
            .model_config
            .provider
            .as_ref()
            .map_or_else(|| RawCStr::new(""), |provider| RawCStr::new(provider));
        let mode_type = config
            .model_config
            .model_type
            .as_ref()
            .map_or_else(|| RawCStr::new(""), |model_type| RawCStr::new(model_type));
        let modeling_unit = config.model_config.modeling_unit.as_ref().map_or_else(
            || RawCStr::new(""),
            |modeling_unit| RawCStr::new(modeling_unit),
        );
        let bpe_vocab = config
            .model_config
            .modeling_unit
            .as_ref()
            .map_or_else(|| RawCStr::new(""), |bpe_vocab| RawCStr::new(bpe_vocab));
        let tele_speech_ctc = config.model_config.tele_speech_ctc.as_ref().map_or_else(
            || RawCStr::new(""),
            |tele_speech_ctc| RawCStr::new(tele_speech_ctc),
        );
        let sense_voice_model = RawCStr::new(&config.model_config.sense_voice.model);
        let sense_voice_language = RawCStr::new(&config.model_config.sense_voice.language);
        let lm_model = RawCStr::new(&config.lm_config.model);
        let decoding_method = RawCStr::new(&config.decoding_method);
        let hotwords_file = RawCStr::new(&config.hotwords_file);
        let rule_fsts = RawCStr::new(&config.rule_fsts);
        let rule_fars = RawCStr::new(&config.rule_fars);

        let c_config = SherpaOnnxOfflineRecognizerConfig {
            feat_config: SherpaOnnxFeatureConfig {
                sample_rate: config.feat_config.sample_rate,
                feature_dim: config.feat_config.feature_dim,
            },
            model_config: SherpaOnnxOfflineModelConfig {
                transducer: SherpaOnnxOfflineTransducerModelConfig {
                    encoder: transducer_encoder.as_ptr(),
                    decoder: transducer_decoder.as_ptr(),
                    joiner: transducer_joiner.as_ptr(),
                },
                paraformer: SherpaOnnxOfflineParaformerModelConfig {
                    model: paraformer_model.as_ptr(),
                },
                nemo_ctc: SherpaOnnxOfflineNemoEncDecCtcModelConfig {
                    model: nemo_enc_dec_ctc_mode.as_ptr(),
                },
                whisper: SherpaOnnxOfflineWhisperModelConfig {
                    encoder: whisper_encoder.as_ptr(),
                    decoder: whisper_decoder.as_ptr(),
                    language: whisper_language.as_ptr(),
                    task: whisper_task.as_ptr(),
                    tail_paddings: config.model_config.whisper.tail_paddings,
                },
                tdnn: SherpaOnnxOfflineTdnnModelConfig {
                    model: tdnn_model.as_ptr(),
                },
                tokens: tokens.as_ptr(),
                num_threads: config.model_config.num_threads,
                debug: config.model_config.debug,
                provider: provider.as_ptr(),
                model_type: mode_type.as_ptr(),
                modeling_unit: modeling_unit.as_ptr(),
                bpe_vocab: bpe_vocab.as_ptr(),
                telespeech_ctc: tele_speech_ctc.as_ptr(),
                sense_voice: SherpaOnnxOfflineSenseVoiceModelConfig {
                    model: sense_voice_model.as_ptr(),
                    language: sense_voice_language.as_ptr(),
                    use_itn: config
                        .model_config
                        .sense_voice
                        .use_inverse_text_normalization,
                },
            },
            lm_config: SherpaOnnxOfflineLMConfig {
                model: lm_model.as_ptr(),
                scale: config.lm_config.scale,
            },
            decoding_method: decoding_method.as_ptr(),
            max_active_paths: config.max_active_paths,
            hotwords_file: hotwords_file.as_ptr(),
            hotwords_score: config.hotwords_score,
            rule_fsts: rule_fsts.as_ptr(),
            rule_fars: rule_fars.as_ptr(),
            blank_penalty: config.blank_penalty,
        };

        let recognizer = unsafe { SherpaOnnxCreateOfflineRecognizer(&c_config) };

        OfflineRecognizer {
            pointer: recognizer,
        }
    }

    /// The user is responsible to invoke [OfflineStream::drop] to free
    /// the returned stream to avoid memory leak.
    pub fn new_stream(&self) -> OfflineStream {
        let stream = unsafe { SherpaOnnxCreateOfflineStream(self.pointer) };
        OfflineStream { pointer: stream }
    }

    /// Decode the offline stream.
    pub fn decode(&self, stream: &OfflineStream) {
        unsafe {
            SherpaOnnxDecodeOfflineStream(self.pointer, stream.pointer);
        }
    }

    /// Decode multiple streams in parallel, i.e., in batch.
    pub fn decode_streams(&self, streams: &mut [OfflineStream]) {
        let mut ss: Vec<*const SherpaOnnxOfflineStream> =
            streams.iter_mut().map(|s| s.pointer).collect();
        unsafe {
            SherpaOnnxDecodeMultipleOfflineStreams(self.pointer, ss.as_mut_ptr(), ss.len() as i32);
        }
    }
}
