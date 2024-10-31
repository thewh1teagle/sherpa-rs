use crate::common_config::FeatureConfig;
use crate::stream::online_stream::{InitialState, OnlineStream, State};
use crate::utils;
use crate::utils::RawCStr;
use sherpa_rs_sys::{
    SherpaOnnxCreateOnlineRecognizer, SherpaOnnxCreateOnlineStream,
    SherpaOnnxDecodeMultipleOnlineStreams, SherpaOnnxDecodeOnlineStream,
    SherpaOnnxDestroyOnlineRecognizer, SherpaOnnxDestroyOnlineRecognizerResult,
    SherpaOnnxFeatureConfig, SherpaOnnxGetOnlineStreamResult, SherpaOnnxIsOnlineStreamReady,
    SherpaOnnxOnlineCtcFstDecoderConfig, SherpaOnnxOnlineModelConfig,
    SherpaOnnxOnlineParaformerModelConfig, SherpaOnnxOnlineRecognizer,
    SherpaOnnxOnlineRecognizerConfig, SherpaOnnxOnlineStream, SherpaOnnxOnlineStreamIsEndpoint,
    SherpaOnnxOnlineStreamReset, SherpaOnnxOnlineTransducerModelConfig,
    SherpaOnnxOnlineZipformer2CtcModelConfig,
};
use std::marker::PhantomData;

/// Configuration for online/streaming transducer models
///
/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/index.html
/// to download pre-trained models
pub struct OnlineTransducerModelConfig {
    pub encoder: String, // Path to the encoder model, e.g., encoder.onnx or encoder.int8.onnx
    pub decoder: String, // Path to the decoder model.
    pub joiner: String,  // Path to the joiner model.
}

/// Configuration for online/streaming paraformer models
///
/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/index.html
/// to download pre-trained models
pub struct OnlineParaformerModelConfig {
    pub encoder: String, // Path to the encoder model, e.g., encoder.onnx or encoder.int8.onnx
    pub decoder: String, // Path to the decoder model.
}

/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-ctc/index.html
/// to download pre-trained models
pub struct OnlineZipformer2CtcModelConfig {
    pub model: String, // Path to the onnx model
}

/// Configuration for online/streaming models
///
/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/index.html
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/index.html
/// to download pre-trained models
pub struct OnlineModelConfig {
    pub transducer: OnlineTransducerModelConfig,
    pub paraformer: OnlineParaformerModelConfig,
    pub zipformer2_ctc: OnlineZipformer2CtcModelConfig,
    pub tokens: String,                // Path to tokens.txt
    pub num_threads: i32,              // Number of threads to use for neural network computation
    pub provider: Option<String>,      // Optional. Valid values are: cpu, cuda, coreml
    pub debug: i32,                    // 1 to show model meta information while loading it.
    pub model_type: Option<String>, // Optional. You can specify it for faster model initialization
    pub modeling_unit: Option<String>, // Optional. cjkchar, bpe, cjkchar+bpe
    pub bpe_vocab: Option<String>,  // Optional.
    pub tokens_buf: Option<String>, // Optional.
    pub tokens_buf_size: Option<i32>, // Optional.
}

#[derive(Default)]
pub struct OnlineCtcFstDecoderConfig {
    pub graph: String,
    pub max_active: i32,
}

/// Configuration for the online/streaming recognizer.
pub struct OnlineRecognizerConfig {
    pub feat_config: FeatureConfig,
    pub model_config: OnlineModelConfig,

    /// Valid decoding methods: greedy_search, modified_beam_search
    pub decoding_method: String,

    /// Used only when DecodingMethod is modified_beam_search. It specifies
    /// the maximum number of paths to keep during the search
    pub max_active_paths: i32,

    pub enable_endpoint: i32, // 1 to enable endpoint detection.

    /// Please see
    /// https://k2-fsa.github.io/sherpa/ncnn/endpoint.html
    /// for the meaning of Rule1MinTrailingSilence, Rule2MinTrailingSilence
    /// and Rule3MinUtteranceLength.
    pub rule1_min_trailing_silence: f32,
    pub rule2_min_trailing_silence: f32,
    pub rule3_min_utterance_length: f32,
    pub hotwords_file: String,
    pub hotwords_score: f32,
    pub blank_penalty: f32,
    pub ctc_fst_decoder_config: OnlineCtcFstDecoderConfig,
    pub rule_fsts: String,
    pub rule_fars: String,
    pub hotwords_buf: String,
    pub hotwords_buf_size: i32,
}

/// It contains the recognition result for a online stream.
pub struct OnlineRecognizerResult {
    pub text: String,
}

/// The online recognizer class. It wraps a pointer from C.
pub struct OnlineRecognizer {
    pointer: *const SherpaOnnxOnlineRecognizer,
}

impl Drop for OnlineRecognizer {
    fn drop(&mut self) {
        self.delete();
    }
}

impl OnlineRecognizer {
    /// The user is responsible to invoke [`Self::drop`] to free
    /// the returned recognizer to avoid memory leak
    pub fn new(config: &OnlineRecognizerConfig) -> Self {
        let transducer_encoder = RawCStr::new(&config.model_config.transducer.encoder);
        let transducer_decoder = RawCStr::new(&config.model_config.transducer.decoder);
        let transducer_joiner = RawCStr::new(&config.model_config.transducer.joiner);
        let paraformer_encoder = RawCStr::new(&config.model_config.paraformer.encoder);
        let paraformer_decoder = RawCStr::new(&config.model_config.paraformer.decoder);
        let zipformer2_ctc_model = RawCStr::new(&config.model_config.zipformer2_ctc.model);
        let tokens = RawCStr::new(&config.model_config.tokens);
        let tokens_buf = config
            .model_config
            .tokens_buf
            .as_ref()
            .map_or_else(|| RawCStr::new(""), |tokens_buf| RawCStr::new(tokens_buf));
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
            .bpe_vocab
            .as_ref()
            .map_or_else(|| RawCStr::new(""), |bpe_vocab| RawCStr::new(bpe_vocab));
        let decoding_method = RawCStr::new(&config.decoding_method);
        let hotwords_file = RawCStr::new(&config.hotwords_file);
        let hotwords_buf = RawCStr::new(&config.hotwords_buf);
        let rule_fsts = RawCStr::new(&config.rule_fsts);
        let rule_fars = RawCStr::new(&config.rule_fars);
        let graph = RawCStr::new(&config.ctc_fst_decoder_config.graph);

        let c_config = SherpaOnnxOnlineRecognizerConfig {
            feat_config: SherpaOnnxFeatureConfig {
                sample_rate: config.feat_config.sample_rate,
                feature_dim: config.feat_config.feature_dim,
            },
            model_config: SherpaOnnxOnlineModelConfig {
                transducer: SherpaOnnxOnlineTransducerModelConfig {
                    encoder: transducer_encoder.as_ptr(),
                    decoder: transducer_decoder.as_ptr(),
                    joiner: transducer_joiner.as_ptr(),
                },
                paraformer: SherpaOnnxOnlineParaformerModelConfig {
                    encoder: paraformer_encoder.as_ptr(),
                    decoder: paraformer_decoder.as_ptr(),
                },
                zipformer2_ctc: SherpaOnnxOnlineZipformer2CtcModelConfig {
                    model: zipformer2_ctc_model.as_ptr(),
                },
                tokens: tokens.as_ptr(),
                tokens_buf: tokens_buf.as_ptr(),
                tokens_buf_size: config.model_config.tokens_buf_size.unwrap_or(0),
                num_threads: config.model_config.num_threads,
                provider: provider.as_ptr(),
                debug: config.model_config.debug,
                model_type: mode_type.as_ptr(),
                modeling_unit: modeling_unit.as_ptr(),
                bpe_vocab: bpe_vocab.as_ptr(),
            },
            decoding_method: decoding_method.as_ptr(),
            max_active_paths: config.max_active_paths,
            enable_endpoint: config.enable_endpoint,
            rule1_min_trailing_silence: config.rule1_min_trailing_silence,
            rule2_min_trailing_silence: config.rule2_min_trailing_silence,
            rule3_min_utterance_length: config.rule3_min_utterance_length,
            hotwords_file: hotwords_file.as_ptr(),
            hotwords_buf: hotwords_buf.as_ptr(),
            hotwords_buf_size: config.hotwords_buf_size,
            hotwords_score: config.hotwords_score,
            blank_penalty: config.blank_penalty,
            rule_fsts: rule_fsts.as_ptr(),
            rule_fars: rule_fars.as_ptr(),
            ctc_fst_decoder_config: SherpaOnnxOnlineCtcFstDecoderConfig {
                graph: graph.as_ptr(),
                max_active: config.ctc_fst_decoder_config.max_active,
            },
        };

        let recognizer = unsafe { SherpaOnnxCreateOnlineRecognizer(&c_config) };

        OnlineRecognizer {
            pointer: recognizer,
        }
    }

    /// Free the internal pointer inside the recognizer to avoid memory leak.
    fn delete(&mut self) {
        unsafe {
            SherpaOnnxDestroyOnlineRecognizer(self.pointer);
        }
    }

    /// The user is responsible to invoke [`OnlineStream::drop`] to free
    /// the returned stream to avoid memory leak
    pub fn new_stream(&self) -> OnlineStream<InitialState> {
        let stream = unsafe { SherpaOnnxCreateOnlineStream(self.pointer) };
        OnlineStream {
            pointer: stream,
            _marker: PhantomData,
        }
    }

    /// Check whether the stream has enough feature frames for decoding.
    /// Return true if this stream is ready for decoding. Return false otherwise.
    ///
    /// You will usually use it like below:
    /// ```text
    /// while recognizer.is_ready(&stream) {
    ///     recognizer.decode(&stream);
    /// }
    /// ```
    pub fn is_ready(&self, stream: &OnlineStream<impl State>) -> bool {
        unsafe { SherpaOnnxIsOnlineStreamReady(self.pointer, stream.pointer) == 1 }
    }

    /// Return true if an endpoint is detected.
    ///
    /// You usually use it like below:
    /// ```text
    /// if recognizer.is_endpoint(&stream) {
    ///     // do your own stuff after detecting an endpoint
    ///
    ///     recognizer.reset(&stream)
    /// }
    /// ```
    pub fn is_endpoint(&self, stream: &OnlineStream<impl State>) -> bool {
        unsafe { SherpaOnnxOnlineStreamIsEndpoint(self.pointer, stream.pointer) == 1 }
    }

    /// After calling this function, the internal neural network model states
    /// are reset and [`Self::is_endpoint(s)`] would return false. [`Self::get_result(s)`] would also
    /// return an empty string.
    pub fn reset(&self, stream: &OnlineStream<impl State>) {
        unsafe {
            SherpaOnnxOnlineStreamReset(self.pointer, stream.pointer);
        }
    }

    /// Decode the stream. Before calling this function, you have to ensure
    /// that recognizer.IsReady(s) returns true. Otherwise, you will be SAD.
    ///
    /// You usually use it like below:
    ///
    /// ```text
    /// while recognizer.is_ready(&stream) {
    ///     recognizer.decode(&stream);
    /// }
    /// ```
    pub fn decode(&self, stream: &OnlineStream<impl State>) {
        unsafe {
            SherpaOnnxDecodeOnlineStream(self.pointer, stream.pointer);
        }
    }

    /// Decode multiple streams in parallel, i.e., in batch.
    /// You have to ensure that each stream is ready for decoding. Otherwise,
    /// you will be SAD.
    pub fn decode_streams(&self, streams: &[OnlineStream<impl State>]) {
        let mut c_streams: Vec<*const SherpaOnnxOnlineStream> =
            streams.iter().map(|s| s.pointer).collect();
        unsafe {
            SherpaOnnxDecodeMultipleOnlineStreams(
                self.pointer,
                c_streams.as_mut_ptr(),
                c_streams.len() as i32,
            );
        }
    }

    /// Get the current result of stream since the last invoke of [`Self::reset()`]
    pub fn get_result(&self, stream: &OnlineStream<impl State>) -> OnlineRecognizerResult {
        let result = unsafe { SherpaOnnxGetOnlineStreamResult(self.pointer, stream.pointer) };
        let text = utils::cstr_to_string((unsafe { *result }).text);
        unsafe {
            SherpaOnnxDestroyOnlineRecognizerResult(result);
        }
        OnlineRecognizerResult { text }
    }
}
