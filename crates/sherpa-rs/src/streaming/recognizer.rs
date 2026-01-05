use super::config::*;
use super::stream::OnlineStream;
use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::ffi::CString;
use std::mem;

/// Online (streaming) speech recognizer
pub struct OnlineRecognizer {
    pub(crate) recognizer: *const sherpa_rs_sys::SherpaOnnxOnlineRecognizer,
}

impl OnlineRecognizer {
    /// Create a new online recognizer with the given configuration
    pub fn new(config: OnlineRecognizerConfig) -> Result<Self> {
        let provider = config.provider.unwrap_or(get_default_provider());
        let provider_ptr = cstring_from_str(&provider);
        let tokens_ptr = cstring_from_str(&config.tokens);
        let decoding_method_ptr = cstring_from_str(&config.decoding_method);
        let hotwords_file_ptr =
            cstring_from_str(&config.hotwords_file.clone().unwrap_or_default());
        let rule_fsts_ptr = cstring_from_str(&config.rule_fsts.clone().unwrap_or_default());
        let rule_fars_ptr = cstring_from_str(&config.rule_fars.clone().unwrap_or_default());

        // Pre-create all CStrings to ensure they live long enough
        let (model_cstrings, model_config) = build_model_config(
            &config.model,
            &tokens_ptr,
            &provider_ptr,
            config.num_threads.unwrap_or(1),
            config.debug,
        );

        // Build CTC FST decoder config
        let graph_ptr = config
            .ctc_fst_decoder
            .as_ref()
            .map(|fst| cstring_from_str(&fst.graph));
        let ctc_fst_decoder_config = if let Some(ref fst) = config.ctc_fst_decoder {
            sherpa_rs_sys::SherpaOnnxOnlineCtcFstDecoderConfig {
                graph: graph_ptr.as_ref().unwrap().as_ptr(),
                max_active: fst.max_active,
            }
        } else {
            unsafe { mem::zeroed::<_>() }
        };

        let recognizer_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineRecognizerConfig {
                feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                    sample_rate: config.sample_rate,
                    feature_dim: config.feature_dim,
                },
                model_config,
                decoding_method: decoding_method_ptr.as_ptr(),
                max_active_paths: config.max_active_paths,
                enable_endpoint: config.endpoint.enable as i32,
                rule1_min_trailing_silence: config.endpoint.rule1_min_trailing_silence,
                rule2_min_trailing_silence: config.endpoint.rule2_min_trailing_silence,
                rule3_min_utterance_length: config.endpoint.rule3_min_utterance_length,
                hotwords_file: hotwords_file_ptr.as_ptr(),
                hotwords_score: config.hotwords_score,
                ctc_fst_decoder_config,
                rule_fsts: rule_fsts_ptr.as_ptr(),
                rule_fars: rule_fars_ptr.as_ptr(),
                blank_penalty: config.blank_penalty,
                hotwords_buf: mem::zeroed::<_>(),
                hotwords_buf_size: 0,
                hr: mem::zeroed::<_>(),
            }
        };

        let recognizer =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateOnlineRecognizer(&recognizer_config) };

        // Keep model_cstrings alive until after recognizer is created
        drop(model_cstrings);

        if recognizer.is_null() {
            bail!("Failed to create online recognizer");
        }

        Ok(Self { recognizer })
    }

    /// Create a new online stream for accepting audio samples
    pub fn create_stream(&self) -> Result<OnlineStream> {
        OnlineStream::new(self.recognizer)
    }

    /// Create a new online stream with custom hotwords
    pub fn create_stream_with_hotwords(&self, hotwords: &str) -> Result<OnlineStream> {
        OnlineStream::new_with_hotwords(self.recognizer, hotwords)
    }
}

/// Helper struct to hold CStrings that need to outlive the config
struct ModelCStrings {
    _encoder: Option<CString>,
    _decoder: Option<CString>,
    _joiner: Option<CString>,
    _model: Option<CString>,
}

fn build_model_config(
    model: &OnlineModelType,
    tokens_ptr: &CString,
    provider_ptr: &CString,
    num_threads: i32,
    debug: bool,
) -> (ModelCStrings, sherpa_rs_sys::SherpaOnnxOnlineModelConfig) {
    match model {
        OnlineModelType::Transducer(t) => {
            let encoder_ptr = cstring_from_str(&t.encoder);
            let decoder_ptr = cstring_from_str(&t.decoder);
            let joiner_ptr = cstring_from_str(&t.joiner);

            let config = unsafe {
                sherpa_rs_sys::SherpaOnnxOnlineModelConfig {
                    transducer: sherpa_rs_sys::SherpaOnnxOnlineTransducerModelConfig {
                        encoder: encoder_ptr.as_ptr(),
                        decoder: decoder_ptr.as_ptr(),
                        joiner: joiner_ptr.as_ptr(),
                    },
                    tokens: tokens_ptr.as_ptr(),
                    num_threads,
                    provider: provider_ptr.as_ptr(),
                    debug: debug.into(),
                    paraformer: mem::zeroed::<_>(),
                    zipformer2_ctc: mem::zeroed::<_>(),
                    nemo_ctc: mem::zeroed::<_>(),
                    t_one_ctc: mem::zeroed::<_>(),
                    model_type: mem::zeroed::<_>(),
                    modeling_unit: mem::zeroed::<_>(),
                    bpe_vocab: mem::zeroed::<_>(),
                    tokens_buf: mem::zeroed::<_>(),
                    tokens_buf_size: 0,
                }
            };

            (
                ModelCStrings {
                    _encoder: Some(encoder_ptr),
                    _decoder: Some(decoder_ptr),
                    _joiner: Some(joiner_ptr),
                    _model: None,
                },
                config,
            )
        }
        OnlineModelType::Paraformer(p) => {
            let encoder_ptr = cstring_from_str(&p.encoder);
            let decoder_ptr = cstring_from_str(&p.decoder);

            let config = unsafe {
                sherpa_rs_sys::SherpaOnnxOnlineModelConfig {
                    paraformer: sherpa_rs_sys::SherpaOnnxOnlineParaformerModelConfig {
                        encoder: encoder_ptr.as_ptr(),
                        decoder: decoder_ptr.as_ptr(),
                    },
                    tokens: tokens_ptr.as_ptr(),
                    num_threads,
                    provider: provider_ptr.as_ptr(),
                    debug: debug.into(),
                    transducer: mem::zeroed::<_>(),
                    zipformer2_ctc: mem::zeroed::<_>(),
                    nemo_ctc: mem::zeroed::<_>(),
                    t_one_ctc: mem::zeroed::<_>(),
                    model_type: mem::zeroed::<_>(),
                    modeling_unit: mem::zeroed::<_>(),
                    bpe_vocab: mem::zeroed::<_>(),
                    tokens_buf: mem::zeroed::<_>(),
                    tokens_buf_size: 0,
                }
            };

            (
                ModelCStrings {
                    _encoder: Some(encoder_ptr),
                    _decoder: Some(decoder_ptr),
                    _joiner: None,
                    _model: None,
                },
                config,
            )
        }
        OnlineModelType::Zipformer2Ctc(z) => {
            let model_ptr = cstring_from_str(&z.model);

            let config = unsafe {
                sherpa_rs_sys::SherpaOnnxOnlineModelConfig {
                    zipformer2_ctc: sherpa_rs_sys::SherpaOnnxOnlineZipformer2CtcModelConfig {
                        model: model_ptr.as_ptr(),
                    },
                    tokens: tokens_ptr.as_ptr(),
                    num_threads,
                    provider: provider_ptr.as_ptr(),
                    debug: debug.into(),
                    transducer: mem::zeroed::<_>(),
                    paraformer: mem::zeroed::<_>(),
                    nemo_ctc: mem::zeroed::<_>(),
                    t_one_ctc: mem::zeroed::<_>(),
                    model_type: mem::zeroed::<_>(),
                    modeling_unit: mem::zeroed::<_>(),
                    bpe_vocab: mem::zeroed::<_>(),
                    tokens_buf: mem::zeroed::<_>(),
                    tokens_buf_size: 0,
                }
            };

            (
                ModelCStrings {
                    _encoder: None,
                    _decoder: None,
                    _joiner: None,
                    _model: Some(model_ptr),
                },
                config,
            )
        }
        OnlineModelType::NemoCtc(n) => {
            let model_ptr = cstring_from_str(&n.model);

            let config = unsafe {
                sherpa_rs_sys::SherpaOnnxOnlineModelConfig {
                    nemo_ctc: sherpa_rs_sys::SherpaOnnxOnlineNemoCtcModelConfig {
                        model: model_ptr.as_ptr(),
                    },
                    tokens: tokens_ptr.as_ptr(),
                    num_threads,
                    provider: provider_ptr.as_ptr(),
                    debug: debug.into(),
                    transducer: mem::zeroed::<_>(),
                    paraformer: mem::zeroed::<_>(),
                    zipformer2_ctc: mem::zeroed::<_>(),
                    t_one_ctc: mem::zeroed::<_>(),
                    model_type: mem::zeroed::<_>(),
                    modeling_unit: mem::zeroed::<_>(),
                    bpe_vocab: mem::zeroed::<_>(),
                    tokens_buf: mem::zeroed::<_>(),
                    tokens_buf_size: 0,
                }
            };

            (
                ModelCStrings {
                    _encoder: None,
                    _decoder: None,
                    _joiner: None,
                    _model: Some(model_ptr),
                },
                config,
            )
        }
    }
}

unsafe impl Send for OnlineRecognizer {}
unsafe impl Sync for OnlineRecognizer {}

impl Drop for OnlineRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizer(self.recognizer);
        }
    }
}
