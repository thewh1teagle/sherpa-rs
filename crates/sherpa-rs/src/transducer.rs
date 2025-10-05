use crate::utils::cstr_to_string;
use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::mem;

pub struct TransducerRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

#[derive(Debug, Clone)]
pub struct TransducerConfig {
    pub decoder: String,
    pub encoder: String,
    pub joiner: String,
    pub tokens: String,
    pub num_threads: i32,
    pub sample_rate: i32,
    pub feature_dim: i32,
    pub decoding_method: String,
    pub hotwords_file: String,
    pub hotwords_score: f32,
    pub modeling_unit: String,
    pub bpe_vocab: String,
    pub blank_penalty: f32,
    pub model_type: String,
    pub debug: bool,
    pub provider: Option<String>,
}

impl Default for TransducerConfig {
    fn default() -> Self {
        TransducerConfig {
            decoder: String::new(),
            encoder: String::new(),
            joiner: String::new(),
            tokens: String::new(),
            model_type: String::from("transducer"),
            num_threads: 1,
            sample_rate: 0,
            feature_dim: 0,
            decoding_method: String::new(),
            hotwords_file: String::new(),
            hotwords_score: 0.0,
            modeling_unit: String::new(),
            bpe_vocab: String::new(),
            blank_penalty: 0.0,
            debug: false,
            provider: None,
        }
    }
}

impl TransducerRecognizer {
    pub fn new(config: TransducerConfig) -> Result<Self> {
        let recognizer = unsafe {
            let debug = config.debug.into();
            let provider = config.provider.unwrap_or(get_default_provider());
            let provider_ptr = cstring_from_str(&provider);

            let encoder = cstring_from_str(&config.encoder);
            let decoder = cstring_from_str(&config.decoder);
            let joiner = cstring_from_str(&config.joiner);
            let model_type = cstring_from_str(&config.model_type);
            let modeling_unit = cstring_from_str(&config.modeling_unit);
            let bpe_vocab = cstring_from_str(&config.bpe_vocab);
            let hotwords_file = cstring_from_str(&config.hotwords_file);
            let tokens = cstring_from_str(&config.tokens);
            let decoding_method = cstring_from_str(&config.decoding_method);

            let offline_model_config = sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
                transducer: sherpa_rs_sys::SherpaOnnxOfflineTransducerModelConfig {
                    encoder: encoder.as_ptr(),
                    decoder: decoder.as_ptr(),
                    joiner: joiner.as_ptr(),
                },
                tokens: tokens.as_ptr(),
                num_threads: config.num_threads,
                debug,
                provider: provider_ptr.as_ptr(),
                model_type: model_type.as_ptr(),
                modeling_unit: modeling_unit.as_ptr(),
                bpe_vocab: bpe_vocab.as_ptr(),

                // NULLs
                telespeech_ctc: mem::zeroed::<_>(),
                paraformer: mem::zeroed::<_>(),
                tdnn: mem::zeroed::<_>(),
                nemo_ctc: mem::zeroed::<_>(),
                whisper: mem::zeroed::<_>(),
                sense_voice: mem::zeroed::<_>(),
                moonshine: mem::zeroed::<_>(),
                fire_red_asr: mem::zeroed::<_>(),
                dolphin: mem::zeroed::<_>(),
                zipformer_ctc: mem::zeroed::<_>(),
                canary: mem::zeroed::<_>(),
            };

            let recognizer_config = sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
                model_config: offline_model_config,
                feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                    sample_rate: config.sample_rate,
                    feature_dim: config.feature_dim,
                },
                hotwords_file: hotwords_file.as_ptr(),
                blank_penalty: config.blank_penalty,
                decoding_method: decoding_method.as_ptr(),
                hotwords_score: config.hotwords_score,

                // NULLs
                lm_config: mem::zeroed::<_>(),
                rule_fsts: mem::zeroed::<_>(),
                rule_fars: mem::zeroed::<_>(),
                max_active_paths: mem::zeroed::<_>(),
                hr: mem::zeroed::<_>(),
            };

            let recognizer = sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&recognizer_config);
            if recognizer.is_null() {
                bail!("SherpaOnnxCreateOfflineRecognizer failed");
            }
            recognizer
        };

        Ok(Self { recognizer })
    }

    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> String {
        unsafe {
            let stream = sherpa_rs_sys::SherpaOnnxCreateOfflineStream(self.recognizer);
            sherpa_rs_sys::SherpaOnnxAcceptWaveformOffline(
                stream,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len().try_into().unwrap(),
            );
            sherpa_rs_sys::SherpaOnnxDecodeOfflineStream(self.recognizer, stream);
            let result_ptr = sherpa_rs_sys::SherpaOnnxGetOfflineStreamResult(stream);
            let raw_result = result_ptr.read();
            let text = cstr_to_string(raw_result.text as _);

            // Free
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
            text
        }
    }
}

unsafe impl Send for TransducerRecognizer {}
unsafe impl Sync for TransducerRecognizer {}

impl Drop for TransducerRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}
