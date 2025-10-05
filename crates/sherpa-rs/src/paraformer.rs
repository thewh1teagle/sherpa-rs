use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::{mem, ptr::null};

#[derive(Debug)]
pub struct ParaformerRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

pub type ParaformerRecognizerResult = super::OfflineRecognizerResult;

#[derive(Debug, Clone)]
pub struct ParaformerConfig {
    pub model: String,
    pub tokens: String,
    pub provider: Option<String>,
    pub num_threads: Option<i32>,
    pub debug: bool,
}

impl Default for ParaformerConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            tokens: String::new(),
            debug: false,
            provider: None,
            num_threads: Some(1),
        }
    }
}

impl ParaformerRecognizer {
    pub fn new(config: ParaformerConfig) -> Result<Self> {
        let debug = config.debug.into();
        let provider = config.provider.unwrap_or(get_default_provider());

        // Prepare C strings
        let provider_ptr = cstring_from_str(&provider);
        let model_ptr = cstring_from_str(&config.model);
        let tokens_ptr = cstring_from_str(&config.tokens);

        // 创建 decoding_method 的 CString 对象并绑定到变量
        let decoding_method_ptr = cstring_from_str("greedy_search");

        // Paraformer model config
        let paraformer_config = sherpa_rs_sys::SherpaOnnxOfflineParaformerModelConfig {
            model: model_ptr.as_ptr(),
        };

        // Offline model config
        let model_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
                debug,
                num_threads: config.num_threads.unwrap_or(1),
                provider: provider_ptr.as_ptr(),
                tokens: tokens_ptr.as_ptr(),
                paraformer: paraformer_config,

                // Null other model types
                bpe_vocab: mem::zeroed::<_>(),
                model_type: mem::zeroed::<_>(),
                modeling_unit: mem::zeroed::<_>(),
                nemo_ctc: mem::zeroed::<_>(),
                tdnn: mem::zeroed::<_>(),
                telespeech_ctc: null(),
                fire_red_asr: mem::zeroed::<_>(),
                transducer: mem::zeroed::<_>(),
                whisper: mem::zeroed::<_>(),
                sense_voice: mem::zeroed::<_>(),
                moonshine: mem::zeroed::<_>(),
                dolphin: mem::zeroed::<_>(),
                zipformer_ctc: mem::zeroed::<_>(),
                canary: mem::zeroed::<_>(),
            }
        };

        // Recognizer config
        let recognizer_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
                decoding_method: decoding_method_ptr.as_ptr(),
                feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                    sample_rate: 16000,
                    feature_dim: 80,
                },
                model_config,
                hotwords_file: null(),
                hotwords_score: 0.0,
                lm_config: mem::zeroed::<_>(),
                max_active_paths: 0,
                rule_fars: null(),
                rule_fsts: null(),
                blank_penalty: 0.0,
                hr: mem::zeroed::<_>(),
            }
        };

        let recognizer =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&recognizer_config) };
        if recognizer.is_null() {
            bail!("Failed to create Paraformer recognizer");
        }

        Ok(Self { recognizer })
    }

    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> ParaformerRecognizerResult {
        unsafe {
            let stream = sherpa_rs_sys::SherpaOnnxCreateOfflineStream(self.recognizer);
            sherpa_rs_sys::SherpaOnnxAcceptWaveformOffline(
                stream,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len() as i32,
            );
            sherpa_rs_sys::SherpaOnnxDecodeOfflineStream(self.recognizer, stream);
            let result_ptr = sherpa_rs_sys::SherpaOnnxGetOfflineStreamResult(stream);
            let raw_result = result_ptr.read();
            let result = ParaformerRecognizerResult::new(&raw_result);

            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);

            result
        }
    }
}

unsafe impl Send for ParaformerRecognizer {}
unsafe impl Sync for ParaformerRecognizer {}

impl Drop for ParaformerRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}
