use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::mem;

#[derive(Debug)]
pub struct DolphinRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

pub type DolphinRecognizerResult = super::OfflineRecognizerResult;

#[derive(Debug, Clone)]
pub struct DolphinConfig {
    pub model: String,
    pub tokens: String,
    pub decoding_method: String,

    pub provider: Option<String>,
    pub num_threads: Option<i32>,
    pub debug: bool,
}

impl Default for DolphinConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            tokens: String::new(),
            decoding_method: String::from("greedy_search"),
            debug: false,
            provider: None,
            num_threads: Some(1),
        }
    }
}

impl DolphinRecognizer {
    pub fn new(config: DolphinConfig) -> Result<Self> {
        let debug = config.debug.into();
        let provider = config.provider.unwrap_or(get_default_provider());

        let provider_ptr = cstring_from_str(&provider);
        let num_threads = config.num_threads.unwrap_or(2);
        let model_ptr = cstring_from_str(&config.model);
        let tokens_ptr = cstring_from_str(&config.tokens);
        let decoding_method_ptr = cstring_from_str(&config.decoding_method);

        let model_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
                debug,
                num_threads,
                provider: provider_ptr.as_ptr(),
                dolphin: sherpa_rs_sys::SherpaOnnxOfflineDolphinModelConfig {
                    model: model_ptr.as_ptr(),
                },
                tokens: tokens_ptr.as_ptr(),

                // Zeros
                nemo_ctc: mem::zeroed::<_>(),
                paraformer: mem::zeroed::<_>(),
                tdnn: mem::zeroed::<_>(),
                telespeech_ctc: mem::zeroed::<_>(),
                fire_red_asr: mem::zeroed::<_>(),
                transducer: mem::zeroed::<_>(),
                whisper: mem::zeroed::<_>(),
                sense_voice: mem::zeroed::<_>(),
                moonshine: mem::zeroed::<_>(),
                bpe_vocab: mem::zeroed::<_>(),
                model_type: mem::zeroed::<_>(),
                modeling_unit: mem::zeroed::<_>(),
                zipformer_ctc: mem::zeroed::<_>(),
                canary: mem::zeroed::<_>(),
            }
        };

        let config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
                decoding_method: decoding_method_ptr.as_ptr(),
                model_config,
                feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                    sample_rate: 16000,
                    feature_dim: 80,
                },
                hotwords_file: mem::zeroed::<_>(),
                hotwords_score: mem::zeroed::<_>(),
                lm_config: mem::zeroed::<_>(),
                max_active_paths: mem::zeroed::<_>(),
                rule_fars: mem::zeroed::<_>(),
                rule_fsts: mem::zeroed::<_>(),
                blank_penalty: mem::zeroed::<_>(),
                hr: mem::zeroed::<_>(),
            }
        };

        let recognizer = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&config) };

        if recognizer.is_null() {
            bail!("Failed to create recognizer");
        }

        Ok(Self { recognizer })
    }

    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> DolphinRecognizerResult {
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
            let result = DolphinRecognizerResult::new(&raw_result);
            // Free
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
            result
        }
    }
}

unsafe impl Send for DolphinRecognizer {}
unsafe impl Sync for DolphinRecognizer {}

impl Drop for DolphinRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}
