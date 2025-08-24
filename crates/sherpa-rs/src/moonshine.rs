use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::{mem, ptr::null};

#[derive(Debug)]
pub struct MoonshineRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

pub type MoonshineRecognizerResult = super::OfflineRecognizerResult;

#[derive(Debug, Clone)]
pub struct MoonshineConfig {
    pub preprocessor: String,

    pub encoder: String,
    pub uncached_decoder: String,
    pub cached_decoder: String,

    pub tokens: String,

    pub provider: Option<String>,
    pub num_threads: Option<i32>,
    pub debug: bool,
}

impl Default for MoonshineConfig {
    fn default() -> Self {
        Self {
            preprocessor: String::new(),
            encoder: String::new(),
            cached_decoder: String::new(),
            uncached_decoder: String::new(),
            tokens: String::new(),

            debug: false,
            provider: None,
            num_threads: Some(1),
        }
    }
}

impl MoonshineRecognizer {
    pub fn new(config: MoonshineConfig) -> Result<Self> {
        let debug = config.debug.into();
        let provider = config.provider.unwrap_or(get_default_provider());

        // Onnx
        let provider_ptr = cstring_from_str(&provider);
        let num_threads = config.num_threads.unwrap_or(2);

        // Moonshine
        let preprocessor_ptr = cstring_from_str(&config.preprocessor);
        let encoder_ptr = cstring_from_str(&config.encoder);
        let cached_decoder_ptr = cstring_from_str(&config.cached_decoder);
        let uncached_decoder_ptr = cstring_from_str(&config.uncached_decoder);
        let tokens_ptr = cstring_from_str(&config.tokens);

        let model_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
                debug,
                num_threads,
                moonshine: sherpa_rs_sys::SherpaOnnxOfflineMoonshineModelConfig {
                    preprocessor: preprocessor_ptr.as_ptr(),
                    encoder: encoder_ptr.as_ptr(),
                    uncached_decoder: uncached_decoder_ptr.as_ptr(),
                    cached_decoder: cached_decoder_ptr.as_ptr(),
                },
                tokens: tokens_ptr.as_ptr(),
                provider: provider_ptr.as_ptr(),

                model_type: mem::zeroed::<_>(),
                modeling_unit: mem::zeroed::<_>(),
                dolphin: mem::zeroed::<_>(),
                bpe_vocab: mem::zeroed::<_>(),
                nemo_ctc: mem::zeroed::<_>(),
                paraformer: mem::zeroed::<_>(),
                tdnn: mem::zeroed::<_>(),
                telespeech_ctc: mem::zeroed::<_>(),
                fire_red_asr: mem::zeroed::<_>(),
                transducer: mem::zeroed::<_>(),
                whisper: mem::zeroed::<_>(),
                sense_voice: mem::zeroed::<_>(),
                zipformer_ctc: mem::zeroed(),
                canary: mem::zeroed::<_>(),
            }
        };

        let config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
                decoding_method: null(),
                feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                    sample_rate: 16000,
                    feature_dim: 512,
                },
                hotwords_file: null(),
                hotwords_score: 0.0,
                lm_config: sherpa_rs_sys::SherpaOnnxOfflineLMConfig {
                    model: null(),
                    scale: 0.0,
                },
                max_active_paths: 0,
                model_config,
                rule_fars: null(),
                rule_fsts: null(),
                blank_penalty: 0.0,
                hr: mem::zeroed::<_>(),
            }
        };

        let recognizer = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&config) };

        if recognizer.is_null() {
            bail!("Failed to create recognizer");
        }

        Ok(Self { recognizer })
    }

    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> MoonshineRecognizerResult {
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
            let result = MoonshineRecognizerResult::new(&raw_result);
            // Free
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
            result
        }
    }
}

unsafe impl Send for MoonshineRecognizer {}
unsafe impl Sync for MoonshineRecognizer {}

impl Drop for MoonshineRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}
