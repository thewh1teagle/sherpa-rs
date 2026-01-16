use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::{mem, ptr::null};

#[derive(Debug)]
pub struct FunAsrNanoRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

pub type FunAsrNanoRecognizerResult = super::OfflineRecognizerResult;

#[derive(Debug, Clone)]
pub struct FunAsrNanoConfig {
    pub encoder_adaptor: String,
    pub llm: String,
    pub embedding: String,
    pub tokenizer: String,
    pub tokens: String,
    pub system_prompt: Option<String>,
    pub user_prompt: Option<String>,
    pub max_new_tokens: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: i32,
    pub decoding_method: String,

    pub provider: Option<String>,
    pub num_threads: Option<i32>,
    pub debug: bool,
}

impl Default for FunAsrNanoConfig {
    fn default() -> Self {
        Self {
            encoder_adaptor: String::new(),
            llm: String::new(),
            embedding: String::new(),
            tokenizer: String::new(),
            tokens: String::new(),
            system_prompt: None,
            user_prompt: None,
            max_new_tokens: 200,
            temperature: 0.0,
            top_p: 1.0,
            seed: 0,
            decoding_method: "greedy_search".into(),
            debug: false,
            provider: None,
            num_threads: Some(1),
        }
    }
}

impl FunAsrNanoRecognizer {
    pub fn new(config: FunAsrNanoConfig) -> Result<Self> {
        let debug = config.debug.into();
        let provider = config.provider.unwrap_or(get_default_provider());

        let provider_ptr = cstring_from_str(&provider);
        let num_threads = config.num_threads.unwrap_or(1);
        let tokens_ptr = cstring_from_str(&config.tokens);
        let decoding_method_ptr = cstring_from_str(&config.decoding_method);

        // FunASR Nano specific fields
        let encoder_adaptor_ptr = cstring_from_str(&config.encoder_adaptor);
        let llm_ptr = cstring_from_str(&config.llm);
        let embedding_ptr = cstring_from_str(&config.embedding);
        let tokenizer_ptr = cstring_from_str(&config.tokenizer);
        let system_prompt_ptr = config.system_prompt.as_ref().map(|s| cstring_from_str(s));
        let user_prompt_ptr = config.user_prompt.as_ref().map(|s| cstring_from_str(s));

        let funasr_nano_config = sherpa_rs_sys::SherpaOnnxOfflineFunASRNanoModelConfig {
            encoder_adaptor: encoder_adaptor_ptr.as_ptr(),
            llm: llm_ptr.as_ptr(),
            embedding: embedding_ptr.as_ptr(),
            tokenizer: tokenizer_ptr.as_ptr(),
            system_prompt: system_prompt_ptr.as_ref().map_or(null(), |s| s.as_ptr()),
            user_prompt: user_prompt_ptr.as_ref().map_or(null(), |s| s.as_ptr()),
            max_new_tokens: config.max_new_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            seed: config.seed,
        };

        let model_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
                debug,
                num_threads,
                provider: provider_ptr.as_ptr(),
                funasr_nano: funasr_nano_config,
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
                dolphin: mem::zeroed::<_>(),
                bpe_vocab: mem::zeroed::<_>(),
                model_type: mem::zeroed::<_>(),
                modeling_unit: mem::zeroed::<_>(),
                zipformer_ctc: mem::zeroed::<_>(),
                canary: mem::zeroed::<_>(),
                wenet_ctc: mem::zeroed::<_>(),
                omnilingual: mem::zeroed::<_>(),
                medasr: mem::zeroed::<_>(),
            }
        };

        let recognizer_config = unsafe {
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

        let recognizer =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&recognizer_config) };

        if recognizer.is_null() {
            bail!("Failed to create FunASR Nano recognizer");
        }

        Ok(Self { recognizer })
    }

    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> FunAsrNanoRecognizerResult {
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
            let result = FunAsrNanoRecognizerResult::new(&raw_result);
            // Free
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
            result
        }
    }
}

unsafe impl Send for FunAsrNanoRecognizer {}
unsafe impl Sync for FunAsrNanoRecognizer {}

impl Drop for FunAsrNanoRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}
