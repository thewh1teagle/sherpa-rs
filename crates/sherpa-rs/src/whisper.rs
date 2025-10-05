use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::mem;

#[derive(Debug)]
pub struct WhisperRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

pub type WhisperRecognizerResult = super::OfflineRecognizerResult;

#[derive(Debug, Clone)]
pub struct WhisperConfig {
    pub decoder: String,
    pub encoder: String,
    pub tokens: String,
    pub language: String,
    pub bpe_vocab: Option<String>,
    pub tail_paddings: Option<i32>,

    pub provider: Option<String>,
    pub num_threads: Option<i32>,
    pub debug: bool,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            decoder: String::new(),
            encoder: String::new(),
            tokens: String::new(),
            language: String::from("en"),
            bpe_vocab: None,
            tail_paddings: None,
            debug: false,
            provider: None,
            num_threads: Some(1),
        }
    }
}

impl WhisperRecognizer {
    pub fn new(config: WhisperConfig) -> Result<Self> {
        let debug = config.debug.into();
        let provider = config.provider.unwrap_or(get_default_provider());

        // Onnx
        let provider_ptr = cstring_from_str(&provider);
        let num_threads = config.num_threads.unwrap_or(2);

        // Whisper
        let bpe_vocab_ptr = cstring_from_str(&config.bpe_vocab.unwrap_or("".into()));
        let tail_paddings = config.tail_paddings.unwrap_or(0);
        let decoder_ptr = cstring_from_str(&config.decoder);
        let encoder_ptr = cstring_from_str(&config.encoder);
        let language_ptr = cstring_from_str(&config.language);
        let task_ptr = cstring_from_str("transcribe");
        let tokens_ptr = cstring_from_str(&config.tokens);
        let decoding_method_ptr = cstring_from_str("greedy_search");

        let whisper_config = sherpa_rs_sys::SherpaOnnxOfflineWhisperModelConfig {
            decoder: decoder_ptr.as_ptr(),
            encoder: encoder_ptr.as_ptr(),
            language: language_ptr.as_ptr(),
            task: task_ptr.as_ptr(),
            tail_paddings,
        };
        let model_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
                whisper: whisper_config,
                debug,
                num_threads,
                provider: provider_ptr.as_ptr(),
                bpe_vocab: bpe_vocab_ptr.as_ptr(),
                tokens: tokens_ptr.as_ptr(),

                // nulls
                model_type: std::ptr::null(),
                modeling_unit: mem::zeroed::<_>(),
                nemo_ctc: mem::zeroed::<_>(),
                paraformer: mem::zeroed::<_>(),
                tdnn: mem::zeroed::<_>(),
                telespeech_ctc: mem::zeroed::<_>(),
                transducer: mem::zeroed::<_>(),
                fire_red_asr: mem::zeroed::<_>(),
                sense_voice: mem::zeroed::<_>(),
                moonshine: mem::zeroed::<_>(),
                dolphin: mem::zeroed::<_>(),
                zipformer_ctc: mem::zeroed::<_>(),
                canary: mem::zeroed::<_>(),
            }
        };

        let config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
                decoding_method: decoding_method_ptr.as_ptr(), // greedy_search, modified_beam_search
                feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                    sample_rate: 16000,
                    feature_dim: 512,
                },
                model_config,

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

    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> WhisperRecognizerResult {
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
            let result = WhisperRecognizerResult::new(&raw_result);
            // Free
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
            result
        }
    }
}

unsafe impl Send for WhisperRecognizer {}
unsafe impl Sync for WhisperRecognizer {}

impl Drop for WhisperRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}
