use crate::{
    get_default_provider,
    utils::{cstr_to_string, cstring_from_str},
};
use eyre::{bail, Result};
use std::mem;

#[derive(Debug, Default)]
pub struct ZipFormerConfig {
    pub decoder: String,
    pub encoder: String,
    pub joiner: String,
    pub tokens: String,

    pub num_threads: Option<i32>,
    pub provider: Option<String>,
    pub debug: bool,
}

pub struct ZipFormer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

impl ZipFormer {
    pub fn new(config: ZipFormerConfig) -> Result<Self> {
        // Zipformer config
        let decoder_ptr = cstring_from_str(&config.decoder);
        let encoder_ptr = cstring_from_str(&config.encoder);
        let joiner_ptr = cstring_from_str(&config.joiner);
        let provider_ptr = cstring_from_str(&config.provider.unwrap_or(get_default_provider()));
        let tokens_ptr = cstring_from_str(&config.tokens);
        let decoding_method_ptr = cstring_from_str("greedy_search");

        let transcuder_config = sherpa_rs_sys::SherpaOnnxOfflineTransducerModelConfig {
            decoder: decoder_ptr.as_ptr(),
            encoder: encoder_ptr.as_ptr(),
            joiner: joiner_ptr.as_ptr(),
        };
        // Offline model config
        let model_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
                num_threads: config.num_threads.unwrap_or(1),
                debug: config.debug.into(),
                provider: provider_ptr.as_ptr(),
                transducer: transcuder_config,
                tokens: tokens_ptr.as_ptr(),
                // NULLs
                bpe_vocab: mem::zeroed::<_>(),
                model_type: mem::zeroed::<_>(),
                modeling_unit: mem::zeroed::<_>(),
                paraformer: mem::zeroed::<_>(),
                tdnn: mem::zeroed::<_>(),
                fire_red_asr: mem::zeroed::<_>(),
                telespeech_ctc: mem::zeroed::<_>(),
                nemo_ctc: mem::zeroed::<_>(),
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
                model_config,
                decoding_method: decoding_method_ptr.as_ptr(),
                // NULLs
                blank_penalty: mem::zeroed::<_>(),
                feat_config: mem::zeroed::<_>(),
                hotwords_file: mem::zeroed::<_>(),
                hotwords_score: mem::zeroed::<_>(),
                lm_config: mem::zeroed::<_>(),
                max_active_paths: mem::zeroed::<_>(),
                rule_fars: mem::zeroed::<_>(),
                rule_fsts: mem::zeroed::<_>(),
                hr: mem::zeroed::<_>(),
            }
        };

        let recognizer =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&recognizer_config) };

        if recognizer.is_null() {
            bail!("Failed to create recognizer");
        }
        Ok(Self { recognizer })
    }

    pub fn decode(&mut self, sample_rate: u32, samples: Vec<f32>) -> String {
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

unsafe impl Send for ZipFormer {}
unsafe impl Sync for ZipFormer {}

impl Drop for ZipFormer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}
