use crate::{cstr, get_default_provider};
use std::{
    ffi::{CStr, CString},
    ptr::null,
};

#[derive(Debug)]
pub struct WhisperRecognizer {
    recognizer: *mut sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

#[derive(Debug)]
pub struct WhisperRecognizerResult {
    pub text: String,
    // pub timestamps: Vec<f32>,
}

impl WhisperRecognizer {
    pub fn new(
        decoder: String,
        encoder: String,
        tokens: String,
        language: String,
        debug: Option<bool>,
        provider: Option<String>,
        num_threads: Option<i32>,
        bpe_vocab: Option<String>,
    ) -> Self {
        let decoder_c = cstr!(decoder);
        let encoder_c = cstr!(encoder);
        let langauge_c = cstr!(language);
        let task_c = cstr!("transcribe".to_string());
        let tail_paddings = 0;
        let tokens_c = cstr!(tokens);

        let debug = debug.unwrap_or_default();
        let debug = if debug { 1 } else { 0 };
        let provider = provider.unwrap_or(get_default_provider());
        let provider_c = cstr!(provider);
        let num_threads = num_threads.unwrap_or(2);
        let bpe_vocab = bpe_vocab.unwrap_or("".into());
        let bpe_vocab_c = cstr!(bpe_vocab);

        let whisper = sherpa_rs_sys::SherpaOnnxOfflineWhisperModelConfig {
            decoder: decoder_c.into_raw(),
            encoder: encoder_c.into_raw(),
            language: langauge_c.into_raw(),
            task: task_c.into_raw(),
            tail_paddings,
        };
        let model_config = sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
            bpe_vocab: bpe_vocab_c.into_raw(),
            debug,
            model_type: null(),
            modeling_unit: null(),
            nemo_ctc: sherpa_rs_sys::SherpaOnnxOfflineNemoEncDecCtcModelConfig { model: null() },
            num_threads,
            paraformer: sherpa_rs_sys::SherpaOnnxOfflineParaformerModelConfig { model: null() },
            provider: provider_c.into_raw(),
            tdnn: sherpa_rs_sys::SherpaOnnxOfflineTdnnModelConfig { model: null() },
            telespeech_ctc: null(),
            tokens: tokens_c.into_raw(),
            transducer: sherpa_rs_sys::SherpaOnnxOfflineTransducerModelConfig {
                encoder: null(),
                decoder: null(),
                joiner: null(),
            },
            whisper,
        };
        let decoding_method_c = CString::new("greedy_search").unwrap();
        let config = sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
            decoding_method: decoding_method_c.into_raw(), // greedy_search, modified_beam_search
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
        };
        let recognizer = unsafe { sherpa_rs_sys::CreateOfflineRecognizer(&config) };

        Self { recognizer }
    }

    pub fn transcribe(&mut self, sample_rate: i32, samples: Vec<f32>) -> WhisperRecognizerResult {
        unsafe {
            let stream = sherpa_rs_sys::CreateOfflineStream(self.recognizer);
            sherpa_rs_sys::AcceptWaveformOffline(
                stream,
                sample_rate,
                samples.as_ptr(),
                samples.len().try_into().unwrap(),
            );
            sherpa_rs_sys::DecodeOfflineStream(self.recognizer, stream);
            let result_ptr = sherpa_rs_sys::GetOfflineStreamResult(stream);
            let raw_result = result_ptr.read();
            let text = CStr::from_ptr(raw_result.text);
            let text = text.to_str().unwrap().to_string();
            // let timestamps: &[f32] =
            // std::slice::from_raw_parts(raw_result.timestamps, raw_result.count as usize);
            let result = WhisperRecognizerResult { text };
            // Free
            sherpa_rs_sys::DestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::DestroyOfflineStream(stream);
            return result;
        }
    }
}

unsafe impl Send for WhisperRecognizer {}
unsafe impl Sync for WhisperRecognizer {}

impl Drop for WhisperRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::DestroyOfflineRecognizer(self.recognizer);
        }
    }
}
