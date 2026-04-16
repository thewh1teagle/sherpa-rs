use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};

#[derive(Debug)]
pub struct CohereTranscribeRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

pub type CohereTranscribeResult = super::OfflineRecognizerResult;

#[derive(Debug, Clone)]
pub struct CohereTranscribeConfig {
    pub encoder: String,
    pub decoder: String,
    pub tokens: String,
    pub language: String,
    pub use_punct: bool,
    pub use_itn: bool,

    pub provider: Option<String>,
    pub num_threads: Option<i32>,
    pub debug: bool,
}

impl Default for CohereTranscribeConfig {
    fn default() -> Self {
        Self {
            encoder: String::new(),
            decoder: String::new(),
            tokens: String::new(),
            language: String::from("en"),
            use_punct: true,
            use_itn: true,
            debug: false,
            provider: None,
            num_threads: Some(1),
        }
    }
}

impl CohereTranscribeRecognizer {
    pub fn new(config: CohereTranscribeConfig) -> Result<Self> {
        let debug = config.debug.into();
        let provider = config.provider.unwrap_or(get_default_provider());

        let provider_ptr = cstring_from_str(&provider);
        let num_threads = config.num_threads.unwrap_or(2);

        let encoder_ptr = cstring_from_str(&config.encoder);
        let decoder_ptr = cstring_from_str(&config.decoder);
        let tokens_ptr = cstring_from_str(&config.tokens);
        let language_ptr = cstring_from_str(&config.language);
        let decoding_method_ptr = cstring_from_str("greedy_search");

        let cohere_config = sherpa_rs_sys::SherpaOnnxOfflineCohereTranscribeModelConfig {
            encoder: encoder_ptr.as_ptr(),
            decoder: decoder_ptr.as_ptr(),
            language: language_ptr.as_ptr(),
            use_punct: config.use_punct as i32,
            use_itn: config.use_itn as i32,
        };

        let model_config = sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
                cohere_transcribe: cohere_config,
                debug,
                num_threads,
                provider: provider_ptr.as_ptr(),
                tokens: tokens_ptr.as_ptr(),
                ..Default::default()
            };

        let config = sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
            decoding_method: decoding_method_ptr.as_ptr(),
            feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                sample_rate: 16000,
                feature_dim: 80,
            },
            model_config,
            ..Default::default()
        };

        let recognizer = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&config) };

        if recognizer.is_null() {
            bail!("Failed to create Cohere Transcribe recognizer");
        }

        Ok(Self { recognizer })
    }

    pub fn transcribe(
        &mut self,
        sample_rate: u32,
        samples: &[f32],
    ) -> CohereTranscribeResult {
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
            let result = CohereTranscribeResult::new(&raw_result);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
            result
        }
    }
}

unsafe impl Send for CohereTranscribeRecognizer {}
unsafe impl Sync for CohereTranscribeRecognizer {}

impl Drop for CohereTranscribeRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}
