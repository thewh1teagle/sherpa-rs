use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::ptr::null;

#[derive(Debug)]
pub struct SenseVoiceRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

pub type SenseVoiceRecognizerResult = super::OfflineRecognizerResult;

#[derive(Debug, Clone)]
pub struct SenseVoiceConfig {
    pub model: String,
    pub language: String,
    pub use_itn: bool,
    pub provider: Option<String>,
    pub num_threads: Option<i32>,
    pub debug: bool,
    pub tokens: String,
    pub hotwords_file: Option<String>,
    pub hotwords_score: Option<f32>,
}

impl Default for SenseVoiceConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            language: "auto".into(),
            use_itn: true,
            provider: None,
            num_threads: Some(1),
            debug: false,
            tokens: String::new(),
            hotwords_file: None,
            hotwords_score: Some(0.0),
        }
    }
}

impl SenseVoiceRecognizer {
    pub fn new(config: SenseVoiceConfig) -> Result<Self> {
        let debug = config.debug.into();
        let provider = config.provider.unwrap_or(get_default_provider());
        let provider_ptr = cstring_from_str(&provider);
        let num_threads = config.num_threads.unwrap_or(1);

        // SenseVoice specific config
        let model_ptr = cstring_from_str(&config.model);
        let language_ptr = cstring_from_str(&config.language);
        let use_itn = if config.use_itn { 1 } else { 0 };

        let sense_voice_config = sherpa_rs_sys::SherpaOnnxOfflineSenseVoiceModelConfig {
            model: model_ptr.as_ptr(),
            language: language_ptr.as_ptr(),
            use_itn,
        };

        // General model config
        let tokens_ptr = cstring_from_str(&config.tokens);
        let model_config = sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
            tokens: tokens_ptr.as_ptr(),
            provider: provider_ptr.as_ptr(),
            num_threads,
            debug,
            sense_voice: sense_voice_config,
            // Other fields set to default/null
            bpe_vocab: null(),
            model_type: null(),
            modeling_unit: null(),
            nemo_ctc: sherpa_rs_sys::SherpaOnnxOfflineNemoEncDecCtcModelConfig { model: null() },
            paraformer: sherpa_rs_sys::SherpaOnnxOfflineParaformerModelConfig { model: null() },
            tdnn: sherpa_rs_sys::SherpaOnnxOfflineTdnnModelConfig { model: null() },
            telespeech_ctc: null(),
            fire_red_asr: sherpa_rs_sys::SherpaOnnxOfflineFireRedAsrModelConfig {
                encoder: null(),
                decoder: null(),
            },
            transducer: sherpa_rs_sys::SherpaOnnxOfflineTransducerModelConfig {
                encoder: null(),
                decoder: null(),
                joiner: null(),
            },
            whisper: sherpa_rs_sys::SherpaOnnxOfflineWhisperModelConfig {
                encoder: null(),
                decoder: null(),
                language: null(),
                task: null(),
                tail_paddings: 0,
            },
            moonshine: sherpa_rs_sys::SherpaOnnxOfflineMoonshineModelConfig {
                preprocessor: null(),
                encoder: null(),
                uncached_decoder: null(),
                cached_decoder: null(),
            },
        };

        let hotwords_score = config.hotwords_score.unwrap_or(0.0);
        let hotwords_file_cstr = config.hotwords_file.map(|f| cstring_from_str(&f));
        // Recognizer config
        let config = sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
            decoding_method: cstring_from_str("modified_beam_search").as_ptr(),
            feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                sample_rate: 16000,
                feature_dim: 80,
            },
            hotwords_file: hotwords_file_cstr
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(null()),
            hotwords_score,
            lm_config: sherpa_rs_sys::SherpaOnnxOfflineLMConfig {
                model: null(),
                scale: 0.0,
            },
            max_active_paths: 0,
            model_config,
            rule_fars: null(),
            rule_fsts: null(),
            blank_penalty: 0.0,
        };

        let recognizer = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&config) };
        if recognizer.is_null() {
            bail!("Failed to create recognizer");
        }

        Ok(Self { recognizer })
    }

    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> SenseVoiceRecognizerResult {
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
            let result = SenseVoiceRecognizerResult::new(&raw_result);
            // Free resources
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
            result
        }
    }
}

unsafe impl Send for SenseVoiceRecognizer {}
unsafe impl Sync for SenseVoiceRecognizer {}

impl Drop for SenseVoiceRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}
