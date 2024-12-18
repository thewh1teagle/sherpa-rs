use crate::{
    get_default_provider,
    utils::{cstr_to_string, RawCStr},
};
use eyre::{bail, Result};
use std::ptr::null;

#[derive(Debug)]
pub struct MoonshineRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

#[derive(Debug)]
pub struct MoonshineRecognizerResult {
    pub text: String,
    // pub timestamps: Vec<f32>,
}

#[derive(Debug)]
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
        let provider_ptr = RawCStr::new(&provider);
        let num_threads = config.num_threads.unwrap_or(2);

        // Moonshine
        let preprocessor_ptr = RawCStr::new(&config.preprocessor);
        let encoder_ptr = RawCStr::new(&config.encoder);
        let cached_decoder_ptr = RawCStr::new(&config.cached_decoder);
        let uncached_decoder_ptr = RawCStr::new(&config.uncached_decoder);
        let tokens_ptr = RawCStr::new(&config.tokens);

        let model_config = sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
            bpe_vocab: null(),
            debug,
            model_type: null(),
            modeling_unit: null(),
            nemo_ctc: sherpa_rs_sys::SherpaOnnxOfflineNemoEncDecCtcModelConfig { model: null() },
            num_threads,
            paraformer: sherpa_rs_sys::SherpaOnnxOfflineParaformerModelConfig { model: null() },
            provider: provider_ptr.as_ptr(),
            tdnn: sherpa_rs_sys::SherpaOnnxOfflineTdnnModelConfig { model: null() },
            telespeech_ctc: null(),
            tokens: tokens_ptr.as_ptr(),
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
            sense_voice: sherpa_rs_sys::SherpaOnnxOfflineSenseVoiceModelConfig {
                model: null(),
                language: null(),
                use_itn: 0,
            },
            moonshine: sherpa_rs_sys::SherpaOnnxOfflineMoonshineModelConfig {
                preprocessor: preprocessor_ptr.as_ptr(),
                encoder: encoder_ptr.as_ptr(),
                uncached_decoder: uncached_decoder_ptr.as_ptr(),
                cached_decoder: cached_decoder_ptr.as_ptr(),
            },
        };

        let config = sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
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
        };

        let recognizer = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&config) };

        if recognizer.is_null() {
            bail!("Failed to create recognizer")
        }

        Ok(Self { recognizer })
    }

    pub fn transcribe(&mut self, sample_rate: u32, samples: Vec<f32>) -> MoonshineRecognizerResult {
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
            let text = cstr_to_string(raw_result.text);
            let result = MoonshineRecognizerResult { text };
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
