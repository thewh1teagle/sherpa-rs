use crate::{
    get_default_provider,
    utils::{cstr_to_string, RawCStr},
};
use eyre::{bail, Result};
use std::ptr::null;

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
        let decoder_ptr = RawCStr::new(&config.decoder);
        let encoder_ptr = RawCStr::new(&config.encoder);
        let joiner_ptr = RawCStr::new(&config.joiner);
        let provider_ptr = RawCStr::new(&config.provider.unwrap_or(get_default_provider()));
        let tokens_ptr = RawCStr::new(&config.tokens);
        let decoding_method_ptr = RawCStr::new("greedy_search");

        let transcuder_config = sherpa_rs_sys::SherpaOnnxOfflineTransducerModelConfig {
            decoder: decoder_ptr.as_ptr(),
            encoder: encoder_ptr.as_ptr(),
            joiner: joiner_ptr.as_ptr(),
        };
        // Offline model config
        let model_config = sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
            num_threads: config.num_threads.unwrap_or(1),
            debug: config.debug.into(),
            provider: provider_ptr.as_ptr(),
            transducer: transcuder_config,
            tokens: tokens_ptr.as_ptr(),
            // NULLs
            bpe_vocab: null(),
            model_type: null(),
            modeling_unit: null(),
            paraformer: sherpa_rs_sys::SherpaOnnxOfflineParaformerModelConfig { model: null() },
            tdnn: sherpa_rs_sys::SherpaOnnxOfflineTdnnModelConfig { model: null() },
            telespeech_ctc: null(),

            nemo_ctc: sherpa_rs_sys::SherpaOnnxOfflineNemoEncDecCtcModelConfig { model: null() },
            whisper: sherpa_rs_sys::SherpaOnnxOfflineWhisperModelConfig {
                encoder: null(),
                decoder: null(),
                language: null(),
                task: null(),
                tail_paddings: 0,
            },
            sense_voice: sherpa_rs_sys::SherpaOnnxOfflineSenseVoiceModelConfig {
                language: null(),
                model: null(),
                use_itn: 0,
            },
        };
        // Recognizer config
        let recognizer_config = sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
            model_config,
            decoding_method: decoding_method_ptr.as_ptr(),
            // NULLs
            blank_penalty: 0.0,
            feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                sample_rate: 0,
                feature_dim: 0,
            },
            hotwords_file: null(),
            hotwords_score: 0.0,
            lm_config: sherpa_rs_sys::SherpaOnnxOfflineLMConfig {
                model: null(),
                scale: 0.0,
            },
            max_active_paths: 0,
            rule_fars: null(),
            rule_fsts: null(),
        };

        let recognizer =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&recognizer_config) };

        if recognizer.is_null() {
            bail!("Failed to create recognizer")
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
            let text = cstr_to_string(raw_result.text);

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
