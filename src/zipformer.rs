use crate::{cstr, cstr_to_string, get_default_provider};
use eyre::{bail, Result};
use std::ptr::null;

#[derive(Debug, Default)]
pub struct ZipFormerConfig {
    pub decoder: String,
    pub encoder: String,
    pub joiner: String,
    pub tokens: String,
    pub debug: Option<bool>,
    pub num_threads: Option<i32>,
    pub provider: Option<String>,
}

pub struct ZipFormer {
    recognizer: *mut sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

impl ZipFormer {
    pub fn new(config: ZipFormerConfig) -> Result<Self> {
        // Zipformer config
        let transcuder_config = sherpa_rs_sys::SherpaOnnxOfflineTransducerModelConfig {
            decoder: cstr!(config.decoder).into_raw(),
            encoder: cstr!(config.encoder).into_raw(),
            joiner: cstr!(config.joiner).into_raw(),
        };
        // Offline model config
        let model_config = sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
            num_threads: config.num_threads.unwrap_or(1),
            debug: config.debug.unwrap_or_default().into(),
            provider: cstr!(config.provider.unwrap_or(get_default_provider())).into_raw(),
            transducer: transcuder_config,
            tokens: cstr!(config.tokens).into_raw(),
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
            decoding_method: cstr!("greedy_search").into_raw(),
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
            let text = cstr_to_string!(raw_result.text);

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
