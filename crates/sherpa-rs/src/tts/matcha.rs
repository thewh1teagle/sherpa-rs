use std::{mem, ptr::null};

use crate::{utils::cstring_from_str, OnnxConfig};
use eyre::Result;
use sherpa_rs_sys;

use super::{CommonTtsConfig, TtsAudio};

pub struct MatchaTts {
    tts: *const sherpa_rs_sys::SherpaOnnxOfflineTts,
}

#[derive(Default)]
pub struct MatchaTtsConfig {
    pub model: String,
    pub lexicon: String,
    pub dict_dir: String,
    pub tokens: String,
    pub data_dir: String,
    pub acoustic_model: String,
    pub vocoder: String,
    pub length_scale: f32,
    pub noise_scale: f32,
    pub noise_scale_w: f32,
    pub silence_scale: f32,

    pub common_config: CommonTtsConfig,
    pub onnx_config: OnnxConfig,
}

impl MatchaTts {
    pub fn new(config: MatchaTtsConfig) -> Self {
        let tts = unsafe {
            let tokens = cstring_from_str(&config.tokens);
            let data_dir = cstring_from_str(&config.data_dir);
            let lexicon = cstring_from_str(&config.lexicon);
            let dict_dir = cstring_from_str(&config.dict_dir);

            let vocoder = cstring_from_str(&config.vocoder);
            let acoustic_model = cstring_from_str(&config.acoustic_model);

            let provider = cstring_from_str(&config.onnx_config.provider);

            let tts_config = config.common_config.to_raw();

            let model_config = sherpa_rs_sys::SherpaOnnxOfflineTtsModelConfig {
                num_threads: config.onnx_config.num_threads,
                vits: mem::zeroed::<_>(),
                debug: config.onnx_config.debug.into(),
                provider: provider.as_ptr(),
                matcha: sherpa_rs_sys::SherpaOnnxOfflineTtsMatchaModelConfig {
                    acoustic_model: acoustic_model.as_ptr(),
                    vocoder: vocoder.as_ptr(),
                    lexicon: lexicon.as_ptr(),
                    tokens: tokens.as_ptr(),
                    data_dir: data_dir.as_ptr(),
                    noise_scale: config.noise_scale,
                    length_scale: config.length_scale,
                    dict_dir: dict_dir.as_ptr(),
                },
                kokoro: mem::zeroed::<_>(),
                kitten: mem::zeroed::<_>(),
            };
            let config = sherpa_rs_sys::SherpaOnnxOfflineTtsConfig {
                max_num_sentences: config.common_config.max_num_sentences,
                model: model_config,
                rule_fars: tts_config.rule_fars.map(|v| v.as_ptr()).unwrap_or(null()),
                rule_fsts: tts_config.rule_fsts.map(|v| v.as_ptr()).unwrap_or(null()),
                silence_scale: config.silence_scale,
            };
            sherpa_rs_sys::SherpaOnnxCreateOfflineTts(&config)
        };

        Self { tts }
    }

    pub fn create(&mut self, text: &str, sid: i32, speed: f32) -> Result<TtsAudio> {
        unsafe { super::create(self.tts, text, sid, speed) }
    }
}

unsafe impl Send for MatchaTts {}
unsafe impl Sync for MatchaTts {}

impl Drop for MatchaTts {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineTts(self.tts);
        }
    }
}
