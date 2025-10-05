use std::{mem, ptr::null};

use crate::{utils::cstring_from_str, OnnxConfig};
use eyre::Result;
use sherpa_rs_sys;

use super::{CommonTtsConfig, TtsAudio};

pub struct KittenTts {
    tts: *const sherpa_rs_sys::SherpaOnnxOfflineTts,
}

#[derive(Default)]
pub struct KittenTtsConfig {
    pub model: String,
    pub voices: String,
    pub tokens: String,
    pub data_dir: String,
    pub length_scale: f32,
    pub onnx_config: OnnxConfig,
    pub common_config: CommonTtsConfig,
}

impl KittenTts {
    pub fn new(config: KittenTtsConfig) -> Self {
        let tts = unsafe {
            let model = cstring_from_str(&config.model);
            let voices = cstring_from_str(&config.voices);
            let tokens = cstring_from_str(&config.tokens);
            let data_dir = cstring_from_str(&config.data_dir);

            let provider = cstring_from_str(&config.onnx_config.provider);

            let tts_config = config.common_config.to_raw();

            let model_config = sherpa_rs_sys::SherpaOnnxOfflineTtsModelConfig {
                vits: mem::zeroed::<_>(),
                num_threads: config.onnx_config.num_threads,
                debug: config.onnx_config.debug.into(),
                provider: provider.as_ptr(),
                matcha: mem::zeroed::<_>(),
                kokoro: mem::zeroed(),
                kitten: sherpa_rs_sys::SherpaOnnxOfflineTtsKittenModelConfig {
                    model: model.as_ptr(),
                    voices: voices.as_ptr(),
                    tokens: tokens.as_ptr(),
                    data_dir: data_dir.as_ptr(),
                    length_scale: config.length_scale,
                },
            };
            let config = sherpa_rs_sys::SherpaOnnxOfflineTtsConfig {
                max_num_sentences: config.common_config.max_num_sentences,
                model: model_config,
                rule_fars: tts_config.rule_fars.map(|v| v.as_ptr()).unwrap_or(null()),
                rule_fsts: tts_config.rule_fsts.map(|v| v.as_ptr()).unwrap_or(null()),
                silence_scale: 1.0,
            };
            sherpa_rs_sys::SherpaOnnxCreateOfflineTts(&config)
        };

        Self { tts }
    }

    pub fn create(&mut self, text: &str, sid: i32, speed: f32) -> Result<TtsAudio> {
        unsafe { super::create(self.tts, text, sid, speed) }
    }
}

unsafe impl Send for KittenTts {}
unsafe impl Sync for KittenTts {}

impl Drop for KittenTts {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineTts(self.tts);
        }
    }
}
