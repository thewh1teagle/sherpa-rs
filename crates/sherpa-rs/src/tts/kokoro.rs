use std::{ mem, ptr::null };

use eyre::Result;
use sherpa_rs_sys;
use crate::{ utils::RawCStr, OnnxConfig };

use super::TtsAudio;

pub struct KokoroTts {
    tts: *const sherpa_rs_sys::SherpaOnnxOfflineTts,
}

#[derive(Default)]
pub struct KokoroTtsConfig {
    pub model: String,
    pub voices: String,
    pub tokens: String,
    pub data_dir: String,
    pub length_scale: f32,
    pub onnx_config: OnnxConfig,
}

impl KokoroTts {
    pub fn new(config: KokoroTtsConfig) -> Self {
        let tts = unsafe {
            let model = RawCStr::new(&config.model);
            let voices = RawCStr::new(&config.voices);
            let tokens = RawCStr::new(&config.tokens);
            let data_dir = RawCStr::new(&config.data_dir);

            let provider = RawCStr::new(&config.onnx_config.provider);

            let model_config = sherpa_rs_sys::SherpaOnnxOfflineTtsModelConfig {
                vits: mem::zeroed::<_>(),
                num_threads: config.onnx_config.num_threads,
                debug: config.onnx_config.debug.into(),
                provider: provider.as_ptr(),
                matcha: mem::zeroed::<_>(),
                kokoro: sherpa_rs_sys::SherpaOnnxOfflineTtsKokoroModelConfig {
                    model: model.as_ptr(),
                    voices: voices.as_ptr(),
                    tokens: tokens.as_ptr(),
                    data_dir: data_dir.as_ptr(),
                    length_scale: config.length_scale,
                },
            };
            let config = sherpa_rs_sys::SherpaOnnxOfflineTtsConfig {
                max_num_sentences: 0,
                model: model_config,
                rule_fars: null(),
                rule_fsts: null(),
            };
            sherpa_rs_sys::SherpaOnnxCreateOfflineTts(&config)
        };

        Self {
            tts,
        }
    }

    pub fn create(&mut self, text: &str, sid: i32, speed: f32) -> Result<TtsAudio> {
        unsafe { super::create(self.tts, text, sid, speed) }
    }
}
