use std::{mem, ptr::null};

use crate::{utils::cstring_from_str, OnnxConfig};
use eyre::Result;
use sherpa_rs_sys;

use super::{CommonTtsConfig, TtsAudio};

pub struct KokoroTts {
    tts: *const sherpa_rs_sys::SherpaOnnxOfflineTts,
}

#[derive(Default)]
pub struct KokoroTtsConfig {
    pub model: String,
    pub voices: String,
    pub tokens: String,
    pub data_dir: String,
    pub dict_dir: String,
    pub lexicon: String,
    pub length_scale: f32,
    pub onnx_config: OnnxConfig,
    pub common_config: CommonTtsConfig,
    pub lang: String,
}

impl KokoroTts {
    pub fn new(config: KokoroTtsConfig) -> Self {
        let tts = unsafe {
            let model = cstring_from_str(&config.model);
            let voices = cstring_from_str(&config.voices);
            let tokens = cstring_from_str(&config.tokens);
            let data_dir = cstring_from_str(&config.data_dir);
            let dict_dir = cstring_from_str(&config.dict_dir);
            let lexicon = cstring_from_str(&config.lexicon);
            let lang = cstring_from_str(&config.lang);

            let provider = cstring_from_str(&config.onnx_config.provider);

            let tts_config = config.common_config.to_raw();

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
                    dict_dir: dict_dir.as_ptr(),
                    lexicon: lexicon.as_ptr(),
                    lang: lang.as_ptr(),
                },
                kitten: mem::zeroed::<_>(),
                zipvoice: mem::zeroed::<_>(),
            };
            let config = sherpa_rs_sys::SherpaOnnxOfflineTtsConfig {
                max_num_sentences: config.common_config.max_num_sentences,
                model: model_config,
                rule_fars: tts_config.rule_fars.map(|v| v.as_ptr()).unwrap_or(null()),
                rule_fsts: tts_config.rule_fsts.map(|v| v.as_ptr()).unwrap_or(null()),
                silence_scale: config.common_config.silence_scale,
            };
            sherpa_rs_sys::SherpaOnnxCreateOfflineTts(&config)
        };

        Self { tts }
    }

    pub fn create(&mut self, text: &str, sid: i32, speed: f32) -> Result<TtsAudio> {
        unsafe { super::create(self.tts, text, sid, speed) }
    }
}

unsafe impl Send for KokoroTts {}
unsafe impl Sync for KokoroTts {}

impl Drop for KokoroTts {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineTts(self.tts);
        }
    }
}
