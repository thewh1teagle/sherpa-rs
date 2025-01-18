use std::{ mem, ptr::null };

use crate::{ utils::RawCStr, OnnxConfig };
use eyre::Result;
use sherpa_rs_sys;

use super::{ CommonTtsConfig, TtsAudio };

pub struct VitsTts {
    tts: *const sherpa_rs_sys::SherpaOnnxOfflineTts,
}

#[derive(Default)]
pub struct VitsTtsConfig {
    pub model: String,
    pub lexicon: String,
    pub dict_dir: String,
    pub tokens: String,
    pub data_dir: String,
    pub length_scale: f32,
    pub noise_scale: f32,
    pub noise_scale_w: f32,

    pub onnx_config: OnnxConfig,
    pub tts_config: CommonTtsConfig,
}

impl VitsTts {
    pub fn new(config: VitsTtsConfig) -> Self {
        let tts = unsafe {
            let model = RawCStr::new(&config.model);
            let tokens = RawCStr::new(&config.tokens);
            let data_dir = RawCStr::new(&config.data_dir);
            let lexicon = RawCStr::new(&config.lexicon);
            let dict_dir = RawCStr::new(&config.dict_dir);

            let provider = RawCStr::new(&config.onnx_config.provider);

            let tts_config = config.tts_config.to_raw();

            let model_config = sherpa_rs_sys::SherpaOnnxOfflineTtsModelConfig {
                num_threads: config.onnx_config.num_threads,
                vits: sherpa_rs_sys::SherpaOnnxOfflineTtsVitsModelConfig {
                    model: model.as_ptr(),
                    lexicon: lexicon.as_ptr(),
                    tokens: tokens.as_ptr(),
                    data_dir: data_dir.as_ptr(),
                    noise_scale: config.noise_scale,
                    noise_scale_w: config.noise_scale_w,
                    length_scale: config.length_scale,
                    dict_dir: dict_dir.as_ptr(),
                },
                debug: config.onnx_config.debug.into(),
                provider: provider.as_ptr(),
                matcha: mem::zeroed::<_>(),
                kokoro: mem::zeroed::<_>(),
            };
            let config = sherpa_rs_sys::SherpaOnnxOfflineTtsConfig {
                max_num_sentences: config.tts_config.max_num_sentences,
                model: model_config,
                rule_fars: tts_config.rule_fars.map(|v| v.as_ptr()).unwrap_or(null()),
                rule_fsts: tts_config.rule_fsts.map(|v| v.as_ptr()).unwrap_or(null()),
            };
            sherpa_rs_sys::SherpaOnnxCreateOfflineTts(&config)
        };

        Self { tts }
    }

    pub fn create(&mut self, text: &str, sid: i32, speed: f32) -> Result<TtsAudio> {
        unsafe { super::create(self.tts, text, sid, speed) }
    }
}
