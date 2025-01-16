use std::ptr::null;

use crate::{get_default_provider, utils::RawCStr};
use eyre::{bail, Result};
use hound::{WavSpec, WavWriter};

#[derive(Debug)]
pub struct OfflineTtsConfig {
    pub model: String,

    // Piper / Vits
    pub rule_fars: String,
    pub rule_fsts: String,
    pub max_num_sentences: i32,

    // speed
    pub length_scale: f32,

    // Kokoro
    pub voices_path: String,
    pub data_dir: String,

    // Onnx options
    pub num_threads: Option<i32>,
    pub debug: bool,
    pub provider: Option<String>,
    pub tokens: String,
}

#[derive(Debug)]
pub struct VitsConfig {
    pub lexicon: String,

    pub dict_dir: String,

    pub noise_scale: f32,
    pub noise_scale_w: f32,
}

impl Default for VitsConfig {
    fn default() -> Self {
        Self {
            lexicon: String::new(),

            dict_dir: String::new(),
            noise_scale: 0.0,
            noise_scale_w: 0.0,
        }
    }
}

impl Default for OfflineTtsConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            rule_fars: String::new(),
            rule_fsts: String::new(),
            voices_path: String::new(),
            data_dir: String::new(),
            max_num_sentences: 2,
            tokens: String::new(),
            num_threads: None,
            debug: false,
            provider: None,
            length_scale: 1.0,
        }
    }
}

#[derive(Debug)]
pub struct OfflineTts {
    pub(crate) tts: *const sherpa_rs_sys::SherpaOnnxOfflineTts,
}

impl OfflineTts {
    pub fn new(config: OfflineTtsConfig, vits_config: VitsConfig) -> Self {
        let provider = config.provider.unwrap_or(get_default_provider());

        let model = RawCStr::new(&config.model);

        let provider = RawCStr::new(&provider);

        // Vits / Piper
        let lexicon = RawCStr::new(&vits_config.lexicon);
        let tokens = RawCStr::new(&config.tokens);
        let rule_fars = RawCStr::new(&config.rule_fars);
        let rule_fsts = RawCStr::new(&config.rule_fsts);

        // Espeak
        let data_dir = RawCStr::new(&config.data_dir);
        let dict_dir = RawCStr::new(&vits_config.dict_dir);

        // Kokoro
        let voices_path = RawCStr::new(&config.voices_path);

        let vits: sherpa_rs_sys::SherpaOnnxOfflineTtsVitsModelConfig =
            unsafe { std::mem::zeroed() };
        let matcha: sherpa_rs_sys::SherpaOnnxOfflineTtsMatchaModelConfig =
            unsafe { std::mem::zeroed() };

        println!(
            "{:?} {} {} {} {}",
            config.model, config.voices_path, config.tokens, config.data_dir, config.length_scale,
        );
        let tts_config = sherpa_rs_sys::SherpaOnnxOfflineTtsConfig {
            max_num_sentences: config.max_num_sentences,
            model: sherpa_rs_sys::SherpaOnnxOfflineTtsModelConfig {
                vits,
                matcha,
                num_threads: config.num_threads.unwrap_or(1),
                debug: config.debug.into(),
                provider: provider.as_ptr(),
                kokoro: sherpa_rs_sys::SherpaOnnxOfflineTtsKokoroModelConfig {
                    model: model.as_ptr(),
                    voices: voices_path.as_ptr(),
                    tokens: tokens.as_ptr(),
                    data_dir: data_dir.as_ptr(),
                    length_scale: config.length_scale,
                },
            },
            rule_fars: null(),
            rule_fsts: null(),
        };

        let tts = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineTts(&tts_config) };
        Self { tts }
    }

    pub fn generate(&mut self, text: String, sid: i32, speed: f32) -> Result<TtsSample> {
        unsafe {
            let text = RawCStr::new(&text);
            let audio_ptr =
                sherpa_rs_sys::SherpaOnnxOfflineTtsGenerate(self.tts, text.as_ptr(), sid, speed);

            if audio_ptr.is_null() {
                bail!("audio is null")
            }
            let audio = audio_ptr.read();

            if audio.n.is_negative() {
                bail!("no samples found")
            }
            if audio.samples.is_null() {
                bail!("audio samples are null")
            }
            let samples: &[f32] = std::slice::from_raw_parts(audio.samples, audio.n as usize);
            let samples = samples.to_vec();
            let sample_rate = audio.sample_rate;
            let duration = samples.len() as i32 / sample_rate;

            // Free
            sherpa_rs_sys::SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio_ptr);

            Ok(TtsSample {
                samples,
                sample_rate: sample_rate as u32,
                duration,
            })
        }
    }
}

#[derive(Debug)]
pub struct TtsSample {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub duration: i32,
}

impl TtsSample {
    pub fn write_to_wav(&self, filename: &str) -> Result<()> {
        let spec = WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = WavWriter::create(filename, spec)?;

        for &sample in &self.samples {
            writer.write_sample(sample)?;
        }

        writer.finalize()?;

        Ok(())
    }
}

unsafe impl Send for OfflineTts {}
unsafe impl Sync for OfflineTts {}

impl Drop for OfflineTts {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineTts(self.tts);
        }
    }
}
