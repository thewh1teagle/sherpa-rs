use crate::{cstr, get_default_provider};
use eyre::{bail, Result};
use hound::{WavSpec, WavWriter};

#[derive(Debug)]
pub struct OfflineTtsConfig {
    pub model: String,
    pub rule_fars: String,
    pub rule_fsts: String,
    pub max_num_sentences: i32,
    pub num_threads: Option<i32>,
    pub debug: Option<bool>,
    pub provider: Option<String>,
}

#[derive(Debug)]
pub struct VitsConfig {
    pub lexicon: String,
    pub tokens: String,
    pub data_dir: String,
    pub dict_dir: String,

    pub noise_scale: f32,
    pub noise_scale_w: f32,
    pub length_scale: f32,
}

impl Default for VitsConfig {
    fn default() -> Self {
        Self {
            lexicon: String::new(),
            tokens: String::new(),
            data_dir: String::new(),
            dict_dir: String::new(),
            noise_scale: 0.0,
            noise_scale_w: 0.0,
            length_scale: 1.0,
        }
    }
}

impl Default for OfflineTtsConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            rule_fars: String::new(),
            rule_fsts: String::new(),
            max_num_sentences: 2,
            num_threads: None,
            debug: None,
            provider: None,
        }
    }
}

#[derive(Debug)]
pub struct OfflineTts {
    pub(crate) tts: *mut sherpa_rs_sys::SherpaOnnxOfflineTts,
}

impl OfflineTts {
    pub fn new(config: OfflineTtsConfig, vits_config: VitsConfig) -> Self {
        let provider = config.provider.unwrap_or(get_default_provider());
        let tts_config = sherpa_rs_sys::SherpaOnnxOfflineTtsConfig {
            max_num_sentences: config.max_num_sentences,
            model: sherpa_rs_sys::SherpaOnnxOfflineTtsModelConfig {
                vits: sherpa_rs_sys::SherpaOnnxOfflineTtsVitsModelConfig {
                    data_dir: cstr!(vits_config.data_dir).into_raw(),
                    dict_dir: cstr!(vits_config.dict_dir).into_raw(),
                    length_scale: vits_config.length_scale,
                    lexicon: cstr!(vits_config.lexicon).into_raw(),
                    model: cstr!(config.model).into_raw(),
                    noise_scale: vits_config.noise_scale,
                    noise_scale_w: vits_config.noise_scale_w,
                    tokens: cstr!(vits_config.tokens).into_raw(),
                },
                num_threads: config.num_threads.unwrap_or(1),
                debug: config.debug.unwrap_or(false).into(),
                provider: cstr!(provider).into_raw(),
            },
            rule_fars: cstr!(config.rule_fars).into_raw(),
            rule_fsts: cstr!(config.rule_fsts).into_raw(),
        };
        let tts = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineTts(&tts_config) };
        Self { tts }
    }

    pub fn generate(&mut self, text: String, sid: i32, speed: f32) -> Result<TtsSample> {
        unsafe {
            let audio_ptr = sherpa_rs_sys::SherpaOnnxOfflineTtsGenerate(
                self.tts,
                cstr!(text).into_raw(),
                sid,
                speed,
            );
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
                sample_rate,
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
            sample_rate: self.sample_rate as u32,
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
