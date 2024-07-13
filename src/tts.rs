use crate::get_default_provider;
use eyre::{bail, Result};
use hound::{WavSpec, WavWriter};
use std::ffi::CString;

#[derive(Debug)]
pub struct TtsVitsModelConfig {
    pub(crate) cfg: sherpa_rs_sys::SherpaOnnxOfflineTtsVitsModelConfig,
}

#[derive(Debug)]
pub struct OfflineTtsModelConfig {
    pub(crate) cfg: sherpa_rs_sys::SherpaOnnxOfflineTtsModelConfig,
}

#[derive(Debug)]
pub struct OfflineTtsConfig {
    pub(crate) cfg: sherpa_rs_sys::SherpaOnnxOfflineTtsConfig,
}

#[derive(Debug)]
pub struct OfflineTts {
    pub(crate) tts: *mut sherpa_rs_sys::SherpaOnnxOfflineTts,
}

impl TtsVitsModelConfig {
    pub fn new(
        model: String,
        lexicon: String,
        tokens: String,
        data_dir: String,
        noise_scale: f32,
        noise_scale_w: f32,
        dict_dir: String,
        length_scale: f32,
    ) -> Self {
        let c_model = CString::new(model).unwrap();
        let c_lexicon = CString::new(lexicon).unwrap();
        let c_tokens = CString::new(tokens).unwrap();
        let c_data_dir = CString::new(data_dir).unwrap();
        let c_dict_dir = CString::new(dict_dir).unwrap();

        let cfg = sherpa_rs_sys::SherpaOnnxOfflineTtsVitsModelConfig {
            model: c_model.into_raw(),
            lexicon: c_lexicon.into_raw(),
            tokens: c_tokens.into_raw(),
            data_dir: c_data_dir.into_raw(),
            noise_scale,
            noise_scale_w,
            dict_dir: c_dict_dir.into_raw(),
            length_scale,
        };
        Self { cfg }
    }
}

impl OfflineTtsModelConfig {
    pub fn new(
        debug: bool,
        vits_config: TtsVitsModelConfig,
        provider: Option<String>,
        num_threads: i32,
    ) -> Self {
        let debug = if debug { 1 } else { 0 };

        let provider = provider.unwrap_or(get_default_provider());
        let provider_c = CString::new(provider).unwrap();

        let cfg = sherpa_rs_sys::SherpaOnnxOfflineTtsModelConfig {
            debug,
            num_threads,
            vits: vits_config.cfg,
            provider: provider_c.into_raw(),
        };
        Self { cfg }
    }
}

impl OfflineTtsConfig {
    pub fn new(
        model: OfflineTtsModelConfig,
        max_num_sentences: i32,
        rule_fars: String,
        rule_fsts: String,
    ) -> Self {
        let rule_fars_c = CString::new(rule_fars).unwrap();
        let rule_fsts_c = CString::new(rule_fsts).unwrap();

        let cfg = sherpa_rs_sys::SherpaOnnxOfflineTtsConfig {
            max_num_sentences,
            model: model.cfg,
            rule_fars: rule_fars_c.into_raw(),
            rule_fsts: rule_fsts_c.into_raw(),
        };
        OfflineTtsConfig { cfg }
    }
}

#[derive(Debug)]
pub struct TtsSample {
    pub samples: Vec<f32>,
    pub sample_rate: i32,
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

impl OfflineTts {
    pub fn new(config: OfflineTtsConfig) -> Self {
        let tts = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineTts(&config.cfg) };
        Self { tts }
    }

    pub fn generate(&mut self, text: String, sid: i32, speed: f32) -> Result<TtsSample> {
        let text_c = CString::new(text).unwrap();
        unsafe {
            let audio_ptr = sherpa_rs_sys::SherpaOnnxOfflineTtsGenerate(
                self.tts,
                text_c.into_raw(),
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

unsafe impl Send for OfflineTts {}
unsafe impl Sync for OfflineTts {}

impl Drop for OfflineTts {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineTts(self.tts);
        }
    }
}
