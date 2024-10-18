use eyre::{bail, Result};

use crate::{cstr, cstr_to_string, get_default_provider};

#[derive(Debug, Default, Clone)]
pub struct AudioTagConfig {
    pub model: String,
    pub labels: String,
    pub top_k: i32,
    pub debug: Option<bool>,
    pub num_threads: Option<i32>,
    pub provider: Option<String>,
    pub ced: Option<String>,
}

pub struct AudioTag {
    audio_tag: *const sherpa_rs_sys::SherpaOnnxAudioTagging,
    config: AudioTagConfig,
}

impl AudioTag {
    pub fn new(config: AudioTagConfig) -> Result<Self> {
        let config_clone = config.clone();
        let sherpa_config = sherpa_rs_sys::SherpaOnnxAudioTaggingConfig {
            model: sherpa_rs_sys::SherpaOnnxAudioTaggingModelConfig {
                zipformer: sherpa_rs_sys::SherpaOnnxOfflineZipformerAudioTaggingModelConfig {
                    model: cstr!(config.model).into_raw(),
                },
                ced: cstr!(config.ced.unwrap_or_default()).into_raw(),
                num_threads: config.num_threads.unwrap_or(1),
                debug: config.debug.unwrap_or_default().into(),
                provider: cstr!(config.provider.unwrap_or(get_default_provider())).into_raw(),
            },
            labels: cstr!(config.labels).into_raw(),
            top_k: config.top_k,
        };
        unsafe {
            let audio_tag = sherpa_rs_sys::SherpaOnnxCreateAudioTagging(&sherpa_config);
            if audio_tag.is_null() {
                bail!("Failed to create audio tagging")
            }
            Ok(Self {
                audio_tag,
                config: config_clone,
            })
        }
    }

    pub fn compute(&mut self, samples: Vec<f32>, sample_rate: u32) -> Vec<String> {
        let mut events = Vec::new();
        unsafe {
            let stream = sherpa_rs_sys::SherpaOnnxAudioTaggingCreateOfflineStream(self.audio_tag);
            sherpa_rs_sys::SherpaOnnxAcceptWaveformOffline(
                stream,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len() as i32,
            );
            let results = sherpa_rs_sys::SherpaOnnxAudioTaggingCompute(
                self.audio_tag,
                stream,
                self.config.top_k,
            );

            for i in 0..self.config.top_k {
                let event = *results.add(i.try_into().unwrap());
                let event_name = cstr_to_string!((*event).name);
                events.push(event_name);
            }
        }
        events
    }
}
