use eyre::{bail, Result};

use crate::{
    get_default_provider,
    utils::{cstr_to_string, RawCStr},
};

#[derive(Debug, Default, Clone)]
pub struct AudioTagConfig {
    pub model: String,
    pub labels: String,
    pub top_k: i32,
    pub ced: Option<String>,
    pub debug: bool,
    pub num_threads: Option<i32>,
    pub provider: Option<String>,
}

pub struct AudioTag {
    audio_tag: *const sherpa_rs_sys::SherpaOnnxAudioTagging,
    config: AudioTagConfig,
}

impl AudioTag {
    pub fn new(config: AudioTagConfig) -> Result<Self> {
        let config_clone = config.clone();

        let model = RawCStr::new(&config.model);
        let ced = RawCStr::new(&config.ced.unwrap_or_default());
        let labels = RawCStr::new(&config.labels);
        let provider = RawCStr::new(&config.provider.unwrap_or(get_default_provider()));

        let sherpa_config = sherpa_rs_sys::SherpaOnnxAudioTaggingConfig {
            model: sherpa_rs_sys::SherpaOnnxAudioTaggingModelConfig {
                zipformer: sherpa_rs_sys::SherpaOnnxOfflineZipformerAudioTaggingModelConfig {
                    model: model.as_ptr(),
                },
                ced: ced.as_ptr(),
                num_threads: config.num_threads.unwrap_or(1),
                debug: config.debug.into(),
                provider: provider.as_ptr(),
            },
            labels: labels.as_ptr(),
            top_k: config.top_k,
        };
        let audio_tag = unsafe { sherpa_rs_sys::SherpaOnnxCreateAudioTagging(&sherpa_config) };

        if audio_tag.is_null() {
            bail!("Failed to create audio tagging")
        }
        Ok(Self {
            audio_tag,
            config: config_clone,
        })
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
                let event_name = cstr_to_string((*event).name);
                events.push(event_name);
            }
        }
        events
    }
}
