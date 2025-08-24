mod kitten;
mod kokoro;
mod matcha;
mod vits;

use std::ffi::CString;

use eyre::{bail, Result};

pub use kitten::{KittenTts, KittenTtsConfig};
pub use kokoro::{KokoroTts, KokoroTtsConfig};
pub use matcha::{MatchaTts, MatchaTtsConfig};
pub use vits::{VitsTts, VitsTtsConfig};

use crate::utils::cstring_from_str;

#[derive(Debug)]
pub struct TtsAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub duration: i32,
}

#[derive(Default)]
pub struct CommonTtsConfig {
    pub rule_fars: String,
    pub rule_fsts: String,
    pub max_num_sentences: i32,
    pub silence_scale: f32,
}

pub struct CommonTtsRaw {
    pub rule_fars: Option<CString>,
    pub rule_fsts: Option<CString>,
    pub max_num_sentences: i32,
}

impl CommonTtsConfig {
    pub fn to_raw(&self) -> CommonTtsRaw {
        let rule_fars = if self.rule_fars.is_empty() {
            None
        } else {
            Some(cstring_from_str(&self.rule_fars))
        };

        let rule_fsts = if self.rule_fsts.is_empty() {
            None
        } else {
            Some(cstring_from_str(&self.rule_fsts))
        };

        CommonTtsRaw {
            rule_fars,
            rule_fsts,
            max_num_sentences: self.max_num_sentences,
        }
    }
}

/// # Safety
///
/// This function dereference sherpa_rs_sys::SherpaOnnxOfflineTts
pub unsafe fn create(
    tts: *const sherpa_rs_sys::SherpaOnnxOfflineTts,
    text: &str,
    sid: i32,
    speed: f32,
) -> Result<TtsAudio> {
    let text = cstring_from_str(text);
    let audio_ptr = sherpa_rs_sys::SherpaOnnxOfflineTtsGenerate(tts, text.as_ptr(), sid, speed);

    if audio_ptr.is_null() {
        bail!("audio is null");
    }
    let audio = audio_ptr.read();

    if audio.n.is_negative() {
        bail!("no samples found");
    }
    if audio.samples.is_null() {
        bail!("audio samples are null");
    }
    let samples: &[f32] = std::slice::from_raw_parts(audio.samples, audio.n as usize);
    let samples = samples.to_vec();
    let sample_rate = audio.sample_rate;
    let duration = (samples.len() as i32) / sample_rate;

    // Free
    sherpa_rs_sys::SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio_ptr);

    Ok(TtsAudio {
        samples,
        sample_rate: sample_rate as u32,
        duration,
    })
}
