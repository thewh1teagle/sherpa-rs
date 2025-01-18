mod kokoro;
mod vits;
mod matcha;

use eyre::{ bail, Result };

pub use kokoro::{ KokoroTts, KokoroTtsConfig };
pub use vits::{ VitsTts, VitsTtsConfig };
pub use matcha::{ MatchaTts, MatchaTtsConfig };

use crate::utils::RawCStr;

#[derive(Debug)]
pub struct TtsAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub duration: i32,
}

/// # Safety
///
/// This function dereference sherpa_rs_sys::SherpaOnnxOfflineTts
pub unsafe fn create(
    tts: *const sherpa_rs_sys::SherpaOnnxOfflineTts,
    text: &str,
    sid: i32,
    speed: f32
) -> Result<TtsAudio> {
    let text = RawCStr::new(text);
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
