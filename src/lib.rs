pub mod audio_tag;
pub mod diarize;
pub mod embedding_manager;
pub mod keyword_spot;
pub mod language_id;
pub mod punctuate;
pub mod speaker_id;
pub mod vad;
pub mod whisper;
pub mod zipformer;

mod utils;

#[cfg(feature = "tts")]
pub mod tts;

use eyre::{bail, Result};

pub fn get_default_provider() -> String {
    if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(target_os = "macos") {
        "coreml"
    } else if cfg!(feature = "directml") {
        "directml"
    } else {
        "cpu"
    }
    .into()
}

pub fn read_audio_file(path: &str) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)?;
    let sample_rate = reader.spec().sample_rate;

    // Check if the sample rate is 16000
    if sample_rate != 16000 {
        bail!("The sample rate must be 16000.");
    }

    // Collect samples into a Vec<f32>
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    Ok((samples, sample_rate))
}
