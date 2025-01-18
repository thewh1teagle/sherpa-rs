pub mod audio_tag;
pub mod diarize;
pub mod embedding_manager;
pub mod keyword_spot;
pub mod language_id;
pub mod moonshine;
pub mod punctuate;
pub mod speaker_id;
pub mod vad;
pub mod whisper;
pub mod zipformer;

mod utils;

#[cfg(feature = "tts")]
pub mod tts;

#[cfg(feature = "sys")]
pub use sherpa_rs_sys;

use eyre::{ bail, Result };

pub fn get_default_provider() -> String {
    "cpu".into()
    // Other providers has many issues with different models!!
    // if cfg!(feature = "cuda") {
    //     "cuda"
    // } else if cfg!(target_os = "macos") {
    //     "coreml"
    // } else if cfg!(feature = "directml") {
    //     "directml"
    // } else {
    //     "cpu"
    // }
    // .into()
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
        .map(|s| (s.unwrap() as f32) / (i16::MAX as f32))
        .collect();

    Ok((samples, sample_rate))
}

pub fn write_audio_file(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    // Create a WAV file writer
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;

    // Convert samples from f32 to i16 and write them to the WAV file
    for &sample in samples {
        let scaled_sample = (sample * (i16::MAX as f32)).clamp(
            i16::MIN as f32,
            i16::MAX as f32
        ) as i16;
        writer.write_sample(scaled_sample)?;
    }

    writer.finalize()?;
    Ok(())
}

pub struct OnnxConfig {
    pub provider: String,
    pub debug: bool,
    pub num_threads: i32,
}

impl Default for OnnxConfig {
    fn default() -> Self {
        Self { provider: get_default_provider(), debug: false, num_threads: 1 }
    }
}
