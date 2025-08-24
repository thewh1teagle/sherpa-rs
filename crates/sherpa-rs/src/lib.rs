pub mod audio_tag;
pub mod diarize;
pub mod dolphin;
pub mod embedding_manager;
pub mod keyword_spot;
pub mod language_id;
pub mod moonshine;
pub mod paraformer;
pub mod punctuate;
pub mod sense_voice;
pub mod silero_vad;
pub mod speaker_id;
pub mod ten_vad;
pub mod transducer;
pub mod whisper;
pub mod zipformer;

mod utils;

#[cfg(feature = "tts")]
pub mod tts;

use std::ffi::CStr;

#[cfg(feature = "sys")]
pub use sherpa_rs_sys;

use eyre::{bail, Result};
use utils::cstr_to_string;

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
        let scaled_sample =
            (sample * (i16::MAX as f32)).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
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

#[derive(Debug, Clone)]
pub struct OfflineRecognizerResult {
    pub lang: String,
    pub text: String,
    pub timestamps: Vec<f32>,
    pub tokens: Vec<String>,
}

impl OfflineRecognizerResult {
    fn new(result: &sherpa_rs_sys::SherpaOnnxOfflineRecognizerResult) -> Self {
        let lang = unsafe { cstr_to_string(result.lang) };
        let text = unsafe { cstr_to_string(result.text) };
        let count = result.count.try_into().unwrap();
        let timestamps = if result.timestamps.is_null() {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(result.timestamps, count).to_vec() }
        };
        let mut tokens = Vec::with_capacity(count);
        let mut next_token = result.tokens;

        for _ in 0..count {
            let token = unsafe { CStr::from_ptr(next_token) };
            tokens.push(token.to_string_lossy().into_owned());
            next_token = next_token
                .wrapping_byte_offset(token.to_bytes_with_nul().len().try_into().unwrap());
        }

        Self {
            lang,
            text,
            timestamps,
            tokens,
        }
    }
}

impl Default for OnnxConfig {
    fn default() -> Self {
        Self {
            provider: get_default_provider(),
            debug: false,
            num_threads: 1,
        }
    }
}
