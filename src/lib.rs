pub mod audio_tag;
pub mod diarize;
pub mod embedding_manager;
pub mod language_id;
pub mod speaker_id;
pub mod vad;
pub mod whisper;
pub mod zipformer;

use eyre::{bail, Result};

#[cfg(feature = "tts")]
pub mod tts;

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

pub fn read_audio_file(path: &str) -> Result<(u32, Vec<f32>)> {
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

    Ok((sample_rate, samples))
}

#[macro_export]
macro_rules! cstr {
    ($s:expr) => {
        std::ffi::CString::new($s)
            .expect("Failed to create CString")
            .into_raw()
    };
}

#[macro_export]
macro_rules! free_cstr {
    ($ptr:expr) => {
        if !$ptr.is_null() {
            // Reclaim the CString to ensure it gets dropped and its memory is freed
            let _ = std::ffi::CString::from_raw($ptr as *mut i8);
        }
    };
}

#[macro_export]
macro_rules! cstr_to_string {
    ($ptr:expr) => {
        std::ffi::CStr::from_ptr($ptr).to_string_lossy().to_string()
    };
}
