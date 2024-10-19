pub mod audio_tag;
pub mod diarize;
pub mod embedding_manager;
pub mod language_id;
pub mod speaker_id;
pub mod vad;
pub mod whisper;
pub mod zipformer;

#[cfg(feature = "tts")]
pub mod tts;

use eyre::{bail, Result};
use std::ffi::CString;

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

// Smart pointer for CString
struct RawCStr {
    ptr: *mut i8,
}

impl RawCStr {
    /// Creates a new `CStr` from a given Rust string slice.
    pub fn new(s: &str) -> Self {
        let cstr = CString::new(s).expect("CString::new failed");
        let ptr = cstr.into_raw();
        Self { ptr }
    }

    /// Returns the raw pointer to the internal C string.
    ///
    /// # Safety
    /// This function only returns the raw pointer and does not transfer ownership.
    /// The pointer remains valid as long as the `CStr` instance exists.
    /// Be cautious not to deallocate or modify the pointer after using `CStr::new`.
    pub fn as_ptr(&self) -> *const i8 {
        self.ptr
    }
}

impl Drop for RawCStr {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = CString::from_raw(self.ptr);
            }
        }
    }
}

fn cstr_to_string(ptr: *const i8) -> String {
    unsafe {
        if ptr.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}
