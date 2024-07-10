pub mod embedding_manager;
pub mod language_id;
pub mod speaker_id;
pub mod transcribe;
pub mod vad;

#[cfg(feature = "tts")]
pub mod tts;

pub fn get_default_provider() -> String {
    if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(target_os = "macos") {
        "coreml"
    } else {
        "cpu"
    }
    .into()
}

#[macro_export]
macro_rules! cstr {
    ($s:expr) => {
        CString::new($s).expect("Failed to create CString")
    };
}

#[macro_export]
macro_rules! cstr_to_string {
    ($ptr:expr) => {
        std::ffi::CStr::from_ptr($ptr).to_string_lossy().to_string()
    };
}
