pub mod language_id;
pub mod speaker_id;
pub mod tts;
pub mod vad;

pub fn get_default_provider() -> String {
    if cfg!(target_os = "macos") {
        "coreml"
    } else {
        "cpu"
    }
    .into()
}
