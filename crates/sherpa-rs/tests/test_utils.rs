//! Shared test utilities: model directory resolution and auto-download.
//!
//! Set `SHERPA_TEST_MODELS` to override the default location.
//! Set `SHERPA_DOWNLOAD_MODELS=1` to auto-download missing models.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Once;

pub fn test_data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("SHERPA_TEST_MODELS") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_data")
}

pub fn should_download() -> bool {
    std::env::var("SHERPA_DOWNLOAD_MODELS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Model archive definitions.
pub struct ModelArchive {
    pub name: &'static str,
    pub url: &'static str,
    /// A file that must exist after extraction to consider the model ready.
    pub check_file: &'static str,
}

pub const WHISPER_TINY: ModelArchive = ModelArchive {
    name: "sherpa-onnx-whisper-tiny",
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2",
    check_file: "tiny-encoder.onnx",
};

pub const COHERE_TRANSCRIBE_INT8: ModelArchive = ModelArchive {
    name: "sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01",
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2",
    check_file: "encoder.int8.onnx",
};

pub const MOTIVATION_WAV_URL: &str =
    "https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/motivation.wav";

static WHISPER_ONCE: Once = Once::new();
static COHERE_ONCE: Once = Once::new();
static WAV_ONCE: Once = Once::new();

fn download_once(archive: &ModelArchive) -> &'static Once {
    match archive.name {
        "sherpa-onnx-whisper-tiny" => &WHISPER_ONCE,
        _ => &COHERE_ONCE,
    }
}

/// Ensure a model archive is present. Downloads if `SHERPA_DOWNLOAD_MODELS=1`.
/// Returns the model directory path, or None if unavailable.
pub fn ensure_model(archive: &ModelArchive) -> Option<PathBuf> {
    let dir = test_data_dir();
    let model_dir = dir.join(archive.name);
    let check = model_dir.join(archive.check_file);

    if check.exists() {
        return Some(model_dir);
    }

    if !should_download() {
        eprintln!(
            "SKIP: {} not found at {}. Set SHERPA_DOWNLOAD_MODELS=1 to auto-download.",
            archive.name,
            model_dir.display()
        );
        return None;
    }

    // Serialize concurrent downloads per model.
    download_once(archive).call_once(|| {
        eprintln!("Downloading {} ...", archive.name);
        std::fs::create_dir_all(&dir).ok();
        let status = Command::new("sh")
            .arg("-c")
            .arg(format!(
                "curl -sL '{}' | tar xjf - -C '{}'",
                archive.url,
                dir.display()
            ))
            .status();
        match status {
            Ok(s) if s.success() => {
                eprintln!("Downloaded {} to {}", archive.name, model_dir.display());
            }
            _ => {
                eprintln!("Failed to download {}", archive.name);
            }
        }
    });

    if check.exists() {
        Some(model_dir)
    } else {
        None
    }
}

/// Ensure motivation.wav test audio is present.
pub fn ensure_motivation_wav() -> Option<PathBuf> {
    let path = test_data_dir().join("motivation.wav");
    if path.exists() {
        return Some(path);
    }

    if !should_download() {
        eprintln!(
            "SKIP: motivation.wav not found. Set SHERPA_DOWNLOAD_MODELS=1 to auto-download."
        );
        return None;
    }

    WAV_ONCE.call_once(|| {
        eprintln!("Downloading motivation.wav ...");
        std::fs::create_dir_all(test_data_dir()).ok();
        let _ = Command::new("curl")
            .args(["-sLo", path.to_str().unwrap(), MOTIVATION_WAV_URL])
            .status();
    });

    if path.exists() {
        Some(path)
    } else {
        None
    }
}