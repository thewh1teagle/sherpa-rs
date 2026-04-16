//! Integration tests for offline recognizers.
//!
//! Tests skip gracefully when model files are not present.
//!
//! Quick run (models already downloaded):
//! ```sh
//! cargo test -p sherpa-rs --test offline_recognizers
//! ```
//!
//! Full run (auto-downloads models ~3GB):
//! ```sh
//! SHERPA_DOWNLOAD_MODELS=1 cargo test -p sherpa-rs --test offline_recognizers -- --nocapture
//! ```
//!
//! Override model directory:
//! ```sh
//! SHERPA_TEST_MODELS=/path/to/models cargo test -p sherpa-rs --test offline_recognizers
//! ```

mod test_utils;

// ─── Whisper ──────────────────────────────────────

mod whisper {
    use super::test_utils::*;
    use super::*;
    use sherpa_rs::whisper::{WhisperConfig, WhisperRecognizer};

    fn config() -> Option<(WhisperConfig, std::path::PathBuf)> {
        let d = ensure_model(&WHISPER_TINY)?;
        Some((
            WhisperConfig {
                encoder: d.join("tiny-encoder.onnx").to_string_lossy().into(),
                decoder: d.join("tiny-decoder.onnx").to_string_lossy().into(),
                tokens: d.join("tiny-tokens.txt").to_string_lossy().into(),
                language: "en".into(),
                num_threads: Some(4),
                ..Default::default()
            },
            d,
        ))
    }

    #[test]
    fn creates_recognizer() {
        let Some((cfg, _)) = config() else { return };
        assert!(WhisperRecognizer::new(cfg).is_ok());
    }

    #[test]
    fn transcribes_motivation_wav() {
        let Some((cfg, _)) = config() else { return };
        let Some(wav) = ensure_motivation_wav() else { return };

        let mut rec = WhisperRecognizer::new(cfg).unwrap();
        let (samples, sr) = sherpa_rs::read_audio_file(wav.to_str().unwrap()).unwrap();
        let dur_s = samples.len() as f64 / sr as f64;

        // 5 iterations, same recognizer — cold vs warm.
        let mut times_ms = Vec::new();
        let mut last_text = String::new();
        for i in 0..5 {
            let start = std::time::Instant::now();
            let result = rec.transcribe(sr, &samples);
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            times_ms.push(elapsed_ms);
            last_text = result.text.clone();
            println!(
                "[whisper] run {} — {:.0}ms / {:.1}s (RTF {:.3})",
                i + 1,
                elapsed_ms,
                dur_s,
                elapsed_ms / 1000.0 / dur_s
            );
            assert!(!result.text.is_empty());
        }

        let lower = last_text.to_lowercase();
        assert!(
            lower.contains("people") || lower.contains("person"),
            "expected recognizable English, got: {:?}",
            last_text
        );

        let warm: Vec<f64> = times_ms.iter().skip(1).copied().collect();
        let warm_avg = warm.iter().sum::<f64>() / warm.len() as f64;
        let warm_rtf = warm_avg / 1000.0 / dur_s;
        println!(
            "[whisper] cold={:.0}ms, warm avg (runs 2-5)={:.0}ms, warm RTF={:.3}",
            times_ms[0], warm_avg, warm_rtf
        );
        assert!(warm_rtf < 1.0, "warm RTF {} should be < 1.0", warm_rtf);
    }

    #[test]
    fn rejects_missing_model() {
        let cfg = WhisperConfig {
            encoder: "/nonexistent/encoder.onnx".into(),
            decoder: "/nonexistent/decoder.onnx".into(),
            tokens: "/nonexistent/tokens.txt".into(),
            ..Default::default()
        };
        assert!(WhisperRecognizer::new(cfg).is_err());
    }
}

// ─── Cohere Transcribe ────────────────────────────

mod cohere {
    use super::test_utils::*;
    use super::*;
    use sherpa_rs::cohere_transcribe::{CohereTranscribeConfig, CohereTranscribeRecognizer};

    /// Create a recognizer with CWD set to the model dir (required for
    /// external .onnx.data files).
    fn make_recognizer() -> Option<(CohereTranscribeRecognizer, std::path::PathBuf)> {
        let d = ensure_model(&COHERE_TRANSCRIBE_INT8)?;
        let prev = std::env::current_dir().ok();
        let _ = std::env::set_current_dir(&d);
        let cfg = CohereTranscribeConfig {
            encoder: d.join("encoder.int8.onnx").to_string_lossy().into(),
            decoder: d.join("decoder.int8.onnx").to_string_lossy().into(),
            tokens: d.join("tokens.txt").to_string_lossy().into(),
            language: "en".into(),
            num_threads: Some(4),
            use_punct: true,
            use_itn: true,
            ..Default::default()
        };
        let rec = CohereTranscribeRecognizer::new(cfg);
        if let Some(p) = prev {
            let _ = std::env::set_current_dir(p);
        }
        Some((rec.ok()?, d))
    }

    #[test]
    fn creates_recognizer() {
        if ensure_model(&COHERE_TRANSCRIBE_INT8).is_none() {
            return;
        }
        assert!(make_recognizer().is_some());
    }

    #[test]
    fn transcribes_english() {
        let Some((mut rec, _)) = make_recognizer() else { return };
        let Some(wav) = ensure_motivation_wav() else { return };

        let (samples, sr) = sherpa_rs::read_audio_file(wav.to_str().unwrap()).unwrap();
        let dur_s = samples.len() as f64 / sr as f64;

        // Run 5 iterations with the same recognizer to measure warm perf.
        // First run includes session init overhead; subsequent runs are steady-state.
        let mut times_ms = Vec::new();
        let mut last_text = String::new();
        for i in 0..5 {
            let start = std::time::Instant::now();
            let result = rec.transcribe(sr, &samples);
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            times_ms.push(elapsed_ms);
            last_text = result.text.clone();
            println!(
                "[cohere/en] run {} — {:.0}ms / {:.1}s (RTF {:.3})",
                i + 1,
                elapsed_ms,
                dur_s,
                elapsed_ms / 1000.0 / dur_s
            );
            assert!(!result.text.is_empty());
        }

        let lower = last_text.to_lowercase();
        assert!(
            lower.contains("people") || lower.contains("person"),
            "expected recognizable English, got: {:?}",
            last_text
        );

        // Warm-run statistics (exclude the first cold run).
        let warm: Vec<f64> = times_ms.iter().skip(1).copied().collect();
        let warm_avg = warm.iter().sum::<f64>() / warm.len() as f64;
        let warm_rtf = warm_avg / 1000.0 / dur_s;
        println!(
            "[cohere/en] cold={:.0}ms, warm avg (runs 2-5)={:.0}ms, warm RTF={:.3}",
            times_ms[0], warm_avg, warm_rtf
        );
        assert!(warm_rtf < 1.0, "warm RTF {} should be < 1.0", warm_rtf);
    }

    #[test]
    fn rejects_missing_model() {
        let cfg = CohereTranscribeConfig {
            encoder: "/nonexistent/encoder.onnx".into(),
            decoder: "/nonexistent/decoder.onnx".into(),
            tokens: "/nonexistent/tokens.txt".into(),
            ..Default::default()
        };
        assert!(CohereTranscribeRecognizer::new(cfg).is_err());
    }
}
