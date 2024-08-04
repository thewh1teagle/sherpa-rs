/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.tar.bz2
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/motivation.wav -O motivation.wav
cargo run --example transcribe motivation.wav
*/

use eyre::{bail, Result};
use sherpa_rs::transcribe::whisper::WhisperRecognizer;
use std::time::Instant;

fn read_audio_file(path: &str) -> Result<(i32, Vec<f32>)> {
    let mut reader = hound::WavReader::open(path)?;
    let sample_rate = reader.spec().sample_rate as i32;

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

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("Missing file path argument");
    let provider = std::env::args().nth(2).unwrap_or("cpu".into());
    let (sample_rate, samples) = read_audio_file(&path)?;

    // Check if the sample rate is 16000
    if sample_rate != 16000 {
        bail!("The sample rate must be 16000.");
    }

    let mut recognizer = WhisperRecognizer::new(
        "sherpa-onnx-whisper-tiny/tiny-decoder.onnx".into(),
        "sherpa-onnx-whisper-tiny/tiny-encoder.onnx".into(),
        "sherpa-onnx-whisper-tiny/tiny-tokens.txt".into(),
        "en".into(),
        Some(true),
        Some(provider),
        None,
        None,
    );

    let start_t = Instant::now();
    let result = recognizer.transcribe(sample_rate, samples);
    println!("{:?}", result);
    println!("Time taken for transcription: {:?}", start_t.elapsed());
    Ok(())
}
