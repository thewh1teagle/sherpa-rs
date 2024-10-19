/*
Transcribe wav file using OpenAI Whisper

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.tar.bz2
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/motivation.wav -O motivation.wav
cargo run --example whisper motivation.wav
*/

use sherpa_rs::{
    read_audio_file,
    whisper::{WhisperConfig, WhisperRecognizer},
};

fn main() {
    let path = std::env::args().nth(1).expect("Missing file path argument");
    let provider = std::env::args().nth(2).unwrap_or("cpu".into());
    let (samples, sample_rate) = read_audio_file(&path).unwrap();
    assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");

    let config = WhisperConfig {
        decoder: "sherpa-onnx-whisper-tiny/tiny-decoder.onnx".into(),
        encoder: "sherpa-onnx-whisper-tiny/tiny-encoder.onnx".into(),
        tokens: "sherpa-onnx-whisper-tiny/tiny-tokens.txt".into(),
        language: "en".into(),
        provider: Some(provider),
        num_threads: None,
        bpe_vocab: None,
        ..Default::default() // fill in any missing fields with defaults
    };

    let mut recognizer = WhisperRecognizer::new(config).unwrap();

    let start_t = std::time::Instant::now();
    let result = recognizer.transcribe(sample_rate as u32, samples);
    println!("✅ Text: {}", result.text);
    println!("⏱️ Time taken for transcription: {:?}", start_t.elapsed());
}
