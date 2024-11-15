/*
Transcribe wav file using Moonshine (English only)

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
tar xvf sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
rm sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/motivation.wav -O motivation.wav
cargo run --example moonshine motivation.wav
*/

use sherpa_rs::{
    moonshine::{MoonshineConfig, MoonshineRecognizer},
    read_audio_file,
};

fn main() {
    let path = std::env::args().nth(1).expect("Missing file path argument");
    let provider = std::env::args().nth(2).unwrap_or("cpu".into());
    let (samples, sample_rate) = read_audio_file(&path).unwrap();
    assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");

    let config = MoonshineConfig {
        preprocessor: "./sherpa-onnx-moonshine-tiny-en-int8/preprocess.onnx".into(),
        encoder: "./sherpa-onnx-moonshine-tiny-en-int8/encode.int8.onnx".into(),
        uncached_decoder: "./sherpa-onnx-moonshine-tiny-en-int8/uncached_decode.int8.onnx".into(),
        cached_decoder: "./sherpa-onnx-moonshine-tiny-en-int8/cached_decode.int8.onnx".into(),
        tokens: "./sherpa-onnx-moonshine-tiny-en-int8/tokens.txt".into(),
        provider: Some(provider),
        num_threads: None,
        ..Default::default() // fill in any missing fields with defaults
    };
    let mut recognizer = MoonshineRecognizer::new(config).unwrap();

    let start_t = std::time::Instant::now();
    let result = recognizer.transcribe(sample_rate as u32, samples);
    println!("✅ Text: {}", result.text);
    println!("⏱️ Time taken for transcription: {:?}", start_t.elapsed());
}
