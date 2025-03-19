/*
Transcribe wav file using SenseVoice

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2
tar xvf sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/motivation.wav -O motivation.wav
cargo run --example paraformer motivation.wav
*/

use sherpa_rs::{
    paraformer::{ParaformerConfig, ParaformerRecognizer},
    read_audio_file,
};

fn main() {
    let path = std::env::args().nth(1).expect("Missing file path argument");
    let provider = std::env::args().nth(2).unwrap_or("cpu".into());
    let (samples, sample_rate) = read_audio_file(&path).unwrap();
    assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");

    let config = ParaformerConfig {
        model: "sherpa-onnx-paraformer-zh-2024-03-09/model.int8.onnx".into(),
        tokens: "sherpa-onnx-paraformer-zh-2024-03-09/tokens.txt".into(),
        provider: Some(provider),

        ..Default::default()
    };

    let mut recognizer: ParaformerRecognizer = ParaformerRecognizer::new(config).unwrap();

    let start_t = std::time::Instant::now();
    let result = recognizer.transcribe(sample_rate, &samples);
    println!("✅ Text: {}", result.text);
    println!("⏱️ Time taken for transcription: {:?}", start_t.elapsed());
}
