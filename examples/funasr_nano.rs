/*
Transcribe wav file using FunASR-Nano

Download model files from:
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models

Example:
cargo run --example funasr_nano -- audio.wav
*/

use sherpa_rs::{
    funasr_nano::{FunasrNanoConfig, FunasrNanoRecognizer},
    read_audio_file,
};

fn main() {
    let path = std::env::args().nth(1).expect("Missing file path argument");
    let provider = std::env::args().nth(2).unwrap_or("cpu".into());
    let (samples, sample_rate) = read_audio_file(&path).unwrap();
    assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");

    let config = FunasrNanoConfig {
        encoder_adaptor: "funasr-nano/encoder_adaptor.int8.onnx".into(),
        llm_prefill: "funasr-nano/llm_prefill.int8.onnx".into(),
        llm_decode: "funasr-nano/llm_decode.int8.onnx".into(),
        embedding: "funasr-nano/embedding.int8.onnx".into(),
        // tokenizer 是目录路径，需要包含 tokenizer.json, vocab.json, merges.txt
        tokenizer: "funasr-nano/Qwen3-0.6B".into(),
        provider: Some(provider),
        ..Default::default()
    };

    let mut recognizer = FunasrNanoRecognizer::new(config).unwrap();

    let start_t = std::time::Instant::now();
    let result = recognizer.transcribe(sample_rate, &samples);
    println!("Text: {}", result.text);
    println!("Time taken for transcription: {:?}", start_t.elapsed());
}

