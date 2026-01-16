/*
Transcribe wav file using FunASR Nano model with LLM support

Download model:
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
tar xvf sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
rm sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2

Example usage:
ENCODER_ADAPTOR=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
LLM=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
EMBEDDING=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
TOKENIZER=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
TOKENS=./sherpa-onnx-funasr-nano-int8-2025-12-30/tokens.txt \
cargo run --example funasr_nano -- audio.wav
*/

use sherpa_rs::{
    funasr_nano::{FunAsrNanoConfig, FunAsrNanoRecognizer},
    read_audio_file,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        eprintln!("Environment variables:");
        eprintln!("  ENCODER_ADAPTOR - path to encoder_adaptor.int8.onnx");
        eprintln!("  LLM - path to llm.int8.onnx");
        eprintln!("  EMBEDDING - path to embedding.int8.onnx");
        eprintln!("  TOKENIZER - path to tokenizer directory");
        eprintln!("  TOKENS - path to tokens.txt");
        std::process::exit(1);
    }

    let path = &args[1];
    let (samples, sample_rate) = read_audio_file(path).unwrap();
    assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");

    // Get paths from environment or use defaults
    let encoder_adaptor = std::env::var("ENCODER_ADAPTOR")
        .unwrap_or_else(|_| "./encoder_adaptor.int8.onnx".into());
    let llm = std::env::var("LLM")
        .unwrap_or_else(|_| "./llm.int8.onnx".into());
    let embedding = std::env::var("EMBEDDING")
        .unwrap_or_else(|_| "./embedding.int8.onnx".into());
    let tokenizer = std::env::var("TOKENIZER")
        .unwrap_or_else(|_| "./tokenizer".into());
    let tokens = std::env::var("TOKENS")
        .unwrap_or_else(|_| "./tokens.txt".into());

    let config = FunAsrNanoConfig {
        encoder_adaptor,
        llm,
        embedding,
        tokenizer,
        tokens,
        max_new_tokens: 200,
        temperature: 0.0,
        top_p: 1.0,
        seed: 0,
        ..Default::default()
    };

    let mut recognizer = FunAsrNanoRecognizer::new(config).unwrap();

    let start_t = std::time::Instant::now();
    let result = recognizer.transcribe(sample_rate, &samples);
    println!("Text: {}", result.text);
    println!("Time taken: {:?}", start_t.elapsed());
}
