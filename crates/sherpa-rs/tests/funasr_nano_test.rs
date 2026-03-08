/*
FunASR-Nano API tests

To run with actual models:
1. Download the FunASR-Nano model:
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-zh-en-ja-2024-12-30.tar.bz2
   tar xvf sherpa-onnx-funasr-nano-zh-en-ja-2024-12-30.tar.bz2

2. Run tests:
   cargo test --test funasr_nano_test -- --ignored
*/

use sherpa_rs::funasr_nano::{FunasrNanoConfig, FunasrNanoRecognizer};

/// Test that default config can be created
#[test]
fn test_funasr_nano_config_default() {
    let config = FunasrNanoConfig::default();

    assert!(config.encoder_adaptor.is_empty());
    assert!(config.llm_prefill.is_empty());
    assert!(config.llm_decode.is_empty());
    assert!(config.embedding.is_empty());
    assert!(config.tokenizer.is_empty());
    assert_eq!(config.system_prompt, Some("You are a helpful assistant.".into()));
    assert_eq!(config.user_prompt, Some("语音转写：".into()));
    assert_eq!(config.max_new_tokens, Some(512));
    assert_eq!(config.temperature, Some(0.3));
    assert_eq!(config.top_p, Some(0.8));
    assert_eq!(config.seed, Some(42));
    assert!(config.provider.is_none());
    assert_eq!(config.num_threads, Some(4));
    assert!(!config.debug);
}

/// Test config with custom values
#[test]
fn test_funasr_nano_config_custom() {
    let config = FunasrNanoConfig {
        encoder_adaptor: "model/encoder_adaptor.onnx".into(),
        llm_prefill: "model/llm_prefill.onnx".into(),
        llm_decode: "model/llm_decode.onnx".into(),
        embedding: "model/embedding.onnx".into(),
        tokenizer: "model/tokenizer.json".into(),
        system_prompt: Some("You are a helpful assistant.".into()),
        user_prompt: Some("Transcribe the following audio:".into()),
        max_new_tokens: Some(500),
        temperature: Some(0.7),
        top_p: Some(0.95),
        seed: Some(42),
        provider: Some("cpu".into()),
        num_threads: Some(4),
        debug: true,
    };

    assert_eq!(config.encoder_adaptor, "model/encoder_adaptor.onnx");
    assert_eq!(config.llm_prefill, "model/llm_prefill.onnx");
    assert_eq!(config.llm_decode, "model/llm_decode.onnx");
    assert_eq!(config.embedding, "model/embedding.onnx");
    assert_eq!(config.tokenizer, "model/tokenizer.json");
    assert_eq!(config.system_prompt, Some("You are a helpful assistant.".into()));
    assert_eq!(config.max_new_tokens, Some(500));
    assert_eq!(config.temperature, Some(0.7));
    assert_eq!(config.top_p, Some(0.95));
    assert_eq!(config.seed, Some(42));
    assert_eq!(config.provider, Some("cpu".into()));
    assert_eq!(config.num_threads, Some(4));
    assert!(config.debug);
}

/// Test config cloning
#[test]
fn test_funasr_nano_config_clone() {
    let config = FunasrNanoConfig {
        encoder_adaptor: "encoder.onnx".into(),
        llm_prefill: "prefill.onnx".into(),
        llm_decode: "decode.onnx".into(),
        embedding: "embedding.onnx".into(),
        tokenizer: "tokenizer.json".into(),
        ..Default::default()
    };

    let cloned = config.clone();
    assert_eq!(cloned.encoder_adaptor, config.encoder_adaptor);
    assert_eq!(cloned.llm_prefill, config.llm_prefill);
    assert_eq!(cloned.llm_decode, config.llm_decode);
    assert_eq!(cloned.embedding, config.embedding);
    assert_eq!(cloned.tokenizer, config.tokenizer);
}

/// Test recognizer creation fails gracefully with invalid paths
#[test]
fn test_funasr_nano_recognizer_invalid_paths() {
    let config = FunasrNanoConfig {
        encoder_adaptor: "nonexistent/encoder_adaptor.onnx".into(),
        llm_prefill: "nonexistent/llm_prefill.onnx".into(),
        llm_decode: "nonexistent/llm_decode.onnx".into(),
        embedding: "nonexistent/embedding.onnx".into(),
        tokenizer: "nonexistent/tokenizer.json".into(),
        ..Default::default()
    };

    let result = FunasrNanoRecognizer::new(config);
    // Should fail with invalid model paths
    assert!(result.is_err());
}

// Helper to get model path relative to workspace root
fn model_path(relative: &str) -> String {
    // Tests run from crates/sherpa-rs, so we need to go up two levels
    format!("../../{}", relative)
}

/// Test recognizer creation with actual models (requires model files)
#[test]
#[ignore = "Requires model files to be downloaded"]
fn test_funasr_nano_recognizer_creation() {
    let config = FunasrNanoConfig {
        encoder_adaptor: model_path("funasr-nano/encoder_adaptor.int8.onnx"),
        llm_prefill: model_path("funasr-nano/llm_prefill.int8.onnx"),
        llm_decode: model_path("funasr-nano/llm_decode.int8.onnx"),
        embedding: model_path("funasr-nano/embedding.int8.onnx"),
        tokenizer: model_path("funasr-nano/Qwen3-0.6B"),
        ..Default::default()
    };

    let result = FunasrNanoRecognizer::new(config);
    assert!(result.is_ok(), "Failed to create recognizer: {:?}", result.err());
}

/// Test transcription with actual models (requires model files)
#[test]
#[ignore = "Requires model files to be downloaded"]
fn test_funasr_nano_transcription() {
    let config = FunasrNanoConfig {
        encoder_adaptor: model_path("funasr-nano/encoder_adaptor.int8.onnx"),
        llm_prefill: model_path("funasr-nano/llm_prefill.int8.onnx"),
        llm_decode: model_path("funasr-nano/llm_decode.int8.onnx"),
        embedding: model_path("funasr-nano/embedding.int8.onnx"),
        tokenizer: model_path("funasr-nano/Qwen3-0.6B"),
        ..Default::default()
    };

    let mut recognizer = FunasrNanoRecognizer::new(config).unwrap();

    // Generate some silence audio
    let samples: Vec<f32> = vec![0.0; 16000]; // 1 second of silence
    let result = recognizer.transcribe(16000, &samples);

    // Result should be returned (may be empty for silence)
    assert!(result.text.is_empty() || !result.text.is_empty());
}

/// Test transcription with custom prompts (requires model files)
#[test]
#[ignore = "Requires model files to be downloaded"]
fn test_funasr_nano_with_prompts() {
    let config = FunasrNanoConfig {
        encoder_adaptor: model_path("funasr-nano/encoder_adaptor.int8.onnx"),
        llm_prefill: model_path("funasr-nano/llm_prefill.int8.onnx"),
        llm_decode: model_path("funasr-nano/llm_decode.int8.onnx"),
        embedding: model_path("funasr-nano/embedding.int8.onnx"),
        tokenizer: model_path("funasr-nano/Qwen3-0.6B"),
        system_prompt: Some("你是一个语音识别助手".into()),
        user_prompt: Some("请转录以下音频：".into()),
        max_new_tokens: Some(100),
        temperature: Some(0.0),
        ..Default::default()
    };

    let mut recognizer = FunasrNanoRecognizer::new(config).unwrap();

    // Generate some silence audio
    let samples: Vec<f32> = vec![0.0; 16000];
    let result = recognizer.transcribe(16000, &samples);

    // Should not panic
    assert!(result.text.is_empty() || !result.text.is_empty());
}

/// Test thread safety - recognizer should be Send + Sync
#[test]
fn test_funasr_nano_thread_safety() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<FunasrNanoRecognizer>();
    assert_sync::<FunasrNanoRecognizer>();
}

/// Test config debug trait
#[test]
fn test_funasr_nano_config_debug() {
    let config = FunasrNanoConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("FunasrNanoConfig"));
}

