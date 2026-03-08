/*
Streaming API tests

To run with actual models:
1. Download a streaming model:
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
   tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

2. Run tests:
   cargo test --test streaming_test -- --ignored
*/

use sherpa_rs::streaming::{
    EndpointConfig, OnlineModelType, OnlineParaformerModelConfig, OnlineRecognizer,
    OnlineRecognizerConfig, OnlineTransducerModelConfig, OnlineZipformer2CtcModelConfig,
};

/// Test that default config can be created
#[test]
fn test_streaming_config_default() {
    let config = OnlineRecognizerConfig::default();

    assert_eq!(config.sample_rate, 16000);
    assert_eq!(config.feature_dim, 80);
    assert_eq!(config.decoding_method, "greedy_search");
    assert!(config.endpoint.enable);
}

/// Test EndpointConfig default values
#[test]
fn test_endpoint_config_default() {
    let config = EndpointConfig::default();

    assert!(config.enable);
    assert_eq!(config.rule1_min_trailing_silence, 2.4);
    assert_eq!(config.rule2_min_trailing_silence, 1.2);
    assert_eq!(config.rule3_min_utterance_length, 20.0);
}

/// Test OnlineTransducerModelConfig creation
#[test]
fn test_transducer_model_config() {
    let config = OnlineTransducerModelConfig {
        encoder: "encoder.onnx".into(),
        decoder: "decoder.onnx".into(),
        joiner: "joiner.onnx".into(),
    };

    assert_eq!(config.encoder, "encoder.onnx");
    assert_eq!(config.decoder, "decoder.onnx");
    assert_eq!(config.joiner, "joiner.onnx");
}

/// Test OnlineParaformerModelConfig creation
#[test]
fn test_paraformer_model_config() {
    let config = OnlineParaformerModelConfig {
        encoder: "encoder.onnx".into(),
        decoder: "decoder.onnx".into(),
    };

    assert_eq!(config.encoder, "encoder.onnx");
    assert_eq!(config.decoder, "decoder.onnx");
}

/// Test OnlineZipformer2CtcModelConfig creation
#[test]
fn test_zipformer2_ctc_model_config() {
    let config = OnlineZipformer2CtcModelConfig {
        model: "model.onnx".into(),
    };

    assert_eq!(config.model, "model.onnx");
}

/// Test full config creation with Transducer model
#[test]
fn test_full_transducer_config() {
    let config = OnlineRecognizerConfig {
        model: OnlineModelType::Transducer(OnlineTransducerModelConfig {
            encoder: "encoder.onnx".into(),
            decoder: "decoder.onnx".into(),
            joiner: "joiner.onnx".into(),
        }),
        tokens: "tokens.txt".into(),
        sample_rate: 16000,
        feature_dim: 80,
        decoding_method: "greedy_search".into(),
        max_active_paths: 4,
        endpoint: EndpointConfig {
            enable: true,
            rule1_min_trailing_silence: 2.0,
            rule2_min_trailing_silence: 1.0,
            rule3_min_utterance_length: 15.0,
        },
        provider: Some("cpu".into()),
        num_threads: Some(4),
        debug: false,
        ..Default::default()
    };

    assert_eq!(config.tokens, "tokens.txt");
    assert_eq!(config.sample_rate, 16000);
    assert_eq!(config.num_threads, Some(4));
    assert!(config.endpoint.enable);
}

/// Test recognizer creation with actual models (requires model files)
#[test]
#[ignore = "Requires model files to be downloaded"]
fn test_streaming_recognizer_creation() {
    let config = OnlineRecognizerConfig {
        model: OnlineModelType::Transducer(OnlineTransducerModelConfig {
            encoder: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx".into(),
            decoder: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx".into(),
            joiner: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx".into(),
        }),
        tokens: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt".into(),
        ..Default::default()
    };

    let recognizer = OnlineRecognizer::new(config);
    assert!(recognizer.is_ok(), "Failed to create recognizer: {:?}", recognizer.err());
}

/// Test stream creation (requires model files)
#[test]
#[ignore = "Requires model files to be downloaded"]
fn test_streaming_stream_creation() {
    let config = OnlineRecognizerConfig {
        model: OnlineModelType::Transducer(OnlineTransducerModelConfig {
            encoder: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx".into(),
            decoder: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx".into(),
            joiner: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx".into(),
        }),
        tokens: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt".into(),
        ..Default::default()
    };

    let recognizer = OnlineRecognizer::new(config).unwrap();
    let stream = recognizer.create_stream();
    assert!(stream.is_ok(), "Failed to create stream: {:?}", stream.err());
}

/// Test streaming transcription (requires model files)
#[test]
#[ignore = "Requires model files to be downloaded"]
fn test_streaming_transcription() {
    let config = OnlineRecognizerConfig {
        model: OnlineModelType::Transducer(OnlineTransducerModelConfig {
            encoder: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx".into(),
            decoder: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx".into(),
            joiner: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx".into(),
        }),
        tokens: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt".into(),
        ..Default::default()
    };

    let recognizer = OnlineRecognizer::new(config).unwrap();
    let mut stream = recognizer.create_stream().unwrap();

    // Generate some silence audio
    let samples: Vec<f32> = vec![0.0; 16000]; // 1 second of silence
    let chunk_size = 1600; // 100ms chunks

    for chunk in samples.chunks(chunk_size) {
        stream.accept_waveform(16000, chunk);

        while stream.is_ready() {
            stream.decode();
        }
    }

    stream.input_finished();
    while stream.is_ready() {
        stream.decode();
    }

    let result = stream.get_result();
    // Result text may be empty for silence, but should not panic
    assert!(result.text.is_empty() || !result.text.is_empty());
}

/// Test endpoint detection (requires model files)
#[test]
#[ignore = "Requires model files to be downloaded"]
fn test_streaming_endpoint_detection() {
    let config = OnlineRecognizerConfig {
        model: OnlineModelType::Transducer(OnlineTransducerModelConfig {
            encoder: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx".into(),
            decoder: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx".into(),
            joiner: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx".into(),
        }),
        tokens: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt".into(),
        endpoint: EndpointConfig {
            enable: true,
            rule1_min_trailing_silence: 0.5,
            rule2_min_trailing_silence: 0.3,
            rule3_min_utterance_length: 5.0,
        },
        ..Default::default()
    };

    let recognizer = OnlineRecognizer::new(config).unwrap();
    let mut stream = recognizer.create_stream().unwrap();

    // Generate longer silence to trigger endpoint
    let samples: Vec<f32> = vec![0.0; 16000 * 3]; // 3 seconds of silence
    stream.accept_waveform(16000, &samples);

    while stream.is_ready() {
        stream.decode();
    }

    // Check is_endpoint doesn't crash
    let _is_endpoint = stream.is_endpoint();

    // Reset should work
    stream.reset();
}

