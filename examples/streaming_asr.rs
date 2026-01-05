/*
Streaming (online) speech recognition example

This example demonstrates how to use the streaming API for real-time
speech recognition. It simulates streaming by processing audio in chunks.

Download model files from:
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html

Example with Zipformer Transducer:
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

cargo run --example streaming_asr -- audio.wav
*/

use sherpa_rs::{
    read_audio_file,
    streaming::{
        EndpointConfig, OnlineModelType, OnlineRecognizer, OnlineRecognizerConfig,
        OnlineTransducerModelConfig,
    },
};

fn main() {
    let path = std::env::args().nth(1).expect("Missing file path argument");
    let provider = std::env::args().nth(2).unwrap_or("cpu".into());
    let (samples, sample_rate) = read_audio_file(&path).unwrap();
    assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");

    // Configure the streaming recognizer
    let config = OnlineRecognizerConfig {
        model: OnlineModelType::Transducer(OnlineTransducerModelConfig {
            encoder: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx".into(),
            decoder: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx".into(),
            joiner: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx".into(),
        }),
        tokens: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt".into(),
        provider: Some(provider),
        endpoint: EndpointConfig {
            enable: true,
            rule1_min_trailing_silence: 2.4,
            rule2_min_trailing_silence: 1.2,
            rule3_min_utterance_length: 20.0,
        },
        ..Default::default()
    };

    let recognizer = OnlineRecognizer::new(config).unwrap();
    let mut stream = recognizer.create_stream().unwrap();

    // Simulate streaming by processing audio in chunks
    // In real applications, you would get chunks from a microphone or network
    let chunk_size = 1600; // 100ms at 16kHz
    let mut last_text = String::new();

    println!("Processing audio in streaming mode...\n");

    let start_t = std::time::Instant::now();

    for (i, chunk) in samples.chunks(chunk_size).enumerate() {
        // Accept audio chunk
        stream.accept_waveform(sample_rate as i32, chunk);

        // Decode while ready
        while stream.is_ready() {
            stream.decode();
        }

        // Get partial result
        let result = stream.get_result();
        if !result.text.is_empty() && result.text != last_text {
            let time_offset = (i * chunk_size) as f32 / sample_rate as f32;
            println!("[{:.2}s] {}", time_offset, result.text);
            last_text = result.text;
        }

        // Check for endpoint (sentence boundary)
        if stream.is_endpoint() {
            let result = stream.get_result();
            if !result.text.is_empty() {
                println!("\n[Endpoint detected] Final: {}\n", result.text);
            }
            stream.reset();
            last_text.clear();
        }
    }

    // Signal end of input and get final result
    stream.input_finished();
    while stream.is_ready() {
        stream.decode();
    }

    let final_result = stream.get_result();
    if !final_result.text.is_empty() {
        println!("\n[Final result] {}", final_result.text);
    }

    println!("\nTotal time: {:?}", start_t.elapsed());
}

