/*
Transcribe wav file using streaming Paraformer and punctuate the result

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
tar xvf sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

cargo run --example paraformer_streaming motivation.wav
*/

use sherpa_rs::{
    paraformer::{ParaformerOnlineConfig, ParaformerOnlineRecognizer},
    read_audio_file,
};

fn main() {
    let path = std::env::args().nth(1).expect("Missing file path argument");
    let provider = std::env::args().nth(2).unwrap_or("cpu".into());
    let (samples, sample_rate) = read_audio_file(&path).unwrap();
    assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");

    let config = ParaformerOnlineConfig {
        tokens: "sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt".into(),
        provider: Some(provider),
        debug: true,
        encoder_model_path: "sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx"
            .into(),
        decoder_model_path: "sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx"
            .into(),
        enable_endpoint: Some(true),
        ..Default::default()
    };

    let mut recognizer = ParaformerOnlineRecognizer::new(config).unwrap();

    for chunk in samples.chunks(1600) {
        let result = recognizer.transcribe(sample_rate, &chunk);
        if result.text.is_empty() {
            continue;
        }
        if result.is_final {
            println!("🎉 Final: {}", result.text);
        } else {
            println!("💬 Partial: {}", result.text);
        }
    }
}
