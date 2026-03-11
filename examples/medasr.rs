/*
Transcribe wav file using Google MedASR CTC model
Optimized for medical speech recognition

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
tar xvf sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
rm sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/motivation.wav -O motivation.wav
cargo run --example medasr -- motivation.wav [provider]
# provider is optional, defaults to "cpu". Can be "cuda", "coreml", etc.
*/

use sherpa_rs::{
    medasr::{MedAsrConfig, MedAsrRecognizer},
    read_audio_file,
};

fn main() {
    let path = std::env::args().nth(1).expect("Missing file path argument");
    let provider = std::env::args().nth(2).unwrap_or("cpu".into());
    let (samples, sample_rate) = read_audio_file(&path).unwrap();
    assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");

    let model_dir = "./sherpa-onnx-medasr-ctc-en-int8-2025-12-25";

    let config = MedAsrConfig {
        model: format!("{}/model.int8.onnx", model_dir),
        tokens: format!("{}/tokens.txt", model_dir),
        provider: Some(provider),
        ..Default::default()
    };
    let mut recognizer = MedAsrRecognizer::new(config).unwrap();

    let start_t = std::time::Instant::now();
    let result = recognizer.transcribe(sample_rate, &samples);
    println!("Text: {}", result.text);
    println!("Time taken for transcription: {:?}", start_t.elapsed());
}
