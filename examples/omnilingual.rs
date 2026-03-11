/*
Transcribe wav file using Omnilingual ASR CTC model
Supports 1600+ languages

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2
tar xvf sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2
rm sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/motivation.wav -O motivation.wav
cargo run --example omnilingual -- motivation.wav [provider]
# provider is optional, defaults to "cpu". Can be "cuda", "coreml", etc.
*/

use sherpa_rs::{
    omnilingual::{OmnilingualConfig, OmnilingualRecognizer},
    read_audio_file,
};

fn main() {
    let path = std::env::args().nth(1).expect("Missing file path argument");
    let provider = std::env::args().nth(2).unwrap_or("cpu".into());
    let (samples, sample_rate) = read_audio_file(&path).unwrap();
    assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");

    let model_dir = "./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12";

    let config = OmnilingualConfig {
        model: format!("{}/model.int8.onnx", model_dir),
        tokens: format!("{}/tokens.txt", model_dir),
        provider: Some(provider),
        ..Default::default()
    };
    let mut recognizer = OmnilingualRecognizer::new(config).unwrap();

    let start_t = std::time::Instant::now();
    let result = recognizer.transcribe(sample_rate, &samples);
    println!("Text: {}", result.text);
    println!("Time taken for transcription: {:?}", start_t.elapsed());
}
