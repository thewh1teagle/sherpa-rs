/*
Transcribe wav file using Dolphin ASR

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2
tar xvf sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2
rm sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/motivation.wav -O motivation.wav
cargo run --example dolphin motivation.wav
*/

use sherpa_rs::{
    dolphin::{DolphinConfig, DolphinRecognizer},
    read_audio_file,
};

fn main() {
    let path = std::env::args().nth(1).expect("Missing file path argument");
    let provider = std::env::args().nth(2).unwrap_or("cpu".into());
    let (samples, sample_rate) = read_audio_file(&path).unwrap();
    assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");

    let config = DolphinConfig {
        model: "./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/model.int8.onnx".into(),
        tokens: "./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/tokens.txt".into(),
        provider: Some(provider),
        ..Default::default() // fill in any missing fields with defaults
    };
    let mut recognizer = DolphinRecognizer::new(config).unwrap();

    let start_t = std::time::Instant::now();
    let result = recognizer.transcribe(sample_rate, &samples);
    println!("✅ Text: {}", result.text);
    println!("⏱️ Time taken for transcription: {:?}", start_t.elapsed());
}
