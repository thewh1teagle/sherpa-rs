/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
tar xvf sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
rm sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/motivation.wav -O motivation.wav
cargo run --example parakeet motivation.wav
*/

use sherpa_rs::read_audio_file;
use sherpa_rs::transducer::{TransducerConfig, TransducerRecognizer};
use std::time::Instant;

pub fn main() {
    let path = std::env::args().nth(1).expect("Missing file path argument");
    let (samples, sample_rate) = read_audio_file(&path).unwrap();

    // Check if the sample rate is 16000
    if sample_rate != 16000 {
        panic!("The sample rate must be 16000.");
    }

    let config = TransducerConfig {
        decoder: "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx".to_string(),
        encoder: "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx".to_string(),
        joiner: "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx".to_string(),
        tokens: "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt".to_string(),
        num_threads: 1,
        sample_rate: 16_000,
        feature_dim: 80,
        debug: true,
        model_type: "nemo_transducer".to_string(),
        ..Default::default()
    };

    let mut recognizer = TransducerRecognizer::new(config).unwrap();

    let start_t = Instant::now();
    let result = recognizer.transcribe(sample_rate, &samples);
    let lower_case = result.to_lowercase();
    let trimmed_result = lower_case.trim();

    println!("Time taken for decode: {:?}", start_t.elapsed());
    println!("Transcribe result: {:?}", trimmed_result);
}
