use sherpa_rs::read_audio_file;
use sherpa_rs::transducer::{TransducerConfig, TransducerRecognizer};
use std::time::Instant;

/*
wget https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230926-small/resolve/main/decoder-epoch-90-avg-20.onnx
wget https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230926-small/resolve/main/encoder-epoch-90-avg-20.onnx
wget https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230926-small/resolve/main/joiner-epoch-90-avg-20.onnx
wget https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230926-small/resolve/main/tokens.txt
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/motivation.wav -O motivation.wav
cargo run --example transducer motivation.wav
*/

pub fn main() {
    let path = std::env::args().nth(1).expect("Missing file path argument");
    let (samples, sample_rate) = read_audio_file(&path).unwrap();

    // Check if the sample rate is 16000
    if sample_rate != 16000 {
        panic!("The sample rate must be 16000.");
    }

    let config = TransducerConfig {
        decoder: "decoder-epoch-90-avg-20.onnx".to_string(),
        encoder: "encoder-epoch-90-avg-20.onnx".to_string(),
        joiner: "joiner-epoch-90-avg-20.onnx".to_string(),
        tokens: "tokens.txt".to_string(),
        num_threads: 1,
        sample_rate: 16_000,
        feature_dim: 80,

        // NULLs
        bpe_vocab: "".to_string(),
        decoding_method: "".to_string(),
        hotwords_file: "".to_string(),
        hotwords_score: 0.0,
        modeling_unit: "".to_string(),
        blank_penalty: 0.0,
        debug: true,
        provider: None,
    };

    let mut recognizer = TransducerRecognizer::new(config).unwrap();

    let start_t = Instant::now();
    let result = recognizer.transcribe(sample_rate, &samples);
    let lower_case = result.to_lowercase();
    let trimmed_result = lower_case.trim();

    println!("Time taken for decode: {:?}", start_t.elapsed());
    println!("Transcribe result: {:?}", trimmed_result);
}
