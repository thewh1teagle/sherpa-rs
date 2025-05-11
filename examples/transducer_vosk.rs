use sherpa_rs::read_audio_file;
use sherpa_rs::transducer::{TransducerConfig, TransducerRecognizer};
use std::time::Instant;

/*
wget https://huggingface.co/alphacep/vosk-model-ru/resolve/main/am-onnx/decoder.onnx
wget https://huggingface.co/alphacep/vosk-model-ru/resolve/main/am-onnx/encoder.onnx
wget https://huggingface.co/alphacep/vosk-model-ru/resolve/main/am-onnx/joiner.onnx
wget https://huggingface.co/alphacep/vosk-model-ru/resolve/main/lang/tokens.txt
wget https://huggingface.co/alphacep/vosk-model-ru/resolve/main/lang/unigram_500.vocab
wget https://huggingface.co/alphacep/vosk-model-ru/resolve/main/test.wav
touch hotwords.txt
cargo run --example transducer_vosk test.wav
*/
pub fn main() {
    let path = std::env::args().nth(1).expect("Missing file path argument");
    let (samples, sample_rate) = read_audio_file(&path).unwrap();

    // Check if the sample rate is 16000
    if sample_rate != 16000 {
        panic!("The sample rate must be 16000.");
    }

    let config = TransducerConfig {
        decoder: "decoder.onnx".to_string(),
        encoder: "encoder.onnx".to_string(),
        joiner: "joiner.onnx".to_string(),
        tokens: "tokens.txt".to_string(),
        bpe_vocab: "unigram_500.vocab".to_string(),
        hotwords_file: "hotwords.txt".to_string(),
        hotwords_score: 1.2,
        num_threads: 1,
        sample_rate: 16_000,
        feature_dim: 80,
        modeling_unit: "bpe".to_string(),
        decoding_method: "modified_beam_search".to_string(),
        debug: true,
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
