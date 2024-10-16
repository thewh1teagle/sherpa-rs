/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-small-en-2023-06-26.tar.bz2
tar xvf sherpa-onnx-zipformer-small-en-2023-06-26.tar.bz2
rm sherpa-onnx-zipformer-small-en-2023-06-26.tar.bz2

cargo run --example zipformer -- \
    "sherpa-onnx-zipformer-small-en-2023-06-26/test_wavs/0.wav" \
    "sherpa-onnx-zipformer-small-en-2023-06-26/encoder-epoch-99-avg-1.onnx" \
    "sherpa-onnx-zipformer-small-en-2023-06-26/decoder-epoch-99-avg-1.onnx" \
    "sherpa-onnx-zipformer-small-en-2023-06-26/joiner-epoch-99-avg-1.onnx" \
    "sherpa-onnx-zipformer-small-en-2023-06-26/tokens.txt"
*/

use sherpa_rs::zipformer::ZipFormer;
use std::env::args;
fn main() {
    let wav_path = args().nth(1).expect("Missing wav file path argument");
    let encoder_path = args().nth(2).expect("Missing encoder path argument");
    let decoder_path = args().nth(3).expect("Missing decoder path argument");
    let joiner_path = args().nth(4).expect("Missing joiner path argument");
    let tokens_path = args().nth(5).expect("Missing tokens path argument");

    // Read the WAV file
    let reader = hound::WavReader::open(&wav_path).expect("Failed to open WAV file");
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let samples: Vec<f32> = reader
        .into_samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    let config = sherpa_rs::zipformer::ZipFormerConfig {
        encoder: encoder_path.into(),
        decoder: decoder_path.into(),
        joiner: joiner_path.into(),
        tokens: tokens_path.into(),
        ..Default::default()
    };
    let mut zipformer = ZipFormer::new(config).unwrap();
    let text = zipformer.decode(sample_rate, samples);
    println!("Text: {}", text);
}
