/*
Use ASR models for extract text from audio

English:
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-small-en-2023-06-26.tar.bz2
tar xvf sherpa-onnx-zipformer-small-en-2023-06-26.tar.bz2
rm sherpa-onnx-zipformer-small-en-2023-06-26.tar.bz2

cargo run --example zipformer -- \
    "sherpa-onnx-zipformer-small-en-2023-06-26/test_wavs/0.wav" \
    "sherpa-onnx-zipformer-small-en-2023-06-26/encoder-epoch-99-avg-1.onnx" \
    "sherpa-onnx-zipformer-small-en-2023-06-26/decoder-epoch-99-avg-1.onnx" \
    "sherpa-onnx-zipformer-small-en-2023-06-26/joiner-epoch-99-avg-1.onnx" \
    "sherpa-onnx-zipformer-small-en-2023-06-26/tokens.txt"

Japanse:
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2
tar xvf sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2
rm sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2

cargo run --example zipformer -- \
    "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/test_wavs/1.wav" \
    "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/encoder-epoch-99-avg-1.onnx" \
    "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/decoder-epoch-99-avg-1.onnx" \
    "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/joiner-epoch-99-avg-1.onnx" \
    "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/tokens.txt"
*/

use sherpa_rs::zipformer::ZipFormer;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (wav_path, encoder_path, decoder_path, joiner_path, tokens_path) = (
        args.get(1).expect("Missing wav file path argument"),
        args.get(2).expect("Missing encoder path argument"),
        args.get(3).expect("Missing decoder path argument"),
        args.get(4).expect("Missing joiner path argument"),
        args.get(5).expect("Missing tokens path argument"),
    );

    // Read the WAV file
    let (samples, sample_rate) = sherpa_rs::read_audio_file(wav_path).unwrap();

    let config = sherpa_rs::zipformer::ZipFormerConfig {
        encoder: encoder_path.into(),
        decoder: decoder_path.into(),
        joiner: joiner_path.into(),
        tokens: tokens_path.into(),
        ..Default::default()
    };
    let mut zipformer = ZipFormer::new(config).unwrap();
    let text = zipformer.decode(sample_rate, samples);
    println!("✅ Text: {}", text);
}
