/*
We assume you have pre-downloaded the model files for testing
from https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html


English:
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2
tar xvf sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2
rm sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2

cargo run --example keyword_spot \
    "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/1.wav" \
    "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx" \
    "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx" \
    "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx" \
    "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/tokens.txt" \
    "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt"

Chinese:
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
tar xf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
rm sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2

cargo run --example keyword_spot \
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile/test_wavs/6.wav \
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx \
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile/decoder-epoch-12-avg-2-chunk-16-left-64.onnx \
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx \
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile/tokens.txt \
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile/test_wavs/test_keywords.txt

*/

use std::env::args;

fn main() {
    let audio_path = args().nth(1).expect("Please specify the audio file path");
    let zipformer_encoder = args()
        .nth(2)
        .expect("Please specify the zipformer encoder file");
    let zipformer_decoder = args()
        .nth(3)
        .expect("Please specify the zipformer decoder file");
    let zipformer_joiner = args()
        .nth(4)
        .expect("Please specify the zipformer joiner file");
    let tokens = args().nth(5).expect("Please specify the tokens file");
    let keywords = args().nth(6).expect("Please specify the keywords file");

    let (samples, sample_rate) = sherpa_rs::read_audio_file(&audio_path).unwrap();

    let config = sherpa_rs::keyword_spot::KeywordSpotConfig {
        zipformer_encoder,
        zipformer_decoder,
        zipformer_joiner,
        tokens,
        keywords,
        max_active_path: 4,
        keywords_threshold: 0.1,
        keywords_score: 3.0,
        num_trailing_blanks: 1,
        sample_rate: 16000,
        feature_dim: 80,
        ..Default::default()
    };
    let mut spotter = sherpa_rs::keyword_spot::KeywordSpot::new(config).unwrap();

    let keyword = spotter
        .extract_keyword(samples, sample_rate)
        .unwrap()
        .unwrap_or("?".into());
    println!("Keyword: {}", keyword);
}
