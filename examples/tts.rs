use std::path::PathBuf;

/// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
/// tar xf vits-piper-en_US-amy-low.tar.bz2
/// cargo run --example tts

fn main() {
    let assets = PathBuf::from("vits-piper-en_US-amy-low");
    let vits_tokens = assets.join("tokens.txt");
    let vits_model = assets.join("en_US-amy-low.onnx");
    let vits_data_dir = assets.join("espeak-ng-data");

    let text = "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.";
    let vits_cfg = sherpa_rs::tts::TtsVitsModelConfig::new(
        vits_model.to_str().unwrap().into(),
        "".into(),
        vits_tokens.to_str().unwrap().into(),
        vits_data_dir.to_str().unwrap().into(),
        0.0,
        0.0,
        "".into(),
        0.0,
    );
    let max_num_sentences = 2;
    let model_cfg = sherpa_rs::tts::OfflineTtsModelConfig::new(true, vits_cfg, None, 2);
    let tts_cfg =
        sherpa_rs::tts::OfflineTtsConfig::new(model_cfg, max_num_sentences, "".into(), "".into());
    let mut tts = sherpa_rs::tts::OfflineTts::new(tts_cfg);
    let speed = 1.0;
    let audio = tts.generate(text.into(), 0, speed).unwrap();
    audio.write_to_wav("audio.wav").unwrap();
}
