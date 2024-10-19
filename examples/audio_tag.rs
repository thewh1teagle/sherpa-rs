/*
Audio tagging identifies specific audio events from audio file.

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
tar xvf sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
rm sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2

cargo run --example audio_tag
*/

fn main() {
    let model = "./sherpa-onnx-zipformer-audio-tagging-2024-04-09/model.int8.onnx";
    let labels_path = "./sherpa-onnx-zipformer-audio-tagging-2024-04-09/class_labels_indices.csv";
    let wav_path = "./sherpa-onnx-zipformer-audio-tagging-2024-04-09/test_wavs/1.wav";
    let top_k = 5;

    let (samples, sample_rate) = sherpa_rs::read_audio_file(&wav_path).unwrap();

    let config = sherpa_rs::audio_tag::AudioTagConfig {
        model: model.into(),
        labels: labels_path.into(),
        top_k,
        ..Default::default()
    };
    let mut audio_tag = sherpa_rs::audio_tag::AudioTag::new(config).unwrap();
    let events = audio_tag.compute(samples, sample_rate);
    println!("✅ Events ({}): {}", events.len(), events.join(", "));
}
