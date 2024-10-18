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

    let reader = hound::WavReader::open(&wav_path).expect("Failed to open WAV file");
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let samples: Vec<f32> = reader
        .into_samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    let config = sherpa_rs::audio_tag::AudioTagConfig {
        model: model.into(),
        labels: labels_path.into(),
        top_k: 5,
        ..Default::default()
    };
    let mut audio_tag = sherpa_rs::audio_tag::AudioTag::new(config).unwrap();
    let events = audio_tag.compute(samples, sample_rate);
    println!("✅ Events ({}): {}", events.len(), events.join(", "));
}
