/*
Create voice embedding for voice in audio file

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_speakerverification_speakernet.onnx
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/16hz_mono_pcm_s16le.wav -O 16hz_mono_pcm_s16le.wav
cargo run --example speaker_embedding 16hz_mono_pcm_s16le.wav
*/

fn main() {
    let file_path = std::env::args().nth(1).expect("Missing file path argument");
    let (samples, sample_rate) = sherpa_rs::read_audio_file(&file_path).unwrap();
    assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");

    let config = sherpa_rs::speaker_id::ExtractorConfig {
        model: "nemo_en_speakerverification_speakernet.onnx".into(),
        ..Default::default()
    };
    let mut extractor = sherpa_rs::speaker_id::EmbeddingExtractor::new(config).unwrap();

    // Compute the speaker embedding
    let embedding = extractor
        .compute_speaker_embedding(samples, sample_rate)
        .unwrap();

    // Use the embedding as needed
    println!("Speaker embedding: {:?}", embedding);
}
