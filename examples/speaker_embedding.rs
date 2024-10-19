/*
Create voice embedding for voice in audio file

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_speakerverification_speakernet.onnx
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/16hz_mono_pcm_s16le.wav -O 16hz_mono_pcm_s16le.wav
cargo run --example speaker_embedding 16hz_mono_pcm_s16le.wav
*/

fn main() {
    let file_path = std::env::args().nth(1).expect("Missing file path argument");
    let mut reader = hound::WavReader::open(file_path).unwrap();
    let sample_rate = reader.spec().sample_rate;

    // Check if the sample rate is 16000
    if sample_rate != 16000 {
        panic!("The sample rate must be 16000.");
    }

    // Collect samples into a Vec<f32>
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

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
