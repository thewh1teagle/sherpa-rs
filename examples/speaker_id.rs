/*
Recognize speakers in audio file

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_speakerverification_speakernet.onnx
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/biden.wav -O biden.wav
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/obama.wav -O obama.wav
cargo run --example speaker_id
*/
use sherpa_rs::{embedding_manager, speaker_id};
use std::collections::HashMap;

fn main() {
    // Define paths to the audio files
    let audio_files = vec!["obama.wav", "biden.wav"];

    let config = speaker_id::ExtractorConfig {
        model: "nemo_en_speakerverification_speakernet.onnx".into(),
        ..Default::default()
    };
    let mut extractor = speaker_id::EmbeddingExtractor::new(config).unwrap();

    // Read and process each audio file, compute embeddings
    let mut embeddings = Vec::new();
    for file in &audio_files {
        let (samples, sample_rate) = sherpa_rs::read_audio_file(&file).unwrap();
        assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");
        let embedding = extractor
            .compute_speaker_embedding(samples, sample_rate)
            .unwrap();
        embeddings.push((file.to_string(), embedding));
    }

    // Create the embedding manager
    let mut embedding_manager =
        embedding_manager::EmbeddingManager::new(extractor.embedding_size.try_into().unwrap()); // Assuming dimension 512 for embeddings

    // Map to store speakers and their corresponding files
    let mut speaker_map: HashMap<String, Vec<String>> = HashMap::new();
    let mut speaker_counter = 0;

    // Process each embedding and identify speakers
    for (file, embedding) in &embeddings {
        if let Some(speaker_name) = embedding_manager.search(embedding, 0.5) {
            // Add file to existing speaker
            speaker_map
                .entry(speaker_name)
                .or_default()
                .push(file.clone());
        } else {
            // Register a new speaker and add the embedding
            embedding_manager
                .add(
                    format!("speaker {}", speaker_counter),
                    &mut embedding.clone(),
                )
                .unwrap();
            speaker_map
                .entry(format!("speaker {}", speaker_counter))
                .or_default()
                .push(file.clone());
            speaker_counter += 1;
        }
    }

    // Print results
    println!("--------");
    println!("📊 Speaker Identification Summary:");
    for (speaker_id, files) in &speaker_map {
        println!("Speaker {}: {:?}", speaker_id, files);
    }
}
