/*
wget https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_speakerverification_speakernet.onnx
cargo run --example diarize
*/

use eyre::{bail, Result};
use sherpa_rs::{
    embedding_manager, speaker_id,
    vad::{Vad, VadConfig},
};
use std::io::Cursor;

fn main() -> Result<()> {
    // Read audio data from the file
    let audio_data: &[u8] = include_bytes!("../samples/motivation.wav");

    let cursor = Cursor::new(audio_data);
    let mut reader = hound::WavReader::new(cursor)?;
    let sample_rate = reader.spec().sample_rate as i32;

    if sample_rate != 16000 {
        bail!("The sample rate must be 16000.");
    }

    let mut samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    let extractor_config = speaker_id::ExtractorConfig::new(
        "nemo_en_speakerverification_speakernet.onnx".into(),
        None,
        None,
        false,
    );
    let mut extractor = speaker_id::EmbeddingExtractor::new_from_config(extractor_config).unwrap();
    let mut embedding_manager =
        embedding_manager::EmbeddingManager::new(extractor.embedding_size.try_into().unwrap()); // Assuming dimension 512 for embeddings

    let mut speaker_counter = 0;

    let vad_model = "silero_vad.onnx".into();
    let window_size: usize = 512;
    let config = VadConfig::new(
        vad_model,
        0.4,
        0.4,
        0.5,
        sample_rate,
        window_size.try_into().unwrap(),
        None,
        None,
        Some(false),
    );

    let mut vad = Vad::new_from_config(config, 3.0).unwrap();
    while samples.len() > window_size {
        let window = &samples[..window_size];
        vad.accept_waveform(window.to_vec()); // Convert slice to Vec
        if vad.is_speech() {
            while !vad.is_empty() {
                let segment = vad.front();
                let start_sec = (segment.start as f32) / sample_rate as f32;
                let duration_sec = (segment.samples.len() as f32) / sample_rate as f32;

                // Compute the speaker embedding
                let embedding =
                    extractor.compute_speaker_embedding(sample_rate, segment.samples)?;

                let name = if let Some(speaker_name) = embedding_manager.search(&embedding, 0.45) {
                    speaker_name
                } else {
                    // Register a new speaker and add the embedding
                    let name = format!("speaker {}", speaker_counter);
                    embedding_manager.add(name.clone(), &mut embedding.clone())?;

                    speaker_counter += 1;
                    name
                };
                println!("({}) start={}s duration={}s", name, start_sec, duration_sec);
                vad.pop();
            }
            vad.clear();
        }
        samples = samples[window_size..].to_vec(); // Move the remaining samples to the next iteration
    }
    Ok(())
}
