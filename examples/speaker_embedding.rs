/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_speakerverification_speakernet.onnx
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/16hz_mono_pcm_s16le.wav -O 16hz_mono_pcm_s16le.wav
cargo run --example speaker_embedding 16hz_mono_pcm_s16le.wav
*/

use eyre::{bail, Result};
use sherpa_rs::speaker_id;
use std::io::Cursor;
use std::path::PathBuf;

fn main() -> Result<()> {
    let file_path = std::env::args().nth(1).expect("Missing file path argument");
    let audio_data = std::fs::read(file_path)?;

    // Use Cursor to create a reader from the byte slice
    let cursor = Cursor::new(audio_data);
    let mut reader = hound::WavReader::new(cursor)?;
    let sample_rate = reader.spec().sample_rate as i32;

    // Check if the sample rate is 16000
    if sample_rate != 16000 {
        bail!("The sample rate must be 16000.");
    }

    // Collect samples into a Vec<f32>
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    // Create the extractor configuration and extractor
    let mut model_path = PathBuf::from(std::env::current_dir()?);
    model_path.push("nemo_en_speakerverification_speakernet.onnx");

    println!("loading model from {}", model_path.display());

    // Create the extractor configuration and extractor
    let config = speaker_id::ExtractorConfig::new(
        model_path.into_os_string().into_string().unwrap(),
        None,
        None,
        false,
    );
    let mut extractor = speaker_id::EmbeddingExtractor::new_from_config(config).unwrap();

    // Compute the speaker embedding
    let embedding = extractor.compute_speaker_embedding(sample_rate, samples)?;

    // Use the embedding as needed
    println!("Speaker embedding: {:?}", embedding);

    Ok(())
}
