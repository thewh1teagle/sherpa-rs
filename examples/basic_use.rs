use eyre::{bail, Result};
use sherpa_rs::speaker_identify;
use std::io::Cursor;

fn main() -> Result<()> {
    // Read audio data from the file
    let audio_data: &[u8] = include_bytes!("../samples/16hz_mono_pcm_s16le.wav");

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
    let config = speaker_identify::ExtractorConfig::new(
        "wespeaker_zh_cnceleb_resnet34.onnx".into(),
        None,
        None,
        false,
    );
    let mut extractor = speaker_identify::EmbeddingExtractor::new_from_config(config).unwrap();

    // Compute the speaker embedding
    let embedding = extractor.compute_speaker_embedding(sample_rate, samples)?;

    // Use the embedding as needed
    println!("Speaker embedding: {:?}", embedding);

    Ok(())
}
