use eyre::{bail, Result};
use sherpa_rs::speaker_identify;
use std::path::PathBuf;

fn read_audio_file(path: &str) -> Result<(i32, Vec<f32>)> {
    let mut reader = hound::WavReader::open(path)?;
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

    Ok((sample_rate, samples))
}

fn main() -> Result<()> {
    env_logger::init();
    // Define paths to the audio files
    let audio_files = vec![
        "samples/obama.wav",
        "samples/trump.wav",
        "samples/biden.wav",
        "samples/biden1.wav",
    ];

    // Create the extractor configuration and extractor
    let mut model_path = PathBuf::from(std::env::current_dir()?);
    model_path.push("wespeaker_en_voxceleb_resnet152_LM.onnx");

    println!("Loading model from {}", model_path.display());

    let config = speaker_identify::ExtractorConfig::new(
        model_path.into_os_string().into_string().unwrap(),
        None,
        None,
        false,
    );
    let mut extractor = speaker_identify::EmbeddingExtractor::new_from_config(config).unwrap();

    // Read and process each audio file
    let mut embeddings = Vec::new();
    for file in audio_files {
        let (sample_rate, samples) = read_audio_file(file)?;
        let embedding = extractor.compute_speaker_embedding(sample_rate, samples)?;
        embeddings.push((file, embedding));
    }

    // Compute cosine similarities
    let mut same_speaker_count = 0;
    for i in 0..embeddings.len() {
        for j in i + 1..embeddings.len() {
            let sim =
                speaker_identify::compute_cosine_similarity(&embeddings[i].1, &embeddings[j].1);
            println!(
                "Cosine similarity between {} and {}: {:.4}",
                embeddings[i].0, embeddings[j].0, sim
            );

            if sim > speaker_identify::DEFAULT_SIMILARITY_THRESHOLD {
                println!(
                    "{} and {} are likely the same speaker.",
                    embeddings[i].0, embeddings[j].0
                );
                same_speaker_count += 1;
            } else {
                println!(
                    "{} and {} are likely different speakers.",
                    embeddings[i].0, embeddings[j].0
                );
            }
        }
    }

    // Print summary
    println!("--------");
    println!("Summary:");
    if same_speaker_count == 0 {
        println!("No pairs of files are likely to be from the same speaker.");
    } else {
        println!(
            "There are {} pairs of files that are likely from the same speaker.",
            same_speaker_count
        );
    }

    Ok(())
}
