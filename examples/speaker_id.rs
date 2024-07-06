use eyre::{bail, Result};
use sherpa_rs::speaker_id;
use std::collections::HashMap;
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
    model_path.push("nemo_en_speakerverification_speakernet.onnx");

    println!("🎤 Loading model from {}", model_path.display());

    let config = speaker_id::ExtractorConfig::new(
        model_path.into_os_string().into_string().unwrap(),
        None,
        None,
        false,
    );
    let mut extractor = speaker_id::EmbeddingExtractor::new_from_config(config)?;

    // Read and process each audio file
    let mut embeddings = Vec::new();
    for file in &audio_files {
        let (sample_rate, samples) = read_audio_file(file)?;
        let embedding = extractor.compute_speaker_embedding(sample_rate, samples)?;
        embeddings.push((file.to_string(), embedding));
    }

    // Identify speakers
    let mut speaker_id_counter = 0;
    let mut speaker_map: HashMap<usize, Vec<String>> = HashMap::new();
    let mut file_speaker_id: HashMap<String, usize> = HashMap::new();

    for i in 0..embeddings.len() {
        let mut assigned = false;
        for j in 0..i {
            let sim = speaker_id::compute_cosine_similarity(&embeddings[i].1, &embeddings[j].1);
            if sim > speaker_id::DEFAULT_SIMILARITY_THRESHOLD {
                let speaker_id = file_speaker_id[&embeddings[j].0];
                speaker_map
                    .entry(speaker_id)
                    .or_default()
                    .push(embeddings[i].0.clone());
                file_speaker_id.insert(embeddings[i].0.clone(), speaker_id);
                assigned = true;
                break;
            }
        }
        if !assigned {
            speaker_map
                .entry(speaker_id_counter)
                .or_default()
                .push(embeddings[i].0.clone());
            file_speaker_id.insert(embeddings[i].0.clone(), speaker_id_counter);
            speaker_id_counter += 1;
        }
    }

    // Print results
    println!("--------");
    println!("📊 Speaker Identification Summary:");
    for (speaker_id, files) in &speaker_map {
        println!("Speaker {}: {:?}", speaker_id, files);
    }

    Ok(())
}
