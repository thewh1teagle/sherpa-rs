/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
wget https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_speakerverification_speakernet.onnx
tar xvf sherpa-onnx-whisper-tiny.tar.bz2
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/sam_altman.wav -O sam_altman.wav
cargo run --example vad_whisper sam_altman.wav
*/

use eyre::{bail, Result};
use sherpa_rs::{
    embedding_manager, speaker_id,
    transcribe::whisper::WhisperRecognizer,
    vad::{Vad, VadConfig},
};

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
    // Read audio data from the file
    let path = std::env::args().nth(1).expect("Missing file path argument");
    let (sample_rate, mut samples) = read_audio_file(&path)?;

    // Pad with 3 seconds of slience so vad will able to detect stop
    for _ in 0..3 * sample_rate {
        samples.push(0.0);
    }

    let extractor_config = speaker_id::ExtractorConfig::new(
        "nemo_en_speakerverification_speakernet.onnx".into(),
        None,
        None,
        false,
    );
    let mut extractor = speaker_id::EmbeddingExtractor::new_from_config(extractor_config).unwrap();
    let mut embedding_manager =
        embedding_manager::EmbeddingManager::new(extractor.embedding_size.try_into().unwrap()); // Assuming dimension 512 for embeddings

    let mut recognizer = WhisperRecognizer::new(
        "sherpa-onnx-whisper-tiny/tiny-decoder.onnx".into(),
        "sherpa-onnx-whisper-tiny/tiny-encoder.onnx".into(),
        "sherpa-onnx-whisper-tiny/tiny-tokens.txt".into(),
        "en".into(),
        Some(false),
        None,
        None,
        None,
    );

    let mut speaker_counter = 0;

    let vad_model = "silero_vad.onnx".into();
    let window_size: usize = 512;
    let config = VadConfig::new(
        vad_model,
        0.4,
        0.4,
        0.5,
        0.5,
        sample_rate,
        window_size.try_into().unwrap(),
        None,
        None,
        Some(false),
    );

    let mut vad = Vad::new_from_config(config, 60.0 * 10.0).unwrap();
    let mut index = 0;
    while index + window_size <= samples.len() {
        let window = &samples[index..index + window_size];
        vad.accept_waveform(window.to_vec()); // Convert slice to Vec
        if vad.is_speech() {
            while !vad.is_empty() {
                let segment = vad.front();
                let start_sec = (segment.start as f32) / sample_rate as f32;
                let duration_sec = (segment.samples.len() as f32) / sample_rate as f32;
                let transcript = recognizer.transcribe(sample_rate, segment.samples.clone());

                // Compute the speaker embedding
                let mut embedding =
                    extractor.compute_speaker_embedding(sample_rate, segment.samples)?;
                let name = if let Some(speaker_name) = embedding_manager.search(&embedding, 0.4) {
                    speaker_name
                } else {
                    // Register a new speaker and add the embedding
                    let name = format!("speaker {}", speaker_counter);
                    embedding_manager.add(name.clone(), &mut embedding)?;

                    speaker_counter += 1;
                    name
                };
                println!(
                    "({}) {} | {}s - {}s",
                    name,
                    transcript.text,
                    start_sec,
                    start_sec + duration_sec
                );
                vad.pop();
            }
        }
        index += window_size;
    }

    if index < samples.len() {
        let remaining_samples = &samples[index..];
        vad.accept_waveform(remaining_samples.to_vec());
        while !vad.is_empty() {
            let segment = vad.front();
            let start_sec = (segment.start as f32) / sample_rate as f32;
            let duration_sec = (segment.samples.len() as f32) / sample_rate as f32;
            let transcript = recognizer.transcribe(sample_rate, segment.samples.clone());

            // Compute the speaker embedding
            let mut embedding =
                extractor.compute_speaker_embedding(sample_rate, segment.samples)?;

            let name = if let Some(speaker_name) = embedding_manager.search(&embedding, 0.4) {
                speaker_name
            } else {
                // Register a new speaker and add the embedding
                let name = format!("speaker {}", speaker_counter);
                embedding_manager.add(name.clone(), &mut embedding)?;

                speaker_counter += 1;
                name
            };
            println!(
                "({}) {} | {}s - {}s",
                name,
                transcript.text,
                start_sec,
                start_sec + duration_sec
            );
            vad.pop();
        }
    }
    Ok(())
}
