/*
Detect voice in audio file and mark start and stop.

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_speakerverification_speakernet.onnx
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/motivation.wav -O motivation.wav
cargo run --example vad_silero motivation.wav
*/
use sherpa_rs::{
    embedding_manager,
    silero_vad::{SileroVad, SileroVadConfig},
    speaker_id,
};

fn get_speaker_name(
    embedding_manager: &mut embedding_manager::EmbeddingManager,
    embedding: &mut [f32],
    speaker_counter: &mut i32,
    max_speakers: i32,
) -> String {
    let mut name = String::from("unknown");

    if *speaker_counter == 0 {
        name = format!("speaker {}", speaker_counter);
        embedding_manager.add(name.clone(), embedding).unwrap();
        *speaker_counter += 1;
    } else if *speaker_counter <= max_speakers {
        if let Some(search_result) = embedding_manager.search(embedding, 0.5) {
            name = search_result;
        } else {
            name = format!("speaker {}", speaker_counter);
            embedding_manager.add(name.clone(), embedding).unwrap();
            *speaker_counter += 1;
        }
    } else {
        let matches = embedding_manager.get_best_matches(embedding, 0.2, *speaker_counter);
        if let Some(name_match) = matches.first().map(|m| m.name.clone()) {
            name = name_match;
        }
    }

    name
}

fn process_speech_segment(
    vad: &mut SileroVad,
    sample_rate: u32,
    embedding_manager: &mut embedding_manager::EmbeddingManager,
    extractor: &mut speaker_id::EmbeddingExtractor,
    speaker_counter: &mut i32,
    max_speakers: i32,
) {
    while !vad.is_empty() {
        let segment = vad.front();
        let start_sec = (segment.start as f32) / sample_rate as f32;
        let duration_sec = (segment.samples.len() as f32) / sample_rate as f32;

        // Compute the speaker embedding
        let mut embedding = extractor
            .compute_speaker_embedding(segment.samples, sample_rate)
            .unwrap();

        let name = get_speaker_name(
            embedding_manager,
            &mut embedding,
            speaker_counter,
            max_speakers,
        );
        println!(
            "({}) start={}s end={}s",
            name,
            start_sec,
            start_sec + duration_sec
        );
        vad.pop();
    }
}

fn main() {
    let file_path = std::env::args().nth(1).expect("Missing file path argument");
    let max_speakers = 2;

    let (mut samples, sample_rate) = sherpa_rs::read_audio_file(&file_path).unwrap();
    assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");

    // Pad with 3 seconds of silence so vad will be able to detect stop
    samples.extend(vec![0.0; (3 * sample_rate) as usize]);

    let extractor_config = speaker_id::ExtractorConfig {
        model: "nemo_en_speakerverification_speakernet.onnx".into(),
        ..Default::default()
    };
    let mut extractor = speaker_id::EmbeddingExtractor::new(extractor_config).unwrap();
    let mut embedding_manager =
        embedding_manager::EmbeddingManager::new(extractor.embedding_size.try_into().unwrap()); // Assuming dimension 512 for embeddings

    let mut speaker_counter = 1;

    let window_size = 512;
    let vad_config = SileroVadConfig {
        model: "silero_vad.onnx".into(),
        window_size: window_size as i32,
        ..Default::default()
    };

    let mut vad = SileroVad::new(vad_config, 60.0 * 10.0).unwrap();
    let mut index = 0;
    while index + window_size <= samples.len() {
        let window = &samples[index..index + window_size];
        vad.accept_waveform(window.to_vec()); // Convert slice to Vec
        if vad.is_speech() {
            while !vad.is_empty() {
                process_speech_segment(
                    &mut vad,
                    sample_rate,
                    &mut embedding_manager,
                    &mut extractor,
                    &mut speaker_counter,
                    max_speakers,
                );
            }
        }

        index += window_size;
    }
    vad.flush();
    // process reamaining
    while !vad.is_empty() {
        process_speech_segment(
            &mut vad,
            sample_rate,
            &mut embedding_manager,
            &mut extractor,
            &mut speaker_counter,
            max_speakers,
        );
    }
}
