/*
Detect speech in audio file and transcribe it

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
wget https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_speakerverification_speakernet.onnx
tar xvf sherpa-onnx-whisper-tiny.tar.bz2
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/sam_altman.wav -O sam_altman.wav
cargo run --example vad_whisper sam_altman.wav
*/
use sherpa_rs::{
    embedding_manager, read_audio_file,
    silero_vad::{SileroVad, SileroVadConfig},
    speaker_id,
    whisper::{WhisperConfig, WhisperRecognizer},
};

fn main() {
    // Read audio data from the file
    let path = std::env::args().nth(1).expect("Missing file path argument");
    let (mut samples, sample_rate) = read_audio_file(&path).unwrap();
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

    let config = WhisperConfig {
        decoder: "sherpa-onnx-whisper-tiny/tiny-decoder.onnx".into(),
        encoder: "sherpa-onnx-whisper-tiny/tiny-encoder.onnx".into(),
        tokens: "sherpa-onnx-whisper-tiny/tiny-tokens.txt".into(),
        language: "en".into(),
        ..Default::default() // fill in any missing fields with defaults
    };

    let mut recognizer = WhisperRecognizer::new(config).unwrap();

    let mut speaker_counter = 0;

    let window_size: usize = 512;
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
                let segment = vad.front();
                let start_sec = (segment.start as f32) / sample_rate as f32;
                let duration_sec = (segment.samples.len() as f32) / sample_rate as f32;
                let transcript = recognizer.transcribe(sample_rate, &segment.samples);

                // Compute the speaker embedding
                let mut embedding = extractor
                    .compute_speaker_embedding(segment.samples, sample_rate)
                    .unwrap();
                let name = if let Some(speaker_name) = embedding_manager.search(&embedding, 0.4) {
                    speaker_name
                } else {
                    // Register a new speaker and add the embedding
                    let name = format!("speaker {}", speaker_counter);
                    embedding_manager.add(name.clone(), &mut embedding).unwrap();

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
            let transcript = recognizer.transcribe(sample_rate, &segment.samples);

            // Compute the speaker embedding
            let mut embedding = extractor
                .compute_speaker_embedding(segment.samples, sample_rate)
                .unwrap();

            let name = if let Some(speaker_name) = embedding_manager.search(&embedding, 0.4) {
                speaker_name
            } else {
                // Register a new speaker and add the embedding
                let name = format!("speaker {}", speaker_counter);
                embedding_manager.add(name.clone(), &mut embedding).unwrap();

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
}
