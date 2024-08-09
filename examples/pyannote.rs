/*

*Prepare Whisper*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.tar.bz2

*Prepare Pyannote*
wget https://github.com/pengzhendong/pyannote-onnx/raw/master/pyannote_onnx/segmentation-3.0.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx

*Run example*
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/6_speakers.wav
cargo run --features example-pyannote --example pyannote

*Prepare DLLs for pyannote's onnxruntime (18.x) since pyannote-rs load them dynamic and sherpa load static*
https://github.com/microsoft/onnxruntime/releases
Place them in target/debug/examples/ AND target/debug/
*/
use sherpa_rs::transcribe::whisper::WhisperRecognizer;
use std::time::Instant;

fn main() {
    let audio_path = std::env::args().nth(1).expect("Please specify audio file");

    // Sherpa
    let mut recognizer = WhisperRecognizer::new(
        "sherpa-onnx-whisper-tiny/tiny-decoder.onnx".into(),
        "sherpa-onnx-whisper-tiny/tiny-encoder.onnx".into(),
        "sherpa-onnx-whisper-tiny/tiny-tokens.txt".into(),
        "en".into(),
        Some(true),
        None,
        None,
        None,
    );

    // Pyannote
    let search_threshold = 0.5;
    let embedding_model_path = "wespeaker_en_voxceleb_CAM++.onnx";
    let segmentation_model_path = "segmentation-3.0.onnx";

    let (samples, sample_rate) = pyannote_rs::read_wav(&audio_path).unwrap();
    let mut embedding_extractor =
        pyannote_rs::EmbeddingExtractor::new(embedding_model_path).unwrap();
    let mut embedding_manager = pyannote_rs::EmbeddingManager::new(usize::MAX);

    // Store start time
    let now = Instant::now();

    // Get segments
    let segments = pyannote_rs::segment(&samples, sample_rate, segmentation_model_path).unwrap();

    for segment in segments {
        // Extract text
        // Collect samples into a Vec<f32>
        let sherpa_samples: Vec<f32> = segment
            .samples
            .iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .collect();
        let whisper_result = recognizer.transcribe(sample_rate as i32, sherpa_samples);
        let text = whisper_result.text;

        // Compute the embedding result
        let embedding_result: Vec<f32> = match embedding_extractor.compute(&segment.samples) {
            Ok(result) => result.collect(),
            Err(error) => {
                println!(
                    "[ERROR] in {:.2}s: {:.2}s: {:?}",
                    segment.start, segment.end, error
                );
                println!(
                    "start = {:.2}, end = {:.2}, speaker = ?, text = {}",
                    segment.start, segment.end, text
                );
                continue; // Skip to the next segment
            }
        };

        // Find the speaker
        let speaker = embedding_manager
            .search_speaker(embedding_result.clone(), search_threshold)
            .ok_or_else(|| embedding_manager.search_speaker(embedding_result, 0.0)) // Ensure always to return speaker
            .map(|r| r.to_string())
            .unwrap_or("?".into());

        println!(
            "start = {:.2}, end = {:.2}, speaker = {}, text = {}",
            segment.start, segment.end, speaker, text
        );

        // Show elapsed time
        let elapsed = now.elapsed();
        println!("Finished in {} seconds.", elapsed.as_secs());
    }
}
