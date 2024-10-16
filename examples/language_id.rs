/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.tar.bz2
rm sherpa-onnx-whisper-tiny.tar.bz2
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/16hz_mono_pcm_s16le.wav -O 16hz_mono_pcm_s16le.wav
cargo run --example language_id 16hz_mono_pcm_s16le.wav
*/

use eyre::{bail, Result};
use sherpa_rs::language_id;
use std::io::Cursor;

fn main() -> Result<()> {
    let file_path = std::env::args().nth(1).expect("Missing file path argument");
    let audio_data = std::fs::read(file_path)?;

    let cursor = Cursor::new(audio_data);
    let mut reader = hound::WavReader::new(cursor)?;
    let sample_rate = reader.spec().sample_rate as i32;

    if sample_rate != 16000 {
        bail!("The sample rate must be 16000.");
    }

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    let config = language_id::SpokenLanguageIdConfig {
        encoder: "sherpa-onnx-whisper-tiny/tiny-encoder.onnx".into(),
        decoder: "sherpa-onnx-whisper-tiny/tiny-decoder.onnx".into(),
        ..Default::default()
    };
    let mut extractor = language_id::SpokenLanguageId::new(config);

    let language = extractor.compute(samples, sample_rate)?;
    println!("Spoken language: {}", language);

    Ok(())
}
