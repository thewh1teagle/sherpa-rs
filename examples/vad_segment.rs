/// wget https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx
/// cargo run --example vad_segment
use eyre::{bail, Result};
use sherpa_rs::vad::{Vad, VadConfig};
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

    let model = "silero_vad.onnx".into();
    let window_size: usize = 512;
    let config = VadConfig::new(
        model,
        0.5,
        0.5,
        0.5,
        sample_rate,
        window_size.try_into().unwrap(),
        None,
        None,
        Some(true),
    );

    let mut vad = Vad::new_from_config(config, 3.0).unwrap();
    while samples.len() > window_size {
        let window = &samples[..window_size];
        vad.accept_waveform(window.to_vec()); // Convert slice to Vec
        if vad.is_speech() {
            while !vad.is_empty() {
                let segment = vad.front();
                // let start = segment.start / sample_rate;
                // let duration = segment.samples.len() as i32 / sample_rate;
                let start_seconds = (segment.start as f32) / sample_rate as f32;
                let duration_seconds = (segment.samples.len() as f32) / sample_rate as f32;
                println!("start={}s duration={}s", start_seconds, duration_seconds);
                vad.pop();
            }
        }
        samples = samples[window_size..].to_vec(); // Move the remaining samples to the next iteration
    }
    Ok(())
}
