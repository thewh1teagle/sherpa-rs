/*
Detect speech in audio file and segment it (mark start and slow time)

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/motivation.wav -O motivation.wav
cargo run --example vad_segment motivation.wav
*/
use eyre::{bail, Result};
use sherpa_rs::vad::{Vad, VadConfig};

fn main() -> Result<()> {
    let file_path = std::env::args().nth(1).expect("Missing file path argument");
    let mut reader = hound::WavReader::open(file_path)?;
    let sample_rate = reader.spec().sample_rate as i32;

    if sample_rate != 16000 {
        bail!("The sample rate must be 16000.");
    }

    let mut samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    let window_size: usize = 512;
    let config = VadConfig {
        model: "silero_vad.onnx".into(),
        window_size: window_size as i32,
        ..Default::default()
    };

    let mut vad = Vad::new(config, 3.0).unwrap();
    while samples.len() > window_size {
        let window = &samples[..window_size];
        vad.accept_waveform(window.to_vec()); // Convert slice to Vec
        if vad.is_speech() {
            while !vad.is_empty() {
                let segment = vad.front();
                let start_sec = (segment.start as f32) / sample_rate as f32;
                let duration_sec = (segment.samples.len() as f32) / sample_rate as f32;
                println!("start={}s duration={}s", start_sec, duration_sec);
                vad.pop();
            }
        }
        samples = samples[window_size..].to_vec(); // Move the remaining samples to the next iteration
    }
    Ok(())
}
