/*
Detect speech in audio file and segment it (mark start and slow time)

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/motivation.wav -O motivation.wav
cargo run --example vad_segment motivation.wav
*/
use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};

fn main() {
    let file_path = std::env::args().nth(1).expect("Missing file path argument");
    let (mut samples, sample_rate) = sherpa_rs::read_audio_file(&file_path).unwrap();
    assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");

    let window_size: usize = 512;
    let config = SileroVadConfig {
        model: "silero_vad.onnx".into(),
        window_size: window_size as i32,
        ..Default::default()
    };

    let mut vad = SileroVad::new(config, 3.0).unwrap();
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
}
