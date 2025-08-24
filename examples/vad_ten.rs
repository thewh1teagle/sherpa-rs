/*
Detect voice in audio file and mark start and stop.

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_speakerverification_speakernet.onnx
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/motivation.wav -O motivation.wav
cargo run --example vad_ten motivation.wav
*/
use sherpa_rs::ten_vad::{TenVad, TenVadConfig};

fn process_speech_segment(vad: &mut TenVad, sample_rate: u32) {
    while !vad.is_empty() {
        let segment = vad.front();
        let start_sec = (segment.start as f32) / sample_rate as f32;
        let duration_sec = (segment.samples.len() as f32) / sample_rate as f32;

        println!("start={}s end={}s", start_sec, start_sec + duration_sec);
        vad.pop();
    }
}

fn main() {
    let file_path = std::env::args().nth(1).expect("Missing file path argument");

    let (mut samples, sample_rate) = sherpa_rs::read_audio_file(&file_path).unwrap();
    assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");

    // Pad with 3 seconds of silence so vad will be able to detect stop
    samples.extend(vec![0.0; (3 * sample_rate) as usize]);

    let window_size = 256;
    let vad_config = TenVadConfig {
        model: "ten-vad.onnx".into(),
        window_size: window_size as i32,
        ..Default::default()
    };

    let mut vad = TenVad::new(vad_config, 60.0).unwrap();
    let mut index = 0;
    while index + window_size <= samples.len() {
        let window = &samples[index..index + window_size];
        vad.accept_waveform(window.to_vec()); // Convert slice to Vec
        if vad.is_speech() {
            while !vad.is_empty() {
                process_speech_segment(&mut vad, sample_rate);
            }
        }

        index += window_size;
    }
    vad.flush();
    // process reamaining
    while !vad.is_empty() {
        process_speech_segment(&mut vad, sample_rate);
    }
}
