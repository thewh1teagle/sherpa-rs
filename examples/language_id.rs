/*
Detect language spoken in audio file

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.tar.bz2
rm sherpa-onnx-whisper-tiny.tar.bz2
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/16hz_mono_pcm_s16le.wav -O 16hz_mono_pcm_s16le.wav
cargo run --example language_id 16hz_mono_pcm_s16le.wav
*/

fn main() {
    let file_path = std::env::args().nth(1).expect("Missing file path argument");

    let (samples, sample_rate) = sherpa_rs::read_audio_file(&file_path).unwrap();
    assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");

    let config = sherpa_rs::language_id::SpokenLanguageIdConfig {
        encoder: "sherpa-onnx-whisper-tiny/tiny-encoder.onnx".into(),
        decoder: "sherpa-onnx-whisper-tiny/tiny-decoder.onnx".into(),
        ..Default::default()
    };
    let mut extractor = sherpa_rs::language_id::SpokenLanguageId::new(config);

    let language = extractor.compute(samples, sample_rate).unwrap();
    println!("Spoken language: {}", language);
}
