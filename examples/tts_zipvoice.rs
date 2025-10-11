/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-zh-en-emilia.tar.bz2
tar xf sherpa-onnx-zipvoice-distill-zh-en-emilia.tar.bz2
rm sherpa-onnx-zipvoice-distill-zh-en-emilia.tar.bz2
cargo run --example tts_zipvoice
*/
use hound;
use sherpa_rs::{
    tts::{CommonTtsConfig, ZipVoiceTts, ZipVoiceTtsConfig},
    OnnxConfig,
};

fn main() {
    let model_dir = "./sherpa-onnx-zipvoice-distill-zh-en-emilia";

    // Load prompt audio
    let prompt_path = format!("{}/prompt.wav", model_dir);
    let mut reader = hound::WavReader::open(&prompt_path).expect("Failed to open prompt.wav");
    let spec = reader.spec();
    let prompt_sr = spec.sample_rate as i32;
    
    println!("Prompt audio spec: {:?}", spec);

    // Read samples and convert to f32 (mono only for now)
    let mut prompt_samples: Vec<f32> = if spec.sample_format == hound::SampleFormat::Int {
        reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect()
    } else {
        reader.samples::<f32>().map(|s| s.unwrap()).collect()
    };
    
    // If stereo, convert to mono by averaging channels
    if spec.channels == 2 {
        prompt_samples = prompt_samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect();
    }
    
    println!("Loaded {} prompt samples at {} Hz", prompt_samples.len(), prompt_sr);

    let config = ZipVoiceTtsConfig {
        tokens: format!("{}/tokens.txt", model_dir),
        text_model: format!("{}/text_encoder.onnx", model_dir),
        flow_matching_model: format!("{}/fm_decoder.onnx", model_dir),
        vocoder: format!("{}/vocos_24khz.onnx", model_dir),
        data_dir: format!("{}/espeak-ng-data", model_dir),
        pinyin_dict: format!("{}/pinyin.raw", model_dir),
        feat_scale: 0.1,
        t_shift: 0.5, // Match the default
        target_rms: 0.1,
        guidance_scale: 3.0,
        common_config: CommonTtsConfig {
            silence_scale: 1.0,
            ..Default::default()
        },
        onnx_config: OnnxConfig {
            debug: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut tts = ZipVoiceTts::new(config);

    let prompt_text = ""; // Can be empty if no prompt text
    let text = "Hello world."; // Start with simple English text
    let speed = 1.0;
    let num_steps = 4; // Number of inference steps
    
    println!("Generating speech for: {}", text);

    let audio = tts
        .create(
            &text,
            &prompt_text,
            &prompt_samples,
            prompt_sr,
            speed,
            num_steps,
        )
        .unwrap();

    dbg!(audio.sample_rate);
    sherpa_rs::write_audio_file("zipvoice_audio.wav", &audio.samples, audio.sample_rate).unwrap();
    println!(
        "Created zipvoice_audio.wav with {} samples",
        audio.samples.len()
    );
}
