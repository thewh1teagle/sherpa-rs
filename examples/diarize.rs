use std::io::Cursor;

/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

cargo run --example diarize ./sherpa-onnx-pyannote-segmentation-3-0/model.onnx ./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx ./0-four-speakers-zh.wav
*/

fn main() {
    let segment_model_path = std::env::args()
        .nth(1)
        .expect("Missing path argument for segmentation model");
    let embedding_model_path = std::env::args()
        .nth(2)
        .expect("Missing path argument for embedding model");
    let wav_path = std::env::args()
        .nth(3)
        .expect("Missing path argument for wav file");

    let config = sherpa_rs::diarize::DiarizeConfig {
        num_clusters: Some(5),
        ..Default::default()
    };

    let progress_callback = |n_computed_chunks: i32, n_total_chunks: i32| -> i32 {
        let progress = 100 * n_computed_chunks / n_total_chunks;
        println!("Progress: {}%", progress);
        0
    };

    let mut sd =
        sherpa_rs::diarize::Diarize::new(segment_model_path, embedding_model_path, config).unwrap();

    let audio_data = std::fs::read(wav_path).unwrap();
    let cursor = Cursor::new(audio_data);
    let mut reader = hound::WavReader::new(cursor).unwrap();
    let _sample_rate = reader.spec().sample_rate as i32;

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    let segments = sd
        .compute(samples, Some(Box::new(progress_callback)))
        .unwrap();
    for segment in segments {
        println!(
            "start = {} end = {} speaker = {}",
            segment.start, segment.end, segment.speaker
        );
    }
}
