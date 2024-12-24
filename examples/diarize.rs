/*
Diarize audio file with pyannote-audio for segmentation (start and stop marks) and 3dspeaker for speaker recognition.

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

cargo run --example diarize ./sherpa-onnx-pyannote-segmentation-3-0/model.onnx ./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx ./0-four-speakers-zh.wav
*/

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (segment_model_path, embedding_model_path, wav_path) = (
        args.get(1)
            .expect("Missing path argument for segmentation model"),
        args.get(2)
            .expect("Missing path argument for embedding model"),
        args.get(3).expect("Missing path argument for wav file"),
    );

    let config = sherpa_rs::diarize::DiarizeConfig {
        num_clusters: Some(5),
        ..Default::default()
    };

    let progress_callback = |n_computed_chunks: i32, n_total_chunks: i32| -> i32 {
        let progress = 100 * n_computed_chunks / n_total_chunks;
        println!("🗣️ Diarizing... {}% 🎯", progress);
        0
    };

    let mut sd =
        sherpa_rs::diarize::Diarize::new(segment_model_path, embedding_model_path, config).unwrap();
    let (samples, _) = sherpa_rs::read_audio_file(wav_path).unwrap();

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
