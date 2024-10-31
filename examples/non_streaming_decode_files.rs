/*
Zh & En:
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
rm sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst

cargo run --example streaming_decode_files -- \
  --paraformer ./sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx \
  --tokens ./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt \
  --model-type paraformer \
  --rule-fsts ./itn_zh_number.fst \
  --debug 0 \
  ./itn-zh-number.wav
*/

use clap::{arg, Parser};
use sherpa_rs::common_config::FeatureConfig;
use sherpa_rs::recognizer::offline_recognizer::{
    OfflineLMConfig, OfflineModelConfig, OfflineNemoEncDecCtcModelConfig,
    OfflineParaformerModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineSenseVoiceModelConfig, OfflineTdnnModelConfig, OfflineTransducerModelConfig,
    OfflineWhisperModelConfig,
};
use sherpa_rs::stream::offline_stream::OfflineStream;

/// NonStreaming
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Please provide one wave file
    wave_file: String,

    /// Sample rate of the data used to train the model
    #[arg(long, default_value = "16000")]
    sample_rate: i32,

    /// Dimension of the features used to train the model
    #[arg(long, default_value = "80")]
    feat_dim: i32,

    /// Path to the transducer encoder model
    #[arg(long, default_value = "")]
    encoder: String,

    /// Path to the transducer decoder model
    #[arg(long, default_value = "")]
    decoder: String,

    /// Path to the transducer joiner model
    #[arg(long, default_value = "")]
    joiner: String,

    /// Path to the paraformer model
    #[arg(long, default_value = "")]
    paraformer: String,

    /// Path to the NeMo CTC model
    #[arg(long, default_value = "")]
    nemo_ctc: String,

    /// Path to the whisper encoder model
    #[arg(long, default_value = "")]
    whisper_encoder: String,

    /// Path to the whisper decoder model
    #[arg(long, default_value = "")]
    whisper_decoder: String,

    /// Language of the input wave. You can leave it empty
    #[arg(long, default_value = "")]
    whisper_language: String,

    /// transcribe or translate
    #[arg(long, default_value = "transcribe")]
    whisper_task: String,

    /// tail paddings for whisper
    #[arg(long, default_value = "-1")]
    whisper_tail_paddings: i32,

    /// Path to the tdnn model
    #[arg(long, default_value = "")]
    tdnn_model: String,

    /// Path to the SenseVoice model
    #[arg(long, default_value = "")]
    sense_voice_model: String,

    /// If not empty, specify the Language for the input wave
    #[arg(long, default_value = "")]
    sense_voice_language: String,

    ///  1 to use inverse text normalization
    #[arg(long, default_value = "1")]
    sense_voice_use_itn: i32,

    /// Path to the tokens file
    #[arg(long, default_value = "")]
    tokens: String,

    /// Number of threads for computing
    #[arg(long, default_value = "1")]
    num_threads: i32,

    /// Whether to show debug message
    #[arg(long, default_value = "0")]
    debug: i32,

    /// Optional. Used for loading the model in a faster way
    #[arg(long)]
    model_type: Option<String>,

    /// Provider to use
    #[arg(long, default_value = "cpu")]
    provider: String,

    /// cjkchar, bpe, cjkchar+bpe, or leave it to empty
    #[arg(long, default_value = "cjkchar")]
    modeling_unit: String,

    ///
    #[arg(long, default_value = "")]
    bpe_vocab: String,

    /// Used for TeleSpeechCtc model
    #[arg(long, default_value = "")]
    telespeech_ctc: String,

    /// Optional. Path to the LM model
    #[arg(long, default_value = "")]
    lm_model: String,

    /// Optional. Scale for the LM model
    #[arg(long, default_value = "1.0")]
    lm_scale: f32,

    /// Decoding method. Possible values: greedy_search, modified_beam_search
    #[arg(long, default_value = "greedy_search")]
    decoding_method: String,

    /// Used only when --decoding-method is modified_beam_search
    #[arg(long, default_value = "4")]
    max_active_paths: i32,

    /// If not empty, path to rule fst for inverse text normalization
    #[arg(long, default_value = "")]
    rule_fsts: String,

    /// If not empty, path to rule fst archives for inverse text normalization
    #[arg(long, default_value = "")]
    rule_fars: String,
}

fn main() {
    // Parse command-line arguments into `Args` struct
    let args = Args::parse();

    println!("Reading {}", args.wave_file);

    let (samples, sample_rate) = read_wave(&args.wave_file);

    println!("Initializing recognizer (may take several seconds)");
    let config = OfflineRecognizerConfig {
        feat_config: FeatureConfig {
            sample_rate: 16000,
            feature_dim: 80,
        },
        model_config: OfflineModelConfig {
            transducer: OfflineTransducerModelConfig {
                encoder: args.encoder,
                decoder: args.decoder,
                joiner: args.joiner,
            },
            paraformer: OfflineParaformerModelConfig {
                model: args.paraformer,
            },
            nemo_ctc: OfflineNemoEncDecCtcModelConfig {
                model: args.nemo_ctc,
            },
            whisper: OfflineWhisperModelConfig {
                encoder: args.whisper_encoder,
                decoder: args.whisper_decoder,
                language: args.whisper_language,
                task: args.whisper_task,
                tail_paddings: args.whisper_tail_paddings,
            },
            tdnn: OfflineTdnnModelConfig {
                model: args.tdnn_model,
            },
            sense_voice: OfflineSenseVoiceModelConfig {
                model: args.sense_voice_model,
                language: args.sense_voice_language,
                use_inverse_text_normalization: args.sense_voice_use_itn,
            },
            tokens: args.tokens,
            num_threads: args.num_threads,
            provider: Some(args.provider),
            debug: args.debug,
            model_type: args.model_type,
            modeling_unit: Some(args.modeling_unit),
            bpe_vocab: Some(args.bpe_vocab),
            tele_speech_ctc: Some(args.telespeech_ctc),
        },
        lm_config: OfflineLMConfig {
            model: args.lm_model,
            scale: args.lm_scale,
        },
        decoding_method: args.decoding_method,
        max_active_paths: args.max_active_paths,
        hotwords_file: "".to_string(),
        hotwords_score: 0.0,
        blank_penalty: 0.0,
        rule_fsts: args.rule_fsts,
        rule_fars: args.rule_fars,
    };

    let recognizer = OfflineRecognizer::new(&config);
    println!("Recognizer created!");

    println!("Start decoding!");
    let mut stream = OfflineStream::new(&recognizer);

    stream.accept_waveform(sample_rate, &samples);

    recognizer.decode(&stream);
    println!("Decoding done!");
    let result = stream.get_result().unwrap();

    println!("Text: {}", result.text.to_lowercase());
    println!("Emotion: {}", result.emotion);
    println!("Lang: {}", result.lang);
    println!("Event: {}", result.event);

    for v in &result.timestamps {
        println!("Timestamp: {:?}", v);
    }

    for v in &result.tokens {
        println!("Token: {}", v);
    }

    println!(
        "Wave duration: {} seconds",
        samples.len() as f32 / sample_rate as f32
    );
}

/// Reads a WAV file and returns the samples and sample rate
///
/// # Parameters
///
/// * `filename` - Path to the WAV file
///
/// # Returns
///
/// * `samples` - Sample data
/// * `sample_rate` - Sample rate
fn read_wave(filename: &str) -> (Vec<f32>, i32) {
    let mut reader = hound::WavReader::open(filename).expect("Failed to open WAV file");
    let spec = reader.spec();

    if spec.sample_format != hound::SampleFormat::Int {
        panic!("Support only PCM format. Given: {:?}", spec.sample_format);
    }

    if spec.channels != 1 {
        panic!("Support only 1 channel wave file. Given: {}", spec.channels);
    }

    if spec.bits_per_sample != 16 {
        panic!(
            "Support only 16-bit per sample. Given: {}",
            spec.bits_per_sample
        );
    }

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    (samples, spec.sample_rate as i32)
}
