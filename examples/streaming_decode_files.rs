/*
Zh & En:
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst

cargo run --example streaming_decode_files -- \
  --encoder ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx \
  --decoder ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
  --joiner ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx \
  --tokens ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
  --model-type zipformer \
  --rule-fsts ./itn_zh_number.fst \
  --debug 0 \
  ./itn-zh-number.wav
*/

use clap::{arg, Parser};
use sherpa_rs::common_config::FeatureConfig;
use sherpa_rs::recognizer::online_recognizer::{
    OnlineCtcFstDecoderConfig, OnlineModelConfig, OnlineParaformerModelConfig, OnlineRecognizer,
    OnlineRecognizerConfig, OnlineTransducerModelConfig, OnlineZipformer2CtcModelConfig,
};
use sherpa_rs::stream::online_stream::OnlineStream;

/// Streaming
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Please provide one wave file
    wave_file: String,

    /// Path to the transducer encoder model
    #[arg(long, default_value = "")]
    encoder: String,

    /// Path to the transducer decoder model
    #[arg(long, default_value = "")]
    decoder: String,

    /// Path to the transducer joiner model
    #[arg(long, default_value = "")]
    joiner: String,

    /// Path to the paraformer encoder model
    #[arg(long, default_value = "")]
    paraformer_encoder: String,

    /// Path to the paraformer decoder model
    #[arg(long, default_value = "")]
    paraformer_decoder: String,

    /// Path to the zipformer2 CTC model
    #[arg(long, default_value = "")]
    zipformer2_ctc: String,

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
    let config = OnlineRecognizerConfig {
        feat_config: FeatureConfig {
            sample_rate: 16000,
            feature_dim: 80,
        },
        model_config: OnlineModelConfig {
            transducer: OnlineTransducerModelConfig {
                encoder: args.encoder,
                decoder: args.decoder,
                joiner: args.joiner,
            },
            paraformer: OnlineParaformerModelConfig {
                encoder: args.paraformer_encoder,
                decoder: args.paraformer_decoder,
            },
            zipformer2_ctc: OnlineZipformer2CtcModelConfig {
                model: args.zipformer2_ctc,
            },
            tokens: args.tokens,
            num_threads: args.num_threads,
            provider: Some(args.provider),
            debug: args.debug,
            model_type: args.model_type,
            modeling_unit: None,
            bpe_vocab: None,
            tokens_buf: None,
            tokens_buf_size: None,
        },
        decoding_method: args.decoding_method,
        max_active_paths: args.max_active_paths,
        enable_endpoint: 0,
        rule1_min_trailing_silence: 0.0,
        rule2_min_trailing_silence: 0.0,
        rule3_min_utterance_length: 0.0,
        hotwords_file: "".to_string(),
        hotwords_score: 0.0,
        blank_penalty: 0.0,
        ctc_fst_decoder_config: OnlineCtcFstDecoderConfig::default(),
        rule_fsts: args.rule_fsts,
        rule_fars: args.rule_fars,
        hotwords_buf: "".to_string(),
        hotwords_buf_size: 0,
    };

    let recognizer = OnlineRecognizer::new(&config);
    println!("Recognizer created!");

    println!("Start decoding!");
    let stream = OnlineStream::new(&recognizer);

    stream.accept_waveform(sample_rate, &samples);

    let tail_padding = vec![0.0; (sample_rate as f32 * 0.3) as usize];
    stream.accept_waveform(sample_rate, &tail_padding);

    while recognizer.is_ready(&stream) {
        recognizer.decode(&stream);
    }
    println!("Decoding done!");

    let result = recognizer.get_result(&stream);
    println!("{}", result.text.to_lowercase());
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
