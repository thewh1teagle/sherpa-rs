/*
Convert text to speech

Piper English model
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
tar xf vits-piper-en_US-amy-low.tar.bz2
cargo run --example tts --features="tts" -- --text 'liliana, the most beautiful and lovely assistant of our team!' --output audio.wav --tokens "vits-piper-en_US-amy-low/tokens.txt" --model "vits-piper-en_US-amy-low/en_US-amy-low.onnx" --data-dir "vits-piper-en_US-amy-low/espeak-ng-data"

High quality vits-ljs with emotions voice
wget https://huggingface.co/csukuangfj/vits-ljs/resolve/main/vits-ljs.onnx
wget https://huggingface.co/csukuangfj/vits-ljs/resolve/main/lexicon.txt
wget https://huggingface.co/csukuangfj/vits-ljs/resolve/main/tokens.txt
cargo run --example tts --features="tts" -- --text "liliana, the most beautiful and lovely assistant of our team!" --output audio.wav --tokens "tokens.txt" --model "vits-ljs.onnx" --lexicon lexicon.txt

MMS Hebrew model
wget https://huggingface.co/thewh1teagle/mms-tts-heb/resolve/main/model_sherpa.onnx
wget https://huggingface.co/thewh1teagle/mms-tts-heb/resolve/main/tokens.txt
cargo run --example tts --features="tts" -- --text "שלום וברכה, ניפרד בשמחה" --output audio.wav --tokens "tokens.txt" --model "model_sherpa.onnx"
*/
use clap::Parser;

/// TTS
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    tokens: String,

    #[arg(short, long)]
    model: String,

    #[arg(long)]
    text: Option<String>,

    #[arg(long)]
    text_file_input: Option<String>,

    #[arg(short, long)]
    output: String,

    #[arg(long)]
    dict_dir: Option<String>,

    #[arg(long)]
    data_dir: Option<String>,

    #[arg(long)]
    lexicon: Option<String>,

    #[arg(long)]
    sid: Option<i32>,

    #[arg(long)]
    speed: Option<f32>,

    #[arg(long)]
    max_num_sentences: Option<i32>,

    #[arg(long)]
    provider: Option<String>,

    #[arg(long)]
    debug: bool,
}

fn main() {
    // Parse command-line arguments into `Args` struct
    let args = Args::parse();
    let text;
    if args.text.is_some() {
        text = args.text.unwrap();
    } else {
        text = std::fs::read_to_string(args.text_file_input.unwrap()).unwrap();
    }

    let vits_config = sherpa_rs::tts::VitsConfig {
        lexicon: args.lexicon.unwrap_or_default(),
        tokens: args.tokens,
        data_dir: args.data_dir.unwrap_or_default(),
        dict_dir: args.dict_dir.unwrap_or_default(),
        ..Default::default()
    };

    let max_num_sentences = args.max_num_sentences.unwrap_or(2);
    let tts_config = sherpa_rs::tts::OfflineTtsConfig {
        model: args.model,
        max_num_sentences,
        ..Default::default()
    };
    let mut tts = sherpa_rs::tts::OfflineTts::new(tts_config, vits_config);
    let speed = args.speed.unwrap_or(1.0);
    let sid = args.sid.unwrap_or(0);
    let audio = tts.generate(text, sid, speed).unwrap();
    audio.write_to_wav(&args.output).unwrap(); // Use the provided output path
    println!("Created {}", args.output);
}
