/*
Piper English model
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
tar xf vits-piper-en_US-amy-low.tar.bz2
cargo run --example tts --features="tts" -- --text "liliana, the most beautiful and lovely assistant of our team!" --output audio.wav --tokens "vits-piper-en_US-amy-low/tokens.txt" --model "vits-piper-en_US-amy-low/en_US-amy-low.onnx" --data-dir "vits-piper-en_US-amy-low/espeak-ng-data"

High quality vits-ljs with emotions voice
wget https://huggingface.co/csukuangfj/vits-ljs/resolve/main/vits-ljs.onnx
wget https://huggingface.co/csukuangfj/vits-ljs/resolve/main/lexicon.txt
wget https://huggingface.co/csukuangfj/vits-ljs/resolve/main/tokens.txt
cargo run --example tts --features="tts" -- --text "liliana, the most beautiful and lovely assistant of our team!"" --output audio.wav --tokens "tokens.txt" --model "vits-ljs.onnx" --lexicon lexicon.txt

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

    let vits_cfg = sherpa_rs::tts::TtsVitsModelConfig::new(
        args.model,
        args.lexicon.unwrap_or_default(),
        args.tokens,
        args.data_dir.unwrap_or_default(),
        0.0,
        0.0,
        args.dict_dir.unwrap_or_default(),
        0.0,
    );
    let max_num_sentences = 2;
    let model_cfg =
        sherpa_rs::tts::OfflineTtsModelConfig::new(args.debug, vits_cfg, args.provider, 1);
    let tts_cfg =
        sherpa_rs::tts::OfflineTtsConfig::new(model_cfg, max_num_sentences, "".into(), "".into());
    let mut tts = sherpa_rs::tts::OfflineTts::new(tts_cfg);
    let speed = 1.0;
    let audio = tts.generate(text, 0, speed).unwrap();
    audio.write_to_wav(&args.output).unwrap(); // Use the provided output path
    println!("Created {}", args.output);
}
