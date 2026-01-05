/*
转录测试 - 测试 FunASR-Nano 和 Streaming API

使用方法:
1. 首先运行 download_models.sh 下载模型
2. 运行测试:
   cargo run --example transcribe_test -- /path/to/audio.wav

或指定测试类型:
   cargo run --example transcribe_test -- /path/to/audio.wav funasr
   cargo run --example transcribe_test -- /path/to/audio.wav streaming
   cargo run --example transcribe_test -- /path/to/audio.wav all
*/

use std::time::Instant;

/// 读取音频文件并重采样到 16000 Hz
fn read_audio_file_resample(path: &str) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    // 读取所有样本
    let samples: Vec<f32> = if spec.bits_per_sample == 16 {
        reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect()
    } else if spec.bits_per_sample == 32 {
        if spec.sample_format == hound::SampleFormat::Float {
            reader.samples::<f32>().map(|s| s.unwrap()).collect()
        } else {
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / i32::MAX as f32)
                .collect()
        }
    } else {
        return Err(format!("Unsupported bits per sample: {}", spec.bits_per_sample).into());
    };

    // 转换为单声道（如果是立体声）
    let mono_samples: Vec<f32> = if channels > 1 {
        samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    // 重采样到 16000 Hz（如果需要）
    let target_rate = 16000u32;
    let resampled = if sample_rate != target_rate {
        resample(&mono_samples, sample_rate, target_rate)
    } else {
        mono_samples
    };

    Ok((resampled, target_rate))
}

/// 简单的线性重采样
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    let ratio = from_rate as f64 / to_rate as f64;
    let new_len = (samples.len() as f64 / ratio) as usize;
    let mut resampled = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_idx = i as f64 * ratio;
        let idx = src_idx as usize;
        let frac = src_idx - idx as f64;

        let sample = if idx + 1 < samples.len() {
            samples[idx] * (1.0 - frac as f32) + samples[idx + 1] * frac as f32
        } else {
            samples[idx.min(samples.len() - 1)]
        };
        resampled.push(sample);
    }

    resampled
}

fn test_funasr_nano(samples: &[f32], sample_rate: u32) {
    use sherpa_rs::funasr_nano::{FunasrNanoConfig, FunasrNanoRecognizer};

    println!("\n{:=^60}", " FunASR-Nano 测试 ");
    
    // 打印当前工作目录
    println!("当前工作目录: {:?}", std::env::current_dir().unwrap());
    
    // 检查模型文件是否存在
    let model_files = [
        "funasr-nano/encoder_adaptor.int8.onnx",
        "funasr-nano/llm_prefill.int8.onnx",
        "funasr-nano/llm_decode.int8.onnx",
        "funasr-nano/embedding.int8.onnx",
        "funasr-nano/Qwen3-0.6B/vocab.json",
        "funasr-nano/Qwen3-0.6B/merges.txt",
    ];
    
    println!("检查模型文件...");
    for file in &model_files {
        let exists = std::path::Path::new(file).exists();
        println!("  {} - {}", if exists { "✅" } else { "❌" }, file);
        if !exists {
            eprintln!("❌ 模型文件不存在: {}", file);
            eprintln!("   请运行 ./download_models.sh 下载模型");
            return;
        }
    }
    
    let config = FunasrNanoConfig {
        encoder_adaptor: "funasr-nano/encoder_adaptor.int8.onnx".into(),
        llm_prefill: "funasr-nano/llm_prefill.int8.onnx".into(),
        llm_decode: "funasr-nano/llm_decode.int8.onnx".into(),
        embedding: "funasr-nano/embedding.int8.onnx".into(),
        // tokenizer 是目录路径，需要包含 vocab.json, merges.txt
        tokenizer: "funasr-nano/Qwen3-0.6B".into(),
        // 必须设置 prompt
        system_prompt: Some("You are a helpful assistant.".into()),
        user_prompt: Some("语音转写：".into()),
        num_threads: Some(4),
        debug: true,
        ..Default::default()
    };

    println!("创建识别器...");
    let mut recognizer = match FunasrNanoRecognizer::new(config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("❌ 创建 FunASR-Nano 识别器失败: {}", e);
            eprintln!("   请确保已运行 download_models.sh 下载模型");
            return;
        }
    };

    println!("开始转录...");
    let start = Instant::now();
    let result = recognizer.transcribe(sample_rate, samples);
    let elapsed = start.elapsed();

    println!("✅ 转录完成!");
    println!("📝 文本: {}", result.text);
    println!("⏱️  耗时: {:?}", elapsed);
    println!(
        "📊 实时率: {:.2}x",
        (samples.len() as f64 / sample_rate as f64) / elapsed.as_secs_f64()
    );
}

fn test_streaming(samples: &[f32], sample_rate: u32) {
    use sherpa_rs::streaming::{
        EndpointConfig, OnlineModelType, OnlineRecognizer, OnlineRecognizerConfig,
        OnlineTransducerModelConfig,
    };

    println!("\n{:=^60}", " Streaming API 测试 ");

    let model_dir = "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20";
    
    let config = OnlineRecognizerConfig {
        model: OnlineModelType::Transducer(OnlineTransducerModelConfig {
            encoder: format!("{}/encoder-epoch-99-avg-1.onnx", model_dir),
            decoder: format!("{}/decoder-epoch-99-avg-1.onnx", model_dir),
            joiner: format!("{}/joiner-epoch-99-avg-1.onnx", model_dir),
        }),
        tokens: format!("{}/tokens.txt", model_dir),
        num_threads: Some(4),
        endpoint: EndpointConfig {
            enable: true,
            rule1_min_trailing_silence: 2.4,
            rule2_min_trailing_silence: 1.2,
            rule3_min_utterance_length: 20.0,
        },
        debug: false,
        ..Default::default()
    };

    println!("创建流式识别器...");
    let recognizer = match OnlineRecognizer::new(config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("❌ 创建 Streaming 识别器失败: {}", e);
            eprintln!("   请确保已运行 download_models.sh 下载模型");
            return;
        }
    };

    let mut stream = match recognizer.create_stream() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("❌ 创建流失败: {}", e);
            return;
        }
    };

    println!("开始流式转录...");
    let start = Instant::now();

    // 模拟流式输入，每次处理 100ms 的音频
    let chunk_size = (sample_rate as usize) / 10; // 100ms
    let mut last_text = String::new();
    let mut segment_count = 0;

    for (i, chunk) in samples.chunks(chunk_size).enumerate() {
        stream.accept_waveform(sample_rate as i32, chunk);

        while stream.is_ready() {
            stream.decode();
        }

        let result = stream.get_result();
        if !result.text.is_empty() && result.text != last_text {
            let time_offset = (i * chunk_size) as f32 / sample_rate as f32;
            println!("[{:>6.2}s] {}", time_offset, result.text);
            last_text = result.text;
        }

        if stream.is_endpoint() {
            let result = stream.get_result();
            if !result.text.is_empty() {
                segment_count += 1;
                println!("\n  📍 端点 #{}: {}\n", segment_count, result.text);
            }
            stream.reset();
            last_text.clear();
        }
    }

    // 处理剩余音频
    stream.input_finished();
    while stream.is_ready() {
        stream.decode();
    }

    let final_result = stream.get_result();
    let elapsed = start.elapsed();

    println!("\n✅ 流式转录完成!");
    if !final_result.text.is_empty() {
        println!("📝 最终文本: {}", final_result.text);
    }
    println!("📊 检测到 {} 个端点", segment_count);
    println!("⏱️  总耗时: {:?}", elapsed);
    println!(
        "📊 实时率: {:.2}x",
        (samples.len() as f64 / sample_rate as f64) / elapsed.as_secs_f64()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        eprintln!("用法: {} <音频文件路径> [funasr|streaming|all]", args[0]);
        eprintln!("\n示例:");
        eprintln!("  {} audio.wav            # 运行所有测试", args[0]);
        eprintln!("  {} audio.wav sense      # 仅测试 SenseVoice", args[0]);
        eprintln!("  {} audio.wav funasr     # 仅测试 FunASR-Nano", args[0]);
        eprintln!("  {} audio.wav streaming  # 仅测试 Streaming", args[0]);
        std::process::exit(1);
    }

    let audio_path = &args[1];
    let test_type = args.get(2).map(|s| s.as_str()).unwrap_or("all");

    println!("{:=^60}", " 音频转录测试 ");
    println!("📁 音频文件: {}", audio_path);

    // 读取音频文件（自动重采样到 16000 Hz）
    println!("\n读取音频文件...");
    let (samples, sample_rate) = match read_audio_file_resample(audio_path) {
        Ok((s, sr)) => (s, sr),
        Err(e) => {
            eprintln!("❌ 读取音频文件失败: {}", e);
            std::process::exit(1);
        }
    };

    let duration = samples.len() as f32 / sample_rate as f32;
    println!("✅ 音频加载成功");
    println!("   采样率: {} Hz", sample_rate);
    println!("   样本数: {}", samples.len());
    println!("   时长: {:.2} 秒", duration);

    // 根据测试类型运行测试
    match test_type {
        "funasr" => {
            test_funasr_nano(&samples, sample_rate);
        }
        "streaming" => {
            test_streaming(&samples, sample_rate);
        }
        "sense" | "sensevoice" => {
            test_sense_voice(&samples, sample_rate);
        }
        "all" | _ => {
            test_sense_voice(&samples, sample_rate);
            test_funasr_nano(&samples, sample_rate);
            test_streaming(&samples, sample_rate);
        }
    }

    println!("\n{:=^60}", " 测试完成 ");
}

fn test_sense_voice(samples: &[f32], sample_rate: u32) {
    use sherpa_rs::sense_voice::{SenseVoiceConfig, SenseVoiceRecognizer};

    println!("\n{:=^60}", " SenseVoice 测试 ");

    let model_dir = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17";

    // 检查模型文件
    let model_path = format!("{}/model.int8.onnx", model_dir);
    let tokens_path = format!("{}/tokens.txt", model_dir);

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("❌ 模型文件不存在: {}", model_path);
        eprintln!("   请运行以下命令下载模型:");
        eprintln!("   curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2");
        eprintln!("   tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2");
        return;
    }

    let config = SenseVoiceConfig {
        model: model_path,
        tokens: tokens_path,
        language: "auto".into(),
        use_itn: true,
        num_threads: Some(4),
        debug: true,
        ..Default::default()
    };

    println!("创建识别器...");
    let mut recognizer = match SenseVoiceRecognizer::new(config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("❌ 创建 SenseVoice 识别器失败: {}", e);
            return;
        }
    };

    println!("开始转录...");
    let start = std::time::Instant::now();
    let result = recognizer.transcribe(sample_rate, samples);
    let elapsed = start.elapsed();

    println!("✅ 转录完成!");
    println!("📝 文本: {}", result.text);
    println!("⏱️  耗时: {:?}", elapsed);
    println!(
        "📊 实时率: {:.2}x",
        (samples.len() as f64 / sample_rate as f64) / elapsed.as_secs_f64()
    );
}

