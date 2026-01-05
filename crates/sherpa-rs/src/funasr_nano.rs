use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::mem;

#[derive(Debug)]
pub struct FunasrNanoRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

pub type FunasrNanoRecognizerResult = super::OfflineRecognizerResult;

#[derive(Debug, Clone)]
pub struct FunasrNanoConfig {
    /// Path to encoder_adaptor.onnx
    pub encoder_adaptor: String,
    /// Path to llm_prefill.onnx
    pub llm_prefill: String,
    /// Path to llm_decode.onnx
    pub llm_decode: String,
    /// Path to embedding.onnx
    pub embedding: String,
    /// Path to tokenizer directory (containing vocab.json, merges.txt)
    pub tokenizer: String,

    /// System prompt for the model
    pub system_prompt: Option<String>,
    /// User prompt for the model
    pub user_prompt: Option<String>,

    /// Maximum number of new tokens to generate
    pub max_new_tokens: Option<i32>,
    /// Temperature for sampling (0.0 for greedy)
    pub temperature: Option<f32>,
    /// Top-p (nucleus) sampling parameter
    pub top_p: Option<f32>,
    /// Random seed for reproducibility
    pub seed: Option<i32>,

    /// Provider (cpu, cuda, coreml, etc.)
    pub provider: Option<String>,
    /// Number of threads
    pub num_threads: Option<i32>,
    /// Enable debug mode
    pub debug: bool,
}

impl Default for FunasrNanoConfig {
    fn default() -> Self {
        Self {
            encoder_adaptor: String::new(),
            llm_prefill: String::new(),
            llm_decode: String::new(),
            embedding: String::new(),
            tokenizer: String::new(),
            // 必须提供 prompt，否则模型无法正确工作
            system_prompt: Some("You are a helpful assistant.".into()),
            user_prompt: Some("语音转写：".into()),
            max_new_tokens: Some(512),
            temperature: Some(0.3),
            top_p: Some(0.8),
            seed: Some(42),
            provider: None,
            num_threads: Some(4),
            debug: false,
        }
    }
}

impl FunasrNanoRecognizer {
    pub fn new(config: FunasrNanoConfig) -> Result<Self> {
        let debug = config.debug.into();
        let provider = config.provider.unwrap_or(get_default_provider());
        let provider_ptr = cstring_from_str(&provider);
        let num_threads = config.num_threads.unwrap_or(1);

        // FunASR-Nano specific config
        let encoder_adaptor_ptr = cstring_from_str(&config.encoder_adaptor);
        let llm_prefill_ptr = cstring_from_str(&config.llm_prefill);
        let llm_decode_ptr = cstring_from_str(&config.llm_decode);
        let embedding_ptr = cstring_from_str(&config.embedding);
        let tokenizer_ptr = cstring_from_str(&config.tokenizer);
        let system_prompt_ptr =
            cstring_from_str(&config.system_prompt.clone().unwrap_or_default());
        let user_prompt_ptr = cstring_from_str(&config.user_prompt.clone().unwrap_or_default());

        let funasr_nano_config = sherpa_rs_sys::SherpaOnnxOfflineFunASRNanoModelConfig {
            encoder_adaptor: encoder_adaptor_ptr.as_ptr(),
            llm_prefill: llm_prefill_ptr.as_ptr(),
            llm_decode: llm_decode_ptr.as_ptr(),
            embedding: embedding_ptr.as_ptr(),
            tokenizer: tokenizer_ptr.as_ptr(),
            system_prompt: system_prompt_ptr.as_ptr(),
            user_prompt: user_prompt_ptr.as_ptr(),
            max_new_tokens: config.max_new_tokens.unwrap_or(200),
            temperature: config.temperature.unwrap_or(0.0),
            top_p: config.top_p.unwrap_or(0.9),
            seed: config.seed.unwrap_or(0),
        };

        // Tokens file
        let tokens_ptr = cstring_from_str("");
        let decoding_method_ptr = cstring_from_str("greedy_search");

        let model_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
                funasr_nano: funasr_nano_config,
                debug,
                num_threads,
                provider: provider_ptr.as_ptr(),
                tokens: tokens_ptr.as_ptr(),

                // nulls - use mem::zeroed for unused fields
                model_type: std::ptr::null(),
                bpe_vocab: mem::zeroed::<_>(),
                modeling_unit: mem::zeroed::<_>(),
                nemo_ctc: mem::zeroed::<_>(),
                paraformer: mem::zeroed::<_>(),
                tdnn: mem::zeroed::<_>(),
                telespeech_ctc: mem::zeroed::<_>(),
                transducer: mem::zeroed::<_>(),
                whisper: mem::zeroed::<_>(),
                fire_red_asr: mem::zeroed::<_>(),
                sense_voice: mem::zeroed::<_>(),
                moonshine: mem::zeroed::<_>(),
                dolphin: mem::zeroed::<_>(),
                zipformer_ctc: mem::zeroed::<_>(),
                canary: mem::zeroed::<_>(),
                wenet_ctc: mem::zeroed::<_>(),
                omnilingual: mem::zeroed::<_>(),
                medasr: mem::zeroed::<_>(),
            }
        };

        let recognizer_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
                decoding_method: decoding_method_ptr.as_ptr(),
                feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                    sample_rate: 16000,
                    feature_dim: 80,
                },
                model_config,

                // 显式设置为 null 指针，避免被错误解析
                hotwords_file: std::ptr::null(),
                hotwords_score: 0.0,
                lm_config: sherpa_rs_sys::SherpaOnnxOfflineLMConfig {
                    model: std::ptr::null(),
                    scale: 0.0,
                },
                max_active_paths: 0,
                rule_fars: std::ptr::null(),
                rule_fsts: std::ptr::null(),
                blank_penalty: 0.0,
                hr: sherpa_rs_sys::SherpaOnnxHomophoneReplacerConfig {
                    lexicon: std::ptr::null(),
                    rule_fsts: std::ptr::null(),
                    dict_dir: std::ptr::null(),
                },
            }
        };

        let recognizer =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&recognizer_config) };
        if recognizer.is_null() {
            bail!("Failed to create FunASR-Nano recognizer");
        }

        Ok(Self { recognizer })
    }

    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> FunasrNanoRecognizerResult {
        unsafe {
            let stream = sherpa_rs_sys::SherpaOnnxCreateOfflineStream(self.recognizer);
            sherpa_rs_sys::SherpaOnnxAcceptWaveformOffline(
                stream,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len().try_into().unwrap(),
            );
            sherpa_rs_sys::SherpaOnnxDecodeOfflineStream(self.recognizer, stream);
            let result_ptr = sherpa_rs_sys::SherpaOnnxGetOfflineStreamResult(stream);
            let raw_result = result_ptr.read();
            let result = FunasrNanoRecognizerResult::new(&raw_result);
            // Free resources
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
            result
        }
    }
}

unsafe impl Send for FunasrNanoRecognizer {}
unsafe impl Sync for FunasrNanoRecognizer {}

impl Drop for FunasrNanoRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}
