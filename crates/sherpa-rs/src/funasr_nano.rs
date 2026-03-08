use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::mem;

#[derive(Debug)]
pub struct FunASRNanoRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

pub type FunASRNanoRecognizerResult = super::OfflineRecognizerResult;

#[derive(Debug, Clone)]
pub struct FunASRNanoConfig {
    /// Path to the encoder adaptor ONNX model
    pub encoder_adaptor: String,
    /// Path to the LLM ONNX model (unified KV-cache version)
    pub llm: String,
    /// Path to the embedding ONNX model
    pub embedding: String,
    /// Path to the tokenizer directory (e.g., Qwen3-0.6B)
    pub tokenizer: String,
    /// System prompt for the LLM decoder
    pub system_prompt: String,
    /// User prompt for the LLM decoder
    pub user_prompt: String,
    /// Maximum number of new tokens to generate
    pub max_new_tokens: i32,
    /// Sampling temperature
    pub temperature: f32,
    /// Top-p (nucleus) sampling probability
    pub top_p: f32,
    /// Random seed for reproducibility
    pub seed: i32,
    /// Execution provider (e.g., "cpu", "cuda", "coreml")
    pub provider: Option<String>,
    /// Number of threads for inference
    pub num_threads: Option<i32>,
    /// Enable debug logging
    pub debug: bool,
}

impl Default for FunASRNanoConfig {
    fn default() -> Self {
        Self {
            encoder_adaptor: String::new(),
            llm: String::new(),
            embedding: String::new(),
            tokenizer: String::new(),
            system_prompt: "You are a helpful assistant.".into(),
            user_prompt: "Transcription:".into(),
            max_new_tokens: 512,
            temperature: 0.3,
            top_p: 0.8,
            seed: 42,
            provider: None,
            num_threads: Some(4),
            debug: false,
        }
    }
}

impl FunASRNanoRecognizer {
    pub fn new(config: FunASRNanoConfig) -> Result<Self> {
        let debug = config.debug.into();
        let provider = config.provider.unwrap_or(get_default_provider());
        let provider_ptr = cstring_from_str(&provider);
        let num_threads = config.num_threads.unwrap_or(4);

        // FunASR Nano specific config
        let encoder_adaptor_ptr = cstring_from_str(&config.encoder_adaptor);
        let llm_ptr = cstring_from_str(&config.llm);
        let embedding_ptr = cstring_from_str(&config.embedding);
        let tokenizer_ptr = cstring_from_str(&config.tokenizer);
        let system_prompt_ptr = cstring_from_str(&config.system_prompt);
        let user_prompt_ptr = cstring_from_str(&config.user_prompt);

        let funasr_nano_config = sherpa_rs_sys::SherpaOnnxOfflineFunASRNanoModelConfig {
            encoder_adaptor: encoder_adaptor_ptr.as_ptr(),
            llm: llm_ptr.as_ptr(),
            embedding: embedding_ptr.as_ptr(),
            tokenizer: tokenizer_ptr.as_ptr(),
            system_prompt: system_prompt_ptr.as_ptr(),
            user_prompt: user_prompt_ptr.as_ptr(),
            max_new_tokens: config.max_new_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            seed: config.seed,
        };

        // General model config
        let model_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
                tokens: mem::zeroed::<_>(), // FunASR Nano uses tokenizer directory instead
                provider: provider_ptr.as_ptr(),
                num_threads,
                debug,
                funasr_nano: funasr_nano_config,
                // Other fields set to default/null
                bpe_vocab: mem::zeroed::<_>(),
                model_type: mem::zeroed::<_>(),
                modeling_unit: mem::zeroed::<_>(),
                nemo_ctc: mem::zeroed::<_>(),
                paraformer: mem::zeroed::<_>(),
                tdnn: mem::zeroed::<_>(),
                telespeech_ctc: mem::zeroed::<_>(),
                fire_red_asr: mem::zeroed::<_>(),
                transducer: mem::zeroed::<_>(),
                whisper: mem::zeroed::<_>(),
                moonshine: mem::zeroed::<_>(),
                dolphin: mem::zeroed::<_>(),
                zipformer_ctc: mem::zeroed(),
                canary: mem::zeroed(),
                wenet_ctc: mem::zeroed::<_>(),
                sense_voice: mem::zeroed::<_>(),
                omnilingual: mem::zeroed::<_>(),
                medasr: mem::zeroed::<_>(),
            }
        };

        // Recognizer config
        let recognizer_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
                decoding_method: mem::zeroed::<_>(),
                feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                    sample_rate: 16000,
                    feature_dim: 80,
                },
                hotwords_file: mem::zeroed::<_>(),
                hotwords_score: 0.0,
                lm_config: sherpa_rs_sys::SherpaOnnxOfflineLMConfig {
                    model: mem::zeroed::<_>(),
                    scale: 0.0,
                },
                max_active_paths: 0,
                model_config,
                rule_fars: mem::zeroed::<_>(),
                rule_fsts: mem::zeroed::<_>(),
                blank_penalty: 0.0,
                hr: mem::zeroed::<_>(),
            }
        };

        let recognizer =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&recognizer_config) };
        if recognizer.is_null() {
            bail!("Failed to create FunASR Nano recognizer");
        }

        Ok(Self { recognizer })
    }

    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> FunASRNanoRecognizerResult {
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
            let result = FunASRNanoRecognizerResult::new(&raw_result);
            // Free resources
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
            result
        }
    }
}

unsafe impl Send for FunASRNanoRecognizer {}
unsafe impl Sync for FunASRNanoRecognizer {}

impl Drop for FunASRNanoRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}
