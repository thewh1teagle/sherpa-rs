use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::{mem, ptr::null};

#[derive(Debug)]
pub struct ParaformerRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

pub type ParaformerRecognizerResult = super::OfflineRecognizerResult;
pub type ParaformerOnlineRecognizerResult = super::OnlineRecognizerResult;

#[derive(Debug, Clone)]
pub struct ParaformerConfig {
    pub model: String,
    pub tokens: String,
    pub provider: Option<String>,
    pub num_threads: Option<i32>,
    pub debug: bool,
}

impl Default for ParaformerConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            tokens: String::new(),
            debug: false,
            provider: None,
            num_threads: Some(1),
        }
    }
}

impl ParaformerRecognizer {
    pub fn new(config: ParaformerConfig) -> Result<Self> {
        let debug = config.debug.into();
        let provider = config.provider.unwrap_or(get_default_provider());

        // Prepare C strings
        let provider_ptr = cstring_from_str(&provider);
        let model_ptr = cstring_from_str(&config.model);
        let tokens_ptr = cstring_from_str(&config.tokens);

        // 创建 decoding_method 的 CString 对象并绑定到变量
        let decoding_method_ptr = cstring_from_str("greedy_search");

        // Paraformer model config
        let paraformer_config = sherpa_rs_sys::SherpaOnnxOfflineParaformerModelConfig {
            model: model_ptr.as_ptr(),
        };

        // Offline model config
        let model_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
                debug,
                num_threads: config.num_threads.unwrap_or(1),
                provider: provider_ptr.as_ptr(),
                tokens: tokens_ptr.as_ptr(),
                paraformer: paraformer_config,

                // Null other model types
                bpe_vocab: mem::zeroed::<_>(),
                model_type: mem::zeroed::<_>(),
                modeling_unit: mem::zeroed::<_>(),
                nemo_ctc: mem::zeroed::<_>(),
                tdnn: mem::zeroed::<_>(),
                telespeech_ctc: null(),
                fire_red_asr: mem::zeroed::<_>(),
                transducer: mem::zeroed::<_>(),
                whisper: mem::zeroed::<_>(),
                sense_voice: mem::zeroed::<_>(),
                moonshine: mem::zeroed::<_>(),
                dolphin: mem::zeroed::<_>(),
                zipformer_ctc: mem::zeroed::<_>(),
                canary: mem::zeroed::<_>(),
                wenet_ctc: mem::zeroed::<_>(),
            }
        };

        // Recognizer config
        let recognizer_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
                decoding_method: decoding_method_ptr.as_ptr(),
                feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                    sample_rate: 16000,
                    feature_dim: 80,
                },
                model_config,
                hotwords_file: null(),
                hotwords_score: 0.0,
                lm_config: mem::zeroed::<_>(),
                max_active_paths: 0,
                rule_fars: null(),
                rule_fsts: null(),
                blank_penalty: 0.0,
                hr: mem::zeroed::<_>(),
            }
        };

        let recognizer =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&recognizer_config) };
        if recognizer.is_null() {
            bail!("Failed to create Paraformer recognizer");
        }

        Ok(Self { recognizer })
    }

    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> ParaformerRecognizerResult {
        unsafe {
            let stream = sherpa_rs_sys::SherpaOnnxCreateOfflineStream(self.recognizer);
            sherpa_rs_sys::SherpaOnnxAcceptWaveformOffline(
                stream,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len() as i32,
            );
            sherpa_rs_sys::SherpaOnnxDecodeOfflineStream(self.recognizer, stream);
            let result_ptr = sherpa_rs_sys::SherpaOnnxGetOfflineStreamResult(stream);
            let raw_result = result_ptr.read();
            let result = ParaformerRecognizerResult::new(&raw_result);

            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);

            result
        }
    }
}

unsafe impl Send for ParaformerRecognizer {}
unsafe impl Sync for ParaformerRecognizer {}

impl Drop for ParaformerRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParaformerOnlineConfig {
    pub encoder_model_path: String,
    pub decoder_model_path: String,
    pub tokens: String,
    pub provider: Option<String>,
    pub num_threads: Option<i32>,
    pub debug: bool,
    pub enable_endpoint: Option<bool>,
    pub rule1_min_trailing_silence: Option<f32>,
    pub rule2_min_trailing_silence: Option<f32>,
    pub rule3_min_utterance_length: Option<f32>,
}

impl Default for ParaformerOnlineConfig {
    fn default() -> Self {
        Self {
            encoder_model_path: String::new(),
            decoder_model_path: String::new(),
            tokens: String::new(),
            provider: None,
            num_threads: None,
            debug: false,
            enable_endpoint: None,
            rule1_min_trailing_silence: None,
            rule2_min_trailing_silence: None,
            rule3_min_utterance_length: None,
        }
    }
}

#[derive(Debug)]
pub struct ParaformerOnlineRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOnlineRecognizer,
    stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream,
    segment_id: i32,
}

impl ParaformerOnlineRecognizer {
    pub fn new(config: ParaformerOnlineConfig) -> Result<Self> {
        let debug = config.debug.into();
        let provider = config.provider.unwrap_or(get_default_provider());
        let provider_ptr = cstring_from_str(&provider);
        let tokens_ptr = cstring_from_str(&config.tokens);

        let encoder_model_path = if config.encoder_model_path.is_empty() {
            bail!("Encoder model path is required for online Paraformer")
        } else {
            cstring_from_str(&config.encoder_model_path)
        };
        let decoder_model_path = if config.decoder_model_path.is_empty() {
            bail!("Decoder model path is required for online Paraformer")
        } else {
            cstring_from_str(&config.decoder_model_path)
        };
        let paraformer_config = sherpa_rs_sys::SherpaOnnxOnlineParaformerModelConfig {
            encoder: encoder_model_path.as_ptr(),
            decoder: decoder_model_path.as_ptr(),
        };
        let empty_str = cstring_from_str("");
        let mut model_config = sherpa_rs_sys::SherpaOnnxOnlineModelConfig::default();
        model_config.debug = debug;
        model_config.num_threads = config.num_threads.unwrap_or(1);
        model_config.provider = provider_ptr.as_ptr();
        model_config.tokens = tokens_ptr.as_ptr();
        model_config.paraformer = paraformer_config;

        // Recognizer config
        let mut recognizer_config = sherpa_rs_sys::SherpaOnnxOnlineRecognizerConfig::default();
        recognizer_config.feat_config = sherpa_rs_sys::SherpaOnnxFeatureConfig {
            sample_rate: 16000,
            feature_dim: 80,
        };
        recognizer_config.model_config = model_config;
        recognizer_config.rule_fsts = empty_str.as_ptr();
        recognizer_config.rule_fars = empty_str.as_ptr();

        recognizer_config.enable_endpoint = config.enable_endpoint.unwrap_or(false).into();
        recognizer_config.rule1_min_trailing_silence =
            config.rule1_min_trailing_silence.unwrap_or(2.4);
        recognizer_config.rule2_min_trailing_silence =
            config.rule2_min_trailing_silence.unwrap_or(1.2);
        recognizer_config.rule3_min_utterance_length =
            config.rule3_min_utterance_length.unwrap_or(300.0);

        let recognizer =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateOnlineRecognizer(&recognizer_config) };
        if recognizer.is_null() {
            bail!("Failed to create online Paraformer recognizer");
        }
        let stream = unsafe { sherpa_rs_sys::SherpaOnnxCreateOnlineStream(recognizer) };
        if stream.is_null() {
            unsafe {
                sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizer(recognizer);
            }
            bail!("Failed to create online Paraformer stream");
        }
        Ok(Self {
            recognizer,
            stream,
            segment_id: 0,
        })
    }

    pub fn transcribe(
        &mut self,
        sample_rate: u32,
        samples: &[f32],
    ) -> ParaformerOnlineRecognizerResult {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamAcceptWaveform(
                self.stream,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len() as i32,
            );

            while sherpa_rs_sys::SherpaOnnxIsOnlineStreamReady(self.recognizer, self.stream) == 1 {
                sherpa_rs_sys::SherpaOnnxDecodeOnlineStream(self.recognizer, self.stream);
            }

            let result_ptr =
                sherpa_rs_sys::SherpaOnnxGetOnlineStreamResult(self.recognizer, self.stream);
            let raw_result = result_ptr.read();
            let mut result = ParaformerOnlineRecognizerResult::from(&raw_result);
            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizerResult(result_ptr);

            if sherpa_rs_sys::SherpaOnnxOnlineStreamIsEndpoint(self.recognizer, self.stream) == 1 {
                self.segment_id += 1;
                sherpa_rs_sys::SherpaOnnxOnlineStreamReset(self.recognizer, self.stream);
                result.is_final = true;
            }

            result.segment = self.segment_id;
            result
        }
    }
}

unsafe impl Send for ParaformerOnlineRecognizer {}
unsafe impl Sync for ParaformerOnlineRecognizer {}

impl Drop for ParaformerOnlineRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOnlineStream(self.stream);
            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizer(self.recognizer);
        }
    }
}
