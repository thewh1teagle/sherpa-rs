use crate::{cstr, cstr_to_string, get_default_provider};
use eyre::{bail, Result};
use std::ptr::null;

#[derive(Debug)]
pub struct WhisperRecognizer {
    recognizer: *mut sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

#[derive(Debug)]
pub struct WhisperRecognizerResult {
    pub text: String,
    // pub timestamps: Vec<f32>,
}

#[derive(Debug)]
pub struct WhisperConfig {
    pub decoder: String,
    pub encoder: String,
    pub tokens: String,
    pub language: String,
    pub debug: Option<bool>,
    pub provider: Option<String>,
    pub num_threads: Option<i32>,
    pub bpe_vocab: Option<String>,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            decoder: String::new(),
            encoder: String::new(),
            tokens: String::new(),
            language: String::from("en"),
            debug: None,
            provider: None,
            num_threads: Some(1),
            bpe_vocab: None,
        }
    }
}

impl WhisperRecognizer {
    pub fn new(config: WhisperConfig) -> Result<Self> {
        let decoder_c = cstr!(config.decoder);
        let encoder_c = cstr!(config.encoder);
        let language_c = cstr!(config.language);
        let task_c = cstr!("transcribe".to_string());
        let tail_paddings = 0;
        let tokens_c = cstr!(config.tokens);

        let debug = config.debug.unwrap_or_default();
        let debug = if debug { 1 } else { 0 };
        let provider = config.provider.unwrap_or(get_default_provider());
        let provider_c = cstr!(provider);
        let num_threads = config.num_threads.unwrap_or(2);
        let bpe_vocab = config.bpe_vocab.unwrap_or("".into());
        let bpe_vocab_c = cstr!(bpe_vocab);

        let whisper = sherpa_rs_sys::SherpaOnnxOfflineWhisperModelConfig {
            decoder: decoder_c.into_raw(),
            encoder: encoder_c.into_raw(),
            language: language_c.into_raw(),
            task: task_c.into_raw(),
            tail_paddings,
        };

        let sense_voice_model_c = cstr!("".to_string());
        let sense_voice_language_c = cstr!("".to_string());
        let sense_voice = sherpa_rs_sys::SherpaOnnxOfflineSenseVoiceModelConfig {
            model: sense_voice_model_c.into_raw(),
            language: sense_voice_language_c.into_raw(),
            use_itn: 0,
        };

        let model_config = sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
            bpe_vocab: bpe_vocab_c.into_raw(),
            debug,
            model_type: null(),
            modeling_unit: null(),
            nemo_ctc: sherpa_rs_sys::SherpaOnnxOfflineNemoEncDecCtcModelConfig { model: null() },
            num_threads,
            paraformer: sherpa_rs_sys::SherpaOnnxOfflineParaformerModelConfig { model: null() },
            provider: provider_c.into_raw(),
            tdnn: sherpa_rs_sys::SherpaOnnxOfflineTdnnModelConfig { model: null() },
            telespeech_ctc: null(),
            tokens: tokens_c.into_raw(),
            transducer: sherpa_rs_sys::SherpaOnnxOfflineTransducerModelConfig {
                encoder: null(),
                decoder: null(),
                joiner: null(),
            },
            whisper,
            sense_voice,
        };

        let config = sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
            decoding_method: cstr!("greedy_search").into_raw(), // greedy_search, modified_beam_search
            feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                sample_rate: 16000,
                feature_dim: 512,
            },
            hotwords_file: null(),
            hotwords_score: 0.0,
            lm_config: sherpa_rs_sys::SherpaOnnxOfflineLMConfig {
                model: null(),
                scale: 0.0,
            },
            max_active_paths: 0,
            model_config,
            rule_fars: null(),
            rule_fsts: null(),
            blank_penalty: 0.0,
        };

        let recognizer = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&config) };
        if recognizer.is_null() {
            bail!("Failed to create recognizer")
        }

        Ok(Self { recognizer })
    }

    pub fn transcribe(&mut self, sample_rate: u32, samples: Vec<f32>) -> WhisperRecognizerResult {
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
            let text = cstr_to_string!(raw_result.text);
            // let timestamps: &[f32] =
            // std::slice::from_raw_parts(raw_result.timestamps, raw_result.count as usize);
            let result = WhisperRecognizerResult { text };
            // Free
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
            result
        }
    }
}

unsafe impl Send for WhisperRecognizer {}
unsafe impl Sync for WhisperRecognizer {}

impl Drop for WhisperRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read_audio_file;
    use std::time::Instant;

    #[test]
    fn test_whisper_transcribe() {
        let path = "motivation.wav";
        let (sample_rate, samples) = read_audio_file(&path).expect("file not found");

        // Check if the sample rate is 16000
        if sample_rate != 16000 {
            panic!("The sample rate must be 16000.");
        }

        let config = WhisperConfig {
            decoder: "sherpa-onnx-whisper-tiny/tiny-decoder.onnx".into(),
            encoder: "sherpa-onnx-whisper-tiny/tiny-encoder.onnx".into(),
            tokens: "sherpa-onnx-whisper-tiny/tiny-tokens.txt".into(),
            language: "en".into(),
            debug: Some(true),
            provider: None,
            num_threads: None,
            bpe_vocab: None,
            ..Default::default() // fill in any missing fields with defaults
        };

        let mut recognizer = WhisperRecognizer::new(config).unwrap();

        let start_t = Instant::now();
        let result = recognizer.transcribe(sample_rate, samples);
        println!("{:?}", result);
        println!("Time taken for transcription: {:?}", start_t.elapsed());
    }
}
