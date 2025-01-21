use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::ptr::null;

#[derive(Debug)]
pub struct WhisperRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

pub type WhisperRecognizerResult = super::OfflineRecognizerResult;

#[derive(Debug, Clone)]
pub struct WhisperConfig {
    pub decoder: String,
    pub encoder: String,
    pub tokens: String,
    pub language: String,
    pub bpe_vocab: Option<String>,

    pub provider: Option<String>,
    pub num_threads: Option<i32>,
    pub debug: bool,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            decoder: String::new(),
            encoder: String::new(),
            tokens: String::new(),
            language: String::from("en"),
            bpe_vocab: None,
            debug: false,
            provider: None,
            num_threads: Some(1),
        }
    }
}

impl WhisperRecognizer {
    pub fn new(config: WhisperConfig) -> Result<Self> {
        let debug = config.debug.into();
        let provider = config.provider.unwrap_or(get_default_provider());

        // Onnx
        let provider_ptr = cstring_from_str(&provider);
        let num_threads = config.num_threads.unwrap_or(2);

        // Whisper
        let bpe_vocab_ptr = cstring_from_str(&config.bpe_vocab.unwrap_or("".into()));
        let tail_paddings = 0;
        let decoder_ptr = cstring_from_str(&config.decoder);
        let encoder_ptr = cstring_from_str(&config.encoder);
        let language_ptr = cstring_from_str(&config.language);
        let task_ptr = cstring_from_str("transcribe");
        let tokens_ptr = cstring_from_str(&config.tokens);
        let decoding_method_ptr = cstring_from_str("greedy_search");
        // Sense voice
        let sense_voice_model_ptr = cstring_from_str("");
        let sense_voice_language_ptr = cstring_from_str("");

        let whisper = sherpa_rs_sys::SherpaOnnxOfflineWhisperModelConfig {
            decoder: decoder_ptr.as_ptr(),
            encoder: encoder_ptr.as_ptr(),
            language: language_ptr.as_ptr(),
            task: task_ptr.as_ptr(),
            tail_paddings,
        };

        let sense_voice = sherpa_rs_sys::SherpaOnnxOfflineSenseVoiceModelConfig {
            model: sense_voice_model_ptr.as_ptr(),
            language: sense_voice_language_ptr.as_ptr(),
            use_itn: 0,
        };

        let model_config = sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
            bpe_vocab: bpe_vocab_ptr.as_ptr(),
            debug,
            model_type: null(),
            modeling_unit: null(),
            nemo_ctc: sherpa_rs_sys::SherpaOnnxOfflineNemoEncDecCtcModelConfig { model: null() },
            num_threads,
            paraformer: sherpa_rs_sys::SherpaOnnxOfflineParaformerModelConfig { model: null() },
            provider: provider_ptr.as_ptr(),
            tdnn: sherpa_rs_sys::SherpaOnnxOfflineTdnnModelConfig { model: null() },
            telespeech_ctc: null(),
            tokens: tokens_ptr.as_ptr(),
            transducer: sherpa_rs_sys::SherpaOnnxOfflineTransducerModelConfig {
                encoder: null(),
                decoder: null(),
                joiner: null(),
            },
            whisper,
            sense_voice,
            moonshine: sherpa_rs_sys::SherpaOnnxOfflineMoonshineModelConfig {
                preprocessor: null(),
                encoder: null(),
                uncached_decoder: null(),
                cached_decoder: null(),
            },
        };

        let config = sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
            decoding_method: decoding_method_ptr.as_ptr(), // greedy_search, modified_beam_search
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
            bail!("Failed to create recognizer");
        }

        Ok(Self { recognizer })
    }

    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> WhisperRecognizerResult {
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
            let result = WhisperRecognizerResult::new(&raw_result);
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
        let (samples, sample_rate) = read_audio_file(path).unwrap();

        // Check if the sample rate is 16000
        if sample_rate != 16000 {
            panic!("The sample rate must be 16000.");
        }

        let config = WhisperConfig {
            decoder: "sherpa-onnx-whisper-tiny/tiny-decoder.onnx".into(),
            encoder: "sherpa-onnx-whisper-tiny/tiny-encoder.onnx".into(),
            tokens: "sherpa-onnx-whisper-tiny/tiny-tokens.txt".into(),
            language: "en".into(),
            debug: true,
            provider: None,
            num_threads: None,
            bpe_vocab: None,
        };

        let mut recognizer = WhisperRecognizer::new(config).unwrap();

        let start_t = Instant::now();
        let result = recognizer.transcribe(sample_rate, &samples);
        println!("{:?}", result);
        println!("Time taken for transcription: {:?}", start_t.elapsed());
    }
}
