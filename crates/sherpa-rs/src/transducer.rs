use crate::utils::cstr_to_string;
use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::ptr::null;

pub struct TransducerRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

#[derive(Debug, Clone)]
pub struct TransducerConfig {
    pub decoder: String,
    pub encoder: String,
    pub joiner: String,
    pub tokens: String,
    pub num_threads: i32,
    pub sample_rate: i32,
    pub feature_dim: i32,
    pub decoding_method: String,
    pub hotwords_file: String,
    pub hotwords_score: f32,
    pub modeling_unit: String,
    pub bpe_vocab: String,
    pub blank_penalty: f32,
    pub debug: bool,
    pub provider: Option<String>,
}

impl Default for TransducerConfig {
    fn default() -> Self {
        TransducerConfig {
            decoder: String::new(),
            encoder: String::new(),
            joiner: String::new(),
            tokens: String::new(),
            num_threads: 1,
            sample_rate: 0,
            feature_dim: 0,
            decoding_method: String::new(),
            hotwords_file: String::new(),
            hotwords_score: 0.0,
            modeling_unit: String::new(),
            bpe_vocab: String::new(),
            blank_penalty: 0.0,
            debug: false,
            provider: None,
        }
    }
}

impl TransducerRecognizer {
    pub fn new(config: TransducerConfig) -> Result<Self> {
        let debug = config.debug.into();
        let provider = config.provider.unwrap_or(get_default_provider());
        let provider_ptr = cstring_from_str(&provider);

        let encoder = cstring_from_str(&config.encoder);
        let decoder = cstring_from_str(&config.decoder);
        let joiner = cstring_from_str(&config.joiner);
        let model_type = cstring_from_str("transducer");
        let modeling_unit = cstring_from_str(&config.modeling_unit);
        let bpe_vocab = cstring_from_str(&config.bpe_vocab);
        let hotwords_file = cstring_from_str(&config.hotwords_file);
        let tokens = cstring_from_str(&config.tokens);
        let decoding_method = cstring_from_str(&config.decoding_method);

        let offline_model_config = sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
            transducer: sherpa_rs_sys::SherpaOnnxOfflineTransducerModelConfig {
                encoder: encoder.as_ptr(),
                decoder: decoder.as_ptr(),
                joiner: joiner.as_ptr(),
            },
            tokens: tokens.as_ptr(),
            num_threads: config.num_threads,
            debug,
            provider: provider_ptr.as_ptr(),
            model_type: model_type.as_ptr(),
            modeling_unit: modeling_unit.as_ptr(),
            bpe_vocab: bpe_vocab.as_ptr(),

            // NULLs
            telespeech_ctc: null(),
            paraformer: sherpa_rs_sys::SherpaOnnxOfflineParaformerModelConfig { model: null() },
            tdnn: sherpa_rs_sys::SherpaOnnxOfflineTdnnModelConfig { model: null() },
            nemo_ctc: sherpa_rs_sys::SherpaOnnxOfflineNemoEncDecCtcModelConfig { model: null() },
            whisper: sherpa_rs_sys::SherpaOnnxOfflineWhisperModelConfig {
                encoder: null(),
                decoder: null(),
                language: null(),
                task: null(),
                tail_paddings: 0,
            },
            sense_voice: sherpa_rs_sys::SherpaOnnxOfflineSenseVoiceModelConfig {
                model: null(),
                language: null(),
                use_itn: 0,
            },
            moonshine: sherpa_rs_sys::SherpaOnnxOfflineMoonshineModelConfig {
                preprocessor: null(),
                encoder: null(),
                uncached_decoder: null(),
                cached_decoder: null(),
            },
            fire_red_asr: sherpa_rs_sys::SherpaOnnxOfflineFireRedAsrModelConfig {
                encoder: null(),
                decoder: null(),
            },
        };

        let recognizer_config = sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
            model_config: offline_model_config,
            feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                sample_rate: config.sample_rate,
                feature_dim: config.feature_dim,
            },
            hotwords_file: hotwords_file.as_ptr(),
            blank_penalty: config.blank_penalty,
            decoding_method: decoding_method.as_ptr(),
            hotwords_score: config.hotwords_score,

            // NULLs
            lm_config: sherpa_rs_sys::SherpaOnnxOfflineLMConfig {
                model: null(),
                scale: 0.0,
            },
            rule_fsts: null(),
            rule_fars: null(),
            max_active_paths: 0,
        };

        let recognizer =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&recognizer_config) };
        if recognizer.is_null() {
            bail!("SherpaOnnxCreateOfflineRecognizer failed");
        }

        Ok(Self { recognizer })
    }

    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> String {
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
            let text = cstr_to_string(raw_result.text as _);

            // Free
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
            text
        }
    }
}

unsafe impl Send for TransducerRecognizer {}
unsafe impl Sync for TransducerRecognizer {}

impl Drop for TransducerRecognizer {
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
    fn test_transducer_decode() {
        let path = "motivation.wav";
        let (samples, sample_rate) = read_audio_file(path).unwrap();

        // Check if the sample rate is 16000
        if sample_rate != 16000 {
            panic!("The sample rate must be 16000.");
        }

        let config = TransducerConfig {
            decoder: "vosk-en/decoder-epoch-90-avg-20.onnx".to_string(),
            encoder: "vosk-en/encoder-epoch-90-avg-20.onnx".to_string(),
            joiner: "vosk-en/joiner-epoch-90-avg-20.onnx".to_string(),
            tokens: "vosk-en/tokens.txt".to_string(),
            num_threads: 1,
            sample_rate: 16_000,
            feature_dim: 80,

            // NULLs
            bpe_vocab: "".to_string(),
            decoding_method: "".to_string(),
            hotwords_file: "".to_string(),
            hotwords_score: 0.0,
            modeling_unit: "".to_string(),
            blank_penalty: 0.0,
            debug: true,
            provider: None,
        };

        let mut recognizer = TransducerRecognizer::new(config).unwrap();

        let start_t = Instant::now();
        let result = recognizer.transcribe(sample_rate, &samples);
        let lower_case = result.to_lowercase();
        let trimmed = lower_case.trim();

        println!("Time taken for decode: {:?}", start_t.elapsed());

        let expected_result = "the person you can control people are not who you want them to be kill your idols here there are things we can learn from people but people aren't going to be what you think they are what they should";

        assert_eq!(trimmed, expected_result);
    }
}
