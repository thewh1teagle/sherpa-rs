use std::mem;

use crate::{
    get_default_provider,
    utils::{cstr_to_string, cstring_from_str},
};
use eyre::{bail, Result};

#[derive(Debug, Clone)]
pub struct KeywordSpotConfig {
    pub zipformer_encoder: String,
    pub zipformer_decoder: String,
    pub zipformer_joiner: String,

    pub tokens: String,
    pub keywords: String,
    pub max_active_path: i32,
    pub keywords_threshold: f32,
    pub keywords_score: f32,

    pub num_trailing_blanks: i32,

    pub sample_rate: i32,
    pub feature_dim: i32,

    pub debug: bool,
    pub num_threads: Option<i32>,
    pub provider: Option<String>,
}

impl Default for KeywordSpotConfig {
    fn default() -> Self {
        Self {
            keywords_threshold: 0.1,
            max_active_path: 4,
            keywords_score: 3.0,
            keywords: String::new(),
            tokens: String::new(),

            sample_rate: 16000,
            feature_dim: 80,
            num_trailing_blanks: 1,

            zipformer_decoder: String::new(),
            zipformer_encoder: String::new(),
            zipformer_joiner: String::new(),

            debug: false,
            num_threads: None,
            provider: Some("cpu".into()),
        }
    }
}

pub struct KeywordSpot {
    spotter: *const sherpa_rs_sys::SherpaOnnxKeywordSpotter,
    stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream,
}

impl KeywordSpot {
    // Create new keyboard spotter along with stream
    // Ready for streaming or regular use
    pub fn new(config: KeywordSpotConfig) -> Result<Self> {
        let provider = cstring_from_str(&config.provider.unwrap_or(get_default_provider()));

        let zipformer_encoder = cstring_from_str(&config.zipformer_encoder);
        let zipformer_decoder = cstring_from_str(&config.zipformer_decoder);
        let zipformer_joiner = cstring_from_str(&config.zipformer_joiner);

        let tokens = cstring_from_str(&config.tokens);
        let keywords = cstring_from_str(&config.keywords);

        let sherpa_config = unsafe {
            sherpa_rs_sys::SherpaOnnxKeywordSpotterConfig {
                feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                    sample_rate: config.sample_rate,
                    feature_dim: config.feature_dim,
                },
                keywords_buf: mem::zeroed::<_>(),
                keywords_buf_size: 0,
                keywords_file: keywords.as_ptr(),
                max_active_paths: config.max_active_path,
                keywords_score: config.keywords_score,
                keywords_threshold: config.keywords_threshold,
                num_trailing_blanks: config.num_trailing_blanks,
                model_config: sherpa_rs_sys::SherpaOnnxOnlineModelConfig {
                    transducer: sherpa_rs_sys::SherpaOnnxOnlineTransducerModelConfig {
                        encoder: zipformer_encoder.as_ptr(),
                        decoder: zipformer_decoder.as_ptr(),
                        joiner: zipformer_joiner.as_ptr(),
                    },
                    num_threads: config.num_threads.unwrap_or(1),
                    provider: provider.as_ptr(),
                    debug: config.debug.into(),
                    tokens: tokens.as_ptr(),

                    paraformer: mem::zeroed::<_>(),
                    zipformer2_ctc: mem::zeroed::<_>(),
                    model_type: mem::zeroed::<_>(),
                    modeling_unit: mem::zeroed::<_>(),
                    bpe_vocab: mem::zeroed::<_>(),
                    tokens_buf: mem::zeroed::<_>(),
                    tokens_buf_size: mem::zeroed::<_>(),
                    nemo_ctc: mem::zeroed::<_>(),
                },
            }
        };
        let spotter = unsafe { sherpa_rs_sys::SherpaOnnxCreateKeywordSpotter(&sherpa_config) };

        if spotter.is_null() {
            bail!("Failed to create keyword spotter");
        }
        let stream = unsafe { sherpa_rs_sys::SherpaOnnxCreateKeywordStream(spotter) };
        if stream.is_null() {
            bail!("Failed to create SherpaOnnx keyword stream");
        }

        Ok(Self { spotter, stream })
    }

    pub fn extract_keyword(
        &mut self,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> Result<Option<String>> {
        // Create keyword spotting stream
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamAcceptWaveform(
                self.stream,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len() as i32,
            );
            sherpa_rs_sys::SherpaOnnxOnlineStreamInputFinished(self.stream);
            while sherpa_rs_sys::SherpaOnnxIsKeywordStreamReady(self.spotter, self.stream) == 1 {
                sherpa_rs_sys::SherpaOnnxDecodeKeywordStream(self.spotter, self.stream);
            }
            let result_ptr = sherpa_rs_sys::SherpaOnnxGetKeywordResult(self.spotter, self.stream);
            let mut keyword = None;
            if !result_ptr.is_null() {
                let decoded_keyword = cstr_to_string((*result_ptr).keyword as _);
                if !decoded_keyword.is_empty() {
                    keyword = Some(decoded_keyword);
                }
                sherpa_rs_sys::SherpaOnnxDestroyKeywordResult(result_ptr);
            }
            Ok(keyword)
        }
    }
}

unsafe impl Send for KeywordSpot {}
unsafe impl Sync for KeywordSpot {}

impl Drop for KeywordSpot {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOnlineStream(self.stream);
            sherpa_rs_sys::SherpaOnnxDestroyKeywordSpotter(self.spotter);
        }
    }
}
