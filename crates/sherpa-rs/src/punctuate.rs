use eyre::{bail, Result};

use crate::{
    get_default_provider,
    utils::{cstr_to_string, cstring_from_str},
};

#[derive(Debug, Default, Clone)]
pub struct PunctuationConfig {
    pub model: String,
    pub debug: bool,
    pub num_threads: Option<i32>,
    pub provider: Option<String>,
}

pub struct Punctuation {
    audio_punctuation: *const sherpa_rs_sys::SherpaOnnxOfflinePunctuation,
}

impl Punctuation {
    pub fn new(config: PunctuationConfig) -> Result<Self> {
        let model = cstring_from_str(&config.model);
        let provider = cstring_from_str(&config.provider.unwrap_or(if cfg!(target_os = "macos") {
            // TODO: sherpa-onnx/issues/1448
            "cpu".into()
        } else {
            get_default_provider()
        }));

        let sherpa_config = sherpa_rs_sys::SherpaOnnxOfflinePunctuationConfig {
            model: sherpa_rs_sys::SherpaOnnxOfflinePunctuationModelConfig {
                ct_transformer: model.as_ptr(),
                num_threads: config.num_threads.unwrap_or(1),
                debug: config.debug.into(),
                provider: provider.as_ptr(),
            },
        };
        let audio_punctuation =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflinePunctuation(&sherpa_config) };

        if audio_punctuation.is_null() {
            bail!("Failed to create audio punctuation");
        }
        Ok(Self { audio_punctuation })
    }

    pub fn add_punctuation(&mut self, text: &str) -> String {
        let text = cstring_from_str(text);
        unsafe {
            let text_with_punct_ptr = sherpa_rs_sys::SherpaOfflinePunctuationAddPunct(
                self.audio_punctuation,
                text.as_ptr(),
            );
            let text_with_punct = cstr_to_string(text_with_punct_ptr as _);
            sherpa_rs_sys::SherpaOfflinePunctuationFreeText(text_with_punct_ptr);
            text_with_punct
        }
    }
}

unsafe impl Send for Punctuation {}
unsafe impl Sync for Punctuation {}

impl Drop for Punctuation {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflinePunctuation(self.audio_punctuation);
        }
    }
}
