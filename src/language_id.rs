use crate::get_default_provider;
use eyre::{bail, Result};
use std::ffi::{CStr, CString};

#[derive(Debug)]
pub struct SpokenLanguageId {
    slid: *const sherpa_rs_sys::SherpaOnnxSpokenLanguageIdentification,
}

#[derive(Debug)]
pub struct SpokenLanguageIdConfig {
    pub encoder: String,
    pub decoder: String,
    pub debug: Option<bool>,
    pub provider: Option<String>,
    pub num_threads: Option<i32>,
}

impl Default for SpokenLanguageIdConfig {
    fn default() -> Self {
        Self {
            encoder: String::new(),
            decoder: String::new(),
            debug: None,
            provider: None,
            num_threads: None,
        }
    }
}

impl SpokenLanguageId {
    pub fn new(config: SpokenLanguageIdConfig) -> Self {
        let provider = config.provider.unwrap_or_else(get_default_provider);
        let provider_c = CString::new(provider).unwrap();
        let debug = config.debug.unwrap_or_default();
        let debug = if debug { 0 } else { 1 };

        let encoder_c = CString::new(config.encoder).unwrap();
        let decoder_c = CString::new(config.decoder).unwrap();
        let whisper = sherpa_rs_sys::SherpaOnnxSpokenLanguageIdentificationWhisperConfig {
            decoder: decoder_c.into_raw(),
            encoder: encoder_c.into_raw(),
            tail_paddings: 0,
        };
        let sherpa_config = sherpa_rs_sys::SherpaOnnxSpokenLanguageIdentificationConfig {
            debug,
            num_threads: config.num_threads.unwrap_or(2),
            provider: provider_c.into_raw(),
            whisper,
        };
        let slid =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateSpokenLanguageIdentification(&sherpa_config) };
        Self { slid }
    }

    pub fn compute(&mut self, samples: Vec<f32>, sample_rate: i32) -> Result<String> {
        unsafe {
            let stream =
                sherpa_rs_sys::SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream(self.slid);
            sherpa_rs_sys::SherpaOnnxAcceptWaveformOffline(
                stream,
                sample_rate,
                samples.as_ptr(),
                samples.len().try_into().unwrap(),
            );
            let language_result_ptr =
                sherpa_rs_sys::SherpaOnnxSpokenLanguageIdentificationCompute(self.slid, stream);
            if language_result_ptr.is_null() || (*language_result_ptr).lang.is_null() {
                bail!("language ptr is null")
            }
            let language_ptr = (*language_result_ptr).lang;
            let c_language = CStr::from_ptr(language_ptr);
            let language = c_language.to_str().unwrap().to_string();
            // Free
            sherpa_rs_sys::SherpaOnnxDestroySpokenLanguageIdentificationResult(language_result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);

            Ok(language)
        }
    }
}

unsafe impl Send for SpokenLanguageId {}
unsafe impl Sync for SpokenLanguageId {}

impl Drop for SpokenLanguageId {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroySpokenLanguageIdentification(self.slid);
        }
    }
}
