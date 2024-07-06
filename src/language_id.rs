use crate::get_default_provider;
use eyre::{bail, Result};
use std::ffi::{CStr, CString};

#[derive(Debug)]
pub struct SpokenLanguageId {
    slid: *const sherpa_rs_sys::SherpaOnnxSpokenLanguageIdentification,
}

impl SpokenLanguageId {
    pub fn new(
        encoder: String,
        decoder: String,
        debug: Option<bool>,
        provider: Option<String>,
        num_threads: Option<i32>,
    ) -> Self {
        let provider = provider.unwrap_or(get_default_provider());
        let provider_c = CString::new(provider).unwrap();
        let debug = debug.unwrap_or_default();
        let debug = if debug { 0 } else { 1 };

        let encoder_c = CString::new(encoder).unwrap();
        let decoder_c = CString::new(decoder).unwrap();
        let whisper = sherpa_rs_sys::SherpaOnnxSpokenLanguageIdentificationWhisperConfig {
            decoder: decoder_c.into_raw(),
            encoder: encoder_c.into_raw(),
            tail_paddings: 0,
        };
        let config = sherpa_rs_sys::SherpaOnnxSpokenLanguageIdentificationConfig {
            debug,
            num_threads: num_threads.unwrap_or(2),
            provider: provider_c.into_raw(),
            whisper,
        };
        let slid = unsafe { sherpa_rs_sys::SherpaOnnxCreateSpokenLanguageIdentification(&config) };
        Self { slid }
    }

    pub fn compute(&mut self, samples: Vec<f32>, sample_rate: i32) -> Result<String> {
        unsafe {
            let stream =
                sherpa_rs_sys::SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream(self.slid);
            sherpa_rs_sys::AcceptWaveformOffline(
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
            sherpa_rs_sys::DestroyOfflineStream(stream);

            Ok(language)
        }
    }
}

impl Drop for SpokenLanguageId {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroySpokenLanguageIdentification(self.slid);
        }
    }
}
