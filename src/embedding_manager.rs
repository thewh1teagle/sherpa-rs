use eyre::{bail, Result};
use std::ffi::{CStr, CString};

#[derive(Debug)]
pub struct EmbeddingManager {
    pub(crate) manager: *const sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingManager,
}

impl EmbeddingManager {
    pub fn new(dimension: i32) -> Self {
        unsafe {
            let manager = sherpa_rs_sys::SherpaOnnxCreateSpeakerEmbeddingManager(dimension);
            Self { manager }
        }
    }

    pub fn search(&mut self, embedding: &[f32], threshold: f32) -> Option<String> {
        unsafe {
            let name = sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingManagerSearch(
                self.manager,
                embedding.to_owned().as_mut_ptr(),
                threshold,
            );
            if name.is_null() {
                return None;
            }
            let cstr = CStr::from_ptr(name);
            Some(cstr.to_str().unwrap_or_default().to_string())
        }
    }

    pub fn add(&mut self, name: String, embedding: &mut [f32]) -> Result<()> {
        let name_cstr = CString::new(name.clone())?;

        unsafe {
            let status = sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingManagerAdd(
                self.manager,
                name_cstr.into_raw(),
                embedding.as_mut_ptr(),
            );
            if status.is_negative() {
                bail!("Failed to register {}", name)
            }
            Ok(())
        }
    }
}

impl Drop for EmbeddingManager {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroySpeakerEmbeddingManager(self.manager);
        };
    }
}
