use eyre::{bail, Result};
use std::ffi::{CStr, CString};

use crate::cstr_to_string;

#[derive(Debug, Clone)]
pub struct EmbeddingManager {
    pub(crate) manager: *const sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingManager,
}

#[derive(Debug, Clone)]
pub struct SpeakerMatch {
    pub name: String,
    pub score: f32,
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

    pub fn get_best_matches(
        &mut self,
        embedding: &[f32],
        threshold: f32,
        n: i32,
    ) -> Vec<SpeakerMatch> {
        unsafe {
            let result_ptr = sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingManagerGetBestMatches(
                self.manager,
                embedding.to_owned().as_mut_ptr(),
                threshold,
                n,
            );
            if result_ptr.is_null() {
                return Vec::new();
            }
            let result = result_ptr.read();

            let matches_c = std::slice::from_raw_parts(result.matches, result.count as usize);
            let mut matches: Vec<SpeakerMatch> = Vec::new();
            for i in 0..result.count {
                let match_c = matches_c[i as usize];
                let name = cstr_to_string!(match_c.name);
                let score = match_c.score;
                matches.push(SpeakerMatch { name, score });
            }
            sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingManagerFreeBestMatches(result_ptr);
            matches
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
