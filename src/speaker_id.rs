use eyre::{bail, Result};
use std::{
    ffi::{CStr, CString},
    path::PathBuf,
};

/// If similarity is greater or equal to thresold than it's a match!
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.5;

#[derive(Debug)]
pub struct ExtractorConfig {
    pub(crate) cfg: sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorConfig,
    model: String,
}

#[derive(Debug)]
pub struct EmbeddingExtractor {
    pub(crate) extractor: *const sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractor,
    pub embedding_size: usize,
}

impl ExtractorConfig {
    pub fn new(
        model: String,
        provider: Option<String>,
        num_threads: Option<i32>,
        debug: bool,
    ) -> Self {
        let default_provider = if cfg!(target_os = "macos") {
            "coreml"
        } else {
            "cpu"
        };
        let provider = provider.unwrap_or(default_provider.into());
        let num_threads = num_threads.unwrap_or(2);
        let debug = if debug { 1 } else { 0 };
        let model_cstr = CString::new(model.clone()).unwrap();
        let provider = CString::new(provider).unwrap();
        let cfg = sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorConfig {
            debug,
            model: model_cstr.into_raw(),
            num_threads,
            provider: provider.into_raw(),
        };
        Self { cfg, model }
    }

    pub fn as_ptr(&self) -> *const sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorConfig {
        &self.cfg
    }
}

impl EmbeddingExtractor {
    pub fn new_from_config(config: ExtractorConfig) -> Result<Self> {
        let model_path = PathBuf::from(&config.model);
        if !model_path.exists() {
            bail!("model not found at {}", model_path.display())
        }
        let extractor =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateSpeakerEmbeddingExtractor(config.as_ptr()) };
        // Assume embedding size is known or can be retrieved
        let embedding_size =
            unsafe { sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorDim(extractor) }
                .try_into()
                .unwrap();
        Ok(Self {
            extractor,
            embedding_size,
        })
    }

    pub fn compute_speaker_embedding(
        &mut self,
        sample_rate: i32,
        samples: Vec<f32>,
    ) -> Result<Vec<f32>> {
        unsafe {
            let stream =
                sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorCreateStream(self.extractor);
            if stream.is_null() {
                bail!("Failed to create SherpaOnnxOnlineStream");
            }

            sherpa_rs_sys::AcceptWaveform(
                stream,
                sample_rate,
                samples.as_ptr(),
                samples.len() as i32,
            );
            sherpa_rs_sys::InputFinished(stream);

            if !self.is_ready(stream) {
                bail!("Embedding extractor is not ready");
            }

            let embedding_ptr = sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(
                self.extractor,
                stream,
            );
            if embedding_ptr.is_null() {
                bail!("Failed to compute speaker embedding");
            }
            log::debug!("using dimensions {}", self.embedding_size);
            let embedding = std::slice::from_raw_parts(embedding_ptr, self.embedding_size).to_vec();
            // Free ptr
            sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(embedding_ptr);
            Ok(embedding)
        }
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn is_ready(
        &mut self,
        stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream,
    ) -> bool {
        unsafe {
            let result =
                sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorIsReady(self.extractor, stream);
            result != 0
        }
    }
}

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
