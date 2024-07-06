use eyre::{bail, Result};
use nalgebra::DVector;
use std::{ffi::CString, path::PathBuf};

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
}

impl ExtractorConfig {
    pub fn new(
        model: String,
        provider: Option<String>,
        num_threads: Option<i32>,
        debug: bool,
    ) -> Self {
        let provider = provider.unwrap_or("cpu".into());
        let num_threads = num_threads.unwrap_or(2);
        let debug = if debug { 1 } else { 0 };
        let model_cstr = CString::new(model.clone()).unwrap();
        let provider = CString::new(provider).unwrap();
        let cfg = sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorConfig {
            debug,
            model: model_cstr.as_ptr(),
            num_threads,
            provider: provider.as_ptr(),
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
        Ok(Self { extractor })
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

            // Assume embedding size is known or can be retrieved
            let embedding_size = self.get_dimension();
            log::debug!("using dimensions {}", embedding_size);
            let embedding =
                std::slice::from_raw_parts(embedding_ptr, embedding_size as usize).to_vec();

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

    /// Return the dimension of the embedding
    pub fn get_dimension(&mut self) -> i32 {
        unsafe { sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorDim(self.extractor) }
    }
}

pub fn compute_cosine_similarity(embedding1: &[f32], embedding2: &[f32]) -> f32 {
    // Convert embeddings to DVector (dynamic vector) from nalgebra
    let vec1 = DVector::from_iterator(embedding1.len(), embedding1.iter().cloned());
    let vec2 = DVector::from_iterator(embedding2.len(), embedding2.iter().cloned());

    // Compute cosine similarity using nalgebra
    let similarity = vec1.dot(&vec2) / (vec1.norm() * vec2.norm());

    similarity
}
