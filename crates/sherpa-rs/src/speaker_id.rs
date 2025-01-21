use eyre::{bail, Result};
use std::path::PathBuf;

use crate::{get_default_provider, utils::cstring_from_str};

/// If similarity is greater or equal to thresold than it's a match!
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.5;

#[derive(Debug, Default)]
pub struct ExtractorConfig {
    pub model: String,
    pub provider: Option<String>,
    pub num_threads: Option<usize>,
    pub debug: bool,
}

#[derive(Debug)]
pub struct EmbeddingExtractor {
    pub(crate) extractor: *const sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractor,
    pub embedding_size: usize,
}

impl EmbeddingExtractor {
    pub fn new(config: ExtractorConfig) -> Result<Self> {
        let provider = config.provider.unwrap_or(get_default_provider());

        let num_threads = config.num_threads.unwrap_or(1);
        let debug = config.debug.into();

        let model_path = PathBuf::from(&config.model);
        if !model_path.exists() {
            bail!("model not found at {}", model_path.display())
        }
        let model = cstring_from_str(&config.model);
        let provider = cstring_from_str(&provider);

        let extractor_config = sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorConfig {
            debug,
            model: model.as_ptr(),
            num_threads: num_threads as i32,
            provider: provider.as_ptr(),
        };
        let extractor =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateSpeakerEmbeddingExtractor(&extractor_config) };
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
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> Result<Vec<f32>> {
        unsafe {
            let stream =
                sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorCreateStream(self.extractor);
            if stream.is_null() {
                bail!("Failed to create SherpaOnnxOnlineStream");
            }

            sherpa_rs_sys::SherpaOnnxOnlineStreamAcceptWaveform(
                stream,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len() as i32,
            );
            sherpa_rs_sys::SherpaOnnxOnlineStreamInputFinished(stream);

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
            tracing::debug!("using dimensions {}", self.embedding_size);
            let embedding = std::slice::from_raw_parts(embedding_ptr, self.embedding_size).to_vec();
            // Free
            sherpa_rs_sys::SherpaOnnxDestroyOnlineStream(stream);
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

unsafe impl Send for EmbeddingExtractor {}
unsafe impl Sync for EmbeddingExtractor {}

impl Drop for EmbeddingExtractor {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroySpeakerEmbeddingExtractor(self.extractor);
        }
    }
}
