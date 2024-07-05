use eyre::{bail, Result};
use std::ffi::CString;

#[derive(Debug)]
pub struct ExtractorConfig {
    pub(crate) cfg: sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorConfig,
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
        let model = CString::new(model).unwrap();
        let provider = CString::new(provider).unwrap();
        let cfg = sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorConfig {
            debug,
            model: model.as_ptr(),
            num_threads,
            provider: provider.as_ptr(),
        };
        Self { cfg }
    }

    pub fn as_ptr(&self) -> *const sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorConfig {
        &self.cfg
    }
}

impl EmbeddingExtractor {
    pub fn new(config: ExtractorConfig) -> Self {
        let extractor =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateSpeakerEmbeddingExtractor(config.as_ptr()) };
        Self { extractor }
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
            let embedding_size = 256; // This should be replaced with the actual size
            let embedding = std::slice::from_raw_parts(embedding_ptr, embedding_size).to_vec();

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
