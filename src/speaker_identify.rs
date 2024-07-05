use sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractor;
use sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorConfig;
use std::ffi::CString;

#[derive(Debug)]
pub struct SherpaOnnxSpeakerEmbeddingExtractorInnerConfig {
    pub(crate) cfg: SherpaOnnxSpeakerEmbeddingExtractorConfig,
}

#[derive(Debug)]
pub struct SherpaOnnxSpeakerEmbeddingExtractorInner {
    pub(crate) extractor: SherpaOnnxSpeakerEmbeddingExtractor,
}

impl SherpaOnnxSpeakerEmbeddingExtractorInnerConfig {
    pub fn new(
        model: String,
        provider: Option<String>,
        num_threads: Option<i32>,
        debug: bool,
    ) -> Self {
        let provider = provider.unwrap_or("cpu".into());
        let num_threads = num_threads.unwrap_or(2);
        let debug = if debug { 1 } else { 0 };
        let cfg = SherpaOnnxSpeakerEmbeddingExtractorConfig {
            debug,
            model: CString::new(model).unwrap().as_ptr(),
            num_threads,
            provider: CString::new(provider).unwrap().as_ptr(),
        };
        Self { cfg }
    }
}

// impl SherpaOnnxSpeakerEmbeddingExtractorInner {
//     pub fn new(config: SherpaOnnxSpeakerEmbeddingExtractorInnerConfig) -> Self {
//         SherpaOnnxSpeakerEmbeddingExtractor
//     }
// }
