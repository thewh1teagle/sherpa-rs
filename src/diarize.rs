use crate::get_default_provider;
use eyre::{bail, Result};
use std::{ffi::CString, path::Path};

#[derive(Debug)]
pub struct Diarize {
    sd: *const sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarization,
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub speaker: i32,
}

#[derive(Debug, Clone)]
pub struct DiarizeConfig {
    pub num_clusters: Option<i32>,
    pub threshold: Option<f32>,
    pub min_duration_on: Option<f32>,
    pub min_duration_off: Option<f32>,
    pub provider: Option<String>,
    pub debug: Option<bool>,
}

impl Default for DiarizeConfig {
    fn default() -> Self {
        Self {
            num_clusters: Some(4),
            threshold: Some(0.5),
            min_duration_on: Some(0.0),
            min_duration_off: Some(0.0),
            provider: None,
            debug: Some(false),
        }
    }
}

impl Diarize {
    pub fn new<P: AsRef<Path>>(
        segmentation_model: P,
        embedding_model: P,
        config: DiarizeConfig,
    ) -> Result<Self> {
        let provider = config.provider.unwrap_or(get_default_provider());
        let provider = CString::new(provider).unwrap();

        let debug = config.debug.unwrap_or(false);
        let debug = if debug { 1 } else { 0 };

        let embedding_model = CString::new(embedding_model.as_ref().to_str().unwrap()).unwrap();
        let segmentation_model =
            CString::new(segmentation_model.as_ref().to_str().unwrap()).unwrap();

        let clustering_config = sherpa_rs_sys::SherpaOnnxFastClusteringConfig {
            num_clusters: config.num_clusters.unwrap_or(4),
            threshold: config.threshold.unwrap_or(0.5),
        };

        let config = sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationConfig {
            embedding: sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorConfig {
                model: embedding_model.into_raw(),
                num_threads: 1,
                debug,
                provider: provider.clone().into_raw(),
            },
            clustering: clustering_config,
            min_duration_off: config.min_duration_off.unwrap_or(0.0),
            min_duration_on: config.min_duration_on.unwrap_or(0.0),
            segmentation: sherpa_rs_sys::SherpaOnnxOfflineSpeakerSegmentationModelConfig {
                pyannote: sherpa_rs_sys::SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig {
                    model: segmentation_model.into_raw(),
                },
                num_threads: 1,
                debug,
                provider: provider.clone().into_raw(),
            },
        };

        let sd = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineSpeakerDiarization(&config) };
        if sd.is_null() {
            bail!("Failed to initialize offline speaker diarization")
        }

        Ok(Self { sd })
    }

    pub fn compute(&mut self, mut samples: Vec<f32>) -> Result<Vec<Segment>> {
        let samples_ptr = samples.as_mut_ptr();
        let mut segments = Vec::new();
        unsafe {
            let result = sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
                self.sd,
                samples_ptr,
                samples.len() as i32,
                None,
                std::ptr::null_mut(),
            );

            let num_segments =
                sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(result);
            let segments_ptr: *const sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationSegment =
                sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(result);

            if !segments_ptr.is_null() && num_segments > 0 {
                let segments_result: &[sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationSegment] =
                    std::slice::from_raw_parts(segments_ptr, num_segments as usize);

                for segment in segments_result {
                    // Use segment here

                    segments.push(Segment {
                        start: segment.start,
                        end: segment.end,
                        speaker: segment.speaker,
                    });
                }
            } else {
                bail!("No segments found or invalid pointer.");
            }

            sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationDestroySegment(segments_ptr);
            sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationDestroyResult(result);

            Ok(segments)
        }
    }
}

impl Drop for Diarize {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineSpeakerDiarization(self.sd);
        }
    }
}
