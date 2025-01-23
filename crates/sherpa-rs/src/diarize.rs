use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::{path::Path, ptr::null_mut};

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

type ProgressCallback = Box<dyn (Fn(i32, i32) -> i32) + Send + 'static>;

#[derive(Debug, Clone)]
pub struct DiarizeConfig {
    pub num_clusters: Option<i32>,
    pub threshold: Option<f32>,
    pub min_duration_on: Option<f32>,
    pub min_duration_off: Option<f32>,
    pub provider: Option<String>,
    pub debug: bool,
}

impl Default for DiarizeConfig {
    fn default() -> Self {
        Self {
            num_clusters: Some(4),
            threshold: Some(0.5),
            min_duration_on: Some(0.0),
            min_duration_off: Some(0.0),
            provider: None,
            debug: false,
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

        let debug = config.debug;
        let debug = if debug { 1 } else { 0 };

        let embedding_model = embedding_model.as_ref().to_str().unwrap();
        let segmentation_model = segmentation_model.as_ref().to_str().unwrap();

        let clustering_config = sherpa_rs_sys::SherpaOnnxFastClusteringConfig {
            num_clusters: config.num_clusters.unwrap_or(4),
            threshold: config.threshold.unwrap_or(0.5),
        };

        let embedding_model = cstring_from_str(embedding_model);
        let provider = cstring_from_str(&provider.clone());
        let segmentation_model = cstring_from_str(segmentation_model);

        let config = sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationConfig {
            embedding: sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorConfig {
                model: embedding_model.as_ptr(),
                num_threads: 1,
                debug,
                provider: provider.as_ptr(),
            },
            clustering: clustering_config,
            min_duration_off: config.min_duration_off.unwrap_or(0.0),
            min_duration_on: config.min_duration_on.unwrap_or(0.0),
            segmentation: sherpa_rs_sys::SherpaOnnxOfflineSpeakerSegmentationModelConfig {
                pyannote: sherpa_rs_sys::SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig {
                    model: segmentation_model.as_ptr(),
                },
                num_threads: 1,
                debug,
                provider: provider.as_ptr(),
            },
        };

        let sd = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineSpeakerDiarization(&config) };

        if sd.is_null() {
            bail!("Failed to initialize offline speaker diarization");
        }
        Ok(Self { sd })
    }

    pub fn compute(
        &mut self,
        mut samples: Vec<f32>,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<Vec<Segment>> {
        let samples_ptr = samples.as_mut_ptr();
        let mut segments = Vec::new();
        unsafe {
            let mut callback_box =
                progress_callback.map(|cb| Box::new(cb) as Box<ProgressCallback>);
            let callback_ptr = callback_box
                .as_mut()
                .map(|b| b.as_mut() as *mut ProgressCallback as *mut std::ffi::c_void)
                .unwrap_or(null_mut());

            let result = sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
                self.sd,
                samples_ptr,
                samples.len() as i32,
                if callback_box.is_some() {
                    Some(progress_callback_wrapper)
                } else {
                    None
                },
                callback_ptr,
            );

            let num_segments =
                sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(result);
            let segments_ptr: *const sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationSegment =
                sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(result);

            if !segments_ptr.is_null() && num_segments > 0 {
                let segments_result: &[
                    sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationSegment
                ] = std::slice::from_raw_parts(segments_ptr, num_segments as usize);

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

unsafe extern "C" fn progress_callback_wrapper(
    num_processed_chunk: i32,
    num_total_chunks: i32,
    arg: *mut std::ffi::c_void,
) -> i32 {
    let callback = &mut *(arg as *mut ProgressCallback);
    callback(num_processed_chunk, num_total_chunks)
}

unsafe impl Send for Diarize {}
unsafe impl Sync for Diarize {}

impl Drop for Diarize {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineSpeakerDiarization(self.sd);
        }
    }
}
