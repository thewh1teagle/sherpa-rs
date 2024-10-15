use crate::get_default_provider;
use eyre::Result;
use std::{ffi::CString, path::Path};

#[derive(Debug)]
pub struct VadConfig {
    pub(crate) cfg: sherpa_rs_sys::SherpaOnnxVadModelConfig,
}

#[derive(Debug)]
pub struct Vad {
    pub(crate) vad: *mut sherpa_rs_sys::SherpaOnnxVoiceActivityDetector,
}

#[derive(Debug)]
pub struct UserVadConfig {
    pub min_silence_duration: f32,
    pub min_speech_duration: f32,
    pub max_speech_duration: f32,
    pub threshold: f32,
    pub sample_rate: i32,
    pub window_size: i32,
    pub provider: Option<String>,
    pub num_threads: Option<i32>,
    pub debug: Option<bool>,
}

impl Default for UserVadConfig {
    fn default() -> Self {
        Self {
            min_silence_duration: 0.5,
            min_speech_duration: 0.5,
            max_speech_duration: 0.5,
            threshold: 0.5,
            sample_rate: 16000,
            window_size: 512,
            provider: None,
            num_threads: Some(1),
            debug: Some(false),
        }
    }
}

impl VadConfig {
    pub fn new<P: AsRef<Path>>(model: P, user_config: UserVadConfig) -> Self {
        let provider = user_config.provider.unwrap_or(get_default_provider());
        let provider = CString::new(provider).unwrap();
        let model = CString::new(model.as_ref().to_str().unwrap()).unwrap();

        let silero_vad = sherpa_rs_sys::SherpaOnnxSileroVadModelConfig {
            model: model.into_raw(),
            min_silence_duration: user_config.min_silence_duration,
            min_speech_duration: user_config.min_speech_duration,
            threshold: user_config.threshold,
            window_size: user_config.window_size,
            max_speech_duration: user_config.max_speech_duration,
        };
        let debug = user_config.debug.unwrap_or(false);
        let debug = if debug { 1 } else { 0 };
        let cfg = sherpa_rs_sys::SherpaOnnxVadModelConfig {
            debug,
            provider: provider.into_raw(),
            num_threads: user_config.num_threads.unwrap_or(1),
            sample_rate: user_config.sample_rate,
            silero_vad,
        };
        Self { cfg }
    }

    pub fn as_ptr(&self) -> *const sherpa_rs_sys::SherpaOnnxVadModelConfig {
        &self.cfg
    }
}

#[derive(Debug)]
pub struct SpeechSegment {
    pub start: i32,
    pub samples: Vec<f32>,
}

impl Vad {
    pub fn new_from_config(config: VadConfig, buffer_size_in_seconds: f32) -> Result<Self> {
        unsafe {
            let vad = sherpa_rs_sys::SherpaOnnxCreateVoiceActivityDetector(
                config.as_ptr(),
                buffer_size_in_seconds,
            );
            Ok(Self { vad })
        }
    }

    pub fn is_empty(&mut self) -> bool {
        unsafe { sherpa_rs_sys::SherpaOnnxVoiceActivityDetectorEmpty(self.vad) == 1 }
    }

    pub fn front(&mut self) -> SpeechSegment {
        unsafe {
            let segment_ptr = sherpa_rs_sys::SherpaOnnxVoiceActivityDetectorFront(self.vad);
            let raw_segment = segment_ptr.read();
            let samples: &[f32] =
                std::slice::from_raw_parts(raw_segment.samples, raw_segment.n as usize);

            let segment = SpeechSegment {
                samples: samples.to_vec(),
                start: raw_segment.start,
            };

            // Free
            sherpa_rs_sys::SherpaOnnxDestroySpeechSegment(segment_ptr);

            segment
        }
    }

    pub fn flush(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxVoiceActivityDetectorFlush(self.vad);
        }
    }

    pub fn accept_waveform(&mut self, mut samples: Vec<f32>) {
        let samples_ptr = samples.as_mut_ptr();
        let samples_length = samples.len();
        unsafe {
            sherpa_rs_sys::SherpaOnnxVoiceActivityDetectorAcceptWaveform(
                self.vad,
                samples_ptr,
                samples_length.try_into().unwrap(),
            );
        };
    }

    pub fn pop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxVoiceActivityDetectorPop(self.vad);
        }
    }

    pub fn is_speech(&mut self) -> bool {
        unsafe { sherpa_rs_sys::SherpaOnnxVoiceActivityDetectorDetected(self.vad) == 1 }
    }

    pub fn clear(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxVoiceActivityDetectorClear(self.vad);
        }
    }
}

unsafe impl Send for Vad {}
unsafe impl Sync for Vad {}

impl Drop for Vad {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyVoiceActivityDetector(self.vad);
        }
    }
}
