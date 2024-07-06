use crate::get_default_provider;
use eyre::Result;
use std::ffi::CString;

#[derive(Debug)]
pub struct VadConfig {
    pub(crate) cfg: sherpa_rs_sys::SherpaOnnxVadModelConfig,
}

#[derive(Debug)]
pub struct Vad {
    pub(crate) vad: *mut sherpa_rs_sys::SherpaOnnxVoiceActivityDetector,
}

impl VadConfig {
    pub fn new(
        model: String,
        min_silence_duration: f32,
        min_speech_duration: f32,
        threshold: f32,
        sample_rate: i32,
        window_size: i32,
        provider: Option<String>,
        num_threads: Option<i32>,
        debug: Option<bool>,
    ) -> Self {
        let provider = provider.unwrap_or(get_default_provider());
        let provider = CString::new(provider).unwrap();
        let model = CString::new(model).unwrap();

        let silero_vad = sherpa_rs_sys::SherpaOnnxSileroVadModelConfig {
            model: model.into_raw(),
            min_silence_duration,
            min_speech_duration,
            threshold,
            window_size,
        };
        let debug = debug.unwrap_or(false);
        let debug = if debug { 1 } else { 0 };
        let cfg = sherpa_rs_sys::SherpaOnnxVadModelConfig {
            debug,
            provider: provider.into_raw(),
            num_threads: num_threads.unwrap_or(2),
            sample_rate,
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
    pub stop: i32,
    pub samples: *mut f32,
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
        unsafe { sherpa_rs_sys::SherpaOnnxVoiceActivityDetectorEmpty(self.vad) != 0 }
    }

    pub fn front(&mut self, sample_rate: i32) -> SpeechSegment {
        unsafe {
            let segment_ptr = sherpa_rs_sys::SherpaOnnxVoiceActivityDetectorFront(self.vad);
            let raw_segment = segment_ptr.read();
            // Free
            sherpa_rs_sys::SherpaOnnxDestroySpeechSegment(segment_ptr);
            let stop = raw_segment.start + (raw_segment.n / sample_rate);

            SpeechSegment {
                samples: raw_segment.samples,
                start: raw_segment.start,
                stop: stop,
            }
        }
    }

    pub fn accept_waveform(&mut self, samples: Vec<f32>) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxVoiceActivityDetectorAcceptWaveform(
                self.vad,
                samples.as_ptr(),
                samples.len().try_into().unwrap(),
            )
        };
    }

    pub fn pop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxVoiceActivityDetectorPop(self.vad);
        }
    }
}

impl Drop for Vad {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyVoiceActivityDetector(self.vad);
        }
    }
}
