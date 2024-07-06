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
        let default_provider = if cfg!(target_os = "macos") {
            "coreml"
        } else {
            "cpu"
        };
        let provider = provider.unwrap_or(default_provider.into());
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

    pub fn front(&mut self) -> sherpa_rs_sys::SherpaOnnxSpeechSegment {
        unsafe {
            let segment_ptr = sherpa_rs_sys::SherpaOnnxVoiceActivityDetectorFront(self.vad);
            let segment = segment_ptr.read();
            sherpa_rs_sys::SherpaOnnxDestroySpeechSegment(segment_ptr);
            return segment;
        };
    }

    pub fn pop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxVoiceActivityDetectorPop(self.vad);
        }
    }
}
