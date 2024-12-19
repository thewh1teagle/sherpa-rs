use crate::recognizer::offline_recognizer::{OfflineRecognizer, OfflineRecognizerResult};
use crate::utils;
use sherpa_rs_sys::{
    SherpaOnnxAcceptWaveformOffline, SherpaOnnxDestroyOfflineRecognizerResult,
    SherpaOnnxDestroyOfflineStream, SherpaOnnxGetOfflineStreamResult, SherpaOnnxOfflineStream,
};

/// It wraps a pointer from C
pub struct OfflineStream {
    pub(crate) pointer: *const SherpaOnnxOfflineStream,
}

impl Drop for OfflineStream {
    fn drop(&mut self) {
        self.delete()
    }
}

impl OfflineStream {
    /// Frees the internal pointer of the stream to avoid memory leak.
    fn delete(&mut self) {
        unsafe {
            SherpaOnnxDestroyOfflineStream(self.pointer);
        }
    }

    /// The user is responsible to invoke [Self::drop] to free
    /// the returned stream to avoid memory leak.
    pub fn new(recognizer: &OfflineRecognizer) -> OfflineStream {
        recognizer.new_stream()
    }

    /// Input audio samples for the offline stream.
    /// Please only call it once. That is, input all samples at once.
    ///
    /// `sample_rate` is the sample rate of the input audio samples. If it is different
    /// from the value expected by the feature extractor, we will do resampling inside.
    ///
    /// `samples` contains the actual audio samples. Each sample is in the range [-1, 1].
    pub fn accept_waveform(&mut self, sample_rate: i32, samples: &[f32]) {
        unsafe {
            SherpaOnnxAcceptWaveformOffline(
                self.pointer,
                sample_rate,
                samples.as_ptr(),
                samples.len() as i32,
            );
        }
    }

    /// Get the recognition result of the offline stream.
    pub fn get_result(&self) -> Option<OfflineRecognizerResult> {
        let p = unsafe { SherpaOnnxGetOfflineStreamResult(self.pointer) };
        if p.is_null() {
            return None;
        }
        let n = unsafe { (*p).count } as usize;
        if n == 0 {
            return None;
        }
        let text = utils::cstr_to_string((unsafe { *p }).text);
        let lang = utils::cstr_to_string((unsafe { *p }).lang);
        let emotion = utils::cstr_to_string((unsafe { *p }).emotion);
        let event = utils::cstr_to_string((unsafe { *p }).event);
        let mut result = OfflineRecognizerResult {
            text,
            lang,
            emotion,
            event,
            tokens: Vec::with_capacity(n),
            timestamps: Vec::with_capacity(n),
        };
        let tokens = unsafe { std::slice::from_raw_parts((*p).tokens_arr, n) };
        for &token in tokens {
            result.tokens.push(utils::cstr_to_string(token));
        }
        unsafe {
            if !(*p).timestamps.is_null() {
                let timestamps = std::slice::from_raw_parts((*p).timestamps, n);
                for &timestamp in timestamps {
                    result.timestamps.push(timestamp);
                }
            }
        }
        unsafe {
            SherpaOnnxDestroyOfflineRecognizerResult(p);
        }
        Some(result)
    }
}
