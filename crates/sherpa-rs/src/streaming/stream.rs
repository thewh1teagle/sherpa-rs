use crate::utils::{cstr_to_string, cstring_from_str};
use eyre::{bail, Result};
use std::ffi::CStr;

/// Result from online (streaming) speech recognition
#[derive(Debug, Clone)]
pub struct OnlineRecognizerResult {
    /// Recognized text
    pub text: String,
    /// Decoded tokens
    pub tokens: Vec<String>,
    /// Timestamps for each token (in seconds)
    pub timestamps: Vec<f32>,
    /// JSON representation of the result
    pub json: String,
}

/// Online stream for accepting audio samples
pub struct OnlineStream {
    stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream,
    recognizer: *const sherpa_rs_sys::SherpaOnnxOnlineRecognizer,
}

impl OnlineStream {
    /// Create a new online stream
    pub(crate) fn new(
        recognizer: *const sherpa_rs_sys::SherpaOnnxOnlineRecognizer,
    ) -> Result<Self> {
        let stream = unsafe { sherpa_rs_sys::SherpaOnnxCreateOnlineStream(recognizer) };
        if stream.is_null() {
            bail!("Failed to create online stream");
        }
        Ok(Self { stream, recognizer })
    }

    /// Create a new online stream with custom hotwords
    pub(crate) fn new_with_hotwords(
        recognizer: *const sherpa_rs_sys::SherpaOnnxOnlineRecognizer,
        hotwords: &str,
    ) -> Result<Self> {
        let hotwords_ptr = cstring_from_str(hotwords);
        let stream = unsafe {
            sherpa_rs_sys::SherpaOnnxCreateOnlineStreamWithHotwords(
                recognizer,
                hotwords_ptr.as_ptr(),
            )
        };
        if stream.is_null() {
            bail!("Failed to create online stream with hotwords");
        }
        Ok(Self { stream, recognizer })
    }

    /// Accept input audio samples
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate of the input samples
    /// * `samples` - Audio samples normalized to [-1, 1]
    pub fn accept_waveform(&mut self, sample_rate: i32, samples: &[f32]) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamAcceptWaveform(
                self.stream,
                sample_rate,
                samples.as_ptr(),
                samples.len() as i32,
            );
        }
    }

    /// Signal that no more audio samples would be available
    ///
    /// After this call, you cannot call `accept_waveform()` anymore.
    pub fn input_finished(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamInputFinished(self.stream);
        }
    }

    /// Check if there are enough feature frames for decoding
    pub fn is_ready(&self) -> bool {
        unsafe {
            sherpa_rs_sys::SherpaOnnxIsOnlineStreamReady(self.recognizer, self.stream) == 1
        }
    }

    /// Run the neural network model and decoding
    ///
    /// Should only be called when `is_ready()` returns true.
    pub fn decode(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDecodeOnlineStream(self.recognizer, self.stream);
        }
    }

    /// Get the current recognition result
    pub fn get_result(&self) -> OnlineRecognizerResult {
        unsafe {
            let result_ptr =
                sherpa_rs_sys::SherpaOnnxGetOnlineStreamResult(self.recognizer, self.stream);
            let raw = result_ptr.read();

            let text = cstr_to_string(raw.text);
            let json = cstr_to_string(raw.json);

            // Parse tokens
            let count = raw.count as usize;
            let timestamps = if raw.timestamps.is_null() {
                Vec::new()
            } else {
                std::slice::from_raw_parts(raw.timestamps, count).to_vec()
            };

            let mut tokens = Vec::with_capacity(count);
            if !raw.tokens_arr.is_null() {
                for i in 0..count {
                    let token_ptr = *raw.tokens_arr.add(i);
                    let token = CStr::from_ptr(token_ptr).to_string_lossy().into_owned();
                    tokens.push(token);
                }
            }

            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizerResult(result_ptr);

            OnlineRecognizerResult {
                text,
                tokens,
                timestamps,
                json,
            }
        }
    }

    /// Check if an endpoint has been detected
    ///
    /// An endpoint is detected when:
    /// - Trailing silence exceeds rule1_min_trailing_silence (if nothing decoded)
    /// - Trailing silence exceeds rule2_min_trailing_silence (if something decoded)
    /// - Utterance length exceeds rule3_min_utterance_length
    pub fn is_endpoint(&self) -> bool {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamIsEndpoint(self.recognizer, self.stream) == 1
        }
    }

    /// Reset the stream
    ///
    /// Clears the neural network model state and decoding state.
    /// Should be called after detecting an endpoint to start a new utterance.
    pub fn reset(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamReset(self.recognizer, self.stream);
        }
    }
}

unsafe impl Send for OnlineStream {}
unsafe impl Sync for OnlineStream {}

impl Drop for OnlineStream {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOnlineStream(self.stream);
        }
    }
}

