use crate::recognizer::online_recognizer::OnlineRecognizer;
use sherpa_rs_sys::{
    SherpaOnnxDestroyOnlineStream, SherpaOnnxOnlineStream, SherpaOnnxOnlineStreamAcceptWaveform,
    SherpaOnnxOnlineStreamInputFinished,
};
use std::marker::PhantomData;

/// The online stream class. It wraps a pointer from C.
pub struct OnlineStream<T: State> {
    pub(crate) pointer: *const SherpaOnnxOnlineStream,
    pub(crate) _marker: PhantomData<T>,
}

pub trait State {}

pub struct InitialState;
pub struct InputFinishedCalledState;

impl State for InitialState {}

impl State for InputFinishedCalledState {}

/// Delete the internal pointer inside the stream to avoid memory leak.
impl<T: State> Drop for OnlineStream<T> {
    fn drop(&mut self) {
        self.delete();
    }
}

impl<T: State> OnlineStream<T> {
    /// Signal that there will be no incoming audio samples.
    /// After calling this function, you cannot call [OnlineStream.AcceptWaveform] any longer.
    ///
    /// The main purpose of this function is to flush the remaining audio samples
    /// buffered inside for feature extraction.
    pub fn input_finished(self) -> OnlineStream<InputFinishedCalledState> {
        unsafe {
            SherpaOnnxOnlineStreamInputFinished(self.pointer);
        }
        OnlineStream {
            pointer: self.pointer,
            _marker: PhantomData,
        }
    }

    /// Delete the internal pointer inside the stream to avoid memory leak.
    fn delete(&mut self) {
        unsafe {
            SherpaOnnxDestroyOnlineStream(self.pointer);
        }
    }
}

impl OnlineStream<InitialState> {
    /// The user is responsible to invoke [DeleteOnlineStream]() to free
    /// the returned stream to avoid memory leak
    pub fn new(recognizer: &OnlineRecognizer) -> Self {
        recognizer.new_stream()
    }

    /// Input audio samples for the stream.
    ///
    /// sampleRate is the actual sample rate of the input audio samples. If it
    /// is different from the sample rate expected by the feature extractor, we will
    /// do resampling inside.
    ///
    /// samples contains audio samples. Each sample is in the range [-1, 1]
    pub fn accept_waveform(&self, sample_rate: i32, samples: &[f32]) {
        unsafe {
            SherpaOnnxOnlineStreamAcceptWaveform(
                self.pointer,
                sample_rate,
                samples.as_ptr(),
                samples.len() as i32,
            );
        }
    }
}
