//! Speech recognition with [Next-gen Kaldi].
//!
//! [sherpa-onnx] is an open-source speech recognition framework for [Next-gen Kaldi].
//! It depends only on [onnxruntime], supporting both streaming and non-streaming
//! speech recognition.
//!
//! It does not need to access the network during recognition and everything
//! runs locally.
//!
//! It supports a variety of platforms, such as Linux (x86_64, aarch64, arm),
//! Windows (x86_64, x86), macOS (x86_64, arm64), etc.
//!
//! Usage examples:
//!
//! 1. Real-time speech recognition from a microphone
//!
//!    Please see
//!    https://github.com/k2-fsa/sherpa-onnx/tree/master/go-api-examples/real-time-speech-recognition-from-microphone
//!
//! 2. Decode files using a non-streaming model
//!
//!    Please see
//!    https://github.com/k2-fsa/sherpa-onnx/tree/master/go-api-examples/non-streaming-decode-files
//!
//! 3. Decode files using a streaming model
//!
//!    Please see
//!    https://github.com/k2-fsa/sherpa-onnx/tree/master/go-api-examples/streaming-decode-files
//!
//! 4. Convert text to speech using a non-streaming model
//!
//!    Please see
//!    https://github.com/k2-fsa/sherpa-onnx/tree/master/go-api-examples/non-streaming-tts
//!
//! [sherpa-onnx]: https://github.com/k2-fsa/sherpa-onnx
//! [onnxruntime]: https://github.com/microsoft/onnxruntime
//! [Next-gen Kaldi]: https://github.com/k2-fsa/

pub mod offline_recognizer;
pub mod online_recognizer;
