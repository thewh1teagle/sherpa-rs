//! Online (streaming) speech recognition module
//!
//! This module provides real-time streaming speech recognition capabilities.
//! Unlike offline recognition which processes complete audio files,
//! streaming recognition can process audio chunks as they arrive.
//!
//! # Example
//!
//! ```ignore
//! use sherpa_rs::streaming::{
//!     OnlineRecognizer, OnlineRecognizerConfig,
//!     OnlineModelType, OnlineTransducerModelConfig,
//! };
//!
//! let config = OnlineRecognizerConfig {
//!     model: OnlineModelType::Transducer(OnlineTransducerModelConfig {
//!         encoder: "encoder.onnx".into(),
//!         decoder: "decoder.onnx".into(),
//!         joiner: "joiner.onnx".into(),
//!     }),
//!     tokens: "tokens.txt".into(),
//!     ..Default::default()
//! };
//!
//! let recognizer = OnlineRecognizer::new(config).unwrap();
//! let mut stream = recognizer.create_stream().unwrap();
//!
//! // Feed audio chunks
//! stream.accept_waveform(16000, &samples);
//!
//! // Decode when ready
//! while stream.is_ready() {
//!     stream.decode();
//! }
//!
//! // Get result
//! let result = stream.get_result();
//! println!("Text: {}", result.text);
//! ```

mod config;
mod recognizer;
mod stream;

pub use config::*;
pub use recognizer::OnlineRecognizer;
pub use stream::{OnlineRecognizerResult, OnlineStream};

