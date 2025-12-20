use axum::{
    extract::{Multipart, State},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use sherpa_rs::{
    whisper::{WhisperConfig, WhisperRecognizer},
    tts::{KokoroTts, KokoroTtsConfig},
};
use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct TtsRequest {
    text: String,
    sid: Option<i32>,
}

#[derive(Serialize)]
struct AsrResponse {
    text: String,
}

// AppState to share AI models across requests
struct AppState {
    recognizer: Mutex<WhisperRecognizer>,
    tts: Mutex<KokoroTts>,
}

#[tokio::main]
async fn main() {
    // 1. Whisper Config (ASR) - Using split paths for ONNX models
    let whisper_config = WhisperConfig {
        encoder: "./sherpa-onnx-whisper-tiny/tiny-encoder.onnx".into(),
        decoder: "./sherpa-onnx-whisper-tiny/tiny-decoder.onnx".into(),
        tokens: "./sherpa-onnx-whisper-tiny/tiny-tokens.txt".into(),
        ..Default::default()
    };
    
    // 2. Kokoro Config (TTS)
    let tts_config = KokoroTtsConfig {
        model: "./kokoro-multi-lang-v1_0/model.onnx".into(),
        voices: "./kokoro-multi-lang-v1_0/voices.bin".into(),
        tokens: "./kokoro-multi-lang-v1_0/tokens.txt".into(),
        data_dir: "./kokoro-multi-lang-v1_0/espeak-ng-data".into(),
        dict_dir: "./kokoro-multi-lang-v1_0/dict".into(),
        lexicon: "./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt".into(),
        ..Default::default()
    };

    // Initialize models and wrap in thread-safe State
    let state = Arc::new(AppState {
        recognizer: Mutex::new(WhisperRecognizer::new(whisper_config).expect("Failed to init Whisper")),
        tts: Mutex::new(KokoroTts::new(tts_config)),
    });

    // 3. Setup Router
    let app = Router::new()
        .route("/asr", post(handle_asr))
        .route("/tts", post(handle_tts))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("🚀 Sherpa-RS HCI Bridge Engine Online | http://0.0.0.0:8080");
    axum::serve(listener, app).await.unwrap();
}

// Handler for Transcribing Speech (ASR)
async fn handle_asr(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    if let Some(field) = multipart.next_field().await.unwrap() {
        let bytes = field.bytes().await.unwrap();
        
        // Explicitly map bytes to f32 to avoid type inference errors
        let samples: Vec<f32> = bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        
        let mut recognizer = state.recognizer.lock().await;
        let result = recognizer.transcribe(16000, &samples);
        return Json(AsrResponse { text: result.text }).into_response();
    }
    (axum::http::StatusCode::BAD_REQUEST, "Missing audio field").into_response()
}

// Handler for Synthesizing Speech (TTS)
async fn handle_tts(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TtsRequest>,
) -> impl IntoResponse {
    let mut tts = state.tts.lock().await;
    
    // Use &req.text to pass as &str to satisfy the compiler
    let audio = tts.create(&req.text, req.sid.unwrap_or(0), 1.0).expect("TTS Failed");
    
    // Convert f32 samples to little-endian bytes for the response
    let pcm_bytes: Vec<u8> = audio.samples
        .iter()
        .flat_map(|sample: &f32| sample.to_le_bytes())
        .collect();

    Response::builder()
        .header("Content-Type", "audio/pcm")
        .body(axum::body::Body::from(pcm_bytes))
        .unwrap()
}
