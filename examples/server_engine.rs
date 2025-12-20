use axum::{
    extract::{Multipart, State},
    response::IntoResponse,
    routing::{post, get},
    Json, Router,
};
use sherpa_rs::{
    whisper::{WhisperConfig, WhisperRecognizer},
    tts::{KokoroTts, KokoroTtsConfig},
};
use std::sync::Arc;
use tokio::sync::Mutex;
use serde::Deserialize;

#[derive(Deserialize)]
struct TtsRequest {
    text: String,
    sid: Option<i32>,
}

struct AppState {
    recognizer: Mutex<WhisperRecognizer>,
    tts: Mutex<KokoroTts>,
}

#[tokio::main]
async fn main() {
    // Whisper Config (ASR)
    let whisper_config = WhisperConfig {
        model: "./sherpa-onnx-whisper-tiny/tiny.en.onnx".to_string(),
        tokens: "./sherpa-onnx-whisper-tiny/tiny.en-tokens.txt".to_string(),
        ..Default::default()
    };
    
    // Kokoro Config (TTS)
    let tts_config = KokoroTtsConfig {
        model: "./kokoro-multi-lang-v1_0/model.onnx".into(),
        voices: "./kokoro-multi-lang-v1_0/voices.bin".into(),
        tokens: "./kokoro-multi-lang-v1_0/tokens.txt".into(),
        data_dir: "./kokoro-multi-lang-v1_0/espeak-ng-data".into(),
        dict_dir: "./kokoro-multi-lang-v1_0/dict".into(),
        lexicon: "./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt".into(),
        ..Default::default()
    };

    let state = Arc::new(AppState {
        recognizer: Mutex::new(WhisperRecognizer::new(whisper_config).unwrap()),
        tts: Mutex::new(KokoroTts::new(tts_config).unwrap()),
    });

    let app = Router::new()
        .route("/asr", post(handle_asr))
        .route("/tts", post(handle_tts))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("🚀 Sherpa-RS HCI Bridge running on http://0.0.0.0:8080");
    axum::serve(listener, app).await.unwrap();
}

async fn handle_asr(State(state): State<Arc<AppState>>, mut multipart: Multipart) -> impl IntoResponse {
    if let Some(field) = multipart.next_field().await.unwrap() {
        let bytes = field.bytes().await.unwrap();
        let samples: Vec<f32> = bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        
        let mut recognizer = state.recognizer.lock().await;
        let result = recognizer.transcribe(16000, &samples);
        return Json(serde_json::json!({ "text": result.text })).into_response();
    }
    (axum::http::StatusCode::BAD_REQUEST, "No audio").into_response()
}

async fn handle_tts(State(state): State<Arc<AppState>>, Json(req): Json<TtsRequest>) -> impl IntoResponse {
    let mut tts = state.tts.lock().await;
    let audio = tts.create_speech(req.text, req.sid.unwrap_or(0), 1.0).unwrap();
    axum::response::Response::builder()
        .header("Content-Type", "audio/wav")
        .body(axum::body::Body::from(audio.to_vec())).unwrap()
}
