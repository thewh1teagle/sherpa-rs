use once_cell::sync::OnceCell;
use std::sync::{Arc, Mutex};
use tauri::Manager;


type PunctuatorHandleType = Arc<Mutex<sherpa_rs::punctuate::Punctuation>>;
static PUNCTUATOR: OnceCell<Result<PunctuatorHandleType, String>> =
    OnceCell::new();


#[tauri::command]
fn punctuate(sentence: &str, app: tauri::AppHandle) -> Result<String, String> {
    println!("Input text: {}", sentence);

    // Initialize the Punctuation object if it hasn't been done already
    let punctuater = PUNCTUATOR.get_or_init(|| {

        // TODO: download the model? or load it from downloads?
        // let resource_dir = app.path().resource_dir().map_err(|e| e.to_string())?;
        // let model_path = resource_dir.join("model.onnx");
        // let temp_model_path = std::env::temp_dir().join(&model_path);
        // std::fs::copy(&model_path, std::env::temp_dir().join(&model_path)).map_err(|e| e.to_string())?;

        let config = sherpa_rs::punctuate::PunctuationConfig {
            // model: temp_model_path.to_string_lossy().to_string(),
            model: "/data/local/tmp/model.onnx".to_string(),
            ..Default::default()
        };
        let punctuator = sherpa_rs::punctuate::Punctuation::new(config)
            .map_err(|e| format!("Failed to initialize Punctuation object: {}", e))?;

        Ok(Arc::new(Mutex::new(punctuator)))
    });

    // Lock the mutex to safely access the Punctuation object
    let punctuater = punctuater.clone().map_err(|e| e.to_string())?;
    let mut punctuater = punctuater
        .lock()
        .map_err(|e| format!("Failed to get punctuator: {:?}", e))?;
    let punctuated = punctuater.add_punctuation(sentence);

    Ok(punctuated)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![punctuate])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
