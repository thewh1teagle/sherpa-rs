extern crate bindgen;

use cmake::Config;
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let sherpa_root = out.join("sherpa-onnx");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("Failed to get CARGO_MANIFEST_DIR");

    let sherpa_onnx_path = Path::new(&manifest_dir).join("sherpa-onnx");
    if !sherpa_root.exists() {
        std::fs::create_dir_all(&sherpa_root).expect("Failed to create sherpa-onnx directory");

        // Two problematic git files
        // Otherwise copy will fail
        #[cfg(not(windows))]
        {
            let path_to_remove = sherpa_onnx_path
                .join("scripts")
                .join("go")
                .join("_internal")
                .join("vad-spoken-language-identification");
            std::fs::remove_file(path_to_remove.join("run.sh")).unwrap();
            std::fs::remove_file(path_to_remove.join("main.go")).unwrap();
        }
        fs_extra::dir::copy(sherpa_onnx_path.clone(), &out, &Default::default()).unwrap_or_else(
            |e| {
                panic!(
                    "Failed to copy sherpa sources from {} into {}: {}",
                    sherpa_onnx_path.display(),
                    sherpa_root.display(),
                    e
                )
            },
        );
    }

    // Set up bindgen builder
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", sherpa_root.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Failed to generate bindings");

    // Write the generated bindings to an output file
    let out_path = out.join("bindings.rs");
    bindings
        .write_to_file(out_path)
        .expect("Failed to write bindings");

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=./sherpa-onnx");

    let mut config = Config::new(&sherpa_root);
    config
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("SHERPA_ONNX_ENABLE_C_API", "ON")
        .define("SHERPA_ONNX_ENABLE_WEBSOCKET", "OFF")
        .define("SHERPA_ONNX_ENABLE_BINARY", "ON")
        .define("SHERPA_ONNX_ENABLE_TTS", "OFF");

    #[cfg(windows)]
    {
        config.define("SHERPA_ONNX_ENABLE_PORTAUDIO", "ON");
    }

    let destination = config.
        very_verbose(true)
        .build();

    println!("cargo:rustc-link-search=native={}", destination.display());
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-search={}", out.join("lib").display());
    }
    #[cfg(windows)]
    {
        println!("cargo:rustc-link-search={}", out.join("lib").display());
    }

    
    println!("cargo:rustc-link-lib=static=sherpa-onnx-c-api");
    println!("cargo:rustc-link-lib=static=sherpa-onnx-core");
    println!("cargo:rustc-link-lib=static=onnxruntime");
    println!("cargo:rustc-link-lib=static=kaldi-native-fbank-core");
    

    #[cfg(windows)]
    {
        println!("cargo:rustc-link-lib=static=kaldi-decoder-core");
        println!("cargo:rustc-link-lib=static=sherpa-onnx-kaldifst-core");
        println!("cargo:rustc-link-lib=static=sherpa-onnx-fst");
        println!("cargo:rustc-link-lib=static=sherpa-onnx-fstfar");
        println!("cargo:rustc-link-lib=static=ssentencepiece_core");
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=c++");
    }
}
