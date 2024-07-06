use cmake::Config;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let sherpa_root = out.join("sherpa-onnx");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("Failed to get CARGO_MANIFEST_DIR");

    let sherpa_onnx_path = Path::new(&manifest_dir).join("sherpa-onnx");
    if !sherpa_root.exists() {
        std::fs::create_dir_all(&sherpa_root).expect("Failed to create sherpa-onnx directory");

        #[cfg(windows)]
        {
            fs_extra::dir::copy(sherpa_onnx_path.clone(), &out, &Default::default())
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed to copy sherpa sources from {} into {}: {}",
                        sherpa_onnx_path.display(),
                        sherpa_root.display(),
                        e
                    )
                });
        }

        // There's some invalid files. better to use cp
        #[cfg(unix)]
        {
            Command::new("cp")
                .arg("-rf")
                .arg(sherpa_onnx_path.clone())
                .arg(out.clone())
                .status()
                .expect("Failed to execute cp command");
        }
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
        .define("SHERPA_ONNX_ENABLE_BINARY", "OFF");

    // TTS
    config.define(
        "SHERPA_ONNX_ENABLE_TTS",
        if cfg!(feature = "tts") { "ON" } else { "OFF" },
    );

    // Cuda
    // https://k2-fsa.github.io/k2/installation/cuda-cudnn.html
    #[cfg(feature = "cuda")]
    {
        config.define("SHERPA_ONNX_ENABLE_GPU", "ON")
    }

    #[cfg(windows)]
    {
        config.define("SHERPA_ONNX_ENABLE_PORTAUDIO", "ON");
    }

    let destination = config.very_verbose(true).build();

    // Common
    println!("cargo:rustc-link-search=native={}", destination.display());
    println!("cargo:rustc-link-search={}", out.join("lib").display());
    println!("cargo:rustc-link-lib=static=sherpa-onnx-c-api");
    println!("cargo:rustc-link-lib=static=sherpa-onnx-core");
    println!("cargo:rustc-link-lib=static=onnxruntime");
    println!("cargo:rustc-link-lib=static=kaldi-native-fbank-core");

    // macOS
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=c++");
    }

    // Linux
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // Linux and Windows
    #[cfg(any(target_os = "linux", windows))]
    {
        println!("cargo:rustc-link-lib=static=kaldi-decoder-core");
        println!("cargo:rustc-link-lib=static=sherpa-onnx-kaldifst-core");
        println!("cargo:rustc-link-lib=static=sherpa-onnx-fst");
        println!("cargo:rustc-link-lib=static=sherpa-onnx-fstfar");
        println!("cargo:rustc-link-lib=static=ssentencepiece_core");
    }

    // TTS
    #[cfg(feature = "tts")]
    {
        println!("cargo:rustc-link-lib=static=espeak-ng");
        println!("cargo:rustc-link-lib=static=kaldi-decoder-core");
        println!("cargo:rustc-link-lib=static=sherpa-onnx-kaldifst-core");
        println!("cargo:rustc-link-lib=static=sherpa-onnx-fst");
        println!("cargo:rustc-link-lib=static=sherpa-onnx-fstfar");
        println!("cargo:rustc-link-lib=static=ssentencepiece_core");
        println!("cargo:rustc-link-lib=static=piper_phonemize");
        println!("cargo:rustc-link-lib=static=ucd");
    }
}
