use cmake::Config;
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let sherpa_dst = out_dir.join("sherpa-onnx");
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("Failed to get CARGO_MANIFEST_DIR");
    let sherpa_src = Path::new(&manifest_dir).join("sherpa-onnx");

    if !sherpa_dst.exists() {
        std::fs::create_dir_all(&sherpa_dst).expect("Failed to create sherpa-onnx directory");

        // There's some invalid files. better to use cp
        #[cfg(unix)]
        {
            std::process::Command::new("cp")
                .arg("-rf")
                .arg(sherpa_src.clone())
                .arg(out_dir.clone())
                .status()
                .expect("Failed to execute cp command");
        }

        #[cfg(windows)]
        {
            std::process::Command::new("cmd")
                .args(&[
                    "/C",
                    "xcopy",
                    "/E",
                    "/I",
                    "/Y",
                    "/H",
                    sherpa_src.to_str().unwrap(),
                    sherpa_dst.to_str().unwrap(),
                ])
                .status()
                .expect("Failed to execute xcopy command");
        }
    }

    // Speed up build
    env::set_var(
        "CMAKE_BUILD_PARALLEL_LEVEL",
        std::thread::available_parallelism()
            .unwrap()
            .get()
            .to_string(),
    );

    // Set up bindgen builder
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", sherpa_dst.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Failed to generate bindings");

    // Write the generated bindings to an output file
    let bindings_path = out_dir.join("bindings.rs");
    bindings
        .write_to_file(bindings_path)
        .expect("Failed to write bindings");

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=./sherpa-onnx");

    let mut config = Config::new(&sherpa_dst);

    config
        .profile("Release")
        .define("SHERPA_ONNX_ENABLE_C_API", "ON")
        .define("SHERPA_ONNX_ENABLE_BINARY", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("SHERPA_ONNX_ENABLE_WEBSOCKET", "OFF");

    // TTS
    config.define(
        "SHERPA_ONNX_ENABLE_TTS",
        if cfg!(feature = "tts") { "ON" } else { "OFF" },
    );

    // Cuda
    // https://k2-fsa.github.io/k2/installation/cuda-cudnn.html
    #[cfg(feature = "cuda")]
    {
        config.define("SHERPA_ONNX_ENABLE_GPU", "ON");
    }

    #[cfg(any(windows, target_os = "linux"))]
    {
        config.define("SHERPA_ONNX_ENABLE_PORTAUDIO", "ON");
    }

    let bindings_dir = config.very_verbose(true).build();

    // Common
    println!("cargo:rustc-link-search={}", out_dir.join("lib").display());
    println!("cargo:rustc-link-search=native={}", bindings_dir.display());
    println!("cargo:rustc-link-lib=static=onnxruntime");
    println!("cargo:rustc-link-lib=static=kaldi-native-fbank-core");
    println!("cargo:rustc-link-lib=static=sherpa-onnx-core");
    println!("cargo:rustc-link-lib=static=sherpa-onnx-c-api");

    // macOS
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=static=sherpa-onnx-kaldifst-core");
        println!("cargo:rustc-link-lib=static=kaldi-decoder-core");
        println!("cargo:rustc-link-lib=static=sherpa-onnx-fst");
        println!("cargo:rustc-link-lib=static=sherpa-onnx-fstfar");
        println!("cargo:rustc-link-lib=static=ssentencepiece_core");
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
