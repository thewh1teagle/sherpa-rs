use cmake::Config;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let target = env::var("TARGET").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let sherpa_dst = out_dir.join("sherpa-onnx");
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("Failed to get CARGO_MANIFEST_DIR");
    let sherpa_src = Path::new(&manifest_dir).join("sherpa-onnx");

    // Prepare sherpa-onnx source
    if !sherpa_dst.exists() {
        std::fs::create_dir_all(&sherpa_dst).expect("Failed to create sherpa-onnx directory");

        // There's some invalid files. better to use cp
        if cfg!(unix) {
            std::process::Command::new("cp")
                .arg("-rf")
                .arg(sherpa_src.clone())
                .arg(out_dir.clone())
                .status()
                .expect("Failed to execute cp command");
        }

        if cfg!(windows) {
            std::process::Command::new("robocopy.exe")
                .args(&[
                    "/e",
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

    // Bindings

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

    // Build with Cmake

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
    if cfg!(feature = "cuda") {
        config.define("SHERPA_ONNX_ENABLE_GPU", "ON");
        config.define("BUILD_SHARED_LIBS", "ON");
    }

    if cfg!(any(windows, target_os = "linux")) {
        config.define("SHERPA_ONNX_ENABLE_PORTAUDIO", "ON");
    }

    let bindings_dir = config.very_verbose(true).build();

    // Search paths
    println!("cargo:rustc-link-search={}", out_dir.join("lib").display());
    println!("cargo:rustc-link-search=native={}", bindings_dir.display());

    if cfg!(feature = "cuda") && cfg!(windows) {
        println!(
            "cargo:rustc-link-search=native={}",
            out_dir.join("build\\_deps\\onnxruntime-src\\lib").display()
        );
    }

    // Link libraries

    println!("cargo:rustc-link-lib=static=onnxruntime");

    // Sherpa API
    println!("cargo:rustc-link-lib=static=kaldi-native-fbank-core");
    println!("cargo:rustc-link-lib=static=sherpa-onnx-core");
    println!("cargo:rustc-link-lib=static=sherpa-onnx-c-api");
    println!("cargo:rustc-link-lib=static=kaldi-decoder-core");
    println!("cargo:rustc-link-lib=static=sherpa-onnx-kaldifst-core");
    println!("cargo:rustc-link-lib=static=sherpa-onnx-fstfar");
    println!("cargo:rustc-link-lib=static=ssentencepiece_core");

    // Cuda
    if cfg!(feature = "cuda") && cfg!(windows) {
        println!("cargo:rustc-link-lib=static=onnxruntime_providers_cuda");
        println!("cargo:rustc-link-lib=static=onnxruntime_providers_shared");
        println!("cargo:rustc-link-lib=static=onnxruntime_providers_tensorrt");
    }

    // macOS
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=c++");
    }

    // Linux
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // TTS
    if cfg!(feature = "tts") {
        println!("cargo:rustc-link-lib=static=espeak-ng");
        println!("cargo:rustc-link-lib=static=piper_phonemize");
        println!("cargo:rustc-link-lib=static=ucd");
    }

    if target.contains("apple") {
        // On (older) OSX we need to link against the clang runtime,
        // which is hidden in some non-default path.
        //
        // More details at https://github.com/alexcrichton/curl-rust/issues/279.
        if let Some(path) = macos_link_search_path() {
            println!("cargo:rustc-link-lib=clang_rt.osx");
            println!("cargo:rustc-link-search={}", path);
        }
    }
}

fn macos_link_search_path() -> Option<String> {
    let output = Command::new("clang")
        .arg("--print-search-dirs")
        .output()
        .ok()?;
    if !output.status.success() {
        println!(
            "failed to run 'clang --print-search-dirs', continuing without a link search path"
        );
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.contains("libraries: =") {
            let path = line.split('=').nth(1)?;
            return Some(format!("{}/lib/darwin", path));
        }
    }

    println!("failed to determine link search path, continuing without it");
    None
}
