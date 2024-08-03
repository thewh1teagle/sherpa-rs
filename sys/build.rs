use cmake::Config;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn copy_folder(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).expect("Failed to create dst directory");
    if cfg!(unix) {
        std::process::Command::new("cp")
            .arg("-rf")
            .arg(src)
            .arg(dst.parent().unwrap())
            .status()
            .expect("Failed to execute cp command");
    }

    if cfg!(windows) {
        std::process::Command::new("robocopy.exe")
            .arg("/e")
            .arg(src)
            .arg(dst)
            .status()
            .expect("Failed to execute xcopy command");
    }
}

fn main() {
    let target = env::var("TARGET").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_dir = out_dir.join("../../../").canonicalize().unwrap();
    let sherpa_dst = out_dir.join("sherpa-onnx");
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("Failed to get CARGO_MANIFEST_DIR");
    let sherpa_src = Path::new(&manifest_dir).join("sherpa-onnx");
    let build_shared_libs = cfg!(feature = "directml") || cfg!(feature = "cuda");
    let profile = if cfg!(debug_assertions) {
        "Debug"
    } else {
        "Release"
    };

    let shared_lib_suffix = if cfg!(windows) {
        ".dll"
    } else if cfg!(target_os = "macos") {
        ".dylib"
    } else {
        ".so"
    };
    let sherpa_libs_kind = if build_shared_libs { "dylib" } else { "static" };
    let sherpa_libs: &[&str] = if build_shared_libs {
        // shared
        &["sherpa-onnx-c-api", "onnxruntime"]
    } else if cfg!(feature = "tts") {
        // static with tts
        &[
            "sherpa-onnx-c-api",
            "sherpa-onnx-core",
            "kaldi-decoder-core",
            "sherpa-onnx-kaldifst-core",
            "sherpa-onnx-fstfar",
            "sherpa-onnx-fst",
            "kaldi-native-fbank-core",
            "piper_phonemize",
            "espeak-ng",
            "ucd",
            "onnxruntime",
            "ssentencepiece_core",
        ]
    } else {
        // static without tts
        &[
            "sherpa-onnx-c-api",
            "sherpa-onnx-core",
            "kaldi-decoder-core",
            "sherpa-onnx-kaldifst-core",
            "sherpa-onnx-fst",
            "kaldi-native-fbank-core",
            "onnxruntime",
            "ssentencepiece_core",
        ]
    };

    // Prepare sherpa-onnx source
    if !sherpa_dst.exists() {
        copy_folder(&sherpa_src, &sherpa_dst);
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
        .define("SHERPA_ONNX_ENABLE_C_API", "ON")
        .define("SHERPA_ONNX_ENABLE_BINARY", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("SHERPA_ONNX_ENABLE_WEBSOCKET", "OFF")
        .define("SHERPA_ONNX_ENABLE_TTS", "OFF");

    if cfg!(windows) {
        config.static_crt(true);
    }

    // TTS
    if cfg!(feature = "tts") {
        config.define("SHERPA_ONNX_ENABLE_TTS", "ON");
    }

    // Cuda https://k2-fsa.github.io/k2/installation/cuda-cudnn.html
    if cfg!(feature = "cuda") {
        config.define("SHERPA_ONNX_ENABLE_GPU", "ON");
        config.define("BUILD_SHARED_LIBS", "ON");
    }

    // DirectML https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
    if cfg!(feature = "directml") {
        config.define("SHERPA_ONNX_ENABLE_DIRECTML", "ON");
        config.define("BUILD_SHARED_LIBS", "ON");
    }

    if cfg!(any(windows, target_os = "linux")) {
        config.define("SHERPA_ONNX_ENABLE_PORTAUDIO", "ON");
    }

    // General
    config
        .profile(profile)
        .very_verbose(false)
        .always_configure(false);

    let bindings_dir = config.build();

    // Search paths
    println!("cargo:rustc-link-search={}", out_dir.join("lib").display());
    println!("cargo:rustc-link-search=native={}", bindings_dir.display());

    // Cuda
    if cfg!(feature = "cuda") {
        println!(
            "cargo:rustc-link-search={}",
            out_dir.join(format!("build/lib/{}", profile)).display()
        );
        if cfg!(windows) {
            println!(
                "cargo:rustc-link-search=native={}",
                out_dir.join("build/_deps/onnxruntime-src/lib").display()
            );
        }
        if cfg!(target_os = "linux") {
            println!(
                "cargo:rustc-link-search=native={}",
                out_dir.join("build/lib").display()
            );
        }
    }

    // Link libraries
    for lib in sherpa_libs {
        println!(
            "{}",
            format!("cargo:rustc-link-lib={}={}", sherpa_libs_kind, lib)
        );
    }

    // Windows debug
    if cfg!(all(debug_assertions, windows)) {
        println!("cargo:rustc-link-lib=dylib=msvcrtd");
    }

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

    // copy DLLs to target
    if build_shared_libs {
        for entry in glob::glob(&format!(
            "{}/*{}",
            out_dir.join("lib").to_str().unwrap(),
            shared_lib_suffix
        ))
        .unwrap()
        .flatten()
        {
            let dst = target_dir.join(entry.file_name().unwrap());
            std::fs::copy(entry, dst).unwrap();
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
