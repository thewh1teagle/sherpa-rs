use cmake::Config;
use glob::glob;
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
            .expect("Failed to execute robocopy command");
    }
}

fn extract_lib_names(out_dir: &Path) -> Vec<String> {
    // Construct the pattern based on the target platform
    let lib_suffix = if cfg!(windows) { "*.lib" } else { "*.a" };
    let pattern = out_dir.join(format!("build/lib/{}", lib_suffix));

    let mut lib_names = Vec::new();

    // Process the libraries based on the pattern
    for entry in glob(pattern.to_str().unwrap()).unwrap() {
        match entry {
            Ok(path) => {
                let stem = path.file_stem().unwrap();
                let stem_str = stem.to_str().unwrap();

                // Remove the "lib" prefix if present
                let lib_name = if stem_str.starts_with("lib") {
                    stem_str.strip_prefix("lib").unwrap_or(stem_str)
                } else {
                    stem_str
                };

                lib_names.push(lib_name.to_string());
            }
            Err(e) => println!("cargo:warning=error={}", e),
        }
    }

    lib_names
}

fn extract_lib_assets(out_dir: &Path) -> Vec<PathBuf> {
    let shared_lib_pattern = if cfg!(windows) {
        "*.dll"
    } else if cfg!(target_os = "macos") {
        "*.dylib"
    } else {
        "*.so"
    };

    let pattern = out_dir.join(format!("lib/{}", shared_lib_pattern));
    let mut files = Vec::new();

    for entry in glob(pattern.to_str().unwrap()).unwrap() {
        match entry {
            Ok(path) => {
                files.push(path);
            }
            Err(e) => eprintln!("cargo:warning=error={}", e),
        }
    }

    files
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

    // Cuda https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
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
    println!(
        "cargo:rustc-link-search={}",
        out_dir.join("build/lib").display()
    );
    println!("cargo:rustc-link-search={}", bindings_dir.display());

    // Link libraries
    let sherpa_libs_kind = if build_shared_libs { "dylib" } else { "static" };
    let sherpa_libs = extract_lib_names(&out_dir);
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
        let libs_assets = extract_lib_assets(&out_dir);
        for asset in libs_assets {
            let asset_clone = asset.clone();
            let filename = asset_clone.file_name().unwrap();
            let filename = filename.to_str().unwrap();
            let dst = target_dir.join(filename);
            std::fs::copy(asset.clone(), dst).unwrap();
        }
    }
}
