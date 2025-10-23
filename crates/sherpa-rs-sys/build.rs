use cmake::Config;
use glob::glob;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "src/download_binaries.rs"]
#[cfg(feature = "download-binaries")]
mod download_binaries;

macro_rules! debug_log {
    ($($arg:tt)*) => {
        // SHERPA_BUILD_DEBUG=1 cargo build
        if std::env::var("SHERPA_BUILD_DEBUG").unwrap_or_default() == "1" {
            println!("cargo:warning=[DEBUG] {}", format!($($arg)*));
        }
    };
}

lazy_static::lazy_static! {
    // clang --print-targets
    // rustc --print target-list
    static ref RUST_CLANG_TARGET_MAP: HashMap<String, String> = {
        let mut m = HashMap::new();
        m.insert("aarch64-linux-android".to_string(), "armv8-linux-androideabi".to_string());
        m.insert("aarch64-apple-ios-sim".to_string(), "arm64-apple-ios-simulator".to_string());
        m
    };
}

fn link_lib(lib: &str, is_dynamic: bool) {
    let lib_kind = if is_dynamic { "dylib" } else { "static" };
    debug_log!("cargo:rustc-link-lib={}={}", lib_kind, lib);
    println!("cargo:rustc-link-lib={}={}", lib_kind, lib);
}

fn link_framework(framework: &str) {
    debug_log!("cargo:rustc-link-lib=framework={}", framework);
    println!("cargo:rustc-link-lib=framework={}", framework);
}

fn add_search_path<P: AsRef<Path>>(path: P) {
    debug_log!("cargo:rustc-link-search={}", path.as_ref().display());
    println!("cargo:rustc-link-search={}", path.as_ref().display());
}

fn copy_file(src: PathBuf, dst: PathBuf) {
    if let Err(err) = std::fs::hard_link(&src, &dst) {
        debug_log!("Failed to hardlink {:?}. fallback to copy.", err);
        fs::copy(&src, &dst)
            .unwrap_or_else(|_| panic!("Failed to copy {} to {}", src.display(), dst.display()));
    }
}

fn get_cargo_target_dir() -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    let profile = std::env::var("PROFILE")?;
    let mut target_dir = None;
    let mut sub_path = out_dir.as_path();
    while let Some(parent) = sub_path.parent() {
        if parent.ends_with(&profile) {
            target_dir = Some(parent);
            break;
        }
        sub_path = parent;
    }
    let target_dir = target_dir.ok_or("not found")?;
    Ok(target_dir.to_path_buf())
}

fn delete_folder(src: &Path) -> std::io::Result<()> {
    if src.exists() {
        fs::remove_dir_all(src)?;
    }
    Ok(())
}

fn copy_folder(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).expect("Failed to create dst directory");
    if cfg!(windows) {
        std::process::Command::new("robocopy.exe")
            .arg("/e")
            .arg(src)
            .arg(dst)
            .status()
            .expect("Failed to execute robocopy command");
    } else {
        std::process::Command::new("cp")
            .arg("-rf")
            .arg(src)
            .arg(dst.parent().unwrap())
            .status()
            .expect("Failed to execute cp command");
    }
}

fn extract_lib_names(out_dir: &Path, is_dynamic: bool, target_os: &str) -> Vec<String> {
    let lib_pattern = if target_os == "windows" {
        "*.lib"
    } else if target_os == "macos" {
        if is_dynamic {
            "*.dylib"
        } else {
            "*.a"
        }
    } else if
    // Linux, Android
    is_dynamic {
        "*.so"
    } else {
        "*.a"
    };

    let libs_dir = out_dir.join("lib");
    let pattern = libs_dir.join(lib_pattern);
    debug_log!("Extract libs {}", pattern.display());

    let mut lib_names: Vec<String> = Vec::new();

    // Process the libraries based on the pattern
    for entry in glob(pattern.to_str().unwrap()).unwrap() {
        match entry {
            Ok(path) => {
                let stem = path.file_stem().unwrap();
                let stem_str = stem.to_str().unwrap();

                // Skip certain libraries that should not be linked
                // cargs.lib is a command line argument parser that shouldn't be linked
                if stem_str == "cargs" {
                    debug_log!("Skipping library: {}", stem_str);
                    continue;
                }

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

fn extract_lib_assets(out_dir: &Path, target_os: &str) -> Vec<PathBuf> {
    let shared_lib_pattern = if target_os == "windows" {
        "*.dll"
    } else if target_os == "macos" {
        "*.dylib"
    } else {
        "*.so"
    };

    let libs_dir = out_dir.join("lib");
    let pattern = libs_dir.join(shared_lib_pattern);
    debug_log!("Extract lib assets {}", pattern.display());
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

fn rerun_on_env_changes(vars: &[&str]) {
    for env in vars {
        println!("cargo::rerun-if-env-changed={}", env);
    }
}

fn rerun_if_changed(vars: &[&str]) {
    for var in vars {
        println!("cargo:rerun-if-changed={}", var);
    }
}

fn verify_checksum(actual_hash: &str, expected_hash: &str) {
    if env::var("UNSAFE_DISABLE_CHECKSUM_VALIDATION").unwrap_or_default() == "1" {
        println!("cargo:warning=UNSAFE: Checksum validation disabled!");
        return;
    }

    if actual_hash != expected_hash {
        panic!(
            "Checksum validation failed!\n\
            Expected: {}\n\
            Got:      {}\n\
            \n\
            This usually means the downloaded file is corrupted or has been tampered with.\n\
            \n\
            Possible solutions:\n\
            1. Try cleaning the cache and rebuilding: cargo clean && cargo build\n\
            2. Check your internet connection and try again\n\
            3. If you trust the source and want to bypass validation (NOT RECOMMENDED):\n\
               UNSAFE_DISABLE_CHECKSUM_VALIDATION=1 cargo build",
            expected_hash, actual_hash
        );
    }
}

fn main() {
    rerun_if_changed(&["wrapper.h", "dist.json", "checksum.txt", "./sherpa-onnx"]);
    rerun_on_env_changes(&[
        "SHERPA_BUILD_SHARED_LIBS",
        "CMAKE_BUILD_PARALLEL_LEVEL",
        "CMAKE_VERBOSE",
        "SHERPA_LIB_PATH",
        "SHERPA_STATIC_CRT",
        "SHERPA_LIB_PROFILE",
        "BUILD_DEBUG",
        "UNSAFE_DISABLE_CHECKSUM_VALIDATION",
    ]);

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    debug_log!("target_os = {}", target_os);

    // Show warning if static enabled on Linux without RUSTFLAGS
    #[cfg(all(
        feature = "static",
        target_os = "linux",
        target_arch = "x86_64",
        feature = "download-binaries"
    ))]
    {
        if !env::var("RUSTFLAGS")
            .unwrap_or_default()
            .contains("relocation-model=dynamic-no-pic")
            && !env::var("CARGO_ENCODED_RUSTFLAGS")
                .unwrap_or_default()
                .contains("relocation-model=dynamic-no-pic")
        {
            panic!(
                "cargo:warning=\
            Please enable the following environment variable when static feature enabled on Linux: RUSTFLAGS=\"-C relocation-model=dynamic-no-pic\""
            )
        }
    }

    let target = env::var("TARGET").unwrap();
    let is_mobile = target.contains("android") || target.contains("ios");
    debug_log!("TARGET: {:?}", target);
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let target_dir = get_cargo_target_dir().unwrap();
    let sherpa_dst = out_dir.join("sherpa-onnx");
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("Failed to get CARGO_MANIFEST_DIR");
    let sherpa_src = Path::new(&manifest_dir).join("sherpa-onnx");

    // Dynamic by default
    #[allow(unused_mut)]
    let mut is_dynamic = if cfg!(feature = "static") {
        false // Static feature is enabled
    } else if cfg!(any(feature = "directml", feature = "cuda")) {
        true // DirectML or CUDA feature is enabled
    } else if let Ok(val) = env::var("SHERPA_BUILD_SHARED_LIBS") {
        val == "1" // Environment variable determines dynamic state
    } else {
        true // Default to true
    };

    debug_log!("TARGET: {}", target);
    debug_log!("CARGO_MANIFEST_DIR: {}", manifest_dir);
    debug_log!("TARGET_DIR: {}", target_dir.display());
    debug_log!("OUT_DIR: {}", out_dir.display());

    // Prepare sherpa-onnx source
    if !sherpa_dst.exists() {
        debug_log!("Copy {} to {}", sherpa_src.display(), sherpa_dst.display());
        delete_folder(&sherpa_src.join("scripts")).unwrap();
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
    if env::var("SHERPA_SKIP_GENERATE_BINDINGS").is_ok() {
        debug_log!("Skip generate bindings");
        std::fs::copy("src/bindings.rs", out_dir.join("bindings.rs"))
            .expect("Failed to copy bindings.rs");
    } else {
        let mut bindings_builder = bindgen::Builder::default()
            .header("wrapper.h")
            .clang_arg(format!("-I{}", sherpa_dst.display()))
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

        if let Some(clang_target) = RUST_CLANG_TARGET_MAP.get(&target) {
            // Explicitly set target in case we are cross-compiling.
            // See https://github.com/rust-lang/rust-bindgen/issues/1780 for context.
            debug_log!("mapped clang target: {}", clang_target);
            bindings_builder = bindings_builder.clang_arg(format!("--target={}", clang_target));
        }

        debug_log!("Generating bindings...");
        let bindings_builder = bindings_builder
            .generate()
            .expect("Failed to generate bindings");

        // Write the generated bindings to an output file
        let bindings_path = out_dir.join("bindings.rs");

        debug_log!("Writing bindings to {:?}", bindings_path);
        bindings_builder
            .write_to_file(bindings_path)
            .expect("Failed to write bindings");
        debug_log!("Bindings Created");
    }

    // Skip build when docs.rs website built this crate
    // Only build the bindings.rs file.
    if env::var("DOCS_RS") == Ok("1".to_string()) {
        println!("cargo:warning=Detected DOCS_RS. Skipping build / fetch.");
        return;
    }

    #[cfg(feature = "download-binaries")]
    let mut optional_dist: Option<download_binaries::Dist> = None;

    let mut sherpa_libs: Vec<String> = Vec::new();

    #[cfg(feature = "download-binaries")]
    {
        // Download libraries, cache and set SHERPA_LIB_PATH
        use download_binaries::{extract_tbz, fetch_file, get_cache_dir, sha256, DIST_TABLE};
        debug_log!("Download binaries enabled");
        // debug_log!("Dist table: {:?}", DIST_TABLE.targets);
        // Try download sherpa libs and set SHERPA_LIB_PATH
        if let Some(dist) = DIST_TABLE.get(&target, &mut is_dynamic) {
            debug_log!("is_dynamic after: {}", is_dynamic);
            optional_dist = Some(dist.clone());
            let mut cache_dir = if let Some(dir) = get_cache_dir() {
                dir.join(target.clone()).join(&dist.checksum)
            } else {
                println!("cargo:warning=Could not determine cache directory, using OUT_DIR");
                PathBuf::from(env::var("OUT_DIR").unwrap())
            };
            if fs::create_dir_all(&cache_dir).is_err() {
                println!("cargo:warning=Could not create cache directory, using OUT_DIR");
                cache_dir = env::var("OUT_DIR").unwrap().into();
            }
            debug_log!("Cache dir: {}", cache_dir.display());

            let lib_dir = cache_dir.join(&dist.name);

            // if is mobile then check if cache dir not empty
            // Sherpa uses special directory structure for mobile
            let cache_dir_empty = cache_dir
                .read_dir()
                .map(|mut entries| entries.next().is_none())
                .unwrap_or(true);

            if (is_mobile && cache_dir_empty) || (!is_mobile && !lib_dir.exists()) {
                let downloaded_file = fetch_file(&dist.url);
                let hash = sha256(&downloaded_file);
                verify_checksum(&hash, &dist.checksum);
                extract_tbz(&downloaded_file, &cache_dir);
            } else {
                debug_log!("Skip fetch file. Using cache from {}", lib_dir.display());
            }

            // In Android, we need to set SHERPA_LIB_PATH to the cache directory sincie it has jniLibs
            if is_mobile {
                env::set_var("SHERPA_LIB_PATH", &cache_dir);
            } else {
                env::set_var("SHERPA_LIB_PATH", cache_dir.join(&dist.name));
            }

            debug_log!("dist libs: {:?}", dist.libs);
            if let Some(libs) = dist.libs {
                for lib in libs.iter() {
                    let lib_path = cache_dir.join(lib);
                    let lib_parent = lib_path.parent().unwrap();
                    add_search_path(lib_parent);
                }

                sherpa_libs = libs
                    .iter()
                    .map(download_binaries::extract_lib_name)
                    .collect();
            } else {
                sherpa_libs = extract_lib_names(&lib_dir, is_dynamic, &target_os);
            }
        } else {
            println!("cargo:warning=Failed to download binaries. fallback to manual build.");
        }
    }

    if let Ok(sherpa_lib_path) = env::var("SHERPA_LIB_PATH") {
        // Skip build if SHERPA_LIB_PATH specified
        debug_log!("Skpping build with Cmake...");
        debug_log!("SHERPA_LIB_PATH: {}", sherpa_lib_path);
        add_search_path(Path::new(&sherpa_lib_path).join("lib"));
        if sherpa_libs.is_empty() {
            sherpa_libs = extract_lib_names(Path::new(&sherpa_lib_path), is_dynamic, &target_os);
        }
    } else {
        // Build with CMake
        let profile = env::var("SHERPA_LIB_PROFILE").unwrap_or("Release".to_string());
        let static_crt = env::var("SHERPA_STATIC_CRT")
            .map(|v| v == "1")
            .unwrap_or(true);
        let mut config = Config::new(&sherpa_dst);

        config.define("CMAKE_POLICY_VERSION_MINIMUM", "3.5");

        config
            .define("SHERPA_ONNX_ENABLE_C_API", "ON")
            .define("SHERPA_ONNX_ENABLE_BINARY", "OFF")
            .define("BUILD_SHARED_LIBS", if is_dynamic { "ON" } else { "OFF" })
            .define("SHERPA_ONNX_ENABLE_WEBSOCKET", "OFF")
            .define("SHERPA_ONNX_ENABLE_TTS", "OFF")
            .define("SHERPA_ONNX_BUILD_C_API_EXAMPLES", "OFF");

        if target_os == "windows" {
            config.static_crt(static_crt);
        }

        // TTS
        if cfg!(feature = "tts") {
            config.define("SHERPA_ONNX_ENABLE_TTS", "ON");
        }

        // Cuda https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
        if cfg!(feature = "cuda") {
            debug_log!("Cuda enabled");
            config.define("SHERPA_ONNX_ENABLE_GPU", "ON");
            config.define("BUILD_SHARED_LIBS", "ON");
        }

        // DirectML https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
        if cfg!(feature = "directml") {
            debug_log!("DirectML enabled");
            config.define("SHERPA_ONNX_ENABLE_DIRECTML", "ON");
            config.define("BUILD_SHARED_LIBS", "ON");
        }

        if target_os == "windows" || target_os == "linux" || target == "android" {
            config.define("SHERPA_ONNX_ENABLE_PORTAUDIO", "ON");
        }

        // General
        config
            .profile(&profile)
            .very_verbose(std::env::var("CMAKE_VERBOSE").is_ok()) // Not verbose by default
            .always_configure(false);

        let build_dir = config.build();
        add_search_path(&build_dir);

        // Extract libs on desktop platforms
        if !is_mobile {
            sherpa_libs = extract_lib_names(&build_dir, is_dynamic, &target_os);
        }
    }

    // Linking

    debug_log!("Sherpa libs: {:?}", sherpa_libs);
    add_search_path(out_dir.join("lib"));

    for lib in sherpa_libs {
        if lib.contains("cxx") {
            continue;
        }
        link_lib(&lib, is_dynamic);
    }

    // Windows debug
    if cfg!(all(debug_assertions, windows)) {
        link_lib("msvcrtd", true);
    }

    // macOS
    if target_os == "macos" || target_os == "ios" {
        link_framework("CoreML");
        link_framework("Foundation");
        link_lib("c++", true);
    }

    // Linux
    if target_os == "linux" || target == "android" {
        link_lib("stdc++", true);
    }

    // macOS
    if target_os == "macos" {
        // On (older) OSX we need to link against the clang runtime,
        // which is hidden in some non-default path.
        //
        // More details at https://github.com/alexcrichton/curl-rust/issues/279.
        if let Some(path) = macos_link_search_path() {
            add_search_path(path);
            link_lib("clang_rt.osx", is_dynamic);
        }
    }

    // Copy dynamic libraries

    if is_dynamic {
        let mut libs_assets = extract_lib_assets(&out_dir, &target_os);
        if let Ok(sherpa_lib_path) = env::var("SHERPA_LIB_PATH") {
            libs_assets.extend(extract_lib_assets(Path::new(&sherpa_lib_path), &target_os));
        }

        #[cfg(feature = "download-binaries")]
        if let Some(dist) = optional_dist {
            if let Some(assets) = dist.libs {
                if let Ok(sherpa_lib_path) = env::var("SHERPA_LIB_PATH") {
                    let sherpa_lib_path = Path::new(&sherpa_lib_path);
                    libs_assets.extend(assets.iter().map(|p| sherpa_lib_path.join(p)));
                }
            }
        }

        // Filter out cargs dynamic libraries from being copied to target output.
        // This keeps runtime artifacts clean and avoids shipping CLI-only helpers.
        libs_assets.retain(|p| {
            let fname = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            let lower = fname.to_ascii_lowercase();
            let is_cargs = lower == "cargs.dll"
                || lower == "libcargs.so"
                || lower == "libcargs.dylib"
                || lower.starts_with("cargs.")
                || lower.starts_with("libcargs.");
            if is_cargs {
                debug_log!("Skipping asset {}", fname);
            }
            !is_cargs
        });

        for asset in libs_assets {
            let asset_clone = asset.clone();
            let filename = asset_clone.file_name().unwrap();
            let filename = filename.to_str().unwrap();
            let dst = target_dir.join(filename);
            // debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
            if !dst.exists() {
                copy_file(asset.clone(), dst);
            }

            // Copy DLLs to examples as well
            if target_dir.join("examples").exists() {
                let dst = target_dir.join("examples").join(filename);
                // debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
                if !dst.exists() {
                    copy_file(asset.clone(), dst);
                }
            }

            // Copy DLLs to target/profile/deps as well for tests
            let dst = target_dir.join("deps").join(filename);
            // debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
            if !dst.exists() {
                copy_file(asset.clone(), dst);
            }
        }

        // TODO: add rpath for Android and iOS so it can find its dependencies in the same directory of executable
        // if is_mobile {
        //     // Add rpath for Android and iOS so that the shared library can find its dependencies in the same directory as well
        //     println!("cargo:rustc-link-arg=-Wl,-rpath,'$ORIGIN'");
        // }
    }
}
