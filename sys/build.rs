use cmake::Config;
use glob::glob;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// Prebuilt sherpa-onnx doesn't have Cuda support
#[cfg(all(
    any(target_os = "windows", target_os = "linux"),
    feature = "download-binaries",
    feature = "cuda"
))]
compile_error!(
    "The 'download-binaries' and 'cuda' features cannot be enabled at the same time.\n\
    To resolve this, please disable the 'download-binaries' feature when using 'cuda'.\n\
    For example, in your Cargo.toml:\n\
    [dependencies]\n\
    sherpa-rs = { default-features = false, features = [\"cuda\"] }"
);

// Prebuilt sherpa-onnx doesn't have DirectML support
#[cfg(all(windows, feature = "download-binaries", feature = "directml"))]
compile_error!(
    "The 'download-binaries' and 'directml' features cannot be enabled at the same time.\n\
    To resolve this, please disable the 'download-binaries' feature when using 'directml'.\n\
    For example, in your Cargo.toml:\n\
    [dependencies]\n\
    sherpa-rs = { default-features = false, features = [\"directml\"] }"
);

// Prebuilt sherpa-onnx does not include TTS in static builds.
#[cfg(all(
    windows,
    feature = "download-binaries",
    feature = "static",
    feature = "tts"
))]
compile_error!(
    "The 'download-binaries', 'static', and 'tts' features cannot be enabled at the same time.\n\
    To resolve this, please disable the 'tts' feature when using 'static' and 'download-binaries' together.\n\
    For example, in your Cargo.toml:\n\
    [dependencies]\n\
    sherpa-rs = { default-features = false, features = [\"static\", \"tts\"] }"
);

#[path = "src/internal/mod.rs"]
#[cfg(feature = "download-binaries")]
mod internal;

#[cfg(feature = "download-binaries")]
use internal::dirs::cache_dir;

const DIST_TABLE: &str = include_str!("dist.txt");

macro_rules! debug_log {
    ($($arg:tt)*) => {
        if std::env::var("BUILD_DEBUG").unwrap_or_default() == "1" {
            println!("cargo:warning=[DEBUG] {}", format!($($arg)*));
        }
    };
}

#[cfg(feature = "download-binaries")]
fn fetch_file(source_url: &str) -> Vec<u8> {
    let resp = ureq::AgentBuilder::new()
        .try_proxy_from_env(true)
        .build()
        .get(source_url)
        .timeout(std::time::Duration::from_secs(1800))
        .call()
        .unwrap_or_else(|err| panic!("Failed to GET `{source_url}`: {err}"));

    let len = resp
        .header("Content-Length")
        .and_then(|s| s.parse::<usize>().ok())
        .expect("Content-Length header should be present on archive response");
    debug_log!("Fetch file {} {}", source_url, len);
    let mut reader = resp.into_reader();
    let mut buffer = Vec::new();
    reader
        .read_to_end(&mut buffer)
        .unwrap_or_else(|err| panic!("Failed to download from `{source_url}`: {err}"));
    assert_eq!(buffer.len(), len);
    buffer
}

#[derive(Debug)]
struct Dist {
    url: String,
    sha256: String,
    name: String,
    is_dynamic: bool,
}

fn get_feature_set() -> String {
    let mut features = Vec::new();
    if cfg!(feature = "static") {
        features.push("static");
    }
    features.sort();
    if features.is_empty() {
        "none".into()
    } else {
        features.join(",")
    }
}

fn find_dist(target: &str, feature_set: &str) -> Option<Dist> {
    let table_content = DIST_TABLE
        .split('\n')
        .skip(5)
        .filter(|l| !l.is_empty() && !l.starts_with('#')); // Skip headers
    debug_log!(
        "table content: {:?}",
        table_content.clone().collect::<String>()
    );
    let mut table = table_content.map(|l| l.split_whitespace().collect::<Vec<_>>());

    table
        .find(|row| row[0] == feature_set && row[1] == target)
        .map(|row| Dist {
            url: row[2].into(),
            name: row[3].into(),
            sha256: row[4].into(),
            is_dynamic: row[5].trim() == "1",
        })
}

#[cfg(feature = "download-binaries")]
fn hex_str_to_bytes(c: impl AsRef<[u8]>) -> Vec<u8> {
    fn nibble(c: u8) -> u8 {
        match c {
            b'A'..=b'F' => c - b'A' + 10,
            b'a'..=b'f' => c - b'a' + 10,
            b'0'..=b'9' => c - b'0',
            _ => panic!(),
        }
    }

    c.as_ref()
        .chunks(2)
        .map(|n| nibble(n[0]) << 4 | nibble(n[1]))
        .collect()
}

#[cfg(feature = "download-binaries")]
fn verify_file(buf: &[u8], hash: impl AsRef<[u8]>) -> bool {
    <sha2::Sha256 as sha2::Digest>::digest(buf)[..] == hex_str_to_bytes(hash)
}

#[cfg(feature = "download-binaries")]
#[allow(unused)]
fn extract_tgz(buf: &[u8], output: &Path) {
    let buf: std::io::BufReader<&[u8]> = std::io::BufReader::new(buf);
    let tar = flate2::read::GzDecoder::new(buf);
    let mut archive = tar::Archive::new(tar);
    archive.unpack(output).expect("Failed to extract .tgz file");
}

#[cfg(feature = "download-binaries")]
fn extract_tbz(buf: &[u8], output: &Path) {
    debug_log!("extracging tbz to {}", output.display());
    let buf: std::io::BufReader<&[u8]> = std::io::BufReader::new(buf);
    let tar = bzip2::read::BzDecoder::new(buf); // Use BzDecoder for .bz2
    let mut archive = tar::Archive::new(tar);
    archive.unpack(output).expect("Failed to extract .tbz file");
}

fn hard_link(src: PathBuf, dst: PathBuf) {
    if let Err(err) = std::fs::hard_link(&src, &dst) {
        debug_log!("Failed to hardlink {:?}. fallback to copy.", err);
        fs::copy(src, dst).unwrap();
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

fn extract_lib_names(out_dir: &Path, dynamic: bool) -> Vec<String> {
    let lib_pattern = if cfg!(windows) {
        "*.lib"
    } else if cfg!(target_os = "macos") {
        if dynamic {
            "*.dylib"
        } else {
            "*.a"
        }
    } else if dynamic {
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

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=./sherpa-onnx");
    println!("cargo:rerun-if-changed=dist.txt");

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
        {
            panic!(
                "cargo:warning=\
            Please enable the following environment variable when static feature enabled on Linux: RUSTFLAGS=\"-C relocation-model=dynamic-no-pic\""
            )
        }
    }

    // Rerun on these environment changes
    rerun_on_env_changes(&[
        "SHERPA_BUILD_SHARED_LIBS",
        "CMAKE_BUILD_PARALLEL_LEVEL",
        "CMAKE_VERBOSE",
        "SHERPA_LIB_PATH",
        "SHERPA_STATIC_CRT",
        "SHERPA_LIB_PROFILE",
        "BUILD_DEBUG",
    ]);

    let target = env::var("TARGET").unwrap();
    debug_log!("TARGET: {:?}", target);
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let target_dir = get_cargo_target_dir().unwrap();
    let sherpa_dst = out_dir.join("sherpa-onnx");
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("Failed to get CARGO_MANIFEST_DIR");
    let sherpa_src = Path::new(&manifest_dir).join("sherpa-onnx");
    let mut is_dynamic = !cfg!(feature = "directml") || !cfg!(feature = "cuda");

    is_dynamic = std::env::var("SHERPA_BUILD_SHARED_LIBS")
        .map(|v| v == "1")
        .unwrap_or(is_dynamic);
    let profile = env::var("SHERPA_LIB_PROFILE").unwrap_or("Release".to_string());
    let static_crt = env::var("SHERPA_STATIC_CRT")
        .map(|v| v == "1")
        .unwrap_or(true);

    debug_log!("TARGET: {}", target);
    debug_log!("CARGO_MANIFEST_DIR: {}", manifest_dir);
    debug_log!("TARGET_DIR: {}", target_dir.display());
    debug_log!("OUT_DIR: {}", out_dir.display());

    // Prepare sherpa-onnx source
    if !sherpa_dst.exists() {
        debug_log!("Copy {} to {}", sherpa_src.display(), sherpa_dst.display());
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
        debug_log!("Bindings Created");
    }

    // Build with Cmake

    let mut config = Config::new(&sherpa_dst);

    config
        .define("SHERPA_ONNX_ENABLE_C_API", "ON")
        .define("SHERPA_ONNX_ENABLE_BINARY", "OFF")
        .define("BUILD_SHARED_LIBS", if is_dynamic { "ON" } else { "OFF" })
        .define("SHERPA_ONNX_ENABLE_WEBSOCKET", "OFF")
        .define("SHERPA_ONNX_ENABLE_TTS", "OFF")
        .define("SHERPA_ONNX_BUILD_C_API_EXAMPLES", "OFF");

    if cfg!(windows) {
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

    if cfg!(any(windows, target_os = "linux")) {
        config.define("SHERPA_ONNX_ENABLE_PORTAUDIO", "ON");
    }

    // General
    config
        .profile(&profile)
        .very_verbose(std::env::var("CMAKE_VERBOSE").is_ok()) // Not verbose by default
        .always_configure(false);

    #[cfg(feature = "download-binaries")]
    {
        // Try download sherpa libs and set SHERPA_LIB_PATH
        if let Some(dist) = find_dist(&target, &get_feature_set()) {
            debug_log!("Dist: {:?}", dist);
            let mut cache_dir = cache_dir()
                .expect("could not determine cache directory")
                .join("sherpa-bin")
                .join(&target)
                .join(&dist.sha256);
            if fs::create_dir_all(&cache_dir).is_err() {
                cache_dir = env::var("OUT_DIR").unwrap().into();
            }
            debug_log!("Cache dir: {}", cache_dir.display());
            let lib_dir = cache_dir.join(&dist.name);
            if !lib_dir.exists() {
                let downloaded_file = fetch_file(&dist.url);
                assert!(
                    verify_file(&downloaded_file, &dist.sha256),
                    "hash of downloaded Sherpa-ONNX Runtime binary does not match!"
                );
                extract_tbz(&downloaded_file, &cache_dir);
            } else {
                debug_log!("Skip fetch file. Using cache from {}", lib_dir.display());
            }
            env::set_var("SHERPA_LIB_PATH", cache_dir.join(&dist.name));
            is_dynamic = dist.is_dynamic;
        } else {
            println!("cargo:warning=Failed to download binaries. fallback to manual build.");
        }
    }

    let sherpa_libs: Vec<String>;
    let sherpa_libs_kind = if is_dynamic { "dylib" } else { "static" };

    if let Ok(sherpa_lib_path) = env::var("SHERPA_LIB_PATH") {
        // Skip build if SHERPA_LIB_PATH specified
        debug_log!("Skpping build with Cmake...");
        debug_log!("SHERPA_LIB_PATH: {}", sherpa_lib_path);
        println!(
            "cargo:rustc-link-search={}",
            Path::new(&sherpa_lib_path).join("lib").display()
        );
        sherpa_libs = extract_lib_names(Path::new(&sherpa_lib_path), is_dynamic);
    } else {
        // Build with CMake
        let bindings_dir = config.build();
        println!("cargo:rustc-link-search={}", bindings_dir.display());
        // Link libraries
        sherpa_libs = extract_lib_names(&out_dir, is_dynamic);
    }

    // Search paths
    println!("cargo:rustc-link-search={}", out_dir.join("lib").display());

    for lib in sherpa_libs {
        debug_log!(
            "LINK {}",
            format!("cargo:rustc-link-lib={}={}", sherpa_libs_kind, lib)
        );
        println!("cargo:rustc-link-lib={}={}", sherpa_libs_kind, lib);
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
    if is_dynamic {
        let mut libs_assets = extract_lib_assets(&out_dir);
        if let Ok(sherpa_lib_path) = env::var("SHERPA_LIB_PATH") {
            libs_assets.extend(extract_lib_assets(Path::new(&sherpa_lib_path)));
        }

        for asset in libs_assets {
            let asset_clone = asset.clone();
            let filename = asset_clone.file_name().unwrap();
            let filename = filename.to_str().unwrap();
            let dst = target_dir.join(filename);
            debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
            if !dst.exists() {
                hard_link(asset.clone(), dst);
            }

            // Copy DLLs to examples as well
            if target_dir.join("examples").exists() {
                let dst = target_dir.join("examples").join(filename);
                debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
                if !dst.exists() {
                    hard_link(asset.clone(), dst);
                }
            }

            // Copy DLLs to target/profile/deps as well for tests
            let dst = target_dir.join("deps").join(filename);
            debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
            if !dst.exists() {
                hard_link(asset.clone(), dst);
            }
        }
    }
}
