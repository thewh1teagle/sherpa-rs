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
        std::fs::remove_file(
            sherpa_onnx_path.join("scripts/go/_internal/vad-spoken-language-identification/run.sh"),
        )
        .unwrap();
        std::fs::remove_file(
            sherpa_onnx_path
                .join("scripts/go/_internal/vad-spoken-language-identification/main.go"),
        )
        .unwrap();

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
        .define("SHERPA_ONNX_ENABLE_C_API", "ON");
    let destination = config.build();
    println!("cargo:rustc-link-search={}", out.join("lib").display());
    println!("cargo:rustc-link-search=native={}", destination.display());
    println!("cargo:rustc-link-lib=static=sherpa-onnx-c-api");
}
