# Building

### Prerequisites

[Cargo](https://www.rust-lang.org/tools/install) | [Clang](https://releases.llvm.org/download.html) | [Cmake](https://cmake.org/download/)

### Linux

```console
sudo apt-get update
sudo apt-get install -y pkg-config build-essential clang cmake
```

### Prepare repository

```console
git clone https://github.com/thewh1teagle/sherpa-rs --recursive
cd sherpa-rs
```

### Build

```console
cargo build --release
```

### Instructions (for builds with `cuda` enabled)

1. Download [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows)
2. Download [Visual Studio with Desktop C++ and Clang enabled](https://visualstudio.microsoft.com/de/downloads/) (see clang link below for installer walkthrough)
3. Run `where.exe clang`, then `setx LIBCLANG_PATH "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin"` or something like that
4. Restart your shell!!!
5. Cargo build

### Resample wav file for 16khz

```console
ffmpeg -i <file> -ar 16000 -ac 1 -c:a pcm_s16le <out>
```

### Update sherpa-onnx

```console
cd sys/sherpa-onnx
git pull origin master
```

### Gotachas

---

When running `--example` with dynamic libraries eg. with `directml` or `cuda` you need to have the DLLs from `target` folder in PATH.
Example:

```console
cargo build --features "directml" --example transcribe
copy target\debug\examples\transcribe.exe target\debug
target\debug\transcribe.exe motivation.wav
```

---

Currently whisper can transcribe only chunks of 30s max.

---

When building with cuda you should use cuda `11.x`
In addition install `cudnn` with `sudo apt install nvidia-cudnn`

<details>
<summary>Static linking failed on Windows</summary>

You can resolve it by creating `.cargo/config.toml` next to `Cargo.toml` with the following:

```toml
[target.'cfg(windows)']
rustflags = ["-C target-feature=+crt-static"]
```

Or set the environment variable `RUSTFLAGS` to `-C target-feature=+crt-static`

If it doesn't help make sure all of your dependencies also links MSVC runtime statically.
You can inspect the build with the following:

1. Set `RUSTC_LOG` to `rustc_codegen_ssa::back::link=info`
2. Build with

```console
cargo build -vv
```

Since there's a lot of output, it's good idea to pipe it to file and check later:

```console
cargo build -vv >log.txt 2>&1
```

Look for the flags `/MD` (Meaning it links it dynamically) and `/MT` or `-MT` (Meaning it links it statically). See [MSVC_RUNTIME_LIBRARY](https://cmake.org/cmake/help/latest/prop_tgt/MSVC_RUNTIME_LIBRARY.html) and [pyannote-rs/issues/1](https://github.com/thewh1teagle/pyannote-rs/issues/1)

## </details>

Controlling build flags
Please see `env::var` calls in `build.rs`.

### Debug build

For debug the build process of sherpa-onnx, please set `BUILD_DEBUG=1` environment variable before build.
