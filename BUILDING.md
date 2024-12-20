# Building

### Prerequisites

[Cargo](https://www.rust-lang.org/tools/install) | [Clang](https://releases.llvm.org/download.html) | [Cmake](https://cmake.org/download/)

### Linux

```console
sudo apt-get update
sudo apt-get install -y pkg-config build-essential clang cmake
```

### Windows

For convenience, I recommend installing these packages.
Additionally, when using wget to run examples, use `wget.exe` instead.

```console
winget install -e --id GnuWin32.Tar
winget install -e --id JernejSimoncic.Wget
```

### Prepare repository

```console
git clone --recursive https://github.com/thewh1teagle/sherpa-rs
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

### Compile with prebuild sherpa-onnx manually (For fast compilation)

Note: sherpa-onnx already download and cache the sherpa-onnx binaries. You can do this manually instead.
Note: to link sherpa-onnx libs dynamically set `SHERPA_BUILD_SHARED_LIBS` to `1`.
Note: you should disable rust-analyzer while doing this. otherwise it will rebuild it with different environment variable on each save which will take long.... time.
Note: on Linux when linking statically you should set this env: `RUSTFLAGS="-C relocation-model=dynamic-no-pic"`

<details>
<summary>macOS (arm64/x86-64)</summary>

```console
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.10.28/sherpa-onnx-v1.10.28-osx-universal2-static.tar.bz2
tar xf sherpa-onnx-v1.10.28-osx-universal2-static.tar.bz2
export SHERPA_LIB_PATH="$(pwd)/sherpa-onnx-v1.10.28-osx-universal2-static"
cargo build
```

</details>

<details>
<summary>Windows (x86-64)</summary>

```console
wget.exe https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.10.28/sherpa-onnx-v1.10.28-win-x64-static.tar.bz2
tar.exe xf sherpa-onnx-v1.10.28-win-x64-static.tar.bz2
$env:SHERPA_LIB_PATH="$pwd/sherpa-onnx-v1.10.28-win-x64-static"
cargo build
```

</details>

<details>
<summary>Linux (x86-64)</summary>

```console
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.10.28/sherpa-onnx-v1.10.28-linux-x64-static.tar.bz2
tar xf sherpa-onnx-v1.10.28-linux-x64-static.tar.bz2
export SHERPA_LIB_PATH="$(pwd)/sherpa-onnx-v1.10.28-linux-x64-static"
export RUSTFLAGS="-C relocation-model=dynamic-no-pic"
cargo build
```

</details>

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

On Linux you should set `RUSTFLAGS="-C relocation-model=dynamic-no-pic"`

<details>
<summary>When using GPU such as DirectML or Cuda</summary>

---

When running `--example` with dynamic libraries eg. with `directml` or `cuda` you need to have the DLLs from `target` folder in PATH.
Example:

```console
cargo build --features "directml" --example transcribe
copy target\debug\examples\transcribe.exe target\debug
target\debug\transcribe.exe motivation.wav
```

When building with cuda you should use cuda `11.x`
In addition install `cudnn` with `sudo apt install nvidia-cudnn`

</details>

<details>
<summary>Whisper limits</summary>

---

Currently whisper can transcribe only chunks of 30s max.

---

</details>

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

</details>

<details>
<summary>Fix build issues with build flags</summary>

Controlling build flags
Please see `env::var` calls in `build.rs`.

</details>

<details>
<summary>Cmake error: path exceeded</summary>

Cmake filed with error about maxium paths exceeded. eg. `The fully qualified file name must be less than 260 characters.`

1. Open PowerShell as admin and execute:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
-Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

2. Restart PC

</details>

### Debug build

For debug the build process of sherpa-onnx, please set `BUILD_DEBUG=1` environment variable before build.

## Release new version

```console
gh release create v0.4.1 --title v0.4.1 --generate-notes
```

## Calculate sha256 for dist table

```console
shasum -a 256 <path> | tr 'a-z' 'A-Z'
```

## See debug log from build

```
BUILD_DEBUG=1 cargo build -vv
```

## Build for Android

You must install NDK from Android Studio settings.

```console
rustup target add aarch64-linux-android
cargo install cargo-ndk
export NDK_HOME="$HOME/Library/Android/sdk/ndk/27.0.12077973"
cargo ndk -t arm64-v8a -o ./jniLibs build --release
```
