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
