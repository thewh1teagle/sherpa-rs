# sherpa-rs

[![Crates](https://img.shields.io/crates/v/sherpa-rs?logo=rust)](https://crates.io/crates/sherpa-rs/)
[![License](https://img.shields.io/github/license/thewh1teagle/sherpa-rs?color=00aaaa&logo=license)](https://github.com/thewh1teagle/sherpa-rs/blob/main/LICENSE)

Rust bindings to [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)

## Features

- Spoken language detection
- Speaker embedding (labeling)
- Speech to text
- Text to speech
- Voice activity detection

## Supported Platforms
- Windows
- Linux
- macOS

## Install

```console
cargo add sherpa-rs
```

## Usage

```console
git clone --recursive https://github.com/thewh1teagle/sherpa-rs.git

cd sherpa-rs

cargo run --example speaker_id
```

## Docs

See [sherpa/intro.html](https://k2-fsa.github.io/sherpa/intro.html)

## Examples

See [examples](examples)
