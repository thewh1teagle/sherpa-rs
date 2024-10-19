# sherpa-rs

[![Crates](https://img.shields.io/crates/v/sherpa-rs?logo=rust)](https://crates.io/crates/sherpa-rs/)
[![License](https://img.shields.io/github/license/thewh1teagle/sherpa-rs?color=00aaaa&logo=license)](https://github.com/thewh1teagle/sherpa-rs/blob/main/LICENSE)

Rust bindings to [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)

## Features

- Spoken language detection
- Speaker embedding (labeling)
- Speaker diarization
- Speech to text
- Text to speech
- Text punctuation
- Voice activity detection
- Audio tagging

## Supported Platforms

- Windows
- Linux
- macOS

## Install

```console
cargo add sherpa-rs
```

## Build

Please see [BUILDING.md](BUILDING.md).

## Feature flags

- `cuda`: enable CUDA support
- `directml`: enable DirectML support
- `tts`: enable TTS
- `download-binaries`: use prebuilt sherpa-onnx libraries for faster builds. cached.
- `static`: use static sherpa-onnx libraries and link them statically.

## Docs

See [sherpa/intro.html](https://k2-fsa.github.io/sherpa/intro.html)

## Examples

See [examples](examples)
