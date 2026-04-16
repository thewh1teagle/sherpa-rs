# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] — unreleased

### Added

- **Cohere Transcribe support** — new `cohere_transcribe` module exposing
  `CohereTranscribeRecognizer` / `CohereTranscribeConfig`. Wraps the
  `SherpaOnnxOfflineCohereTranscribeModelConfig` C API added upstream in
  sherpa-onnx v1.12.x. Supports 14 languages with native punctuation and
  inverse-text-normalization toggles.
- **Integration test suite and model-download helper** — first
  cargo-runnable tests in the repo (previously only `examples/` existed
  and required manual model setup). `tests/offline_recognizers.rs`
  exercises both the Whisper path (validating that existing C API fields
  still work after the v1.12.38 bump) and the new Cohere Transcribe
  module end-to-end against real audio. `tests/test_utils.rs` is a
  reusable helper exposing `ensure_model(&ModelArchive)` /
  `ensure_motivation_wav()` — it resolves a model cache directory
  (`SHERPA_TEST_MODELS` env var or workspace-root `test_data/`), skips
  tests gracefully when files are missing (CI-friendly default), and
  auto-downloads from the k2-fsa/sherpa-onnx release assets when
  `SHERPA_DOWNLOAD_MODELS=1` is set. Downloads are serialised via
  `std::sync::Once` so the suite is safe under `cargo test`'s default
  parallel runner. Covers Whisper-tiny (~250 MB) and Cohere Transcribe
  int8 (~2.7 GB); adding further model archives is a one-const entry.
- `ZipVoiceTtsConfig`: new `encoder`, `decoder`, `lexicon` fields
  matching the upstream C API layout.

### Changed

- **Bumped bundled sherpa-onnx from v1.12.15 → v1.12.38.** Updated
  submodule, `dist.json` tag, and `checksum.txt`.
- **Apple Silicon performance fix.** `dist.json` for `aarch64-apple-darwin`
  now pulls `sherpa-onnx-{tag}-onnxruntime-1.24.4-osx-arm64-shared.tar.bz2`
  instead of `sherpa-onnx-{tag}-osx-arm64-shared.tar.bz2`. The former
  bundles the full-optimization onnxruntime build (same as the Python
  pip wheel); the latter shipped a smaller variant missing
  graph-optimization paths, which blocked the post-first-inference
  kernel-cache warmup. Net effect on our spot-check with Cohere
  Transcribe int8: a multi-x speedup on warm inferences, bringing
  warm-path parity with the Python `sherpa_onnx` pip wheel. Actual
  numbers will vary by chip generation, thread count, and model.
- Windows `dist.json` entry updated to
  `sherpa-onnx-{tag}-win-x64-shared-MD-Release.tar.bz2` (upstream renamed
  the asset to include build configuration).
- `aarch64-apple-darwin` / `x86_64-apple-darwin` split: each arch now
  points at its native single-arch tarball instead of the universal2
  fat binary. Smaller downloads, no lipo-imposed compiler compromises.

### Breaking

- `ZipVoiceTtsConfig` dropped `flow_matching_model`, `text_model`, and
  `pinyin_dict` fields — upstream replaced them with `encoder`,
  `decoder`, and `lexicon` between v1.12.15 and v1.12.38. Callers must
  rename their field usage.
- Minimum bundled sherpa-onnx is v1.12.38. Consumers pinning
  `sherpa-rs` in a workspace that also vendors an older sherpa-onnx
  must upgrade.

## [0.6.8] — 2025-xx-xx

- See GitHub releases.
