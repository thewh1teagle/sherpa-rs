# Tauri + sherpa-rs

<img src="https://github.com/user-attachments/assets/4c0da66d-61b8-481d-b53a-a049fe0b914d" width=250>

## Prepare model

```console
cd src-tauri
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
mv sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx model.onnx
rm -rf rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12
rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
adb push model.onnx /data/local/tmp/model.onnx # currently hardcoded in the APK
```

## Build

See https://v2.tauri.app/start/prerequisites

See [Building](../../BUILDING.md)

```console
# Setup environment variables
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
export ANDROID_HOME="$HOME/Library/Android/sdk"
export NDK_HOME="$HOME/Library/Android/sdk/ndk/27.0.12077973" # ls $HOME/Library/Android/sdk/ndk

# Setup UI
bun install
bunx tauri icon src-tauri/icons/icon.png

cd src-tauri
export CARGO_TARGET_DIR="$(pwd)/target"
cargo ndk -t arm64-v8a build
mkdir -p gen/android/app/src/main/jniLibs/arm64-v8a
ln -s $(pwd)/target/aarch64-linux-android/debug/libonnxruntime.so $(pwd)/gen/android/app/src/main/jniLibs/arm64-v8a/libonnxruntime.so
ln -s $(pwd)/target/aarch64-linux-android/debug/libsherpa-onnx-c-api.so $(pwd)/gen/android/app/src/main/jniLibs/arm64-v8a/libsherpa-onnx-c-api.so
bun run tauri android dev
```

## Debug

```console
adb logcat -c && adb logcat | grep -i -E "tauri|rust|sherpa"
```

## Debug webview

Open `chrome://inspect` in the chrome browser and click `inspect`
