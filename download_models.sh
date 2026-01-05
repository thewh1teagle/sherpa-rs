#!/bin/bash
# 下载 funasr-nano 和 streaming 模型的脚本

set -e

cd "$(dirname "$0")"

echo "=== 下载 FunASR-Nano 模型 ==="
mkdir -p funasr-nano
cd funasr-nano

# FunASR-Nano int8 模型文件
BASE_URL="https://huggingface.co/csukuangfj/sherpa-onnx-funasr-nano-int8-2025-12-30/resolve/main"

echo "下载 encoder_adaptor.int8.onnx..."
curl -L -o encoder_adaptor.int8.onnx "${BASE_URL}/encoder_adaptor.int8.onnx"

echo "下载 llm_prefill.int8.onnx..."
curl -L -o llm_prefill.int8.onnx "${BASE_URL}/llm_prefill.int8.onnx"

echo "下载 llm_decode.int8.onnx..."
curl -L -o llm_decode.int8.onnx "${BASE_URL}/llm_decode.int8.onnx"

echo "下载 embedding.int8.onnx..."
curl -L -o embedding.int8.onnx "${BASE_URL}/embedding.int8.onnx"

# 下载 Qwen3-0.6B tokenizer 目录
echo ""
echo "下载 Qwen3-0.6B tokenizer..."
mkdir -p Qwen3-0.6B
cd Qwen3-0.6B

echo "  - merges.txt..."
curl -L -o merges.txt "${BASE_URL}/Qwen3-0.6B/merges.txt"

echo "  - vocab.json..."
curl -L -o vocab.json "${BASE_URL}/Qwen3-0.6B/vocab.json"

echo "  - tokenizer.json..."
curl -L -o tokenizer.json "${BASE_URL}/Qwen3-0.6B/tokenizer.json"

cd ../..

echo ""
echo "=== 下载 Streaming Zipformer 模型 ==="
echo "下载 sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20..."
if [ ! -f "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2" ]; then
    curl -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
fi

if [ ! -d "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20" ]; then
    echo "解压模型..."
    tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
fi

echo ""
echo "=== 下载完成 ==="
echo ""
echo "FunASR-Nano 模型位置:"
echo "  $(pwd)/funasr-nano/"
echo "  └── Qwen3-0.6B/ (tokenizer 目录)"
echo ""
echo "Streaming 模型位置:"
echo "  $(pwd)/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/"
echo ""
echo "运行测试:"
echo "  cargo run --example transcribe_test -- <audio.wav> funasr"
echo "  cargo run --example transcribe_test -- <audio.wav> streaming"
echo "  cargo run --example transcribe_test -- <audio.wav> all"
