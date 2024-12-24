/*
We assume you have pre-downloaded the model files for testing
from https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

cargo run --example punctuate
*/

fn main() {
    let model = "./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx";
    let sentences = [
        "这是一个测试你好吗How are you我很好thank you are you ok谢谢你", 
        "我们都是木头人不会说话不会动", 
        "The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry"
    ];

    let config = sherpa_rs::punctuate::PunctuationConfig {
        model: model.into(),
        ..Default::default()
    };
    let mut punctuate = sherpa_rs::punctuate::Punctuation::new(config).unwrap();

    println!("--------------------");
    for sentence in sentences {
        let punctuated = punctuate.add_punctuation(sentence);
        println!("Input text: {}", sentence);
        println!("Output text: {}", punctuated);
        println!("--------------------");
    }
}
