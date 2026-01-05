/// Configuration for online (streaming) Transducer model
#[derive(Debug, Clone, Default)]
pub struct OnlineTransducerModelConfig {
    pub encoder: String,
    pub decoder: String,
    pub joiner: String,
}

/// Configuration for online (streaming) Paraformer model
#[derive(Debug, Clone, Default)]
pub struct OnlineParaformerModelConfig {
    pub encoder: String,
    pub decoder: String,
}

/// Configuration for online (streaming) Zipformer2-CTC model
#[derive(Debug, Clone, Default)]
pub struct OnlineZipformer2CtcModelConfig {
    pub model: String,
}

/// Configuration for online (streaming) NeMo CTC model
#[derive(Debug, Clone, Default)]
pub struct OnlineNemoCtcModelConfig {
    pub model: String,
}

/// Enum representing different online model types
#[derive(Debug, Clone)]
pub enum OnlineModelType {
    Transducer(OnlineTransducerModelConfig),
    Paraformer(OnlineParaformerModelConfig),
    Zipformer2Ctc(OnlineZipformer2CtcModelConfig),
    NemoCtc(OnlineNemoCtcModelConfig),
}

impl Default for OnlineModelType {
    fn default() -> Self {
        Self::Transducer(OnlineTransducerModelConfig::default())
    }
}

/// Endpoint detection configuration
#[derive(Debug, Clone)]
pub struct EndpointConfig {
    /// Enable endpoint detection
    pub enable: bool,
    /// An endpoint is detected if trailing silence in seconds is larger than
    /// this value even if nothing has been decoded.
    pub rule1_min_trailing_silence: f32,
    /// An endpoint is detected if trailing silence in seconds is larger than
    /// this value after something that is not blank has been decoded.
    pub rule2_min_trailing_silence: f32,
    /// An endpoint is detected if the utterance in seconds is larger than
    /// this value.
    pub rule3_min_utterance_length: f32,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            enable: true,
            rule1_min_trailing_silence: 2.4,
            rule2_min_trailing_silence: 1.2,
            rule3_min_utterance_length: 20.0,
        }
    }
}

/// CTC FST decoder configuration for streaming recognition
#[derive(Debug, Clone, Default)]
pub struct OnlineCtcFstDecoderConfig {
    pub graph: String,
    pub max_active: i32,
}

/// Main configuration for online (streaming) recognizer
#[derive(Debug, Clone)]
pub struct OnlineRecognizerConfig {
    /// Model configuration
    pub model: OnlineModelType,
    /// Path to tokens.txt
    pub tokens: String,

    // Feature configuration
    /// Sample rate (default: 16000)
    pub sample_rate: i32,
    /// Feature dimension (default: 80)
    pub feature_dim: i32,

    // Decoding configuration
    /// Decoding method: "greedy_search" or "modified_beam_search"
    pub decoding_method: String,
    /// Used only when decoding_method is "modified_beam_search"
    pub max_active_paths: i32,

    // Endpoint detection
    /// Endpoint detection configuration
    pub endpoint: EndpointConfig,

    // Hotwords
    /// Path to hotwords file
    pub hotwords_file: Option<String>,
    /// Bonus score for each token in hotwords
    pub hotwords_score: f32,

    // CTC FST decoder
    pub ctc_fst_decoder: Option<OnlineCtcFstDecoderConfig>,

    // Rule FSTs
    pub rule_fsts: Option<String>,
    pub rule_fars: Option<String>,

    // Blank penalty
    pub blank_penalty: f32,

    // Runtime configuration
    /// Provider (cpu, cuda, coreml, etc.)
    pub provider: Option<String>,
    /// Number of threads
    pub num_threads: Option<i32>,
    /// Enable debug mode
    pub debug: bool,
}

impl Default for OnlineRecognizerConfig {
    fn default() -> Self {
        Self {
            model: OnlineModelType::default(),
            tokens: String::new(),
            sample_rate: 16000,
            feature_dim: 80,
            decoding_method: "greedy_search".to_string(),
            max_active_paths: 4,
            endpoint: EndpointConfig::default(),
            hotwords_file: None,
            hotwords_score: 1.5,
            ctc_fst_decoder: None,
            rule_fsts: None,
            rule_fars: None,
            blank_penalty: 0.0,
            provider: None,
            num_threads: Some(1),
            debug: false,
        }
    }
}

