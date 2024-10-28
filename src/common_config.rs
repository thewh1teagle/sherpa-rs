/// Configuration for the feature extractor
pub struct FeatureConfig {
    /// Sample rate expected by the model. It is 16000 for all
    /// pre-trained models provided by us
    pub sample_rate: i32,
    /// Feature dimension expected by the model. It is 80 for all
    /// pre-trained models provided by us
    pub feature_dim: i32,
}
