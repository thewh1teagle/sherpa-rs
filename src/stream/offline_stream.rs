use sherpa_rs_sys::SherpaOnnxOfflineStream;

/// It wraps a pointer from C
struct OfflineStream {
    pointer: *const SherpaOnnxOfflineStream,
}
