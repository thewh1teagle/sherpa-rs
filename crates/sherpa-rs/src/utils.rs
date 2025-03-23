use std::ffi::{c_char, CString};

pub fn cstring_from_str(s: &str) -> CString {
    CString::new(s).expect("CString::new failed")
}

pub unsafe fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}
