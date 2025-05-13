use std::ffi::{c_char, CString};

pub fn cstring_from_str(s: &str) -> RawCStr {
    RawCStr::new(s)
}

pub fn cstr_to_string(ptr: *const c_char) -> String {
    unsafe {
        if ptr.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

pub struct RawCStr {
    ptr: *mut std::ffi::c_char,
}

impl RawCStr {
    /// Creates a new `CStr` from a given Rust string slice.
    pub fn new(s: &str) -> Self {
        let cstr = CString::new(s).expect("CString::new failed");
        let ptr = cstr.into_raw();
        Self { ptr }
    }

    /// Returns the raw pointer to the internal C string.
    ///
    /// # Safety
    /// This function only returns the raw pointer and does not transfer ownership.
    /// The pointer remains valid as long as the `CStr` instance exists.
    /// Be cautious not to deallocate or modify the pointer after using `CStr::new`.
    pub fn as_ptr(&self) -> *const c_char {
        self.ptr
    }
}

impl Drop for RawCStr {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = CString::from_raw(self.ptr);
            }
        }
    }
}
