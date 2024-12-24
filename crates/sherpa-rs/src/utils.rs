use std::ffi::CString;

// Smart pointer for CString
pub struct RawCStr {
    #[cfg(target_os = "android")]
    ptr: *mut u8,

    #[cfg(not(target_os = "android"))]
    ptr: *mut i8,
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
    #[cfg(target_os = "android")]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    #[cfg(not(target_os = "android"))]
    pub fn as_ptr(&self) -> *const i8 {
        self.ptr
    }
}

impl Drop for RawCStr {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                #[cfg(target_os = "android")]
                let _ = CString::from_raw(self.ptr as *mut u8);

                #[cfg(not(target_os = "android"))]
                let _ = CString::from_raw(self.ptr);
            }
        }
    }
}

#[cfg(target_os = "android")]
pub fn cstr_to_string(ptr: *const u8) -> String {
    unsafe {
        if ptr.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

#[cfg(not(target_os = "android"))]
pub fn cstr_to_string(ptr: *const i8) -> String {
    unsafe {
        if ptr.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}
