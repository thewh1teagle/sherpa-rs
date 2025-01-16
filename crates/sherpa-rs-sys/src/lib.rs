#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

macro_rules! impl_default_for_bindgen {
    ($struct_name:ident { $($field_name:ident : $field_type:ty),* $(,)? }) => {
        impl Default for $struct_name {
            fn default() -> Self {
                Self {
                    $(
                        $field_name: default_value::<$field_type>(),
                    )*
                }
            }
        }
    };
}

macro_rules! impl_default_for_bindgen {
    ($struct_name:ident) => {
        impl Default for $struct_name {
            fn default() -> Self {
                unsafe { std::mem::zeroed() }
            }
        }
    };
}

impl_default_for_bindgen!(SherpaOnnxOfflineTtsMatchaModelConfig);
impl_default_for_bindgen!(SherpaOnnxOfflineTtsVitsModelConfig);
