use std::convert::AsRef;
use std::path::Path;

use crate::Language;

use pyo3::prelude::*;
use pyo3::types::PyDict;

pub struct Model {
    model: Py<PyAny>,
}

impl Model {
    pub fn load(size: ModelSize) -> Result<Self, PyErr> {
        Python::with_gil(|py| {
            let whisper = PyModule::import(py, "whisper")?;

            whisper
                .getattr("load_model")?
                .call1((size.to_string(),))
                .map(|m| Model { model: m.into() })
        })
    }

    pub fn transcribe(
        &self,
        path: impl AsRef<Path>,
        language: Language,
    ) -> Result<Transcription, PyErr> {
        let full_path_string = path.as_ref().canonicalize().unwrap().display().to_string();

        Python::with_gil(|py| {
            let model = self.model.as_ref(py);

            let kwargs = PyDict::new(py);
            kwargs.set_item("language", language.to_string())?;

            let dict: &PyDict = model
                .call_method("transcribe", (full_path_string,), Some(kwargs))?
                .downcast()?;

            Ok(Transcription {
                text: dict.get_item("text").unwrap().extract()?,
                language,
            })
        })
    }
}

#[derive(Clone, Copy, strum_macros::Display, Debug)]
#[strum(serialize_all = "lowercase")]
pub enum ModelSize {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
}

#[derive(Clone, Debug)]
pub struct Transcription {
    pub text: String,
    pub language: Language,
}
