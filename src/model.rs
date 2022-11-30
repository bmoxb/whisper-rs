use std::convert::{AsRef, TryFrom};
use std::path::Path;

use crate::Language;

use pyo3::prelude::*;
use pyo3::types::*;

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
        language: Option<Language>,
    ) -> Result<Transcription, PyErr> {
        let full_path_string = path.as_ref().canonicalize().unwrap().display().to_string();

        Python::with_gil(|py| {
            let model = self.model.as_ref(py);

            let kwargs = PyDict::new(py);
            if let Some(language) = language {
                kwargs.set_item("language", language.to_string())?;
            }

            let dict: &PyDict = model
                .call_method("transcribe", (full_path_string,), Some(kwargs))?
                .downcast()?;

            dict.try_into()
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
    pub segments: Vec<Segment>,
    pub language: Language,
}

impl TryFrom<&PyDict> for Transcription {
    type Error = PyErr;

    fn try_from(dict: &PyDict) -> Result<Self, Self::Error> {
        let text = dict.get_item("text").unwrap().extract()?;

        let segments = dict
            .get_item("segments")
            .unwrap()
            .iter()?
            .map(|s| {
                let dict: &PyDict = s?.downcast()?;
                Ok(Segment {
                    text: dict.get_item("text").unwrap().extract()?,
                    start: dict.get_item("start").unwrap().extract()?,
                    end: dict.get_item("end").unwrap().extract()?,
                })
            })
            .collect::<Result<Vec<Segment>, PyErr>>()?;

        let language = dict
            .get_item("language")
            .unwrap()
            .extract::<&str>()?
            .try_into()
            .unwrap();

        Ok(Transcription {
            text,
            segments,
            language,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Segment {
    pub text: String,
    pub start: f32,
    pub end: f32,
}
