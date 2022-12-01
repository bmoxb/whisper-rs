use std::convert::{AsRef, TryFrom};
use std::fmt;
use std::path::Path;

use crate::{Error, Language, Result};

use numpy::convert::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::*;

pub struct Model {
    model: Py<PyAny>,
}

impl Model {
    pub fn new(
        size: ModelSize,
        specific_device: Option<Device>,
        download_path: Option<&Path>,
        in_memory: bool,
    ) -> Result<Self> {
        Python::with_gil(|py| {
            let whisper = PyModule::import(py, "whisper")?;

            let kwargs = PyDict::new(py);
            if let Some(device) = specific_device {
                kwargs.set_item("device", device.to_string())?;
            }
            if let Some(path) = download_path {
                let path = path.canonicalize()?;
                kwargs.set_item("download_root", path.display().to_string())?;
            }
            kwargs.set_item("in_memory", in_memory)?;

            whisper
                .getattr("load_model")?
                .call((size.to_string(),), Some(kwargs))
                .map(|m| Model { model: m.into() })
                .map_err(Into::into)
        })
    }

    pub fn from_size(size: ModelSize) -> Result<Self> {
        Model::new(size, None, None, false)
    }

    pub fn default() -> Result<Self> {
        Model::from_size(ModelSize::default())
    }

    pub fn transcribe_file(
        &self,
        path: impl AsRef<Path>,
        language: Option<Language>,
    ) -> Result<Transcription> {
        let full_path_string = path.as_ref().canonicalize()?.display().to_string();
        Python::with_gil(|py| self.transcribe(py, (full_path_string,), language))
    }

    pub fn transcribe_audio(
        &self,
        audio: Vec<f32>,
        language: Option<Language>,
    ) -> Result<Transcription> {
        Python::with_gil(|py| {
            let array = audio.into_pyarray(py);
            self.transcribe(py, (array,), language)
        })
    }

    fn transcribe(
        &self,
        py: Python<'_>,
        args: impl IntoPy<Py<PyTuple>>,
        language: Option<Language>,
    ) -> Result<Transcription> {
        let model = self.model.as_ref(py);

        let kwargs = PyDict::new(py);
        if let Some(language) = language {
            kwargs.set_item("language", language.to_string())?;
        }

        let dict: &PyDict = model
            .call_method("transcribe", args, Some(kwargs))?
            .downcast()
            .map_err(PyErr::from)?;

        dict.try_into()
    }
}

#[derive(Clone, Debug)]
pub struct Transcription {
    pub text: String,
    pub segments: Vec<Segment>,
    pub language: Language,
}

impl TryFrom<&PyDict> for Transcription {
    type Error = Error;

    fn try_from(dict: &PyDict) -> Result<Self> {
        let text = dict
            .get_item("text")
            .ok_or(Error::MissingResult("text"))?
            .extract()?;

        let segments = dict
            .get_item("segments")
            .ok_or(Error::MissingResult("segments"))?
            .iter()?
            .map(|s| {
                let dict: &PyDict = s?.downcast().map_err(PyErr::from)?;

                Ok(Segment {
                    text: dict
                        .get_item("text")
                        .ok_or(Error::MissingResult("segment text"))?
                        .extract()?,
                    start: dict
                        .get_item("start")
                        .ok_or(Error::MissingResult("segment start"))?
                        .extract()?,
                    end: dict
                        .get_item("end")
                        .ok_or(Error::MissingResult("segment end"))?
                        .extract()?,
                })
            })
            .collect::<Result<Vec<Segment>>>()?;

        let language = dict
            .get_item("language")
            .ok_or(Error::MissingResult("language"))?
            .extract::<&str>()?
            .try_into()?;

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

#[derive(Clone, Copy, Debug)]
pub enum ModelSize {
    Tiny,
    TinyEnglishOnly,
    Base,
    BaseEnglishOnly,
    Small,
    SmallEnglishOnly,
    Medium,
    MediumEnglishOnly,
    Large,
}

impl Default for ModelSize {
    fn default() -> Self {
        ModelSize::Small
    }
}

impl fmt::Display for ModelSize {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ModelSize::Tiny => write!(f, "tiny"),
            ModelSize::TinyEnglishOnly => write!(f, "tiny.en"),
            ModelSize::Base => write!(f, "base"),
            ModelSize::BaseEnglishOnly => write!(f, "base.en"),
            ModelSize::Small => write!(f, "small"),
            ModelSize::SmallEnglishOnly => write!(f, "small.en"),
            ModelSize::Medium => write!(f, "medium"),
            ModelSize::MediumEnglishOnly => write!(f, "medium.en"),
            ModelSize::Large => write!(f, "large"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Device {
    Cpu,
    Cuda(Option<usize>),
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(None) => write!(f, "cuda"),
            Device::Cuda(Some(index)) => write!(f, "cuda:{}", index),
        }
    }
}
