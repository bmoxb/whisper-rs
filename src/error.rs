use std::io;

use pyo3::PyErr;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("string could not be matched to language: {0:?}")]
    InvalidLanguageString(String),
    #[error("IO error: {0}")]
    InputOutput(#[from] io::Error),
    #[error("result missing in Whisper output: {0}")]
    MissingResult(&'static str),
    #[error("Python error: {0}")]
    Python(#[from] PyErr),
}

pub type Result<T> = std::result::Result<T, Error>;
