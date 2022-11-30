mod language;
mod model;

pub use language::Language;
pub use model::*;

use pyo3::prelude::*;
use pyo3::types::PyDict;

fn testing123() -> PyResult<String> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| {
        let os = PyModule::import(py, "os")?;
        let s: &str = os.getattr("getcwd")?.call0()?.extract()?;
        println!("{}", s);

        let whisper = PyModule::import(py, "whisper")?;

        let args = ("base",);
        let model = whisper.getattr("load_model")?.call1(args)?;
        let result: &PyDict = model
            .call_method1("transcribe", ("audio.mp3",))?
            .downcast()?;

        result.get_item("text").unwrap().extract()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn my_test() {
        assert_eq!(testing123().unwrap(), " Hello world.");
    }
}
