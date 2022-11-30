mod language;
mod model;

pub use language::Language;
pub use model::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn my_test() {
        println!(
            "{:?}",
            Model::new(ModelSize::Base, Some(Device::Cuda(None)), None, true)
                .unwrap()
                .transcribe("br2049.mp3", None)
                .unwrap()
        );
    }
}
