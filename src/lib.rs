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
            Model::new(ModelSize::Base, Some(Device::Cpu), None, true)
                .unwrap()
                .transcribe_audio(x, None)
                .unwrap()
        );
    }
}
