use std::convert::AsRef;
use std::path::Path;

use crate::Language;

struct Model {}

impl Model {
    fn load(size: ModelSize) -> Result<Model, ()> {
        unimplemented!()
    }

    fn transcribe(file: impl AsRef<Path>, lang: Language) -> Result<(), ()> {
        unimplemented!()
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ModelSize {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
}
