#[derive(strum_macros::Display, Debug)]
#[strum(serialize_all = "lowercase")]
pub enum Language {
    English,
    Chinese,
    German,
    Spanish,
    Russian,
    Korean,
    French,
    Japanese,
    Portuguese,
    Turkish,
    Polish,
    Catalan,
    Dutch,
    Arabic,
    Swedish,
    Italian,
    Indonesian,
    Hindi,
    Finnish,
    Vietnamese,
    Hebrew,
    Ukrainian,
    Greek,
    Malay,
    Czech,
    Romanian,
    Danish,
    Hungarian,
    Tamil,
    Norwegian,
    Thai,
    Urdu,
    Croatian,
    Bulgarian,
    Lithuanian,
    Latin,
    Maori,
    Malayalam,
    Welsh,
    Slovak,
    Telugu,
    Persian,
    Latvian,
    Bengali,
    Serbian,
    Azerbaijani,
    Slovenian,
    Kannada,
    Estonian,
    Macedonian,
    Breton,
    Basque,
    Icelandic,
    Armenian,
    Nepali,
    Mongolian,
    Bosnian,
    Kazakh,
    Albanian,
    Swahili,
    Galician,
    Marathi,
    Punjabi,
    Sinhala,
    Khmer,
    Shona,
    Yoruba,
    Somali,
    Afrikaans,
    Occitan,
    Georgian,
    Belarusian,
    Tajik,
    Sindhi,
    Gujarati,
    Amharic,
    Yiddish,
    Lao,
    Uzbek,
    Faroese,
    #[strum(serialize = "haitian creole")]
    HaitianCreole,
    Pashto,
    Turkmen,
    Nynorsk,
    Maltese,
    Sanskrit,
    Luxembourgish,
    Myanmar,
    Tibetan,
    Tagalog,
    Malagasy,
    Assamese,
    Tatar,
    Hawaiian,
    Lingala,
    Hausa,
    Bashkir,
    Javanese,
    Sundanese,
}

#[cfg(test)]
mod tests {
    use super::Language;

    #[test]
    fn language_to_string() {
        assert_eq!(Language::English.to_string(), "english");
        assert_eq!(Language::Japanese.to_string(), "japanese");
        assert_eq!(Language::HaitianCreole.to_string(), "haitian creole");
    }
}
