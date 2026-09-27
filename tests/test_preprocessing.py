from preprocessing import NatashaPreprocessor, preprocess_as_dicts

def test_preprocess_returns_lemmas_pos_and_offsets() -> None:
    tokens = NatashaPreprocessor().process("Кошки бегут!")

    assert [token.text for token in tokens] == ["Кошки", "бегут", "!"]
    assert [token.lemma for token in tokens] == ["кошка", "бежать", "!"]
    assert [token.pos for token in tokens] == ["NOUN", "VERB", "PUNCT"]
    assert [(token.start, token.stop) for token in tokens] == [(0, 5), (6, 11), (11, 12)]

def test_preprocess_as_dicts_is_serializable() -> None:
    assert preprocess_as_dicts("Привет.") == [
        {
            "text": "Привет",
            "lemma": "привет",
            "pos": "NOUN",
            "start": 0,
            "stop": 6,
        },
        {
            "text": ".",
            "lemma": ".",
            "pos": "PUNCT",
            "start": 6,
            "stop": 7,
        },
    ]

def test_empty_text_returns_no_tokens() -> None:
    assert NatashaPreprocessor().process("") == []
