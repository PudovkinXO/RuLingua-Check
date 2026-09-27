from dataclasses import asdict, dataclass
from typing import Any

from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter

@dataclass(frozen=True, slots=True)
class Token:

    text: str
    lemma: str
    pos: str | None
    start: int
    stop: int

    def as_dict(self) -> dict[str, Any]:

        return asdict(self)


class NatashaPreprocessor:

    def __init__(self) -> None:
        self._segmenter = Segmenter()
        self._morph_vocab = MorphVocab()
        self._morph_tagger = NewsMorphTagger(NewsEmbedding())

    def process(self, text: str) -> list[Token]:

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text:
            return []
        
        doc = Doc(text)
        doc.segment(self._segmenter)
        doc.tag_morph(self._morph_tagger)
        
        for token in doc.tokens:
            token.lemmatize(self._morph_vocab)

        return [
            Token(
                text=token.text,
                lemma=token.lemma or token.text,
                pos=token.pos,
                start=token.start,
                stop=token.stop,
            )
            for token in doc.tokens
        ]

_DEFAULT_PREPROCESSOR = NatashaPreprocessor()

def preprocess(text: str) -> list[Token]:

    return _DEFAULT_PREPROCESSOR.process(text)

def preprocess_as_dicts(text: str) -> list[dict[str, Any]]:

    return [token.as_dict() for token in preprocess(text)]