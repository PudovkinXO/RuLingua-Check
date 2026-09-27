from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from natasha import (
    Doc,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Segmenter,
)

from preprocessing import Token

_CONFIDENCE = {"низкая": 1, "средняя": 2, "высокая": 3}
_RISK = {"низкий": 3, "средний": 2, "высокий": 1}

@dataclass(frozen=True, slots=True)
class Finding:

    layer: int
    rule_id: str
    text: str
    start: int
    stop: int
    category: str
    confidence: str
    false_positive_risk: str
    priority: int
    explanation: str | None = None
    alternative: str | list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "layer": self.layer,
            "rule_id": self.rule_id,
            "text": self.text,
            "start": self.start,
            "stop": self.stop,
            "category": self.category,
            "confidence": self.confidence,
            "false_positive_risk": self.false_positive_risk,
            "priority": self.priority,
        }
        if self.explanation is not None:
            result["explanation"] = self.explanation
        if self.alternative is not None:
            result["alternative"] = self.alternative
        return result

class RuleBasedDetector:

    def __init__(
        self,
        *,
        borrowings_path: str | Path | None = None,
        patterns_path: str | Path | None = None,
    ) -> None:
        root = Path(__file__).resolve().parent / "f_files"
        self._borrowings = self._load_json(
            Path(borrowings_path) if borrowings_path else root / "anglicisms_borrowings.json"
        )
        patterns = self._load_json(
            Path(patterns_path) if patterns_path else root / "syntactic_patterns.json"
        )
        self._patterns = patterns["паттерны"]
        self._segmenter = Segmenter()
        self._morph_vocab = MorphVocab()
        embeddings = NewsEmbedding()
        self._morph_tagger = NewsMorphTagger(embeddings)
        self._syntax_parser = NewsSyntaxParser(embeddings)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Dictionary not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid dictionary JSON: {path}") from exc

    def detect(self, text: str, *, include_disputed: bool = False) -> list[Finding]:

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        doc = Doc(text)
        doc.segment(self._segmenter)
        doc.tag_morph(self._morph_tagger)

        for token in doc.tokens:
            token.lemmatize(self._morph_vocab)
            
        tokens = [
            Token(
                text=token.text,
                lemma=token.lemma or token.text,
                pos=token.pos,
                start=token.start,
                stop=token.stop,
            )
            for token in doc.tokens
        ]

        findings = self._layer_one(tokens)
        findings.extend(self._regex_patterns(text))

        deferred = [rule for rule in self._patterns if rule.get("детекция") in {"syntax_tree", "pos_pattern"}]
        if deferred:
            doc.parse_syntax(self._syntax_parser)
            findings.extend(self._syntax_patterns(doc, tokens, deferred))

        findings = self._deduplicate_overlapping(findings)

        visible = [
            finding
            for finding in findings
            if include_disputed
            or not (
                finding.category == "спорно"
                and finding.confidence == "низкая"
            )
        ]
        return sorted(visible, key=lambda item: (-item.priority, item.start, item.stop))

    @staticmethod
    def _deduplicate_overlapping(findings: list[Finding]) -> list[Finding]:
        ordered = sorted(
            findings,
            key=lambda finding: (-finding.priority, -(finding.stop - finding.start)),
        )
        kept: list[Finding] = []
        for finding in ordered:
            overlaps_existing = any(
                finding.start < other.stop and finding.stop > other.start
                for other in kept
            )
            if not overlaps_existing:
                kept.append(finding)
        return kept

    def _layer_one(self, tokens: list[Token]) -> list[Finding]:
        findings: list[Finding] = []
        entries = [(term, data) for term, data in self._borrowings.items() if term != "_meta"]
        lemmas = [token.lemma.lower() for token in tokens]
        for term, data in entries:
            phrase = bool(data.get("многословный"))
            expected = term.lower().split()
            if phrase:
                matches: Iterable[tuple[int, int]] = (
                    (index, index + len(expected))
                    for index in range(len(lemmas) - len(expected) + 1)
                    if lemmas[index : index + len(expected)] == expected
                )
            else:
                matches = (
                    (index, index + 1)
                    for index, lemma in enumerate(lemmas)
                    if lemma == expected[0]
                )
            for start_index, stop_index in matches:
                start = tokens[start_index].start
                stop = tokens[stop_index - 1].stop
                findings.append(
                    self._finding(
                        layer=1,
                        rule_id=term,
                        text=tokens[start_index].text
                        if stop_index == start_index + 1
                        else " ".join(token.text for token in tokens[start_index:stop_index]),
                        start=start,
                        stop=stop,
                        data=data,
                        context_sensitive=bool(data.get("контекст_чувствительность")),
                    )
                )
        return findings

    def _regex_patterns(self, text: str) -> list[Finding]:
        findings = []
        for rule in self._patterns:
            if rule.get("детекция") != "regex" or not rule.get("regex"):
                continue
            for match in re.finditer(rule["regex"], text, flags=re.IGNORECASE):
                findings.append(
                    self._finding(
                        layer=3,
                        rule_id=rule["id"],
                        text=match.group(0),
                        start=match.start(),
                        stop=match.end(),
                        data=rule,
                    )
                )
        return findings

    def _syntax_patterns(
        self, doc: Doc, tokens: list[Token], rules: list[dict[str, Any]]
    ) -> list[Finding]:
        findings: list[Finding] = []
        for rule in rules:
            if rule.get("детекция") == "pos_pattern" and rule["id"] == "genitive_chain":
                findings.extend(self._genitive_chains(doc, tokens, rule))
                continue
            if rule.get("regex"):
                for match in re.finditer(rule["regex"], doc.text, flags=re.IGNORECASE):
                    if self._has_dependency_context(doc, match.start(), match.end()):
                        findings.append(
                            self._finding(
                                layer=3,
                                rule_id=rule["id"],
                                text=match.group(0),
                                start=match.start(),
                                stop=match.end(),
                                data=rule,
                            )
                        )
        return findings

    @staticmethod
    def _has_dependency_context(doc: Doc, start: int, stop: int) -> bool:
        return any(
            token.start is not None
            and token.stop is not None
            and token.start < stop
            and token.stop > start
            and getattr(token, "rel", None)
            for token in doc.tokens
        )

    def _genitive_chains(
        self, doc: Doc, tokens: list[Token], rule: dict[str, Any]
    ) -> list[Finding]:
        syntax_tokens = doc.tokens
        findings = []
        for index in range(len(syntax_tokens) - 2):
            group = syntax_tokens[index : index + 3]
            ids = {token.id for token in group}
            has_internal_dependency = any(
                token.head_id in ids and str(token.rel or "").startswith("nmod")
                for token in group
            )
            if (
                has_internal_dependency
                and all(token.pos == "NOUN" and token.feats.get('Case') == 'Gen' for token in group)
            ):
                findings.append(
                    self._finding(
                        layer=3,
                        rule_id=rule["id"],
                        text=doc.text[group[0].start : group[-1].stop],
                        start=group[0].start,
                        stop=group[-1].stop,
                        data=rule,
                    )
                )
        return findings

    @staticmethod
    def _finding(
        *,
        layer: int,
        rule_id: str,
        text: str,
        start: int,
        stop: int,
        data: dict[str, Any],
        context_sensitive: bool = False,
    ) -> Finding:
        confidence = data.get("уверенность", "средняя")
        risk = data.get("риск_ложного_срабатывания", "средний")
        priority = _CONFIDENCE.get(confidence, 2) + _RISK.get(risk, 2)
        if context_sensitive:
            priority -= 1
        return Finding(
            layer=layer,
            rule_id=rule_id,
            text=text,
            start=start,
            stop=stop,
            category=data.get("категория") or data.get("тип") or "не_указано",
            confidence=confidence,
            false_positive_risk=risk,
            priority=priority,
            explanation=data.get("объяснение"),
            alternative=data.get("альтернатива") or data.get("альтернативы"),
        )
