import json

import pytest

from detector import RuleBasedDetector


def test_disputed_low_confidence_rules_are_hidden_by_default(tmp_path) -> None:
    borrowings = tmp_path / "borrowings.json"
    patterns = tmp_path / "patterns.json"
    borrowings.write_text(json.dumps({"_meta": {}, "тест": {}}), encoding="utf-8")
    patterns.write_text(
        json.dumps(
            {
                "_meta": {},
                "паттерны": [
                    {
                        "id": "disputed",
                        "детекция": "regex",
                        "regex": r"\bтест\b",
                        "категория": "спорно",
                        "уверенность": "низкая",
                        "риск_ложного_срабатывания": "высокий",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    detector = RuleBasedDetector(
        borrowings_path=borrowings,
        patterns_path=patterns,
    )

    assert detector.detect("Тест") == []
    assert detector.detect("Тест", include_disputed=True)[0].rule_id == "disputed"
