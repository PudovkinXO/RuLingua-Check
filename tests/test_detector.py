import json

from detector import RuleBasedDetector


def test_disputed_low_confidence_rules_are_hidden_by_default(tmp_path) -> None:
    borrowings = tmp_path / "borrowings.json"
    patterns = tmp_path / "patterns.json"
    borrowings.write_text(json.dumps({"_meta": {}}), encoding="utf-8")
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
    findings = detector.detect("Тест", include_disputed=True)
    assert [finding.rule_id for finding in findings] == ["disputed"]
