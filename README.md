# Предобработка русского текста

Модуль [`preprocessing.py`](./preprocessing.py) использует Natasha для сегментации
русского текста, лемматизации и определения универсальных POS-тегов.

Установка зависимостей:

```bash
python -m pip install -r requirements.txt
```

Пример:

```python
from preprocessing import preprocess_as_dicts

tokens = preprocess_as_dicts("Кошки бегут!")
```

Каждый результат содержит исходный текст токена, лемму, POS-тег и диапазон
`start`/`stop` в исходной строке. Пунктуация возвращается отдельными токенами.

## Rule-based детектор

[`detector.py`](./detector.py) применяет словари:

```python
from detector import RuleBasedDetector

detector = RuleBasedDetector()
findings = detector.detect("Мы работаем на ежедневной основе.")
```

Слой 1 сопоставляется по леммам. Записи с `многословный: true` сопоставляются
как последовательности лемм. Regex-правила слоя 3 выполняются первым проходом;
`syntax_tree` и `pos_pattern` — вторым проходом после dependency parsing Natasha.
Результаты сортируются по уверенности и риску ложного срабатывания. Правила
категории `спорно` с низкой уверенностью по умолчанию скрыты; для явного показа
используйте `detect(text, include_disputed=True)`.
