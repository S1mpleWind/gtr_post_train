"""Optional outcome reward examples; choose explicitly with --reward-fn."""
import re
from decimal import Decimal, InvalidOperation


def _answer(text):
    text = str(text).strip()
    if "####" in text:
        text = text.rsplit("####", 1)[1]
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed[-1]
    numbers = re.findall(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)", text)
    if not numbers:
        return None
    try:
        return Decimal(numbers[-1].replace(",", ""))
    except InvalidOperation:
        return None


def gsm8k(tokenizer, record, response):
    """Numeric final-answer reward; expects `answer` or `reference` in the JSONL."""
    reference = record.get("answer", record.get("reference"))
    if reference is None:
        raise ValueError("gsm8k reward requires answer or reference")
    expected, actual = _answer(reference), _answer(response)
    if expected is None:
        raise ValueError("No numeric answer in reference")
    return float(actual is not None and actual == expected)
