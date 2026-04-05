import re
from text_to_num import alpha2digit

def collapse_spaced_digits(text: str) -> str:
    # collapse sequences like "6 1 6" -> "616"
    return re.sub(r"\b\d(?:\s+\d)+\b", lambda m: m.group(0).replace(" ", ""), text)

def normalize_for_wer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    text = alpha2digit(text, "en")
    text = collapse_spaced_digits(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

ref = "bctl3 target is nine hundred fifty feet"
pred = "busan tower singapore six one six"

print(normalize_for_wer(ref))
print(normalize_for_wer(pred))