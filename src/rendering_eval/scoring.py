import re


def score_answer(raw, answer, *, truncated=False):
    match = re.fullmatch(r"\s*(?:([A-D])\.?|\(([A-D])\)\.?)\s*", raw, flags=re.IGNORECASE)
    predicted = next((x.upper() for x in match.groups() if x), None) if match else None
    valid = predicted is not None and not truncated
    return {"predicted": predicted, "valid": valid, "truncated": truncated,
            "correct": valid and predicted == answer}
