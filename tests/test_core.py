from dataclasses import replace
import json

import pytest

from rendering_eval.config import Config, INSTRUCTION
from rendering_eval.data import align_and_select
from rendering_eval.prompts import messages_for
from rendering_eval.scoring import score_answer


def row(lang, i, answer="1"):
    return {"link": f"https://example.org/{i}", "question_number": 1, "dialect": lang,
            "flores_passage": f"Passage {i}", "question": "Which choice?",
            **{f"mc_answer{n}": f"Option {n}" for n in range(1, 5)}, "correct_answer_num": answer}


@pytest.fixture
def config(tmp_path):
    return Config(output_dir=str(tmp_path / "run"), languages=("eng_Latn", "arb_Arab"), n_ids=2)


@pytest.fixture
def data(config):
    return {lang: [row(lang, i, str(i % 4 + 1)) for i in range(6)] for lang in config.languages}


def test_selection_is_aligned_and_order_independent(config, data):
    examples = align_and_select(data, config)
    assert len(examples) == 4
    assert examples == align_and_select({lang: list(reversed(rows)) for lang, rows in data.items()}, config)
    assert examples[:2] == align_and_select(data, replace(config, n_ids=1))
    assert len({e.sample_id for e in examples}) == 2
    assert all(a.answer == b.answer for a, b in zip(examples[::2], examples[1::2]))


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "answer", "empty", "dialect"])
def test_alignment_rejects_corrupt_data(config, data, corruption):
    rows = data["arb_Arab"]
    if corruption == "duplicate":
        rows.append(rows[0])
    elif corruption == "missing":
        rows.pop()
    elif corruption == "answer":
        rows[0]["correct_answer_num"] = "4"
    elif corruption == "empty":
        rows[0]["question"] = " "
    else:
        rows[0]["dialect"] = "wrong"
    with pytest.raises(ValueError):
        align_and_select(data, config)


def test_optional_source_split_disambiguates(config):
    data = {}
    for lang in config.languages:
        data[lang] = [{**row(lang, 0), "split": part} for part in ("dev", "test")]
    assert len(align_and_select(data, config)) == 4


def test_prompt_parity_and_no_answer_leak(config, data):
    example = align_and_select(data, config)[0]
    sentinel = object()
    text = messages_for(example, "text")[0]["content"]
    image = messages_for(example, "image", sentinel)[0]["content"]
    assert text[0]["text"] == example.task_text()
    assert text[-1] == image[-1] == {"type": "text", "text": INSTRUCTION}
    assert image[0] == {"type": "image", "image": sentinel}
    changed_key = replace(example, answer="D" if example.answer != "D" else "A")
    assert changed_key.task_text() == example.task_text()
    assert messages_for(changed_key, "text") == messages_for(example, "text")
    assert "correct_answer" not in json.dumps(text)


@pytest.mark.parametrize("raw", ["A", "a", " A.\n", "(a)", "(A)."])
def test_valid_scoring(raw):
    assert score_answer(raw, "A")["correct"]
    assert not score_answer(raw, "B")["correct"]
    assert not score_answer(raw, "A", truncated=True)["valid"]


@pytest.mark.parametrize("raw", ["", "Answer: A", "A or B", "Option 1", "AA", "(A", "A)", "A\nExplanation"])
def test_invalid_scoring(raw):
    assert not score_answer(raw, "A")["valid"]
