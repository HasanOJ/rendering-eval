import json
from pathlib import Path

import pytest

from rendering_eval.error_analysis import ErrorExplorer, response_description
from rendering_eval.scoring import score_answer
from test_reporting import completed_run  # noqa: F401


def test_invalid_accounting_does_not_rescore_or_hide_valid_errors(completed_run):
    path = completed_run / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(r for r in rows if r["language"] == "eng_Latn" and r["mode"] == "image" and not r["correct"])
    row["raw_response"] = "I need the passage. Please provide the passage"
    row.update(score_answer(row["raw_response"], row["answer"], truncated=True))
    row["inference"]["generated_tokens"] = 16
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    original = path.read_bytes()
    explorer = ErrorExplorer(completed_run)
    result = explorer.diagnosis()
    assert result["image_errors"] == 4
    assert result["invalid_image"] == 1
    assert result["valid_wrong_image"] == 3
    assert result["max_gain_from_invalid_pp"] == pytest.approx(100 / 6)
    assert result["all_invalid_correct_upper_bound_pct"] == pytest.approx(50)
    assert result["invalid_details"][0]["text_correct"] is True
    assert result["invalid_behavior_counts"] == {"requests passage; token limit": 1}
    assert (explorer.pairs.outcome == "invalid image").sum() == 1
    assert path.read_bytes() == original
    confusion = [r for r in result["confusion"] if r["mode"] == "image"]
    assert sum(sum(r[key] for key in (*"ABCD", "Invalid")) for r in confusion) == 6


def test_truncated_letter_is_not_misclassified_as_prose():
    row = {"raw_response": "A", **score_answer("A", "A", truncated=True)}
    assert response_description(row) == "single choice not completed; token limit"
    row = {"raw_response": "(a).", **score_answer("(a).", "A")}
    assert response_description(row) == "valid correct"


def test_error_notebook_is_clean_and_inference_is_absent():
    notebook = json.loads(Path("Belebele_errors.ipynb").read_text())
    assert all(not cell.get("outputs") and cell.get("execution_count") is None for cell in notebook["cells"])
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "SHOW_INPUT_PREVIEW = False" in code
    assert "local_files_only=True" in code
    assert "GemmaBackend" not in code
    assert "from rendering_eval.runner" not in code
