import json
from pathlib import Path

import pytest

from rendering_eval.reporting import (
    build_report, load_pairs, plot_answer_bias,
    plot_diagnostics, plot_overview,
)
from rendering_eval.scoring import score_answer


@pytest.fixture
def completed_run(tmp_path):
    languages = ["eng_Latn", "arb_Arab"]
    selection, rows = [], []
    for language in languages:
        for question in range(3):
            passage = "p1" if question < 2 else "p2"
            selection.append({"sample_id": str(question), "passage_id": passage, "language": language})
            for mode in ("text", "image"):
                correct = (question < 2) == (mode == "text")
                raw = "A" if correct else "B"
                rows.append({
                    "sample_id": str(question), "passage_id": passage, "language": language,
                    "mode": mode, "answer": "A", "raw_response": raw,
                    **score_answer(raw, "A"), "manifest_fingerprint": "fixture",
                    "rendering": {"height": 500 + question * 100, "width": 1024,
                                  "font_size": 24, "unknown_glyphs": 0} if mode == "image" else {},
                    "inference": {"processed_width": 1008, "processed_height": 576, "soft_tokens": 252,
                                  "latency_seconds": .5},
                })
    (tmp_path / "manifest.json").write_text(json.dumps({
        "fingerprint": "fixture", "selection": selection,
        "experiment": {"n_ids": 3, "languages": languages, "font_size": 24},
    }))
    (tmp_path / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return tmp_path


def test_audit_replays_scores_and_leaves_originals_unchanged(completed_run):
    before = {p.name: p.read_bytes() for p in completed_run.iterdir()}
    report = build_report(completed_run)
    assert report["audit"]["predictions"] == 12
    assert report["audit"]["paired_examples"] == 6
    assert report["audit"]["unique_passages"] == 2
    assert report["groups"][0]["text_accuracy"] == pytest.approx(2 / 3)
    assert report["groups"][0]["image_accuracy"] == pytest.approx(1 / 3)
    assert report["groups"][0]["image_minus_text_pp"] == pytest.approx(-100 / 3)
    assert "uncertainty" not in report
    assert all(not any("ci95" in key for key in group) for group in report["groups"])
    assert {p.name: p.read_bytes() for p in completed_run.iterdir()} == before
    path = completed_run / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["correct"] = not rows[0]["correct"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Stored scores"):
        load_pairs(completed_run)


def test_analysis_figures_never_save_images(completed_run, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    def forbidden(*args, **kwargs):
        raise AssertionError("Plot files must not be saved")

    monkeypatch.setattr(Figure, "savefig", forbidden)
    report = build_report(completed_run)
    for plot in (plot_overview, plot_answer_bias, plot_diagnostics):
        figure = plot(report)
        assert len(figure.axes) >= 2
        plt.close(figure)
    assert {p.suffix for p in completed_run.iterdir()} <= {".json", ".jsonl"}


def test_analysis_notebook_is_clean_and_has_no_inference():
    notebook = json.loads(Path("Belebele_analysis.ipynb").read_text())
    for cell in notebook["cells"]:
        assert not cell.get("outputs")
        assert cell.get("execution_count") is None
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "build_report" in code
    assert "from rendering_eval.runner" not in code
    assert "from rendering_eval.backends" not in code
    assert "savefig" not in code
    assert 'RESULTS_DIR = Path("results/belebele-e4b-8bit-full")' in code
    assert "bootstrap" not in code.lower()
    assert "ci95" not in code


@pytest.mark.parametrize("model,quantization,precision", [
    ("google/gemma-4-E4B-it", "bitsandbytes-int8", "INT8"),
    ("google/gemma-4-12B-it", "bitsandbytes-nf4-double", "NF4"),
])
def test_overview_title_uses_saved_model_and_precision(completed_run, model, quantization, precision):
    import matplotlib.pyplot as plt
    report = build_report(completed_run)
    report["experiment"].update(model_id=model, quantization=quantization)
    figure = plot_overview(report)
    assert model.rsplit("/", 1)[-1] in figure._suptitle.get_text()
    assert precision in figure._suptitle.get_text()
    plt.close(figure)
