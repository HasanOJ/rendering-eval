from dataclasses import replace
import json
from pathlib import Path

from PIL import Image
import pytest

from rendering_eval.analysis import analyze, plot_summary
from rendering_eval.artifacts import read_jsonl
from rendering_eval.backends import Generation
from rendering_eval.data import align_and_select
from rendering_eval.runner import prepare, run
from test_core import config, data  # noqa: F401


class Renderer:
    font_metadata = {"fixture": True}

    def render(self, example):
        return Image.new("RGB", (20, 10), "white"), {"width": 20, "height": 10}


class Model:
    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after

    def describe(self):
        return {"backend": "test-fixture"}

    def generate(self, example, mode, image=None):
        if self.calls == self.fail_after:
            raise RuntimeError("simulated interruption")
        self.calls += 1
        assert (image is not None) == (mode == "image")
        return Generation(example.answer if mode == "text" else "not a letter")


def test_resume_and_no_image_writes(config, data, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Images must never be saved")
    monkeypatch.setattr(Image.Image, "save", forbidden)
    examples = align_and_select(data, config)
    prepare(config, examples=examples, renderer=Renderer())
    assert analyze(config.output_dir)["groups"][0]["text_accuracy"] is None
    with pytest.raises(RuntimeError, match="interruption"):
        run(config, examples=examples, renderer=Renderer(), backend=Model(fail_after=3), progress=lambda _: None)
    path = Path(config.output_dir) / "predictions.jsonl"
    original = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(b'{"interrupted":')
    model = Model()
    run(config, examples=examples, renderer=Renderer(), backend=model, progress=lambda _: None)
    assert model.calls == 5
    assert path.read_bytes().startswith(original)
    assert len(read_jsonl(path)) == 8
    completed = path.read_bytes()
    run(config, examples=examples, progress=lambda _: None)  # Must not construct a real model/renderer.
    assert path.read_bytes() == completed
    monkeypatch.setattr("rendering_eval.artifacts.implementation_digest", lambda: "new-code-version")
    run(config, examples=examples, progress=lambda _: None)
    assert path.read_bytes() == completed
    summary = analyze(config.output_dir)
    assert summary["status"] == "complete"
    assert summary["groups"][0]["text_accuracy"] == 1
    assert summary["groups"][0]["image_accuracy"] == 0
    assert summary["groups"][0]["invalid_responses"] == 4
    assert summary["execution_error_attempts"] == 1
    figure = plot_summary(summary)
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert labels == ["Chance", "Text", "Image"]
    from matplotlib import pyplot as plt
    plt.close(figure)
    assert {p.suffix for p in Path(config.output_dir).iterdir()} <= {".json", ".jsonl", ".lock"}


def test_mismatch_and_duplicates_rejected(config, data):
    examples = align_and_select(data, config)
    run(config, examples=examples, renderer=Renderer(), backend=Model(), progress=lambda _: None)
    with pytest.raises(ValueError, match="Incompatible"):
        run(replace(config, font_size=32), examples=examples, renderer=Renderer(), backend=Model())
    path = Path(config.output_dir) / "predictions.jsonl"
    with path.open("a") as handle:
        handle.write(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        analyze(config.output_dir)


def test_corrupt_complete_line_fails(tmp_path):
    path = tmp_path / "corrupt.jsonl"
    path.write_bytes(b'{bad}\n')
    with pytest.raises(ValueError, match="Corrupt complete"):
        read_jsonl(path, repair=True)
    assert path.read_bytes() == b'{bad}\n'


def test_valid_final_line_without_newline(tmp_path):
    path = tmp_path / "valid.jsonl"
    path.write_text('{"x": 1}')
    assert read_jsonl(path, repair=True) == [{"x": 1}]
    assert path.read_bytes().endswith(b"\n")


@pytest.mark.parametrize("fail_after", [None, 1])
def test_owned_model_released_on_success_or_error(config, data, monkeypatch, fail_after):
    models = []
    class OwnedModel(Model):
        def __init__(self, cfg):
            super().__init__(fail_after=fail_after)
            self.closed = False
            models.append(self)
        def close(self):
            self.closed = True
    monkeypatch.setattr("rendering_eval.backends.GemmaBackend", OwnedModel)
    examples = align_and_select(data, config)
    if fail_after is None:
        run(config, examples=examples, renderer=Renderer(), progress=lambda _: None)
    else:
        with pytest.raises(RuntimeError, match="interruption"):
            run(config, examples=examples, renderer=Renderer(), progress=lambda _: None)
    assert len(models) == 1 and models[0].closed


def test_notebook_is_clean():
    notebook = json.loads(Path("Belebele.ipynb").read_text())
    for cell in notebook["cells"]:
        assert not cell.get("outputs")
        assert cell.get("execution_count") is None
    source = "".join("".join(c["source"]) for c in notebook["cells"])
    assert "RUN_EVALUATION = False" in source
    assert "clear_previews" in source
