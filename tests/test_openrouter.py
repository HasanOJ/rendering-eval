from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest

from rendering_eval.config import Config
from rendering_eval.data import Example
from rendering_eval.openrouter import OpenRouterBackend, remote_messages


def example():
    return Example("id", "passage", "eng_Latn", "Paris is in France.", "Where is Paris?",
                   ("France", "Italy", "Spain", "Germany"), "A")


def config():
    return Config(backend="openrouter", model_id="google/gemma-4-26b-a4b-it:free", openrouter_free_only=True)


class Transport:
    def __init__(self, *, image_support=True, price="0", finish="stop"):
        self.calls, self.finish = [], finish
        self.model = {"id": config().model_id, "pricing": {"prompt": price, "completion": price},
                      "architecture": {"input_modalities": ["text", "image"] if image_support else ["text"]},
                      "supported_parameters": ["seed", "temperature", "reasoning"]}

    def __call__(self, path, **kwargs):
        if path == "/models":
            assert "api_key" not in kwargs
            return {"data": [self.model]}
        self.calls.append(kwargs)
        return {"id": "generation-test", "model": config().model_id, "provider": "fixture",
                "choices": [{"message": {"content": "A"}, "finish_reason": self.finish}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0}}


def test_remote_pair_parity_and_memory_only(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    original_save = Image.Image.save
    def only_memory(self, fp, *args, **kwargs):
        assert isinstance(fp, BytesIO)
        return original_save(self, fp, *args, **kwargs)
    monkeypatch.setattr(Image.Image, "save", only_memory)
    transport = Transport()
    backend = OpenRouterBackend(config(), transport=transport)
    result = backend.generate(example(), "text")
    with Image.new("RGB", (40, 40), "white") as image:
        backend.generate(example(), "image", image)
    text, image = [c["payload"] for c in transport.calls]
    assert text["messages"][0]["content"][-1] == image["messages"][0]["content"][-1]
    assert image["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert text["provider"]["max_price"] == {"prompt": 0, "completion": 0}
    assert text["provider"]["allow_fallbacks"] is False
    assert text["reasoning"]["enabled"] is False
    assert result.raw_response == "A" and result.metadata["cost"] == 0
    assert "test-secret" not in json.dumps(backend.describe())
    assert "data:image" not in json.dumps(result.metadata)


@pytest.mark.parametrize("kwargs", [{"image_support": False}, {"price": "0.01"}])
def test_catalog_rejects_incompatible_or_paid_model(monkeypatch, kwargs):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    with pytest.raises(ValueError):
        OpenRouterBackend(config(), transport=Transport(**kwargs))


def test_auth_required_and_paid_model_not_accepted_in_free_mode(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterBackend(config(), transport=Transport())
    with pytest.raises(ValueError, match="Free-only"):
        replace(config(), model_id="qwen/qwen3.6-27b")


def test_remote_truncation_and_backend_identity(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    backend = OpenRouterBackend(config(), transport=Transport(finish="length"))
    assert backend.generate(example(), "text").truncated
    assert config().experiment()["quantization"] is None
    assert config().experiment()["max_soft_tokens"] is None
    assert Config().experiment()["quantization"] == "bitsandbytes-prequantized-nf4"
    assert replace(Config(), quantization="8bit", prequantized=False).experiment() != Config().experiment()


def test_full_configuration_and_cli_overrides(monkeypatch):
    from rendering_eval.cli import main
    assert Config.from_yaml("configs/full.yaml").n_ids == 900
    assert Config.from_yaml("configs/full.yaml").quantization == "4bit"
    received = []
    monkeypatch.setattr("rendering_eval.runner.run", lambda cfg, **kwargs: received.append(cfg))
    monkeypatch.setattr("rendering_eval.analysis.analyze", lambda *args, **kwargs: {})
    main(["run", "--n-ids", "900", "--output-dir", "results/full", "--quantization", "4bit"])
    assert received[0].n_ids == 900 and received[0].output_dir == "results/full"
