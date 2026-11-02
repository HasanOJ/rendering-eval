from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from rendering_eval.config import Config


def test_model_configs_and_historical_baseline():
    configs = [Config.from_yaml(f"configs/{name}.yaml") for name in (
        "smoke", "full", "smoke-12b-4bit", "full-12b-4bit",
        "smoke-e4b-4bit", "full-e4b-4bit",
    )]
    assert len({c.output_dir for c in configs}) == 6
    for smoke, full in zip(configs[::2], configs[1::2]):
        assert smoke.n_ids == 2 and full.n_ids == 900
        assert replace(smoke, n_ids=900).experiment() == full.experiment()
    assert Config() == configs[0]
    assert configs[2].model_id == "google/gemma-4-12B-it"
    assert configs[2].experiment()["quantization"] == "bitsandbytes-nf4-double"
    # Historical configurations must still describe the saved experiments exactly.
    for config in configs[-2:]:
        manifest = Path(config.output_dir) / "manifest.json"
        if manifest.exists():
            assert config.experiment() == json.loads(manifest.read_text())["experiment"]


@pytest.mark.parametrize("unified", [False, True])
def test_actual_processors_reconstruct_patches_and_count_tokens(unified):
    from rendering_eval.backends import image_input_metadata, reconstruct_processor_image
    if unified:
        from transformers import Gemma4UnifiedImageProcessor as Processor
    else:
        from transformers import Gemma4ImageProcessor as Processor
    # Exact multiples of 48: exercise 3x3 merging and padding, without interpolation.
    pixels = np.random.default_rng(42).integers(0, 256, (96, 144, 3), dtype=np.uint8)
    processor = SimpleNamespace(image_processor=Processor(do_resize=False, max_soft_tokens=70))
    with Image.fromarray(pixels) as original:
        encoded = processor.image_processor(images=original, return_tensors="pt")
    metadata = image_input_metadata(encoded, processor)
    assert metadata["soft_tokens"] == 6
    assert (metadata["processed_width"], metadata["processed_height"]) == (144, 96)
    with reconstruct_processor_image(encoded, processor) as reconstructed:
        np.testing.assert_array_equal(np.asarray(reconstructed), pixels)


@pytest.mark.parametrize("name", ["smoke", "smoke-12b-4bit", "smoke-e4b-4bit"])
def test_loading_uses_requested_quantization_and_device(monkeypatch, name):
    import torch
    import transformers
    from rendering_eval.backends import GemmaBackend
    config = Config.from_yaml(f"configs/{name}.yaml")
    saved = SimpleNamespace(quantization_config={
        "quant_method": "bitsandbytes", "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bfloat16",
        "llm_int8_skip_modules": ["vision_tower"],
    } if config.prequantized else None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (12 * 1024**3, 12 * 1024**3))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: saved)
    calls = []
    class Model:
        is_loaded_in_4bit = config.quantization == "4bit"
        is_loaded_in_8bit = config.quantization == "8bit"
        def eval(self):
            return self
        def named_modules(self):
            return []
    def load(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return Model()
    monkeypatch.setattr(transformers.AutoModelForMultimodalLM, "from_pretrained", load)
    GemmaBackend(config, processor=object())
    model_id, options = calls[0]
    assert model_id == config.model_id and options["revision"] == config.model_revision
    assert options["device_map"] == {"": 0}
    assert options["dtype"] == torch.bfloat16
    if config.prequantized:
        assert options["config"] is saved and "quantization_config" not in options
        assert saved.quantization_config["llm_int8_skip_modules"] != ["vision_tower"]
    else:
        quant = options["quantization_config"]
        assert quant.load_in_4bit == (config.quantization == "4bit")
        assert quant.load_in_8bit == (config.quantization == "8bit")
        if quant.load_in_4bit:
            assert quant.bnb_4bit_quant_type == "nf4" and quant.bnb_4bit_use_double_quant
    # Requantizing a stored checkpoint must fail before model weights are loaded.
    saved.quantization_config = {"quant_method": "bitsandbytes", "load_in_4bit": True}
    with pytest.raises(ValueError, match="unquantized source"):
        GemmaBackend(replace(config, prequantized=False), processor=object())
    assert len(calls) == 1


def test_new_model_notebook_is_clean_and_opt_in():
    notebook = json.loads(Path("Belebele_models.ipynb").read_text())
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in notebook["cells"])
    sources = "".join("".join(c["source"]) for c in notebook["cells"])
    assert "RUN_EVALUATION = False" in sources
    assert "configs/smoke-12b-4bit.yaml" in sources
