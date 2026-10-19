import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1", reason="Set RUN_INTEGRATION=1 to download pinned inputs")]


def test_real_languages_render_and_processor(tmp_path):
    from rendering_eval.config import Config
    from rendering_eval.data import load_examples
    from rendering_eval.rendering import DocumentRenderer
    from rendering_eval.backends import load_processor, encode, reconstruct_processor_image, image_input_metadata
    from rendering_eval.runner import prepare

    config = Config(output_dir=str(tmp_path / "integration"))
    examples = load_examples(config)
    assert len(examples) == 24
    assert len({e.sample_id for e in examples}) == 2
    renderer = DocumentRenderer(config)
    prepare(config, examples=examples, renderer=renderer)
    processor = load_processor(config)
    for example in examples:
        original, info = renderer.render(example)
        assert info["unknown_glyphs"] == 0
        assert all(name.startswith("Noto Sans") for name in info["actual_fonts"])
        encoded = encode(processor, config, example, "image", original)
        metadata = image_input_metadata(encoded, processor)
        assert 0 < metadata["soft_tokens"] <= 280
        preview = reconstruct_processor_image(encoded, processor)
        assert preview.size == (metadata["processed_width"], metadata["processed_height"])
        original.close()
        preview.close()
    assert not list(Path(config.output_dir).glob("*.png"))
