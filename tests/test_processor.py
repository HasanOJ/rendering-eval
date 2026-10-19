from types import SimpleNamespace

import numpy as np
import pytest


def test_prequantized_subtree_exclusions_match_current_loader():
    from rendering_eval.backends import checkpoint_skip_patterns
    from transformers.quantizers.quantizers_utils import should_convert_module
    patterns = checkpoint_skip_patterns(["vision_tower", "model.language_model.layers.21.mlp"])
    assert not should_convert_module("model.vision_tower.patch_embedder.input_proj", patterns)
    assert not should_convert_module("model.language_model.layers.21.mlp.gate_proj", patterns)
    assert should_convert_module("model.language_model.layers.2.mlp.gate_proj", patterns)
    assert should_convert_module("model.language_model.layers.210.mlp.gate_proj", patterns)


def test_preview_reconstructs_real_patches_and_ignores_padding():
    torch = pytest.importorskip("torch")
    from rendering_eval.backends import reconstruct_processor_image
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    patches, positions = [], []
    for y in range(2):
        for x in range(3):
            patches.append(image[y*2:y*2+2, x*2:x*2+2].reshape(-1) / 255)
            positions.append([x, y])
    patches.append(np.full(12, 99))
    positions.append([-1, -1])
    inputs = {"pixel_values": torch.tensor(np.array([patches])),
              "image_position_ids": torch.tensor([positions])}
    processor = SimpleNamespace(image_processor=SimpleNamespace(
        patch_size=2, pooling_kernel_size=1, do_normalize=False, do_rescale=True, rescale_factor=1/255))
    reconstructed = reconstruct_processor_image(inputs, processor)
    np.testing.assert_array_equal(np.array(reconstructed), image)
