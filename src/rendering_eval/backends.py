"""Backend interface and the initial local Gemma implementation."""

from dataclasses import dataclass, field
from typing import Protocol
import time
import re
import math

from .prompts import messages_for


@dataclass
class Generation:
    raw_response: str
    truncated: bool = False
    metadata: dict = field(default_factory=dict)


class Backend(Protocol):
    def describe(self) -> dict: ...
    def generate(self, example, mode: str, image=None) -> Generation: ...


def checkpoint_skip_patterns(module_names):
    """Match whole checkpoint module names anywhere in the instantiated module tree.

    Current Transformers iterates descendants independently; excluding a bare parent
    name no longer excludes its children unless the pattern includes the full path.
    """
    return [r"(?:^|.*\.)" + re.escape(name) + r"(?:\.|$)" for name in module_names]


def load_processor(config):
    if config.backend != "local":
        raise ValueError("OpenRouter image preprocessing is provider-managed; no local processor preview is available")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(config.model_id, revision=config.model_revision)
    processor.image_processor.max_soft_tokens = config.max_soft_tokens
    return processor


def encode(processor, config, example, mode, image=None):
    return processor.apply_chat_template(
        messages_for(example, mode, image), tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=True, enable_thinking=False,
        processor_kwargs={"images_kwargs": {"max_soft_tokens": config.max_soft_tokens}} if mode == "image" else {},
    )


def processor_patch_geometry(inputs, processor):
    """E4B emits teacher patches; Unified 12B emits already merged patches."""
    features = inputs["pixel_values"].shape[-1]
    patch = math.isqrt(features // 3)
    base = processor.image_processor.patch_size
    pooled = base * processor.image_processor.pooling_kernel_size
    if features != patch * patch * 3 or patch not in (base, pooled):
        raise ValueError(f"Unsupported processor patch geometry: {features} values")
    return patch, (pooled // patch) ** 2


def image_input_metadata(inputs, processor):
    positions = inputs["image_position_ids"][0].detach().cpu().numpy()
    valid = (positions >= 0).all(axis=1)
    positions = positions[valid]
    patch, patches_per_token = processor_patch_geometry(inputs, processor)
    return {
        "processed_width": int(positions[:, 0].max() + 1) * patch,
        "processed_height": int(positions[:, 1].max() + 1) * patch,
        "real_patches": int(valid.sum()), "padded_patches": int((~valid).sum()),
        "soft_tokens": int(valid.sum()) // patches_per_token,
    }


def reconstruct_processor_image(inputs, processor):
    """Invert actual patches, discarding padding; never approximate by resizing the original."""
    import numpy as np
    from PIL import Image

    info = image_input_metadata(inputs, processor)
    image_processor = processor.image_processor
    patch, _ = processor_patch_geometry(inputs, processor)
    positions = inputs["image_position_ids"][0].detach().cpu().numpy()
    values = inputs["pixel_values"][0].detach().float().cpu().numpy()
    canvas = np.zeros((info["processed_height"], info["processed_width"], 3), dtype=np.float32)
    for flat, (x, y) in zip(values, positions):
        if x < 0 or y < 0:
            continue
        canvas[y * patch:(y + 1) * patch, x * patch:(x + 1) * patch] = flat.reshape(patch, patch, 3)
    if image_processor.do_normalize:
        canvas = canvas * np.asarray(image_processor.image_std) + np.asarray(image_processor.image_mean)
    if image_processor.do_rescale:
        canvas = canvas / image_processor.rescale_factor
    return Image.fromarray(np.clip(np.rint(canvas), 0, 255).astype(np.uint8))


class GemmaBackend:
    def __init__(self, config, processor=None):
        import torch
        from transformers import AutoConfig, AutoModelForMultimodalLM, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Run in GPU-enabled WSL or a GPU cloud environment.")
        self.config = config
        self.processor = processor if processor is not None else load_processor(config)
        self.device = torch.device("cuda:0")
        self.compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        saved = AutoConfig.from_pretrained(config.model_id, revision=config.model_revision)
        if config.prequantized:
            quantization = getattr(saved, "quantization_config", {})
            if quantization.get("quant_method") != "bitsandbytes" or not quantization.get("load_in_4bit"):
                raise ValueError("The selected checkpoint does not contain prequantized 4-bit weights")
            if quantization.get("bnb_4bit_quant_type") != "nf4":
                raise ValueError("The selected checkpoint does not use the configured NF4 format")
            if quantization.get("bnb_4bit_compute_dtype") == "bfloat16" and self.compute_dtype != torch.bfloat16:
                raise RuntimeError("This prequantized checkpoint requires a BF16-capable GPU")
            self.checkpoint_skip_modules = list(quantization.get("llm_int8_skip_modules", []))
            # Preserve the stored BF16 subtrees under the current loader's pattern matching.
            saved.quantization_config = {**quantization, "llm_int8_skip_modules":
                                         checkpoint_skip_patterns(self.checkpoint_skip_modules)}
            loading_options = {"config": saved}
        else:
            if getattr(saved, "quantization_config", None):
                raise ValueError("Quantize-on-load requires an unquantized source checkpoint; "
                                 "select the official BF16 model instead of requantizing stored weights")
            quantization = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.compute_dtype,
            ) if config.quantization == "4bit" else BitsAndBytesConfig(load_in_8bit=True)
            loading_options = {"quantization_config": quantization}
        free, total = torch.cuda.mem_get_info()
        self.initial_memory = {"free_bytes": free, "total_bytes": total}
        self.model = AutoModelForMultimodalLM.from_pretrained(
            config.model_id, revision=config.model_revision,
            **loading_options,
            dtype=self.compute_dtype, device_map={"": 0}, attn_implementation=config.attention_implementation,
        ).eval()
        if not getattr(self.model, "is_loaded_in_4bit" if config.quantization == "4bit" else "is_loaded_in_8bit", False):
            raise RuntimeError(f"Model did not load with the requested {config.quantization} quantization")
        if config.prequantized:
            from bitsandbytes.nn import Linear4bit
            missing = [name for name, module in self.model.named_modules()
                       if isinstance(module, Linear4bit) and getattr(module.weight, "quant_state", None) is None]
            if missing:
                self.close()
                raise RuntimeError(f"Checkpoint quantization state is missing for: {missing[:5]}")

    def describe(self):
        import torch
        device_map = getattr(self.model, "hf_device_map", {"": self.device})
        quantization = self.model.config.quantization_config
        if hasattr(quantization, "to_dict"):
            quantization = quantization.to_dict()
        return {
            "backend": "transformers-gemma", "model_id": self.config.model_id,
            "model_revision": self.config.model_revision,
            "loaded_in_4bit": bool(getattr(self.model, "is_loaded_in_4bit", False)),
            "loaded_in_8bit": bool(getattr(self.model, "is_loaded_in_8bit", False)),
            "quantization_config": quantization,
            "prequantized_checkpoint": self.config.prequantized,
            "checkpoint_skip_modules": getattr(self, "checkpoint_skip_modules", None),
            "attention_implementation": self.config.attention_implementation,
            "compute_dtype": str(self.compute_dtype), "gpu": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "device_map": {k: str(v) for k, v in device_map.items()},
            "model_footprint_bytes": self.model.get_memory_footprint(),
            "initial_memory": self.initial_memory,
            "image_processor": self.processor.image_processor.to_dict(),
        }

    def generate(self, example, mode, image=None):
        import torch
        inputs = encode(self.processor, self.config, example, mode, image)
        extra = image_input_metadata(inputs, self.processor) if mode == "image" else {}
        input_length = inputs["input_ids"].shape[-1]
        inputs = inputs.to(self.device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.config.max_new_tokens,
                use_cache=True,
            )
        torch.cuda.synchronize()
        generated = outputs[0, input_length:].detach().cpu()
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [])
        truncated = len(generated) >= self.config.max_new_tokens and int(generated[-1]) not in eos
        raw = self.processor.decode(generated, skip_special_tokens=True)
        return Generation(raw, truncated, {
            **extra, "input_tokens": input_length, "generated_tokens": len(generated),
            "latency_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        })

    def close(self):
        """Release owned model weights and the allocator's unused GPU blocks."""
        import gc
        import torch
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
