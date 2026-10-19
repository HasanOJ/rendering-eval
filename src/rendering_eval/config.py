from dataclasses import asdict, dataclass
from pathlib import Path
import re

LANGUAGES = (
    "eng_Latn", "rus_Cyrl", "arb_Arab", "hin_Deva", "ben_Beng", "tel_Telu",
    "tha_Thai", "khm_Khmr", "mya_Mymr", "zho_Hans", "jpn_Jpan", "kor_Hang",
)
DATASET_REVISION = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
MODEL_REVISION = "6f10b2aa522ba471321d8246fd71a3dfa8c5b9a2"
RENDERER_REVISION = "9706296afca3dc156d76ff82450fff0fefc6f3d7"
FONT_REVISION = "601944d2ee97e76d4c456dc3d790db358c0c6ec5"
INSTRUCTION = (
    "Read the passage and answer the multiple-choice question. "
    "Reply with exactly one letter: A, B, C, or D."
)


@dataclass(frozen=True)
class Config:
    output_dir: str = "results/belebele-smoke-4bit"
    n_ids: int = 2
    seed: int = 42
    languages: tuple[str, ...] = LANGUAGES
    dataset_id: str = "facebook/belebele"
    dataset_revision: str = DATASET_REVISION
    model_id: str = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
    model_revision: str = MODEL_REVISION
    backend: str = "local"
    quantization: str = "4bit"
    prequantized: bool = True
    attention_implementation: str = "eager"
    openrouter_free_only: bool = False
    openrouter_provider: str | None = None
    request_timeout: int = 120
    font_size: int = 24
    width: int = 1024
    margin: int = 24
    max_soft_tokens: int = 280
    max_new_tokens: int = 16

    def __post_init__(self):
        object.__setattr__(self, "languages", tuple(self.languages))
        if not self.languages or len(set(self.languages)) != len(self.languages):
            raise ValueError("Languages must be nonempty and unique")
        if set(self.languages) - set(LANGUAGES):
            raise ValueError("This version supports the documented 12-language subset")
        if not 1 <= self.n_ids <= 900:
            raise ValueError("n_ids must be between 1 and 900")
        if self.font_size < 1 or self.margin < 1 or self.width <= 2 * self.margin:
            raise ValueError("Invalid rendering dimensions")
        if self.max_soft_tokens not in (70, 140, 280, 560, 1120):
            raise ValueError("Unsupported Gemma vision-token budget")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.backend not in ("local", "openrouter"):
            raise ValueError("backend must be local or openrouter")
        if self.quantization not in ("4bit", "8bit"):
            raise ValueError("quantization must be 4bit or 8bit")
        if self.backend == "local" and self.prequantized and self.quantization != "4bit":
            raise ValueError("Direct prequantized loading requires a 4-bit checkpoint")
        if self.attention_implementation not in ("eager", "sdpa"):
            raise ValueError("attention_implementation must be eager or sdpa")
        if self.request_timeout < 1:
            raise ValueError("request_timeout must be positive")
        if self.openrouter_free_only and (self.backend != "openrouter" or not self.model_id.endswith(":free")):
            raise ValueError("Free-only requests require an explicit OpenRouter :free model")
        revisions = [self.dataset_revision]
        if self.backend == "local":
            revisions.append(self.model_revision)
        for revision in revisions:
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Dataset and model revisions must be full commit hashes")

    @classmethod
    def from_yaml(cls, path):
        import yaml
        return cls(**(yaml.safe_load(Path(path).read_text()) or {}))

    def experiment(self):
        result = asdict(self)
        result.pop("output_dir")
        result.pop("request_timeout")
        if self.backend == "openrouter":
            result.update(model_revision=None, quantization=None, attention_implementation=None,
                          prequantized=None, max_soft_tokens=None, image_preprocessing="provider-managed")
        else:
            result["quantization"] = ("bitsandbytes-prequantized-nf4" if self.prequantized else
                                      "bitsandbytes-nf4-double" if self.quantization == "4bit" else "bitsandbytes-int8")
        return {
            **result, "languages": list(self.languages), "instruction": INSTRUCTION,
            "renderer_revision": RENDERER_REVISION, "font_revision": FONT_REVISION,
            "batch_size": 1,
            "enable_thinking": False, "do_sample": False,
            "layout_version": 1, "scoring_version": 1,
        }
