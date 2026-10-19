"""Inspect saved mistakes without new inference or changing benchmark scores."""

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import prediction_index
from .config import Config
from .data import align_and_select, digest
from .reporting import LANGUAGE_NAMES, load_pairs


def response_description(row):
    """Describe observable response format, not a speculative model mechanism."""
    if row["valid"]:
        return "valid correct" if row["correct"] else "valid wrong"
    raw = row["raw_response"].lower()
    if row.get("predicted") is not None:
        behavior = "single choice not completed"
    elif "please provide the passage" in raw or "i need the passage" in raw:
        behavior = "requests passage"
    elif "incomplete" in raw:
        behavior = "claims incomplete input"
    else:
        behavior = "prose instead of a choice"
    return behavior + ("; token limit" if row["truncated"] else "; invalid format")


class ErrorExplorer:
    def __init__(self, output_dir):
        self.root = Path(output_dir)
        self.manifest, self.rows, self.pairs = load_pairs(self.root)
        self.index = prediction_index(self.rows)
        self._sources = None  # Source text is read only when a case is requested.
        p = self.pairs
        self.pairs["outcome"] = np.select(
            [~p.image_valid, p.text_correct & p.image_correct, p.text_correct,
             p.image_correct], ["invalid image", "both correct", "text only", "image only"],
            default="both wrong",
        )

    def diagnosis(self):
        p = self.pairs
        languages = self.manifest["experiment"]["languages"]
        invalids = [r for r in self.rows if not r["valid"]]
        conditional, confusion, language_errors, invalid_comparisons = [], [], [], []
        for language in languages:
            g = p[p.language == language]
            image_wrong = g[g.image_valid & ~g.image_correct]
            wrong_answers = Counter(image_wrong.image_predicted)
            dominant = max(wrong_answers, key=wrong_answers.get) if wrong_answers else None
            language_errors.append({
                "language": language, "n": len(g), "text_wrong": int((~g.text_correct).sum()),
                "image_valid_wrong": len(image_wrong), "image_invalid": int((~g.image_valid).sum()),
                "wrong_image_most_common_choice": dominant,
                "wrong_image_most_common_choice_n": wrong_answers.get(dominant, 0),
            })
            for mode in ("text", "image"):
                for answer in "ABCD":
                    a = g[g.answer == answer]
                    conditional.append({"language": language, "mode": mode, "gold": answer,
                                        "n": len(a), "correct": int(a[f"{mode}_correct"].sum()),
                                        "accuracy_pct": float(a[f"{mode}_correct"].mean() * 100) if len(a) else None})
            if (~g.image_valid).any():
                a, b = g[~g.image_valid], g[g.image_valid]
                invalid_comparisons.append({
                    "language": language, "invalid_n": len(a), "valid_n": len(b),
                    "invalid_height_median": float(a.render_height.median()),
                    "valid_height_median": float(b.render_height.median()),
                    "invalid_text_correct": int(a.text_correct.sum()),
                    "invalid_nominal_size_median": float(a.nominal_font_px_after_resize.median()),
                    "valid_nominal_size_median": float(b.nominal_font_px_after_resize.median()),
                })
        for mode in ("text", "image"):
            for gold in "ABCD":
                g = p[p.answer == gold]
                answers = Counter(g[f"{mode}_predicted"].where(g[f"{mode}_valid"], "Invalid"))
                confusion.append({"mode": mode, "gold": gold, "n": len(g),
                                  **{choice: answers[choice] for choice in (*"ABCD", "Invalid")}})
        invalid_details = []
        for row in invalids:
            paired = self.index.get((row["sample_id"], row["language"], "text"))
            english = self.index.get((row["sample_id"], "eng_Latn", "image"))
            invalid_details.append({
                "sample_id": row["sample_id"], "language": row["language"], "mode": row["mode"],
                "gold": row["answer"], "raw_response": row["raw_response"],
                "behavior": response_description(row), "truncated": row["truncated"],
                "generated_tokens": row["inference"].get("generated_tokens"),
                "text_correct": paired["correct"] if paired else None,
                "english_image_correct": english["correct"] if english else None,
            })
        image_errors = int((~p.image_correct).sum())
        invalid_image = int((~p.image_valid).sum())
        correct = int(p.image_correct.sum())
        return {
            "pairs": len(p), "image_errors": image_errors, "invalid_image": invalid_image,
            "valid_wrong_image": image_errors - invalid_image,
            "image_accuracy_pct": correct / len(p) * 100,
            "all_invalid_correct_upper_bound_pct": (correct + invalid_image) / len(p) * 100,
            "max_gain_from_invalid_pp": invalid_image / len(p) * 100,
            "invalid_prose_templates": len({r["raw_response"] for r in invalids}),
            "invalid_behavior_counts": dict(Counter(response_description(r) for r in invalids)),
            "invalid_details": invalid_details, "language_errors": language_errors,
            "invalid_comparisons": invalid_comparisons,
            "accuracy_by_gold": conditional, "confusion": confusion,
        }

    def config(self):
        experiment = self.manifest["experiment"]
        fields = {key: value for key, value in experiment.items() if key in Config.__dataclass_fields__}
        fields["quantization"] = "4bit" if "nf4" in (experiment.get("quantization") or "") else "8bit"
        return Config(**{**fields, "output_dir": str(self.root)})

    def sources(self):
        """Read pinned source JSONL strictly from the Hub cache; never download it."""
        if self._sources is None:
            from huggingface_hub import try_to_load_from_cache
            cfg = self.config()
            source_rows = {}
            for language in cfg.languages:
                path = try_to_load_from_cache(cfg.dataset_id, f"data/{language}.jsonl",
                                              repo_type="dataset", revision=cfg.dataset_revision)
                if not isinstance(path, str):
                    raise FileNotFoundError(f"Pinned source text is not cached for {language}; use the evaluation environment.")
                source_rows[language] = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
            examples = align_and_select(source_rows, cfg)
            selection = {(s["sample_id"], s["language"]): s for s in self.manifest["selection"]}
            for example in examples:
                if digest(asdict(example)) != selection[example.sample_id, example.language]["content_sha256"]:
                    raise ValueError("Cached source text differs from the evaluated manifest")
                for mode in ("text", "image"):
                    row = self.index.get((example.sample_id, example.language, mode))
                    if row is not None and row["answer"] != example.answer:
                        raise ValueError("Saved answer differs from the pinned source answer")
            self._sources = {(e.sample_id, e.language): e for e in examples}
        return self._sources

    def case(self, sample_id, language):
        example = self.sources()[sample_id, language]
        english = self.sources().get((sample_id, "eng_Latn"))
        comparison = []
        for lang in dict.fromkeys([language, "eng_Latn"]):
            for mode in ("text", "image"):
                r = self.index.get((sample_id, lang, mode))
                if r:
                    comparison.append({
                        "language": lang, "mode": mode, "gold": r["answer"],
                        "raw_response": r["raw_response"], "correct": r["correct"],
                        "valid": r["valid"], "truncated": r["truncated"],
                        "generated_tokens": r["inference"].get("generated_tokens"),
                        "source_height": r["rendering"].get("height"),
                        "processed_width": r["inference"].get("processed_width"),
                        "processed_height": r["inference"].get("processed_height"),
                    })
        return {"example": example, "english": english, "comparison": comparison}


def plot_error_patterns(diagnosis):
    import matplotlib.pyplot as plt

    languages = [r["language"] for r in diagnosis["language_errors"]]
    figure, axes = plt.subplots(2, 2, figsize=(14, 11), layout="constrained")
    for column, mode in enumerate(("text", "image")):
        values = {(r["language"], r["gold"]): r["accuracy_pct"] for r in diagnosis["accuracy_by_gold"]
                  if r["mode"] == mode}
        matrix = np.array([[values[lang, label] if values[lang, label] is not None else np.nan
                            for label in "ABCD"] for lang in languages])
        axes[0, column].imshow(matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        axes[0, column].set(xticks=range(4), xticklabels=list("ABCD"), xlabel="Correct answer label",
                            yticks=range(len(languages)), yticklabels=[LANGUAGE_NAMES.get(x, x) for x in languages],
                            title=f"{mode.capitalize()} accuracy (%) conditional on gold label")
        for y in range(len(languages)):
            for x in range(4):
                if np.isfinite(matrix[y, x]):
                    axes[0, column].text(x, y, f"{matrix[y, x]:.1f}", ha="center", va="center",
                                            fontsize=9, color="white" if matrix[y, x] > 55 else "#17334a")
        rows = [r for r in diagnosis["confusion"] if r["mode"] == mode]
        matrix = np.array([[r[c] / r["n"] * 100 if r["n"] else np.nan for c in (*"ABCD", "Invalid")] for r in rows])
        axes[1, column].imshow(matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        axes[1, column].set(xticks=range(5), xticklabels=[*"ABCD", "Invalid"], xlabel="Model answer",
                            yticks=range(4), yticklabels=list("ABCD"), ylabel="Correct answer",
                            title=f"{mode.capitalize()} confusion · row percentages, all languages")
        for y in range(4):
            for x in range(5):
                if np.isfinite(matrix[y, x]):
                    axes[1, column].text(x, y, f"{matrix[y, x]:.1f}%", ha="center", va="center",
                                            color="white" if matrix[y, x] > 55 else "#17334a")
    figure.suptitle("Most failures are valid wrong choices · label-conditional accuracy exposes the pattern", fontsize=15)
    return figure


# Fixed examples from this pinned full run; these are illustrative, not a random sample.
CASE_STUDIES = [
    {"title": "Burmese: asks for a passage that is present", "language": "mya_Mymr",
     "sample_id": "ec6ee41709d98fddee37cb465f725e1aad24841122ddc030277a53e61f072f64",
     "note": "The image response requests the passage; the same-language text response is correct. Inspect the full page and processor reconstruction before interpreting the request as actual missing input."},
    {"title": "Arabic: claims incomplete input", "language": "arb_Arab",
     "sample_id": "20fe38e977b81e084ad76c78caa1cb5baeb86f2d0ac916f8a49b764190839992",
     "note": "The English reference asks what Singapore is renowned for (urban greenery, A). The Arabic image response claims incomplete text; check the actual page rather than treating that statement as an input audit."},
    {"title": "Khmer: discusses translating instead of answering", "language": "khm_Khmr",
     "sample_id": "db6fc973a4cc3e26748f2b167845dac2dd6ed1801b464df57613c8e8d87f695e",
     "note": "Hostel guests are typically in their twenties; the gold answer is the group of 22-year-olds (B). The image output begins a translation/explanation preamble and uses its entire token allowance."},
    {"title": "Khmer: selects A on a short numeric question", "language": "khm_Khmr",
     "sample_id": "561206ca4b4d74cc6709e55684ea3a09a75480550926b67c39f24f603f520a2d",
     "note": "The English passage says the typhoon was 70 km from Fujian (C). Khmer text and both English modes answer correctly, while Khmer image selects A (9 km). This documents a modality-specific failure, without deciding whether its cause is reading or reasoning."},
    {"title": "Arabic: selects the frequent B distractor", "language": "arb_Arab",
     "sample_id": "617f326892020aefd14ccc7040bd9de3c8aa53d66f5ce4bca57913a771a3ba4a",
     "note": "The acquisition is Skype (C); Apple (B) is the image answer. Arabic text and both English modes are correct. This is consistent with the aggregate B preference, but an individual error does not establish its mechanism."},
    {"title": "English: chooses euros when asked for dollars", "language": "eng_Latn",
     "sample_id": "c75336fa952a85daf31d7ba47a0f94db2e57d72cac92e6f7e63919d862016e36",
     "note": "The passage gives 10 billion euros (US$14.7 billion). The question asks for about how many billion dollars: gold C (14), text C, image D (10). The selected number matches the euro amount; the saved output cannot distinguish a reading failure from a unit-selection error."},
    {"title": "Both English modes: origin versus destination", "language": "eng_Latn",
     "sample_id": "4b8f353b40ce301a3f60b83ff8d73aec3577dfa3a8f53cd82a806463b3485783",
     "note": "Vidal joined Barca from Sevilla. The question asks which club he joined three seasons ago: gold C (Barca), both modes A (Sevilla). This is a shared error, rather than an error unique to image input."},
    {"title": "Image succeeds where text fails: a negated question", "language": "eng_Latn",
     "sample_id": "44c1a3fd444f7f4b48a41d3efa3d33bc0452de2a1eb4b7789b76435de954a73a",
     "note": "The question asks which statement does NOT describe nerve impulses' purpose. Gold A (slowing blood flow), image A, text B (managing blood flow). Include image-only successes when reviewing mistakes; images do not fail on every question."},
]
