"""Read-only run auditing, paired passage bootstrap, and in-memory result plots.

This module never loads a model, downloads data, or writes image files. Bootstrap
units are passage IDs: their questions AND all translations move together.
"""

from collections import Counter
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import analyze
from .artifacts import prediction_index, read_jsonl
from .scoring import score_answer


LANGUAGE_NAMES = {
    "eng_Latn": "English", "rus_Cyrl": "Russian", "arb_Arab": "Arabic (MSA)",
    "hin_Deva": "Hindi", "ben_Beng": "Bengali", "tel_Telu": "Telugu",
    "tha_Thai": "Thai", "khm_Khmr": "Khmer", "mya_Mymr": "Burmese",
    "zho_Hans": "Chinese (simplified)", "jpn_Jpan": "Japanese", "kor_Hang": "Korean",
}
TEXT_COLOR = "#2563a6"
IMAGE_COLOR = "#df752d"


def _name(language):
    return LANGUAGE_NAMES.get(language, language)


def paired_passage_bootstrap(pairs, languages, *, repetitions=5000, seed=42):
    """Percentile 95% intervals for text, image, and paired image-minus-text.

    Ratio estimates weight questions equally, even when passage sizes differ.
    The same resampled passages are used for every language and both modalities.
    These are corpus-sampling intervals, not model-seed/quantization uncertainty.
    """
    if repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    passages = sorted(pairs.passage_id.unique())
    if not passages:
        raise ValueError("No complete pairs to bootstrap")
    columns = ["ALL", *languages]
    totals = np.zeros((len(passages), len(columns), 3), dtype=float)
    passage_index = {value: i for i, value in enumerate(passages)}
    for column, language in enumerate(columns):
        group = pairs if language == "ALL" else pairs[pairs.language == language]
        for row in group.itertuples():
            totals[passage_index[row.passage_id], column] += (
                row.text_correct, row.image_correct, 1,
            )
    # Multinomial multiplicities implement sampling N passage IDs with replacement.
    weights = np.random.default_rng(seed).multinomial(
        len(passages), np.full(len(passages), 1 / len(passages)), size=repetitions,
    )
    samples = (weights @ totals.reshape(len(passages), -1)).reshape(repetitions, len(columns), 3)
    intervals = {}
    for i, language in enumerate(columns):
        nonempty = samples[:, i, 2] > 0
        values = samples[nonempty, i, :2] / samples[nonempty, i, 2, None] * 100
        if not len(values):
            continue
        values = np.column_stack([values, values[:, 1] - values[:, 0]])
        intervals[language] = {
            key: np.quantile(values[:, j], [0.025, 0.975]).tolist()
            for j, key in enumerate(("text_ci95_pct", "image_ci95_pct", "gap_ci95_pp"))
        }
    return intervals


def load_pairs(output_dir):
    """Validate original records and replay strict scoring without changing them."""
    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    rows = read_jsonl(root / "predictions.jsonl")
    indexed = prediction_index(rows)
    selection = {(x["sample_id"], x["language"]): x for x in manifest["selection"]}
    if len(selection) != len(manifest["selection"]):
        raise ValueError("Duplicate manifest selections")
    expected = {(*key, mode) for key in selection for mode in ("text", "image")}
    if set(indexed) - expected:
        raise ValueError("Unexpected prediction identity")
    answers = {}
    for row in rows:
        if row["manifest_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Prediction fingerprint differs from manifest")
        if row["passage_id"] != selection[row["sample_id"], row["language"]]["passage_id"]:
            raise ValueError("Prediction passage differs from manifest")
        if row["answer"] not in ("A", "B", "C", "D"):
            raise ValueError("Invalid answer key")
        previous = answers.setdefault(row["sample_id"], row["answer"])
        if previous != row["answer"]:
            raise ValueError("Answers differ between modalities or translations")
        scored = score_answer(row["raw_response"], row["answer"], truncated=row["truncated"])
        if any(row[key] != value for key, value in scored.items()):
            raise ValueError("Stored scores disagree with strict scoring replay")
    paired = []
    for key in sorted(selection):
        if (*key, "text") not in indexed or (*key, "image") not in indexed:
            continue
        text, image = indexed[(*key, "text")], indexed[(*key, "image")]
        render, inference = image.get("rendering", {}), image.get("inference", {})
        record = {
            "sample_id": key[0], "language": key[1], "passage_id": text["passage_id"],
            "answer": text["answer"],
        }
        for mode, row in (("text", text), ("image", image)):
            for field in ("correct", "valid", "predicted", "truncated"):
                record[f"{mode}_{field}"] = row[field]
        for field in ("width", "height", "font_size", "unknown_glyphs"):
            record[f"render_{field}"] = render.get(field, np.nan)
        for field in ("processed_width", "processed_height", "soft_tokens"):
            record[field] = inference.get(field, np.nan)
        scale = min(record["processed_width"] / record["render_width"],
                    record["processed_height"] / record["render_height"])
        record["nominal_font_px_after_resize"] = record["render_font_size"] * scale
        paired.append(record)
    if not paired:
        raise ValueError("No completed text/image pairs to analyze")
    return manifest, rows, pd.DataFrame(paired)


def build_report(output_dir, *, repetitions=5000, seed=42):
    """Return JSON-serializable derived metrics; leave the original run untouched."""
    root = Path(output_dir)
    manifest, rows, pairs = load_pairs(root)
    summary = analyze(root, save=False)
    languages = manifest["experiment"]["languages"]
    intervals = paired_passage_bootstrap(pairs, languages, repetitions=repetitions, seed=seed)
    groups = [{**g, **intervals.get(g["language"], {})} for g in summary["groups"]]
    answer_distributions, invalid, resizing, quartiles = [], [], [], []
    frame = pd.DataFrame(rows)
    gold = Counter(pairs.answer)
    for language in languages:
        selected = pairs[pairs.language == language].copy()
        for mode in ("text", "image"):
            # A truncated letter is invalid for scoring and is counted as Invalid here.
            answers = selected[f"{mode}_predicted"].where(selected[f"{mode}_valid"], "Invalid")
            counts = Counter(answers)
            answer_distributions.append({
                "language": language, "mode": mode, "n": len(selected),
                "counts": {label: int(counts[label]) for label in ("A", "B", "C", "D", "Invalid")},
            })
            responses = frame[(frame.language == language) & (frame["mode"] == mode)]
            invalid.append({
                "language": language, "mode": mode, "n": len(responses),
                "invalid": int((~responses.valid).sum()), "truncated": int(responses.truncated.sum()),
            })
        if selected.render_height.notna().any():
            resizing.append({
                "language": language,
                "source_height_min": float(selected.render_height.min()),
                "source_height_median": float(selected.render_height.median()),
                "source_height_max": float(selected.render_height.max()),
                "nominal_font_px_median": float(selected.nominal_font_px_after_resize.median()),
                "soft_tokens_min": float(selected.soft_tokens.min()),
                "soft_tokens_median": float(selected.soft_tokens.median()),
                "soft_tokens_max": float(selected.soft_tokens.max()),
            })
            cuts = selected.render_height.quantile([0.25, 0.5, 0.75]).to_numpy()
            selected["height_quartile"] = np.searchsorted(cuts, selected.render_height, side="left") + 1
            for quartile, q in selected.groupby("height_quartile"):
                quartiles.append({
                    "language": language, "height_quartile": int(quartile), "n": len(q),
                    "source_height_median": float(q.render_height.median()),
                    "text_accuracy_pct": float(q.text_correct.mean() * 100),
                    "image_accuracy_pct": float(q.image_correct.mean() * 100),
                    "gap_pp": float((q.image_correct.mean() - q.text_correct.mean()) * 100),
                })
    runtime = {}
    for mode in ("text", "image"):
        metadata = [r["inference"] for r in rows if r["mode"] == mode]
        latency = [r["latency_seconds"] for r in metadata if "latency_seconds" in r]
        runtime[mode] = {
            "latency_sum_seconds": float(sum(latency)),
            "latency_median_seconds": float(np.median(latency)) if latency else None,
            "latency_p95_seconds": float(np.quantile(latency, 0.95)) if latency else None,
        }
    times = [datetime.fromisoformat(r["time"]) for r in rows if r.get("time")]
    if times:
        runtime["prediction_timestamp_span_seconds"] = (max(times) - min(times)).total_seconds()
    cross_language = {}
    # Only comparable IDs with all configured translations in both modalities.
    complete_ids = pairs.groupby("sample_id").language.nunique()
    aligned = pairs[pairs.sample_id.isin(complete_ids[complete_ids == len(languages)].index)]
    for mode in ("text", "image"):
        counts = aligned.groupby("sample_id")[f"{mode}_correct"].sum()
        cross_language[mode] = {str(n): int((counts == n).sum()) for n in range(len(languages) + 1)}
    original_files = ("manifest.json", "selected_ids.json", "predictions.jsonl", "metrics.json",
                      "rendering_metadata.json", "execution.json", "sessions.jsonl", "errors.jsonl")
    return {
        "schema_version": 1, "output_dir": root.as_posix(), "manifest_fingerprint": manifest["fingerprint"],
        "experiment": manifest["experiment"], "status": summary["status"],
        "audit": {
            "expected_predictions": summary["expected_predictions"], "predictions": len(rows),
            "paired_examples": len(pairs), "unique_questions": int(pairs.sample_id.nunique()),
            "unique_passages": int(pairs.passage_id.nunique()),
            "execution_error_attempts": summary["execution_error_attempts"],
            "scores_replayed": True, "duplicates": 0,
            "unknown_glyphs": int(pairs.render_unknown_glyphs.fillna(0).sum()),
            "original_artifact_sha256": {name: sha256((root / name).read_bytes()).hexdigest()
                                         for name in original_files if (root / name).exists()},
        },
        "uncertainty": {
            "method": "paired percentile passage-cluster bootstrap",
            "confidence_level": 0.95, "repetitions": repetitions, "seed": seed,
            "unit": "passage_id; all questions, translations, and modalities resampled together",
            "interpretation": "Corpus-sampling sensitivity; not repeated inference or model-seed uncertainty.",
        },
        "groups": groups, "answer_distributions": answer_distributions,
        "gold_answer_counts": {label: gold[label] for label in "ABCD"},
        "majority_answer_baseline_pct": max(gold.values()) / len(pairs) * 100,
        "invalid_by_language_mode": invalid, "resizing": resizing,
        "within_language_height_quartiles": quartiles,
        "cross_language_correct_histogram": cross_language, "runtime": runtime,
        "invalid_examples": [{key: r[key] for key in ("sample_id", "language", "mode", "raw_response", "truncated")}
                             for r in rows if not r["valid"]][:6],
    }


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def markdown_report(report):
    overall, *languages = report["groups"]
    audit, experiment = report["audit"], report["experiment"]
    ci = overall["gap_ci95_pp"]
    lines = [
        "# Belebele: full text versus rendered-image evaluation", "",
        f"**Text {overall['text_accuracy'] * 100:.2f}%; image {overall['image_accuracy'] * 100:.2f}%; "
        f"image minus text {overall['image_minus_text_pp']:.2f} percentage points.** "
        f"Passage-cluster bootstrap 95% interval for the gap: [{ci[0]:.2f}, {ci[1]:.2f}] points.", "",
        f"{audit['predictions']:,}/{audit['expected_predictions']:,} predictions ({report['status']}), "
        f"{audit['paired_examples']:,} paired examples, {audit['unique_questions']} aligned questions, "
        f"{audit['unique_passages']} distinct passages, {len(experiment['languages'])} languages. "
        f"{audit['execution_error_attempts']} execution error attempts; strict scores replayed for every record. "
        "All means use completed pairs; equal language coverage makes the overall mean equal to the macro language mean.", "",
        f"Model: `{experiment['model_id']}` at `{experiment['model_revision']}`; "
        f"{experiment['quantization']}. Render: {experiment['font_size']} px Noto, "
        f"{experiment['width']} px wide, {experiment['margin']} px margins, natural height; "
        f"image budget {experiment['max_soft_tokens']} soft tokens, answer budget {experiment['max_new_tokens']} tokens.", "",
        "## Per-language results", "",
        _table(["Language", "Pairs", "Text %", "Image %", "Gap (pp)", "Gap 95% interval (pp)", "Invalid image"], [
            [_name(g["language"]), g["paired_examples"], f"{100 * g['text_accuracy']:.2f}",
             f"{100 * g['image_accuracy']:.2f}", f"{g['image_minus_text_pp']:.2f}",
             f"[{g['gap_ci95_pp'][0]:.2f}, {g['gap_ci95_pp'][1]:.2f}]",
             next(x['invalid'] for x in report['invalid_by_language_mode']
                  if x['language'] == g['language'] and x['mode'] == 'image')]
            for g in languages]), "",
        "Intervals resample passages with replacement, keeping both questions from a passage, "
        f"all translations, and both modalities together ({report['uncertainty']['repetitions']:,} replicates, "
        f"seed {report['uncertainty']['seed']}). They describe sensitivity to the passage sample. "
        "They do not measure inference randomness, and are pointwise rather than simultaneous intervals. "
        "The evaluated 900-question corpus itself has no sampling error in its reported accuracy.", "",
        "## Paired outcomes", "",
        _table(["Outcome", "Count", "Share of pairs"], [
            [label, f"{overall[field]:,}", f"{overall[field] / overall['paired_examples'] * 100:.2f}%"]
            for label, field in (("Both correct", "both_correct"), ("Text correct, image wrong", "text_only_correct"),
                                 ("Text wrong, image correct", "image_only_correct"), ("Both wrong", "both_wrong"))]), "",
        f"Among text-correct pairs, the image condition retains "
        f"{overall['both_correct'] / (overall['both_correct'] + overall['text_only_correct']) * 100:.2f}% correct answers. "
        f"There are {overall['text_only_correct'] / max(1, overall['image_only_correct']):.2f} text-only successes "
        "for each image-only success.", "",
        "## Answer-choice behavior and invalid responses", "",
        "Ground-truth answer shares: " + "; ".join(
            f"{label} {count / audit['paired_examples'] * 100:.2f}%"
            for label, count in report["gold_answer_counts"].items()) + ". "
        f"Always choosing the majority label scores {report['majority_answer_baseline_pct']:.2f}%; "
        "uniform random guessing has expected accuracy 25%.", "",
        _table(["Language", "Most frequent image answer", "Share", "Image accuracy %"], [
            [_name(item['language']), max(item['counts'], key=item['counts'].get),
             f"{max(item['counts'].values()) / item['n'] * 100:.2f}%",
             f"{next(g['image_accuracy'] for g in languages if g['language'] == item['language']) * 100:.2f}"]
            for item in report['answer_distributions'] if item['mode'] == 'image']), "",
        f"{overall['invalid_responses']} responses are invalid, including {overall['truncated_responses']} "
        "truncated responses; these categories can overlap. Breakdown: " + "; ".join(
            f"{_name(x['language'])} {x['mode']}: {x['invalid']} invalid, {x['truncated']} truncated"
            for x in report['invalid_by_language_mode'] if x['invalid'] or x['truncated']) + ". "
        "Invalid/truncated responses stay incorrect; excluding them would change the estimand. "
        "The main loss is valid wrong answers, so answer-format failures alone do not explain the modality gap.", "",
        "## Rendering and processor diagnostics", "",
        _table(["Language", "Median source height (px)", "Median nominal size after resize (px)", "Actual soft-token range"], [
            [_name(r['language']), f"{r['source_height_median']:.0f}", f"{r['nominal_font_px_median']:.2f}",
             f"{r['soft_tokens_min']:.0f}–{r['soft_tokens_max']:.0f}"] for r in report['resizing']]), "",
        "Nominal size after resize is the configured font size multiplied by the smaller horizontal/vertical "
        "processor scale; it is a geometric proxy, not a measured glyph height. The 280-token setting is a cap, "
        "not a promise of exactly 280 actual tokens. Tall pages shrink more under the fixed budget. "
        "Within-language height-quartile plots are descriptive: document height also varies with content "
        "and question difficulty. Zero recorded missing glyphs does not establish legibility. "
        "This analysis does not rerender images or claim a fresh visual inspection.", "",
        "## Consistency and runtime", "",
        "On questions with all configured translations present: " + "; ".join(
            f"{mode}: {hist[str(len(experiment['languages']))]} questions correct in every language, "
            f"{hist['0']} correct in none"
            for mode, hist in report['cross_language_correct_histogram'].items()) + ".", "",
        _table(["Mode", "Summed generation time (min)", "Median latency (s)", "95th percentile latency (s)"], [
            [mode, f"{r['latency_sum_seconds'] / 60:.2f}", f"{r['latency_median_seconds']:.3f}",
             f"{r['latency_p95_seconds']:.3f}"] for mode, r in report['runtime'].items() if isinstance(r, dict)]), "",
        "Latency values come from per-prediction metadata and exclude model loading and preparation. "
        "They describe this hardware/software run, not a general model speed comparison.", "",
        "## Interpretation and next experiment", "",
        "This run shows a large end-to-end text/image gap under one quantized checkpoint and rendering/processor "
        "configuration. English has the strongest image accuracy; every tested language loses accuracy in image mode. "
        "The image answer distributions reveal pronounced option-position preferences, particularly in Khmer and Arabic. "
        "These are observed behaviors, not evidence of a specific OCR or script mechanism.", "",
        "A useful next controlled comparison is to rerun identical IDs with a larger image budget (560 or 1120 tokens) "
        "in a new output directory, checking GPU fit first. Compare paired changes within each language, retain "
        "invalid responses as incorrect, and inspect processor previews for the same examples. A second model "
        "would help separate checkpoint-specific behavior from benchmark/rendering behavior. "
        "No additional inference was launched by this analysis.", "",
        "## Reproduce and provenance", "",
        "```bash", f".venv/bin/python scripts/analyze_results.py --output-dir {report['output_dir']}", "```", "",
        "Open `Belebele_analysis.ipynb` for plots. It needs no model, dataset download, renderer, or GPU. "
        "Plots live only in notebook memory; clear outputs before saving. Original run artifacts remain unchanged. "
        "The derived `analysis.json` contains original-file SHA-256 hashes, all tables, bootstrap settings, and run configuration.", "",
        f"Manifest fingerprint: `{report['manifest_fingerprint']}`. "
        f"Dataset: `{experiment['dataset_id']}@{experiment['dataset_revision']}`. "
        "See README for dataset, model, font, and renderer attribution.", "",
    ]
    return "\n".join(lines)


def plot_overview(report):
    import matplotlib.pyplot as plt

    groups = report["groups"][1:]
    overall = report["groups"][0]
    y = np.arange(len(groups))
    figure, axes = plt.subplots(1, 3, figsize=(18, 7.5), gridspec_kw={"width_ratios": [1.5, 1.1, 1]}, layout="constrained")
    for mode, offset, color, label in (("text", -0.17, TEXT_COLOR, "Text"), ("image", 0.17, IMAGE_COLOR, "Image")):
        accuracy = np.array([g[f"{mode}_accuracy"] * 100 for g in groups])
        axes[0].barh(y + offset, accuracy, height=0.30, color=color, label=label)
        for yy, value in zip(y + offset, accuracy):
            axes[0].text(value + 0.8, yy, f"{value:.1f}", va="center", fontsize=9)
    axes[0].axvline(25, color="#7b8794", linestyle=":", linewidth=1, label="25% guessing")
    axes[0].set(yticks=y, yticklabels=[_name(g["language"]) for g in groups], xlim=(0, 100),
                xlabel="Accuracy (%)", title=f"Same {report['audit']['unique_questions']} questions per language")
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.10), ncol=3, frameon=False, fontsize=9)
    gaps = np.array([g["image_minus_text_pp"] for g in groups])
    bounds = np.array([g["gap_ci95_pp"] for g in groups])
    axes[1].errorbar(gaps, y, xerr=np.vstack([gaps - bounds[:, 0], bounds[:, 1] - gaps]),
                     fmt="o", color=IMAGE_COLOR, capsize=3)
    axes[1].axvline(0, color="#7b8794", linestyle=":")
    axes[1].set(yticks=y, yticklabels=[], ylim=axes[0].get_ylim(), xlim=(min(-56, bounds.min() - 3), 2),
                xlabel="Image minus text (percentage points)", title="Paired gap · 95% passage-bootstrap interval")
    transitions = np.array([[overall["both_correct"], overall["text_only_correct"]],
                            [overall["image_only_correct"], overall["both_wrong"]]])
    axes[2].imshow(transitions / transitions.sum() * 100, cmap="Blues", vmin=0, vmax=50)
    axes[2].set(xticks=[0, 1], xticklabels=["Image\ncorrect", "Image\nwrong"],
                yticks=[0, 1], yticklabels=["Text\ncorrect", "Text\nwrong"],
                title=f"All {report['audit']['paired_examples']:,} paired outcomes")
    for yy in range(2):
        for xx in range(2):
            count = transitions[yy, xx]
            axes[2].text(xx, yy, f"{count:,}\n{count / transitions.sum():.1%}", ha="center", va="center",
                         fontsize=13, color="white" if count / transitions.sum() > .35 else "#172b4d")
    figure.suptitle(f"Belebele · Gemma 4 E4B NF4 · Text {100 * overall['text_accuracy']:.2f}% → "
                   f"Image {100 * overall['image_accuracy']:.2f}% ({overall['image_minus_text_pp']:.2f} pp)", fontsize=17)
    return figure


def plot_answer_bias(report):
    import matplotlib.pyplot as plt

    labels = ("A", "B", "C", "D", "Invalid")
    figure, axes = plt.subplots(1, 2, figsize=(13, 7.5), layout="constrained")
    for axis, mode in zip(axes, ("text", "image")):
        rows = [x for x in report["answer_distributions"] if x["mode"] == mode]
        matrix = np.array([[x["counts"][label] / x["n"] * 100 for label in labels] for x in rows])
        im = axis.imshow(matrix, cmap="YlOrBr", vmin=0, vmax=90, aspect="auto")
        axis.set(xticks=range(5), xticklabels=labels, yticks=range(len(rows)),
                 yticklabels=[_name(x["language"]) for x in rows], title=mode.capitalize(), xlabel="Model's answer")
        for yy in range(len(rows)):
            for xx in range(5):
                axis.text(xx, yy, f"{matrix[yy, xx]:.1f}", ha="center", va="center",
                          color="white" if matrix[yy, xx] > 55 else "#382916", fontsize=10)
    figure.colorbar(im, ax=axes, label="Share of responses (%)", shrink=.8)
    gold = report["gold_answer_counts"]
    total = sum(gold.values())
    figure.suptitle("Image answers show strong answer-choice preferences\nGold shares: " +
                   " · ".join(f"{label} {gold[label] / total:.1%}" for label in "ABCD"), fontsize=15)
    return figure


def plot_diagnostics(report):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(18, 6.5), layout="constrained")
    rows = report["resizing"]
    if rows:
        values = [r["nominal_font_px_median"] for r in rows]
        axes[0].barh(np.arange(len(rows)), values, color="#5e9c97")
        axes[0].set(yticks=np.arange(len(rows)), yticklabels=[_name(r["language"]) for r in rows],
                    xlabel="24 px × processor resize scale (px)", xlim=(0, 27), title="Median nominal size after resizing")
        axes[0].axvline(report["experiment"]["font_size"], color="#555", linestyle=":")
        axes[0].invert_yaxis()
        for y, value in enumerate(values):
            axes[0].text(value - .4, y, f"{value:.1f}", va="center", ha="right", color="white", fontsize=9)
    else:
        axes[0].text(.5, .5, "Processor resizing is provider-managed", ha="center")
        axes[0].set_axis_off()
    quartiles = report["within_language_height_quartiles"]
    languages = report["experiment"]["languages"]
    matrix = np.full((len(languages), 4), np.nan)
    for r in quartiles:
        matrix[languages.index(r["language"]), r["height_quartile"] - 1] = r["image_accuracy_pct"]
    im = axes[1].imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=80)
    axes[1].set(yticks=range(len(languages)), yticklabels=[_name(lang) for lang in languages],
                xticks=range(4), xticklabels=["Q1\nshorter", "Q2", "Q3", "Q4\ntaller"],
                title="Image accuracy (%) by page height", xlabel="Quartiles defined within each language")
    for yy in range(len(languages)):
        for xx in range(4):
            if np.isfinite(matrix[yy, xx]):
                axes[1].text(xx, yy, f"{matrix[yy, xx]:.1f}", ha="center", va="center",
                             color="white" if matrix[yy, xx] > 50 else "#172b4d", fontsize=9)
    for mode, offset, color in (("text", -.18, TEXT_COLOR), ("image", .18, IMAGE_COLOR)):
        hist = report["cross_language_correct_histogram"][mode]
        axes[2].bar(np.arange(len(languages) + 1) + offset, [hist[str(n)] for n in range(len(languages) + 1)],
                    width=.35, color=color, label=mode.capitalize())
    axes[2].set(xticks=range(len(languages) + 1), xlabel="Languages answered correctly for the same question",
                ylabel="Number of aligned questions", title="Cross-language consistency")
    axes[2].legend()
    figure.suptitle("Diagnostics · resizing and height associations are descriptive, not causal", fontsize=15)
    return figure
