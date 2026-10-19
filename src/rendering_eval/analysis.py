"""Paired summaries and notebook-only plots. No plot/image files are written."""

from pathlib import Path
import json

from .artifacts import atomic_json, prediction_index, read_jsonl


def analyze(output_dir, *, save=True):
    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    rows = read_jsonl(root / "predictions.jsonl")
    indexed = prediction_index(rows)
    expected = {(s["sample_id"], s["language"], mode)
                for s in manifest["selection"] for mode in ("text", "image")}
    if set(indexed) - expected or any(r["manifest_fingerprint"] != manifest["fingerprint"] for r in rows):
        raise ValueError("Predictions do not match the manifest")
    languages = manifest["experiment"]["languages"]
    groups = []
    for lang in ["ALL", *languages]:
        selected = [r for r in rows if lang == "ALL" or r["language"] == lang]
        pair_keys = sorted({(r["sample_id"], r["language"]) for r in selected})
        pairs = [(indexed[(*key, "text")], indexed[(*key, "image")])
                 for key in pair_keys if (*key, "text") in indexed and (*key, "image") in indexed]
        text_correct = sum(a["correct"] for a, _ in pairs)
        image_correct = sum(b["correct"] for _, b in pairs)
        n = len(pairs)
        groups.append({
            "language": lang, "completed_predictions": len(selected), "paired_examples": n,
            "text_accuracy": text_correct / n if n else None,
            "image_accuracy": image_correct / n if n else None,
            "image_minus_text_pp": 100 * (image_correct - text_correct) / n if n else None,
            "both_correct": sum(a["correct"] and b["correct"] for a, b in pairs),
            "text_only_correct": sum(a["correct"] and not b["correct"] for a, b in pairs),
            "image_only_correct": sum(not a["correct"] and b["correct"] for a, b in pairs),
            "both_wrong": sum(not a["correct"] and not b["correct"] for a, b in pairs),
            "invalid_responses": sum(not r["valid"] for r in selected),
            "truncated_responses": sum(r["truncated"] for r in selected),
        })
    summary = {
        "label": f"{manifest['experiment']['n_ids']}-ID paired evaluation; modality gaps do not establish causal script effects.",
        "status": "complete" if len(rows) == len(expected) else ("partial" if rows else "pending"),
        "expected_predictions": len(expected), "completed_predictions": len(rows),
        "missing_predictions": len(expected) - len(rows),
        "execution_error_attempts": len(read_jsonl(root / "errors.jsonl")),
        "groups": groups,
    }
    if save:
        from filelock import FileLock
        with FileLock(str(root / ".run.lock"), timeout=0):
            atomic_json(root / "metrics.json", summary)
    return summary


def plot_summary(summary):
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame(summary["groups"]).set_index("language")
    if frame.loc["ALL", "paired_examples"] == 0:
        print("No completed text/image pairs yet; accuracy is pending.")
        return None
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), layout="constrained")
    languages = frame.drop(index="ALL")
    (languages[["text_accuracy", "image_accuracy"]].rename(
        columns={"text_accuracy": "Text", "image_accuracy": "Image"}) * 100).plot.bar(ax=axes[0], rot=45)
    axes[0].set(ylabel="Accuracy (%) on completed pairs", xlabel="", ylim=(0, 100))
    axes[0].axhline(25, color="gray", linestyle=":", label="Chance")
    axes[0].legend()
    total = frame.loc["ALL"]
    transitions = [[total["both_correct"], total["text_only_correct"]],
                   [total["image_only_correct"], total["both_wrong"]]]
    axes[1].imshow(transitions, cmap="Blues")
    axes[1].set(xticks=[0, 1], xticklabels=["Image correct", "Image wrong"],
                yticks=[0, 1], yticklabels=["Text correct", "Text wrong"], title="Paired outcomes")
    for y in range(2):
        for x in range(2):
            axes[1].text(x, y, str(int(transitions[y][x])), ha="center", va="center", color="black")
    figure.suptitle(f"Belebele text/image evaluation — {int(total['paired_examples'])} pairs ({summary['status']})")
    return figure
