"""Small text-only artifacts; lock writers and tolerate an interrupted final line."""

from dataclasses import asdict
from importlib import metadata
import json
import os
from pathlib import Path
import platform

from .data import digest


def environment():
    versions = {}
    for name in ("rendering-eval", "transformers", "torch", "torchvision", "accelerate", "bitsandbytes",
                 "pixel-renderer", "huggingface-hub", "numpy", "Pillow", "PyGObject", "pycairo"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def implementation_digest():
    return digest({p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))})


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path, *, repair=False):
    path = Path(path)
    if not path.exists():
        return []
    content = path.read_bytes()
    rows, offset = [], 0
    lines = content.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Corrupt complete record in {path}, line {index + 1}")
            if repair:
                with path.open("r+b") as handle:
                    handle.truncate(offset)
            break
        offset += len(line)
    else:
        if content and not content.endswith(b"\n") and repair:
            with path.open("ab") as handle:
                handle.write(b"\n")
    return rows


def ensure_manifest(config, examples, *, allow_completed=False):
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    selection = [
        {"sample_id": e.sample_id, "passage_id": e.passage_id, "language": e.language,
         "content_sha256": digest(asdict(e))} for e in examples
    ]
    contract = {
        "schema_version": 1, "experiment": config.experiment(),
        "selection": selection, "implementation_sha256": implementation_digest(),
    }
    manifest = {**contract, "fingerprint": digest(contract), "environment": environment(),
                "expected_predictions": len(examples) * 2}
    path = root / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["fingerprint"] != manifest["fingerprint"]:
            # A complete historical run can be returned unchanged after a code update.
            # No new inference is mixed into it and its original provenance is retained.
            if allow_completed and previous["experiment"] == manifest["experiment"] and previous["selection"] == selection:
                rows = read_jsonl(root / "predictions.jsonl")
                indexed = prediction_index(rows)
                expected = {(e.sample_id, e.language, mode) for e in examples for mode in ("text", "image")}
                if set(indexed) == expected and all(r.get("manifest_fingerprint") == previous["fingerprint"] for r in rows):
                    return previous
            raise ValueError("Incompatible output directory: config, examples, or code changed. Use a new output_dir.")
        return previous
    if (root / "predictions.jsonl").exists():
        raise ValueError("Predictions exist without a manifest; use a new output directory")
    atomic_json(path, manifest)
    atomic_json(root / "selected_ids.json", list(dict.fromkeys(e.sample_id for e in examples)))
    return manifest


def prediction_index(rows):
    indexed = {}
    for row in rows:
        key = (row["sample_id"], row["language"], row["mode"])
        if key in indexed:
            raise ValueError(f"Duplicate prediction: {key}")
        indexed[key] = row
    return indexed
