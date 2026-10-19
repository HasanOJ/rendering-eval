from datetime import datetime, timezone
from pathlib import Path
import json

from filelock import FileLock

from .artifacts import (
    append_jsonl, atomic_json, ensure_manifest, environment, prediction_index, read_jsonl,
)
from .data import load_examples
from .scoring import score_answer


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(config, *, examples=None, renderer=None):
    """Prepare text-only artifacts and validate all selected renderings in memory."""
    from .rendering import DocumentRenderer

    examples = load_examples(config) if examples is None else examples
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".run.lock"), timeout=0):
        ensure_manifest(config, examples)
        renderer = renderer if renderer is not None else DocumentRenderer(config)
        audits = []
        for example in examples:
            image, metadata = renderer.render(example)
            image.close()
            audits.append({"sample_id": example.sample_id, "language": example.language, **metadata})
        atomic_json(root / "rendering_metadata.json", {"fonts": renderer.font_metadata, "examples": audits})
    return examples


def run(config, *, examples=None, renderer=None, backend=None, progress=print):
    """Load only when work is missing; release internally created models on return or error."""
    owned_backends = []
    try:
        return _run(config, examples=examples, renderer=renderer, backend=backend,
                    progress=progress, owned_backends=owned_backends)
    finally:
        for owned in owned_backends:
            close = getattr(owned, "close", None)
            if close is not None:
                close()


def _run(config, *, examples=None, renderer=None, backend=None, progress=print, owned_backends=None):
    """Run or resume. First run tests the longest text/image inputs before the remainder."""
    from .backends import GemmaBackend
    from .rendering import DocumentRenderer

    examples = load_examples(config) if examples is None else examples
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".run.lock"), timeout=0):
        manifest = ensure_manifest(config, examples, allow_completed=True)
        for journal in ("errors.jsonl", "sessions.jsonl"):
            read_jsonl(root / journal, repair=True)
        existing = prediction_index(read_jsonl(root / "predictions.jsonl", repair=True))
        expected = {(e.sample_id, e.language, mode) for e in examples for mode in ("text", "image")}
        if set(existing) - expected:
            raise ValueError("Unexpected predictions in output directory")
        if any(row.get("manifest_fingerprint") != manifest["fingerprint"] for row in existing.values()):
            raise ValueError("Prediction fingerprint does not match the manifest")
        if len(existing) == len(expected):
            progress(f"Already complete: {len(existing)}/{len(expected)} predictions")
            return list(existing.values())
        try:
            renderer = renderer if renderer is not None else DocumentRenderer(config)
            audits = []
            for example in examples:
                image, metadata = renderer.render(example)
                image.close()
                audits.append({"sample_id": example.sample_id, "language": example.language, **metadata})
            atomic_json(root / "rendering_metadata.json", {"fonts": renderer.font_metadata, "examples": audits})
            if backend is None:
                if config.backend == "openrouter":
                    from .openrouter import OpenRouterBackend
                    backend = OpenRouterBackend(config)
                else:
                    backend = GemmaBackend(config)
                owned_backends.append(backend)
            details = backend.describe()
            execution = {
                "environment": environment(), "backend": details, "fonts": renderer.font_metadata,
            }
            # Hardware may differ between hosts; numerical implementation must agree.
            compatibility = {
                "packages": execution["environment"]["packages"], "fonts": renderer.font_metadata,
                "backend": {k: v for k, v in details.items() if k not in (
                    "gpu", "initial_memory", "model_footprint_bytes", "cuda_version",
                )},
            }
            execution_path = root / "execution.json"
            if execution_path.exists():
                previous = json.loads(execution_path.read_text())
                if previous["compatibility"] != compatibility:
                    raise ValueError("Inference environment differs from existing run; use a new output_dir")
            else:
                atomic_json(execution_path, {**execution, "compatibility": compatibility})
            append_jsonl(root / "sessions.jsonl", {"started": now(), **execution})
        except Exception as error:
            append_jsonl(root / "errors.jsonl", {"time": now(), "stage": "preflight", "type": type(error).__name__, "message": str(error)})
            raise

        schedule = [(e, mode) for e in examples for mode in ("text", "image")]
        if hasattr(backend, "processor"):
            longest_text = max(examples, key=lambda e: len(backend.processor.tokenizer.encode(e.task_text())))
        else:
            longest_text = max(examples, key=lambda e: len(e.task_text()))
        tallest = max(audits, key=lambda a: a["height"])
        longest_image = next(e for e in examples if (e.sample_id, e.language) == (tallest["sample_id"], tallest["language"]))
        preflight = [(longest_text, "text"), (longest_image, "image")]
        schedule = preflight + [item for item in schedule if item not in preflight]
        for example, mode in schedule:
            key = (example.sample_id, example.language, mode)
            if key in existing:
                continue
            image, render_meta = None, {}
            try:
                if mode == "image":
                    image, render_meta = renderer.render(example)
                generated = backend.generate(example, mode, image)
                row = {
                    "sample_id": example.sample_id, "passage_id": example.passage_id,
                    "language": example.language, "mode": mode, "answer": example.answer,
                    "raw_response": generated.raw_response,
                    **score_answer(generated.raw_response, example.answer, truncated=generated.truncated),
                    "rendering": render_meta, "inference": generated.metadata,
                    "manifest_fingerprint": manifest["fingerprint"], "time": now(),
                }
                append_jsonl(root / "predictions.jsonl", row)
                existing[key] = row
                progress(f"{len(existing)}/{len(expected)} {example.language} {mode}: {row['predicted'] or 'invalid'}")
            except Exception as error:
                append_jsonl(root / "errors.jsonl", {
                    "time": now(), "stage": "generation", "sample_id": example.sample_id,
                    "language": example.language, "mode": mode,
                    "type": type(error).__name__, "message": str(error),
                })
                # Stop on execution failure rather than turn it into an incorrect answer.
                raise
            finally:
                if image is not None:
                    image.close()
        return list(existing.values())
