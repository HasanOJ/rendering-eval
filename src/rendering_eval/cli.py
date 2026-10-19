import argparse
from dataclasses import replace
import json

from .config import Config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paired Belebele text/image evaluation (no image files)")
    parser.add_argument("command", choices=["prepare", "preview", "run", "analyze"])
    parser.add_argument("--config", help="YAML config; defaults match configs/smoke.yaml")
    parser.add_argument("--output-dir", help="Override the results directory")
    parser.add_argument("--n-ids", type=int, help="Aligned IDs: 2 for smoke, 900 for the full evaluation")
    parser.add_argument("--languages", nargs="+", help="Language config codes; defaults to the selected 12")
    parser.add_argument("--backend", choices=["local", "openrouter"])
    parser.add_argument("--model-id", help="Hugging Face model ID locally, or OpenRouter model slug remotely")
    parser.add_argument("--model-revision", help="Pinned Hugging Face commit for a local model")
    parser.add_argument("--quantization", choices=["4bit", "8bit"])
    parser.add_argument("--quantize-on-load", dest="prequantized", action="store_false", default=None,
                        help="Explicitly allow conversion of a BF16 checkpoint instead of direct quantized loading")
    parser.add_argument("--provider", dest="openrouter_provider", help="Optional OpenRouter provider slug")
    parser.add_argument("--free-only", dest="openrouter_free_only", action="store_true", default=None)
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args(argv)
    config = Config.from_yaml(args.config) if args.config else Config()
    overrides = {name: getattr(args, name) for name in (
        "output_dir", "n_ids", "languages", "backend", "model_id", "model_revision", "quantization",
        "openrouter_provider", "openrouter_free_only", "max_new_tokens", "prequantized",
    ) if getattr(args, name) is not None}
    config = replace(config, **overrides)
    if args.command == "analyze":
        from .analysis import analyze
        print(json.dumps(analyze(config.output_dir), indent=2))
    elif args.command == "run":
        from .runner import run
        from .analysis import analyze
        run(config, progress=lambda message: print(message, flush=True))
        print(json.dumps(analyze(config.output_dir), indent=2))
    else:
        from .runner import prepare
        examples = prepare(config)
        print(f"Prepared {len(examples)} examples / {len(examples) * 2} predictions. No images saved.")
        if args.command == "preview":
            if config.backend == "openrouter":
                print("OpenRouter preprocessing is provider-managed. Open Belebele.ipynb to view original renderings.")
                return
            # CLI validates actual processor inputs; visual display belongs to the notebook.
            from .backends import encode, image_input_metadata, load_processor
            from .rendering import DocumentRenderer
            renderer, processor = DocumentRenderer(config), load_processor(config)
            for lang in config.languages:
                example = next(e for e in examples if e.language == lang)
                image, info = renderer.render(example)
                try:
                    inputs = encode(processor, config, example, "image", image)
                    print(json.dumps({"language": lang, **info, **image_input_metadata(inputs, processor)}))
                finally:
                    image.close()
            print("Open Belebele.ipynb to view the images; this command writes no images.")


if __name__ == "__main__":
    main()
