"""Generate text-only analysis artifacts from an existing run; never infer/render."""

import argparse
from pathlib import Path

from rendering_eval.artifacts import atomic_json
from rendering_eval.reporting import build_report, markdown_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/belebele-full-4bit")
    parser.add_argument("--report", help="Markdown path; defaults to reports/<run-directory-name>.md")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.output_dir)
    report = build_report(root, repetitions=args.bootstrap, seed=args.seed)
    destination = Path(args.report) if args.report else Path("reports") / f"{root.name}.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown_report(report), encoding="utf-8")
    atomic_json(root / "analysis.json", report)
    print(f"Validated {report['audit']['predictions']:,} predictions; no inference or rendering.")
    print(f"Report: {destination}; derived metrics: {root / 'analysis.json'}")
    print("Open Belebele_analysis.ipynb for in-memory plots.")


if __name__ == "__main__":
    main()
