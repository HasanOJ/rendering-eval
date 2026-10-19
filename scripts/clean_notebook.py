"""Remove all outputs, attachments, and saved widget state before sharing/saving a clean copy."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("notebook", nargs="?", default="Belebele.ipynb")
args = parser.parse_args()
path = Path(args.notebook)
notebook = json.loads(path.read_text())
notebook["metadata"].pop("widgets", None)
for cell in notebook["cells"]:
    cell.pop("attachments", None)
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
