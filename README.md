# Belebele: text versus rendered image

Compare the same Belebele passage, question, and four choices delivered as text or as a rendered image. The default smoke test is **2 aligned IDs × 12 languages × 2 modalities = 48 predictions**, using local **Gemma 4 E4B in 4-bit NF4**. [Belebele.ipynb](Belebele.ipynb) provides in-memory previews and paired plots. The CLI supports local and OpenRouter evaluation with resumable JSON results.

## Install in WSL or on a GPU cloud host

Use Python 3.12 and an NVIDIA driver compatible with your PyTorch build. After cloning this repository on the target machine:

```bash
sudo apt-get update
sudo apt-get install -y libcairo2-dev libgirepository1.0-dev gir1.2-pango-1.0 pkg-config python3-dev python3-venv fontconfig
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[notebook,pango,test]'
```

Choose `.venv/bin/python` as the notebook kernel in VS Code/Jupyter. If system `python3-gi`, `python3-gi-cairo`, and `python3-cairo` already match your interpreter, you can create a venv with `--system-site-packages` and install `.[notebook,test]` instead. `constraints-wsl.txt` records the tested package versions; use a driver-compatible PyTorch build on other hosts.

No notebook-specific environment bootstrap is required. The same checkout, configs, and commands work locally and on a cloud GPU. Set the output directory to persistent storage on the cloud host. Model, dataset, and font downloads use their normal library caches; benchmark images do not.

## Smoke test and full local evaluation

```bash
# Two aligned IDs in all 12 languages: 48 predictions.
.venv/bin/rendering-eval run --config configs/smoke.yaml

# Full evaluation: all 900 IDs in the selected 12 languages, 21,600 predictions.
.venv/bin/rendering-eval run --config configs/full.yaml

# Equivalent explicit command.
.venv/bin/rendering-eval run --config configs/smoke.yaml --n-ids 900 --output-dir results/belebele-full-4bit

# Summarize existing results without loading the model.
.venv/bin/rendering-eval analyze --output-dir results/belebele-full-4bit
```

For the completed full run, open [Belebele_analysis.ipynb](Belebele_analysis.ipynb) and **Run All** with the project `.venv` kernel. It reads existing results without model loading, inference, downloads, or rendering. It includes per-language accuracy and paired gap intervals, paired outcomes, answer-choice distributions, invalid responses, processor resizing, page-height comparisons, and multilingual consistency. Bootstrap intervals resample passages, carrying their questions and translations together. Charts stay in notebook output; clear outputs before saving a clean copy. The original evaluation notebook remains separate.

The [full-run report](reports/belebele-full-4bit.md) records the findings. To regenerate its text-only report and derived metrics, use `.venv/bin/python scripts/analyze_results.py --output-dir results/belebele-full-4bit`; this preserves the original evaluation artifacts.

For a closer investigation, open [Belebele_errors.ipynb](Belebele_errors.ipynb). It shows every invalid raw response, label-conditional accuracy/confusion plots, eight annotated cases, and a browser for any saved mistake with its aligned English reference. Source text is verified against manifest hashes and read strictly from the existing Hub cache. Optional previews reconstruct one processor input on CPU; no model weights or new predictions are loaded. The notebook keeps the original scores and result files intact and saves no image files.

Repeat the same `run` command to resume. Successful predictions are preserved, including invalid model answers; missing work and execution failures are retried. Changing sample count, backend, model, quantization, or rendering requires a **new output directory**. Code changes require a fresh directory for incomplete runs; complete historical runs are returned unchanged with their original provenance and no new inference. The earlier 8-bit run remains in `results/belebele-smoke`; the new 4-bit run uses `results/belebele-smoke-4bit`.

The prequantized checkpoint is a third-party Unsloth conversion of Gemma. Its files total about 11 GB because embeddings and selected modules remain in BF16. The complete checkpoint is needed for inference; only loading is deferred until a run has missing predictions.

The 12 variants are `eng_Latn`, `rus_Cyrl`, `arb_Arab`, `hin_Deva`, `ben_Beng`, `tel_Telu`, `tha_Thai`, `khm_Khmr`, `mya_Mymr`, `zho_Hans`, `jpn_Jpan`, and `kor_Hang`. Full evaluation means 900 IDs in this subset, not all 122 Belebele variants. Override `--languages eng_Latn arb_Arab` or `--n-ids 2` for a smaller run.

Local inference directly loads a pinned prequantized bitsandbytes NF4 checkpoint; it does not download or convert the BF16 source checkpoint. It uses the saved double-quantization state and skip list, batch size 1, thinking disabled, greedy decoding, and a 16-token answer budget. Nonquantized modules use BF16 on supported GPUs and FP16 otherwise. Gemma E4B has large embedding tables that remain unquantized, so total GPU storage is considerably higher than 4-bit linear-weight storage alone. Reducing the number of test IDs reduces runtime, not model memory. There is no automatic precision change or CPU offloading. Model loading is deferred until predictions are missing, and internally created models are released after completion or failure. This checkpoint requires a BF16-capable GPU. Conversion of another source checkpoint is available only through the explicit `--quantize-on-load` option with matching model ID/revision and a new output directory.

## OpenRouter

Set `OPENROUTER_API_KEY` through your shell or secret manager. The program reads it from the environment and never includes it in configurations, manifests, or prediction records. Do not commit the key; `.env` files are ignored. The program does not automatically load `.env` files.

```bash
# One English question, text and image: two free requests.
.venv/bin/rendering-eval run --config configs/openrouter-free.yaml

# Paid Qwen full evaluation, launched explicitly when ready.
.venv/bin/rendering-eval run --config configs/openrouter-full.yaml

# Two-ID paid Qwen pilot before a full run.
.venv/bin/rendering-eval run --config configs/openrouter-full.yaml --n-ids 2 --output-dir results/qwen-pilot

# Select another vision model and optionally pin a provider.
.venv/bin/rendering-eval run --backend openrouter --model-id qwen/qwen3.6-27b --n-ids 2 --output-dir results/qwen-custom
```

The free config selects the tested `nvidia/nemotron-3.5-super-vl:free` vision endpoint. A different currently available vision model can be selected with `--model-id <slug>:free --free-only --output-dir <new-directory>`. Free variants still require an API key and can be rate-limited. Free-only mode checks the public catalog pricing and sends a zero-price provider ceiling. There is no fallback to a paid model. See the [OpenRouter free-variant documentation](https://openrouter.ai/docs/guides/routing/model-variants/free).

OpenRouter sends the rendered image as a base64 PNG encoded in memory. It uses temperature 0, requests reasoning disabled where supported, and records served model/provider, request ID, token usage, finish reason, latency, and reported cost. Requests use one model with provider fallback disabled. An optional `--provider <provider-slug>` pins the provider. HTTP or network errors stop the run and leave the request pending; repeat the same command to resume. No automatic request retries risk duplicating a paid call.

Remote image resizing, quantization, and model weight revisions are provider-managed. The local Gemma 280-token processor setting does not apply to OpenRouter, and the notebook shows only original renderings for remote models. Provider previews are not fabricated. The API payload follows the [OpenRouter chat reference](https://openrouter.ai/docs/api_reference/overview).

## Preparation and visualization

```bash
.venv/bin/rendering-eval prepare --config configs/smoke.yaml
.venv/bin/rendering-eval preview --config configs/smoke.yaml
```

`prepare` validates question identity and correct-answer alignment across all source rows, selects the seeded IDs, and checks selected renderings for missing glyphs and clipping. `preview` also checks local processor inputs. Open `Belebele.ipynb` to see original images, actual local processor-patch reconstructions, paired accuracy, and answer transitions. Set `CONFIG_PATH` in the notebook to inspect another run; notebook inference stays disabled by default with `RUN_EVALUATION=False`.

The text and image conditions use the exact same task string and answer order, plus the same short English instruction. Images contain the entire passage, question, and A–D choices. There is no duplicated task text or answer key in the image prompt.

Rendering uses a readable default: 24 px, black on white, 1,024 px width, 24 px margins, and natural height. Script-appropriate Noto Sans fonts and Pango handle shaping, wrapping, and directionality. Font/size sweeps remain deferred. Local Gemma uses a fixed budget of 280 soft image tokens; long pages may shrink.

## Artifacts and reproducibility

**No benchmark image or plot files are saved or cached.** Each rendering is created in memory, used, and released. OpenRouter's PNG buffer also stays in memory. Only JSON/JSONL results and a lock file are written:

- `manifest.json`, `selected_ids.json`: configuration, source hashes, selected IDs, content hashes, and expected prediction count.
- `rendering_metadata.json`: dimensions, actual fonts, font hashes, and glyph/clipping checks.
- `execution.json`, `sessions.jsonl`: runtime, local quantization/device placement or remote catalog/provider metadata.
- `predictions.jsonl`, `errors.jsonl`, `metrics.json`: responses, scores, inference metadata, execution failures, and paired summaries.

Notebook frontends can embed visible outputs during save/autosave. Clear displays with the notebook's final cell before saving, or sanitize a saved notebook:

```bash
python3 scripts/clean_notebook.py Belebele.ipynb
```

The dataset is pinned to `facebook/belebele@7899cdfa4e1e0d733fd77c848e2c273cb1d32be2`, and the local prequantized model to `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit@6f10b2aa522ba471321d8246fd71a3dfa8c5b9a2`. The renderer is installed as a package dependency at `sign/pixel-renderer@9706296afca3dc156d76ff82450fff0fefc6f3d7`; a local document adapter uses its Pango/Cairo infrastructure without modifying upstream. Fonts use Google Fonts revision `601944d2ee97e76d4c456dc3d790db358c0c6ec5`.

The Hub export omits passage-level `split`, so the loader verifies uniqueness of `(link, question_number)`; it includes `split` if present. Languages must have identical IDs and answer labels. Selection uses seeded hash ordering rather than positional joins. Resume checks source/data/configuration hashes and execution compatibility. The loader expands saved module-subtree exclusions to match the current Transformers API and verifies that every quantized layer has quantization state before inference. Stored weight values and intended layer precisions are preserved.

## Scoring and validation

A response must contain one A–D letter, case-insensitive, optionally parenthesized or followed by a period. Invalid or truncated responses count as incorrect. Execution failures remain pending. Accuracy uses completed text/image pairs; pending results have no accuracy rather than zero. The notebook shows both-correct, text-only-correct, image-only-correct, and both-wrong transitions.

Two IDs are a smoke test, not enough for reliable language rankings. Modality differences include rendering, preprocessing, visual encoding, and reasoning; they do not establish OCR ability or causal script effects.

```bash
.venv/bin/python -m pytest -q
RUN_INTEGRATION=1 .venv/bin/python -m pytest tests/test_integration.py -q
```

The integration test downloads pinned data/fonts/processor and checks all 24 selected examples without loading model weights. OpenRouter unit tests use a fake transport; live API testing is reported separately in `VALIDATION.md`.

## Attribution

- [Belebele](https://huggingface.co/datasets/facebook/belebele), Bandarkar et al., *The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants*, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- [sign/pixel-renderer](https://github.com/sign/pixel-renderer), Ilker Kesen and Amit Moryossef, MIT.
- [Noto / Google Fonts](https://github.com/google/fonts), SIL Open Font License.
- [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it), Google DeepMind, with the [Unsloth prequantized conversion](https://huggingface.co/unsloth/gemma-4-E4B-it-unsloth-bnb-4bit); consult the pinned model cards for terms and conversion details.
