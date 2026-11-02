# Belebele: text versus rendered image

Compare the same Belebele passage, question, and four choices delivered as text or as a rendered image. The default smoke test is **2 aligned IDs × 12 languages × 2 modalities = 48 predictions**, using local **Gemma 4 E4B in 8-bit INT8**. **Gemma 4 12B in 4-bit NF4** is also configured. [Belebele_models.ipynb](Belebele_models.ipynb) provides model selection, in-memory previews, and paired plots with inference disabled by default. The CLI supports local and OpenRouter evaluation with resumable JSON results.

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

| Model | Loading | Smoke config | Full config |
| --- | --- | --- | --- |
| Gemma 4 E4B, 8-bit (default) | Official BF16 source → bitsandbytes INT8 on load | `configs/smoke.yaml` | `configs/full.yaml` |
| Gemma 4 12B, 4-bit | Official BF16 source → bitsandbytes NF4 with double quantization on load | `configs/smoke-12b-4bit.yaml` | `configs/full-12b-4bit.yaml` |
| Historical E4B, 4-bit baseline | Pinned Unsloth prequantized NF4 | `configs/smoke-e4b-4bit.yaml` | `configs/full-e4b-4bit.yaml` |

The two new variants use separate `results/belebele-e4b-8bit-{smoke,full}` and `results/belebele-12b-4bit-{smoke,full}` directories. The E4B INT8 full run completed all 21,600 predictions on the RTX 4070 with zero execution errors. The 12B variant has not yet been validated with real inference or for GPU fit. On the 12 GB RTX 4070, begin with a smoke run: unquantized embeddings, other retained modules, activations, and the KV cache also consume memory. The loader uses only GPU 0 and does not automatically offload to CPU or change precision.

Both new variants download the official BF16 source weights into the normal Hub cache and quantize during loading: approximately **16.0 GB for E4B** and **23.9 GB for 12B**, about **39.9 GB combined**, in addition to existing caches. These are download sizes, not GPU-memory estimates. They do not reuse the old prequantized E4B weights. No model weights are downloaded merely by editing configs or analyzing saved results. Use the new notebook for these variants; the original `Belebele.ipynb` retains the saved historical outputs and its previous inference-enabled setting.

```bash
# Two aligned IDs in all 12 languages: 48 predictions.
.venv/bin/rendering-eval run --config configs/smoke.yaml

# Full evaluation: all 900 IDs in the selected 12 languages, 21,600 predictions.
.venv/bin/rendering-eval run --config configs/full.yaml

# 12B 4-bit: smoke first, then explicitly launch the full evaluation.
.venv/bin/rendering-eval run --config configs/smoke-12b-4bit.yaml
.venv/bin/rendering-eval run --config configs/full-12b-4bit.yaml

# Return the completed historical E4B 4-bit full run without new inference.
.venv/bin/rendering-eval run --config configs/full-e4b-4bit.yaml

# Summarize existing results without loading the model.
.venv/bin/rendering-eval analyze --output-dir results/belebele-e4b-8bit-full
```

For the completed **E4B 8-bit** full run in `results/belebele-e4b-8bit-full`, open [Belebele_analysis.ipynb](Belebele_analysis.ipynb) and **Run All** with the project `.venv` kernel. It reads existing results without model loading, inference, downloads, or rendering. It includes per-language accuracy and measured accuracy differences, paired outcomes, answer-choice distributions, invalid responses, processor resizing, page-height comparisons, and multilingual consistency. Charts stay in notebook output; clear outputs before saving a clean copy. The original evaluation notebook remains separate.

The [E4B 8-bit full-run report](reports/belebele-e4b-8bit-full.md) records the findings. To regenerate its text-only report and derived metrics, use `.venv/bin/python scripts/analyze_results.py --output-dir results/belebele-e4b-8bit-full`; this preserves the original evaluation artifacts. Both analysis notebooks default to this run and make no cross-run comparisons.

For a closer investigation, open [Belebele_errors.ipynb](Belebele_errors.ipynb). It shows invalid-response summaries (with an optional full row table), label-conditional accuracy/confusion plots, eight illustrative cases selected from the current run, and a browser for any saved mistake with its aligned English reference. Case notes are derived from the current saved scores and raw-response format. Source text is verified against manifest hashes and read strictly from the existing Hub cache. Optional previews reconstruct one processor input on CPU; no model weights or new predictions are loaded. The notebook keeps the original scores and result files intact and saves no image files.

Repeat the same `run` command to resume. Successful predictions are preserved, including invalid model answers; missing work and execution failures are retried. Changing sample count, backend, model, quantization, or rendering requires a **new output directory**. Code changes require a fresh directory for incomplete runs; complete historical runs are returned unchanged with their original provenance and no new inference. The historical E4B 4-bit results remain in `results/belebele-smoke-4bit` and `results/belebele-full-4bit`; their explicit legacy configs retain the original experiment settings. Existing result files are never converted or rescored for a new model.

The prequantized checkpoint is a third-party Unsloth conversion of Gemma. Its files total about 11 GB because embeddings and selected modules remain in BF16. The complete checkpoint is needed for inference; only loading is deferred until a run has missing predictions.

The 12 variants are `eng_Latn`, `rus_Cyrl`, `arb_Arab`, `hin_Deva`, `ben_Beng`, `tel_Telu`, `tha_Thai`, `khm_Khmr`, `mya_Mymr`, `zho_Hans`, `jpn_Jpan`, and `kor_Hang`. Full evaluation means 900 IDs in this subset, not all 122 Belebele variants. Override `--languages eng_Latn arb_Arab` or `--n-ids 2` for a smaller run.

Local inference uses batch size 1, thinking disabled, greedy decoding, and a 16-token answer budget. Nonquantized modules use BF16 on supported GPUs and FP16 otherwise. The 12B model uses the installed Transformers `gemma4_unified` implementation; processor previews reconstruct its already merged 48 px patches and count one soft token per merged patch. E4B retains its original 16 px patch reconstruction and pooling-based token accounting.

Model loading is deferred until predictions are missing, and internally created models are released after completion or failure. Reducing test IDs reduces runtime, not model-weight memory. To select another unquantized source manually, pass `--model-id`, its full `--model-revision`, `--quantization`, `--quantize-on-load`, and a new `--output-dir`. Quantize-on-load rejects checkpoints with stored quantization metadata so a 4-bit checkpoint cannot silently masquerade as an 8-bit conversion. The historical prequantized config still requires a BF16-capable GPU and verifies stored NF4 state before inference.

### Expected INT8 casting message

With BF16 or FP32 activations, bitsandbytes' INT8 backend casts its inputs to FP16 before activation quantization. The messages `MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization` and `MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization` report that internal cast; quantized linear weights remain INT8. See the [bitsandbytes implementation](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/autograd/_functions.py).

The `rendering-eval` console command and `Belebele_models.ipynb` suppress only these two exact warnings from `bitsandbytes.autograd._functions`. Other warnings, errors, and casts from other dtypes remain visible. This only changes logging; model precision, generation, scoring, and inference-source fingerprints stay unchanged, so existing resumable runs remain compatible.

After updating an existing checkout, refresh the console entry point without downloading dependencies or model weights:

```bash
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
```

Suppression applies to subsequent CLI launches or after running the new notebook's setup cell; an already-running process keeps its current logging. For a custom Python entry point, call `from rendering_eval.diagnostics import silence_expected_int8_cast_warning` and then `silence_expected_int8_cast_warning()` before inference. Direct `python -m rendering_eval` invocation bypasses the console logging setup.

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

`prepare` validates question identity and correct-answer alignment across all source rows, selects the seeded IDs, and checks selected renderings for missing glyphs and clipping. `preview` also checks local processor inputs. Open `Belebele_models.ipynb` to see original images, actual local processor-patch reconstructions, paired accuracy, and answer transitions. Set `CONFIG_PATH` in the notebook to inspect another run; notebook inference stays disabled by default with `RUN_EVALUATION=False`.

The text and image conditions use the exact same task string and answer order, plus the same short English instruction. Images contain the entire passage, question, and A–D choices. There is no duplicated task text or answer key in the image prompt.

Rendering uses a readable default: 24 px, black on white, 1,024 px width, 24 px margins, and natural height. Script-appropriate Noto Sans fonts and Pango handle shaping, wrapping, and directionality. Font/size sweeps remain deferred. Local Gemma uses a fixed budget of 280 soft image tokens; long pages may shrink.

Here, a **soft image token** means a vector of image features, not a word or character. The project retained the processor's default of **280**; this was an implementation choice, not a user-selected limit or the model's maximum. Supported budgets are 70, 140, 280, 560, and **1120**. [Google recommends higher budgets for document reading and small text](https://huggingface.co/google/gemma-4-E4B-it#5-variable-image-resolution). A larger budget can preserve more image detail but needs more compute and memory; GPU fit at 1120 has not been established here. Changing the budget requires a separately named run. The completed INT8 results continue to describe their actual 280-token setting.

Local previews show the original page and the page after Gemma preprocessing. Their headings use explicit 32 px bold text and 24 px dimension labels on a separate header band. The headings, shared canvas, and extra alignment whitespace are display-only and never enter model inputs. Neither view displays dummy tensor-padding patches; ordinary page margins remain visible.

## Artifacts and reproducibility

**No benchmark image or plot files are saved or cached.** Each rendering is created in memory, used, and released. OpenRouter's PNG buffer also stays in memory. Only JSON/JSONL results and a lock file are written:

- `manifest.json`, `selected_ids.json`: configuration, source hashes, selected IDs, content hashes, and expected prediction count.
- `rendering_metadata.json`: dimensions, actual fonts, font hashes, and glyph/clipping checks.
- `execution.json`, `sessions.jsonl`: runtime, local quantization/device placement or remote catalog/provider metadata.
- `predictions.jsonl`, `errors.jsonl`, `metrics.json`: responses, scores, inference metadata, execution failures, and paired summaries.

Notebook frontends can embed visible outputs during save/autosave. Clear displays with the notebook's final cell before saving, or sanitize a saved notebook:

```bash
python3 scripts/clean_notebook.py Belebele_models.ipynb
```

The dataset is pinned to `facebook/belebele@7899cdfa4e1e0d733fd77c848e2c273cb1d32be2`. The new model sources are [Google E4B instruction-tuned](https://huggingface.co/google/gemma-4-E4B-it) at `ee0ef6023621cff504d758262d4e04895a5af4a2` and [Google 12B instruction-tuned](https://huggingface.co/google/gemma-4-12B-it) at `43c9d014b231474fa7212dbb48a762566c2b255d`. Their INT8/NF4 loading follows [Transformers bitsandbytes support](https://huggingface.co/docs/transformers/quantization/bitsandbytes). The historical prequantized model remains pinned to `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit@6f10b2aa522ba471321d8246fd71a3dfa8c5b9a2`. The renderer is installed as a package dependency at `sign/pixel-renderer@9706296afca3dc156d76ff82450fff0fefc6f3d7`; a local document adapter uses its Pango/Cairo infrastructure without modifying upstream. Fonts use Google Fonts revision `601944d2ee97e76d4c456dc3d790db358c0c6ec5`.

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
