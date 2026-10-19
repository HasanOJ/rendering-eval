# Validation record

Date: 2026-10-18. WSL/Linux, Python 3.12.3, NVIDIA GeForce RTX 4070 (12,282 MiB).

## Current implementation

- The smoke config selects **2 aligned IDs × 12 languages × 2 modalities = 48 predictions**. The full local config selects all 900 IDs in the same 12 languages: **21,600 predictions**.
- Local inference defaults to direct loading of `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit@6f10b2aa522ba471321d8246fd71a3dfa8c5b9a2`. This restores saved NF4 quantization state and module exclusions, avoiding a BF16 source conversion. Model loading is deferred until missing predictions exist. Internally created models are closed after success or error.
- The previous BF16-to-NF4 loading attempt was stopped following the user's request for a prequantized checkpoint. Its artifacts and the earlier 13-prediction 8-bit run were preserved separately. The prequantized checkpoint files total about 10.95 GB; embeddings and selected modules remain in BF16.
- Colab-specific setup and packaging were removed. Setup now uses an ordinary repository checkout and Python environment in WSL or on a GPU cloud host. No hosted cloud GPU execution was performed.

## Checks completed

- **37 tests passed**, including real-data integration with all 24 selected multilingual examples and the prequantized checkpoint's processor.
- Dataset alignment checks validate all 900 IDs and answer labels across configured languages before selection. Tests cover duplicate/missing IDs, answer mismatches, empty content, optional passage split keys, and deterministic ordering.
- Rendering/processor checks found no unknown glyphs or clipping in the selected examples. The fixed 280-token local image budget and actual patch reconstruction remain tested.
- Resume tests preserve prior predictions, recover interrupted trailing records, reject duplicates and incompatible configurations, and avoid loading a model for complete runs. Complete historical results can be returned unchanged after code updates; incomplete runs still require compatible code.
- Model-lifetime tests verify cleanup after successful inference and generation failures.
- OpenRouter tests verify text/image prompt parity, PNG encoding only into memory, free-pricing restrictions, credential exclusion from metadata, truncation, vision-model validation, and CLI overrides.
- The simplified notebook executed end to end with inference disabled, including the completed-results plots. Its final cleanup left **zero embedded image outputs**. No executed notebook was saved. The distributed notebook remains clean.
- Installed package compatibility checks pass. Exact top-level package versions are in `constraints-wsl.txt`.

## Live OpenRouter test

- One English Belebele question was tested in both text and rendered-image modes using `nvidia/nemotron-3.5-super-vl:free` and the configured environment key.
- Both predictions were valid and correct; reported total cost was **$0**. Results are in `results/belebele-openrouter-free-nemotron/`.
- Repeating the default free config returned the completed results without additional inference requests.
- The two free Gemma endpoints initially attempted returned HTTP 429 before any prediction. Those attempts were preserved in separate result directories. No paid inference was launched.
- Remote preprocessing is provider-managed. Local Gemma processor settings are not claimed for OpenRouter inputs.

## Local smoke test

The corrected direct-load prequantized smoke test completed **48/48 predictions**, comprising 24 text/image pairs:

- Text: **22/24 (91.67%)**; image: **8/24 (33.33%)**.
- Both correct: 7; text-only correct: 15; image-only correct: 1; both wrong: 1.
- Invalid/truncated model responses: **0**; execution failures in the final run: **0**.
- Actual model footprint: **10.119 GiB**; peak allocated GPU memory: **10.929 GiB**; peak reserved: **11.178 GiB**.
- Summed generation latency: **13.88 seconds**, excluding loading and preparation.
- GPU usage after the CLI released the model: **236 MiB**.
- Results: `results/belebele-smoke-4bit/`. A completed-run resume left prediction bytes unchanged and loaded no model.

The initial direct-load preflight exposed a Transformers subtree-exclusion incompatibility. The adapter now expands saved parent-module names into whole-subtree patterns and checks quantization state on every quantized layer. This preserves the checkpoint's intended BF16 vision/projector subtrees and its stored quantized weights. The failed preflight's one text prediction is preserved in `results/belebele-smoke-4bit-loader-initial/` and is not mixed into the final results.

These are two-ID smoke results and do not establish robust language rankings or causal OCR/script effects.

Run or resume:

```bash
.venv/bin/rendering-eval run --config configs/smoke.yaml
```

Full local evaluation, deliberately not launched:

```bash
.venv/bin/rendering-eval run --config configs/full.yaml
```

Full Qwen evaluation via the configured OpenRouter key, deliberately not launched:

```bash
.venv/bin/rendering-eval run --config configs/openrouter-full.yaml
```

No benchmark image files, processor-preview files, or plot files are written by these workflows. Use `Belebele.ipynb` to visualize them in memory.

## Completed full-run analysis — 2026-10-19

The user completed the full local run after the original validation above. Its existing artifacts in `results/belebele-full-4bit` were audited without inference or rerendering:

- **21,600/21,600 unique predictions**, 10,800 complete pairs, 900 aligned questions from **488 passages**, 12 languages, no execution error attempts. Strict scoring was replayed for every prediction.
- Text: **8,025/10,800 (74.31%)**; image: **3,737/10,800 (34.60%)**; image minus text: **−39.70 percentage points**. The 95% paired passage-cluster bootstrap interval is **[−41.77, −37.61] points**, using 5,000 replicates and seed 42. Questions, translations, and both modalities are resampled together within each passage.
- Paired outcomes: 3,163 both correct; 4,862 text only; 574 image only; 2,201 both wrong. All 49 invalid responses are image responses and truncated: Burmese 44, Khmer 3, Arabic 2.
- `Belebele_analysis.ipynb` executed end to end in a real local Jupyter kernel with Hugging Face access disabled. It loaded no PyTorch/model and made no inference calls. All three figures were visually inspected in memory. Original run artifacts and the user's evaluation notebook retained their SHA-256 hashes.
- The new analysis notebook is distributed with no execution counts or outputs. Validation cleared chart displays in memory and did not save an executed notebook or image files.
- Four new analysis tests pass, covering passage-level dependence, paired counts, strict-score replay, preservation of originals, plots without file writes, and the clean analysis notebook. The overall unit run returned **39 passed, 1 skipped, 1 failed**. The failure is the existing distribution-cleanliness test for the user's already-executed `Belebele.ipynb` (saved outputs and evaluation enabled); that notebook was preserved unchanged. The skipped test is the opt-in dataset/processor integration test; no inference code changed in this analysis.

See `Belebele_analysis.ipynb` for plots and `reports/belebele-full-4bit.md` for the text findings. Derived `analysis.json` records original-artifact hashes and analysis settings; original `metrics.json`, predictions, and manifests were not rewritten. No paid/API evaluation or further local evaluation was launched.

## Error investigation — 2026-10-19

- `Belebele_errors.ipynb` executed end to end offline, including every invalid response, conditional accuracy/confusion plots, eight annotated case cards, pinned-source checks for all 10,800 examples, and a configurable case browser. Its default execution imported no PyTorch and loaded no model weights. Original notebooks and every full-run artifact retained their SHA-256 hashes; the new notebook is saved without outputs.
- All 49 invalid responses were image-mode prose cut at 16 generated tokens. There are 19 distinct raw strings: 47 descriptive/preamble responses, one claim of incomplete input, and one request for the passage. Same-language text was correct for 36/49. The cached chat template confirms that `enable_thinking=False` is a supported switch; generation remains unconstrained rather than restricted to A–D. The saved decoded text does not retain special-token/channel markers.
- The 7,014 valid wrong image answers account for 99.31% of the 7,063 image errors. Making all 49 invalid answers correct would raise image accuracy only to 35.06%, a 0.45-point upper-bound gain; no such rescoring was performed.
- Manually inspected in-memory original and processor-patch previews for the Burmese passage request, Arabic incomplete-input claim, Khmer translation preamble, and Khmer numeric wrong answer. All included passage/question/A–D options with no visible clipping or missing-glyph boxes. Rendering and processor metadata matched the saved run. This is a layout inspection, not native-speaker glyph validation. The inspected Khmer invalid page was enlarged (1024×528 → 1104×576), so its failure does not require downsampling.
- Three new tests pass for invalid-error accounting, preservation of scores/results, confusion totals, truncated-letter classification, and the clean inference-free error notebook. No new inference, GPU model load, image files, or image caches were created. Optional notebook previews use cached processor files and CPU tensors only.
