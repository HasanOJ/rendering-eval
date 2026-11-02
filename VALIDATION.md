# Validation record

## Simplified measured-results analysis — 2026-11-02

- Removed passage resampling, estimated confidence intervals, their report columns, and plot error bars at the user's request. The overview now shows measured per-language accuracy differences. Removed the analysis script's resampling options and the notebook's related settings and explanations.
- Regenerated the E4B INT8 Markdown report and derived `analysis.json` using only observed results. Derived report schema version 2 omits the removed statistical fields. Every measured group value matches the preceding analysis; original predictions, scores, manifests, and other saved run files retain their hashes.
- Targeted reporting/error-analysis checks: **8 passed, 1 failed**. The failure is the existing cleanliness assertion for the user's now-executed error notebook, which was preserved unchanged. The simplified overview notebook was validated offline without loading PyTorch or a model. `git diff --check` passed.

## Readable preview headings — 2026-11-02

- Replaced the tiny Pillow-default preview labels with 32 px bold headings and 24 px details, using fonts bundled with the existing Matplotlib dependency. The two headings explicitly identify the original rendering and the page after Gemma preprocessing.
- Verified on 1024×788 and 912×672 sample arrays that both displayed image regions remain pixel-identical; only the external header changed. Visually inspected the header and validated the notebook schema/code cells. No model or inference was run.
- Clarified visual-token terminology, the inherited 280-token processor default, the supported 1120 maximum, and the distinction between tensor padding and page margins in the README and model notebook. The completed run's 280-token setting is unchanged.

## Completed E4B INT8 analysis notebooks — 2026-11-02

- Both `Belebele_analysis.ipynb` and `Belebele_errors.ipynb` now default to `results/belebele-e4b-8bit-full` and analyze only that run, with no cross-run comparison. The original evaluation notebook is unchanged. Saved analysis notebooks have no execution counts, outputs, or widget state; Run All with the project `.venv` kernel displays the plots.
- Audited **21,600/21,600 predictions**, 10,800 complete pairs, 900 aligned questions from 488 passages in 12 languages. Strict scoring was replayed; there were no duplicate predictions or execution-error attempts. Text accuracy: **77.47%**; image accuracy: **35.23%**; image minus text: **−42.24 percentage points**. The paired passage-cluster bootstrap 95% interval is **[−44.10, −40.30] points**, from 5,000 replicates with seed 42.
- There are **6,995 image errors**: 6,857 valid wrong choices (98.03%) and 138 invalid responses. All invalid responses are image-mode outputs truncated at 16 tokens: Burmese 131, Arabic 3, Khmer 3, English 1. They comprise 35 distinct raw strings; observable behavior counts are 105 prose/preambles, 32 passage requests, and one incomplete-input claim. Same-language text is correct for 105/138. Even making all invalid responses correct would add at most **1.28 points**, raising image accuracy to 36.51%; no rescoring was performed.
- Removed fixed counts, case annotations, and prior visual-inspection claims from the error notebook. Eight illustrative cases are now selected deterministically from the current saved predictions; their notes describe saved outcomes and response formats. Cached source text is verified against manifest hashes. The optional processor-only preview is disabled by default and keyed to the selected checkpoint and image-token budget.
- Overview plot model/precision labels and report answer-choice commentary now derive from saved metadata. The analysis script defaults to the completed INT8 directory. Generated `reports/belebele-e4b-8bit-full.md` and a new derived `analysis.json` without rewriting original run files.
- Executed both notebooks end to end in real local Jupyter kernels with Hub access offline. All four figures were visually inspected using temporary validation outputs outside the repository. Default execution loaded no PyTorch/model, performed no inference, and created no repository image files. Original result-file hashes matched their pre-analysis values.
- Full suite: **54 passed, 1 skipped, 1 failed**. The failure is the existing cleanliness check for the user's executed `Belebele.ipynb`, which was preserved. The skipped test is opt-in data/processor integration. New checks cover current-run case selection and notes, dynamic plot labels, and the notebooks' INT8 defaults. `git diff --check` passed.

## INT8 warning suppression — 2026-11-01

- Added a logging filter for only the exact BF16-to-FP16 and FP32-to-FP16 `MatMul8bitLt` warnings at WARNING level from `bitsandbytes.autograd._functions`. Other dtype-cast messages, warnings, and errors are retained. Model precision and inference settings are unchanged.
- Installed the console entry point through `rendering_eval.diagnostics:main` and added the same setup to `Belebele_models.ipynb`. Existing processes retain their current logging; subsequent console launches and notebook setup executions apply the filter. The README explains the expected cast and the scope of suppression.
- Both targeted logging tests passed, exercising formatted and literal messages for both BF16 and FP32, repeat installation, retained warnings/errors (including FP64 casts), and console setup before CLI execution. The installed entry point and notebook schema/code cells were checked without loading a model or importing PyTorch.
- The inference-source fingerprint and static result artifact hashes matched their pre-change values. No result files, inference source files, model weights, or GPU state were changed. Existing resume compatibility is preserved.

## Model variants — 2026-11-01

- Default local configs now select `google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`, quantized on load with bitsandbytes INT8. Added smoke/full configs for `google/gemma-4-12B-it@43c9d014b231474fa7212dbb48a762566c2b255d`, quantized on load with NF4 and double quantization. Both use fresh, model-specific results directories; explicit historical E4B 4-bit configs reproduce the saved experiment settings.
- Public Hub metadata verified the pinned model IDs, architecture types, processor configurations, and support for `enable_thinking` in both chat templates. The installed Transformers 5.20.1 maps these configs to `Gemma4ForConditionalGeneration` and `Gemma4UnifiedForConditionalGeneration`. No dependency upgrade was needed.
- Official BF16 source weight downloads are 15,992,595,884 bytes for E4B and 23,919,549,408 bytes for 12B. No weights were downloaded or converted, no real model was instantiated, and no GPU inference was launched. GPU fit, real loading behavior, and accuracy of both new variants remain untested.
- Actual installed E4B and Unified 12B image processors were exercised on a synthetic CPU image. Tests verified exact patch reconstruction, dimensions, padding handling, and six image tokens for a 144×96 image. Unified emits already merged 48 px patches; E4B emits 16 px patches pooled 3×3. Runtime metadata and previews now support both formats.
- Mocked model-loader tests verified pinned revisions, GPU-0 placement, INT8 versus NF4 options, preserved legacy prequantized loading, and rejection of attempts to requantize an already quantized source. These tests do not establish real GPU compatibility.
- `Belebele_models.ipynb` is a new clean notebook with inference disabled and config selection for both new variants. Existing notebooks and result artifacts were preserved. Historical experiment equality was checked against both saved E4B 4-bit manifests.
- Full test suite: **47 passed, 1 skipped, 3 failed**. All failures are existing notebook-cleanliness checks: the user's `Belebele.ipynb`, `Belebele_analysis.ipynb`, and `Belebele_errors.ipynb` contain saved execution outputs, which were retained. The skipped check is opt-in data/processor integration. The targeted model/processor/OpenRouter suite passed **15/15**. `pip check` and `git diff --check` passed.

The sections below record the earlier E4B 4-bit baseline; their commands and performance figures describe that historical configuration. Use `configs/smoke-e4b-4bit.yaml` and `configs/full-e4b-4bit.yaml` to select it now.

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
