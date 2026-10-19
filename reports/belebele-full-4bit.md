# Belebele: full text versus rendered-image evaluation

**Text 74.31%; image 34.60%; image minus text -39.70 percentage points.** Passage-cluster bootstrap 95% interval for the gap: [-41.77, -37.61] points.

21,600/21,600 predictions (complete), 10,800 paired examples, 900 aligned questions, 488 distinct passages, 12 languages. 0 execution error attempts; strict scores replayed for every record. All means use completed pairs; equal language coverage makes the overall mean equal to the macro language mean.

Model: `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` at `6f10b2aa522ba471321d8246fd71a3dfa8c5b9a2`; bitsandbytes-prequantized-nf4. Render: 24 px Noto, 1024 px wide, 24 px margins, natural height; image budget 280 soft tokens, answer budget 16 tokens.

## Per-language results

| Language | Pairs | Text % | Image % | Gap (pp) | Gap 95% interval (pp) | Invalid image |
| --- | --- | --- | --- | --- | --- | --- |
| English | 900 | 83.67 | 69.22 | -14.44 | [-17.24, -11.71] | 0 |
| Russian | 900 | 81.11 | 40.00 | -41.11 | [-44.83, -37.29] | 0 |
| Arabic (MSA) | 900 | 77.22 | 27.78 | -49.44 | [-53.50, -45.62] | 2 |
| Hindi | 900 | 68.89 | 32.56 | -36.33 | [-40.13, -32.65] | 0 |
| Bengali | 900 | 73.78 | 29.11 | -44.67 | [-48.60, -40.84] | 0 |
| Telugu | 900 | 66.11 | 26.11 | -40.00 | [-44.31, -35.67] | 0 |
| Thai | 900 | 73.56 | 27.56 | -46.00 | [-50.06, -42.07] | 0 |
| Khmer | 900 | 70.11 | 24.44 | -45.67 | [-49.61, -41.62] | 3 |
| Burmese | 900 | 68.33 | 23.00 | -45.33 | [-49.39, -41.24] | 44 |
| Chinese (simplified) | 900 | 77.89 | 36.56 | -41.33 | [-45.19, -37.65] | 0 |
| Japanese | 900 | 74.00 | 42.33 | -31.67 | [-35.49, -27.78] | 0 |
| Korean | 900 | 77.00 | 36.56 | -40.44 | [-44.21, -36.64] | 0 |

Intervals resample passages with replacement, keeping both questions from a passage, all translations, and both modalities together (5,000 replicates, seed 42). They describe sensitivity to the passage sample. They do not measure inference randomness, and are pointwise rather than simultaneous intervals. The evaluated 900-question corpus itself has no sampling error in its reported accuracy.

## Paired outcomes

| Outcome | Count | Share of pairs |
| --- | --- | --- |
| Both correct | 3,163 | 29.29% |
| Text correct, image wrong | 4,862 | 45.02% |
| Text wrong, image correct | 574 | 5.31% |
| Both wrong | 2,201 | 20.38% |

Among text-correct pairs, the image condition retains 39.41% correct answers. There are 8.47 text-only successes for each image-only success.

## Answer-choice behavior and invalid responses

Ground-truth answer shares: A 22.89%; B 27.89%; C 27.33%; D 21.89%. Always choosing the majority label scores 27.89%; uniform random guessing has expected accuracy 25%.

| Language | Most frequent image answer | Share | Image accuracy % |
| --- | --- | --- | --- |
| English | A | 33.11% | 69.22 |
| Russian | A | 52.56% | 40.00 |
| Arabic (MSA) | B | 70.33% | 27.78 |
| Hindi | A | 45.67% | 32.56 |
| Bengali | A | 41.00% | 29.11 |
| Telugu | A | 60.78% | 26.11 |
| Thai | A | 67.89% | 27.56 |
| Khmer | A | 86.11% | 24.44 |
| Burmese | D | 41.11% | 23.00 |
| Chinese (simplified) | A | 53.67% | 36.56 |
| Japanese | A | 41.78% | 42.33 |
| Korean | A | 50.56% | 36.56 |

49 responses are invalid, including 49 truncated responses; these categories can overlap. Breakdown: Arabic (MSA) image: 2 invalid, 2 truncated; Khmer image: 3 invalid, 3 truncated; Burmese image: 44 invalid, 44 truncated. Invalid/truncated responses stay incorrect; excluding them would change the estimand. The main loss is valid wrong answers, so answer-format failures alone do not explain the modality gap.

## Rendering and processor diagnostics

| Language | Median source height (px) | Median nominal size after resize (px) | Actual soft-token range |
| --- | --- | --- | --- |
| English | 676 | 22.15 | 255–276 |
| Russian | 775 | 20.81 | 255–276 |
| Arabic (MSA) | 941 | 19.12 | 255–273 |
| Hindi | 608 | 22.74 | 252–272 |
| Bengali | 657 | 22.50 | 252–272 |
| Telugu | 675 | 22.19 | 255–276 |
| Thai | 696 | 21.52 | 252–276 |
| Khmer | 682 | 21.98 | 255–276 |
| Burmese | 1214 | 16.88 | 250–273 |
| Chinese (simplified) | 608 | 22.74 | 252–272 |
| Japanese | 713 | 21.05 | 252–272 |
| Korean | 661 | 22.50 | 252–272 |

Nominal size after resize is the configured font size multiplied by the smaller horizontal/vertical processor scale; it is a geometric proxy, not a measured glyph height. The 280-token setting is a cap, not a promise of exactly 280 actual tokens. Tall pages shrink more under the fixed budget. Within-language height-quartile plots are descriptive: document height also varies with content and question difficulty. Zero recorded missing glyphs does not establish legibility. This analysis does not rerender images or claim a fresh visual inspection.

## Consistency and runtime

On questions with all configured translations present: text: 311 questions correct in every language, 52 correct in none; image: 3 questions correct in every language, 72 correct in none.

| Mode | Summed generation time (min) | Median latency (s) | 95th percentile latency (s) |
| --- | --- | --- | --- |
| text | 32.44 | 0.174 | 0.229 |
| image | 183.41 | 1.000 | 1.129 |

Latency values come from per-prediction metadata and exclude model loading and preparation. They describe this hardware/software run, not a general model speed comparison.

## Interpretation and next experiment

This run shows a large end-to-end text/image gap under one quantized checkpoint and rendering/processor configuration. English has the strongest image accuracy; every tested language loses accuracy in image mode. The image answer distributions reveal pronounced option-position preferences, particularly in Khmer and Arabic. These are observed behaviors, not evidence of a specific OCR or script mechanism.

A useful next controlled comparison is to rerun identical IDs with a larger image budget (560 or 1120 tokens) in a new output directory, checking GPU fit first. Compare paired changes within each language, retain invalid responses as incorrect, and inspect processor previews for the same examples. A second model would help separate checkpoint-specific behavior from benchmark/rendering behavior. No additional inference was launched by this analysis.

## Reproduce and provenance

```bash
.venv/bin/python scripts/analyze_results.py --output-dir results/belebele-full-4bit
```

Open `Belebele_analysis.ipynb` for plots. It needs no model, dataset download, renderer, or GPU. Plots live only in notebook memory; clear outputs before saving. Original run artifacts remain unchanged. The derived `analysis.json` contains original-file SHA-256 hashes, all tables, bootstrap settings, and run configuration.

Manifest fingerprint: `f567fe9b4b20b63b648ccfb883b98608a161e02ab1040362070bfd03f505e4a8`. Dataset: `facebook/belebele@7899cdfa4e1e0d733fd77c848e2c273cb1d32be2`. See README for dataset, model, font, and renderer attribution.
