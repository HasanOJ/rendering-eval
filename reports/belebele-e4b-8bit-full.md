# Belebele: full text versus rendered-image evaluation

**Text 77.47%; image 35.23%; image minus text -42.24 percentage points.**

21,600/21,600 predictions (complete), 10,800 paired examples, 900 aligned questions, 488 distinct passages, 12 languages. 0 execution error attempts; strict scores replayed for every record. All means use completed pairs; equal language coverage makes the overall mean equal to the macro language mean.

Model: `google/gemma-4-E4B-it` at `ee0ef6023621cff504d758262d4e04895a5af4a2`; bitsandbytes-int8. Render: 24 px Noto, 1024 px wide, 24 px margins, natural height; image budget 280 soft tokens, answer budget 16 tokens.

## Per-language results

| Language | Pairs | Text % | Image % | Gap (pp) | Invalid image |
| --- | --- | --- | --- | --- | --- |
| English | 900 | 86.67 | 71.78 | -14.89 | 1 |
| Russian | 900 | 85.22 | 42.89 | -42.33 | 0 |
| Arabic (MSA) | 900 | 80.44 | 29.44 | -51.00 | 3 |
| Hindi | 900 | 72.44 | 33.33 | -39.11 | 0 |
| Bengali | 900 | 76.56 | 26.56 | -50.00 | 0 |
| Telugu | 900 | 69.22 | 23.11 | -46.11 | 0 |
| Thai | 900 | 75.78 | 29.11 | -46.67 | 0 |
| Khmer | 900 | 72.89 | 24.22 | -48.67 | 3 |
| Burmese | 900 | 72.00 | 20.78 | -51.22 | 131 |
| Chinese (simplified) | 900 | 80.11 | 40.11 | -40.00 | 0 |
| Japanese | 900 | 78.11 | 42.89 | -35.22 | 0 |
| Korean | 900 | 80.22 | 38.56 | -41.67 | 0 |

## Paired outcomes

| Outcome | Count | Share of pairs |
| --- | --- | --- |
| Both correct | 3,334 | 30.87% |
| Text correct, image wrong | 5,033 | 46.60% |
| Text wrong, image correct | 471 | 4.36% |
| Both wrong | 1,962 | 18.17% |

Among text-correct pairs, the image condition retains 39.85% correct answers. There are 10.69 text-only successes for each image-only success.

## Answer-choice behavior and invalid responses

Ground-truth answer shares: A 22.89%; B 27.89%; C 27.33%; D 21.89%. Always choosing the majority label scores 27.89%; uniform random guessing has expected accuracy 25%.

| Language | Most frequent image answer | Share | Image accuracy % |
| --- | --- | --- | --- |
| English | A | 30.89% | 71.78 |
| Russian | A | 45.56% | 42.89 |
| Arabic (MSA) | B | 38.78% | 29.44 |
| Hindi | A | 40.56% | 33.33 |
| Bengali | A | 46.00% | 26.56 |
| Telugu | A | 65.33% | 23.11 |
| Thai | A | 62.89% | 29.11 |
| Khmer | A | 86.11% | 24.22 |
| Burmese | D | 40.11% | 20.78 |
| Chinese (simplified) | A | 47.11% | 40.11 |
| Japanese | A | 40.44% | 42.89 |
| Korean | A | 42.44% | 38.56 |

138 responses are invalid, including 138 truncated responses; these categories can overlap. Breakdown: English image: 1 invalid, 1 truncated; Arabic (MSA) image: 3 invalid, 3 truncated; Khmer image: 3 invalid, 3 truncated; Burmese image: 131 invalid, 131 truncated. Invalid/truncated responses stay incorrect; excluding them would change the estimand. The main loss is valid wrong answers, so answer-format failures alone do not explain the modality gap.

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

On questions with all configured translations present: text: 328 questions correct in every language, 36 correct in none; image: 1 questions correct in every language, 63 correct in none.

| Mode | Summed generation time (min) | Median latency (s) | 95th percentile latency (s) |
| --- | --- | --- | --- |
| text | 100.57 | 0.549 | 0.659 |
| image | 952.02 | 5.023 | 6.518 |

Latency values come from per-prediction metadata and exclude model loading and preparation. They describe this hardware/software run, not a general model speed comparison.

## Interpretation and next experiment

This run shows a large end-to-end text/image gap under one quantized checkpoint and rendering/processor configuration. English has the strongest image accuracy. The largest image answer-choice concentrations are Khmer: A (86.11% of responses); Telugu: A (65.33% of responses); Thai: A (62.89% of responses). These are observed behaviors, not evidence of a specific OCR or script mechanism.

A useful next controlled comparison is to rerun identical IDs with a larger image budget (560 or 1120 tokens) in a new output directory, checking GPU fit first. Compare paired changes within each language, retain invalid responses as incorrect, and inspect processor previews for the same examples. A second model would help separate checkpoint-specific behavior from benchmark/rendering behavior. No additional inference was launched by this analysis.

## Reproduce and provenance

```bash
.venv/bin/python scripts/analyze_results.py --output-dir results/belebele-e4b-8bit-full
```

Open `Belebele_analysis.ipynb` for plots. It needs no model, dataset download, renderer, or GPU. Plots live only in notebook memory; clear outputs before saving. Original run artifacts remain unchanged. The derived `analysis.json` contains original-file SHA-256 hashes, all measured tables and run configuration.

Manifest fingerprint: `95ead1a901f4dcd1f37993356954886642d2b6b28817814c39e349756c497fd2`. Dataset: `facebook/belebele@7899cdfa4e1e0d733fd77c848e2c273cb1d32be2`. See README for dataset, model, font, and renderer attribution.
