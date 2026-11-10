# Member 4 implementation

## What this implements
- Passage-level word-overlap pruning using Member 1's selected preprocessing:
  lowercase, NLTK stop-word removal (negations retained), Porter stemming, minimum token length 2, and title-token removal.
- Sentence splitting before NLI scoring.
- DeBERTa inference in batches with dynamic padding and `torch.inference_mode()`.
- Max entailment probability over all retained evidence sentences for each fact.
- Evaluation of fixed-threshold predictions and separately development-tuned thresholds.
- Runtime, passage-retention/reduction, number of sentence pairs, and process peak RSS reporting.

## Important integration note
The supplied Member 3 `factcheck.py` still has `raise Exception("Not implemented")` in
`EntailmentModel.check_entailment()` and `raise Exception("Implement me")` in
`EntailmentFactChecker.predict()`. Member 3's `tune.py` also expects a
`score_details(fact, passages)` method. This Member 4 script intentionally bypasses
those unfinished methods and loads the same DeBERTa checkpoint directly. It does not
modify `factcheck.py`.

## Files needed in the repository
Place this folder at:
`experiments/m4_optimization/`

The repository should contain:
- `factcheck.py` (with Member 1's `TextPreprocessor` and `split_sentences`)
- `Framework Code/data/dev_labeled_ChatGPT.jsonl`
- `Framework Code/data/passages_bm25_ChatGPT_humfacts.jsonl`

## Install / environment
Use the project's existing requirements. `resource` is part of Python on Linux/macOS.
The script uses:
- torch
- transformers
- factcheck.py
- existing coursework data files

The model is downloaded from Hugging Face on first run, so internet access is required then.
For CPU:
```bash
python experiments/m4_optimization/run_m4_experiments.py
```
For CUDA, only if PyTorch CUDA is installed and available:
```bash
python experiments/m4_optimization/run_m4_experiments.py --cuda
```

## Useful experiment commands
Run batch-size comparisons with no pruning and pruning thresholds 0.20, 0.30, 0.40:
```bash
python experiments/m4_optimization/run_m4_experiments.py --batch-sizes 1 4 8 16 --pruning-thresholds 0.20 0.30 0.40
```

Use a fixed threshold of 0.5 for direct configuration comparisons:
```bash
python experiments/m4_optimization/run_m4_experiments.py --classification-threshold 0.5
```

The script also reports a threshold optimized on the development set for each configuration.
Do not describe those tuned scores as held-out test results. For a fair optimization comparison,
compare fixed-threshold metrics first, and compare tuned metrics separately.

## Outputs
- `results/m4/results.csv`: one row per experiment configuration.
- `results/m4/report.json`: settings, dataset count, runtime, and results.

## Methodological notes
1. A pruning threshold of 0.0 is equivalent to retaining all passages with nonnegative overlap; it is the no-pruning baseline.
2. If pruning would remove every passage for a fact, the highest-overlap passage is retained. This avoids silently converting an empty evidence set into NS.
3. Passage overlap is computed independently for each retrieved passage; this differs from Member 1's fact-level max aggregation used for classification.
4. Entailment probabilities are obtained from the model's entailment label. The script resolves the label from `id2label`, with the documented checkpoint label order as fallback.
5. Sentence-level evidence is used, and the fact score is the maximum entailment probability over retained sentence pairs, consistent with the coursework brief.
6. Runtime includes sentence preparation, pruning, tokenization, and model inference for all facts. Model download/loading is excluded.
7. Peak RSS is process-wide high-water memory, not per-experiment incremental memory. CUDA peak memory is not included in this initial script.
8. These experiments use the development set because that is the supplied labelled set. Threshold tuning on it is optimistic and must be reported transparently.
