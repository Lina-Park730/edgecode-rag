# EdgeCode — 20-question development tuning

Development tuning selects AST chunk size and BM25–dense fusion weights using a 20-question development set. Evaluation and generation are described in the root README.

## Run on Mac / Linux

Open a terminal in this folder (the one containing scripts/run_development_tuning.py):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_repository.py
python scripts/run_development_tuning.py
```

The first run downloads the BGE model and needs internet. Run `python download_repository.py` first to install Click 8.1.7 under repository/ with its license; this directory is excluded from Git. Only src/click/ is indexed.

For a shorter run using all 20 questions but only AST size 80:

```bash
python scripts/run_development_tuning.py --quick
```

Full run: 3 sizes (60, 80, 120 lines) × 11 BM25 weights (0 to 1) = 33 configurations. Quick run: 11 configurations. Each configuration uses ALL 20 questions. candidate_k=20 is candidates per retriever per question, not the number of questions. Embeddings/candidate lists are reused across weights for each AST size.

The startup message says `20 DEVELOPMENT questions`. Output selected_config.json records dev_question_count=20; each row of development_tuning.csv records question_count=20. Default output directories are outputs/development20 and outputs/development20_quick. Use --output PATH to preserve a run in a separate directory.

## Dataset

The active file is data/questions_dev20.json: 20 questions, all split=dev, four per category. Seven question wordings were revised to include realistic identifier lookups and observed error fragments, while the semantic questions were kept unchanged. These are hypotheses about when lexical retrieval may help; a BM25 advantage is not guaranteed. See [question revisions](../QUESTION_CHANGES.md) for exact edits and annotation corrections.

The original questions are retained in data/questions_original_reference.json for provenance only and are never loaded by the runner. The runner rejects a count other than 20 or any non-dev split.

The gold annotations include a source review against Click 8.1.7. Q02's incorrect longest-option-name explanation was corrected to longest-option-prefix priority. See [source annotations](../GOLD_SOURCE_REVIEW.md) for the passages. Expected answers are review aids, not inputs to retrieval or scoring. The retrievers receive only the question text.

## Development outputs

| File | Purpose |
|---|---|
| development_tuning.csv | Overall metrics for every size and weight |
| development_ranked_configs.csv | Configurations sorted by selection objective |
| development_all_query_metrics.csv | Every question at every size/weight and K=3,5,10 |
| development_category_metrics.csv | Category means at every size/weight and K |
| development_hybrid_vs_dense.csv | Best interior hybrid versus dense-only at each SAME AST size |
| development_all_rankings.json | Top retrieved passages for all configurations |
| development_query_metrics.csv | Baseline versus overall selected configuration, per question |
| development_baseline.csv | Fixed-size dense reference |
| selected_config.json | Chosen parameters for later use |
| development_results.json | Development summary and dataset hash |
| development_dataset_snapshot.json | Exact questions used for this run |
| development_config_snapshot.json | Input configuration used for this run |

To inspect BM25's contribution: filter development_category_metrics.csv to ast_max_lines=80, k=5, then compare bm25_weight=0 (dense), 1 (BM25), and interior values separately in each category. Use development_all_query_metrics.csv and development_all_rankings.json to inspect wins and losses. Comparisons at the same AST size isolate the effect of retrieval method.

Selection maximizes mean MRR@5; ties use Recall@5, Hit@5, Precision@5, then closeness to weight 0.5 and AST size 80. Both pure endpoints remain eligible. A nonzero BM25 weight is never forced. The best interior hybrid is also reported even when dense wins, along with its score difference.

Metric matching requires same-file overlap covering at least 25% of a gold span (and at least one line). Recall counts gold spans found; MRR uses the first matching chunk, so it does not measure complete multi-function answers. Precision divides matching chunks by returned chunks (up to K). These overlap metrics are development proxies, not human correctness judgments; large chunks and broad annotations can affect them.

All 20 questions are development data, so their scores are not independent test results. Question wording was revised after an earlier dense-only optimum; the revision history is recorded in QUESTION_CHANGES.md.

## Implementation checks

```bash
python -m unittest discover -s tests -v
```

These checks use synthetic embeddings and a tiny synthetic repo to test exports and selection. They are not benchmark evidence. The dataset/source check also validates the bundled real question count and gold bounds. These checks do not measure real-embedding retrieval performance.

Zero-weight fusion branches are skipped, so they cannot introduce extra zero-score candidates or affect tie-breaking.
