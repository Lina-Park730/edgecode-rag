# EdgeCode — Code-Aware RAG Assistant

EdgeCode retrieves Python source code and generates explanations with file and line references using a local Ollama model. It compares fixed-size dense retrieval, AST-aware dense retrieval, and AST-aware hybrid BM25–dense retrieval on Click 8.1.7.

## Retrieval pipeline

1. Split Python source into fixed-size windows or AST-aware chunks.
2. Retrieve candidates with BGE embeddings and, for the hybrid method, BM25.
3. Combine hybrid rankings with weighted reciprocal-rank fusion.
4. Pass the selected code excerpts to Ollama.
5. Display the answer with source IDs mapped to filenames and inclusive line ranges.

| Setting | Value |
| --- | --- |
| AST chunk size | Up to 60 lines |
| Fixed-size baseline | Up to 80 lines |
| Window overlap | 10 lines |
| Candidates per retriever | 20 |
| Final retrieval K | 5 |
| Hybrid weights | BM25 0.1, dense 0.9 |
| RRF constant | 60 |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Default generation model | `gemma4:e4b` |

## Setup

Requires Python 3.10+ and Ollama with a locally installed model.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_repository.py
```

The download script fetches Click 8.1.7 and verifies the archive SHA-256. The source is stored in `repository/click-8.1.7/`, which is excluded from Git. Existing source directories are not overwritten or reverified. For an existing official archive:

```bash
python download_repository.py --archive /path/to/8.1.7.zip
```

The first retrieval run downloads the BGE embedding model. Open Ollama or start its server with `ollama serve`, then check the connection:

```bash
ollama list
python ask.py --check
```

## Dataset Setup

This project uses Click v8.1.7 as the reference codebase for retrieval evaluation.

1. Download the Click v8.1.7 source code from:
   https://github.com/pallets/click/archive/refs/tags/8.1.7.zip

2. Extract the archive into the `repository/` directory.

3. Ensure the resulting directory structure is:

   repository/
   └── click-8.1.7/
       ├── src/
       ├── tests/
       └── ...

## Usage

```bash
python ask.py "How does Click resolve a command name?"
python ask.py "How does Click handle missing required parameters?" --output outputs/answer.json
python ask.py "How does Click resolve a command name?" --arm ast_dense
```

Available retrieval methods are `fixed_dense`, `ast_dense`, and `ast_hybrid` (default). Use `--model` to select another installed Ollama model.

To query another Python repository:

```bash
python ask.py "Where is input validation handled?" --repo /path/to/repo --include-prefix src/
```

`--include-prefix` can be repeated. Custom repositories default to all discoverable Python files; the Click example indexes `src/click/`. Each invocation builds an in-memory index.

The default Ollama endpoint is `http://localhost:11434`. `--ollama-url`, `EDGECODE_OLLAMA_URL`, and `EDGECODE_MODEL` configure the endpoint and model. Retrieved code is sent to the selected endpoint.

## Source references

Answers use citations such as `[S1]`. The source list maps each ID to `path:start_line-end_line`, relative to the indexed repository. Only excerpts included in the prompt appear in this list. Missing or unknown citation IDs produce warnings; citation checks do not verify the factual accuracy of each claim.

The default excerpt budget is 24,000 characters. Whole chunks that exceed the remaining budget are skipped. This character limit is not an exact token limit. Context and generation limits are configurable with `--context-chars`, `--num-ctx` (16,384 by default), and `--max-tokens` (1,024 by default).

## Evaluation

The development set contains 20 questions, four per category. Development tuning compares AST sizes of 60, 80, and 120 lines and BM25 weights from 0 to 1. `--quick` evaluates only the 80-line setting. The separate 80-question evaluation set contains 16 questions per category and uses the fixed settings above.

```bash
python scripts/run_development_tuning.py --quick
python scripts/run_evaluation.py --questions data/questions_eval.json --validate-only
python scripts/run_evaluation.py --questions data/questions_eval.json
```

Evaluation compares all three retrieval methods at K=5. Development tuning reports K=3, 5, and 10. Metrics include MRR, recall, precision, and hit rate. A gold-span match requires the same file, at least one overlapping line, and at least 25% coverage of the gold span. These measures assess retrieval, not generated-answer correctness. Development scores reflect parameter selection on the same questions; semantic independence and gold annotations require human review.

Optional generation in `scripts/run_evaluation.py` uses `--ollama-model` and `--tokenizer`, with a matching tokenizer and chat template. It requires `transformers` separately. This evaluation path performs tokenizer-based prompt budgeting, unlike the character budget in `ask.py`.

Results are written to `outputs/`. See [development methodology](docs/DEVELOPMENT.md) for tuning outputs and selection criteria.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover development exports, dataset guards, request formatting, source mapping, citation checks, and connection errors using synthetic embeddings and mocked HTTP responses. They do not measure live model performance.

## Project structure

| Path | Purpose |
| --- | --- |
| `ask.py` | Question-answering CLI |
| `edgecode/` | Chunking, retrieval, fusion, evaluation, and generation |
| `download_repository.py` | Reference source setup |
| `scripts/run_development_tuning.py` | Development grid search |
| `scripts/run_evaluation.py` | Three-method evaluation |
| `scripts/diagnose_retrieval.py` | Retrieval diagnostics |
| `scripts/print_k_results.py` | Development metric summaries |
| `data/`, `config/` | Question sets and development configuration |
| `tests/` | Automated implementation checks |

## Troubleshooting

- Connection errors: start Ollama and run `python ask.py --check`.
- Missing model: use a model name listed by `ollama list`.
- Slow generation: increase `--timeout`.
- Memory pressure: reduce `--num-ctx` or use a smaller installed model.
- Empty answers: increase `--max-tokens`.

## References

- [Click](https://github.com/pallets/click), maintained by Pallets, is the reference codebase. Its BSD license is retained in the downloaded source. Evaluation line numbers are tied to version 8.1.7.
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md).
