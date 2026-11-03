#!/usr/bin/env python3
"""Ask a source-grounded question with AST retrieval and local Ollama."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from edgecode.generation import OllamaClient, answer_question, render_answer

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question', nargs='?')
    parser.add_argument('--model', default=os.getenv('EDGECODE_MODEL', 'gemma4:e4b'))
    parser.add_argument('--ollama-url', default=os.getenv('EDGECODE_OLLAMA_URL', 'http://localhost:11434'))
    parser.add_argument('--check', action='store_true', help='Check server and model without embedding downloads')
    parser.add_argument('--repo', type=Path, default=ROOT/'repository/click-8.1.7')
    parser.add_argument('--include-prefix', action='append', help='Repeat for several directories; defaults to all Python files in custom repositories')
    parser.add_argument('--arm', choices=['fixed_dense', 'ast_dense', 'ast_hybrid'], default='ast_hybrid')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--num-ctx', type=int, default=16384)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--context-chars', type=int, default=24000)
    parser.add_argument('--output', type=Path, help='Save answer and source metadata as JSON')
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > 20:
        parser.error('--top-k must be between 1 and 20')
    if min(args.timeout, args.num_ctx, args.max_tokens, args.context_chars) <= 0:
        parser.error('Timeout and context/generation budgets must be positive')
    if not args.check and (not args.question or not args.question.strip()):
        parser.error('Provide a question or use --check')
    try:
        client = OllamaClient(args.model, args.ollama_url, args.timeout)
        version = client.check()
        if args.check:
            print(f'Ollama {version}; model {args.model} is available.')
            return 0
        from edgecode.chunking import chunk_repository
        from edgecode.dense import DenseIndex
        from edgecode.bm25 import BM25Index
        from edgecode.fusion import weighted_reciprocal_rank_fusion
        prefixes = args.include_prefix
        if prefixes is None and args.repo.resolve() == (ROOT/'repository/click-8.1.7').resolve():
            prefixes = ['src/click/']
        chunks = chunk_repository(args.repo, strategy='fixed' if args.arm == 'fixed_dense' else 'ast',
                                  max_lines=80 if args.arm == 'fixed_dense' else 60,
                                  overlap_lines=10, include_prefixes=prefixes)
        if not chunks:
            raise ValueError('No Python chunks found; check --repo and --include-prefix.')
        print(f'Indexing {len(chunks)} chunks ({args.arm})...', file=sys.stderr)
        dense = DenseIndex(chunks).search(args.question, k=20)
        results = dense[:args.top_k]
        if args.arm == 'ast_hybrid':
            lexical = BM25Index(chunks).search(args.question, k=20)
            results = weighted_reciprocal_rank_fusion({'dense': dense, 'bm25': lexical},
                weights={'dense': 0.9, 'bm25': 0.1}, rrf_k=60, limit=args.top_k)
        result = answer_question(client, args.question, results, max_chars=args.context_chars,
                                 num_ctx=args.num_ctx, max_tokens=args.max_tokens)
        result.update(question=args.question, model=args.model, ollama_version=version, arm=args.arm)
        print(render_answer(result))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        return 0
    except (RuntimeError, OSError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
