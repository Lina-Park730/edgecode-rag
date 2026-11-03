#!/usr/bin/env python3
"""Development-only AST chunk-size and BM25/Dense RRF grid search.

Use only on the DEVELOPMENT questions, never to claim held-out performance.
The dense index and each query's BM25/dense candidates are computed once per
AST size; eleven alpha values re-use those candidates without re-embedding.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from statistics import mean
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edgecode.bm25 import BM25Index
from edgecode.chunking import chunk_repository
from edgecode.dense import DenseIndex
from edgecode.evaluation import GoldSpan, retrieval_metrics_at_k
from edgecode.fusion import weighted_reciprocal_rank_fusion

DEFAULT_ALPHAS = tuple(i / 10 for i in range(11))
DEFAULT_SIZES = (60, 80, 120)


def read_development_inputs(config_file: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    config = json.loads(config_file.read_text(encoding='utf-8'))
    data_path = (ROOT / config['questions_path']).resolve()
    questions = json.loads(data_path.read_text(encoding='utf-8'))
    expected_count = int(config.get('expected_question_count', 20))
    if not isinstance(questions, list) or len(questions) != expected_count:
        raise ValueError(f'Development dataset must contain exactly {expected_count} questions')
    if len(set(q['id'] for q in questions)) != len(questions):
        raise ValueError('Duplicate question ID in development dataset')
    if any(q.get('split') != 'dev' for q in questions):
        raise ValueError('All development questions must have split=dev; do not mix held-out questions')
    for q in questions:
        if not str(q.get('question','')).strip() or not q.get('gold_spans'):
            raise ValueError(f"Missing question text or gold spans: {q.get('id')}")
    repo = (ROOT / config['repository']['path']).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f'Missing source repository: {repo}\nDownload Click 8.1.7 first; see README.')
    prefix = config['repository'].get('include_prefixes',['src/click/'])
    for q in questions:
        for raw in q['gold_spans']:
            GoldSpan.from_dict(raw)
            filename = (repo / raw['path']).resolve()
            if not filename.is_relative_to(repo) or not filename.is_file():
                raise ValueError(f"Missing or unsafe gold source path in {q['id']}: {raw['path']}")
            num_lines = len(filename.read_text(encoding='utf-8-sig').splitlines())
            if int(raw['end_line']) > num_lines:
                raise ValueError(f"Gold span past end of file in {q['id']}: {raw['path']}")
            if not any(raw['path'].startswith(p.rstrip('/') + '/') or raw['path']==p.rstrip('/') for p in prefix):
                raise ValueError(f"Gold path outside indexed corpus: {q['id']}: {raw['path']}")
    return config, questions, repo


def _eval(rankings: list[Any], questions: list[dict[str,Any]], evaluation_ks: tuple[int,...],
          min_gold_coverage: float, min_overlap_lines: int) -> tuple[dict[str,float],list[dict[str, Any]]]:
    if len(rankings) != len(questions):
        raise ValueError('Ranking count differs from development question count')
    records=[]
    for q, ranked in zip(questions,rankings):
        gold=[GoldSpan.from_dict(span) for span in q['gold_spans']]
        metrics={k:retrieval_metrics_at_k(ranked,gold,k=k,
                      min_overlap_lines=min_overlap_lines,
                      min_gold_coverage=min_gold_coverage) for k in evaluation_ks}
        records.append({'question_id':q['id'],'category':q.get('category',''),
             'metrics':{str(k):m.to_dict() for k,m in metrics.items()},
             'gold_spans':q['gold_spans'],
             'min_gold_coverage':min_gold_coverage,'min_overlap_lines':min_overlap_lines,
             'top_results':[{'path':r.path,'start_line':r.start_line,'end_line':r.end_line,
                             'symbol':r.chunk.symbol,'score':r.score} for r in ranked]})
    summary={}
    for k in evaluation_ks:
        for name,attribute in [('precision','precision'),('recall','recall'),
                               ('mrr','reciprocal_rank'),('hit_rate','hit')]:
            summary[f'{name}_at_{k}']=round(mean(getattr(retrieval_metrics_at_k(
                rankings[i],[GoldSpan.from_dict(span) for span in q['gold_spans']], k=k,
                min_overlap_lines=min_overlap_lines,min_gold_coverage=min_gold_coverage),attribute)
                     for i,q in enumerate(questions)),8)
    return summary,records


def _score_key(row: Mapping[str,Any]) -> tuple[Any,...]:
    """Deterministic choice: MRR@5 primary; then Recall, Hit, precision.

    Near-equal floating-point values use 8-decimal rounding, then choose weights
    near 0.5 and sizes near 80 to avoid arbitrary extreme tie choices.
    """
    return (-round(row['mrr_at_5'],8),-round(row['recall_at_5'],8),
            -round(row['hit_rate_at_5'],8),-round(row['precision_at_5'],8),
            abs(row['bm25_weight']-0.5),abs(row['ast_max_lines']-80),
            row['ast_max_lines'],row['bm25_weight'])


def _save_json(path: Path, content: Any) -> None:
    path.write_text(json.dumps(content,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def _save_csv(path: Path, rows: list[dict[str,Any]]) -> None:
    if not rows: raise ValueError('No rows to export')
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_tuning(config_file: Path, output_dir: Path, *, ast_sizes: tuple[int,...]=DEFAULT_SIZES,
               alphas: tuple[float,...]=DEFAULT_ALPHAS, embedder: Any=None, verbose: bool=True) -> dict[str,Any]:
    """Run 3 chunk sizes x 11 weights using reusable embeddings / candidates.

    Provide injected embedder in unit tests only; production CLI uses FastEmbed.
    """
    config, questions, repo = read_development_inputs(config_file)
    retrieval=config['retrieval']
    chunking=config['chunking']
    ks=tuple(sorted(set(int(x) for x in retrieval.get('evaluation_k',[3,5,10]))))
    if ks!=(3,5,10): raise ValueError('Expected evaluation_k to be [3, 5, 10]')
    candidate_k=int(retrieval.get('candidate_k',20))
    if candidate_k < 10: raise ValueError('candidate_k must be >= 10')
    if any(not 0<=a<=1 for a in alphas) or not alphas:
        raise ValueError('Alpha values must all fall in [0, 1]')
    if not ast_sizes or any(s <= int(chunking['ast']['overlap_lines']) for s in ast_sizes):
        raise ValueError('AST size must be greater than overlap')
    if verbose: print(f'{len(questions)} DEVELOPMENT questions | AST sizes={ast_sizes} | alphas={alphas}\n'
                      'Results from this run are for hyperparameter selection ONLY.',flush=True)

    start_all=time.perf_counter()
    if embedder is None:
        from fastembed import TextEmbedding
        if verbose: print('Loading pretrained BGE embedding model (may download once)...',flush=True)
        embedder=TextEmbedding(model_name=retrieval['embedding_model'])

    root_prefixes=config['repository'].get('include_prefixes',['src/click/'])
    batch=int(retrieval['embedding_batch_size'])
    t0=time.perf_counter()
    fixed_chunks=chunk_repository(repo,strategy='fixed',
        max_lines=int(chunking['fixed']['max_lines']),
        overlap_lines=int(chunking['fixed']['overlap_lines']),
        include_prefixes=root_prefixes)
    fixed_dense=DenseIndex(fixed_chunks,model_name=retrieval['embedding_model'],
                          embedder=embedder,batch_size=batch)
    fixed_build_seconds=time.perf_counter()-t0
    if verbose: print(f'Baseline: {len(fixed_chunks)} fixed chunks; build {fixed_build_seconds:.1f}s',flush=True)
    baseline_rankings=[fixed_dense.search(q['question'],k=max(candidate_k,max(ks))) for q in questions]
    evaluation=config['evaluation']
    threshold=float(evaluation.get('min_gold_coverage',0.25))
    overlap=int(evaluation.get('min_overlap_lines',1))
    baseline_metrics,baseline_details=_eval(baseline_rankings,questions,ks,threshold,overlap)
    tuning=[]
    all_query_rows=[]
    category_rows=[]
    ranking_cache={}
    build_info={}
    for size in ast_sizes:
        t0=time.perf_counter()
        chunks=chunk_repository(repo,strategy='ast',max_lines=size,
             overlap_lines=int(chunking['ast']['overlap_lines']),include_prefixes=root_prefixes)
        bm25=BM25Index(chunks)
        dense=DenseIndex(chunks,model_name=retrieval['embedding_model'],
                         embedder=embedder,batch_size=batch)
        build_seconds=time.perf_counter()-t0
        if verbose: print(f'AST {size}: {len(chunks)} chunks; index build {build_seconds:.1f}s',flush=True)
        # Search independently ONCE per question, then cache both top-20 lists.
        candidate_pairs=[]
        for q in questions:
            text=q['question']
            candidate_pairs.append((bm25.search(text,k=candidate_k),dense.search(text,k=candidate_k)))
        build_info[str(size)]={'chunk_count':len(chunks),'build_seconds':round(build_seconds,3)}
        for alpha in alphas:
            rankings=[weighted_reciprocal_rank_fusion({'bm25':b,'dense':d},
                        weights={'bm25':alpha,'dense':1-alpha},
                        rrf_k=int(retrieval['hybrid']['rrf_k']),limit=None)
                      for b,d in candidate_pairs]
            summary,details=_eval(rankings,questions,ks,threshold,overlap)
            for detail,(b,d) in zip(details,candidate_pairs):
                detail['candidates']={'bm25':[r.to_dict() for r in b],
                                      'dense':[r.to_dict() for r in d]}
            for detail in details:
                for k in ks:
                    all_query_rows.append({'ast_max_lines':size,'bm25_weight':alpha,
                        'dense_weight':1-alpha,'question_id':detail['question_id'],
                        'category':detail['category'],**detail['metrics'][str(k)]})
            for category in sorted({x['category'] for x in details}):
                subset=[x for x in details if x['category']==category]
                for k in ks:
                    category_rows.append({'ast_max_lines':size,'bm25_weight':alpha,
                        'dense_weight':1-alpha,'category':category,'question_count':len(subset),'k':k,
                        **{name:round(mean(x['metrics'][str(k)][name] for x in subset),8)
                        for name in ['mrr_at_k','recall_at_k','precision_at_k','hit_at_k']}})
            key=(size,round(alpha,8))
            ranking_cache[key]=details
            tuning.append({'ast_max_lines':size,'bm25_weight':round(alpha,2),
                           'dense_weight':round(1-alpha,2),'question_count':len(questions),
                           'chunk_count':len(chunks),**summary})
            if verbose: print(f'  alpha={alpha:.1f}: MRR@5={summary["mrr_at_5"]:.4f} '
                              f'Recall@5={summary["recall_at_5"]:.4f}',flush=True)
    best=min(tuning,key=_score_key)
    selected=ranking_cache[(best['ast_max_lines'],round(best['bm25_weight'],8))]
    output_dir.mkdir(parents=True,exist_ok=True)
    tuning_sorted=sorted(tuning,key=_score_key)
    _save_csv(output_dir/'development_all_query_metrics.csv',all_query_rows)
    _save_csv(output_dir/'development_category_metrics.csv',category_rows)
    comparisons=[]
    for size in ast_sizes:
        pure=next((x for x in tuning if x['ast_max_lines']==size and x['bm25_weight']==0),None)
        mixed=[x for x in tuning if x['ast_max_lines']==size and 0<x['bm25_weight']<1]
        if pure and mixed:
            hybrid=min(mixed,key=_score_key)
            comparisons.append({'ast_max_lines':size,'best_hybrid_bm25_weight':hybrid['bm25_weight'],
                'dense_mrr_at_5':pure['mrr_at_5'],'hybrid_mrr_at_5':hybrid['mrr_at_5'],
                'hybrid_minus_dense_mrr_at_5':round(hybrid['mrr_at_5']-pure['mrr_at_5'],8),
                'dense_recall_at_5':pure['recall_at_5'],'hybrid_recall_at_5':hybrid['recall_at_5']})
    if comparisons: _save_csv(output_dir/'development_hybrid_vs_dense.csv',comparisons)
    _save_json(output_dir/'development_all_rankings.json',[
        {'ast_max_lines':size,'bm25_weight':alpha,'queries':details}
        for (size,alpha),details in ranking_cache.items()])
    _save_json(output_dir/'development_dataset_snapshot.json',questions)
    _save_json(output_dir/'development_config_snapshot.json',config)
    _save_csv(output_dir/'development_tuning.csv',tuning)
    _save_csv(output_dir/'development_ranked_configs.csv',tuning_sorted)
    _save_csv(output_dir/'development_baseline.csv',[{'method':'fixed_dense',
            'fixed_max_lines':chunking['fixed']['max_lines'],'chunk_count':len(fixed_chunks),
            'question_count':len(questions),**baseline_metrics}])
    detail_rows=[]
    for method,detail in [('fixed_dense',baseline_details),('ast_hybrid_best_dev',selected)]:
        for x in detail:
            for k in ks:
                mm=x['metrics'][str(k)]
                detail_rows.append({'method':method,'question_id':x['question_id'],
                 'category':x['category'],'k':k,'precision_at_k':mm['precision_at_k'],
                 'recall_at_k':mm['recall_at_k'],'mrr_at_k':mm['mrr_at_k'],
                 'hit_at_k':mm['hit_at_k']})
    _save_csv(output_dir/'development_query_metrics.csv',detail_rows)
    selected_config={'experiment_name':'edgecode_dev20_selected_config',
       'status':'development_tuned_NOT_HELD_OUT',
       'source_repository':'pallets/click tag 8.1.7',
       'selection_objective':'maximize mean development MRR@5; tie-break Recall@5, Hit@5, Precision@5; then closeness to 0.5/80',
       'ast_max_lines':best['ast_max_lines'],
       'ast_overlap_lines':chunking['ast']['overlap_lines'],
       'bm25_weight':best['bm25_weight'],'dense_weight':best['dense_weight'],
       'candidate_k':candidate_k,'evaluation_k':list(ks),
       'model':retrieval['embedding_model'],'rrf_k':retrieval['hybrid']['rrf_k'],
       'baseline_fixed_max_lines':chunking['fixed']['max_lines'],
       'min_gold_coverage':threshold,'dev_question_count':len(questions)}
    _save_json(output_dir/'selected_config.json',selected_config)
    report={'dataset_sha256':hashlib.sha256((ROOT / config['questions_path']).read_bytes()).hexdigest(),
       'dev_question_count':len(questions),'eval_ks':list(ks),'candidate_k':candidate_k,
       'weights':list(alphas),'ast_sizes':list(ast_sizes),
       'baseline_fixed_dense':{'metrics':baseline_metrics,'chunks':len(fixed_chunks),'build_seconds':round(fixed_build_seconds,3)},
       'build_by_ast_size':build_info,'best_dev_metrics':best,
       'selected_configuration':selected_config,'elapsed_total_seconds':round(time.perf_counter()-start_all,3),
       'warnings':['Development-set metrics; not held-out generalization estimates.',
                   'Gold annotations require independent human validation.',
                   'MRR@5 is optimized on these same development questions and will be optimistically biased.',
                   'This experiment evaluates neither LLM answer quality nor parent-context retrieval.']}
    _save_json(output_dir/'development_results.json',report)
    _save_json(output_dir/'selected_rankings.json',{'method':'ast_hybrid_best_dev','queries':selected})
    if verbose:
        print('\nBEST DEVELOPMENT SETTINGS (not final test performance):',flush=True)
        print(json.dumps({'ast_max_lines':best['ast_max_lines'],
                          'bm25_weight':best['bm25_weight'],'dense_weight':best['dense_weight'],
                          'dev_mrr_at_5':best['mrr_at_5']},indent=2))
        print('Files saved to:',output_dir,flush=True)
        print('Important: validate gold spans, freeze settings, then evaluate on a NEW held-out set.',flush=True)
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description='EdgeCode: 20-question development grid search (not final test).')
    parser.add_argument('--config',type=Path,default=ROOT/'config/development20.json')
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--quick',action='store_true',help='Only AST size 80; still test all 11 weights on 20 questions.')
    args=parser.parse_args()
    run_tuning(args.config,args.output or ROOT/('outputs/development20_quick' if args.quick else 'outputs/development20'),
               ast_sizes=(80,) if args.quick else DEFAULT_SIZES)


if __name__=='__main__':
    main()
