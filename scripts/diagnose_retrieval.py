"""Analyze diagnostic rankings without embeddings or LLM calls (standard library only)."""
import argparse
import csv
import json
from pathlib import Path


def matches(result, gold, coverage, lines):
    chunk = result.get('chunk', result)
    path = str(chunk.get('path', '')).replace('\\', '/').removeprefix('./')
    target = str(gold.get('path', gold.get('file', ''))).replace('\\', '/').removeprefix('./')
    overlap = max(0, min(int(chunk['end_line']), int(gold['end_line'])) -
                  max(int(chunk['start_line']), int(gold['start_line'])) + 1)
    return path == target and overlap >= lines and overlap / (gold['end_line']-gold['start_line']+1) >= coverage


def first_rank(results, gold, coverage, lines):
    return next((i for i, r in enumerate(results, 1) if matches(r, gold, coverage, lines)), None)


def records(path):
    if path.suffix == '.jsonl':
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if 'diagnostic' not in row:
                raise ValueError('Old rankings have no diagnostics. Rerun with the replacement runner.')
            d = row['diagnostic']
            yield row.get('repo_id', ''), row['question_id'], row['category'], row['arm'], d['gold_spans'], d['full_results'], d['candidates'], d
    else:
        data = json.loads(path.read_text(encoding='utf-8'))
        configs = data if isinstance(data, list) else [data]
        for config in configs:
            method = config.get('method', f"AST{config.get('ast_max_lines')}_bm25_{config.get('bm25_weight')}")
            for q in config['queries']:
                if 'gold_spans' not in q or 'candidates' not in q:
                    raise ValueError('Old development export lacks diagnostics. Rerun the development runner.')
                yield '', q['question_id'], q['category'], method, q['gold_spans'], q['top_results'], q['candidates'], q


def analyze(path, category=None):
    questions, spans = [], []
    for repo, qid, cat, arm, gold, ranking, candidates, settings in records(path):
        if category and cat != category:
            continue
        if not gold:
            raise ValueError(f'{qid}: empty gold spans')
        coverage = settings.get('min_gold_coverage', .25)
        lines = settings.get('min_overlap_lines', 1)
        ranks = []
        for g in gold:
            rank = first_rank(ranking, g, coverage, lines)
            ranks.append(rank)
            dr = first_rank(candidates.get('dense', []), g, coverage, lines)
            br = first_rank(candidates.get('bm25', []), g, coverage, lines)
            if rank and rank <= 5:
                reason = 'covered_at_5'
            elif rank:
                reason = 'below_top_5'
            elif dr or br:
                reason = 'candidate_present_not_in_saved_ranking'
            else:
                reason = 'not_in_saved_candidates'
            spans.append(dict(repo_id=repo, question_id=qid, category=cat, arm=arm,
                path=g.get('path', g.get('file')), start_line=g['start_line'], end_line=g['end_line'],
                first_rank=rank or '', dense_candidate_rank=dr or '', bm25_candidate_rank=br or '',
                diagnosis=reason))
        row = dict(repo_id=repo, question_id=qid, category=cat, arm=arm, gold_span_count=len(gold))
        for k in (5, 10, 20):
            count = sum(r is not None and r <= k for r in ranks)
            row[f'recall_at_{k}'] = count / len(gold)
            row[f'all_gold_matched_at_{k}'] = int(count == len(gold))
        questions.append(row)
    if not questions:
        raise ValueError('No records matched. Check the category or input file.')
    return questions, spans


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rankings', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--category', help='Exact category name; omit to analyze all categories.')
    a = p.parse_args()
    try:
        questions, spans = analyze(a.rankings, a.category)
        a.output.mkdir(parents=True, exist_ok=False)
        write_csv(a.output/'question_coverage.csv', questions)
        write_csv(a.output/'gold_span_diagnostics.csv', spans)
        summary = []
        for arm, category in sorted({(r['arm'], r['category']) for r in questions}):
            group = [r for r in questions if r['arm'] == arm and r['category'] == category]
            row = dict(arm=arm, category=category, question_count=len(group))
            for k in (5, 10, 20):
                row[f'mean_recall_at_{k}'] = sum(r[f'recall_at_{k}'] for r in group)/len(group)
                row[f'all_gold_matched_count_at_{k}'] = sum(r[f'all_gold_matched_at_{k}'] for r in group)
            summary.append(row)
        write_csv(a.output/'category_summary.csv', summary)
        (a.output/'README.txt').write_text(
            'Coverage uses the existing file/line overlap rule, not semantic correctness.\n'
            'not_in_saved_candidates does NOT prove the chunk is absent from the corpus.\n'
            'Inspect source/gold annotations before inferring fragmentation or missing call relationships.\n'
            'No statistical significance or answer-accuracy claim is made.\n', encoding='utf-8')
        print(f'Analyzed {len(questions)} query-method records. Outputs: {a.output}')
    except (ValueError, KeyError, OSError) as error:
        p.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
