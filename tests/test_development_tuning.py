import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Offline end-to-end development grid-search test; fake embedder is NEVER used for report benchmark."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
import numpy as np
from scripts.run_development_tuning import run_tuning, _score_key

class FakeEmbedder:
    def embed(self, texts, batch_size=32):
        return (self._e(t) for t in texts)
    def query_embed(self, texts, batch_size=32):
        return (self._e(t) for t in texts)
    def _e(self,t):
        return np.asarray([t.lower().count('alpha'),t.lower().count('beta'),1],dtype=np.float32)

class TuningTests(unittest.TestCase):
    def test_selects_and_exports_multiple_weights(self):
        with TemporaryDirectory() as d:
            base=Path(d)
            repo=base/'repo'; p=repo/'pkg';p.mkdir(parents=True)
            (p/'values.py').write_text('def alpha():\n    return "alpha"\n\ndef beta():\n    return "beta"\n',encoding='utf8')
            qs=[]
            for i in range(20):
                alpha=(i%2==0)
                qs.append({'id':f'q{i:02d}','question': 'Where is alpha?' if alpha else 'Where is beta?',
                           'split':'dev','category':'test','gold_spans':[{'path':'pkg/values.py',
                           'start_line':1 if alpha else 4,'end_line':2 if alpha else 5}]})
            # read_development_inputs resolves questions relative to project root,
            # so use an absolute questions filename via config.
            qf=base/'questions.json';qf.write_text(json.dumps(qs))
            cfg={'questions_path':str(qf),'repository':{'path':str(repo), 'include_prefixes':['pkg/']},
                 'expected_question_count':20,
                 'chunking':{'fixed':{'max_lines':3,'overlap_lines':0},
                             'ast':{'max_lines':5,'overlap_lines':1}},
                 'retrieval':{'embedding_model':'fake','embedding_batch_size':2,
                   'candidate_k':20,'evaluation_k':[3,5,10], 'hybrid':{'rrf_k':60}},
                 'evaluation':{'min_gold_coverage':0.25,'min_overlap_lines':1}}
            cf=base/'config.json';cf.write_text(json.dumps(cfg))
            output=base/'output'
            result=run_tuning(cf,output,embedder=FakeEmbedder(),
                              ast_sizes=(5,8),alphas=(0,.5,1),verbose=False)
            self.assertEqual(result['dev_question_count'],20)
            self.assertTrue((output/'selected_config.json').is_file())
            import csv
            with (output/'development_tuning.csv').open() as h:
                rows=list(csv.DictReader(h))
            self.assertEqual(len(rows),6)
            self.assertEqual({float(r['bm25_weight']) for r in rows},{0.,.5,1.})
            with (output/'development_query_metrics.csv').open() as h:
                details=list(csv.DictReader(h))
            self.assertEqual(len(details),2*20*3)
            with (output/'development_all_query_metrics.csv').open() as h:
                all_rows=list(csv.DictReader(h))
            self.assertEqual(len(all_rows),6*20*3)
            self.assertEqual({int(x['ast_max_lines']) for x in all_rows},{5,8})
            with (output/'development_category_metrics.csv').open() as h:
                cats=list(csv.DictReader(h))
            self.assertEqual(len(cats),6*3)
            self.assertTrue(all(int(x['question_count'])==20 for x in cats))
            self.assertEqual(len(json.loads((output/'development_all_rankings.json').read_text())),6)
            self.assertEqual(result['selected_configuration']['status'],'development_tuned_NOT_HELD_OUT')

    def test_revised_dataset_has_twenty_balanced_dev_questions(self):
        from scripts.run_development_tuning import ROOT, read_development_inputs
        from collections import Counter
        config, questions, repo=read_development_inputs(ROOT/'config/development20.json')
        self.assertEqual(len(questions),20)
        self.assertEqual(set(Counter(q['category'] for q in questions).values()),{4})
        self.assertTrue(all(q['split']=='dev' for q in questions))

    def test_tie_break_prefers_simple_central_values(self):
        def row(w,size):
            return {'mrr_at_5':.8,'recall_at_5':.9,'hit_rate_at_5':1.,'precision_at_5':.3,
                    'bm25_weight':w,'ast_max_lines':size}
        self.assertLess(_score_key(row(.5,80)),_score_key(row(.1,60)))

if __name__=='__main__': unittest.main()
