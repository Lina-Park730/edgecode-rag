import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from scripts.run_development_tuning import ROOT, read_development_inputs
from edgecode.models import Chunk, SearchResult
from edgecode.fusion import weighted_reciprocal_rank_fusion

class GuardTests(unittest.TestCase):
    def test_rejects_four_questions_and_mixed_split(self):
        config=json.loads((ROOT/'config/development20.json').read_text())
        questions=json.loads((ROOT/'data/questions_dev20.json').read_text())
        with TemporaryDirectory() as d:
            qf=Path(d)/'q.json'; cf=Path(d)/'config.json'
            config['questions_path']=str(qf)
            cf.write_text(json.dumps(config))
            qf.write_text(json.dumps(questions[:4]))
            with self.assertRaisesRegex(ValueError,'exactly 20'):
                read_development_inputs(cf)
            questions[0]['split']='test'
            qf.write_text(json.dumps(questions))
            with self.assertRaisesRegex(ValueError,'split=dev'):
                read_development_inputs(cf)

    def test_zero_weight_adds_no_candidates(self):
        a=SearchResult(chunk=Chunk('a.py',1,1,'alpha','ast'),score=1,rank=1,source='dense')
        b=SearchResult(chunk=Chunk('b.py',1,1,'beta','ast'),score=1,rank=1,source='bm25')
        result=weighted_reciprocal_rank_fusion({'dense':[a],'bm25':[b]},weights={'dense':1,'bm25':0},limit=10)
        self.assertEqual([x.path for x in result],['a.py'])

if __name__=='__main__': unittest.main()
