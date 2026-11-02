import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('audit',Path(__file__).parents[1]/'lora_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class LoraAudit(unittest.TestCase):
    def test_reports_unmatched_and_incomplete_pairs_but_preserves_adaln(self):
        keys={'adaln.lora_A.weight','adaln.lora_B.weight','block.lora_A.weight',
              'unknown.lora_A.weight','unknown.lora_B.weight','alpha'}
        self.assertEqual(audit.unmatched_keys(keys,{'adaln.weight','block.weight'},lambda x:x),
                         ['alpha','block.lora_A.weight','unknown.lora_A.weight','unknown.lora_B.weight'])

if __name__=='__main__':unittest.main()
