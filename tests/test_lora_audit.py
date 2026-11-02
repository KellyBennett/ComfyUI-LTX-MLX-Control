import importlib.util
from pathlib import Path
import unittest
from ltx_core_mlx.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP

spec=importlib.util.spec_from_file_location('audit',Path(__file__).parents[1]/'lora_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class LoraAudit(unittest.TestCase):
    def test_reports_unmatched_and_incomplete_pairs_but_preserves_adaln(self):
        keys={'adaln.lora_A.weight','adaln.lora_B.weight','block.lora_A.weight',
              'unknown.lora_A.weight','unknown.lora_B.weight','alpha'}
        self.assertEqual(audit.unmatched_keys(keys,{'adaln.weight','block.weight'},LTXV_LORA_COMFY_RENAMING_MAP.apply_to_key),
                         ['alpha','block.lora_A.weight','unknown.lora_A.weight','unknown.lora_B.weight'])


class BackendMapperContract(unittest.TestCase):
    def test_real_mapper_preserves_fixture_keys_and_maps_comfy_prefixes(self):
        rename = LTXV_LORA_COMFY_RENAMING_MAP.apply_to_key
        for key in ('adaln.lora_A.weight', 'adaln.lora_B.weight', 'block.lora_A.weight',
                    'unknown.lora_A.weight', 'unknown.lora_B.weight', 'alpha'):
            self.assertEqual(rename(key), key)
        keys = {'diffusion_model.transformer_blocks.0.attn1.to_out.0.lora_A.weight',
                'diffusion_model.transformer_blocks.0.attn1.to_out.0.lora_B.weight'}
        self.assertEqual(rename(next(key for key in keys if '.lora_A.' in key)),
                         'transformer_blocks.0.attn1.to_out.lora_A.weight')
        self.assertEqual(audit.unmatched_keys(keys,
                         {'transformer_blocks.0.attn1.to_out.weight'}, rename), [])

if __name__=='__main__':unittest.main()
