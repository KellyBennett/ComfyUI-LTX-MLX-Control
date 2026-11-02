"""Exercise backend preparation without loading weights or running generation."""
from contextlib import ExitStack
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ltx_pipelines_mlx.utils import blocks
from test_contract import adapter


COMPONENTS = (blocks.PromptEncoder, blocks.ImageConditioner, blocks.AudioConditioner,
              blocks.VideoDecoder, blocks.AudioDecoder)


def safetensors_header(path, keys):
    header = {key: {'dtype': 'F32', 'shape': [0], 'data_offsets': [0, 0]} for key in keys}
    encoded = json.dumps(header).encode()
    encoded += b' ' * (-len(encoded) % 8)
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded)
    return path


class BackendPreparation(unittest.TestCase):
    def test_real_backend_is_freed_after_normal_use(self):
        with TemporaryDirectory() as temp, ExitStack() as patches:
            folder = Path(temp)
            control = safetensors_header(folder / 'control.safetensors', [])
            frees = [patches.enter_context(patch.object(component, 'free', autospec=True,
                                                       side_effect=component.free))
                     for component in COMPONENTS]
            with adapter.controlled_pipeline(folder, str(folder), control, 1.0) as pipe:
                self.assertTrue(pipe.low_memory)
                self.assertFalse(pipe.low_ram_streaming)
                self.assertFalse(pipe.generate_audio)
            for free in frees:
                free.assert_called_once()

    def test_real_character_audit_failure_still_frees_constructed_backend(self):
        with TemporaryDirectory() as temp, ExitStack() as patches:
            folder = Path(temp)
            control = safetensors_header(folder / 'control.safetensors', [])
            character = safetensors_header(folder / 'character.safetensors',
                                           ['unknown.lora_A.weight', 'unknown.lora_B.weight'])
            safetensors_header(folder / 'transformer.safetensors', ['adaln.weight'])
            frees = [patches.enter_context(patch.object(component, 'free', autospec=True,
                                                       side_effect=component.free))
                     for component in COMPONENTS]
            with self.assertRaisesRegex(ValueError, 'no complete pairs'):
                with adapter.controlled_pipeline(folder, str(folder), control, 1.0,
                                                 (character, -0.5)):
                    self.fail('An incompatible character must be rejected before generation')
            for free in frees:
                free.assert_called_once()


if __name__ == '__main__':
    unittest.main()
