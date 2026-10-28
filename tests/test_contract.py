import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from fractions import Fraction
import unittest
import json
import numpy as np
import torch
import av
spec=importlib.util.spec_from_file_location('adapter',Path(__file__).resolve().parents[1]/'__init__.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)

class InputContract(unittest.TestCase):
    def test_fast_motion_request_keeps_timing_and_control(self):
        r=adapter.generation_request((1,1152,2048,3),(33,1152,2048,3),48,7,1,1)
        self.assertEqual((r['num_frames'],r['frame_rate'],r['stage1_steps'],r['refine_steps']),(33,48,8,3))
        self.assertTrue(r['upsample_only'])

    def test_rejects_inputs_that_would_change_framing_or_duration(self):
        cases=[((1,1152,2048,3),(32,1152,2048,3)),((1,576,1024,3),(33,1152,2048,3)),((2,1152,2048,3),(33,1152,2048,3)),((1,1080,1920,3),(33,1080,1920,3))]
        for image,guide in cases:
            with self.subTest(image=image,guide=guide),self.assertRaises(ValueError):
                adapter.generation_request(image,guide,48,7,1,1)

    def test_transport_preserves_each_control_pixel_and_frame(self):
        pixels=np.random.default_rng(7).integers(0,256,(9,32,64,3),dtype=np.uint8)
        with TemporaryDirectory() as temp:
            p=Path(temp)/'control.mkv'
            adapter.write_lossless_video(p,torch.from_numpy(pixels.astype(np.float32)/255),48)
            with av.open(str(p)) as c:
                self.assertEqual(c.streams.video[0].average_rate,Fraction(48))
                actual=np.stack([f.to_ndarray(format='rgb24') for f in c.decode(video=0)])
        np.testing.assert_array_equal(actual,pixels)

class OfflineCache(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache=Path(self.temp.name)
        self.gemma=self.snapshot('mlx-community/gemma-3-12b-it-4bit',
            ['config.json','tokenizer.json','tokenizer.model','model.safetensors'],
            ['README.md','.gitattributes'])
        self.filename='ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors'
        self.lora=self.snapshot('Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control',
            [self.filename],['README.md','.gitattributes','example.mp4'])/self.filename

    def snapshot(self,repo,available,missing):
        root=self.cache/('models--'+repo.replace('/','--'))
        commit='a'*40
        folder=root/'snapshots'/commit
        folder.mkdir(parents=True)
        (root/'refs').mkdir()
        (root/'refs/main').write_text(commit)
        (root/'trees').mkdir()
        # Hub's cached repository listing includes intentionally uncached docs/media.
        tree={'format_version':1,'files':{name:{'size':1,'blob_id':'b'*40}
              for name in available+missing}}
        (root/'trees'/f'{commit}.json').write_text(json.dumps(tree))
        for name in available:(folder/name).write_bytes(b'x')
        return folder

    def test_resolves_inference_assets_without_docs_or_example_media(self):
        gemma,lora=adapter.cached_model_paths(self.cache)
        self.assertTrue(Path(gemma).samefile(self.gemma))
        self.assertTrue(lora.samefile(self.lora))

    def test_missing_gemma_weights_still_fail(self):
        from huggingface_hub.errors import IncompleteSnapshotError
        (self.gemma/'model.safetensors').unlink()
        with self.assertRaises(IncompleteSnapshotError):
            adapter.cached_model_paths(self.cache)

    def test_missing_control_weights_still_fail(self):
        from huggingface_hub.errors import LocalEntryNotFoundError
        self.lora.unlink()
        with self.assertRaises(LocalEntryNotFoundError):
            adapter.cached_model_paths(self.cache)


if __name__=='__main__':unittest.main()
