"""First-party Strum adapter for the installed MLX pipeline. No monkeypatches.

Requires ltx-pipelines-mlx
v0.18.0 (b82b97e3d000e01361f8cfcb745244b15b89a521) and the declared MLX pack.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import math


def generation_request(image_shape, guide_shape, fps, seed, reference_strength, attention_strength):
    if len(image_shape) != 4 or image_shape[0] != 1 or image_shape[-1] != 3:
        raise ValueError('Supply exactly one RGB photographic starting frame.')
    if len(guide_shape) != 4 or guide_shape[-1] != 3:
        raise ValueError('Supply an RGB control-frame batch.')
    count, height, width, _ = guide_shape
    if count < 9 or count > 257 or (count - 1) % 8:
        raise ValueError('The guide must contain 8n+1 consecutive frames (9–257).')
    if tuple(image_shape[1:3]) != (height, width):
        raise ValueError('Starting frame and guide must have identical dimensions and framing.')
    if width % 128 or height % 128:
        raise ValueError('Use dimensions divisible by 128 so both control reductions preserve aspect ratio.')
    if not math.isfinite(fps) or not 1 <= fps <= 120:
        raise ValueError('Frame rate must be between 1 and 120.')
    if not 0 <= reference_strength <= 1 or not 0 <= attention_strength <= 1:
        raise ValueError('Control strengths must be between 0 and 1.')
    return dict(width=width, height=height, num_frames=count, frame_rate=fps,
                seed=seed, stage1_steps=8, stage2_steps=3,
                conditioning_attention_strength=attention_strength,
                upsample_only=True, refine_steps=3)


def write_lossless_video(path, frames, fps):
    import av
    import numpy as np
    from fractions import Fraction
    with av.open(str(path), 'w') as container:
        stream=container.add_stream('ffv1',rate=Fraction(str(fps)))
        stream.width=frames.shape[2];stream.height=frames.shape[1]
        stream.pix_fmt='bgr0'
        for pixels in frames:
            rgb=np.clip(np.rint(pixels.detach().cpu().numpy()*255),0,255).astype(np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():container.mux(packet)


class StrumMLXControlledVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'model':('LTXV_MLX_MODEL',),
            'starting_image':('IMAGE',),
            'control_frames':('IMAGE',),
            'prompt':('STRING',{'multiline':True,'default':''}),
            'fps':('FLOAT',{'default':48.0,'min':1.0,'max':120.0,'step':1.0}),
            'seed':('INT',{'default':7,'min':0,'max':2147483647}),
            'lora_strength':('FLOAT',{'default':1.0,'min':0.0,'max':2.0,'step':0.05}),
            'reference_strength':('FLOAT',{'default':1.0,'min':0.0,'max':1.0,'step':0.05}),
            'attention_strength':('FLOAT',{'default':1.0,'min':0.0,'max':1.0,'step':0.05}),
        }}
    RETURN_TYPES=('IMAGE','FLOAT')
    RETURN_NAMES=('video_frames','fps')
    FUNCTION='generate'
    CATEGORY='Strum/MLX'

    def generate(self,model,starting_image,control_frames,prompt,fps,seed,
                 lora_strength,reference_strength,attention_strength):
        request=generation_request(starting_image.shape,control_frames.shape,fps,seed,
                                   reference_strength,attention_strength)
        if not prompt.strip():raise ValueError('A shot description is required.')
        import gc
        import numpy as np
        import torch
        import mlx.core as mx
        from PIL import Image
        from huggingface_hub import snapshot_download
        from ltx_pipelines_mlx.ic_lora import ICLoraPipeline
        from ltx_core_mlx.model.video_vae.video_vae import _compute_decode_tiling
        from ltx_core_mlx.utils.memory import aggressive_cleanup
        import comfy.model_management
        # All model repositories must already be cached. Never download implicitly.
        gemma=snapshot_download('mlx-community/gemma-3-12b-it-4bit',local_files_only=True)
        lora=Path(snapshot_download('Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control',local_files_only=True))/'ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors'
        if not lora.is_file():raise FileNotFoundError(lora)
        model_dir=Path(model.model_dir)
        if not model_dir.is_dir():raise FileNotFoundError(model_dir)
        # The existing loader is lazy; release any previous run before backend ownership.
        model.unload()
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        pipe=None
        chunks=[]
        try:
            with TemporaryDirectory(prefix='strum_mlx_') as temp:
                folder=Path(temp)
                guide=folder/'control.mkv';first=folder/'first.png'
                write_lossless_video(guide,control_frames,fps)
                pixels=np.clip(np.rint(starting_image[0].detach().cpu().numpy()*255),0,255).astype(np.uint8)
                Image.fromarray(pixels).save(first)
                pipe=ICLoraPipeline(str(model_dir),lora_paths=[(str(lora),lora_strength)],
                                    gemma_model_id=gemma,low_memory=True)
                pipe.generate_audio=False
                video,audio=pipe.generate(prompt=prompt,
                    video_conditioning=[(str(guide),reference_strength)],
                    images=[(str(first),0,1.0)],**request)
                mx.eval(video)
                del audio
                pipe.dit=None;pipe.upsampler=None;pipe._loaded=False
                pipe.prompt_encoder.free();pipe.image_conditioner.free()
                aggressive_cleanup()
                decoder=pipe.video_decoder_block.load()
                tiling=_compute_decode_tiling(tuple(video.shape),frame_rate=fps)
                for chunk in decoder.tiled_decode(video,tiling):
                    mx.eval(chunk)
                    pixels=np.array(chunk.astype(mx.float32))[0].transpose(1,2,3,0)
                    chunks.append(torch.from_numpy(np.clip((pixels+1)*0.5,0,1)))
                frames=torch.cat(chunks,dim=0)
                expected=(request['num_frames'],request['height'],request['width'],3)
                if tuple(frames.shape)!=expected:
                    raise RuntimeError(f'Decoder returned {tuple(frames.shape)}, expected {expected}.')
                return frames,float(fps)
        finally:
            if pipe is not None:
                pipe.dit=None;pipe.upsampler=None
                for name in ('prompt_encoder','image_conditioner','audio_conditioner','video_decoder_block','audio_decoder_block'):
                    getattr(pipe,name).free()
            chunks.clear()
            gc.collect();aggressive_cleanup()


NODE_CLASS_MAPPINGS={'StrumMLXControlledVideo':StrumMLXControlledVideo}
NODE_DISPLAY_NAME_MAPPINGS={'StrumMLXControlledVideo':'Strum MLX controlled video'}
