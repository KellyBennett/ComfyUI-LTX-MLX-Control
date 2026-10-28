"""ComfyUI adapter for LTX MLX video generation with IC-LoRA control.

Requires ltx-pipelines-mlx
v0.18.0 (b82b97e3d000e01361f8cfcb745244b15b89a521) and the declared MLX pack.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import math


def cached_model_paths(cache_dir=None):
    """Resolve inference assets offline without requiring repository documentation."""
    from huggingface_hub import hf_hub_download, snapshot_download
    gemma = snapshot_download(
        'mlx-community/gemma-3-12b-it-4bit',
        allow_patterns=['*.json', '*.safetensors', '*.model'],
        cache_dir=cache_dir, local_files_only=True,
    )
    lora = hf_hub_download(
        'Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control',
        filename='ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors',
        cache_dir=cache_dir, local_files_only=True,
    )
    return gemma, Path(lora)


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


class LTXMLXControlledVideo:
    DESCRIPTION = (
        'Generate silent LTX 2.3 video on Apple Silicon with MLX IC-LoRA Union Control. '
        'Use when a starting image and aligned canny or depth guides must control motion, '
        'with an explicit sampling frame rate and control retained during refinement. '
        'Runs 8 sampling steps at half resolution, then 3 controlled refinement steps. '
        'Accepts one RGB starting image and 9 to 257 RGB guide frames (8n+1), all at the '
        'same dimensions, with width and height divisible by 128. Output size and frame '
        'count come from the guides. Returns RGB frames and fps for CreateVideo; no audio. '
        'Requires LTXVMLXCheckpointLoader, the pinned MLX v0.18.0 backend, and cached '
        'Gemma and Union Control models. See example_workflows/controlled_video.json '
        'in ComfyUI-LTX-MLX-Control. Generation quality and peak memory remain untested.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'model':('LTXV_MLX_MODEL',{'tooltip':'Connect LTXVMLXCheckpointLoader. Use the cached dgrauet/ltx-2.3-mlx-q8 model and low_vram memory profile.'}),
            'starting_image':('IMAGE',{'tooltip':'One RGB image matching the first guide frame, with identical dimensions and framing.'}),
            'control_frames':('IMAGE',{'tooltip':'Aligned RGB canny or depth guides, already preprocessed. Supply 8n+1 consecutive frames, from 9 to 257. Their dimensions set the output size; each dimension must be divisible by 128.'}),
            'prompt':('STRING',{'multiline':True,'default':'','tooltip':'Describe the subject, appearance and intended motion shown by the input image and guides.'}),
            'fps':('FLOAT',{'default':48.0,'min':1.0,'max':120.0,'step':1.0,'tooltip':'Frame rate used inside sampling and returned for playback. Match the guide sampling rate and connect the fps output to CreateVideo.'}),
            'seed':('INT',{'default':7,'min':0,'max':2147483647,'tooltip':'Keep fixed when comparing control strengths.'}),
            'lora_strength':('FLOAT',{'default':1.0,'min':0.0,'max':2.0,'step':0.05,'tooltip':'Weight of the cached Union Control IC-LoRA. It remains applied during refinement.'}),
            'reference_strength':('FLOAT',{'default':1.0,'min':0.0,'max':1.0,'step':0.05,'tooltip':'Strength of the reference video conditioning in the MLX pipeline.'}),
            'attention_strength':('FLOAT',{'default':1.0,'min':0.0,'max':1.0,'step':0.05,'tooltip':'Scales attention to the control reference. 0 ignores it; 1 uses full conditioning attention.'}),
        }}
    RETURN_TYPES=('IMAGE','FLOAT')
    RETURN_NAMES=('video_frames','fps')
    OUTPUT_TOOLTIPS=('Generated RGB frames at the guide dimensions and frame count.',
                     'The sampling frame rate. Connect to CreateVideo to preserve timing.')
    FUNCTION='generate'
    CATEGORY='video/LTX/MLX/IC-LoRA'

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
        from ltx_pipelines_mlx.ic_lora import ICLoraPipeline
        from ltx_core_mlx.model.video_vae.video_vae import _compute_decode_tiling
        from ltx_core_mlx.utils.memory import aggressive_cleanup
        import comfy.model_management
        # All model repositories must already be cached. Never download implicitly.
        gemma,lora=cached_model_paths()
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
            with TemporaryDirectory(prefix='ltx_mlx_control_') as temp:
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


NODE_CLASS_MAPPINGS={'LTXMLXControlledVideo':LTXMLXControlledVideo}
NODE_DISPLAY_NAME_MAPPINGS={'LTXMLXControlledVideo':'LTX 2.3 MLX IC-LoRA Video (Frame Rate + Controlled Refinement)'}
