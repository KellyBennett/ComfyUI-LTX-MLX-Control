# ComfyUI Strum MLX

This node connects ComfyUI to the IC-LoRA pipeline in `ltx-pipelines-mlx`. It passes the selected frame rate into sampling and retains reference control during the final refinement. The defaults use 48 fps and 8 sampling steps followed by 3 refinement steps.

Connect an MLX model from `LTXVMLXCheckpointLoader`, one photographic starting image, and an RGB batch of control frames. The starting image and control frames must have the same dimensions. Width and height must be divisible by 128, and the frame count must be 8n+1. The node returns RGB video frames and their frame rate. Connect both to `CreateVideo` for a silent video.

The first generation stage runs at half the output size. Union Control uses a reference at half that stage's size. During refinement, the reference is encoded again at half the final output size. Inspect the reduced guides before spending compute on small moving details.

## Dependencies

The node uses `ComfyUI-LTXVideo-mlx` at commit `0994c235e22204b4d36e5acdda533bc116dd9267`, with `ltx-core-mlx` and `ltx-pipelines-mlx` from `dgrauet/ltx-2-mlx` v0.18.0, commit `b82b97e3d000e01361f8cfcb745244b15b89a521`. The existing pack and backend remain unchanged.

The following model repositories must already be cached:

- `dgrauet/ltx-2.3-mlx-q8`
- `mlx-community/gemma-3-12b-it-4bit`
- `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control`

The adapter resolves the text encoder and control model from the local cache. It sends no media to an external service. Temporary control video uses lossless FFV1 encoding, and temporary files are removed after execution.

## Validation

Run `python -m unittest discover -s tests` in the ComfyUI Python environment. These checks cover input geometry, frame counts, timing parameters and lossless control transport. They load no model. Generation quality and peak memory still require a real test run.
