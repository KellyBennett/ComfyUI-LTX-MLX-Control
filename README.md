# LTX 2.3 MLX video with frame-rate control

`ComfyUI-LTX-MLX-Control` provides **LTX 2.3 MLX IC-LoRA Video (Frame Rate + Controlled Refinement)** for Apple Silicon. Use it to animate a starting image with aligned canny or depth guides when motion timing and control during refinement are needed.

The node connects ComfyUI to the IC-LoRA pipeline in `ltx-pipelines-mlx`. It passes the selected frame rate into sampling and retains reference control during the final refinement. The defaults use 48 fps and 8 sampling steps followed by 3 refinement steps.

Connect an MLX model from `LTXVMLXCheckpointLoader`, one photographic starting image, and an RGB batch of control frames. The starting image and control frames must have the same dimensions. Width and height must be divisible by 128, and the frame count must be 8n+1. The node returns RGB video frames and their frame rate. Connect both to `CreateVideo` for a silent video.

The first generation stage runs at half the output size. Union Control uses a reference at half that stage's size. During refinement, the reference is encoded again at half the final output size. Inspect the reduced guides before spending compute on small moving details.

## Example workflow

Open [controlled_video.json](example_workflows/controlled_video.json) in ComfyUI. Supply your own `starting_image.png` and `control_video.mp4`; no example media is included. The video must contain 9 to 257 frames with an 8n+1 count, and its dimensions must match the starting image. Both dimensions must be divisible by 128.

The example extracts the video frames and sampling rate, detects canny edges at 0.1/0.3, and passes the edges into the adapter. The adapter's frame-rate output connects to `CreateVideo`, so sampling and playback use the same timing. For precomputed depth or edge guides, bypass the Canny node. The example uses the video loader's frame rate, even when it differs from the adapter's default of 48 fps.

Replace the example prompt with a description of your inputs. The example does not run until you supply its image and video. Its wiring has been checked; generation quality and memory use have not been tested.

## Finding the node in Loom

Within a Project that declares this pack, `list_nodes` searches the node's display name and category. Search for `LTX`, `MLX`, `IC-LoRA`, `Frame Rate`, or `Controlled Refinement`. Call `describe_nodes` with `LTXMLXControlledVideo` to read its purpose, input constraints and tooltips.

In another Project on the same machine, `list_installed_node_packs` lists `ComfyUI-LTX-MLX-Control` and its installed commits. Declare the pack at an exact commit together with its upstream dependencies, then let ComfyUI reload. The pack inventory currently exposes names and versions rather than capability descriptions, so the richer metadata becomes available after the Project declares the pack. This repository is public on GitHub and is not listed in the Comfy Registry.

## Installation and dependencies

The node uses `ComfyUI-LTXVideo-mlx` at commit `0994c235e22204b4d36e5acdda533bc116dd9267`, with `ltx-core-mlx` and `ltx-pipelines-mlx` from `dgrauet/ltx-2-mlx` v0.18.0, commit `b82b97e3d000e01361f8cfcb745244b15b89a521`. The existing pack and backend remain unchanged.

Install this repository under `ComfyUI/custom_nodes/ComfyUI-LTX-MLX-Control`, alongside the upstream `ComfyUI-LTXVideo-mlx` pack. Use the ComfyUI Python environment to install the backend versions below, then restart ComfyUI:

```sh
python -m pip install 'ltx-core-mlx @ git+https://github.com/dgrauet/ltx-2-mlx@v0.18.0#subdirectory=packages/ltx-core-mlx' 'ltx-pipelines-mlx @ git+https://github.com/dgrauet/ltx-2-mlx@v0.18.0#subdirectory=packages/ltx-pipelines-mlx'
```

In Loom, declare both packs at exact commits in your own Project's `loom.toml`, with those two backend requirements as overrides on `ComfyUI-LTXVideo-mlx`. Use `https://github.com/KellyBennett/ComfyUI-LTX-MLX-Control` for this pack. Any Project can use it with its own images and guides. Loom is optional; the node and example also work in ComfyUI directly.

The following model repositories must already be cached:

- `dgrauet/ltx-2.3-mlx-q8`
- `mlx-community/gemma-3-12b-it-4bit`
- `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control`

The adapter resolves the text encoder's JSON, safetensors and tokenizer model files and the exact Union Control weight file from the local cache. Repository documentation, Git metadata and example media do not need to be cached. Missing inference assets still fail without downloading. It sends no media to an external service. Temporary control video uses lossless FFV1 encoding, and temporary files are removed after execution.

## Validation

Run `python -m unittest discover -s tests` in the ComfyUI Python environment. These checks cover input geometry, frame counts, timing parameters and lossless control transport. They load no model. Generation quality and peak memory still require a real test run.
