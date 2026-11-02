# Character LoRAs with Union Control

`LTXMLXControlledVideo` now has an optional `character_lora` input. Connect the
`LTX2MLXLora` picker from
[`tanis2000/comfyui-ltx2-mlx`](https://github.com/tanis2000/comfyui-ltx2-mlx).
The picker returns a file path and strength. The Control node passes that pair
alongside Union Control to the existing `ICLoraPipeline`. Both adapters remain
active during controlled refinement. Existing workflows keep their behavior
when the new input is disconnected.

`example_workflows/controlled_character_video.json` shows the connection. In a
Loom Project, an existing result Product can provide
`character/ltx_lora.safetensors` through `[uses]`, just as with an ordinary LoRA
loader. Choose the actual installed Product in the picker and replace the example
prompt with one containing that character's trigger word.

Use the community pack at `d38b63abfffc32520cb31dd8b8471ba7bba52991` and pin both
`ltx-core-mlx` and `ltx-pipelines-mlx` to `v0.18.0`. The community requirements
otherwise follow Git HEAD. It also requires `soundfile`. No upstream loader
fork is needed for this connection. The existing `LTXVMLXCheckpointLoader`
still supplies the Control model handle; the community model loader is a
different pipeline type and cannot replace that input.

Character strengths must be finite and between -4 and 4. A strength of zero
skips the adapter. Fusion uses the resident path with `low_ram_streaming=False`
so keys outside transformer blocks are retained. Header checks report each
unmatched or incomplete pair and reject an adapter with no complete matches.
The backend performs tensor-shape checks during fusion.

Tests cover the connection contract, absent and disabled characters, positive
and negative strengths, invalid paths and strengths, and unmatched or incomplete
pairs including adaLN keys. An independent header and
shape audit found all 1,152 tensors in the existing FZZ spike LoRA compatible
with the cached distilled 1.1 q8 model. That audit does not demonstrate visual
quality or controlled inference. No new training or rental was performed.

Run `python -m unittest discover -s tests` in the environment with the pinned
backend and its dependencies installed. The LoRA audit tests use its real key
mapper, including a ComfyUI prefix and renamed attention projection. Preparation
tests construct the real backend with header-only temporary files and check
component release after normal use and a rejected character audit. They run no
generation and load no weights. Missing backend dependencies fail the test
instead of skipping these compatibility checks.
