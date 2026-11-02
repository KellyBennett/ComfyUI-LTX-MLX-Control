"""Report adapters the backend would silently skip, without loading tensors."""
import json
import logging
from pathlib import Path
import struct


def header_keys(path):
    with Path(path).open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors header")
        length, = struct.unpack("<Q", prefix)
        if not 2 <= length <= 100_000_000:
            raise ValueError("Invalid safetensors header length")
        header = json.loads(stream.read(length))
    return set(header) - {"__metadata__"}


def unmatched_keys(keys, weights, rename):
    mapped = {key: rename(key) for key in keys}
    available = set(mapped.values())
    missing = []
    for key, target in mapped.items():
        if target is None or not key.endswith((".lora_A.weight", ".lora_B.weight")):
            missing.append(key)
            continue
        base, side = target.rsplit(".lora_", 1)
        other = base + (".lora_B.weight" if side == "A.weight" else ".lora_A.weight")
        if base + ".weight" not in weights or other not in available:
            missing.append(key)
    return sorted(missing)


def audit_lora(path, transformer):
    from ltx_core_mlx.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
    weights = {key.removeprefix("transformer.") for key in header_keys(transformer)}
    keys = header_keys(path)
    missing = unmatched_keys(keys, weights, LTXV_LORA_COMFY_RENAMING_MAP.apply_to_key)
    logger = logging.getLogger(__name__)
    for key in missing:
        logger.warning("Character LoRA %s: unmatched or incomplete key %s", Path(path).name, key)
    logger.info("Character LoRA %s: %d matched keys, %d unmatched keys", Path(path).name,
                len(keys)-len(missing),len(missing))
    if len(missing) == len(keys):
        raise ValueError("The character LoRA has no complete pairs matching this model")
    return missing
