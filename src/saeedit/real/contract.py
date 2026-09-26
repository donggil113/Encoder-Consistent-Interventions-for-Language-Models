"""Model-SAE contract checks that need only the standard library.

The contract itself (revisions, shapes, coordinates, sources) lives in
``configs/p3_contract_gpt2_res_jb_l8.json``. This module checks a downloaded
``cfg.json`` and the header of ``sae_weights.safetensors`` against it without
torch: a safetensors file starts with an 8-byte little-endian header length
followed by a JSON header.
"""

from __future__ import annotations

import importlib
import json
import struct
from typing import Dict, List

REQUIRED_MODULES = ("torch", "transformers", "safetensors", "huggingface_hub", "pyarrow")


def load_contract(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def read_safetensors_header(path: str) -> Dict[str, dict]:
    """Return ``{tensor_name: {"dtype", "shape", "data_offsets"}}`` (metadata key dropped)."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("file too short for a safetensors header")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > 100_000_000:
            raise ValueError(f"implausible safetensors header length {n}")
        header = json.loads(f.read(n).decode("utf-8"))
    header.pop("__metadata__", None)
    return header


def check_safetensors_header(header: Dict[str, dict], contract: dict) -> List[str]:
    """Return a list of contract violations (empty list = check C2 passes)."""
    exp = contract["sae"]["expected_tensors_INFERRED"]
    want = {k: v for k, v in exp.items() if isinstance(v, list)}
    errors = []
    if set(header) != set(want):
        errors.append(f"tensor names {sorted(header)} != expected {sorted(want)}")
    for name, shape in want.items():
        if name not in header:
            continue
        if list(header[name]["shape"]) != shape:
            errors.append(f"{name}: shape {header[name]['shape']} != {shape}")
        if header[name]["dtype"] != exp["dtype"]:
            errors.append(f"{name}: dtype {header[name]['dtype']} != {exp['dtype']}")
    return errors


def check_sae_cfg(cfg: dict, contract: dict) -> List[str]:
    """Check C1: every verified cfg.json field matches; also flag unexpected activation settings."""
    errors = []
    for k, v in contract["sae"]["cfg_json_verified"].items():
        if cfg.get(k) != v:
            errors.append(f"cfg[{k!r}] = {cfg.get(k)!r} != {v!r}")
    for k in ("activation_fn", "activation_fn_str"):
        if k in cfg and cfg[k] != "relu":
            errors.append(f"cfg[{k!r}] = {cfg[k]!r}: contract assumes ReLU")
    if cfg.get("apply_b_dec_to_input") is False:
        errors.append("cfg sets apply_b_dec_to_input=False; contract assumes True")
    if cfg.get("normalize_activations") not in (None, "none", False):
        errors.append(f"cfg normalize_activations={cfg.get('normalize_activations')!r}; contract assumes none")
    return errors


def check_gpt2_config(cfg: dict, contract: dict) -> List[str]:
    return [f"model cfg[{k!r}] = {cfg.get(k)!r} != {v!r}"
            for k, v in contract["model"]["config_verified"].items() if cfg.get(k) != v]


def environment_status() -> dict:
    """Which required Python modules are importable (and their versions)."""
    out = {}
    for name in REQUIRED_MODULES:
        try:
            mod = importlib.import_module(name)
            out[name] = {"available": True, "version": getattr(mod, "__version__", "unknown")}
        except Exception as e:  # ImportError or a broken install
            out[name] = {"available": False, "error": type(e).__name__}
    out["all_available"] = all(v["available"] for v in out.values() if isinstance(v, dict))
    return out
