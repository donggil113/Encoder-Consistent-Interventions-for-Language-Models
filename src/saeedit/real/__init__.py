"""Adapters for real model/SAE evaluation (P3-REAL-01R, P3-REAL-02).

Modules that only need the standard library (``contract``, ``data``, ``dose``,
``drift``, ``lexicon``) are unit-tested here. ``backend`` and the CLIs need
torch/transformers/safetensors/huggingface_hub/pyarrow, which are not installed
in the approved environment; they are written but NOT_RUN.
"""
