"""GPT-2 small + jbloom res-jb SAE backend (torch / transformers / safetensors).

NOT_RUN in the approved environment: none of these packages are installed.
Every contract item is taken from ``configs/p3_contract_gpt2_res_jb_l8.json``;
nothing here downloads unless ``allow_download=True`` is passed explicitly.

Coordinates: the SAE was trained on TransformerLens activations with
``center_writing_weights=True``. By derivation (contract, 'hook.coordinates'),
those equal the HF hidden state minus its per-token mean over d_model. Edits
are defined in those coordinates: the SAE reads ``x + delta`` and the model is
run on ``h + delta``. Contract checks C3/C4 test this empirically before any
measurement.
"""

from __future__ import annotations

import json
import resource
import time
from typing import Dict, List, Optional, Sequence

from .contract import check_gpt2_config, check_safetensors_header, check_sae_cfg, read_safetensors_header


class ContractError(RuntimeError):
    pass


def peak_rss_kb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


class Backend:
    def __init__(self, contract: dict, cache_dir: Optional[str] = None, allow_download: bool = False,
                 threads: int = 2):
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        torch.set_num_threads(threads)
        self.torch = torch
        self.contract = contract
        m, s = contract["model"], contract["sae"]
        lfo = not allow_download
        self.tok = GPT2TokenizerFast.from_pretrained(m["hf_repo"], revision=m["revision"], cache_dir=cache_dir,
                                                     local_files_only=lfo)
        self.model = GPT2LMHeadModel.from_pretrained(m["hf_repo"], revision=m["revision"], cache_dir=cache_dir,
                                                     local_files_only=lfo, torch_dtype=torch.float32).eval()
        cfg_path = hf_hub_download(s["hf_repo"], f"{s['folder']}/cfg.json", revision=s["revision"],
                                   cache_dir=cache_dir, local_files_only=lfo)
        w_path = hf_hub_download(s["hf_repo"], f"{s['folder']}/sae_weights.safetensors", revision=s["revision"],
                                 cache_dir=cache_dir, local_files_only=lfo)
        with open(cfg_path) as f:
            sae_cfg = json.load(f)
        errors = (check_sae_cfg(sae_cfg, contract)
                  + check_safetensors_header(read_safetensors_header(w_path), contract)
                  + check_gpt2_config(self.model.config.to_dict(), contract))
        self.contract_report: Dict[str, object] = {"C1_C2_model_cfg_errors": errors}
        if errors:
            raise ContractError("; ".join(errors))
        w = load_file(w_path)
        self.W_enc = w["W_enc"].float()  # [d, m]
        self.W_dec = w["W_dec"].float()  # [m, d]
        self.b_enc = w["b_enc"].float()  # [m]
        self.b_dec = w["b_dec"].float()  # [d]
        self.layer = int(s["cfg_json_verified"]["hook_point_layer"])
        self.block = self.model.transformer.h[self.layer]
        self.bos_id = int(contract["tokenizer"]["bos_token_id"])
        self.dec_norms = self.W_dec.norm(dim=-1)

    # ------------------------------------------------------------ SAE maths

    @staticmethod
    def center(h):
        return h - h.mean(-1, keepdim=True)

    def sae_pre(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    @staticmethod
    def act(pre):
        return pre.clamp_min(0.0), pre > 0

    # ------------------------------------------------------------ forward passes

    def tokenize(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def hook_states(self, ids) -> "torch.Tensor":
        """Hidden state entering block ``layer`` (= TL hook_resid_pre up to centring). ids: [B, T]."""
        with self.torch.no_grad():
            out = self.model.transformer(ids, output_hidden_states=True, use_cache=False)
        return out.hidden_states[self.layer]

    def _pre_hook(self, pos, delta):
        """Adds delta[b] at position pos[b] (pos may be a LongTensor [B] or 'all')."""
        torch = self.torch

        def hook(module, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            hs = hs.clone()
            if isinstance(pos, str) and pos == "all":
                if delta.dim() == 1:          # one vector for every row and position
                    hs = hs + delta
                elif delta.dim() == 2:        # one vector per row, every position
                    hs = hs + delta[:, None, :]
                else:                         # [B, T, d], per position (T must match)
                    hs = hs + delta
            else:
                hs[torch.arange(hs.shape[0]), pos] += delta
            if args:
                return (hs,) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = hs
            return args, kwargs

        return hook

    def final_hidden(self, ids, pos=None, delta=None):
        """ln_f output [B, T, d]; optional edit at the hook."""
        handle = None
        if delta is not None:
            handle = self.block.register_forward_pre_hook(self._pre_hook(pos, delta), with_kwargs=True)
        try:
            with self.torch.no_grad():
                out = self.model.transformer(ids, use_cache=False)
        finally:
            if handle is not None:
                handle.remove()
        return out.last_hidden_state

    def logits_slice(self, hidden, start: int, stop: int):
        with self.torch.no_grad():
            return self.model.lm_head(hidden[:, start:stop])

    # ------------------------------------------------------------ contract checks C3 / C4

    def check_noop(self, ids) -> dict:
        """C3: identity pre-hook and zero-delta hook reproduce unhooked logits exactly."""
        torch = self.torch
        with torch.no_grad():
            ref = self.model(ids, use_cache=False).logits
            h1 = self.block.register_forward_pre_hook(lambda mod, a, k: (a, k), with_kwargs=True)
            try:
                same = self.model(ids, use_cache=False).logits
            finally:
                h1.remove()
            zero = torch.zeros(ids.shape[0], self.W_dec.shape[1])
            h2 = self.block.register_forward_pre_hook(
                self._pre_hook(torch.ones(ids.shape[0], dtype=torch.long), zero), with_kwargs=True)
            try:
                z = self.model(ids, use_cache=False).logits
            finally:
                h2.remove()
        d1 = (ref - same).abs().max().item()
        d2 = (ref - z).abs().max().item()
        return {"identity_hook_max_abs_diff": d1, "zero_delta_max_abs_diff": d2, "pass": d1 == 0.0 and d2 == 0.0}

    def check_reconstruction(self, windows: Sequence[Sequence[int]], batch: int = 8) -> dict:
        """C4: L0 and FVE on non-BOS tokens, centred (contract) vs uncentred (diagnostic)."""
        torch = self.torch
        d = self.W_dec.shape[1]
        stats = {k: {"l0_sum": 0.0, "sse": 0.0, "n": 0, "x2": 0.0} for k in ("centred", "uncentred")}
        sum_x = {k: torch.zeros(d, dtype=torch.float64) for k in stats}
        for i in range(0, len(windows), batch):
            ids = torch.tensor(windows[i:i + batch])
            h = self.hook_states(ids)[:, 1:, :].reshape(-1, d)
            for k, x in (("centred", self.center(h)), ("uncentred", h)):
                a, mask = self.act(self.sae_pre(x))
                recon = a @ self.W_dec + self.b_dec
                st = stats[k]
                st["l0_sum"] += mask.sum().item()
                st["sse"] += ((x - recon) ** 2).sum().item()
                st["n"] += x.shape[0]
                st["x2"] += (x.double() ** 2).sum().item()
                sum_x[k] += x.double().sum(0)
        out = {}
        for k, st in stats.items():
            n = st["n"]
            mean_sq = (sum_x[k] / n).pow(2).sum().item()
            sst = st["x2"] - n * mean_sq
            out[k] = {"l0": st["l0_sum"] / n, "fve": 1.0 - st["sse"] / sst, "n_tokens": n}
        c = out["centred"]
        out["pass"] = 30.0 <= c["l0"] <= 120.0 and c["fve"] >= 0.7
        out["reference"] = {"l0": 60.0, "variance_explained": 0.9, "source": "SAELens pretrained_saes.yaml entry"}
        return out

    # ------------------------------------------------------------ edits

    def enc_col(self, j: int):
        return self.W_enc[:, j]

    def dec_row(self, j: int):
        return self.W_dec[j]

    def jacobian_ln(self, x, j: int, dp_req: float, rtol: float = 1e-10) -> dict:
        """Least-norm delta with pre_j change dp_req and zero pre-change on active non-targets."""
        torch = self.torch
        t0 = time.perf_counter()
        pre = self.sae_pre(x)
        prot = torch.nonzero(pre > 0).flatten()
        prot = prot[prot != j]
        rows = torch.cat([torch.tensor([j]), prot])
        M = self.W_enc[:, rows].T  # [k, d]
        t = torch.zeros(M.shape[0])
        t[0] = dp_req
        pinv = torch.linalg.pinv(M.double(), rtol=rtol)
        delta = (pinv @ t.double()).float()
        sv = torch.linalg.svdvals(M.double())
        rank = int((sv > sv[0] * rtol).sum().item()) if sv.numel() else 0
        res = (t - M @ delta).norm().item() / max(abs(dp_req), 1e-30)
        return {"delta": delta, "rank": rank, "n_constraints": int(M.shape[0]), "residual_rel": res,
                "solve_seconds": time.perf_counter() - t0}

    def match_scale(self, x, j: int, u, alpha: float) -> Optional[float]:
        """ReLU closed form: s >= 0 with relu(p_j + s * E_j.u) - a_j = alpha, or None if impossible."""
        pre = self.sae_pre(x)
        p_j = pre[j].item()
        a_j = max(p_j, 0.0)
        g = (self.enc_col(j) @ u).item()
        need = a_j + alpha - p_j
        if g <= 0.0 or need < 0.0:
            return None
        return need / g
