"""P3-CONTRACT-EXEC: canonical TransformerLens/SAELens path vs the HF shortcut.

    HF_HOME=<cache> PYTHONPATH=src <venv>/bin/python -m saeedit.real.contract_exec \
        --config configs/p3_contract_exec.json --out results/contract_exec

Checks K0-K9 are defined (with tolerances) in the config before execution. The
report records every measured quantity; the verdict follows the config's
pass rule and never relaxes a tolerance.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import resource
import sys
import time

from . import data as D
from .contract import load_contract


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p3_contract_exec.json")
    ap.add_argument("--out", default="results/contract_exec")
    args = ap.parse_args(argv)
    t_start = time.time()
    cfg = json.load(open(args.config))
    contract = load_contract(cfg["contract"])
    fx = cfg["fixed"]

    import numpy as np
    import pyarrow.parquet as pq
    import sae_lens
    import torch
    import transformer_lens
    import transformers
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from sae_lens import SAE
    from transformer_lens import HookedTransformer
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    torch.set_num_threads(fx["model"]["threads"])
    torch.manual_seed(0)
    rep = {"config_id": cfg["config_id"], "checks": {}, "versions": {
        "python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
        "transformers": transformers.__version__, "sae_lens": sae_lens.__version__,
        "transformer_lens": getattr(transformer_lens, "__version__", "2.18.0 (pinned; module has no __version__)"),
        "numpy": np.__version__}}
    timings = {}

    # ------------------------------------------------------------------ load
    t0 = time.time()
    m, s = fx["model"], fx["sae"]
    tok = GPT2TokenizerFast.from_pretrained(m["hf_repo"], revision=m["revision"])
    hf = GPT2LMHeadModel.from_pretrained(m["hf_repo"], revision=m["revision"], torch_dtype=torch.float32).eval()
    w_path = hf_hub_download(s["hf_repo"], f"{s['folder']}/sae_weights.safetensors", revision=s["revision"])
    W = load_file(w_path)
    W_enc, W_dec, b_enc, b_dec = W["W_enc"], W["W_dec"], W["b_enc"], W["b_dec"]
    tlo = fx["transformerlens"]
    # a separate HF copy for TL: TL processes (folds/centres) the weights it is given
    hf_for_tl = GPT2LMHeadModel.from_pretrained(m["hf_repo"], revision=m["revision"], torch_dtype=torch.float32).eval()
    tl = HookedTransformer.from_pretrained(
        tlo["model_name"], hf_model=hf_for_tl, tokenizer=tok, fold_ln=tlo["fold_ln"],
        center_writing_weights=tlo["center_writing_weights"], center_unembed=tlo["center_unembed"],
        fold_value_biases=tlo["fold_value_biases"], refactor_factored_attn_matrices=tlo["refactor_factored_attn_matrices"],
        default_prepend_bos=tlo["default_prepend_bos"], dtype="float32", device="cpu")
    tl.eval()
    sae = SAE.from_pretrained(s["saelens_release"], s["saelens_sae_id"], device="cpu")
    timings["load_seconds"] = time.time() - t0
    L = fx["layer"]
    hook = fx["hook"]["tl"]
    d = W_dec.shape[1]

    # ------------------------------------------------------------------ K0
    sc = sae.cfg
    got = {"apply_b_dec_to_input": getattr(sc, "apply_b_dec_to_input", None),
           "normalize_activations": getattr(sc, "normalize_activations", None),
           "architecture": sc.architecture() if callable(getattr(sc, "architecture", None)) else getattr(sc, "architecture", None),
           "hook_name": getattr(sc.metadata, "hook_name", None) if hasattr(sc, "metadata") else getattr(sc, "hook_name", None),
           "d_in": sc.d_in, "d_sae": sc.d_sae,
           "activation": type(sae.activation_fn).__name__ if hasattr(sae, "activation_fn") else None}
    req = fx["saelens"]["required_cfg"]
    cfg_ok = (got["apply_b_dec_to_input"] is True and got["normalize_activations"] == "none"
              and got["architecture"] == req["architecture"] and got["hook_name"] == req["hook_name"]
              and got["d_in"] == req["d_in"] and got["d_sae"] == req["d_sae"] and got["activation"] == "ReLU")
    eq = {k: bool(torch.equal(getattr(sae, k).detach(), v)) for k, v in
          (("W_enc", W_enc), ("W_dec", W_dec), ("b_enc", b_enc), ("b_dec", b_dec))}
    rep["checks"]["K0_weights_identity"] = {"saelens_cfg": got, "cfg_ok": cfg_ok, "bitwise_equal": eq,
                                            "pass": cfg_ok and all(eq.values())}

    # ------------------------------------------------------------------ data (calibration only)
    p_val = hf_hub_download(contract["data"]["hf_repo"], "wikitext-103-raw-v1/validation-00000-of-00001.parquet",
                            repo_type="dataset", revision=contract["data"]["revision"])
    docs = D.build_documents({"calibration": pq.read_table(p_val).column("text").to_pylist()})
    rng = random.Random(fx["data"]["window_seed"])
    D.tokenize_documents(docs, lambda t: tok(t, add_special_tokens=False)["input_ids"], 50256)
    all_w = [(di, wi) for di, doc in enumerate(docs) for wi in range(len(doc.windows))]
    pick = rng.sample(all_w, fx["data"]["n_windows_reconstruction"])
    parity = pick[: fx["data"]["n_windows_parity"]]
    rep["data"] = {"n_calibration_docs": len(docs), "n_calibration_windows": len(all_w)}

    def ids_of(dw):
        return torch.tensor([docs[dw[0]].windows[dw[1]]])

    # ------------------------------------------------------------------ K1 tokenizer
    k1 = []
    for dw in parity:
        w = docs[dw[0]].windows[dw[1]]
        text = tok.decode(w[1:])
        a = tl.to_tokens(text, prepend_bos=True)[0].tolist()
        b = [50256] + tok(text, add_special_tokens=False)["input_ids"]
        k1.append(a == b)
    rep["checks"]["K1_tokenizer"] = {"n": len(k1), "n_equal": sum(k1), "pass": all(k1)}

    # helpers
    def hf_hidden(ids):
        with torch.no_grad():
            return hf.transformer(ids, output_hidden_states=True, use_cache=False).hidden_states[L]

    def tl_resid(ids):
        store = {}

        def f(act, hook):
            store["x"] = act.detach().clone()
            return act

        with torch.no_grad():
            logits = tl.run_with_hooks(ids, fwd_hooks=[(hook, f)])
        return store["x"], logits

    def manual_pre(x):
        return (x - b_dec) @ W_enc + b_enc

    # ------------------------------------------------------------------ K2 / K3 / K6 on parity windows
    t0 = time.time()
    k2, k2_unc, k3, k3_bad, k6 = [], [], [], 0, {"resid": [], "prob": [], "kl": [], "shift": []}
    for dw in parity:
        ids = ids_of(dw)
        x_tl, logit_tl = tl_resid(ids)
        h = hf_hidden(ids)
        x_hf = h - h.mean(-1, keepdim=True)
        xt, xh, hh = x_tl[0, 1:], x_hf[0, 1:], h[0, 1:]
        k2.append(((xt - xh).norm(dim=-1) / xt.norm(dim=-1)).max().item())
        k2_unc.append(((xt - hh).norm(dim=-1) / xt.norm(dim=-1)).max().item())
        with torch.no_grad():
            a_sl = sae.encode(xt)
            pre_m = manual_pre(xh)
        a_m = pre_m.clamp_min(0)
        k3.append(((a_sl - a_m).norm(dim=-1) / a_sl.norm(dim=-1).clamp_min(1e-12)).max().item())
        dis = (a_sl > 0) != (pre_m > 0)
        k3_bad += int((dis & (pre_m.abs() > 1e-3)).sum().item())
        with torch.no_grad():
            logit_hf = hf(ids, use_cache=False).logits
        diff = logit_tl - logit_hf
        c = diff.mean(-1, keepdim=True)
        k6["shift"].append(c.abs().max().item())
        k6["resid"].append((diff - c).abs().max().item())
        pt, ph = logit_tl.softmax(-1), logit_hf.softmax(-1)
        k6["prob"].append((pt - ph).abs().max().item())
        k6["kl"].append((pt * (logit_tl.log_softmax(-1) - logit_hf.log_softmax(-1))).sum(-1).max().item())
    timings["parity_seconds"] = time.time() - t0
    rep["checks"]["K2_activation"] = {"max_rel_diff_centred": max(k2), "max_rel_diff_uncentred_negative_control": max(k2_unc),
                                      "tol": 1e-4, "pass": max(k2) <= 1e-4}
    rep["checks"]["K3_features"] = {"max_rel_diff": max(k3), "active_set_disagreements_outside_1e-3": k3_bad,
                                    "tol": 1e-3, "pass": max(k3) <= 1e-3 and k3_bad == 0}
    rep["checks"]["K6_logit_parity"] = {"max_common_shift": max(k6["shift"]), "max_residual_after_shift": max(k6["resid"]),
                                        "max_prob_diff": max(k6["prob"]), "max_kl": max(k6["kl"]),
                                        "pass": max(k6["resid"]) <= 1e-3 and max(k6["prob"]) <= 1e-5 and max(k6["kl"]) <= 1e-6}

    # ------------------------------------------------------------------ K4 reconstruction (canonical path)
    t0 = time.time()
    tot = {k: {"l0": 0.0, "sse": 0.0, "n": 0, "x2": 0.0, "sx": torch.zeros(d, dtype=torch.float64)} for k in ("tl", "hf_uncentred")}
    for dw in pick:
        ids = ids_of(dw)
        x_tl, _ = tl_resid(ids)
        h = hf_hidden(ids)
        for k, x in (("tl", x_tl[0, 1:]), ("hf_uncentred", h[0, 1:])):
            with torch.no_grad():
                a = sae.encode(x)
                rec = sae.decode(a)
            t = tot[k]
            t["l0"] += (a > 0).sum().item()
            t["sse"] += ((x - rec) ** 2).sum().item()
            t["n"] += x.shape[0]
            t["x2"] += (x.double() ** 2).sum().item()
            t["sx"] += x.double().sum(0)
    k4 = {}
    for k, t in tot.items():
        sst = t["x2"] - t["n"] * (t["sx"] / t["n"]).pow(2).sum().item()
        k4[k] = {"l0": t["l0"] / t["n"], "fve": 1.0 - t["sse"] / sst, "n_tokens": t["n"]}
    timings["reconstruction_seconds"] = time.time() - t0
    rep["checks"]["K4_reconstruction"] = {**k4, "reference": {"l0": 60.0, "variance_explained": 0.9},
                                          "pass": 30.0 <= k4["tl"]["l0"] <= 120.0 and k4["tl"]["fve"] >= 0.7}

    # ------------------------------------------------------------------ K5 no-op
    ids = ids_of(parity[0])
    with torch.no_grad():
        ref_tl = tl(ids)
        id_tl = tl.run_with_hooks(ids, fwd_hooks=[(hook, lambda a, hook: a)])
        z_tl = tl.run_with_hooks(ids, fwd_hooks=[(hook, lambda a, hook: a + torch.zeros_like(a))])
        ref_hf = hf(ids, use_cache=False).logits
        h1 = hf.transformer.h[L].register_forward_pre_hook(lambda mod, a, k: (a, k), with_kwargs=True)
        id_hf = hf(ids, use_cache=False).logits
        h1.remove()

        def zero_pre(mod, a, k):
            if a:
                return (a[0] + torch.zeros_like(a[0]),) + tuple(a[1:]), k
            k = dict(k)
            k["hidden_states"] = k["hidden_states"] + torch.zeros_like(k["hidden_states"])
            return a, k

        h2 = hf.transformer.h[L].register_forward_pre_hook(zero_pre, with_kwargs=True)
        z_hf = hf(ids, use_cache=False).logits
        h2.remove()
    k5 = {"tl_identity": (ref_tl - id_tl).abs().max().item(), "tl_zero": (ref_tl - z_tl).abs().max().item(),
          "hf_identity": (ref_hf - id_hf).abs().max().item(), "hf_zero": (ref_hf - z_hf).abs().max().item()}
    rep["checks"]["K5_noop"] = {**k5, "pass": all(v == 0.0 for v in k5.values())}

    # ------------------------------------------------------------------ K7 / K8 / K9 interventions
    t0 = time.time()
    ones = torch.ones(d) / d ** 0.5

    def P(v):
        return v - v.mean()

    med_norm = float(torch.cat([tl_resid(ids_of(dw))[0][0, 1:].norm(dim=-1) for dw in parity[:4]]).median())
    rho = 0.1 * med_norm
    feats = sorted(random.Random("k7-features").sample(range(W_enc.shape[1]), 4))

    def run_tl_edit(ids, pos, delta):
        store = {}

        def f(act, hook):
            act = act.clone()
            act[:, pos, :] += delta
            store["x"] = act[0, pos].detach().clone()
            return act

        with torch.no_grad():
            lg = tl.run_with_hooks(ids, fwd_hooks=[(hook, f)])
        return store["x"], lg

    def run_hf_edit(ids, pos, delta):
        store = {}

        def f(mod, a, k):
            hs = (a[0] if a else k["hidden_states"]).clone()
            hs[:, pos, :] += delta
            store["h"] = hs[0, pos].detach().clone()
            if a:
                return (hs,) + tuple(a[1:]), k
            k = dict(k)
            k["hidden_states"] = hs
            return a, k

        hdl = hf.transformer.h[L].register_forward_pre_hook(f, with_kwargs=True)
        try:
            with torch.no_grad():
                lg = hf(ids, use_cache=False).logits
        finally:
            hdl.remove()
        return store["h"], lg

    k7 = {"feat_rel": [], "kl": [], "prob": [], "dlogp": []}
    k8 = []
    k9 = {"kl_tl": [], "kl_hf": [], "sae_drift_rel": [], "target_like_change": []}
    for wi, dw in enumerate(parity[:8]):
        ids = ids_of(dw)
        pos = random.Random(f"k7-pos-{wi}").randrange(1, ids.shape[1] - 16)
        x_tl, lg_tl0 = tl_resid(ids)
        with torch.no_grad():
            lg_hf0 = hf(ids, use_cache=False).logits
        x0 = x_tl[0, pos]
        pre0 = manual_pre(x0)
        for j in feats:
            dirs = {"decoder": P(W_dec[j]), "encoder_row": P(W_enc[:, j])}
            act = torch.nonzero(pre0 > 0).flatten()
            act = act[act != j]
            M = W_enc[:, torch.cat([torch.tensor([j]), act])].T
            MP = M - M.mean(-1, keepdim=True)  # M P  (J_E P)
            t = torch.zeros(MP.shape[0])
            t[0] = 1.0
            dirs["ln"] = (torch.linalg.pinv(MP.double(), rtol=1e-10) @ t.double()).float()
            for name, u in dirs.items():
                delta = rho * u / u.norm()
                xe_tl, lg_tl = run_tl_edit(ids, pos, delta)
                he_hf, lg_hf = run_hf_edit(ids, pos, delta)
                xe_hf = he_hf - he_hf.mean()
                with torch.no_grad():
                    da_tl = sae.encode(xe_tl[None])[0] - sae.encode(x0[None])[0]
                da_hf = manual_pre(xe_hf).clamp_min(0) - pre0.clamp_min(0)
                k7["feat_rel"].append(((da_tl - da_hf).norm() / da_tl.norm().clamp_min(1e-12)).item())
                sl = slice(pos, ids.shape[1])
                lt, lh = lg_tl[0, sl].log_softmax(-1), lg_hf[0, sl].log_softmax(-1)
                k7["kl"].append((lt.exp() * (lt - lh)).sum(-1).max().item())
                k7["prob"].append((lt.exp() - lh.exp()).abs().max().item())
                ch_tl = lt - lg_tl0[0, sl].log_softmax(-1)
                ch_hf = lh - lg_hf0[0, sl].log_softmax(-1)
                k7["dlogp"].append((ch_tl - ch_hf).abs().max().item())
            # K8: raw decoder direction (with its mean component): TL convention vs HF-shortcut convention
            raw = rho * W_dec[j] / W_dec[j].norm()
            a_tl_conv = manual_pre(x0 + raw).clamp_min(0)
            a_hf_conv = manual_pre(x0 + raw - raw.mean()).clamp_min(0)
            k8.append({"feature": j, "mean_component_rel": (raw.mean() * d ** 0.5 / raw.norm()).abs().item(),
                       "readout_diff_rel": ((a_tl_conv - a_hf_conv).norm() / (a_tl_conv - pre0.clamp_min(0)).norm().clamp_min(1e-12)).item()})
        # K9: mean-only edit
        delta = rho * ones
        xe_tl, lg_tl = run_tl_edit(ids, pos, delta)
        _, lg_hf = run_hf_edit(ids, pos, delta)
        sl = slice(pos, ids.shape[1])
        for key, lg, lg0 in (("kl_tl", lg_tl, lg_tl0), ("kl_hf", lg_hf, lg_hf0)):
            a, b = lg0[0, sl].log_softmax(-1), lg[0, sl].log_softmax(-1)
            k9[key].append((a.exp() * (a - b)).sum(-1).max().item())
        with torch.no_grad():
            da = sae.encode(xe_tl[None])[0] - sae.encode(x0[None])[0]
        k9["sae_drift_rel"].append((da.norm() / rho).item())
        k9["target_like_change"].append(da.abs().max().item())
    timings["intervention_seconds"] = time.time() - t0
    rep["checks"]["K7_intervention_parity"] = {
        "n_edits": len(k7["kl"]), "rho": rho, "median_residual_norm": med_norm, "features": feats,
        "max_feature_change_rel_diff": max(k7["feat_rel"]), "max_kl": max(k7["kl"]), "max_prob_diff": max(k7["prob"]),
        "max_logprob_change_diff": max(k7["dlogp"]),
        "pass": max(k7["feat_rel"]) <= 1e-3 and max(k7["kl"]) <= 1e-6 and max(k7["prob"]) <= 1e-5}
    rep["checks"]["K8_non_admissible_readout"] = {"per_feature": k8, "descriptive": True}
    rep["checks"]["K9_mean_only_control"] = {
        "max_kl_tl": max(k9["kl_tl"]), "max_kl_hf": max(k9["kl_hf"]),
        "sae_drift_rel_median": float(np.median(k9["sae_drift_rel"])), "sae_drift_rel_max": max(k9["sae_drift_rel"]),
        "max_single_feature_change": max(k9["target_like_change"]), "rho": rho,
        "pass_kl_bound": max(k9["kl_tl"]) <= 1e-8 and max(k9["kl_hf"]) <= 1e-8}

    passed = all(rep["checks"][k]["pass"] for k in ("K0_weights_identity", "K1_tokenizer", "K2_activation",
                                                     "K3_features", "K4_reconstruction", "K5_noop",
                                                     "K6_logit_parity", "K7_intervention_parity")) \
        and rep["checks"]["K9_mean_only_control"]["pass_kl_bound"]
    rep["verdict"] = "CONTRACT_PASS" if passed else "CONTRACT_FAIL_BLOCKED"
    timings["total_seconds"] = time.time() - t_start
    timings["peak_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rep["timings"] = timings
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "contract_exec_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: (v.get("pass", v.get("pass_kl_bound", "desc")) if isinstance(v, dict) else v)
                      for k, v in rep["checks"].items()}, indent=1))
    print(rep["verdict"], json.dumps(timings))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
