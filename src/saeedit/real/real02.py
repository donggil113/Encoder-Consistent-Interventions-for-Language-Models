"""CLI for P3-REAL-02 / P3-REAL-02-LX (wedding-lexicon occurrence; label never uses the SAE).

    HF_HOME=<cache> HF_HUB_OFFLINE=1 PYTHONPATH=src <venv>/bin/python -m saeedit.real.real02 <stage> \
        --config configs/p3_real02_lx.json --out results/real02_lx

Stages: check-env, calibrate, run, summarize.

Actor edits are admissible (in range(P), P = I - 11^T/d): decoder, encoder-row and DiffMean
directions are projected by P; the LN direction solves (M P) delta = e_1 and lies in
range(P) by construction. mean_only (c 1/sqrt(d)) is a control outside every ranking and is
run at one fixed amplitude only. The metric is wedding-lexicon occurrence in the generated
continuation: a lexical proxy, not semantic topic success. The lexicon is also used to pick
the SAE feature and the DiffMean direction on calibration data (disclosed in the paper).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List

from . import data as D
from . import dose as Q
from .contract import environment_status, load_contract
from .lexicon import count_hits, label
from .real01r import capped_windows, last_status, load_documents, need_torch, record
from .stats import cluster_bootstrap, paired_unit_differences, per_unit_means

ACTORS = ("decoder", "encoder_grad", "jacobian_ln", "diffmean")
CONTROL = "mean_only"


def _methods(cfg) -> List[str]:
    return list(cfg["methods"].get("ranked", ACTORS))


def _prompts(be, docs, lex, per_doc: int = 4, prompt_len: int = 32):
    """(doc_id, prompt ids) from windows with zero lexicon hits."""
    out = []
    for d in docs:
        cands = [w for w in d.windows if count_hits(be.tok.decode(w[1:]), lex) == 0]
        rng = D.seeded_rng("prompts", d.doc_id)
        for w in (rng.sample(cands, per_doc) if len(cands) > per_doc else cands):
            out.append((d.doc_id, w[:prompt_len]))
    return out


def _deltas(be, method, x, rho, j, vecs, cache=None):
    """Per-position edits [T, d] for hook-input states x [T, d] (centred). ``cache`` holds LN unit
    directions of earlier positions: states at the hook depend only on blocks 0..layer-1, so a
    position's LN direction never changes when later tokens are appended."""
    import torch
    T, d = x.shape
    if method == "no_edit":
        return torch.zeros(T, d, dtype=x.dtype)
    if method == "jacobian_ln":
        cache = [] if cache is None else cache
        for t in range(len(cache), T):
            u = be.jacobian_ln(x[t], j, 1.0)["delta"]  # solves with J_E P, so u is admissible
            cache.append(u / u.norm())
        return rho * torch.stack(cache[:T])
    if method == CONTROL:
        u = torch.ones(d, dtype=x.dtype)
    else:
        u = be.proj(vecs[method])
    return (rho * u / u.norm()).expand(T, d)


class _EditHook:
    """Pre-hook on the hook block that edits every position with deltas computed from the
    incoming hidden state itself (one forward per call). Records x = P h at the hook."""

    def __init__(self, be, method, rho, j, vecs):
        self.be, self.method, self.rho, self.j, self.vecs = be, method, rho, j, vecs
        self.cache: list = []
        self.x = None
        self.delta = None

    def __call__(self, module, args, kwargs):
        hs = args[0] if args else kwargs["hidden_states"]
        x = self.be.center(hs[0])
        delta = _deltas(self.be, self.method, x, self.rho, self.j, self.vecs, self.cache)
        self.x, self.delta = x, delta
        hs = hs + delta[None]
        if args:
            return (hs,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = hs
        return args, kwargs


def _edited_logprobs(be, ids, method, rho, j, vecs, hook=None):
    """Log-probs [T, V] of the edited model on ids (teacher forced), and the hook object."""
    hook = hook or _EditHook(be, method, rho, j, vecs)
    h = be.block.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with be.torch.no_grad():
            hid = be.model.transformer(ids, use_cache=False).last_hidden_state
    finally:
        h.remove()
    return be.logits_slice(hid, 0, ids.shape[1])[0].log_softmax(-1), hook


def _base_logprobs(be, ids):
    return be.logits_slice(be.final_hidden(ids), 0, ids.shape[1])[0].log_softmax(-1)


def _kl(base, edit):
    """KL(P_base || P_edit) per position, averaged over positions."""
    return float((base.exp() * (base - edit)).sum(-1).mean())


def _kl_all_positions(be, ids, method, rho, j, vecs):
    return _kl(_base_logprobs(be, ids), _edited_logprobs(be, ids, method, rho, j, vecs)[0])


def _generate(be, ids, method, rho, j, vecs, new_tokens):
    """Greedy decoding (temperature 0, no sampling) with the edit at every position, prompt and
    generated; no KV cache; one forward per token."""
    import torch
    seq = ids.clone()
    hook = _EditHook(be, method, rho, j, vecs)
    for _ in range(new_tokens):
        lp, hook = _edited_logprobs(be, seq, method, rho, j, vecs, hook)
        seq = torch.cat([seq, lp[-1].argmax().view(1, 1)], dim=1)
    return seq


def _continuation_nll(be, full, n_prompt):
    """Mean NLL of the generated tokens under the unedited model (fluency proxy)."""
    import torch
    lp = _base_logprobs(be, full)[:-1]
    tgt = full[0, 1:]
    idx = torch.arange(n_prompt - 1, full.shape[1] - 1)
    return float(-lp[idx, tgt[idx]].mean())


def _internal(be, x, delta, j):
    """Canonical SAE readout at non-BOS prompt positions: E(x + P delta) - E(x)."""
    a0, m0 = be.act(be.sae_pre(x[1:]))
    a1, m1 = be.act(be.sae_pre(x[1:] + be.proj(delta[1:])))
    d = a1 - a0
    tc = d[:, j]
    d_nt = d.clone()
    d_nt[:, j] = 0
    new = (~m0 & m1)
    new[:, j] = False
    return {"target_change_mean": float(tc.mean()), "target_active_after_frac": float(m1[:, j].double().mean()),
            "drift_all_nontarget_mean": float(d_nt.norm(dim=-1).mean()),
            "n_newly_active_mean": float(new.sum(-1).double().mean())}


def stage_calibrate(args, cfg, contract):
    t0 = time.time()
    if not need_torch(args.out, "calibrate", t0):
        return
    import torch
    from .backend import Backend

    torch.manual_seed(0)
    be = Backend(contract, args.cache_dir, args.allow_download, cfg["budget"]["threads"], "float64")
    lex = cfg["task"]["lexicon"]
    cc = cfg["calibration_only_choices"]
    docs = sorted((d for d in load_documents(contract, args.cache_dir, args.allow_download) if d.split == "calibration"),
                  key=lambda d: d.doc_id)
    D.tokenize_documents(docs, be.tokenize, be.bos_id)
    m = be.W_enc.shape[1]
    s_hit, s_no = torch.zeros(m, dtype=torch.float64), torch.zeros(m, dtype=torch.float64)
    x_hit, x_no = torch.zeros(be.W_dec.shape[1], dtype=torch.float64), torch.zeros(be.W_dec.shape[1], dtype=torch.float64)
    n_hit = n_no = w_hit = w_no = 0
    norms: List[float] = []
    for d in docs:
        for wi in capped_windows(d, cc["windows_per_doc_cap"]):
            w = d.windows[wi]
            x = be.center(be.hook_states(torch.tensor([w]))[0])[1:]
            a = be.act(be.sae_pre(x))[0].double()
            norms.extend(x.norm(dim=-1).tolist())
            if count_hits(be.tok.decode(w[1:]), lex) > 0:
                s_hit += a.sum(0); x_hit += x.double().sum(0); n_hit += x.shape[0]; w_hit += 1
            else:
                s_no += a.sum(0); x_no += x.double().sum(0); n_no += x.shape[0]; w_no += 1
    t_scan = time.time() - t0
    if w_hit < cc["min_hit_windows"]:
        record(args.out, "calibrate", "BLOCKED", t0, reason=f"only {w_hit} calibration windows with lexicon hits")
        return
    diff = s_hit / n_hit - s_no / n_no
    top = torch.topk(diff, 5)
    j = int(top.indices[0])
    diffmean = x_hit / n_hit - x_no / n_no
    budgets = Q.norm_budgets(norms, cc["norm_budget_multipliers"])
    vecs = {"decoder": be.dec_row(j), "encoder_grad": be.enc_col(j), "diffmean": diffmean}
    prompts = _prompts(be, docs, lex, cc["calibration_prompts_per_doc"], cfg["data"]["prompt_len"])[:cc["n_calibration_prompts"]]
    kappa = cc["kappa_nats"]
    chosen: Dict[str, object] = {}
    kl_table = defaultdict(dict)
    t1 = time.time()
    base = [(torch.tensor([p]), _base_logprobs(be, torch.tensor([p]))) for _, p in prompts]
    for meth in _methods(cfg):
        best = None
        for bname, rho in sorted(budgets["budgets"].items(), key=lambda kv: kv[1]):
            kls = [_kl(b, _edited_logprobs(be, ids, meth, rho, j, vecs)[0]) for ids, b in base]
            mk = sum(kls) / len(kls)
            kl_table[meth][bname] = mk
            if mk <= kappa:
                best = bname
        chosen[meth] = best
    t_kl = time.time() - t1
    # cost probe on calibration prompts only: seconds per generation (no_edit and LN)
    t2 = time.time()
    probe = []
    for meth in ("no_edit", "jacobian_ln"):
        for ids, _ in base[:2]:
            s = time.perf_counter()
            _generate(be, ids, meth, budgets["budgets"][sorted(budgets["budgets"], key=budgets["budgets"].get)[0]], j, vecs,
                      cfg["generation"]["new_tokens"])
            probe.append((meth, time.perf_counter() - s))
    sec_gen = {mm: sum(s for (m_, s) in probe if m_ == mm) / 2 for mm in ("no_edit", "jacobian_ln")}
    n_gen_per_doc = 1 + sum(1 for m_ in _methods(cfg) if chosen[m_] is not None) + 1  # no_edit + actors + control
    per_doc = sec_gen["no_edit"] * (n_gen_per_doc - 1) + sec_gen["jacobian_ln"] + 2.0  # + KL/NLL/readout forwards
    budget_s = 60 * cfg["budget"]["run_wall_clock_minutes_max"]
    fits = [n for n in cfg["test"]["n_docs_options"] if n * per_doc <= budget_s]
    frozen = {"config_id": cfg["config_id"], "feature": j, "top5_features": top.indices.tolist(),
              "top5_scores": top.values.tolist(), "n_hit_windows": w_hit, "n_nohit_windows": w_no,
              "budgets": budgets, "calibration_kl": kl_table, "chosen_budget": chosen, "kappa_nats": kappa,
              "control_budget": sorted(budgets["budgets"], key=budgets["budgets"].get)[-1],
              "diffmean": diffmean.tolist(), "diffmean_norm": float(diffmean.norm()),
              "diffmean_mean_component": float(diffmean.mean()),
              "calibration_doc_ids": [d.doc_id for d in docs],
              "calibration_prompt_doc_ids": [di for di, _ in prompts],
              "cost_probe_seconds_per_generation": sec_gen, "projected_seconds_per_test_doc": per_doc,
              "n_test_docs": max(fits) if fits else None,
              "reduction_rule": "largest n in test.n_docs_options with n * projected_seconds_per_test_doc <= budget",
              "test_documents_read": False}
    os.makedirs(args.out, exist_ok=True)
    sha = Q.freeze(frozen, os.path.join(args.out, "calibration_frozen.json"))
    record(args.out, "calibrate", "RAN", t0, calibration_sha256=sha, feature=j, chosen_budget=chosen,
           n_hit_windows=w_hit, n_test_docs=frozen["n_test_docs"], seconds_scan=t_scan, seconds_kl_sweep=t_kl,
           seconds_cost_probe=time.time() - t2, threads=cfg["budget"]["threads"])


def stage_run(args, cfg, contract):
    t0 = time.time()
    if not need_torch(args.out, "run", t0):
        return
    cal = last_status(args.out, "calibrate")
    if cal.get("status") != "RAN":
        record(args.out, "run", "BLOCKED", t0, reason="calibration not frozen")
        return
    import torch
    from .backend import Backend

    fz = Q.load_frozen(os.path.join(args.out, "calibration_frozen.json"), cal["calibration_sha256"])
    if fz["n_test_docs"] is None:
        record(args.out, "run", "NOT_RUN", t0, reason="projected cost exceeds the run budget for every n_docs option")
        return
    be = Backend(contract, args.cache_dir, args.allow_download, cfg["budget"]["threads"], "float64")
    lex = cfg["task"]["lexicon"]
    docs = load_documents(contract, args.cache_dir, args.allow_download)
    test = sorted((d for d in docs if d.split == "test"), key=lambda d: d.doc_id)
    bad = D.check_disjoint(test + [d for d in docs if d.split == "calibration"])
    if bad:
        record(args.out, "run", "BLOCKED", t0, reason="calibration/test overlap", examples=bad[:5])
        return
    D.seeded_rng(cfg["test"]["doc_order_seed"]).shuffle(test)
    D.tokenize_documents(test, be.tokenize, be.bos_id)
    prompts = []
    for d in test:
        p = _prompts(be, [d], lex, 1, cfg["data"]["prompt_len"])
        if p:
            prompts.append(p[0])
        if len(prompts) == fz["n_test_docs"]:
            break
    j = fz["feature"]
    vecs = {"decoder": be.dec_row(j), "encoder_grad": be.enc_col(j), "diffmean": torch.tensor(fz["diffmean"], dtype=be.dtype)}
    budgets = fz["budgets"]["budgets"]
    plan = [("no_edit", "none", 0.0, "reference")]
    for meth in _methods(cfg):
        b = fz["chosen_budget"][meth]
        plan.append((meth, b, budgets[b], "actor") if b is not None else (meth, None, None, "NOT_RUN_NO_BUDGET_WITHIN_KAPPA"))
    plan.append((CONTROL, fz["control_budget"], budgets[fz["control_budget"]], "control_fixed_amplitude"))
    rows = []
    n_new = cfg["generation"]["new_tokens"]
    for doc_id, p in prompts:
        ids = torch.tensor([p])
        base = _base_logprobs(be, ids)
        for meth, bname, rho, role in plan:
            if rho is None:
                rows.append({"doc_id": doc_id, "method": meth, "role": role, "status": role})
                continue
            t1 = time.perf_counter()
            full = _generate(be, ids, meth, rho, j, vecs, n_new)
            t_gen = time.perf_counter() - t1
            text = be.tok.decode(full[0, ids.shape[1]:].tolist())
            lab = label(text, lex)
            lp, hook = _edited_logprobs(be, ids, meth, rho, j, vecs)
            rows.append({"doc_id": doc_id, "method": meth, "role": role, "status": "OK", "budget": bname, "rho": rho,
                         "prompt_ids": " ".join(map(str, p)),
                         **lab, "hits_per_generated_token": lab["hits"] / n_new,
                         "kl_prompt_teacher_forced": _kl(base, lp),
                         "edit_norm_per_position_median": float(hook.delta[1:].norm(dim=-1).median()),
                         "edit_mean_component_max": float(hook.delta.mean(-1).abs().max()),
                         **_internal(be, hook.x, hook.delta, j),
                         "continuation_nll_unedited_model": _continuation_nll(be, full, ids.shape[1]),
                         "seconds_generation": t_gen, "seconds_row": time.perf_counter() - t1,
                         "continuation_ids": " ".join(map(str, full[0, ids.shape[1]:].tolist())),
                         "continuation": text})
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "raw_run.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}), restval="")
        w.writeheader()
        w.writerows(rows)
    record(args.out, "run", "RAN", t0, raw=path, n_rows=len(rows), feature=j, n_test_docs=len(prompts),
           test_doc_ids=[di for di, _ in prompts], threads=cfg["budget"]["threads"])


def stage_summarize(args, cfg, contract):
    t0 = time.time()
    path = os.path.join(args.out, "raw_run.csv")
    if not os.path.exists(path):
        record(args.out, "summarize", "NOT_RUN", t0, reason="no raw_run.csv")
        return
    rows = [r for r in csv.DictReader(open(path)) if r["status"] == "OK"]
    num = lambda k: (lambda r: float(r[k]) if r.get(k, "") != "" else None)
    per = {}
    for m in ["no_edit"] + _methods(cfg) + [CONTROL]:
        rs = [r for r in rows if r["method"] == m]
        per[m] = {"n_docs": len({r["doc_id"] for r in rs}),
                  "any_hit_rate": cluster_bootstrap(per_unit_means(rs, "doc_id", num("success")), stat=_mean),
                  "hits_per_generated_token": cluster_bootstrap(per_unit_means(rs, "doc_id", num("hits_per_generated_token")),
                                                                stat=_mean),
                  **{k: cluster_bootstrap(per_unit_means(rs, "doc_id", num(k)))
                     for k in ("kl_prompt_teacher_forced", "continuation_nll_unedited_model", "target_change_mean",
                               "drift_all_nontarget_mean", "edit_norm_per_position_median")}}
    key = lambda r: (r["doc_id"],)
    pairs = [("jacobian_ln", "decoder"), ("jacobian_ln", "diffmean"), ("jacobian_ln", "encoder_grad"),
             ("decoder", "diffmean")] + [(m, "no_edit") for m in _methods(cfg)]
    paired = {}
    for a, b in pairs:
        for metric in ("success", "hits_per_generated_token", "kl_prompt_teacher_forced", "continuation_nll_unedited_model"):
            paired[f"{a}-{b}|{metric}"] = cluster_bootstrap(
                paired_unit_differences(rows, "doc_id", key, "method", a, b, num(metric)), stat=_mean)
    ctrl = [r for r in rows if r["method"] == CONTROL]
    ref = {r["doc_id"]: r["continuation_ids"] for r in rows if r["method"] == "no_edit"}
    out = {"per_method_at_selected_budget": per, "paired": paired,
           "control_mean_only": {"n": len(ctrl), "identical_continuation_to_no_edit":
                                 sum(r["continuation_ids"] == ref.get(r["doc_id"]) for r in ctrl),
                                 "max_kl": max((float(r["kl_prompt_teacher_forced"]) for r in ctrl), default=None)},
           "interpretation_rule": cfg["interpretation_rule"],
           "note": "unit = test document (one prompt each); statistic = mean over documents; intervals are "
                   "document-bootstrap percentiles, descriptive; an interval that includes 0 is not evidence of equivalence"}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    record(args.out, "summarize", "RAN", t0)


def _mean(v):
    return sum(v) / len(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["check-env", "calibrate", "run", "summarize"])
    ap.add_argument("--config", default="configs/p3_real02_lx.json")
    ap.add_argument("--out", default="results/real02_lx")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--allow-download", action="store_true", help="requires explicit approval")
    args = ap.parse_args(argv)
    cfg = json.load(open(args.config))
    contract = load_contract(cfg["contract"])
    if args.stage == "check-env":
        t0 = time.time()
        env = environment_status()
        record(args.out, "check-env", "RAN", t0, env=env,
               verdict="DEPENDENCIES_AVAILABLE" if env["all_available"] else "BLOCKED_DEPENDENCIES")
    else:
        {"calibrate": stage_calibrate, "run": stage_run, "summarize": stage_summarize}[args.stage](args, cfg, contract)
    return 0


if __name__ == "__main__":
    sys.exit(main())
