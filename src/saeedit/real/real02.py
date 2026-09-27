"""CLI for P3-REAL-02 (wedding-topic steering; lexicon label independent of the SAE).

    PYTHONPATH=src python3 -m saeedit.real.real02 <stage> --config configs/p3_real02.json \
        --out results/real02 [--cache-dir DIR] [--allow-download]

Stages: check-env, calibrate, run, summarize. NOT_RUN in the approved environment.
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

# Directions are projected onto admissible edits (P = I - 11^T/d; contract execution K9).
# mean_only is a negative control: invisible to the model, so it must not change behaviour.
METHODS = ("no_edit", "decoder", "encoder_grad", "jacobian_ln", "diffmean", "random", "mean_only")


def _prompts(be, docs, lex, per_doc: int = 4, prompt_len: int = 32):
    """(doc_id, prompt ids) from windows with zero lexicon hits."""
    out = []
    for d in docs:
        cands = [w for w in d.windows if count_hits(be.tok.decode(w[1:]), lex) == 0]
        rng = D.seeded_rng("prompts", d.doc_id)
        for w in (rng.sample(cands, per_doc) if len(cands) > per_doc else cands):
            out.append((d.doc_id, w[:prompt_len]))
    return out


def _deltas(be, method, x, rho, j, vecs):
    """Per-position edits [T, d] for one sequence of centred hook states x [T, d]."""
    import torch
    T, d = x.shape
    if method == "no_edit":
        return torch.zeros(T, d)
    if method == "jacobian_ln":
        rows = []
        for t in range(T):
            u = be.jacobian_ln(x[t], j, 1.0)["delta"]  # solves with J_E P, so u is admissible
            rows.append(rho * u / u.norm())
        return torch.stack(rows)
    if method == "mean_only":
        u = torch.ones(d, dtype=x.dtype)
    else:
        u = be.proj(vecs[method])
    return (rho * u / u.norm()).expand(T, d)


def _kl_all_positions(be, ids, delta):
    import torch
    clean = be.logits_slice(be.final_hidden(ids), 0, ids.shape[1]).log_softmax(-1)
    edit = be.logits_slice(be.final_hidden(ids, "all", delta[None]), 0, ids.shape[1]).log_softmax(-1)
    return float((clean.exp() * (clean - edit)).sum(-1).mean())


def _generate(be, ids, method, rho, j, vecs, new_tokens):
    """Greedy decoding with the edit at every position (recomputed each step, no KV cache).
    Hook-input states depend only on blocks 0..layer-1, so per-position edits do not
    depend on edits at other positions."""
    import torch
    seq = ids.clone()
    for _ in range(new_tokens):
        x = be.center(be.hook_states(seq)[0])
        delta = _deltas(be, method, x, rho, j, vecs)
        hid = be.final_hidden(seq, "all", delta[None])
        nxt = be.logits_slice(hid, seq.shape[1] - 1, seq.shape[1])[0, -1].argmax()
        seq = torch.cat([seq, nxt.view(1, 1)], dim=1)
    return seq


def _continuation_nll(be, full, n_prompt):
    import torch
    lp = be.logits_slice(be.final_hidden(full), 0, full.shape[1] - 1).log_softmax(-1)[0]
    tgt = full[0, 1:]
    idx = torch.arange(n_prompt - 1, full.shape[1] - 1)
    return float(-lp[idx, tgt[idx]].mean())


def stage_calibrate(args, cfg, contract):
    t0 = time.time()
    if not need_torch(args.out, "calibrate", t0):
        return
    import torch
    from .backend import Backend

    be = Backend(contract, args.cache_dir, args.allow_download, 2, "float64")
    lex = cfg["task"]["lexicon"]
    docs = [d for d in load_documents(contract, args.cache_dir, args.allow_download) if d.split == "calibration"]
    D.tokenize_documents(docs, be.tokenize, be.bos_id)
    m = be.W_enc.shape[1]
    s_hit, s_no = torch.zeros(m, dtype=torch.float64), torch.zeros(m, dtype=torch.float64)
    x_hit, x_no = torch.zeros(be.W_dec.shape[1], dtype=torch.float64), torch.zeros(be.W_dec.shape[1], dtype=torch.float64)
    n_hit = n_no = w_hit = 0
    norms: List[float] = []
    for d in docs:
        for wi in capped_windows(d, args.windows_per_doc):
            w = d.windows[wi]
            x = be.center(be.hook_states(torch.tensor([w]))[0])[1:]
            a = be.act(be.sae_pre(x))[0].double()
            norms.extend(x.norm(dim=-1).tolist())
            if count_hits(be.tok.decode(w[1:]), lex) > 0:
                s_hit += a.sum(0); x_hit += x.double().sum(0); n_hit += x.shape[0]; w_hit += 1
            else:
                s_no += a.sum(0); x_no += x.double().sum(0); n_no += x.shape[0]
    if w_hit < cfg["calibration_only_choices"]["min_hit_windows"]:
        record(args.out, "calibrate", "BLOCKED", t0, reason=f"only {w_hit} calibration windows with lexicon hits")
        return
    diff = s_hit / n_hit - s_no / n_no
    top = torch.topk(diff, 5)
    j = int(top.indices[0])
    diffmean = (x_hit / n_hit - x_no / n_no).float()
    budgets = Q.norm_budgets(norms, cfg["calibration_only_choices"]["norm_budget_multipliers"])
    vecs = {"decoder": be.dec_row(j), "encoder_grad": be.enc_col(j), "diffmean": diffmean,
            "random": torch.randn(be.W_dec.shape[1], generator=torch.Generator().manual_seed(0))}
    # matched-quality budget per method from calibration prompts only
    prompts = _prompts(be, docs, lex)[:32]
    kappa = cfg["calibration_only_choices"]["kappa_nats"]
    chosen = {}
    kl_table = defaultdict(dict)
    for meth in METHODS[1:]:
        best = None
        for bname, rho in sorted(budgets["budgets"].items(), key=lambda kv: kv[1]):
            kls = []
            for _, p in prompts:
                ids = torch.tensor([p])
                x = be.center(be.hook_states(ids)[0])
                kls.append(_kl_all_positions(be, ids, _deltas(be, meth, x, rho, j, vecs)))
            mk = sum(kls) / len(kls)
            kl_table[meth][bname] = mk
            if mk <= kappa:
                best = bname
        chosen[meth] = best
    frozen = {"feature": j, "top5_features": top.indices.tolist(), "top5_scores": top.values.tolist(),
              "n_hit_windows": w_hit, "budgets": budgets, "calibration_kl": kl_table, "chosen_budget": chosen,
              "diffmean": diffmean.tolist(), "calibration_doc_ids": sorted(d.doc_id for d in docs),
              "test_documents_read": False}
    os.makedirs(args.out, exist_ok=True)
    sha = Q.freeze(frozen, os.path.join(args.out, "calibration_frozen.json"))
    record(args.out, "calibrate", "RAN", t0, calibration_sha256=sha, feature=j, chosen_budget=chosen)


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
    be = Backend(contract, args.cache_dir, args.allow_download, 2, "float64")
    lex = cfg["task"]["lexicon"]
    docs = load_documents(contract, args.cache_dir, args.allow_download)
    test = sorted((d for d in docs if d.split == "test"), key=lambda d: d.doc_id)
    D.tokenize_documents(test, be.tokenize, be.bos_id)
    j = fz["feature"]
    vecs = {"decoder": be.dec_row(j), "encoder_grad": be.enc_col(j), "diffmean": torch.tensor(fz["diffmean"]),
            "random": torch.randn(be.W_dec.shape[1], generator=torch.Generator().manual_seed(0))}
    rows = []
    for doc_id, p in _prompts(be, test, lex):
        ids = torch.tensor([p])
        x = be.center(be.hook_states(ids)[0])
        for meth in METHODS:
            for bname, rho in sorted(fz["budgets"]["budgets"].items(), key=lambda kv: kv[1]):
                if meth == "no_edit" and bname != sorted(fz["budgets"]["budgets"])[0]:
                    continue
                t1 = time.perf_counter()
                full = _generate(be, ids, meth, rho, j, vecs, cfg["generation"]["new_tokens"])
                text = be.tok.decode(full[0, ids.shape[1]:].tolist())
                lab = label(text, lex)
                rows.append({"doc_id": doc_id, "method": meth, "budget": bname, "rho": rho,
                             "calibration_selected": int(fz["chosen_budget"].get(meth) == bname),
                             **lab, "kl_prompt": _kl_all_positions(be, ids, _deltas(be, meth, x, rho, j, vecs)),
                             "continuation_nll_unedited_model": _continuation_nll(be, full, ids.shape[1]),
                             "seconds": time.perf_counter() - t1, "continuation": text})
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "raw_run.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    record(args.out, "run", "RAN", t0, raw=path, n_rows=len(rows), feature=j)


def stage_summarize(args, cfg, contract):
    t0 = time.time()
    path = os.path.join(args.out, "raw_run.csv")
    if not os.path.exists(path):
        record(args.out, "summarize", "NOT_RUN", t0, reason="no raw_run.csv")
        return
    rows = list(csv.DictReader(open(path)))
    sel = [r for r in rows if r["calibration_selected"] == "1"]
    num = lambda k: (lambda r: float(r[k]))
    out = {"selected_budget_success": {m: cluster_bootstrap(per_unit_means([r for r in sel if r["method"] == m],
                                                                            "doc_id", num("success")))
                                       for m in METHODS[1:]}}
    key = lambda r: (r["doc_id"],)
    out["paired_success"] = {f"{a}-{b}": cluster_bootstrap(paired_unit_differences(sel, "doc_id", key, "method", a, b,
                                                                                    num("success")))
                             for a, b in (("jacobian_ln", "decoder"), ("jacobian_ln", "diffmean"),
                                          ("jacobian_ln", "encoder_grad"), ("decoder", "diffmean"))}
    out["note"] = "paired_unit_differences averages multiple prompts of one document; intervals are descriptive"
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    record(args.out, "summarize", "RAN", t0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["check-env", "calibrate", "run", "summarize"])
    ap.add_argument("--config", default="configs/p3_real02.json")
    ap.add_argument("--out", default="results/real02")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--allow-download", action="store_true", help="requires explicit approval")
    ap.add_argument("--windows-per-doc", type=int, default=8)
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
