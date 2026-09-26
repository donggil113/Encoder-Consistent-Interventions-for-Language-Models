"""CLI for P3-REAL-01R (same-layer fidelity decomposition on GPT-2 small + res-jb L8).

    PYTHONPATH=src python3 -m saeedit.real.real01r <stage> --config configs/p3_real01r.json \
        --out results/real01r [--cache-dir DIR] [--allow-download]

Stages: check-env, contract, calibrate, smoke, run, summarize. Each stage appends
its status (RAN / NOT_RUN / BLOCKED / FAIL, reason, seconds, peak RSS) to
``<out>/stage_status.json``; a stage never writes PASS for something it did not
execute. Only ``check-env`` works without torch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from typing import Dict, List

from . import data as D
from . import dose as Q
from .contract import environment_status, load_contract
from .stats import cluster_bootstrap, paired_unit_differences, per_unit_means

TARGET_KINDS = ("active", "inactive")
METHODS_EQUAL_NORM = ("decoder", "encoder_grad", "jacobian_ln", "random")
METHODS_MATCHED = ("decoder", "encoder_grad", "jacobian_ln", "random")
UNMATCHED_BASELINES = ("decoder_calib_rescaled", "decoder_plain")


def _status_path(out: str) -> str:
    return os.path.join(out, "stage_status.json")


def record(out: str, stage: str, status: str, t0: float, **info) -> dict:
    os.makedirs(out, exist_ok=True)
    path = _status_path(out)
    hist = json.load(open(path)) if os.path.exists(path) else []
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        rss = None
    entry = {"stage": stage, "status": status, "seconds": round(time.time() - t0, 3), "peak_rss_kb": rss,
             "unix_time": time.time(), **info}
    hist.append(entry)
    with open(path, "w") as f:
        json.dump(hist, f, indent=2, default=str)
    print(json.dumps(entry, default=str))
    return entry


def last_status(out: str, stage: str) -> dict:
    path = _status_path(out)
    if not os.path.exists(path):
        return {}
    hits = [e for e in json.load(open(path)) if e["stage"] == stage]
    return hits[-1] if hits else {}


# ---------------------------------------------------------------- shared loading


def load_documents(contract: dict, cache_dir, allow_download: bool) -> List[D.Document]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    dcfg = contract["data"]
    files = {"calibration": "wikitext-103-raw-v1/validation-00000-of-00001.parquet",
             "test": "wikitext-103-raw-v1/test-00000-of-00001.parquet"}
    lines = {}
    for split, fn in files.items():
        p = hf_hub_download(dcfg["hf_repo"], fn, repo_type="dataset", revision=dcfg["revision"],
                            cache_dir=cache_dir, local_files_only=not allow_download)
        lines[split] = pq.read_table(p).column("text").to_pylist()
    return D.build_documents(lines)


def capped_windows(doc: D.Document, cap: int) -> List[int]:
    idx = list(range(len(doc.windows)))
    if len(idx) <= cap:
        return idx
    return sorted(D.seeded_rng("windows", doc.doc_id).sample(idx, cap))


def need_torch(out: str, stage: str, t0: float) -> bool:
    env = environment_status()
    if not env["all_available"]:
        record(out, stage, "BLOCKED", t0, reason="missing dependencies (installation not approved)", env=env)
        return False
    return True


# ---------------------------------------------------------------- stages


def stage_check_env(args, cfg, contract):
    t0 = time.time()
    env = environment_status()
    record(args.out, "check-env", "RAN", t0, env=env,
           verdict="DEPENDENCIES_AVAILABLE" if env["all_available"] else "BLOCKED_DEPENDENCIES")


def stage_contract(args, cfg, contract):
    t0 = time.time()
    if not need_torch(args.out, "contract", t0):
        return
    import torch
    from .backend import Backend, ContractError

    try:
        be = Backend(contract, args.cache_dir, args.allow_download, cfg["budget"]["threads"])
    except ContractError as e:
        record(args.out, "contract", "BLOCKED", t0, reason=f"contract violation: {e}")
        return
    docs = load_documents(contract, args.cache_dir, args.allow_download)
    calib = [d for d in docs if d.split == "calibration"]
    D.tokenize_documents(calib, be.tokenize, be.bos_id)
    wins = [w for d in calib for i, w in enumerate(d.windows) if i in capped_windows(d, 2)]
    noop = be.check_noop(torch.tensor(wins[:4]))
    recon = be.check_reconstruction(wins[:64])
    ok = noop["pass"] and recon["pass"]
    record(args.out, "contract", "RAN" if ok else "BLOCKED", t0, C3_noop=noop, C4_reconstruction=recon,
           C1_C2=be.contract_report, verdict="CONTRACT_PASS" if ok else "CONTRACT_FAIL_BLOCKED",
           n_calibration_docs=len(calib), dec_norm_min=be.dec_norms.min().item(),
           dec_norm_max=be.dec_norms.max().item())


def _encode_windows(be, doc, cap):
    """Yield (window_idx, ids, x_centred [T, d]) for the capped windows of a document."""
    import torch
    for wi in capped_windows(doc, cap):
        ids = torch.tensor([doc.windows[wi]])
        h = be.hook_states(ids)[0]
        yield wi, ids, be.center(h)


def stage_calibrate(args, cfg, contract):
    t0 = time.time()
    if not need_torch(args.out, "calibrate", t0):
        return
    if last_status(args.out, "contract").get("verdict") != "CONTRACT_PASS":
        record(args.out, "calibrate", "BLOCKED", t0, reason="contract stage has not passed")
        return
    import torch
    from .backend import Backend

    be = Backend(contract, args.cache_dir, args.allow_download, cfg["budget"]["threads"])
    docs = load_documents(contract, args.cache_dir, args.allow_download)
    calib = [d for d in docs if d.split == "calibration"]
    D.tokenize_documents(calib, be.tokenize, be.bos_id)
    cap = args.windows_per_doc
    m = be.W_enc.shape[1]
    fire = torch.zeros(m, dtype=torch.long)
    n_tok = 0
    norms: List[float] = []
    for d in calib:
        for _, _, x in _encode_windows(be, d, cap):
            x = x[1:]
            a, mask = be.act(be.sae_pre(x))
            fire += mask.sum(0)
            n_tok += x.shape[0]
            norms.extend(x.norm(dim=-1).tolist())
    cc = cfg["calibration_only"]
    pool = Q.feature_pool(fire.tolist(), n_tok, 1e-4, 1e-2)
    feats = Q.sample_features(pool, cc["n_features"], cc["feature_sampling_seed"])
    pos_acts: Dict[int, List[float]] = {j: [] for j in feats}
    fidx = torch.tensor(feats)
    for d in calib:
        for _, _, x in _encode_windows(be, d, cap):
            a = be.act(be.sae_pre(x[1:]))[0][:, fidx]
            for c, j in enumerate(feats):
                col = a[:, c]
                pos_acts[j].extend(col[col > 0].tolist())
    frozen = {
        "config_id": cfg["config_id"], "contract_id": contract["contract_id"], "windows_per_doc_cap": cap,
        "calibration_doc_ids": sorted(d.doc_id for d in calib), "n_calibration_tokens": n_tok,
        "pool_size": len(pool), "features": feats,
        "target_doses": Q.target_doses(pos_acts, cc["target_change_doses"]["quantiles_of_positive_calibration_activations"]),
        "norm_budgets": Q.norm_budgets(norms, cc["norm_budget_doses"]["multipliers"]),
        "test_documents_read": False,
    }
    os.makedirs(args.out, exist_ok=True)
    sha = Q.freeze(frozen, os.path.join(args.out, "calibration_frozen.json"))
    record(args.out, "calibrate", "RAN", t0, calibration_sha256=sha, n_features=len(feats), pool_size=len(pool))


def _edit_rows(be, x, j, feat_doses, budgets, kind, rng_seed):
    """All (mode, dose, method) edits for one test point. Returns list of (meta, delta or None, scale, alpha)."""
    import torch
    out = []
    E_j = be.enc_col(j)
    dirs = {"decoder": be.dec_row(j), "encoder_grad": E_j,
            "random": torch.randn(x.shape[0], generator=torch.Generator().manual_seed(rng_seed))}
    ln_unit = be.jacobian_ln(x, j, 1.0)
    dirs["jacobian_ln"] = ln_unit["delta"]
    pre = be.sae_pre(x)
    p_j = pre[j].item()
    a_j = max(p_j, 0.0)
    g_dec = (E_j @ be.dec_row(j)).item()
    for bname, rho in budgets.items():
        for meth in METHODS_EQUAL_NORM:
            u = dirs[meth]
            sign = 1.0 if (E_j @ u).item() >= 0 else -1.0
            delta = sign * rho * u / u.norm()
            out.append(({"mode": "equal_norm", "dose": bname, "method": meth, "status": "OK"}, delta, rho, rho))
    for dname, alpha in feat_doses.items():
        for meth in METHODS_MATCHED:
            if meth == "jacobian_ln":
                sol = be.jacobian_ln(x, j, a_j + alpha - p_j)
                out.append(({"mode": "target_matched", "dose": dname, "method": meth, "status": "OK",
                             "rank": sol["rank"], "n_constraints": sol["n_constraints"],
                             "residual_rel": sol["residual_rel"], "solve_seconds": sol["solve_seconds"]},
                            sol["delta"], alpha, alpha))
                continue
            s = be.match_scale(x, j, dirs[meth], alpha)
            meta = {"mode": "target_matched", "dose": dname, "method": meth}
            if s is None:
                out.append(({**meta, "status": "INFEASIBLE", "reason": "no s >= 0 reaches the target"}, None, alpha, alpha))
            else:
                out.append(({**meta, "status": "OK", "scale_s": s}, s * dirs[meth], alpha, alpha))
        # deployable, unmatched baselines
        if g_dec > 0:
            out.append(({"mode": "target_matched", "dose": dname, "method": "decoder_calib_rescaled",
                         "status": "OK_UNMATCHED"}, (alpha / g_dec) * be.dec_row(j), alpha, alpha))
        else:
            out.append(({"mode": "target_matched", "dose": dname, "method": "decoder_calib_rescaled",
                         "status": "INFEASIBLE", "reason": "E_j . D_j <= 0"}, None, alpha, alpha))
        out.append(({"mode": "target_matched", "dose": dname, "method": "decoder_plain", "status": "OK_UNMATCHED"},
                    alpha * be.dec_row(j), alpha, alpha))
    return out


def _run_group(be, ids, pos, x, j, rows, with_quality, K):
    """Re-encode all edits of one test point; optionally measure quality with one batched forward."""
    import torch
    from .drift import decompose_torch

    ok = [(m, d, s, a) for (m, d, s, a) in rows if d is not None]
    results = [dict(m) for (m, d, s, a) in rows if d is None]
    if not ok:
        return results
    deltas = torch.stack([d for (_, d, _, _) in ok])
    scales = torch.tensor([s for (_, _, s, _) in ok])
    alphas = torch.tensor([a for (_, _, _, a) in ok])
    a0, m0 = be.act(be.sae_pre(x))
    a1, m1 = be.act(be.sae_pre(x[None, :] + deltas))
    B = deltas.shape[0]
    dec = decompose_torch(a0.expand(B, -1), a1, m0.expand(B, -1), m1, torch.full((B,), j), alphas, scales)
    q = {}
    if with_quality:
        t0 = time.perf_counter()
        T = ids.shape[1]
        stop = min(pos + K, T - 1)
        clean = be.logits_slice(be.final_hidden(ids), pos, stop).log_softmax(-1)
        lp_edit = []
        for c in range(0, B, 16):
            sub = deltas[c:c + 16]
            hid = be.final_hidden(ids.expand(sub.shape[0], -1), torch.full((sub.shape[0],), pos), sub)
            lp_edit.append(be.logits_slice(hid, pos, stop).log_softmax(-1))
        lp = torch.cat(lp_edit)
        kl = (clean.exp() * (clean - lp)).sum(-1).mean(-1)  # KL(P_clean || P_edited), mean over positions
        true_next = ids[0, pos + 1:stop + 1]
        nll_clean = -clean[0, torch.arange(stop - pos), true_next].mean()
        nll_edit = -lp[:, torch.arange(stop - pos), true_next].mean(-1)
        q = {"kl": kl, "dnll": nll_edit - nll_clean, "forward_seconds": time.perf_counter() - t0}
    for b, (m, d, s, a) in enumerate(ok):
        r = dict(m)
        for k, v in dec.items():
            r[k] = float(v[b])
        r["edit_norm"] = float(d.norm())
        r["delta_mean_component"] = float(d.mean())
        if q:
            r["kl_next_token"] = float(q["kl"][b])
            r["dnll_true_next"] = float(q["dnll"][b])
            r["forward_seconds_group"] = q["forward_seconds"]
        results.append(r)
    return results


def stage_smoke_or_run(args, cfg, contract, stage: str):
    t0 = time.time()
    if not need_torch(args.out, stage, t0):
        return
    cal = last_status(args.out, "calibrate")
    if cal.get("status") != "RAN":
        record(args.out, stage, "BLOCKED", t0, reason="calibration not frozen")
        return
    import torch
    from .backend import Backend

    frozen = Q.load_frozen(os.path.join(args.out, "calibration_frozen.json"), cal["calibration_sha256"])
    be = Backend(contract, args.cache_dir, args.allow_download, cfg["budget"]["threads"])
    docs = load_documents(contract, args.cache_dir, args.allow_download)
    split = "calibration" if stage == "smoke" else "test"  # smoke never touches test documents
    use = sorted((d for d in docs if d.split == split), key=lambda d: d.doc_id)
    if stage == "run" and len(use) < cfg["data"]["min_test_documents"]:
        record(args.out, stage, "BLOCKED", t0, reason=f"only {len(use)} test documents")
        return
    D.tokenize_documents(use, be.tokenize, be.bos_id)
    if stage == "run":
        bad = D.check_disjoint(use + [d for d in docs if d.split == "calibration"])
        if bad:
            record(args.out, stage, "BLOCKED", t0, reason="calibration/test overlap", examples=bad[:5])
            return
    feats = frozen["features"]
    budgets = frozen["norm_budgets"]["budgets"]
    cap = frozen["windows_per_doc_cap"]
    n_docs = args.n_docs
    n_quality = args.quality_docs
    if stage == "smoke":
        feats, use, n_docs, n_quality = feats[:2], use[:2], 1, 1
    # scan: active positions of selected features per (doc, window)
    fidx = torch.tensor(feats)
    cache = {}
    active_pos: Dict[int, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for d in use:
        for wi, ids, x in _encode_windows(be, d, cap):
            cache[(d.doc_id, wi)] = (ids, x)
            act = (be.sae_pre(x[1:])[:, fidx] > 0)
            T = ids.shape[1]
            for c, j in enumerate(feats):
                # positions 1..T-2 only, so the quality window after the edit is never empty
                ps = [p for p in (torch.nonzero(act[:, c]).flatten() + 1).tolist() if p <= T - 2]
                if ps:
                    active_pos[j][d.doc_id].append((wi, ps))
    rows_out = []
    K = 16
    for j in feats:
        fd = frozen["target_doses"][str(j)] if str(j) in frozen["target_doses"] else frozen["target_doses"][j]
        if fd.get("status") != "OK":
            rows_out.append({"feature": j, "status": "SKIP_NO_DOSE"})
            continue
        doses = fd["doses"]
        order = [d.doc_id for d in use]
        D.seeded_rng("docorder", j).shuffle(order)
        act_docs = [di for di in order if di in active_pos[j]][:n_docs]
        inact_docs = order[:n_docs]
        for kind, dlist in (("active", act_docs), ("inactive", inact_docs)):
            for k_i, di in enumerate(dlist):
                rng = D.seeded_rng("pos", di, j, kind)
                if kind == "active":
                    wi, ps = rng.choice(active_pos[j][di])
                    pos = rng.choice(ps)
                else:
                    wis = sorted(w for (dd, w) in cache if dd == di)
                    wi = rng.choice(wis)
                    ids, x = cache[(di, wi)]
                    cand = [p for p in range(1, ids.shape[1] - 1)
                            if p not in dict(active_pos[j].get(di, [])).get(wi, [])]
                    pos = rng.choice(cand)
                ids, x = cache[(di, wi)]
                xp = x[pos]
                rows = _edit_rows(be, xp, j, doses, budgets, kind,
                                  rng_seed=D.seeded_rng("rand", di, j, kind).randrange(2 ** 31))
                res = _run_group(be, ids, pos, xp, j, rows, with_quality=k_i < n_quality, K=K)
                for r in res:
                    r.update({"doc_id": di, "feature": j, "target_kind": kind, "window": wi, "position": pos})
                rows_out.extend(res)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"raw_{stage}.csv")
    keys = sorted({k for r in rows_out for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_out)
    st = defaultdict(int)
    for r in rows_out:
        st[r.get("status", "?")] += 1
    extra = {}
    if stage == "smoke":
        # Projection rule fixed in the config: the largest N in [8, 16, 32] whose projected wall
        # clock fits the budget. Uses calibration documents only.
        n_groups = sum(1 for _ in {(r["doc_id"], r["feature"], r["target_kind"]) for r in rows_out if "doc_id" in r})
        per_group = (time.time() - t0) / max(n_groups, 1)
        budget_s = 60 * cfg["budget"]["wall_clock_minutes_max"]
        n_feat = len(frozen["features"])
        proj = {n: per_group * n_feat * len(TARGET_KINDS) * n for n in (8, 16, 32)}
        fit = [n for n, sec in proj.items() if sec <= budget_s]
        extra = {"seconds_per_group_upper_bound": per_group, "projected_seconds": proj,
                 "recommended_n_docs": max(fit) if fit else None,
                 "note": "upper bound: includes model/SAE loading and scanning; if None, the run does not fit the budget"}
    record(args.out, stage, "RAN", t0, raw=path, n_rows=len(rows_out), row_status_counts=dict(st),
           split_used=split, n_docs=n_docs, quality_docs=n_quality, windows_per_doc_cap=cap, **extra)


def stage_summarize(args, cfg, contract):
    """Curves, paired differences and G1-G3 readouts with document-cluster bootstrap (stdlib)."""
    t0 = time.time()
    path = os.path.join(args.out, "raw_run.csv")
    if not os.path.exists(path):
        record(args.out, "summarize", "NOT_RUN", t0, reason="no raw_run.csv")
        return
    rows = list(csv.DictReader(open(path)))

    def num(k):
        def f(r):
            v = r.get(k, "")
            return float(v) if v not in ("", None) else None
        return f

    out = {"curves": [], "paired": [], "gates": {}, "infeasible_counts": {}}
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("mode"), r.get("dose"), r.get("method"), r.get("target_kind"))].append(r)
    for (mode, dose, meth, kind), rs in sorted(groups.items(), key=lambda t: str(t[0])):
        st = defaultdict(int)
        for r in rs:
            st[r.get("status")] += 1
        out["infeasible_counts"][f"{mode}|{dose}|{meth}|{kind}"] = dict(st)
        ok = [r for r in rs if r.get("status", "").startswith("OK")]
        for metric in ("target_err_rel", "drift_orig_active_rel", "drift_newly_active_rel", "drift_all_nontarget_rel",
                       "kl_next_token", "edit_norm"):
            ci = cluster_bootstrap(per_unit_means(ok, "doc_id", num(metric)))
            out["curves"].append({"mode": mode, "dose": dose, "method": meth, "target_kind": kind, "metric": metric, **ci})
    key = lambda r: (r["doc_id"], r["feature"], r["target_kind"], r["mode"], r["dose"])
    ok_rows = [r for r in rows if r.get("status", "").startswith("OK")]
    for mode, a, b in (("target_matched", "jacobian_ln", "decoder"), ("target_matched", "jacobian_ln", "encoder_grad"),
                       ("equal_norm", "jacobian_ln", "decoder"), ("equal_norm", "encoder_grad", "decoder")):
        for metric in ("drift_all_nontarget_rel", "drift_orig_active_rel", "drift_newly_active_rel", "kl_next_token"):
            sub = [r for r in ok_rows if r["mode"] == mode]
            for kind in TARGET_KINDS:
                diffs = paired_unit_differences([r for r in sub if r["target_kind"] == kind], "doc_id", key,
                                                "method", a, b, num(metric))
                out["paired"].append({"mode": mode, "a": a, "b": b, "metric": metric, "target_kind": kind,
                                      **cluster_bootstrap(diffs)})
    # G1-G3 readouts at the q99 target dose (operational, not significance tests)
    q99 = [r for r in ok_rows if r["mode"] == "target_matched" and r["dose"] == "q99"]
    g1 = cluster_bootstrap(per_unit_means([r for r in q99 if r["method"] == "decoder_calib_rescaled"], "doc_id",
                                          num("drift_orig_active_rel")))
    g2 = cluster_bootstrap(per_unit_means([r for r in q99 if r["method"] == "decoder_calib_rescaled"], "doc_id",
                                          lambda r: (num("drift_newly_active_rel")(r) or 0.0) - (num("drift_orig_active_rel")(r) or 0.0)))
    g3 = cluster_bootstrap(paired_unit_differences(q99, "doc_id", key, "method", "jacobian_ln", "decoder",
                                                   num("drift_all_nontarget_rel")))

    def side(ci, thr, stop_if_below):
        if ci["lo"] is None:
            return "NOT_RUN"
        if stop_if_below:
            return "STOP_SIDE" if ci["hi"] < thr else ("CONTINUE_SIDE" if ci["lo"] >= thr else "INCONCLUSIVE")
        return "STOP_SIDE" if ci["lo"] >= thr else ("CONTINUE_SIDE" if ci["hi"] < thr else "INCONCLUSIVE")

    out["gates"] = {"G1": {**g1, "readout": side(g1, 0.1, True)}, "G2": {**g2, "readout": side(g2, 0.0, False)},
                    "G3": {**g3, "readout": side(g3, 0.0, False)},
                    "note": "operational readouts (config gate_reinterpretation); not significance tests"}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    record(args.out, "summarize", "RAN", t0, gates={k: v.get("readout") for k, v in out["gates"].items() if isinstance(v, dict)})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["check-env", "contract", "calibrate", "smoke", "run", "summarize"])
    ap.add_argument("--config", default="configs/p3_real01r.json")
    ap.add_argument("--out", default="results/real01r")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--allow-download", action="store_true", help="requires explicit approval")
    ap.add_argument("--windows-per-doc", type=int, default=8)
    ap.add_argument("--n-docs", type=int, default=8, help="per feature and target kind; fixed from the smoke stage")
    ap.add_argument("--quality-docs", type=int, default=2, help="documents per (feature, kind) with quality metrics")
    args = ap.parse_args(argv)
    cfg = json.load(open(args.config))
    contract = load_contract(cfg["contract"])
    {"check-env": stage_check_env, "contract": stage_contract, "calibrate": stage_calibrate,
     "summarize": stage_summarize}.get(args.stage, lambda a, c, k: stage_smoke_or_run(a, c, k, args.stage))(args, cfg, contract)
    return 0


if __name__ == "__main__":
    sys.exit(main())
