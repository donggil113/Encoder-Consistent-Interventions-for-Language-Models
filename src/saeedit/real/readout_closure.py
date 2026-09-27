"""P3-READOUT-CLOSURE: raw off-slice vs canonical reprojected SAE readout.

    HF_HOME=<cache> HF_HUB_OFFLINE=1 PYTHONPATH=src <venv>/bin/python -m saeedit.real.readout_closure \
        --out results/readout_closure

Two readouts of an edit delta applied to the HF hidden state h at the layer-8 hook
(x = Ph is the SAE input, P = I - 11^T/d):

* raw off-slice readout      E(x + delta) - E(x)       (what contract K9 and the REAL-01R
                                                          pilot computed: sae.encode(x0 + delta),
                                                          be.sae_pre(x + deltas))
* canonical reprojected      E(P(h + delta)) - E(Ph) = E(x + P delta) - E(x)

They agree exactly when delta is in range(P). They differ for any delta with a
component along 1, in particular for the mean-only control delta = c 1/sqrt(d),
whose canonical readout is 0. The model sees h + delta; its logits are checked
against h + P delta.

Recomputes on existing small cases only (no new data, no new features):
  part A: the 8 contract-exec K9 windows/positions (HF path; K3 showed the SAELens
          encoder equals the manual encoder in float64), to confirm that the stored
          K9 'sae_drift_rel_median' is the raw off-slice readout;
  part B: 16 pilot test points of results/real01r/raw_run.csv (for each pilot feature and
          target kind, the first document in file order), all methods and doses, with the
          stored raw-readout columns reproduced and the canonical columns added.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import resource
import sys
import time

from . import data as D
from .contract import load_contract

COMPARE = ("target_change", "target_err_rel", "drift_orig_active_rel", "drift_newly_active_rel",
           "drift_all_nontarget_rel", "n_newly_active")


def sha(t) -> str:
    """sha256 of a tensor's float64 little-endian bytes (thread-count dependent at the last bit)."""
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(t.detach().double().cpu().numpy()).tobytes()).hexdigest()


def part_a(be, contract, window_seed: str, stored: dict):
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download

    dc = contract["data"]
    p_val = hf_hub_download(dc["hf_repo"], "wikitext-103-raw-v1/validation-00000-of-00001.parquet",
                            repo_type="dataset", revision=dc["revision"], local_files_only=True)
    docs = D.build_documents({"calibration": pq.read_table(p_val).column("text").to_pylist()})
    rng = random.Random(window_seed)
    D.tokenize_documents(docs, be.tokenize, be.bos_id)
    all_w = [(di, wi) for di, doc in enumerate(docs) for wi in range(len(doc.windows))]
    parity = rng.sample(all_w, 64)[:16]  # same draw as contract_exec (64 reconstruction windows, first 16 parity)

    def ids_of(dw):
        return torch.tensor([docs[dw[0]].windows[dw[1]]])

    d = be.W_dec.shape[1]
    med = float(torch.cat([be.center(be.hook_states(ids_of(dw)))[0, 1:].norm(dim=-1) for dw in parity[:4]]).median())
    rho = 0.1 * med
    ones = torch.ones(d, dtype=be.dtype) / d ** 0.5
    delta = rho * ones
    pdelta = be.proj(delta)
    rows = []
    for wi, dw in enumerate(parity[:8]):
        ids = ids_of(dw)
        T = ids.shape[1]
        pos = random.Random(f"k7-pos-{wi}").randrange(1, T - 16)
        h = be.hook_states(ids)[0]
        x = be.center(h)
        x0 = x[pos]
        a0 = be.act(be.sae_pre(x0))[0]
        a_raw = be.act(be.sae_pre(x0 + delta))[0]
        a_can = be.act(be.sae_pre(be.center(h[pos] + delta)))[0]  # E(P(h + delta))
        sl = slice(pos, T)
        lg0 = be.logits_slice(be.final_hidden(ids), 0, T)[0, sl].log_softmax(-1)
        lg_d = be.logits_slice(be.final_hidden(ids, torch.tensor([pos]), delta[None]), 0, T)[0, sl].log_softmax(-1)
        lg_p = be.logits_slice(be.final_hidden(ids, torch.tensor([pos]), pdelta[None]), 0, T)[0, sl].log_softmax(-1)
        rows.append({
            "window": list(dw), "position": pos,
            "raw_drift_rel": float((a_raw - a0).norm() / rho),
            "raw_max_single_feature_change": float((a_raw - a0).abs().max()),
            "canonical_drift_rel": float((a_can - a0).norm() / rho),
            "canonical_max_single_feature_change": float((a_can - a0).abs().max()),
            "kl_clean_vs_h_plus_delta_max": float((lg0.exp() * (lg0 - lg_d)).sum(-1).max()),
            "max_abs_logprob_diff_h_plus_delta_vs_h_plus_Pdelta": float((lg_d - lg_p).abs().max()),
            "sha256": {"h": sha(h), "x_Ph": sha(x), "a0": sha(a0), "a_raw": sha(a_raw), "a_canonical": sha(a_can),
                       "logprobs_clean": sha(lg0), "logprobs_h_plus_delta": sha(lg_d)},
        })
    raw = sorted(r["raw_drift_rel"] for r in rows)
    med_raw = (raw[3] + raw[4]) / 2
    return {
        "rho": rho, "rho_stored_k9": stored["rho"], "rho_abs_diff": abs(rho - stored["rho"]),
        "norm_P_delta": float(pdelta.norm()), "delta_norm": float(delta.norm()),
        "raw_drift_rel_median": med_raw, "stored_k9_sae_drift_rel_median": stored["sae_drift_rel_median"],
        "raw_vs_stored_rel_diff": abs(med_raw - stored["sae_drift_rel_median"]) / stored["sae_drift_rel_median"],
        "raw_drift_rel_max": max(raw), "stored_k9_sae_drift_rel_max": stored["sae_drift_rel_max"],
        "canonical_drift_rel_max": max(r["canonical_drift_rel"] for r in rows),
        "kl_clean_vs_h_plus_delta_max": max(r["kl_clean_vs_h_plus_delta_max"] for r in rows),
        "max_abs_logprob_diff_delta_vs_Pdelta": max(r["max_abs_logprob_diff_h_plus_delta_vs_h_plus_Pdelta"]
                                                    for r in rows),
        "per_window": rows,
    }


def part_b(be, contract, raw_csv: str, frozen_path: str, frozen_sha: str):
    import torch
    from . import dose as Q
    from .drift import decompose_torch
    from .real01r import _edit_rows, load_documents

    fz = Q.load_frozen(frozen_path, frozen_sha)
    stored = list(csv.DictReader(open(raw_csv)))
    first = {}
    for r in stored:
        k = (int(r["feature"]), r["target_kind"])
        if k not in first:
            first[k] = (r["doc_id"], int(r["window"]), int(r["position"]))
    docs = {d.doc_id: d for d in load_documents(contract, None, False) if d.split == "test"}
    need = sorted({v[0] for v in first.values()})
    D.tokenize_documents([docs[i] for i in need], be.tokenize, be.bos_id)
    budgets = fz["norm_budgets"]["budgets"]
    out_rows, groups = [], []
    for (j, kind), (di, wi, pos) in sorted(first.items()):
        doses = fz["target_doses"][str(j)]["doses"]
        ids = torch.tensor([docs[di].windows[wi]])
        T = ids.shape[1]
        h = be.hook_states(ids)[0]
        x = be.center(h)
        xp = x[pos]
        rows = _edit_rows(be, xp, j, doses, budgets, kind,
                          rng_seed=D.seeded_rng("rand", di, j, kind).randrange(2 ** 31))
        ok = [(m, dl, s, a) for (m, dl, s, a) in rows if dl is not None]
        # extra diagnostic row: unprojected decoder direction at the largest norm budget
        bmax = max(budgets, key=budgets.get)
        raw_dec = budgets[bmax] * be.dec_row(j) / be.dec_row(j).norm()
        ok.append(({"mode": "equal_norm", "dose": bmax, "method": "decoder_unprojected_diagnostic",
                    "status": "DIAGNOSTIC", "admissible": 0}, raw_dec, budgets[bmax], budgets[bmax]))
        deltas = torch.stack([dl for (_, dl, _, _) in ok])
        pdeltas = be.proj(deltas)
        scales = torch.tensor([s for (_, _, s, _) in ok])
        alphas = torch.tensor([a for (_, _, _, a) in ok])
        B = deltas.shape[0]
        a0, m0 = be.act(be.sae_pre(xp))
        a_raw, m_raw = be.act(be.sae_pre(xp[None, :] + deltas))
        a_can, m_can = be.act(be.sae_pre(be.center(h[pos][None, :] + deltas)))  # E(P(h + delta))
        jj = torch.full((B,), j)
        dec_raw = decompose_torch(a0.expand(B, -1), a_raw, m0.expand(B, -1), m_raw, jj, alphas, scales)
        dec_can = decompose_torch(a0.expand(B, -1), a_can, m0.expand(B, -1), m_can, jj, alphas, scales)
        # logits: model on h + delta vs h + P delta (all rows of the group, positions pos..T-1)
        lg0 = be.logits_slice(be.final_hidden(ids), 0, T)[0, pos:].log_softmax(-1)
        dmax, kl_mean_only = 0.0, None
        for c in range(0, B, 16):
            sub, psub = deltas[c:c + 16], pdeltas[c:c + 16]
            n = sub.shape[0]
            pp = torch.full((n,), pos)
            l1 = be.logits_slice(be.final_hidden(ids.expand(n, -1), pp, sub), 0, T)[:, pos:].log_softmax(-1)
            l2 = be.logits_slice(be.final_hidden(ids.expand(n, -1), pp, psub), 0, T)[:, pos:].log_softmax(-1)
            dmax = max(dmax, float((l1 - l2).abs().max()))
            for b in range(n):
                if ok[c + b][0]["method"] == "mean_only":
                    kl = float((lg0.exp() * (lg0 - l1[b])).sum(-1).max())
                    kl_mean_only = kl if kl_mean_only is None else max(kl_mean_only, kl)
        key = {(r["mode"], r["dose"], r["method"]): r for r in stored
               if r["doc_id"] == di and int(r["feature"]) == j and r["target_kind"] == kind}
        repro = 0.0
        for b, (m, dl, s, a) in enumerate(ok):
            row = {"doc_id": di, "feature": j, "target_kind": kind, "window": wi, "position": pos,
                   "mode": m["mode"], "dose": m["dose"], "method": m["method"], "status": m["status"],
                   "admissible": m.get("admissible", ""), "edit_norm": float(dl.norm()),
                   "norm_P_delta": float(pdeltas[b].norm()), "norm_1_component": float((dl - pdeltas[b]).norm())}
            for k in COMPARE:
                row["raw_" + k] = float(dec_raw[k][b])
                row["canonical_" + k] = float(dec_can[k][b])
            sr = key.get((m["mode"], m["dose"], m["method"]))
            if sr is not None and sr.get("drift_all_nontarget_rel"):
                for k in COMPARE:
                    repro = max(repro, abs(float(sr[k]) - row["raw_" + k]) / max(1.0, abs(float(sr[k]))))
                row["stored_row_found"] = 1
            else:
                row["stored_row_found"] = 0
            out_rows.append(row)
        groups.append({"doc_id": di, "feature": j, "target_kind": kind, "window": wi, "position": pos,
                       "n_rows": B, "max_rel_diff_raw_vs_stored_csv": repro,
                       "max_abs_logprob_diff_h_plus_delta_vs_h_plus_Pdelta": dmax,
                       "max_kl_clean_vs_mean_only": kl_mean_only,
                       "sha256": {"h": sha(h), "x_Ph": sha(x), "deltas": sha(deltas), "P_deltas": sha(pdeltas),
                                  "a_raw": sha(a_raw), "a_canonical": sha(a_can), "logprobs_clean": sha(lg0)}})
    return out_rows, groups


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default="configs/p3_contract_gpt2_res_jb_l8.json")
    ap.add_argument("--contract-exec-report", default="results/contract_exec_v2/contract_exec_report.json")
    ap.add_argument("--raw", default="results/real01r/raw_run.csv")
    ap.add_argument("--frozen", default="results/real01r/calibration_frozen.json")
    ap.add_argument("--frozen-sha", default="ba9a5efae1bed14d258664df92451701f4f80914c22e68bf7916a8b5d043a1f6")
    ap.add_argument("--out", default="results/readout_closure")
    args = ap.parse_args(argv)
    t0, c0 = time.time(), time.process_time()
    from .backend import Backend

    contract = load_contract(args.contract)
    be = Backend(contract, None, False, 2, "float64")
    t_load = time.time() - t0
    stored = json.load(open(args.contract_exec_report))["checks"]["K9_mean_only_control"]
    ta = time.time()
    a = part_a(be, contract, "p3-contract-exec", stored)
    t_a = time.time() - ta
    tb = time.time()
    rows, groups = part_b(be, contract, args.raw, args.frozen, args.frozen_sha)
    t_b = time.time() - tb
    os.makedirs(args.out, exist_ok=True)
    rpath = os.path.join(args.out, "pilot_subset_raw_vs_canonical.csv")
    with open(rpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def mx(sel, k):
        v = [abs(r["raw_" + k] - r["canonical_" + k]) for r in rows if sel(r)]
        return max(v) if v else None

    adm = lambda r: r["method"] not in ("mean_only", "decoder_unprojected_diagnostic")
    mo = [r for r in rows if r["method"] == "mean_only"]
    rep = {
        "experiment_id": "P3-READOUT-CLOSURE",
        "definitions": {
            "raw_off_slice": "E(x + delta) - E(x), x = P h (contract K9: sae.encode(x0 + delta); REAL-01R _run_group: be.sae_pre(x + deltas))",
            "canonical_reprojected": "E(P(h + delta)) - E(P h)",
            "mean_only": "delta = c * 1/sqrt(d), 1 = all-ones vector in d = 768; P delta = 0",
        },
        "call_paths_checked": {
            "contract_exec.py K9": "da = sae.encode(xe_tl[None]) - sae.encode(x0[None]) with xe_tl = x0 + rho*1/sqrt(d) captured after the TL hook edit -> raw off-slice",
            "real01r.py _run_group": "a1 = be.act(be.sae_pre(x[None, :] + deltas)) -> raw off-slice; equals canonical for every projected (admissible) method, differs only for mean_only",
            "real01r.py _edit_rows / Backend.match_scale": "target-matched mean_only scale uses E_j . 1/sqrt(d) (raw readout); canonically no scale reaches the target",
        },
        "part_a_k9": {k: v for k, v in a.items() if k != "per_window"},
        "part_b_pilot_subset": {
            "n_groups": len(groups), "n_rows": len(rows),
            "selection_rule": "for each (pilot feature, target kind): first document in results/real01r/raw_run.csv file order",
            "max_rel_diff_raw_vs_stored_csv": max(g["max_rel_diff_raw_vs_stored_csv"] for g in groups),
            "admissible_methods_max_abs_raw_minus_canonical": {k: mx(adm, k) for k in COMPARE},
            "admissible_methods_max_norm_1_component": max(r["norm_1_component"] for r in rows if adm(r)),
            "mean_only_rows": len(mo),
            "mean_only_raw_drift_all_nontarget_rel_median": sorted(r["raw_drift_all_nontarget_rel"] for r in mo)[len(mo) // 2],
            "mean_only_canonical_drift_all_nontarget_rel_max": max(r["canonical_drift_all_nontarget_rel"] for r in mo),
            "mean_only_canonical_abs_target_change_max": max(abs(r["canonical_target_change"]) for r in mo),
            "mean_only_max_kl_clean_vs_edit": max(g["max_kl_clean_vs_mean_only"] for g in groups),
            "unprojected_decoder_max_abs_raw_minus_canonical_drift_all_rel": mx(
                lambda r: r["method"] == "decoder_unprojected_diagnostic", "drift_all_nontarget_rel"),
            "max_abs_logprob_diff_h_plus_delta_vs_h_plus_Pdelta": max(
                g["max_abs_logprob_diff_h_plus_delta_vs_h_plus_Pdelta"] for g in groups),
        },
        "interpretation": "Stored K9 drift and pilot mean_only rows are raw off-slice readouts: the SAE reads an input outside its training slice range(P). Canonically the mean-only edit changes no SAE feature and no logit. These numbers are an off-training-subspace stress test of the encoder, not evidence of internal change without external effect.",
        "per_window_k9": a["per_window"], "per_group_pilot": groups,
        "files": {"pilot_subset_csv": rpath},
        "timing": {"wall_seconds_total": time.time() - t0, "cpu_seconds_process": time.process_time() - c0,
                   "wall_seconds_load": t_load, "wall_seconds_part_a": t_a, "wall_seconds_part_b": t_b,
                   "threads": 2, "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "note": "cpu_seconds_process counts CPU time of all threads of this process; wall is elapsed time"},
    }
    with open(rpath, "rb") as f:
        rep["files"]["pilot_subset_csv_sha256"] = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(args.out, "readout_closure_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({"part_a": rep["part_a_k9"], "part_b": rep["part_b_pilot_subset"], "timing": rep["timing"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
