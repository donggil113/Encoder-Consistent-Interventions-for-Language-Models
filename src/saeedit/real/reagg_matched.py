"""Matched-denominator re-aggregation of the REAL-01R pilot (standard library only).

    PYTHONPATH=src python3 -m saeedit.real.reagg_matched --out results/real01r_reagg

Reads results/real01r/raw_run.csv and calibration_frozen.json (never rewrites them) and writes
new derived files only. Rules, fixed in this file before the tables were produced:

* matched row: status OK and target_err_rel <= TOL (1e-6). The stored alpha and scale were
  float32 tensors in _run_group, so a numerically exact float64 edit shows target_err_rel
  up to ~6e-8 (float32 rounding of the reference alpha); TOL sits above that and far below
  any real miss.
* OK_UNMATCHED rows (decoder_calib_rescaled, decoder_plain) are deployable baselines whose
  target change is not matched; they are counted and summarised, never pooled into
  target-matched effect tables.
* mean_only rows: their stored readout is the raw off-slice readout E(x + delta) - E(x)
  (P3-READOUT-CLOSURE). The canonical readout of a mean-only edit is 0, so no target-matched
  mean_only row exists canonically: they are excluded from matched tables and rankings and
  counted as RAW_OFF_SLICE_ONLY.
* feasible intersection: the test points (doc, feature, kind) at a dose where every method
  in a set is matched. Primary set {decoder, encoder_grad, jacobian_ln}; secondary set adds
  random. Effects are estimated on the intersection and on all available rows.
* unit: test document (document-cluster bootstrap, 2000 resamples, seed 0, median of
  per-document means), as in the pilot summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
from collections import Counter, defaultdict

from .stats import cluster_bootstrap, paired_unit_differences, per_unit_means

TOL = 1e-6
PRIMARY = ("decoder", "encoder_grad", "jacobian_ln")
SECONDARY = PRIMARY + ("random",)
METRICS = ("drift_all_nontarget_rel", "drift_orig_active_rel", "drift_newly_active_rel", "edit_norm",
           "kl_next_token", "dnll_true_next")
PILOT_FEATURES = (2360, 8852, 9608, 15617, 17696, 20347, 20747, 22630)
DOSES = ("q50", "q90", "q99", "2xq99")


def fnum(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else None


def matched(r) -> bool:
    return (r["mode"] == "target_matched" and r["status"] == "OK" and r["method"] != "mean_only"
            and fnum(r, "target_err_rel") is not None and fnum(r, "target_err_rel") <= TOL)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/real01r/raw_run.csv")
    ap.add_argument("--frozen", default="results/real01r/calibration_frozen.json")
    ap.add_argument("--summary", default="results/real01r/summary.json")
    ap.add_argument("--out", default="results/real01r_reagg")
    args = ap.parse_args(argv)
    t0, c0 = time.time(), time.process_time()
    rows = list(csv.DictReader(open(args.raw)))
    os.makedirs(args.out, exist_ok=True)
    point = lambda r: (r["doc_id"], r["feature"], r["target_kind"])

    # 1. status counts and exclusion reasons per (mode, dose, method, kind)
    cnt = defaultdict(Counter)
    for r in rows:
        k = (r["mode"], r["dose"], r["method"], r["target_kind"])
        if r["mode"] == "target_matched" and r["method"] == "mean_only":
            cnt[k]["RAW_OFF_SLICE_ONLY"] += 1
        elif r["status"] == "OK" and r["mode"] == "target_matched" and not matched(r):
            cnt[k]["OK_BUT_ABOVE_TOL"] += 1
        else:
            cnt[k][r["status"]] += 1
        if r["status"] == "INFEASIBLE":
            cnt[k]["reason: " + r["reason"]] += 1
    cols = ["OK", "OK_BUT_ABOVE_TOL", "OK_UNMATCHED", "INFEASIBLE", "RAW_OFF_SLICE_ONLY"]
    with open(os.path.join(args.out, "status_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "dose", "method", "target_kind", "n_rows"] + cols + ["n_docs_with_row", "reasons"])
        for k in sorted(cnt):
            c = cnt[k]
            docs = {r["doc_id"] for r in rows if (r["mode"], r["dose"], r["method"], r["target_kind"]) == k}
            reasons = "; ".join(f"{x[8:]} x{n}" for x, n in c.items() if x.startswith("reason: "))
            w.writerow(list(k) + [sum(c[s] for s in cols)] + [c[s] for s in cols] + [len(docs), reasons])
    totals = Counter()
    for r in rows:
        totals[r["status"]] += 1
    tm_ok = [r for r in rows if r["mode"] == "target_matched" and r["status"] == "OK"]
    target_err_max = {m: max((fnum(r, "target_err_rel") for r in tm_ok if r["method"] == m), default=None)
                      for m in SECONDARY + ("mean_only",)}

    # 2. feasible intersections at each dose and kind
    by_pt = defaultdict(dict)
    for r in rows:
        if r["mode"] == "target_matched":
            by_pt[(r["dose"], r["target_kind"], point(r))][r["method"]] = r
    inter, eff, pair = [], [], []
    for dose in DOSES:
        for kind in ("active", "inactive"):
            pts = {p for (d, k, p) in by_pt if d == dose and k == kind}
            for set_name, mset in (("primary", PRIMARY), ("secondary", SECONDARY)):
                keep = {p for p in pts if all(m in by_pt[(dose, kind, p)] and matched(by_pt[(dose, kind, p)][m])
                                              for m in mset)}
                ex = {m: sum(1 for p in pts if not (m in by_pt[(dose, kind, p)] and matched(by_pt[(dose, kind, p)][m])))
                      for m in mset}
                inter.append({"dose": dose, "target_kind": kind, "set": set_name, "n_points": len(pts),
                              "n_intersection": len(keep), "exclusion_rate": 1 - len(keep) / len(pts),
                              "n_docs_all": len({p[0] for p in pts}), "n_docs_intersection": len({p[0] for p in keep}),
                              **{f"not_matched_{m}": ex[m] for m in mset}})
                if dose != "q99":
                    continue
                for m in mset:
                    for metric in METRICS:
                        allr = [by_pt[(dose, kind, p)][m] for p in pts
                                if m in by_pt[(dose, kind, p)] and matched(by_pt[(dose, kind, p)][m])]
                        inr = [by_pt[(dose, kind, p)][m] for p in keep]
                        a = cluster_bootstrap(per_unit_means(allr, "doc_id", lambda r: fnum(r, metric)))
                        b = cluster_bootstrap(per_unit_means(inr, "doc_id", lambda r: fnum(r, metric)))
                        eff.append({"dose": dose, "target_kind": kind, "set": set_name, "method": m, "metric": metric,
                                    "all_n_docs": a["n_units"], "all_est": a["estimate"], "all_lo": a["lo"], "all_hi": a["hi"],
                                    "int_n_docs": b["n_units"], "int_est": b["estimate"], "int_lo": b["lo"], "int_hi": b["hi"]})
            # paired LN - baseline on the primary intersection, every dose
            keep = {p for p in pts if all(m in by_pt[(dose, kind, p)] and matched(by_pt[(dose, kind, p)][m])
                                          for m in PRIMARY)}
            sub = [r for p in keep for r in by_pt[(dose, kind, p)].values() if r["method"] in PRIMARY]
            for b_m in ("decoder", "encoder_grad"):
                for metric in METRICS:
                    ci = cluster_bootstrap(paired_unit_differences(
                        sub, "doc_id", point, "method", "jacobian_ln", b_m, lambda r: fnum(r, metric)))
                    pair.append({"dose": dose, "target_kind": kind, "a": "jacobian_ln", "b": b_m, "metric": metric,
                                 "n_docs": ci["n_units"], "est": ci["estimate"], "lo": ci["lo"], "hi": ci["hi"]})
    for name, tab in (("intersections.csv", inter), ("q99_effects_all_vs_intersection.csv", eff),
                      ("paired_ln_on_primary_intersection.csv", pair)):
        with open(os.path.join(args.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in tab for k in r)), restval="")
            w.writeheader()
            w.writerows(tab)

    # 3. OK_UNMATCHED baselines: how far from the target they land
    unm = {}
    for m in ("decoder_calib_rescaled", "decoder_plain"):
        for kind in ("active", "inactive"):
            for dose in DOSES:
                rs = [r for r in rows if r["method"] == m and r["target_kind"] == kind and r["dose"] == dose
                      and r["status"] == "OK_UNMATCHED"]
                ci = cluster_bootstrap(per_unit_means(rs, "doc_id", lambda r: fnum(r, "target_err_rel")))
                unm[f"{m}|{kind}|{dose}"] = {"n_rows": len(rs), "n_within_tol": sum(fnum(r, "target_err_rel") <= TOL for r in rs),
                                             "target_err_rel_median_doc": ci["estimate"], "lo": ci["lo"], "hi": ci["hi"]}

    # 4. effective calibration counts behind the q99 dose of the pilot features
    fz = json.load(open(args.frozen))
    cal = []
    for j in PILOT_FEATURES:
        t = fz["target_doses"][str(j)]
        n = t["n_positive"]
        cal.append({"feature": j, "n_positive_calibration": n, "q99": t["doses"]["q99"],
                    "q99_rank_position": 0.99 * (n - 1), "between_top_two_order_stats": 0.99 * (n - 1) >= n - 2,
                    "a_max": t["a_max_recorded_not_used"], "q99_over_max": t["doses"]["q99"] / t["a_max_recorded_not_used"]})
    with open(os.path.join(args.out, "q99_calibration_counts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cal[0].keys()))
        w.writeheader()
        w.writerows(cal)
    all_n = sorted(v["n_positive"] for v in fz["target_doses"].values() if "n_positive" in v)

    # 5. G1-G3 as run (in-run amendment labels) and split by target kind
    q99 = [r for r in rows if r["mode"] == "target_matched" and r["dose"] == "q99" and r["status"].startswith("OK")]
    key = lambda r: (r["doc_id"], r["feature"], r["target_kind"], r["mode"], r["dose"])
    resc = lambda rs: [r for r in rs if r["method"] == "decoder_calib_rescaled"]
    g = {"G1_as_run": cluster_bootstrap(per_unit_means(resc(q99), "doc_id", lambda r: fnum(r, "drift_orig_active_rel")))}
    for kind in ("active", "inactive"):
        rk = [r for r in q99 if r["target_kind"] == kind]
        g[f"G1_{kind}_only"] = {**cluster_bootstrap(per_unit_means(resc(rk), "doc_id", lambda r: fnum(r, "drift_orig_active_rel"))),
                                "rescaled_rows_within_tol": sum(fnum(r, "target_err_rel") <= TOL for r in resc(rk)),
                                "rescaled_rows": len(resc(rk))}
    g["G3_as_run"] = cluster_bootstrap(paired_unit_differences(q99, "doc_id", key, "method", "jacobian_ln", "decoder",
                                                               lambda r: fnum(r, "drift_all_nontarget_rel")))
    stored = json.load(open(args.summary))["gates"]
    g["stored_summary_gates"] = {k: {kk: v.get(kk) for kk in ("estimate", "lo", "hi", "n_units", "readout")}
                                 for k, v in stored.items() if isinstance(v, dict)}
    g["labels"] = {
        "MIN_UNITS_FOR_READOUT=10 and target_gain": "in-run amendment: commit 6a37028 (2026-09-27 00:17:39 UTC) landed while the pilot run stage was executing (started ~00:16:32, raw_run.csv written 00:33:54); not in the frozen pilot config (commit 6b4b273, 00:06:19); written before any test-split row existed on disk",
        "G1_as_run": "pools active and inactive targets; for inactive targets decoder_calib_rescaled is unmatched (see unmatched_baselines), so G1 as run mixes a matched and an unmatched denominator",
    }

    rep = {"experiment_id": "P3-REAL-01R-PILOT re-aggregation (matched denominators)",
           "rules": {"tolerance_target_err_rel": TOL, "primary_set": PRIMARY, "secondary_set": SECONDARY,
                     "mean_only_target_matched": "RAW_OFF_SLICE_ONLY (excluded)",
                     "ok_unmatched": "reported separately, never in matched tables"},
           "inputs": {p: sha256(p) for p in (args.raw, args.frozen, args.summary)},
           "row_status_totals_stored": dict(totals),
           "target_err_rel_max_over_OK_target_matched_rows": target_err_max,
           "target_err_note": "alpha was stored as a float32 tensor in _run_group; ~5e-8 is float32 rounding of the reference, not an edit miss",
           "intersections": inter, "unmatched_baselines": unm,
           "q99_calibration_counts": cal,
           "calibration_n_positive_all_64": {"min": all_n[0], "median": statistics.median(all_n), "max": all_n[-1],
                                              "n_below_100": sum(n < 100 for n in all_n)},
           "gates": g,
           "timing": {"wall_seconds": time.time() - t0, "cpu_seconds_process": time.process_time() - c0, "threads": 1}}
    with open(os.path.join(args.out, "reagg_summary.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: rep[k] for k in ("row_status_totals_stored", "target_err_rel_max_over_OK_target_matched_rows",
                                          "calibration_n_positive_all_64", "timing")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
