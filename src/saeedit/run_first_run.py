"""Run the P3 first-run toy suite and write raw rows, a summary and a manifest.

Usage (from repo root):
    OMP_NUM_THREADS=2 PYTHONPATH=src timeout 120 python3 -m saeedit.run_first_run \
        --config configs/p3_first_run.json --out results

Engineering status (did it run) and verification status (did the identity
hold) are recorded separately. The scientific hypothesis H_MAIN is not tested
by this script and is always recorded as NOT_RUN.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional

from . import experiments as ex

LEAD_COLS = ["experiment_id", "regime", "act", "case", "check", "seed", "base_idx", "target_kind", "target",
             "alpha", "method", "status"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(root: str) -> dict:
    def run(args):
        try:
            return subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as e:  # pragma: no cover
            return f"ERROR:{e}"
    commit = run(["git", "rev-parse", "HEAD"]) or "NO_COMMIT"
    dirty = run(["git", "status", "--porcelain", "--", "src", "configs", "tests"])
    return {"commit": commit, "code_dirty": bool(dirty), "dirty_files": dirty.splitlines()[:20]}


def write_csv(rows: List[dict], path: str) -> None:
    keys = []
    seen = set()
    for k in LEAD_COLS:
        if any(k in r for r in rows):
            keys.append(k)
            seen.add(k)
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, float):
            return "nan" if math.isnan(v) else repr(v)
        return v

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in keys})


# ---------------------------------------------------------------- aggregation


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def aggregate(rows: Iterable[dict], group_keys: List[str], metrics: List[str], unit: str = "seed") -> List[dict]:
    """Average within each independent unit first, then summarise across units."""
    per_unit: Dict[tuple, Dict[object, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    status_counts: Dict[tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        g = tuple(r.get(k) for k in group_keys)
        status_counts[g][r.get("status", "?")] += 1
        if r.get("status") != "OK":
            continue
        for mname in metrics:
            v = r.get(mname)
            if _finite(v):
                per_unit[g][r[unit]][mname].append(float(v))
    out = []
    for g in sorted(status_counts, key=lambda t: tuple(str(x) for x in t)):
        rec = dict(zip(group_keys, g))
        rec["status_counts"] = dict(status_counts[g])
        units = per_unit.get(g, {})
        rec["n_units"] = len(units)
        for mname in metrics:
            means = [statistics.fmean(u[mname]) for u in units.values() if u.get(mname)]
            if means:
                rec[mname] = {"median": statistics.median(means), "min": min(means), "max": max(means),
                              "n_units": len(means)}
            else:
                rec[mname] = None
        out.append(rec)
    return out


# ---------------------------------------------------------------- verification


def _maxv(rows, key):
    vals = [r[key] for r in rows if _finite(r.get(key))]
    return max(vals) if vals else None


def _minv(rows, key):
    vals = [r[key] for r in rows if _finite(r.get(key))]
    return min(vals) if vals else None


def _sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def _verdict(ok: Optional[bool]) -> str:
    return "NOT_RUN" if ok is None else ("PASS" if ok else "FAIL")


def verify(results: Dict[str, List[dict]], cfg: dict) -> Dict[str, dict]:
    tol = cfg["numerics"]["identity_tol"]
    V: Dict[str, dict] = {}
    lin = results.get("P3-T2-MISMATCH")
    if lin is not None:
        r0d = _sel(lin, regime="R0_pinv_undercomplete", method="decoder")
        r0j = _sel(lin, regime="R0_pinv_undercomplete", method="jacobian_ln")
        ev = {"max_target_rel_err_decoder": _maxv(r0d, "target_rel_err"),
              "max_leak_decoder": _maxv(r0d, "leak_rel_all"),
              "max_rel_diff_ln_vs_decoder": _maxv(r0j, "rel_diff_vs_decoder")}
        V["V1"] = {"verdict": _verdict(all(v is not None and v <= tol for v in ev.values())), **ev}

        r1r = _sel(lin, regime="R1_scaled_orthogonal", method="decoder_rescaled")
        r1j = _sel(lin, regime="R1_scaled_orthogonal", method="jacobian_ln")
        r1d = _sel(lin, regime="R1_scaled_orthogonal", method="decoder")
        ev = {"max_target_rel_err_rescaled": _maxv(r1r, "target_rel_err"),
              "max_leak_rescaled": _maxv(r1r, "leak_rel_all"),
              "max_rel_diff_ln_vs_rescaled": _maxv(r1j, "rel_diff_vs_decoder_rescaled")}
        ok = all(v is not None and v <= tol for v in ev.values())
        ev["min_target_rel_err_decoder_unrescaled"] = _minv(r1d, "target_rel_err")
        V["V2"] = {"verdict": _verdict(ok), **ev}

        r2 = _sel(lin, regime="R2_tied_coherent")
        dd = _sel(r2, method="decoder")
        dr = _sel(r2, method="decoder_rescaled")
        dj = _sel(r2, method="jacobian_ln")
        e_dec = max(abs(r["leak_rel_all"] - r["analytic_leak_tied"]) for r in dd)
        e_res = max(abs(r["leak_rel_all"] - r["analytic_leak_tied"]) for r in dr)
        e_nr = max(abs(r["norm_ratio"] - r["analytic_norm_ratio_ln"]) for r in dj)
        ev = {"max_abs_err_decoder_leak_vs_c_sqrt_m_minus_1": e_dec,
              "max_abs_err_rescaled_leak_vs_c_sqrt_m_minus_1": e_res,
              "max_leak_jacobian_ln": _maxv(dj, "leak_rel_all"),
              "max_abs_err_ln_norm_ratio_vs_sqrt_Ginv_jj": e_nr,
              "min_analytic_norm_ratio_ln": _minv(dj, "analytic_norm_ratio_ln")}
        ok = (e_dec <= tol and e_res <= tol and ev["max_leak_jacobian_ln"] <= tol and e_nr <= tol
              and ev["min_analytic_norm_ratio_ln"] > 1.0)
        V["V3"] = {"verdict": _verdict(ok), **ev}

        r4a = _sel(lin, regime="R4a_overcomplete_tied_protect_all", method="jacobian_ln")
        feas = [r for r in lin if r["method"] == "jacobian_ln" and r["regime"] in
                ("R4b_overcomplete_tied_protect_subset", "R5_overcomplete_pinv_protect_subset")]
        ev = {"min_residual_rel_R4a": _minv(r4a, "residual_rel"),
              "max_cert_MTr_rel_R4a": _maxv(r4a, "cert_MTr_rel"),
              "max_abs_cert_rt_minus_1_R4a": max(abs(r["cert_rt_over_rr"] - 1.0) for r in r4a if "cert_rt_over_rr" in r),
              "max_residual_rel_R4b_R5": _maxv(feas, "residual_rel")}
        ok = (ev["min_residual_rel_R4a"] > 1e-6 and ev["max_cert_MTr_rel_R4a"] <= tol
              and ev["max_abs_cert_rt_minus_1_R4a"] <= tol and ev["max_residual_rel_R4b_R5"] <= tol)
        V["V4"] = {"verdict": _verdict(ok), **ev}

        jl = _sel(lin, method="jacobian_ln")
        cg = _sel(lin, method="jacobian_ln_cgls")
        ev = {"max_rel_diff_cgls_vs_dense": _maxv(jl, "rel_diff_vs_jacobian_ln_cgls"),
              "max_cgls_iters": _maxv(cg, "cgls_iters"),
              "all_cgls_converged": all(r.get("cgls_converged") for r in cg)}
        V["V5"] = {"verdict": _verdict(ev["max_rel_diff_cgls_vs_dense"] <= 1e-6), **ev}
    else:
        for k in ("V1", "V2", "V3", "V4", "V5"):
            V[k] = {"verdict": "NOT_RUN"}

    feas = results.get("P3-T3-FEASIBILITY")
    if feas is not None:
        by = defaultdict(list)
        for r in feas:
            by[r["check"]].append(r["pass"])
        V["T3_checks"] = {"verdict": _verdict(all(all(v) for v in by.values())),
                          **{k: f"{sum(v)}/{len(v)} pass" for k, v in sorted(by.items())}}
    else:
        V["T3_checks"] = {"verdict": "NOT_RUN"}

    t4 = results.get("P3-T4-ACTIVESET-NORM")
    if t4 is not None:
        relu_ok = [r for r in t4 if r.get("act") == "relu" and r.get("status") == "OK"]
        nocross = [r for r in relu_ok if r["n_crossings"] == 0 and r["target_crossed"] == 0]
        V["V6"] = {"verdict": _verdict(all(r["local_pred_exact"] == 1 for r in nocross) if nocross else None),
                   "n_relu_rows_without_crossing": len(nocross),
                   "max_local_pred_err_without_crossing": _maxv(nocross, "local_pred_err")}
        tr = [r for r in t4 if str(r.get("method", "")).startswith("trust_region") and r.get("status") == "OK"]
        over = [r for r in tr if r["edit_norm"] > r["budget"] * (1 + 1e-9) + 1e-15]
        key = lambda r: (r["act"], r["seed"], r["base_idx"], r["target_kind"], r["alpha"])
        ln_norm = {key(r): r["edit_norm"] for r in t4 if r.get("method") == "jacobian_ln" and r.get("status") == "OK"}
        inactive_budget = [r for r in tr if not r["budget_active"]]
        mism = [r for r in inactive_budget if abs(r["edit_norm"] - ln_norm[key(r)]) > tol * max(1.0, ln_norm[key(r)])]
        V["V7"] = {"verdict": _verdict(not over and not mism), "n_trust_region_rows": len(tr),
                   "n_budget_violations": len(over), "n_inactive_budget_rows": len(inactive_budget),
                   "n_inactive_budget_norm_mismatch_vs_ln": len(mism)}
    else:
        V["V6"] = {"verdict": "NOT_RUN"}
        V["V7"] = {"verdict": "NOT_RUN"}

    tx = results.get("P3-TX-EXTPROXY")
    if tx is not None:
        out = {}
        for case in cfg["TX_external_proxy"]["cases"]:
            med = {}
            for meth in ("decoder", "jacobian_ln", "decoder_rescaled"):
                per = defaultdict(list)
                for r in _sel(tx, case=case, method=meth, status="OK"):
                    per[r["seed"]].append((math.hypot(r["int_target_rel_err"], r["int_leak_rel"]),
                                           math.hypot(r["ext_target_rel_err"], r["ext_leak_rel"])))
                ints = [statistics.fmean(v[0] for v in vals) for vals in per.values()]
                exts = [statistics.fmean(v[1] for v in vals) for vals in per.values()]
                med[meth] = {"internal_err_median": statistics.median(ints), "external_err_median": statistics.median(exts)}
            # per-instance paired comparison jacobian_ln vs decoder on the external error
            wins = 0
            n = 0
            per_dec = defaultdict(list)
            per_jl = defaultdict(list)
            for r in _sel(tx, case=case, method="decoder", status="OK"):
                per_dec[r["seed"]].append(math.hypot(r["ext_target_rel_err"], r["ext_leak_rel"]))
            for r in _sel(tx, case=case, method="jacobian_ln", status="OK"):
                per_jl[r["seed"]].append(math.hypot(r["ext_target_rel_err"], r["ext_leak_rel"]))
            for s in per_dec:
                n += 1
                wins += int(statistics.fmean(per_jl[s]) < statistics.fmean(per_dec[s]))
            out[case] = {"medians": med, "instances_where_ln_beats_decoder_externally": f"{wins}/{n}",
                         "internal_improvement": med["jacobian_ln"]["internal_err_median"] < med["decoder"]["internal_err_median"],
                         "external_improvement": med["jacobian_ln"]["external_err_median"] < med["decoder"]["external_err_median"]}
        V["V8"] = {"verdict": "DESCRIPTIVE", **out}
    else:
        V["V8"] = {"verdict": "NOT_RUN"}
    return V


# ---------------------------------------------------------------- main


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p3_first_run.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--dev", action="store_true", help="use dev seeds (debugging only; not reported)")
    ap.add_argument("--manifest", default="run_manifest.json")
    args = ap.parse_args(argv)

    t_start = time.time()
    root = os.getcwd()
    with open(args.config) as f:
        cfg = json.load(f)
    budget = cfg["budget"]["wall_clock_seconds_max"]
    sd = cfg["seeds"]
    seeds = sd["dev_seeds"] if args.dev else list(range(sd["report_seeds_start"], sd["report_seeds_start"] + sd["n_report_instances"]))
    raw_dir = os.path.join(args.out, "raw_dev" if args.dev else "raw")
    os.makedirs(raw_dir, exist_ok=True)

    plan: List[tuple] = [
        ("P3-T2-MISMATCH", "linear_regimes.csv", ex.run_linear_regimes),
        ("P3-T3-FEASIBILITY", "feasibility.csv", ex.run_feasibility),
        ("P3-T4-ACTIVESET-NORM", "activeset_norm.csv", ex.run_activeset_norm),
        ("P3-TX-EXTPROXY", "external_proxy.csv", ex.run_external_proxy),
        ("P3-T4b-SPARSITY-EXPLORATORY", "sparsity_exploratory.csv", ex.run_sparsity_exploratory),
    ]
    manifest = {
        "run_kind": "dev" if args.dev else "report",
        "config_id": cfg["config_id"],
        "config_path": args.config,
        "config_sha256": sha256_file(args.config),
        "git": git_info(root),
        "started_unix": t_start,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": "python standard library only (numpy/torch not installed; installation not approved)",
        "threads": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "note": "pure Python, single-threaded"},
        "budget_seconds": budget,
        "seeds": seeds,
        "data_models_saes": "none: fully synthetic toy; no dataset, language model or SAE weights used",
        "H_MAIN": "NOT_RUN",
        "experiments": {},
    }
    results: Dict[str, List[dict]] = {}

    def dump_manifest():
        manifest["elapsed_seconds"] = time.time() - t_start
        with open(args.manifest if not args.dev else os.path.join(args.out, "run_manifest_dev.json"), "w") as f:
            json.dump(manifest, f, indent=2, default=str)

    for eid, fname, fn in plan:
        elapsed = time.time() - t_start
        if elapsed > budget:
            manifest["experiments"][eid] = {"engineering_status": "NOT_RUN", "reason": f"budget exhausted at {elapsed:.1f}s"}
            dump_manifest()
            continue
        t0 = time.time()
        try:
            rows = fn(cfg, seeds)
        except Exception as e:  # record, never convert to PASS
            manifest["experiments"][eid] = {"engineering_status": "ERROR", "error": repr(e)}
            dump_manifest()
            continue
        path = os.path.join(raw_dir, fname)
        write_csv(rows, path)
        results[eid] = rows
        st = defaultdict(int)
        for r in rows:
            st[r.get("status", "OK" if "pass" in r else "?")] += 1
        manifest["experiments"][eid] = {"engineering_status": "RAN", "wall_seconds": time.time() - t0,
                                        "n_rows": len(rows), "row_status_counts": dict(st),
                                        "raw_file": path, "raw_sha256": sha256_file(path)}
        dump_manifest()

    # summaries: aggregate within seed first, then across seeds
    summary: Dict[str, object] = {"note": "per-instance means summarised across instances (median/min/max); n_units = instances"}
    if "P3-T2-MISMATCH" in results:
        summary["P3-T2-MISMATCH"] = aggregate(results["P3-T2-MISMATCH"], ["regime", "method"],
                                              ["target_rel_err", "leak_rel_all", "leak_rel_protected",
                                               "leak_rel_unprotected", "norm_ratio", "residual_rel",
                                               "mismatch_diag_share", "n_jvp", "n_vjp"])
    if "P3-T4-ACTIVESET-NORM" in results:
        summary["P3-T4-ACTIVESET-NORM"] = aggregate(results["P3-T4-ACTIVESET-NORM"],
                                                    ["act", "target_kind", "alpha", "method"],
                                                    ["target_rel_err", "leak_rel_all", "leak_rel_unprotected",
                                                     "n_crossings", "n_cross_on", "n_cross_off", "local_pred_exact",
                                                     "norm_ratio", "residual_rel", "repair_converged",
                                                     "target_attainable"])
    if "P3-T4b-SPARSITY-EXPLORATORY" in results:
        summary["P3-T4b-SPARSITY-EXPLORATORY"] = aggregate(results["P3-T4b-SPARSITY-EXPLORATORY"],
                                                           ["act", "target_kind", "alpha", "method"],
                                                           ["target_rel_err", "leak_rel_all", "n_crossings",
                                                            "norm_ratio", "residual_rel", "n_active_base"])
    if "P3-TX-EXTPROXY" in results:
        summary["P3-TX-EXTPROXY"] = aggregate(results["P3-TX-EXTPROXY"], ["case", "method"],
                                              ["int_target_rel_err", "int_leak_rel", "ext_target_rel_err",
                                               "ext_leak_rel", "norm_ratio"])
    summary["verification"] = verify(results, cfg)
    manifest["verification"] = {k: v["verdict"] for k, v in summary["verification"].items()}
    spath = os.path.join(args.out, "summary_dev.json" if args.dev else "summary.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    manifest["summary_file"] = spath
    manifest["summary_sha256"] = sha256_file(spath)
    manifest["finished_unix"] = time.time()
    dump_manifest()
    print(json.dumps({"elapsed_s": round(time.time() - t_start, 2), "verification": manifest["verification"],
                      "experiments": {k: v.get("engineering_status") for k, v in manifest["experiments"].items()}},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
