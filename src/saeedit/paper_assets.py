"""Re-aggregate existing first-run raw CSVs into paper tables, figure data and number macros.

    PYTHONPATH=src python3 -m saeedit.paper_assets --raw results/raw --paper paper \
        --out results/reaggregated

Reads only ``results/raw/*.csv`` (no experiment is re-run) and reuses the
aggregation helpers of ``run_first_run``. Every number printed in the paper
comes from ``paper/generated_numbers.tex`` written here. Independent unit =
toy instance (seed); rows inside an instance are averaged first.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from typing import Callable, Dict, List

from .run_first_run import _total_err, paired, rescale_vs_interference

INT_FIELDS = {"seed", "base_idx", "target", "n_active_base", "n_protected", "n_crossings", "n_cross_on",
              "n_cross_off", "target_crossed", "local_pred_exact", "rank", "n_constraints", "repair_iters",
              "n_extra_protected", "cgls_iters", "n_jvp", "n_vjp", "d", "m", "k", "pass", "trial",
              "target_entered", "n_active_after", "target_attainable", "attainable_flag"}
BOOL_FIELDS = {"repair_converged", "cgls_converged", "budget_active", "sign_flip"}
STR_FIELDS = {"experiment_id", "regime", "act", "case", "check", "target_kind", "method", "status", "reason"}


def load(path: str) -> List[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                if v == "":
                    continue
                if k in STR_FIELDS:
                    out[k] = v
                elif k in BOOL_FIELDS:
                    out[k] = v in ("1", "True", "true")
                elif k in INT_FIELDS:
                    out[k] = int(float(v))
                else:
                    out[k] = float(v)
            rows.append(out)
    return rows


def per_seed(rows: List[dict], metric: Callable[[dict], float]) -> Dict[int, float]:
    acc = defaultdict(list)
    for r in rows:
        if r.get("status") == "OK":
            v = metric(r)
            if v is not None and math.isfinite(v):
                acc[r["seed"]].append(v)
    return {s: statistics.fmean(v) for s, v in acc.items()}


def med(rows, metric) -> dict:
    ps = per_seed(rows, metric)
    vals = list(ps.values())
    if not vals:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {"median": statistics.median(vals), "min": min(vals), "max": max(vals), "n": len(vals)}


def sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def g(k):
    return lambda r: r.get(k)


def fmt(x, nd=3):
    if x is None:
        return "--"
    if x != 0 and abs(x) < 1e-6:
        return r"$<\!10^{-6}$"
    return f"{x:.{nd}g}" if abs(x) >= 1e-3 else f"{x:.1e}"


# ---------------------------------------------------------------- tables


def table_linear(lin: List[dict], tol: float) -> tuple:
    classes = {c["regime"]: c for c in rescale_vs_interference(lin, tol)}
    names = {
        "R0_pinv_undercomplete": r"R0: $E{=}D^{+}$, $m{<}d$",
        "R1_scaled_orthogonal": r"R1: orth.\ $D$, $E{=}\mathrm{diag}(s)D^{\top}$",
        "R2_tied_coherent": r"R2: tied, $c{=}0.3$",
        "R3_scaled_tied_coherent": r"R3: scaled tied, $c{=}0.3$",
        "R4a_overcomplete_tied_protect_all": r"R4a: tied, $m{>}d$, $P$=all",
        "R4b_overcomplete_tied_protect_subset": r"R4b: tied, $m{>}d$, $|P|{=}7$",
        "R5_overcomplete_pinv_protect_subset": r"R5: $E{=}D^{+}$, $m{>}d$, $|P|{=}7$",
    }
    short = {"NO_MISMATCH": "no mismatch", "RESCALE_SUFFICIENT": "rescale suffices",
             "INTERFERENCE_REMOVED_BY_CORRECTION": "removed by LN", "INFEASIBLE_ON_S_UNION_P": "infeasible",
             "INTERFERENCE_MOVED_TO_UNPROTECTED": "moved to unprot."}
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Regime & Class & Dec.\ tgt err & Dec.\ leak & Resc.\ leak (norm) & LN leak (norm) & LN resid. \\",
             r"\midrule"]
    data = {}
    for reg in names:
        c = classes[reg]
        rows = sel(lin, regime=reg)
        ln_t = med(sel(rows, method="jacobian_ln"), g("target_rel_err"))["median"]
        data[reg] = {**c, "ln_target_rel_err_median": ln_t}
        lines.append(f"{names[reg]} & {short[c['class']]} & {fmt(c['decoder_target_rel_err_median'])} & "
                     f"{fmt(c['decoder_leak_median'])} & {fmt(c['rescaled_leak_median'])} "
                     f"({fmt(c['rescaled_norm_ratio_median'])}) & {fmt(c['ln_leak_median'])} "
                     f"({fmt(c['ln_norm_ratio_median'])}) & {fmt(c['ln_residual_rel_median'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), data


def table_feasibility(feas: List[dict]) -> tuple:
    by = defaultdict(list)
    for r in feas:
        by[r["check"]].append(r["pass"])
    label = {"full_rank_least_norm": "Full row rank: least-norm, minimality, CGLS = dense",
             "rank_deficient_infeasible": "More constraints than $d$: residual $>0$ with certificate",
             "rank_deficient_feasible_control": "Rank-deficient, target in range: residual $\\approx 0$",
             "low_rank_rows": "Low-rank rows ($k{=}8$, rank 5): certificate",
             "duplicate_rows_conflict": r"Duplicate rows, targets $(\alpha,0)$: residual $\alpha/\sqrt2$",
             "range_infeasible_relu": "ReLU: negative target unattainable",
             "range_infeasible_jumprelu": r"JumpReLU: target in $(0,\theta]$ unattainable",
             "topk_eviction": "TopK: entering target evicts a feature"}
    lines = [r"\begin{tabular}{lc}", r"\toprule", r"Check & Pass \\", r"\midrule"]
    counts = {}
    for k in label:
        v = by.get(k, [])
        counts[k] = (sum(v), len(v))
        lines.append(f"{label[k]} & {sum(v)}/{len(v)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), counts


def drift_table(rows: List[dict], acts: List[str], caption_act: Dict[str, str]) -> tuple:
    """Rows: (act, kind, alpha). Drift = toy protected (originally active) / unprotected (newly active) leakage."""
    lines = [r"\begin{tabular}{llc|ccc|cccc|cc}", r"\toprule",
             r" & & & \multicolumn{3}{c|}{Rescaled decoder} & \multicolumn{4}{c|}{LN correction} & "
             r"\multicolumn{2}{c}{LN + repair} \\",
             r"Act. & Target & $\alpha$ & orig.\ act. & new & norm & tgt err & orig.\ act. & new & norm & tgt err & total \\",
             r"\midrule"]
    data = {}
    for act in acts:
        for kind in ("active_increase", "inactive_activate"):
            for alpha in (0.25, 1.0, 4.0):
                base = sel(rows, act=act, target_kind=kind, alpha=alpha)
                rs = sel(base, method="decoder_rescaled")
                ln = sel(base, method="jacobian_ln")
                rp = sel(base, method="jacobian_ln_repair")
                d = {
                    "resc_orig": med(rs, g("leak_rel_protected"))["median"],
                    "resc_new": med(rs, g("leak_rel_unprotected"))["median"],
                    "resc_norm": med(rs, g("norm_ratio"))["median"],
                    "resc_tgt": med(rs, g("target_rel_err"))["median"],
                    "ln_tgt": med(ln, g("target_rel_err"))["median"],
                    "ln_orig": med(ln, g("leak_rel_protected"))["median"],
                    "ln_new": med(ln, g("leak_rel_unprotected"))["median"],
                    "ln_norm": med(ln, g("norm_ratio"))["median"],
                    "rep_tgt": med(rp, g("target_rel_err"))["median"],
                    "rep_total": med(rp, g("leak_rel_all"))["median"],
                    "rep_conv": med(rp, lambda r: float(r.get("repair_converged", False)))["median"],
                    "attainable": med(ln, g("target_attainable"))["median"],
                }
                pw = paired(base, ["act"], "jacobian_ln", "decoder_rescaled", g("leak_rel_all"))
                d["ln_better_total"] = (pw[0]["a_better_units"], pw[0]["n_units"]) if pw else None
                data[f"{act}|{kind}|{alpha}"] = d
                kname = "active" if kind == "active_increase" else "inactive"
                flag = "" if d["attainable"] in (None, 1.0) else r"$^\dagger$"
                lines.append(f"{caption_act.get(act, act)} & {kname}{flag} & {alpha:g} & {fmt(d['resc_orig'])} & "
                             f"{fmt(d['resc_new'])} & {fmt(d['resc_norm'])} & {fmt(d['ln_tgt'])} & "
                             f"{fmt(d['ln_orig'])} & {fmt(d['ln_new'])} & {fmt(d['ln_norm'])} & "
                             f"{fmt(d['rep_tgt'])} & {fmt(d['rep_total'])} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines), data


def table_extproxy(tx: List[dict]) -> tuple:
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Case & Method & Internal err & External err & Norm & LN better ext. \\", r"\midrule"]
    names = {"encoder_accurate": r"encoder $=A^{+}$", "decoder_accurate": r"decoder $=A$", "both_noisy": "both noisy"}
    data = {}
    for case in ("encoder_accurate", "decoder_accurate", "both_noisy"):
        rows = sel(tx, case=case)
        pw = paired(rows, ["case"], "jacobian_ln", "decoder",
                    lambda r: math.hypot(r["ext_target_rel_err"], r["ext_leak_rel"]), unit="seed")
        for meth in ("decoder", "decoder_rescaled", "encoder_grad", "jacobian_ln"):
            rs = sel(rows, method=meth)
            ie = med(rs, lambda r: math.hypot(r["int_target_rel_err"], r["int_leak_rel"]))["median"]
            ee = med(rs, lambda r: math.hypot(r["ext_target_rel_err"], r["ext_leak_rel"]))["median"]
            nr = med(rs, g("norm_ratio"))["median"]
            wins = f"{pw[0]['a_better_units']}/{pw[0]['n_units']}" if (meth == "jacobian_ln" and pw) else ""
            data[f"{case}|{meth}"] = {"internal": ie, "external": ee, "norm": nr}
            mname = meth.replace("_", "\\_")
            lines.append(f"{names[case] if meth == 'decoder' else ''} & {mname} & {fmt(ie)} & "
                         f"{fmt(ee)} & {fmt(nr)} & {wins} \\\\")
        data[f"{case}|paired"] = pw[0] if pw else None
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines), data


# ---------------------------------------------------------------- figure data


def fig_alpha_curves(t4: List[dict], t4b: List[dict], path: str) -> dict:
    """Median total drift vs alpha for ReLU active targets (dense pre-registered vs sparser exploratory)."""
    regimes = [("dense", t4, "relu"), ("bm050", t4b, "relu_b-0.50"), ("bm075", t4b, "relu_b-0.75")]
    out = {}
    with open(path, "w") as f:
        cols = ["alpha"] + [f"{n}_{m}_{k}" for n, _, _ in regimes for m in ("resc", "ln") for k in ("orig", "new", "total")]
        f.write(" ".join(cols) + "\n")
        for alpha in (0.25, 1.0, 4.0):
            vals = [alpha]
            for n, rows, act in regimes:
                base = sel(rows, act=act, target_kind="active_increase", alpha=alpha)
                for m, meth in (("resc", "decoder_rescaled"), ("ln", "jacobian_ln")):
                    rs = sel(base, method=meth)
                    for k, col in (("orig", "leak_rel_protected"), ("new", "leak_rel_unprotected"), ("total", "leak_rel_all")):
                        v = med(rs, g(col))["median"]
                        vals.append(v)
                        out[f"{n}|{m}|{k}|{alpha}"] = v
            f.write(" ".join(f"{v:.6g}" for v in vals) + "\n")
    return out


def fig_norm_budget(t4: List[dict], path: str) -> dict:
    """Target error vs total drift at matched budgets (ReLU, active target, alpha = 1)."""
    base = sel(t4, act="relu", target_kind="active_increase", alpha=1.0)
    out = {}
    with open(path, "w") as f:
        f.write("method norm tgt leak\n")
        for meth in ("decoder", "decoder_rescaled", "jacobian_ln", "jacobian_ln_normmatched", "trust_region_x0.5",
                     "trust_region_normmatched", "trust_region_x2.0", "jacobian_ln_repair"):
            rs = sel(base, method=meth)
            n = med(rs, g("norm_ratio"))["median"]
            t = med(rs, g("target_rel_err"))["median"]
            l = med(rs, g("leak_rel_all"))["median"]
            out[meth] = {"norm": n, "tgt": t, "leak": l}
            f.write(f"{meth} {n:.6g} {t:.6g} {l:.6g}\n")
    return out


def fig_extproxy(tx_data: dict, path: str) -> None:
    with open(path, "w") as f:
        f.write("case dec_int dec_ext ln_int ln_ext\n")
        for i, case in enumerate(("encoder_accurate", "decoder_accurate", "both_noisy")):
            d, l = tx_data[f"{case}|decoder"], tx_data[f"{case}|jacobian_ln"]
            f.write(f"{i} {d['internal']:.6g} {d['external']:.6g} {l['internal']:.6g} {l['external']:.6g}\n")


# ---------------------------------------------------------------- macros


def macro_name(s: str) -> str:
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six",
              "7": "Seven", "8": "Eight", "9": "Nine"}
    out = ""
    for ch in s:
        if ch.isalpha():
            out += ch
        elif ch.isdigit():
            out += digits[ch]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--out", default="results/reaggregated")
    ap.add_argument("--tol", type=float, default=1e-8)
    args = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(os.path.join(args.paper, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.paper, "figures"), exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    files = {k: os.path.join(args.raw, f) for k, f in (("lin", "linear_regimes.csv"), ("feas", "feasibility.csv"),
                                                        ("t4", "activeset_norm.csv"), ("t4b", "sparsity_exploratory.csv"),
                                                        ("tx", "external_proxy.csv"))}
    R = {k: load(p) for k, p in files.items()}
    numbers: Dict[str, object] = {}

    tab, lin_data = table_linear(R["lin"], args.tol)
    open(os.path.join(args.paper, "tables", "tab_linear.tex"), "w").write(tab + "\n")
    tab, feas_counts = table_feasibility(R["feas"])
    open(os.path.join(args.paper, "tables", "tab_feasibility.tex"), "w").write(tab + "\n")
    tab, t4_data = drift_table(R["t4"], ["relu", "jumprelu", "topk"], {"relu": "ReLU", "jumprelu": "JumpReLU", "topk": "TopK"})
    open(os.path.join(args.paper, "tables", "tab_activeset_all.tex"), "w").write(tab + "\n")
    tab, _ = drift_table(R["t4"], ["relu"], {"relu": "ReLU"})
    open(os.path.join(args.paper, "tables", "tab_activeset_relu.tex"), "w").write(tab + "\n")
    tab, t4b_data = drift_table(R["t4b"], ["relu_b-0.50", "relu_b-0.75"], {"relu_b-0.50": "$b{=}{-}0.5$", "relu_b-0.75": "$b{=}{-}0.75$"})
    open(os.path.join(args.paper, "tables", "tab_sparsity_exploratory.tex"), "w").write(tab + "\n")
    tab, tx_data = table_extproxy(R["tx"])
    open(os.path.join(args.paper, "tables", "tab_extproxy.tex"), "w").write(tab + "\n")

    curves = fig_alpha_curves(R["t4"], R["t4b"], os.path.join(args.paper, "figures", "alpha_curves.dat"))
    budget = fig_norm_budget(R["t4"], os.path.join(args.paper, "figures", "norm_budget.dat"))
    names = {"decoder": "decoder", "decoder_rescaled": "rescaled decoder", "jacobian_ln": "LN correction",
             "jacobian_ln_normmatched": "LN, scaled to $\\|\\alpha D_j\\|$", "trust_region_x0.5": "trust region, $0.5\\tau$",
             "trust_region_normmatched": "trust region, $\\tau$", "trust_region_x2.0": "trust region, $2\\tau$",
             "jacobian_ln_repair": "LN + repair"}
    tl = [r"\begin{tabular}{lccc}", r"\toprule", r"Method & $\|\delta\|/\tau$ & Target err & Total drift \\", r"\midrule"]
    for k, v in budget.items():
        tl.append(f"{names[k]} & {fmt(v['norm'])} & {fmt(v['tgt'])} & {fmt(v['leak'])} \\\\")
    tl += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(args.paper, "tables", "tab_norm_budget.tex"), "w").write("\n".join(tl) + "\n")
    fig_extproxy(tx_data, os.path.join(args.paper, "figures", "extproxy.dat"))

    # active-set sizes per base point (unique (act, seed, base_idx))
    def active_sizes(rows):
        seen = {}
        for r in rows:
            if "n_active_base" in r:
                seen[(r["act"], r["seed"], r["base_idx"])] = r["n_active_base"]
        by = defaultdict(list)
        for (act, _, _), v in seen.items():
            by[act].append(v)
        return {a: {"median": statistics.median(v), "min": min(v), "max": max(v), "n_base_points": len(v)} for a, v in by.items()}

    sizes = {**active_sizes(R["t4"]), **active_sizes(R["t4b"])}
    # paired matched-norm trust region vs decoder on total error (circular comparison, reported as such)
    tr = paired(R["t4"], ["act", "target_kind", "alpha"], "trust_region_normmatched", "decoder", _total_err)
    # repair convergence and target error ranges
    rep = {k: (v["rep_conv"], v["rep_tgt"]) for k, v in t4_data.items()}
    rep_tgt_relu_jump = [v[1] for k, v in rep.items() if not k.startswith("topk") and t4_data[k]["attainable"] == 1.0]
    rep_conv_topk = [v[0] for k, v in rep.items() if k.startswith("topk")]
    # row-level repair convergence counts per cell (the per-instance median hides failures)
    conv_rows = defaultdict(lambda: [0, 0])
    for r in R["t4"]:
        if r.get("method") == "jacobian_ln_repair" and r.get("status") == "OK":
            c = conv_rows[(r["act"], r["target_kind"], r["alpha"])]
            c[0] += int(bool(r.get("repair_converged", False)))
            c[1] += 1
    conv_rj = [v for k, v in conv_rows.items() if k[0] in ("relu", "jumprelu")]
    conv_tk = [v for k, v in conv_rows.items() if k[0] == "topk"]
    # linear-regime extras
    lin = R["lin"]
    enc_grad_r0 = med(sel(lin, regime="R0_pinv_undercomplete", method="encoder_grad"), g("leak_rel_all"))["median"]
    r4a_res = med(sel(lin, regime="R4a_overcomplete_tied_protect_all", method="jacobian_ln"), g("residual_rel"))
    tr_r2 = {k: med(sel(lin, regime="R2_tied_coherent", method="trust_region_normmatched"), g(k))["median"]
             for k in ("target_rel_err", "leak_rel_all")}
    r4b_dec = {k: med(sel(lin, regime="R4b_overcomplete_tied_protect_subset", method="decoder"), g(k))["median"]
               for k in ("leak_rel_protected", "leak_rel_unprotected", "leak_rel_all")}
    v6 = [r for r in R["t4"] if r.get("act") == "relu" and r.get("status") == "OK" and r["n_crossings"] == 0
          and r["target_crossed"] == 0]
    cg = [r for r in lin if r.get("method") == "jacobian_ln" and "rel_diff_vs_jacobian_ln_cgls" in r]

    numbers = {
        "linear": lin_data, "feasibility": feas_counts, "activeset_dense": t4_data, "activeset_sparse_exploratory": t4b_data,
        "extproxy": tx_data, "alpha_curves": curves, "norm_budget": budget, "active_set_sizes": sizes,
        "trust_region_vs_decoder_matched_norm_CIRCULAR": tr,
        "repair_target_err_range_relu_jumprelu_attainable": [min(rep_tgt_relu_jump), max(rep_tgt_relu_jump)],
        "repair_convergence_topk_range_median_of_instance_rates": [min(rep_conv_topk), max(rep_conv_topk)],
        "repair_convergence_rows_per_cell": {f"{a}|{k}|{al}": v for (a, k, al), v in conv_rows.items()},
        "encoder_grad_leak_R0": enc_grad_r0, "R4a_residual": r4a_res, "R2_trust_region": tr_r2, "R4b_decoder": r4b_dec,
        "V6_rows_without_crossing": len(v6), "V6_max_local_pred_err": max(r["local_pred_err"] for r in v6),
        "V5_max_cgls_vs_dense": max(r["rel_diff_vs_jacobian_ln_cgls"] for r in cg),
    }
    # macros used in the text
    M = {
        "RtwoDecLeak": fmt(lin_data["R2_tied_coherent"]["decoder_leak_median"]),
        "RtwoLnNorm": fmt(lin_data["R2_tied_coherent"]["ln_norm_ratio_median"]),
        "RthreeLnNorm": fmt(lin_data["R3_scaled_tied_coherent"]["ln_norm_ratio_median"]),
        "RthreeRescLeak": fmt(lin_data["R3_scaled_tied_coherent"]["rescaled_leak_median"]),
        "RoneDecTgt": fmt(lin_data["R1_scaled_orthogonal"]["decoder_target_rel_err_median"]),
        "RfouraRes": fmt(r4a_res["median"]), "RfouraResMin": fmt(r4a_res["min"]),
        "RfouraResRowMin": fmt(min(r["residual_rel"] for r in sel(lin, regime="R4a_overcomplete_tied_protect_all",
                                                                   method="jacobian_ln"))),
        "RfouraLnTgt": fmt(lin_data["R4a_overcomplete_tied_protect_all"]["ln_target_rel_err_median"]),
        "RfourbDecLeak": fmt(r4b_dec["leak_rel_all"]), "RfourbDecLeakP": fmt(r4b_dec["leak_rel_protected"]),
        "RfourbDecLeakU": fmt(r4b_dec["leak_rel_unprotected"]),
        "RfourbLnLeak": fmt(lin_data["R4b_overcomplete_tied_protect_subset"]["ln_leak_median"]),
        "RfourbLnNorm": fmt(lin_data["R4b_overcomplete_tied_protect_subset"]["ln_norm_ratio_median"]),
        "RfiveRescLeak": fmt(lin_data["R5_overcomplete_pinv_protect_subset"]["rescaled_leak_median"]),
        "RfiveRescNorm": fmt(lin_data["R5_overcomplete_pinv_protect_subset"]["rescaled_norm_ratio_median"]),
        "RfiveLnLeak": fmt(lin_data["R5_overcomplete_pinv_protect_subset"]["ln_leak_median"]),
        "RfiveLnNorm": fmt(lin_data["R5_overcomplete_pinv_protect_subset"]["ln_norm_ratio_median"]),
        "RfiveDecTgt": fmt(lin_data["R5_overcomplete_pinv_protect_subset"]["decoder_target_rel_err_median"]),
        "EncGradLeakRzero": fmt(enc_grad_r0),
        "RtwoTrTgt": fmt(tr_r2["target_rel_err"]), "RtwoTrLeak": fmt(tr_r2["leak_rel_all"]),
        "FeasTotalPass": str(sum(p for p, _ in feas_counts.values())),
        "FeasTotal": str(sum(n for _, n in feas_counts.values())),
        "VsixRows": str(len(v6)), "VsixMaxErr": f"{numbers['V6_max_local_pred_err']:.1e}",
        "VfiveMaxDiff": f"{numbers['V5_max_cgls_vs_dense']:.1e}",
        "DenseReluActMed": f"{sizes['relu']['median']:g}", "DenseReluActMin": f"{sizes['relu']['min']:g}",
        "DenseReluActMax": f"{sizes['relu']['max']:g}",
        "DenseJumpActMed": f"{sizes['jumprelu']['median']:g}", "DenseJumpActMin": f"{sizes['jumprelu']['min']:g}",
        "DenseJumpActMax": f"{sizes['jumprelu']['max']:g}",
        "SparseFiftyActMed": f"{sizes['relu_b-0.50']['median']:g}", "SparseFiftyActMax": f"{sizes['relu_b-0.50']['max']:g}",
        "SparseSeventyFiveActMed": f"{sizes['relu_b-0.75']['median']:g}",
        "SparseSeventyFiveActMax": f"{sizes['relu_b-0.75']['max']:g}",
        "RepairTgtMin": fmt(min(rep_tgt_relu_jump), 2), "RepairTgtMax": fmt(max(rep_tgt_relu_jump), 2),
        "RepairConvRJMin": str(min(v[0] for v in conv_rj)), "RepairConvRJMax": str(max(v[0] for v in conv_rj)),
        "RepairConvTKMin": str(min(v[0] for v in conv_tk)), "RepairConvTKMax": str(max(v[0] for v in conv_tk)),
        "RepairRowsPerCell": str(max(v[1] for v in conv_rows.values())),
    }
    for kind, kk in (("active_increase", "Act"), ("inactive_activate", "Inact")):
        for alpha, aa in ((0.25, "Q"), (1.0, "One"), (4.0, "Four")):
            w = t4_data[f"relu|{kind}|{alpha}"]["ln_better_total"]
            M[f"DenseRelu{kk}{aa}Wins"] = f"{w[0]}/{w[1]}"
            for act, an in (("relu_b-0.50", "Fifty"), ("relu_b-0.75", "SeventyFive")):
                w = t4b_data[f"{act}|{kind}|{alpha}"]["ln_better_total"]
                M[f"Sparse{an}{kk}{aa}Wins"] = f"{w[0]}/{w[1]}"
    for case, cc in (("encoder_accurate", "EncAcc"), ("decoder_accurate", "DecAcc"), ("both_noisy", "Both")):
        for meth, mm in (("decoder", "Dec"), ("jacobian_ln", "Ln")):
            M[f"Tx{cc}{mm}Int"] = fmt(tx_data[f"{case}|{meth}"]["internal"])
            M[f"Tx{cc}{mm}Ext"] = fmt(tx_data[f"{case}|{meth}"]["external"])
        p = tx_data[f"{case}|paired"]
        M[f"Tx{cc}Wins"] = f"{p['a_better_units']}/{p['n_units']}"
        M[f"Tx{cc}Diff"] = f"{p['median_diff_a_minus_b']:+.3g}"
    tr_relu = [t for t in tr if t["act"] in ("relu", "jumprelu")]
    M["TrWinsMin"] = str(min(t["a_better_units"] for t in tr_relu))
    M["TrWinsMax"] = str(max(t["a_better_units"] for t in tr_relu))
    real = {}
    if os.path.exists("results/contract_exec/contract_exec_report.json") and os.path.exists("results/contract_exec_v2/contract_exec_report.json"):
        real.update(contract_assets("results/contract_exec/contract_exec_report.json",
                                    "results/contract_exec_v2/contract_exec_report.json", args.paper))
    if os.path.exists("results/real01r/summary.json"):
        real.update(pilot_assets("results/real01r/summary.json", args.paper))
    if _closure():
        real.update(closure_assets())
    if os.path.exists("results/real01r_reagg/reagg_summary.json"):
        real.update(reagg_assets("results/real01r_reagg", args.paper))
    if os.path.exists("results/real02_lx/summary.json"):
        real.update(lx_assets("results/real02_lx", args.paper))
    elif os.path.exists("results/real02_lx/stage_status.json"):
        real.update(lx_blocked_assets("results/real02_lx"))
    M.update(real)
    with open(os.path.join(args.paper, "generated_numbers.tex"), "w") as f:
        f.write("% Generated by src/saeedit/paper_assets.py from results/raw/*.csv. Do not edit by hand.\n")
        for k in sorted(M):
            assert macro_name(k) == k, k
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    numbers["macros"] = M
    inputs = {k: {"path": p, "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()} for k, p in files.items()}
    with open(os.path.join(args.out, "paper_numbers.json"), "w") as f:
        json.dump({"inputs": inputs, "seconds": time.time() - t0, "numbers": numbers}, f, indent=2, default=str)
    print(json.dumps({"seconds": round(time.time() - t0, 2), "n_macros": len(M)}))
    return 0



# ---------------------------------------------------------------- real-model assets (contract execution, pilot)


def _sci(x, nd=2):
    if x is None:
        return "--"
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        m, e = f"{x:.{nd - 1}e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{x:.3g}"


def contract_assets(v1_path: str, v2_path: str, paper: str) -> dict:
    """Table of P3-CONTRACT-EXEC checks: float32 (v1, pre-registered) vs float64 (v2, diagnostic)."""
    v1 = json.load(open(v1_path))["checks"]
    v2 = json.load(open(v2_path))["checks"]
    ok = lambda b: r"\checkmark" if b else r"$\times$"
    rows = [
        ("K0 weights/config identity", "bitwise", "--", ok(v1["K0_weights_identity"]["pass"]), ok(v2["K0_weights_identity"]["pass"])),
        ("K1 tokenizer (16 windows)", "equal", "--", f"{v1['K1_tokenizer']['n_equal']}/16", f"{v2['K1_tokenizer']['n_equal']}/16"),
        ("K2 activation, $x=h-\\bar h$", "rel.\\ $\\ell_2$", "$10^{-4}$", _sci(v1["K2_activation"]["max_rel_diff_centred"]), _sci(v2["K2_activation"]["max_rel_diff_centred"])),
        ("\\quad uncentred $h$ (neg.\\ control)", "rel.\\ $\\ell_2$", "--", _sci(v1["K2_activation"]["max_rel_diff_uncentred_negative_control"]), _sci(v2["K2_activation"]["max_rel_diff_uncentred_negative_control"])),
        ("K3 SAE features", "rel.\\ $\\ell_2$", "$10^{-3}$", _sci(v1["K3_features"]["max_rel_diff"]), _sci(v2["K3_features"]["max_rel_diff"])),
        ("K4 L0 / FVE (canonical)", "band", "[30,120] / 0.7", f"{v1['K4_reconstruction']['tl']['l0']:.1f} / {v1['K4_reconstruction']['tl']['fve']:.3f}", f"{v2['K4_reconstruction']['tl']['l0']:.1f} / {v2['K4_reconstruction']['tl']['fve']:.3f}"),
        ("\\quad uncentred input", "--", "--", f"{v1['K4_reconstruction']['hf_uncentred']['l0']:.0f} / {v1['K4_reconstruction']['hf_uncentred']['fve']:.3f}", f"{v2['K4_reconstruction']['hf_uncentred']['l0']:.0f} / {v2['K4_reconstruction']['hf_uncentred']['fve']:.3f}"),
        ("K5 no-op hooks", "max $|\\Delta|$", "0", _sci(max(v for k, v in v1["K5_noop"].items() if k != "pass")), _sci(max(v for k, v in v2["K5_noop"].items() if k != "pass"))),
        ("K6 common logit shift", "max $|c_t|$", "--", f"{v1['K6_logit_parity']['max_common_shift']:.0f}", f"{v2['K6_logit_parity']['max_common_shift']:.0f}"),
        ("\\quad residual after shift", "max", "$10^{-3}$", _sci(v1["K6_logit_parity"]["max_residual_after_shift"]), _sci(v2["K6_logit_parity"]["max_residual_after_shift"])),
        ("\\quad probabilities", "max $|\\Delta p|$", "$10^{-5}$", _sci(v1["K6_logit_parity"]["max_prob_diff"]), _sci(v2["K6_logit_parity"]["max_prob_diff"])),
        ("\\quad KL", "max", "$10^{-6}$", _sci(v1["K6_logit_parity"]["max_kl"]), _sci(v2["K6_logit_parity"]["max_kl"])),
        ("K7 post-edit feature change", "rel.\\ $\\ell_2$", "$10^{-3}$", _sci(v1["K7_intervention_parity"]["max_feature_change_rel_diff"]), _sci(v2["K7_intervention_parity"]["max_feature_change_rel_diff"])),
        ("\\quad post-edit probabilities", "max $|\\Delta p|$", "$10^{-5}$", _sci(v1["K7_intervention_parity"]["max_prob_diff"]), _sci(v2["K7_intervention_parity"]["max_prob_diff"])),
        ("\\quad post-edit KL", "max", "$10^{-6}$", _sci(v1["K7_intervention_parity"]["max_kl"]), _sci(v2["K7_intervention_parity"]["max_kl"])),
        ("K9 mean-only edit, model KL", "max", "$10^{-8}$", _sci(max(v1["K9_mean_only_control"]["max_kl_tl"], v1["K9_mean_only_control"]["max_kl_hf"])), _sci(max(v2["K9_mean_only_control"]["max_kl_tl"], v2["K9_mean_only_control"]["max_kl_hf"]))),
        ("\\quad SAE drift / $\\rho$, raw off-slice $E(x{+}\\delta)$", "median", "--", f"{v1['K9_mean_only_control']['sae_drift_rel_median']:.0f}", f"{v2['K9_mean_only_control']['sae_drift_rel_median']:.0f}"),
        ("\\quad SAE drift / $\\rho$, canonical $E(P(h{+}\\delta))$", "max", "--", "not run", _sci(_closure()["part_a_k9"]["canonical_drift_rel_max"]) if _closure() else "--"),
        ("K10 no-processing path, KL", "max", "$10^{-6}$", "not run", _sci(v2["K10_no_processing_path"]["max_kl"]) if "K10_no_processing_path" in v2 else "--"),
    ]
    lines = [r"\begin{tabular}{llccc}", r"\toprule", r"Check & Quantity & Tol. & float32 (v1) & float64 (v2) \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\midrule", f"Verdict & & & {json.load(open(v1_path))['verdict'].replace('_', ' ')} & {json.load(open(v2_path))['verdict'].replace('_', ' ')} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(paper, "tables", "tab_contract_exec.tex"), "w").write("\n".join(lines) + "\n")
    k8 = [p["readout_diff_rel"] for p in v2["K8_non_admissible_readout"]["per_feature"]]
    mc = [p["mean_component_rel"] for p in v2["K8_non_admissible_readout"]["per_feature"]]
    return {
        "CxShift": f"{v2['K6_logit_parity']['max_common_shift']:.0f}",
        "CxFThreeTwoProb": _sci(v1["K6_logit_parity"]["max_prob_diff"]), "CxFThreeTwoKL": _sci(v1["K6_logit_parity"]["max_kl"]),
        "CxFSixFourProb": _sci(v2["K6_logit_parity"]["max_prob_diff"]), "CxFSixFourKL": _sci(v2["K6_logit_parity"]["max_kl"]),
        "CxActRel": _sci(v2["K2_activation"]["max_rel_diff_centred"]), "CxActRelUnc": _sci(v2["K2_activation"]["max_rel_diff_uncentred_negative_control"]),
        "CxLZero": f"{v2['K4_reconstruction']['tl']['l0']:.1f}", "CxFVE": f"{v2['K4_reconstruction']['tl']['fve']:.2f}",
        "CxFVEUnc": f"{v2['K4_reconstruction']['hf_uncentred']['fve']:.3f}", "CxLZeroUnc": f"{v2['K4_reconstruction']['hf_uncentred']['l0']:.0f}",
        "CxMeanKLThreeTwo": _sci(max(v1["K9_mean_only_control"]["max_kl_tl"], v1["K9_mean_only_control"]["max_kl_hf"])),
        "CxMeanKLSixFour": _sci(max(v2["K9_mean_only_control"]["max_kl_tl"], v2["K9_mean_only_control"]["max_kl_hf"])),
        "CxMeanDrift": f"{v2['K9_mean_only_control']['sae_drift_rel_median']:.0f}",
        "CxMeanMaxFeat": f"{v2['K9_mean_only_control']['max_single_feature_change']:.1f}",
        "CxEightMed": f"{statistics.median(k8):.2f}", "CxEightMin": f"{min(k8):.2f}", "CxEightMax": f"{max(k8):.2f}",
        "CxEightMeanMin": f"{100 * min(mc):.0f}", "CxEightMeanMax": f"{100 * max(mc):.0f}",
        "CxRho": f"{v2['K7_intervention_parity']['rho']:.1f}", "CxMedNorm": f"{v2['K7_intervention_parity']['median_residual_norm']:.0f}",
    }


# mean_only is excluded: its stored readout is raw off-slice (P3-READOUT-CLOSURE); canonically it changes nothing
PILOT_METHODS = ("decoder", "decoder_calib_rescaled", "encoder_grad", "jacobian_ln", "random")
PILOT_NAMES = {"decoder": "decoder", "decoder_calib_rescaled": "rescaled (weight-only)", "encoder_grad": "encoder row",
               "jacobian_ln": "LN ($J_EP$)", "random": "random", "mean_only": "mean-only control"}
DOSES = ("q50", "q90", "q99", "2xq99")


def _ci(c, nd=3):
    if not c or c.get("estimate") is None:
        return "--"
    f = lambda v: f"{v:.{nd}g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.1e}"
    return f"{f(c['estimate'])} [{f(c['lo'])}, {f(c['hi'])}]"


def pilot_assets(summary_path: str, paper: str) -> dict:
    S = json.load(open(summary_path))
    cur = {(c["mode"], c["target_kind"], c["dose"], c["method"], c["metric"]): c for c in S["curves"]}
    par = {(p["mode"], p["target_kind"], p["dose"], p["a"], p["b"], p["metric"]): p for p in S["paired"]}
    inf = S["infeasible_counts"]
    # target-matched table at q99
    L = [r"\begin{tabular}{llcccccc}", r"\toprule",
         r"Target & Method & Target err & $D_{\mathrm{orig}}/\alpha$ & $D_{\mathrm{new}}/\alpha$ & $D_{\mathrm{all}}/\alpha$ & KL (nats) & $\|\delta\|$ \\",
         r"\midrule"]
    for kind in ("active", "inactive"):
        for m in PILOT_METHODS:
            g = lambda met: cur.get(("target_matched", kind, "q99", m, met))
            st = inf.get(f"target_matched|q99|{m}|{kind}", {})
            n_inf = st.get("INFEASIBLE", 0)
            tag = f"$^{{{n_inf}}}$" if n_inf else ""
            L.append(f"{kind if m == 'decoder' else ''} & {PILOT_NAMES[m]}{tag} & {_ci(g('target_err_rel'))} & "
                     f"{_ci(g('drift_orig_active_rel'))} & {_ci(g('drift_newly_active_rel'))} & "
                     f"{_ci(g('drift_all_nontarget_rel'))} & {_ci(g('kl_next_token'))} & {_ci(g('edit_norm'))} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    open(os.path.join(paper, "tables", "tab_pilot_matched.tex"), "w").write("\n".join(L) + "\n")
    # medians-only variant for the main text (intervals in the appendix table)
    Lm = [r"\begin{tabular}{llcccccc}", r"\toprule",
          r"Target & Method & $n_{\mathrm{doc}}$ & $D_{\mathrm{orig}}/\alpha$ & $D_{\mathrm{new}}/\alpha$ & $D_{\mathrm{all}}/\alpha$ & KL$\times10^{3}$ & $\|\delta\|$ \\",
          r"\midrule"]
    for kind in ("active", "inactive"):
        for m in PILOT_METHODS:
            g = lambda met: cur.get(("target_matched", kind, "q99", m, met))
            est = lambda met, sc=1.0: ("--" if not g(met) or g(met).get("estimate") is None
                                        else ("$\\approx 0$" if abs(sc * g(met)["estimate"]) < 1e-9
                                              else f"{sc * g(met)['estimate']:.3g}"))
            n = g("drift_all_nontarget_rel")["n_units"] if g("drift_all_nontarget_rel") else 0
            Lm.append(f"{kind if m == 'decoder' else ''} & {PILOT_NAMES[m]} & {n} & {est('drift_orig_active_rel')} & "
                      f"{est('drift_newly_active_rel')} & {est('drift_all_nontarget_rel')} & {est('kl_next_token', 1e3)} & "
                      f"{est('edit_norm')} \\\\")
        Lm.append(r"\midrule")
    Lm[-1] = r"\bottomrule"
    Lm.append(r"\end{tabular}")
    open(os.path.join(paper, "tables", "tab_pilot_matched_median.tex"), "w").write("\n".join(Lm) + "\n")
    # paired LN - decoder by dose (target-matched)
    P = [r"\begin{tabular}{llccccc}", r"\toprule",
         r"Target & Dose & $n_{\mathrm{doc}}$ & $\Delta D_{\mathrm{all}}$ & $\Delta D_{\mathrm{new}}$ & $\Delta D_{\mathrm{orig}}$ & $\Delta$KL \\",
         r"\midrule"]
    for kind in ("active", "inactive"):
        for dose in DOSES:
            g = lambda met: par.get(("target_matched", kind, dose, "jacobian_ln", "decoder", met))
            n = g("drift_all_nontarget_rel")["n_units"] if g("drift_all_nontarget_rel") else 0
            P.append(f"{kind if dose == 'q50' else ''} & {dose} & {n} & {_ci(g('drift_all_nontarget_rel'))} & "
                     f"{_ci(g('drift_newly_active_rel'))} & {_ci(g('drift_orig_active_rel'))} & {_ci(g('kl_next_token'))} \\\\")
        P.append(r"\midrule")
    P[-1] = r"\bottomrule"
    P.append(r"\end{tabular}")
    open(os.path.join(paper, "tables", "tab_pilot_paired.tex"), "w").write("\n".join(P) + "\n")
    # equal-norm table at budget 0.1
    E = [r"\begin{tabular}{llccc}", r"\toprule", r"Target & Method & target gain & $D_{\mathrm{all}}/\rho$ & KL (nats) \\", r"\midrule"]
    for kind in ("active", "inactive"):
        for m in ("decoder", "encoder_grad", "jacobian_ln", "random"):
            g = lambda met: cur.get(("equal_norm", kind, "0.1", m, met))
            E.append(f"{kind if m == 'decoder' else ''} & {PILOT_NAMES[m]} & {_ci(g('target_gain'))} & "
                     f"{_ci(g('drift_all_nontarget_rel'))} & {_ci(g('kl_next_token'))} \\\\")
        E.append(r"\midrule")
    E[-1] = r"\bottomrule"
    E.append(r"\end{tabular}")
    open(os.path.join(paper, "tables", "tab_pilot_equalnorm.tex"), "w").write("\n".join(E) + "\n")
    # dose curves (figure data)
    with open(os.path.join(paper, "figures", "pilot_dose.dat"), "w") as f:
        cols = ["dose_idx"] + [f"{k}_{m}_{met}" for k in ("active", "inactive") for m in ("decoder", "jacobian_ln", "mean_only")
                               for met in ("all", "new", "kl")]
        f.write(" ".join(cols) + "\n")
        for i, dose in enumerate(DOSES):
            vals = [i]
            for k in ("active", "inactive"):
                for m in ("decoder", "jacobian_ln", "mean_only"):
                    for met in ("drift_all_nontarget_rel", "drift_newly_active_rel", "kl_next_token"):
                        c = cur.get(("target_matched", k, dose, m, met))
                        vals.append(c["estimate"] if c and c.get("estimate") is not None else float("nan"))
            f.write(" ".join(f"{v:.6g}" for v in vals) + "\n")
    M = {"PilotDocs": str(S["units"]["n_test_documents"]), "PilotFeatures": str(S["units"]["n_features"]),
         "PilotRows": str(S["units"]["n_rows"]),
         "PilotFwdSec": f"{S['cost']['forward_seconds_per_quality_group_median']:.1f}",
         "PilotSolveMs": f"{1000 * S['cost']['ln_solve_seconds_median']:.0f}",
         "PilotQualityGroups": str(S["cost"]["quality_groups"])}
    def cm(mode, kind, dose, m, met, sc=1.0):
        c = cur.get((mode, kind, dose, m, met))
        if not c or c.get("estimate") is None:
            return "--"
        v = sc * c["estimate"]
        return "$\\approx 0$" if abs(v) < 1e-9 else f"{v:.3g}"

    for m, mm in (("decoder", "Dec"), ("jacobian_ln", "Ln"), ("encoder_grad", "Enc"), ("mean_only", "Mean"), ("random", "Rand")):
        M[f"PilotGain{mm}"] = cm("equal_norm", "active", "0.1", m, "target_gain")
        M[f"PilotGain{mm}Inact"] = cm("equal_norm", "inactive", "0.1", m, "target_gain")
        M[f"PilotAll{mm}"] = cm("target_matched", "active", "q99", m, "drift_all_nontarget_rel")
        M[f"PilotAll{mm}Inact"] = cm("target_matched", "inactive", "q99", m, "drift_all_nontarget_rel")
        M[f"PilotNew{mm}"] = cm("target_matched", "active", "q99", m, "drift_newly_active_rel")
        M[f"PilotKL{mm}"] = cm("target_matched", "active", "q99", m, "kl_next_token", 1e3)
        M[f"PilotNorm{mm}"] = cm("target_matched", "active", "q99", m, "edit_norm")
        M[f"PilotEqAll{mm}"] = cm("equal_norm", "active", "0.1", m, "drift_all_nontarget_rel")
    M["PilotRescInactTgt"] = cm("target_matched", "inactive", "q99", "decoder_calib_rescaled", "target_err_rel")
    M["PilotOrigDec"] = cm("target_matched", "active", "q99", "decoder", "drift_orig_active_rel")
    M["PilotOrigLn"] = cm("target_matched", "active", "q99", "jacobian_ln", "drift_orig_active_rel")
    M["PilotRandInf"] = str(sum(v.get("INFEASIBLE", 0) for k, v in inf.items() if "|random|" in k and k.startswith("target_matched")))
    M["PilotRandTotal"] = str(sum(sum(v.values()) for k, v in inf.items() if "|random|" in k and k.startswith("target_matched")))
    for met, mm in (("drift_all_nontarget_rel", "All"), ("kl_next_token", "KL"), ("edit_norm", "Norm")):
        p_ = par.get(("target_matched", "active", "q99", "jacobian_ln", "encoder_grad", met))
        if p_ and p_.get("estimate") is not None:
            M[f"PilotLnEnc{mm}"] = _ci(p_)
    for gk in ("G1", "G2", "G3"):
        M[f"Pilot{macro_name(gk)}"] = S["gates"][gk]["readout"].replace("_", " ").lower()
    for kind, kk in (("active", "Act"), ("inactive", "Inact")):
        for met, mm in (("drift_all_nontarget_rel", "All"), ("drift_newly_active_rel", "New"), ("kl_next_token", "KL")):
            p = par.get(("target_matched", kind, "q99", "jacobian_ln", "decoder", met))
            if p and p.get("estimate") is not None:
                M[f"PilotD{mm}{kk}"] = _ci(p)
        mo = cur.get(("target_matched", kind, "q99", "mean_only", "kl_next_token"))
        if mo and mo.get("estimate") is not None:
            M[f"PilotMeanKL{kk}"] = _ci(mo)
        st = inf.get(f"target_matched|q99|mean_only|{kind}", {})
        M[f"PilotMeanInf{kk}"] = f"{st.get('INFEASIBLE', 0)}/{sum(st.values())}"
    return M


# ---------------------------------------------------------------- readout closure, matched denominators, REAL-02-LX


def _closure():
    p = "results/readout_closure/readout_closure_report.json"
    return json.load(open(p)) if os.path.exists(p) else None


def closure_assets() -> dict:
    C = _closure()
    a, b = C["part_a_k9"], C["part_b_pilot_subset"]
    adm = max(v for v in b["admissible_methods_max_abs_raw_minus_canonical"].values() if v is not None)
    return {"ClRawMed": f"{a['raw_drift_rel_median']:.1f}", "ClRawRepro": _sci(a["raw_vs_stored_rel_diff"]),
            "ClCanonMax": _sci(a["canonical_drift_rel_max"]), "ClKLMax": _sci(a["kl_clean_vs_h_plus_delta_max"]),
            "ClPdeltaNorm": _sci(a["norm_P_delta"]), "ClGroups": str(b["n_groups"]), "ClRows": str(b["n_rows"]),
            "ClAdmMax": _sci(adm), "ClStoredRepro": _sci(b["max_rel_diff_raw_vs_stored_csv"]),
            "ClMeanRawMed": f"{b['mean_only_raw_drift_all_nontarget_rel_median']:.0f}",
            "ClMeanCanonMax": _sci(b["mean_only_canonical_drift_all_nontarget_rel_max"]),
            "ClMeanKL": _sci(b["mean_only_max_kl_clean_vs_edit"]),
            "ClUnprojMax": f"{b['unprojected_decoder_max_abs_raw_minus_canonical_drift_all_rel']:.2f}",
            "ClLogitMax": _sci(b["max_abs_logprob_diff_h_plus_delta_vs_h_plus_Pdelta"]),
            "ClWall": f"{C['timing']['wall_seconds_total']:.0f}", "ClCPU": f"{C['timing']['cpu_seconds_process']:.0f}"}


def reagg_assets(d: str, paper: str) -> dict:
    R = json.load(open(os.path.join(d, "reagg_summary.json")))
    st = list(csv.DictReader(open(os.path.join(d, "status_counts.csv"))))
    names = {**PILOT_NAMES, "decoder_plain": "plain decoder ($\\alpha D_j$)", "mean_only": "mean-only (raw readout)"}
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Method (target-matched, 4 doses) & rows & matched & unmatched & infeasible & raw-only & docs \\", r"\midrule"]
    for m in ("decoder", "encoder_grad", "jacobian_ln", "random", "decoder_calib_rescaled", "decoder_plain", "mean_only"):
        rs = [r for r in st if r["mode"] == "target_matched" and r["method"] == m]
        tot = lambda k: sum(int(r[k]) for r in rs)
        L.append(f"{names[m]} & {tot('n_rows')} & {tot('OK')} & {tot('OK_UNMATCHED')} & {tot('INFEASIBLE')} & "
                 f"{tot('RAW_OFF_SLICE_ONLY')} & {max(int(r['n_docs_with_row']) for r in rs)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(paper, "tables", "tab_matched_denominators.tex"), "w").write("\n".join(L) + "\n")
    inter = R["intersections"]
    prim = [x for x in inter if x["set"] == "primary"]
    sec = [x for x in inter if x["set"] == "secondary"]
    g = R["gates"]
    cal = R["q99_calibration_counts"]
    ns = [c["n_positive_calibration"] for c in cal]
    M = {"RgPrimExclMax": f"{max(x['exclusion_rate'] for x in prim):.0%}".replace("%", "\\%"),
         "RgSecExclMin": f"{min(x['exclusion_rate'] for x in sec):.0%}".replace("%", "\\%"),
         "RgSecExclMax": f"{max(x['exclusion_rate'] for x in sec):.0%}".replace("%", "\\%"),
         "RgOK": str(R["row_status_totals_stored"]["OK"]), "RgUnm": str(R["row_status_totals_stored"]["OK_UNMATCHED"]),
         "RgInf": str(R["row_status_totals_stored"]["INFEASIBLE"]),
         "RgRawOnly": str(sum(int(r["RAW_OFF_SLICE_ONLY"]) for r in st)),
         "RgTgtErrMax": _sci(max(v for v in R["target_err_rel_max_over_OK_target_matched_rows"].values() if v is not None)),
         "RgGOneAsRun": _ci(g["G1_as_run"]), "RgGOneAct": _ci(g["G1_active_only"]), "RgGOneInact": _ci(g["G1_inactive_only"]),
         "RgGOneActN": str(g["G1_active_only"]["n_units"]), "RgGOneInactN": str(g["G1_inactive_only"]["n_units"]),
         "RgRescInactErr": f"{R['unmatched_baselines']['decoder_calib_rescaled|inactive|q99']['target_err_rel_median_doc']:.3f}",
         "RgNPosMin": str(min(ns)), "RgNPosMax": str(max(ns)),
         "RgTopTwo": str(sum(c["between_top_two_order_stats"] for c in cal)),
         "RgAllBelowHundred": str(R["calibration_n_positive_all_64"]["n_below_100"]),
         "RgAllNPosMin": str(R["calibration_n_positive_all_64"]["min"]),
         "RgRescActWithin": str(sum(v["n_within_tol"] for k, v in R["unmatched_baselines"].items()
                                    if k.startswith("decoder_calib_rescaled|active"))),
         "RgRescInactWithin": str(sum(v["n_within_tol"] for k, v in R["unmatched_baselines"].items()
                                      if k.startswith("decoder_calib_rescaled|inactive")))}
    return M


LX_NAMES = {"no_edit": "no edit", "decoder": "decoder ($PD_j$)", "encoder_grad": "encoder row ($PE_j^{\\top}$)",
            "jacobian_ln": "LN ($J_EP$)", "diffmean": "DiffMean ($Pv$)", "mean_only": "mean-only (control)"}


def lx_assets(d: str, paper: str) -> dict:
    S = json.load(open(os.path.join(d, "summary.json")))
    F = json.load(open(os.path.join(d, "calibration_frozen.json")))
    st = json.load(open(os.path.join(d, "stage_status.json")))
    per, par = S["per_method_at_selected_budget"], S["paired"]
    budgets = F["budgets"]["budgets"]
    pct = lambda c: "--" if not c or c.get("estimate") is None else (
        f"{100 * c['estimate']:.0f} [{100 * c['lo']:.0f}, {100 * c['hi']:.0f}]")
    L = [r"\begin{tabular}{lccccccc}", r"\toprule",
         r"Method & budget & cal.\ KL & test KL & any hit (\%) & hits/100 tok. & NLL & $\Delta a_j$ \\", r"\midrule"]
    for m in ("no_edit", "decoder", "encoder_grad", "jacobian_ln", "diffmean", "mean_only"):
        if m not in per or per[m]["n_docs"] == 0:
            b = F["chosen_budget"].get(m, "--")
            bb = "none within $\\kappa$" if b is None else b
            L.append(f"{LX_NAMES[m]} & {bb} & & & \\multicolumn{{4}}{{c}}{{\\notrun}} \\\\")
            continue
        p = per[m]
        b = "--" if m == "no_edit" else (F["control_budget"] if m == "mean_only" else F["chosen_budget"][m])
        ck = "--" if m in ("no_edit", "mean_only") else _sci(F["calibration_kl"][m][b])
        est = lambda c, sc=1.0: "--" if not c or c.get("estimate") is None else _sci(sc * c["estimate"])
        bl = b if b == "--" else b + "$\\bar r$"
        L.append(f"{LX_NAMES[m]} & {bl} & {ck} & {est(p['kl_prompt_teacher_forced'])} & "
                 f"{pct(p['any_hit_rate'])} & {est(p['hits_per_generated_token'], 100)} & "
                 f"{est(p['continuation_nll_unedited_model'])} & {est(p['target_change_mean'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(paper, "tables", "tab_lx.tex"), "w").write("\n".join(L) + "\n")
    P = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Pair ($a-b$) & $\Delta$ any hit (pp) & $\Delta$ hits/100 tok. & $\Delta$ test KL & $\Delta$ NLL \\", r"\midrule"]
    for key in ("jacobian_ln-decoder", "jacobian_ln-diffmean", "jacobian_ln-encoder_grad", "decoder-diffmean",
                "decoder-no_edit", "encoder_grad-no_edit", "jacobian_ln-no_edit", "diffmean-no_edit"):
        a_, b_ = key.split("-")
        g = lambda met: par.get(f"{key}|{met}")
        sc = lambda c, k: "--" if not c or c.get("estimate") is None else (
            f"{k * c['estimate']:.3g} [{k * c['lo']:.3g}, {k * c['hi']:.3g}]")
        na, nb = LX_NAMES[a_].split(" (")[0], LX_NAMES[b_].split(" (")[0]
        P.append(f"{na} $-$ {nb} & {sc(g('success'), 100)} & "
                 f"{sc(g('hits_per_generated_token'), 100)} & {sc(g('kl_prompt_teacher_forced'), 1)} & "
                 f"{sc(g('continuation_nll_unedited_model'), 1)} \\\\")
    P += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(paper, "tables", "tab_lx_paired.tex"), "w").write("\n".join(P) + "\n")
    run = [x for x in st if x["stage"] == "run"][-1]
    cal = [x for x in st if x["stage"] == "calibrate"][-1]
    M = {"LxFeature": str(F["feature"]), "LxDocs": str(run.get("n_test_docs", "--")),
         "LxHitWin": str(F["n_hit_windows"]), "LxNoHitWin": str(F["n_nohit_windows"]),
         "LxMedNorm": f"{F['budgets']['reference_median_norm']:.1f}", "LxKappa": f"{F['kappa_nats']:g}",
         "LxRunSec": f"{run['seconds']:.0f}", "LxCalSec": f"{cal['seconds']:.0f}",
         "LxCtrlSame": f"{S['control_mean_only']['identical_continuation_to_no_edit']}/{S['control_mean_only']['n']}",
         "LxCtrlKL": _sci(S["control_mean_only"]["max_kl"])}
    for m, mm in (("no_edit", "None"), ("decoder", "Dec"), ("encoder_grad", "Enc"), ("jacobian_ln", "Ln"),
                  ("diffmean", "Dm"), ("mean_only", "Mean")):
        p = per.get(m)
        if p and p["n_docs"]:
            M[f"LxHit{mm}"] = pct(p["any_hit_rate"])
            M[f"LxKL{mm}"] = _sci(p["kl_prompt_teacher_forced"]["estimate"])
            M[f"LxNLL{mm}"] = f"{p['continuation_nll_unedited_model']['estimate']:.2f}"
        if m not in ("no_edit", "mean_only"):
            M[f"LxBud{mm}"] = str(F["chosen_budget"][m])
    for key, kk in (("jacobian_ln-decoder", "LnDec"), ("jacobian_ln-diffmean", "LnDm"), ("jacobian_ln-encoder_grad", "LnEnc"),
                    ("decoder-diffmean", "DecDm"), ("jacobian_ln-no_edit", "LnNone"), ("decoder-no_edit", "DecNone"),
                    ("diffmean-no_edit", "DmNone"), ("encoder_grad-no_edit", "EncNone")):
        c = par.get(f"{key}|success")
        if c and c.get("estimate") is not None:
            M[f"LxP{kk}"] = f"{100 * c['estimate']:.1f} [{100 * c['lo']:.1f}, {100 * c['hi']:.1f}]"
            M[f"LxP{kk}N"] = str(c["n_units"])
    return M



def lx_blocked_assets(d: str) -> dict:
    """Macros for a BLOCKED REAL-02-LX (calibration support below the pre-set minimum)."""
    st = json.load(open(os.path.join(d, "stage_status.json")))
    cal = [x for x in st if x["stage"] == "calibrate"]
    hits = lambda x: int(x["reason"].split("only ")[1].split(" ")[0])
    v1, v11 = cal[0], cal[-1]
    return {"LxHitsVOne": str(hits(v1)), "LxHitsVOneOne": str(hits(v11)),
            "LxWindowsAll": str(hits(v11) + int(v11["n_nohit_windows"])),
            "LxCalSecVOne": f"{v1['seconds']:.0f}", "LxCalSecVOneOne": f"{v11['seconds']:.0f}",
            "LxStatus": v11["status"]}

if __name__ == "__main__":
    raise SystemExit(main())
