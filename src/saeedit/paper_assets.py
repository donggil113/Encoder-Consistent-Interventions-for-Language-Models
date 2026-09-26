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


if __name__ == "__main__":
    raise SystemExit(main())
