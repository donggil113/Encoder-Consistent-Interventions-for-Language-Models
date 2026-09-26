"""First-run toy experiments (P3-T2..T5, P3-TX). Each returns raw rows.

Independent unit = toy instance (seed). Rows inside one instance share the
same (D, E, b) and are not independent samples.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List

from . import linalg as la
from .edits import EditProblem, evaluate, run_method
from .sae_toy import (ToySAE, build_decoder, build_encoder, equiangular_decoder, gaussian_vec,
                      normalize, sparse_base_point)

INFO_KEYS = ("residual_rel", "rank", "n_constraints", "n_jvp", "n_vjp", "cgls_iters",
             "cgls_converged", "coef", "sign_flip", "lam", "budget_active", "budget",
             "repair_iters", "repair_converged", "n_extra_protected", "reason")


def _row(base: dict, method: str, res: dict, pb: EditProblem, tol: float) -> dict:
    row = dict(base)
    row["method"] = method
    row["status"] = res["status"]
    for k in INFO_KEYS:
        if k in res:
            row[k] = res[k]
    if res["delta"] is not None:
        row.update(evaluate(pb, res["delta"], tol))
    return row


# ---------------------------------------------------------------- T2 / T5


def run_linear_regimes(cfg: dict, seeds: List[int]) -> List[dict]:
    ecfg = cfg["T2_T5_linear_regimes"]
    num = cfg["numerics"]
    tol = num["identity_tol"]
    rows: List[dict] = []
    for reg in ecfg["regimes"]:
        for seed in seeds:
            rng = random.Random(f"T2:{reg['name']}:{seed}")
            d, m = reg["d"], reg["m"]
            D = build_decoder(reg["decoder"], d, m, rng, reg.get("coherence", 0.0))
            E = build_encoder(reg["encoder"], D, rng, scale_range=reg.get("scale_range"), rtol=num["pinv_rtol"])
            sae = ToySAE(E=E, b=[0.0] * m, D=D, act="identity")
            x = gaussian_vec(d, rng)
            Ginv = la.pinv(la.matmul(la.transpose(D), D), num["pinv_rtol"]) if reg["name"].startswith("R2") else None
            targets = rng.sample(range(m), min(ecfg["targets_per_instance"], m))
            for j in targets:
                others = [i for i in range(m) if i != j]
                if reg["protect"] == "all":
                    prot = others
                else:
                    prot = sorted(rng.sample(others, reg["protect_size"]))
                pb = EditProblem(sae, x, j, ecfg["alpha"], prot, num["pinv_rtol"])
                base = {"experiment_id": "P3-T2-MISMATCH", "regime": reg["name"], "seed": seed,
                        "d": d, "m": m, "target": j, "n_protected": len(prot), "alpha": ecfg["alpha"]}
                if Ginv is not None:
                    base["analytic_leak_tied"] = reg["coherence"] * math.sqrt(m - 1)
                    base["analytic_norm_ratio_ln"] = math.sqrt(Ginv[j][j])
                deltas: Dict[str, List[float]] = {}
                for meth in ecfg["methods"]:
                    res = run_method(meth, pb, random.Random(f"rand:{reg['name']}:{seed}:{j}"), num)
                    row = _row(base, meth, res, pb, tol)
                    if res["delta"] is not None:
                        deltas[meth] = res["delta"]
                    if meth == "jacobian_ln":
                        r = res["residual"]
                        rn = la.norm(r)
                        if rn > 0.0:
                            row["cert_MTr_rel"] = la.norm(la.rmatvec(res["M"], r)) / (la.frob(res["M"]) * rn)
                            row["cert_rt_over_rr"] = la.dot(r, res["t"]) / (rn * rn)
                    rows.append(row)
                # identity comparisons between methods, attached to the jacobian_ln row
                jl = next(r for r in rows[-len(ecfg["methods"]):] if r["method"] == "jacobian_ln")
                if "jacobian_ln" in deltas:
                    for other in ("decoder", "decoder_rescaled", "jacobian_ln_cgls", "encoder_grad"):
                        if other in deltas:
                            jl[f"rel_diff_vs_{other}"] = la.rel_diff(deltas["jacobian_ln"], deltas[other])
    return rows


# ---------------------------------------------------------------- T3


def _gauss_matrix(r: int, c: int, rng: random.Random) -> la.Matrix:
    return [gaussian_vec(c, rng) for _ in range(r)]


def run_feasibility(cfg: dict, seeds: List[int]) -> List[dict]:
    fc = cfg["T3_feasibility"]
    num = cfg["numerics"]
    tol = num["identity_tol"]
    rtol = num["pinv_rtol"]
    d = fc["d"]
    rows: List[dict] = []
    eid = "P3-T3-FEASIBILITY"

    def cert(M, t, r):
        rn = la.norm(r)
        return (la.norm(la.rmatvec(M, r)) / (la.frob(M) * rn), la.dot(r, t) / (rn * rn))

    for seed in seeds:
        rng = random.Random(f"T3:{seed}")
        # (a) full row rank: least-norm solution
        for k in fc["k_values"]:
            M = _gauss_matrix(k, d, rng)
            t = gaussian_vec(k, rng)
            sol = la.lstsq_min_norm(M, t, rtol)
            x = sol["x"]
            Mp = la.pinv(M, rtol)
            proj = la.matvec(Mp, la.matvec(M, x))
            rowspace_err = la.norm(la.sub(x, proj)) / la.norm(x)
            pyth_err, min_ok = 0.0, True
            for _ in range(fc["n_nullspace_probes"]):
                g = gaussian_vec(d, rng)
                n = la.sub(g, la.matvec(Mp, la.matvec(M, g)))
                xn = la.add(x, n)
                lhs = la.dot(xn, xn)
                rhs = la.dot(x, x) + la.dot(n, n)
                pyth_err = max(pyth_err, abs(lhs - rhs) / rhs)
                min_ok = min_ok and (la.norm(xn) >= la.norm(x) * (1 - 1e-12))
            cg = la.cgls(lambda v: la.matvec(M, v), lambda u: la.rmatvec(M, u), t, d,
                         rel_tol=num["cgls_rel_tol"], max_iter=num["cgls_max_iter_factor"] * d)
            cg_rel = la.rel_diff(cg["x"], x)
            res_rel = la.norm(sol["residual"]) / la.norm(t)
            rows.append({"experiment_id": eid, "check": "full_rank_least_norm", "seed": seed, "k": k,
                         "rank": sol["rank"], "residual_rel": res_rel, "rowspace_err": rowspace_err,
                         "pythagoras_rel_err": pyth_err, "nullspace_probes_not_shorter": int(min_ok),
                         "cgls_vs_dense_rel": cg_rel, "cgls_iters": cg["iters"],
                         "pass": int(res_rel <= tol and rowspace_err <= tol and pyth_err <= tol and min_ok
                                     and cg_rel <= 1e-6)})
        # (b) more constraints than dimensions: generic target infeasible
        for k in fc["k_infeasible_values"]:
            M = _gauss_matrix(k, d, rng)
            t = gaussian_vec(k, rng)
            sol = la.lstsq_min_norm(M, t, rtol)
            res_rel = la.norm(sol["residual"]) / la.norm(t)
            c1, c2 = cert(M, t, sol["residual"])
            rows.append({"experiment_id": eid, "check": "rank_deficient_infeasible", "seed": seed, "k": k,
                         "rank": sol["rank"], "residual_rel": res_rel, "cert_MTr_rel": c1, "cert_rt_over_rr": c2,
                         "pass": int(res_rel > 1e-6 and c1 <= tol and abs(c2 - 1.0) <= tol)})
            # (c) feasible control in the same rank-deficient system
            tf = la.matvec(M, gaussian_vec(d, rng))
            solf = la.lstsq_min_norm(M, tf, rtol)
            resf = la.norm(solf["residual"]) / la.norm(tf)
            rows.append({"experiment_id": eid, "check": "rank_deficient_feasible_control", "seed": seed, "k": k,
                         "rank": solf["rank"], "residual_rel": resf, "pass": int(resf <= tol)})
        # (b') low-rank rows with k <= d
        kk, rr = fc["low_rank_case"]["k"], fc["low_rank_case"]["r"]
        M = la.matmul(_gauss_matrix(kk, rr, rng), _gauss_matrix(rr, d, rng))
        t = gaussian_vec(kk, rng)
        sol = la.lstsq_min_norm(M, t, rtol)
        res_rel = la.norm(sol["residual"]) / la.norm(t)
        c1, c2 = cert(M, t, sol["residual"])
        rows.append({"experiment_id": eid, "check": "low_rank_rows", "seed": seed, "k": kk, "rank": sol["rank"],
                     "residual_rel": res_rel, "cert_MTr_rel": c1, "cert_rt_over_rr": c2,
                     "pass": int(sol["rank"] == rr and res_rel > 1e-6 and c1 <= tol and abs(c2 - 1.0) <= tol)})
        # (d) duplicated encoder row with conflicting targets
        e = gaussian_vec(d, rng)
        alpha = 1.0
        M = [e, list(e)] + _gauss_matrix(3, d, rng)
        t = [alpha, 0.0, 0.0, 0.0, 0.0]
        sol = la.lstsq_min_norm(M, t, rtol)
        rn = la.norm(sol["residual"])
        rows.append({"experiment_id": eid, "check": "duplicate_rows_conflict", "seed": seed, "k": 5,
                     "rank": sol["rank"], "residual_abs": rn, "expected_residual": alpha / math.sqrt(2),
                     "pass": int(abs(rn - alpha / math.sqrt(2)) <= tol)})
        # (e)/(f) range infeasibility and (g) TopK eviction
        rows.extend(_range_checks(seed, num))
    return rows


def _range_checks(seed: int, num: dict) -> List[dict]:
    eid = "P3-T3-FEASIBILITY"
    rng = random.Random(f"T3range:{seed}")
    tol = num["identity_tol"]
    rows = []
    d, m = 8, 12
    D = build_decoder("gaussian", d, m, rng)
    E = build_encoder("tied", D, rng)
    # ReLU: requesting a negative post-activation
    sae = ToySAE(E=E, b=[0.0] * m, D=D, act="relu")
    x = gaussian_vec(d, rng)
    p = sae.pre(x)
    act = [i for i in range(m) if p[i] > 0]
    if act:
        j = act[0]
        alpha = -(p[j] + 1.0)  # asks a_j' = -1
        pb = EditProblem(sae, x, j, alpha, [], num["pinv_rtol"])
        res = run_method("jacobian_ln", pb, rng)
        ev = evaluate(pb, res["delta"], tol)
        a_new = sae.encode(la.add(x, res["delta"]))
        rows.append({"experiment_id": eid, "check": "range_infeasible_relu", "seed": seed,
                     "attainable_flag": int(pb.target_attainable()), "achieved_post": a_new[j],
                     "requested_post": pb.v_target, "target_rel_err": ev["target_rel_err"],
                     "pass": int((not pb.target_attainable()) and a_new[j] >= 0.0 and ev["target_rel_err"] > 1e-6)})
    # JumpReLU: requesting a value in (0, theta]
    theta = 0.5
    sae = ToySAE(E=E, b=[0.0] * m, D=D, act="jumprelu", theta=theta)
    p = sae.pre(x)
    inact = [i for i in range(m) if p[i] <= theta]
    if inact:
        j = inact[0]
        alpha = 0.5 * theta  # asks a_j' = theta/2
        pb = EditProblem(sae, x, j, alpha, [], num["pinv_rtol"])
        res = run_method("jacobian_ln", pb, rng)
        a_new = sae.encode(la.add(x, res["delta"]))
        rows.append({"experiment_id": eid, "check": "range_infeasible_jumprelu", "seed": seed,
                     "attainable_flag": int(pb.target_attainable()), "achieved_post": a_new[j],
                     "requested_post": pb.v_target,
                     "pass": int((not pb.target_attainable()) and a_new[j] == 0.0)})
    # TopK: exactly k positive actives, activate an inactive target
    k = 3
    sae = ToySAE(E=E, b=[0.0] * m, D=D, act="topk", k=k)
    for trial in range(20):
        x = gaussian_vec(d, rng)
        p = sae.pre(x)
        mask = sae.active_mask(p)
        if sum(mask) != k:
            continue
        inact = [i for i in range(m) if not mask[i]]
        j = inact[0]
        prot = [i for i in range(m) if mask[i]]
        kth = min(p[i] for i in prot)
        alpha = kth + 0.5  # target value above the current k-th activation
        pb = EditProblem(sae, x, j, alpha, prot, num["pinv_rtol"])
        res = run_method("jacobian_ln", pb, rng)
        ev = evaluate(pb, res["delta"], tol)
        entered = sae.active_mask(sae.pre(la.add(x, res["delta"])))[j]
        n_active_after = sum(sae.active_mask(sae.pre(la.add(x, res["delta"]))))
        rows.append({"experiment_id": eid, "check": "topk_eviction", "seed": seed, "trial": trial,
                     "target_entered": int(entered), "n_cross_off": ev["n_cross_off"],
                     "n_active_after": n_active_after, "leak_rel_all": ev["leak_rel_all"],
                     "pass": int((not entered) or (ev["n_cross_off"] >= 1 and n_active_after == k))})
        break
    return rows


# ---------------------------------------------------------------- T4


def run_activeset_norm(cfg: dict, seeds: List[int], activations: List[dict] = None,
                       experiment_id: str = "P3-T4-ACTIVESET-NORM") -> List[dict]:
    c = cfg["T4_activeset_norm"]
    num = cfg["numerics"]
    tol = num["identity_tol"]
    methods = list(c["methods"]) + [f"trust_region_x{b}" for b in c["norm_budget_multipliers"] if b != 1.0]
    mcfg = dict(num, repair_max_iters=c["repair_max_iters"])
    rows: List[dict] = []
    for acfg in (activations if activations is not None else c["activations"]):
        aname = acfg["name"]
        label = acfg.get("label", aname)
        for seed in seeds:
            rng = random.Random(f"T4:{label}:{seed}")
            D = build_decoder(c["decoder"], c["d"], c["m"], rng)
            E = build_encoder(c["encoder"], D, rng, beta=c["encoder_beta"], scale_range=c["scale_range"],
                              rtol=num["pinv_rtol"])
            bias = acfg.get("bias", c["bias"])
            sae = ToySAE(E=E, b=[bias] * c["m"], D=D, act=aname, theta=acfg.get("theta", 0.0),
                         k=acfg.get("k", 0))
            for bi in range(c["base_points_per_instance"]):
                x = sparse_base_point(D, c["base_point_support"], c["base_point_coef_range"],
                                      c["base_point_noise"], rng)
                mask = sae.active_mask(sae.pre(x))
                active = [i for i in range(c["m"]) if mask[i]]
                inactive = [i for i in range(c["m"]) if not mask[i]]
                for kind in c["target_kinds"]:
                    pool = active if kind == "active_increase" else inactive
                    base = {"experiment_id": experiment_id, "act": label, "seed": seed,
                            "base_idx": bi, "target_kind": kind, "n_active_base": len(active)}
                    if not pool:
                        rows.append(dict(base, method="*", status="SKIP", reason="empty target pool"))
                        continue
                    j = rng.choice(pool)
                    prot = [i for i in active if i != j]
                    for alpha in c["alpha_values"]:
                        pb = EditProblem(sae, x, j, alpha, prot, num["pinv_rtol"])
                        b2 = dict(base, target=j, alpha=alpha, n_protected=len(prot),
                                  target_attainable=int(pb.target_attainable()))
                        for meth in methods:
                            res = run_method(meth, pb, random.Random(f"rand:T4:{label}:{seed}:{bi}:{kind}:{alpha}"),
                                             mcfg)
                            rows.append(_row(b2, meth, res, pb, tol))
    return rows


def run_sparsity_exploratory(cfg: dict, seeds: List[int]) -> List[dict]:
    c = cfg["T4b_sparsity_exploratory"]
    return run_activeset_norm(cfg, seeds, c["activations"], c["experiment_id"])


# ---------------------------------------------------------------- TX


def run_external_proxy(cfg: dict, seeds: List[int]) -> List[dict]:
    c = cfg["TX_external_proxy"]
    num = cfg["numerics"]
    tol = num["identity_tol"]
    d, m, sig = c["d"], c["m"], c["noise_sigma"]
    rows: List[dict] = []
    for case in c["cases"]:
        for seed in seeds:
            rng = random.Random(f"TX:{case}:{seed}")
            A = equiangular_decoder(d, m, c["coherence"], rng)
            Ap = la.pinv(A, num["pinv_rtol"])  # external readout z_true = A^+ x
            row_scale = la.frob(Ap) / math.sqrt(m)

            def noisy_dec():
                cols = [normalize(la.axpy(sig / math.sqrt(d), gaussian_vec(d, rng), la.column(A, j)))
                        for j in range(m)]
                return la.transpose(cols)

            def noisy_enc():
                return [la.axpy(sig * row_scale / math.sqrt(d), gaussian_vec(d, rng), row) for row in Ap]

            if case == "encoder_accurate":
                D, E = noisy_dec(), [list(r) for r in Ap]
            elif case == "decoder_accurate":
                D, E = [list(r) for r in A], noisy_enc()
            else:
                D, E = noisy_dec(), noisy_enc()
            sae = ToySAE(E=E, b=[0.0] * m, D=D, act="identity")
            x = gaussian_vec(d, rng)
            for j in rng.sample(range(m), min(c["targets_per_instance"], m)):
                prot = [i for i in range(m) if i != j]
                pb = EditProblem(sae, x, j, 1.0, prot, num["pinv_rtol"])
                for meth in c["methods"]:
                    res = run_method(meth, pb, rng, num)
                    row = {"experiment_id": "P3-TX-EXTPROXY", "case": case, "seed": seed, "target": j,
                           "method": meth, "status": res["status"]}
                    if res["delta"] is not None:
                        ev = evaluate(pb, res["delta"], tol)
                        dz = la.matvec(Ap, res["delta"])
                        row.update({
                            "int_target_rel_err": ev["target_rel_err"],
                            "int_leak_rel": ev["leak_rel_all"],
                            "ext_target_rel_err": abs(dz[j] - 1.0),
                            "ext_leak_rel": math.sqrt(sum(dz[i] ** 2 for i in range(m) if i != j)),
                            "norm_ratio": ev["norm_ratio"],
                        })
                    rows.append(row)
    return rows
