#!/usr/bin/env python3
"""Analyze OmniIsaacGymEnvs play/eval CSV.

Implements the recommended workflow:
- Layer-0: unadjusted group diffs with bootstrap CIs.
- Layer-0+: regression adjustment (logit for success, OLS+HC1 for continuous).
- Stratified effect curves: bin by conditions and plot delta curves with CIs.

Supports two modes:
1) Single-CSV sim-vs-base (legacy): compares obs_source == 'sim' vs 'base'.
2) Run-vs-Run (two CSVs): compares CSV-A vs CSV-B by assigning a group label.

Usage:
    # Legacy (sim vs base within one CSV)
    python analyze_play_csv.py --csv runs/play_CSV/Mar26_20-37-49.csv --outdir runs/play_analysis/Mar26_20-37-49

    # Run-vs-Run (two CSVs)
    python analyze_play_csv.py --csv-a runs/play_CSV/A.csv --csv-b runs/play_CSV/B.csv \
        --label-a randomized --label-b baseline --outdir runs/play_analysis/A_vs_B

Outputs (in outdir):
  - qc_summary.json
  - layer0_summary.csv
  - layer0plus_regression.txt
  - stratified_effects.csv
  - plots_stratified.png

Notes:
- This script does NOT modify any simulation code; it is offline analysis only.
- Requires: numpy, pandas, matplotlib, statsmodels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _require_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")


def _as_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _as_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def _pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def _safe_str(x: object) -> str:
    try:
        s = str(x)
    except Exception:
        return ""
    return s.strip()


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize dtypes for a few known columns
    for c in ["success", "collision", "out_of_bounds"]:
        if c in df.columns:
            df[c] = _as_float(df[c])
    # Normalize string columns that we frequently group on.
    for c in ["obs_source", "done_reason", "obstacles_hash"]:
        if c in df.columns:
            df[c] = df[c].astype(str).map(_safe_str)
    return df


def _bootstrap_diff(
    x_sim: np.ndarray,
    x_base: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    """IID bootstrap within each group; returns diff=sim-base with percentile CI."""

    rng = np.random.default_rng(seed)
    n_sim = len(x_sim)
    n_base = len(x_base)
    if n_sim == 0 or n_base == 0:
        return {
            "n_sim": float(n_sim),
            "n_base": float(n_base),
            "diff": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
        }

    point = float(stat_fn(x_sim) - stat_fn(x_base))
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        bs_sim = x_sim[rng.integers(0, n_sim, size=n_sim)]
        bs_base = x_base[rng.integers(0, n_base, size=n_base)]
        diffs[i] = stat_fn(bs_sim) - stat_fn(bs_base)

    ci_lo, ci_hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "n_sim": float(n_sim),
        "n_base": float(n_base),
        "diff": float(point),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
    }


def _mean_stat(x: np.ndarray) -> float:
    return float(np.nanmean(x))


def _rate_stat(x: np.ndarray) -> float:
    # x should be 0/1
    return float(np.nanmean(x))


def _median_stat(x: np.ndarray) -> float:
    return float(np.nanmedian(x))


@dataclass(frozen=True)
class QCSummary:
    n_total: int
    group_col: str
    treat: str
    control: str
    n_treat: int
    n_control: int
    obstacles_hash_treat_unique: int
    obstacles_hash_control_unique: int
    obstacles_hash_intersection: int
    obstacles_hash_pair_upper_bound: int
    missingness: Dict[str, Dict[str, int]]
    done_reasons: Dict[str, Dict[str, int]]


def qc_diagnostics_groups(df: pd.DataFrame, *, group_col: str, treat: str, control: str) -> QCSummary:
    _require_column(df, group_col)

    df = df.copy()
    g = df[group_col].astype(str).map(_safe_str)
    df[group_col] = g

    treat_df = df[df[group_col] == str(treat)]
    control_df = df[df[group_col] == str(control)]

    # Obstacles hash diagnostics (strict equality matching)
    if "obstacles_hash" in df.columns:
        treat_hash = treat_df["obstacles_hash"].astype(str).map(_safe_str)
        control_hash = control_df["obstacles_hash"].astype(str).map(_safe_str)

        treat_set = set([h for h in treat_hash.unique().tolist() if h and h.lower() != "nan"])
        control_set = set([h for h in control_hash.unique().tolist() if h and h.lower() != "nan"])
        inter = treat_set & control_set

        treat_counts = treat_hash.value_counts()
        control_counts = control_hash.value_counts()
        pair_ub = 0
        for h in inter:
            pair_ub += int(min(treat_counts.get(h, 0), control_counts.get(h, 0)))

        treat_unique = len(treat_set)
        control_unique = len(control_set)
        inter_size = len(inter)
    else:
        treat_unique = control_unique = inter_size = pair_ub = 0

    # Missingness for a few key metrics
    check_cols = [
        "success",
        "return_scaled",
        "return_raw",
        "time_to_goal_sec",
        "episode_len_steps",
        "path_length",
        "path_efficiency",
        "collision",
        "out_of_bounds",
    ]
    missingness: Dict[str, Dict[str, int]] = {str(treat): {}, str(control): {}}
    for col in check_cols:
        if col not in df.columns:
            continue
        missingness[str(treat)][col] = int(treat_df[col].isna().sum())
        missingness[str(control)][col] = int(control_df[col].isna().sum())

    # Done reasons
    done_reasons: Dict[str, Dict[str, int]] = {str(treat): {}, str(control): {}}
    if "done_reason" in df.columns:
        for mode, sub in [(str(treat), treat_df), (str(control), control_df)]:
            vc = sub["done_reason"].astype(str).value_counts(dropna=False)
            done_reasons[mode] = {str(k): int(v) for k, v in vc.items()}

    return QCSummary(
        n_total=int(len(df)),
        group_col=str(group_col),
        treat=str(treat),
        control=str(control),
        n_treat=int(len(treat_df)),
        n_control=int(len(control_df)),
        obstacles_hash_treat_unique=int(treat_unique),
        obstacles_hash_control_unique=int(control_unique),
        obstacles_hash_intersection=int(inter_size),
        obstacles_hash_pair_upper_bound=int(pair_ub),
        missingness=missingness,
        done_reasons=done_reasons,
    )


def qc_diagnostics(df: pd.DataFrame) -> QCSummary:
    """Backward-compatible wrapper: compares obs_source sim vs base."""
    return qc_diagnostics_groups(df, group_col="obs_source", treat="sim", control="base")


def layer0_bootstrap_groups(
    df: pd.DataFrame,
    *,
    group_col: str,
    treat: str,
    control: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Unadjusted diffs with bootstrap CIs (treat-control)."""

    _require_column(df, group_col)
    _require_column(df, "success")

    treat_df = df[df[group_col] == str(treat)].copy()
    control_df = df[df[group_col] == str(control)].copy()

    out_rows: List[Dict[str, object]] = []

    def add_metric(name: str, treat_arr: np.ndarray, control_arr: np.ndarray, stat: Callable[[np.ndarray], float]):
        res = _bootstrap_diff(treat_arr, control_arr, stat_fn=stat, n_boot=n_boot, seed=seed)
        out_rows.append(
            {
                "metric": name,
                "n_treat": int(res["n_sim"]),
                "n_control": int(res["n_base"]),
                "diff_treat_minus_control": res["diff"],
                "ci95_lo": res["ci_lo"],
                "ci95_hi": res["ci_hi"],
            }
        )

    # Binary rates
    success_t = _as_float(treat_df["success"]).to_numpy(dtype=np.float64)
    success_c = _as_float(control_df["success"]).to_numpy(dtype=np.float64)
    add_metric("success_rate", success_t, success_c, _rate_stat)

    for bin_col in ["collision", "out_of_bounds"]:
        if bin_col in df.columns:
            add_metric(
                f"{bin_col}_rate",
                _as_float(treat_df[bin_col]).to_numpy(dtype=np.float64),
                _as_float(control_df[bin_col]).to_numpy(dtype=np.float64),
                _rate_stat,
            )

    # Continuous outcomes
    for col in ["return_scaled", "return_raw", "path_efficiency", "path_length"]:
        if col not in df.columns:
            continue
        add_metric(
            f"{col}_mean",
            _as_float(treat_df[col]).to_numpy(dtype=np.float64),
            _as_float(control_df[col]).to_numpy(dtype=np.float64),
            _mean_stat,
        )
        add_metric(
            f"{col}_median",
            _as_float(treat_df[col]).to_numpy(dtype=np.float64),
            _as_float(control_df[col]).to_numpy(dtype=np.float64),
            _median_stat,
        )

    # Time to goal for successes only (avoid NaNs)
    if "time_to_goal_sec" in df.columns:
        treat_succ = treat_df[_as_float(treat_df["success"]) == 1.0]
        control_succ = control_df[_as_float(control_df["success"]) == 1.0]
        add_metric(
            "time_to_goal_sec_success_mean",
            _as_float(treat_succ["time_to_goal_sec"]).to_numpy(dtype=np.float64),
            _as_float(control_succ["time_to_goal_sec"]).to_numpy(dtype=np.float64),
            _mean_stat,
        )

    return pd.DataFrame(out_rows)


def layer0_bootstrap(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Backward-compatible wrapper: compares obs_source sim vs base."""
    out = layer0_bootstrap_groups(
        df,
        group_col="obs_source",
        treat="sim",
        control="base",
        n_boot=n_boot,
        seed=seed,
    )
    # Preserve legacy column names.
    rename = {
        "n_treat": "n_sim",
        "n_control": "n_base",
        "diff_treat_minus_control": "diff_sim_minus_base",
    }
    for k, v in rename.items():
        if k in out.columns and v not in out.columns:
            out = out.rename(columns={k: v})
    return out


def _build_design_matrix_groups(
    df: pd.DataFrame, *, group_col: str, treat: str, control: str
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build X, y_success, treatment indicator (treat=1, control=0)."""

    _require_column(df, group_col)
    _require_column(df, "success")

    # Treatment indicator: 1 if sim, 0 if base
    t = (df[group_col].astype(str) == str(treat)).astype(int)

    y = _as_float(df["success"])  # 0/1

    # Covariates (requested)
    covars = {}
    for col in ["straight_line_dist", "min_obs_dist_start", "sim_mass_rel", "start_vx", "start_vy", "start_wz"]:
        if col in df.columns:
            covars[col] = _as_float(df[col])

    # Yaw: use sin/cos if present
    if "start_yaw" in df.columns:
        yaw = _as_float(df["start_yaw"])
        covars["start_yaw_sin"] = np.sin(yaw)
        covars["start_yaw_cos"] = np.cos(yaw)

    X = pd.DataFrame(covars)

    # Add intercept and treatment
    X = X.copy()
    X.insert(0, "intercept", 1.0)
    X.insert(1, "treat", t.astype(float))

    # Drop rows with missing in any used variable
    used = pd.concat([y, X], axis=1)
    used = used.dropna(axis=0)

    y_clean = used["success"]
    t_clean = used["treat"]
    X_clean = used.drop(columns=["success"])

    return X_clean, y_clean, t_clean


def _build_design_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Backward-compatible wrapper: compares obs_source sim vs base."""
    return _build_design_matrix_groups(df, group_col="obs_source", treat="sim", control="base")


def layer0plus_regression_groups(
    df: pd.DataFrame,
    *,
    group_col: str,
    treat: str,
    control: str,
    out_txt_path: Path,
) -> Dict[str, float]:
    """Fit regression-adjusted models and write human-readable summary to text file."""

    # Import locally so the script can still run Layer-0 if statsmodels is missing.
    try:
        import statsmodels.api as sm
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "statsmodels is required for Layer-0+ regression. Install it via: pip install statsmodels"
        ) from e

    X, y_success, _t = _build_design_matrix_groups(df, group_col=group_col, treat=treat, control=control)

    # Logistic regression (GLM Binomial) for success
    glm_binom = sm.GLM(y_success, X, family=sm.families.Binomial())
    res_glm = glm_binom.fit(cov_type="HC1")

    # Continuous outcomes: OLS + HC1
    cont_outcomes = []
    for col in ["return_scaled", "path_efficiency", "path_length"]:
        if col not in df.columns:
            continue
        y = _as_float(df[col])
        used = pd.concat([y.rename(col), X], axis=1).dropna(axis=0)
        y_c = used[col]
        X_c = used.drop(columns=[col])
        ols = sm.OLS(y_c, X_c)
        cont_outcomes.append((col, ols.fit(cov_type="HC1")))

    # Write summary
    lines: List[str] = []
    lines.append("Layer-0+ regression adjustment (robust SE: HC1)\n")
    lines.append("Design matrix columns:\n")
    lines.append("  " + ", ".join(list(X.columns)) + "\n\n")

    # Success model
    beta = float(res_glm.params.get("treat", np.nan))
    se = float(res_glm.bse.get("treat", np.nan))
    ci_lo = beta - 1.96 * se
    ci_hi = beta + 1.96 * se
    lines.append("[Success | GLM Binomial (logit)]\n")
    lines.append(res_glm.summary().as_text())
    lines.append("\n\n")
    lines.append(f"treat ({treat} vs {control}) coefficient:\n")
    lines.append(f"  beta={beta:.6g}, se={se:.6g}, 95%CI=[{ci_lo:.6g}, {ci_hi:.6g}]\n")
    lines.append(f"  odds_ratio=exp(beta)={math.exp(beta):.6g}, 95%CI=[{math.exp(ci_lo):.6g}, {math.exp(ci_hi):.6g}]\n\n")

    # Continuous models
    for col, res in cont_outcomes:
        b = float(res.params.get("treat", np.nan))
        s = float(res.bse.get("treat", np.nan))
        lo = b - 1.96 * s
        hi = b + 1.96 * s
        lines.append(f"[{col} | OLS + HC1]\n")
        lines.append(res.summary().as_text())
        lines.append("\n\n")
        lines.append(f"treat ({treat} minus {control}) coefficient:\n")
        lines.append(f"  beta={b:.6g}, se={s:.6g}, 95%CI=[{lo:.6g}, {hi:.6g}]\n\n")

    out_txt_path.write_text("".join(lines), encoding="utf-8")

    return {
        "n_used_success": int(res_glm.nobs),
        "success_beta_treat": beta,
        "success_or_treat": float(math.exp(beta)),
    }


def layer0plus_regression(df: pd.DataFrame, out_txt_path: Path) -> Dict[str, float]:
    """Backward-compatible wrapper: compares obs_source sim vs base."""
    return layer0plus_regression_groups(df, group_col="obs_source", treat="sim", control="base", out_txt_path=out_txt_path)


def stratified_effects(
    df: pd.DataFrame,
    strat_cols: Sequence[str],
    n_bins: int,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Backward-compatible wrapper: compares obs_source sim vs base."""

    out = stratified_effects_groups(
        df,
        group_col="obs_source",
        treat="sim",
        control="base",
        strat_cols=strat_cols,
        n_bins=n_bins,
        n_boot=n_boot,
        seed=seed,
    )
    # Preserve legacy column names.
    rename = {
        "n_treat": "n_sim",
        "n_control": "n_base",
    }
    for k, v in rename.items():
        if k in out.columns and v not in out.columns:
            out = out.rename(columns={k: v})
    return out


def stratified_effects_groups(
    df: pd.DataFrame,
    *,
    group_col: str,
    treat: str,
    control: str,
    strat_cols: Sequence[str],
    n_bins: int,
    min_per_bin: int = 5,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Compute stratified deltas by quantile bins for each strat col (treat-control)."""

    _require_column(df, group_col)
    _require_column(df, "success")

    rng = np.random.default_rng(seed)

    results: List[Dict[str, object]] = []

    for col in strat_cols:
        if col not in df.columns:
            continue

        x = _as_float(df[col])
        # quantile edges; drop duplicate edges to avoid empty bins
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.nanquantile(x.to_numpy(dtype=np.float64), qs))
        if len(edges) < 3:
            continue

        # assign bins
        bin_idx = np.digitize(x.to_numpy(dtype=np.float64), edges[1:-1], right=True)
        # bins are 0..(len(edges)-2)

        for b in range(len(edges) - 1):
            mask = bin_idx == b
            sub = df.loc[mask].copy()
            if len(sub) < 2 * int(min_per_bin):
                continue

            treat_df = sub[sub[group_col] == str(treat)]
            control_df = sub[sub[group_col] == str(control)]
            if len(treat_df) < int(min_per_bin) or len(control_df) < int(min_per_bin):
                continue

            # point estimates
            succ_t = _as_float(treat_df["success"]).to_numpy(dtype=np.float64)
            succ_c = _as_float(control_df["success"]).to_numpy(dtype=np.float64)
            d_succ = float(np.nanmean(succ_t) - np.nanmean(succ_c))

            d_ret = float("nan")
            if "return_scaled" in df.columns:
                d_ret = float(
                    np.nanmean(_as_float(treat_df["return_scaled"]).to_numpy(dtype=np.float64))
                    - np.nanmean(_as_float(control_df["return_scaled"]).to_numpy(dtype=np.float64))
                )

            # bootstrap CI within bin
            def boot_once(arr: np.ndarray) -> np.ndarray:
                return arr[rng.integers(0, len(arr), size=len(arr))]

            diffs_succ = np.empty(n_boot, dtype=np.float64)
            diffs_ret = np.empty(n_boot, dtype=np.float64)

            for i in range(n_boot):
                bs_s = boot_once(succ_t)
                bs_b = boot_once(succ_c)
                diffs_succ[i] = np.nanmean(bs_s) - np.nanmean(bs_b)

                if "return_scaled" in df.columns:
                    rs = _as_float(treat_df["return_scaled"]).to_numpy(dtype=np.float64)
                    rb = _as_float(control_df["return_scaled"]).to_numpy(dtype=np.float64)
                    diffs_ret[i] = np.nanmean(boot_once(rs)) - np.nanmean(boot_once(rb))
                else:
                    diffs_ret[i] = np.nan

            lo_s, hi_s = np.nanquantile(diffs_succ, [0.025, 0.975])
            lo_r, hi_r = np.nanquantile(diffs_ret, [0.025, 0.975])

            results.append(
                {
                    "group_col": str(group_col),
                    "treat": str(treat),
                    "control": str(control),
                    "strat_col": col,
                    "bin": b,
                    "bin_lo": float(edges[b]),
                    "bin_hi": float(edges[b + 1]),
                    "n": int(len(sub)),
                    "n_treat": int(len(treat_df)),
                    "n_control": int(len(control_df)),
                    "delta_success": d_succ,
                    "delta_success_ci95_lo": float(lo_s),
                    "delta_success_ci95_hi": float(hi_s),
                    "delta_return_scaled": d_ret,
                    "delta_return_scaled_ci95_lo": float(lo_r),
                    "delta_return_scaled_ci95_hi": float(hi_r),
                }
            )

    return pd.DataFrame(results)


def _suggest_bins_for_min_per_bin(
    *,
    n_treat: int,
    n_control: int,
    bins_requested: int,
    min_per_bin: int,
) -> Dict[str, int]:
    """Heuristic to suggest bins so that each bin can have >= min_per_bin per group.

    Note: quantile binning won't guarantee exact balance per group, but this is a
    useful sanity bound.
    """

    n_treat = int(max(0, n_treat))
    n_control = int(max(0, n_control))
    bins_requested = int(max(1, bins_requested))
    min_per_bin = int(max(1, min_per_bin))

    max_bins_treat = n_treat // min_per_bin
    max_bins_control = n_control // min_per_bin
    max_bins = int(max(0, min(max_bins_treat, max_bins_control)))

    # We need at least 2 bins to form a curve.
    suggested = int(min(bins_requested, max(2, max_bins)))
    feasible = int(max_bins >= 2)

    return {
        "bins_requested": bins_requested,
        "min_per_bin": min_per_bin,
        "max_bins": max_bins,
        "bins_suggested": suggested,
        "feasible": feasible,
    }


def randomization_qc(
    df: pd.DataFrame,
    *,
    group_col: str,
    cols: Sequence[str],
    constant_std_eps: float = 1e-6,
) -> pd.DataFrame:
    """Summarize randomization columns per group.

    Returns a long-form DataFrame with one row per (group, col).
    """

    _require_column(df, group_col)

    rows: List[Dict[str, object]] = []
    groups = sorted([g for g in df[group_col].dropna().astype(str).map(_safe_str).unique().tolist() if g])
    for g in groups:
        sub = df[df[group_col].astype(str).map(_safe_str) == g]
        for c in cols:
            if c not in sub.columns:
                rows.append({"group": g, "col": c, "present": False})
                continue
            x = _as_float(sub[c]).to_numpy(dtype=np.float64)
            x = x[np.isfinite(x)]
            if x.size == 0:
                rows.append({"group": g, "col": c, "present": True, "n": 0})
                continue
            p5, p50, p95 = np.quantile(x, [0.05, 0.5, 0.95])
            std = float(np.std(x))
            rows.append(
                {
                    "group": g,
                    "col": c,
                    "present": True,
                    "n": int(x.size),
                    "mean": float(np.mean(x)),
                    "std": std,
                    "min": float(np.min(x)),
                    "p5": float(p5),
                    "p50": float(p50),
                    "p95": float(p95),
                    "max": float(np.max(x)),
                    "is_constant": bool(std < float(constant_std_eps)),
                }
            )
    return pd.DataFrame(rows)


def _sanitize_filename_component(s: str) -> str:
    s = _safe_str(s)
    if not s:
        return ""
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _find_constant_columns(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    std_eps: float = 1e-6,
) -> Dict[str, float]:
    """Return mapping of {col: std} for columns that are (near) constant."""
    out: Dict[str, float] = {}
    for c in cols:
        if c not in df.columns:
            continue
        x = _as_float(df[c]).to_numpy(dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        s = float(np.std(x))
        if s < float(std_eps):
            out[c] = s
    return out


def _add_log_transforms(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Add log_ prefixed columns for positive-valued scalars.

    Uses log(x) with x clipped to a small positive epsilon to avoid -inf.
    """
    out = df.copy()
    eps = 1e-9
    for c in cols:
        if c not in out.columns:
            continue
        x = _as_float(out[c]).to_numpy(dtype=np.float64)
        out[f"log_{c}"] = np.log(np.clip(x, eps, None))
    return out


def _build_attribution_design_matrix(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    param_cols: Sequence[str],
    control_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build design matrix X and outcome y for within-group attribution models."""

    _require_column(df, outcome_col)

    y = _as_float(df[outcome_col])

    covars: Dict[str, pd.Series] = {}
    for c in param_cols:
        if c in df.columns:
            covars[c] = _as_float(df[c])

    for c in control_cols:
        if c in df.columns:
            covars[c] = _as_float(df[c])

    # Optional yaw encoding if user didn't provide sin/cos.
    if "start_yaw" in df.columns and ("start_yaw_sin" not in covars and "start_yaw_cos" not in covars):
        yaw = _as_float(df["start_yaw"])
        covars["start_yaw_sin"] = np.sin(yaw)
        covars["start_yaw_cos"] = np.cos(yaw)

    X = pd.DataFrame(covars)
    X = X.copy()
    X.insert(0, "intercept", 1.0)

    used = pd.concat([y.rename(outcome_col), X], axis=1).dropna(axis=0)
    y_clean = used[outcome_col]
    X_clean = used.drop(columns=[outcome_col])
    return X_clean, y_clean


def attribution_logit_within_group(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    param_cols: Sequence[str],
    control_cols: Sequence[str],
    out_txt_path: Path,
    constant_std_eps: float = 1e-6,
    use_log_params: bool = True,
) -> pd.DataFrame:
    """Fit a within-group GLM Binomial(logit) attribution model.

    Intended use: run-vs-run comparisons where baseline has constant params;
    this fits only on the randomized (varying) subset.
    """

    try:
        import statsmodels.api as sm
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "statsmodels is required for attribution logit. Install it via: pip install statsmodels"
        ) from e

    df_sub = df.copy()

    if use_log_params:
        df_sub = _add_log_transforms(df_sub, param_cols)
        param_cols_model = [f"log_{c}" for c in param_cols if f"log_{c}" in df_sub.columns]
    else:
        param_cols_model = [c for c in param_cols if c in df_sub.columns]

    const = _find_constant_columns(df_sub, param_cols_model, std_eps=constant_std_eps)
    param_cols_model = [c for c in param_cols_model if c not in const]

    X, y = _build_attribution_design_matrix(
        df_sub,
        outcome_col=outcome_col,
        param_cols=param_cols_model,
        control_cols=control_cols,
    )

    glm_binom = sm.GLM(y, X, family=sm.families.Binomial())
    res = glm_binom.fit(cov_type="HC1")

    rows: List[Dict[str, object]] = []
    for name in X.columns:
        beta = float(res.params.get(name, np.nan))
        se = float(res.bse.get(name, np.nan))
        lo = beta - 1.96 * se
        hi = beta + 1.96 * se
        rows.append(
            {
                "term": name,
                "beta": beta,
                "se_hc1": se,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "odds_ratio": float(math.exp(beta)) if np.isfinite(beta) else float("nan"),
                "or_ci95_lo": float(math.exp(lo)) if np.isfinite(lo) else float("nan"),
                "or_ci95_hi": float(math.exp(hi)) if np.isfinite(hi) else float("nan"),
            }
        )
    coef_df = pd.DataFrame(rows)

    lines: List[str] = []
    lines.append(f"[Attribution logit | outcome={outcome_col}]\n")
    lines.append(f"n_used={int(res.nobs)}\n")
    lines.append(f"param_transform={'log' if use_log_params else 'raw'}\n")
    lines.append("\nDesign matrix columns:\n")
    lines.append("  " + ", ".join(list(X.columns)) + "\n\n")
    if const:
        lines.append("Dropped near-constant param columns (std < eps):\n")
        for k, v in const.items():
            lines.append(f"  - {k}: std={v:.3g}\n")
        lines.append("\n")
    lines.append(res.summary().as_text())
    lines.append("\n\n")

    lines.append("Parameter terms (OR=exp(beta), robust SE=HC1):\n")
    if len(param_cols_model) == 0:
        lines.append("  (none; all param cols missing or near-constant)\n")
    else:
        sub = coef_df[coef_df["term"].isin(param_cols_model)].copy()
        for _, r in sub.iterrows():
            lines.append(
                "  {term}: beta={beta:.4g}, OR={odds_ratio:.4g}, 95%CI_OR=[{or_ci95_lo:.4g}, {or_ci95_hi:.4g}]\n".format(
                    term=r["term"],
                    beta=float(r["beta"]),
                    odds_ratio=float(r["odds_ratio"]),
                    or_ci95_lo=float(r["or_ci95_lo"]),
                    or_ci95_hi=float(r["or_ci95_hi"]),
                )
            )

    out_txt_path.write_text("".join(lines), encoding="utf-8")
    return coef_df


def binned_sensitivity_table(
    df: pd.DataFrame,
    *,
    param_col: str,
    n_bins: int,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Quantile-bin a parameter and aggregate selected metrics per bin."""

    if param_col not in df.columns:
        return pd.DataFrame([])

    x = _as_float(df[param_col]).to_numpy(dtype=np.float64)
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.unique(np.nanquantile(x, qs))
    if len(edges) < 3:
        return pd.DataFrame([])

    bin_idx = np.digitize(x, edges[1:-1], right=True)

    out_rows: List[Dict[str, object]] = []
    for b in range(len(edges) - 1):
        sub = df.loc[bin_idx == b]
        if len(sub) == 0:
            continue

        row: Dict[str, object] = {
            "param_col": param_col,
            "bin": int(b),
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "n": int(len(sub)),
        }

        if "success_rate" in metrics and "success" in sub.columns:
            row["success_rate"] = float(np.nanmean(_as_float(sub["success"]).to_numpy(dtype=np.float64)))
        if "collision_rate" in metrics and "collision" in sub.columns:
            row["collision_rate"] = float(np.nanmean(_as_float(sub["collision"]).to_numpy(dtype=np.float64)))
        if "out_of_bounds_rate" in metrics and "out_of_bounds" in sub.columns:
            row["out_of_bounds_rate"] = float(np.nanmean(_as_float(sub["out_of_bounds"]).to_numpy(dtype=np.float64)))

        if "path_efficiency_mean" in metrics and "path_efficiency" in sub.columns:
            row["path_efficiency_mean"] = float(
                np.nanmean(_as_float(sub["path_efficiency"]).to_numpy(dtype=np.float64))
            )

        if "action_saturation_rate_mean" in metrics and "action_saturation_rate" in sub.columns:
            row["action_saturation_rate_mean"] = float(
                np.nanmean(_as_float(sub["action_saturation_rate"]).to_numpy(dtype=np.float64))
            )

        if "time_to_goal_sec_success_mean" in metrics and "time_to_goal_sec" in sub.columns and "success" in sub.columns:
            succ = sub[_as_float(sub["success"]) == 1.0]
            if len(succ) > 0:
                row["time_to_goal_sec_success_mean"] = float(
                    np.nanmean(_as_float(succ["time_to_goal_sec"]).to_numpy(dtype=np.float64))
                )
            else:
                row["time_to_goal_sec_success_mean"] = float("nan")

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def plot_stratified_effects(strat_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if strat_df.empty:
        return

    strat_cols = strat_df["strat_col"].unique().tolist()
    treat = str(strat_df["treat"].iloc[0]) if "treat" in strat_df.columns and len(strat_df) else "treat"
    control = str(strat_df["control"].iloc[0]) if "control" in strat_df.columns and len(strat_df) else "control"

    fig, axes = plt.subplots(len(strat_cols), 2, figsize=(12, 4 * len(strat_cols)), squeeze=False)

    for i, col in enumerate(strat_cols):
        sub = strat_df[strat_df["strat_col"] == col].sort_values("bin")
        x_mid = (sub["bin_lo"].to_numpy() + sub["bin_hi"].to_numpy()) / 2.0

        # Success deltas
        ax = axes[i, 0]
        y = sub["delta_success"].to_numpy()
        yerr = np.vstack(
            [
                y - sub["delta_success_ci95_lo"].to_numpy(),
                sub["delta_success_ci95_hi"].to_numpy() - y,
            ]
        )
        ax.errorbar(x_mid, y, yerr=yerr, fmt="o-", capsize=3)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(f"Δ success ({treat}-{control}) vs {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Δ success")

        # Return deltas
        ax = axes[i, 1]
        y = sub["delta_return_scaled"].to_numpy()
        yerr = np.vstack(
            [
                y - sub["delta_return_scaled_ci95_lo"].to_numpy(),
                sub["delta_return_scaled_ci95_hi"].to_numpy() - y,
            ]
        )
        ax.errorbar(x_mid, y, yerr=yerr, fmt="o-", capsize=3)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(f"Δ return_scaled ({treat}-{control}) vs {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Δ return_scaled")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="", help="Input play CSV path (legacy: sim vs base)")
    p.add_argument("--csv-a", type=str, default="", help="CSV-A path (run-vs-run)")
    p.add_argument("--csv-b", type=str, default="", help="CSV-B path (run-vs-run)")
    p.add_argument("--label-a", type=str, default="A", help="Group label for CSV-A")
    p.add_argument("--label-b", type=str, default="B", help="Group label for CSV-B")
    p.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Output directory. Default: runs/play_analysis/<csv_stem>/",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for bootstrap")
    p.add_argument("--n-boot", type=int, default=5000, help="Bootstrap samples for Layer-0")
    p.add_argument("--n-boot-strat", type=int, default=1000, help="Bootstrap samples per stratum")
    p.add_argument("--bins", type=int, default=5, help="Quantile bins for stratified curves")
    p.add_argument(
        "--min-per-bin",
        type=int,
        default=20,
        help="Minimum samples per group (treat/control) in each stratified bin. Recommend: 20.",
    )
    p.add_argument(
        "--auto-bins",
        action="store_true",
        help="Automatically clamp --bins down to a suggested value based on sample size and --min-per-bin.",
    )
    p.add_argument(
        "--strat-cols",
        type=str,
        default="min_obs_dist_start,straight_line_dist,sim_mass_rel",
        help="Comma-separated stratification columns",
    )
    p.add_argument(
        "--randomization-cols",
        type=str,
        default="thruster_mul,thruster_left_mul,thruster_right_mul,k_drag,k_Iz",
        help="Comma-separated randomization columns to QC (per group)",
    )
    p.add_argument(
        "--attrib-group",
        type=str,
        default="",
        help="Which group label to use for within-group attribution (default: treat group)",
    )
    p.add_argument(
        "--attrib-outcome",
        type=str,
        default="collision",
        help="Binary outcome for attribution logit (collision|success|out_of_bounds)",
    )
    p.add_argument(
        "--attrib-param-cols",
        type=str,
        default="thruster_mul,k_drag,k_Iz",
        help="Comma-separated parameter columns for attribution logit (within-group)",
    )
    p.add_argument(
        "--attrib-control-cols",
        type=str,
        default="straight_line_dist,min_obs_dist_start,sim_mass_rel,start_vx,start_vy,start_wz,start_yaw_sin,start_yaw_cos",
        help="Comma-separated control columns for attribution logit (within-group)",
    )
    p.add_argument(
        "--attrib-log-params",
        action="store_true",
        help="Use log() transform for parameter columns in attribution logit",
    )
    p.add_argument(
        "--bins-params",
        action="store_true",
        help="Also compute within-group binned sensitivity tables for attrib-param-cols",
    )
    p.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    args = p.parse_args(argv)

    # Load mode: either a single CSV (legacy) or two CSVs (run-vs-run).
    csv_a = _safe_str(args.csv_a)
    csv_b = _safe_str(args.csv_b)
    csv_single = _safe_str(args.csv)

    use_run_vs_run = bool(csv_a and csv_b)
    if use_run_vs_run and csv_single:
        raise ValueError("Provide either --csv (legacy) OR (--csv-a and --csv-b), not both")
    if (not use_run_vs_run) and (not csv_single):
        raise ValueError("Missing input: provide --csv OR (--csv-a and --csv-b)")

    if use_run_vs_run:
        path_a = Path(csv_a)
        path_b = Path(csv_b)
        if not path_a.exists():
            raise FileNotFoundError(str(path_a))
        if not path_b.exists():
            raise FileNotFoundError(str(path_b))
        df_a = _load_csv(path_a)
        df_b = _load_csv(path_b)
        label_a = _safe_str(args.label_a) or "A"
        label_b = _safe_str(args.label_b) or "B"
        df_a = df_a.copy()
        df_b = df_b.copy()
        df_a["group"] = label_a
        df_b["group"] = label_b
        df = pd.concat([df_a, df_b], axis=0, ignore_index=True)
        group_col = "group"
        treat = label_a
        control = label_b
        stem = f"{Path(csv_a).stem}_vs_{Path(csv_b).stem}"
    else:
        csv_path = Path(csv_single)
        if not csv_path.exists():
            raise FileNotFoundError(str(csv_path))
        df = _load_csv(csv_path)
        group_col = "obs_source"
        treat = "sim"
        control = "base"
        stem = csv_path.stem

    outdir = Path(args.outdir) if args.outdir else Path("runs/play_analysis") / stem
    _ensure_outdir(outdir)

    # QC
    qc = qc_diagnostics_groups(df, group_col=group_col, treat=treat, control=control)
    qc_payload = dict(qc.__dict__)

    # Suggest bins based on sample size (helps avoid noisy stratified curves)
    bins_hint = _suggest_bins_for_min_per_bin(
        n_treat=qc.n_treat,
        n_control=qc.n_control,
        bins_requested=int(args.bins),
        min_per_bin=int(args.min_per_bin),
    )
    qc_payload["stratified_bins_hint"] = bins_hint

    bins_used = int(args.bins)
    if bool(args.auto_bins):
        bins_used = int(bins_hint["bins_suggested"])
    # Add legacy aliases for older tooling that expects sim/base keys.
    if group_col == "obs_source" and treat == "sim" and control == "base":
        qc_payload.setdefault("n_sim", qc.n_treat)
        qc_payload.setdefault("n_base", qc.n_control)
        qc_payload.setdefault("obstacles_hash_sim_unique", qc.obstacles_hash_treat_unique)
        qc_payload.setdefault("obstacles_hash_base_unique", qc.obstacles_hash_control_unique)
    (outdir / "qc_summary.json").write_text(
        json.dumps(qc_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Randomization QC (per group)
    rand_cols = [s.strip() for s in str(args.randomization_cols).split(",") if s.strip()]
    rand_qc = randomization_qc(df, group_col=group_col, cols=rand_cols)
    rand_qc.to_csv(outdir / "randomization_qc.csv", index=False)

    # Layer-0
    layer0 = layer0_bootstrap_groups(
        df,
        group_col=group_col,
        treat=treat,
        control=control,
        n_boot=int(args.n_boot),
        seed=int(args.seed),
    )
    layer0.to_csv(outdir / "layer0_summary.csv", index=False)

    # Layer-0+
    reg_info = {}
    try:
        reg_info = layer0plus_regression_groups(
            df,
            group_col=group_col,
            treat=treat,
            control=control,
            out_txt_path=outdir / "layer0plus_regression.txt",
        )
    except RuntimeError as e:
        # Write a helpful message but still succeed overall.
        (outdir / "layer0plus_regression.txt").write_text(
            f"Layer-0+ skipped: {e}\n", encoding="utf-8"
        )

    # Stratified
    strat_cols = [s.strip() for s in str(args.strat_cols).split(",") if s.strip()]
    strat = stratified_effects_groups(
        df,
        group_col=group_col,
        treat=treat,
        control=control,
        strat_cols=strat_cols,
        n_bins=int(bins_used),
        min_per_bin=int(args.min_per_bin),
        n_boot=int(args.n_boot_strat),
        seed=int(args.seed) + 12345,
    )
    strat.to_csv(outdir / "stratified_effects.csv", index=False)

    if not args.no_plots:
        plot_stratified_effects(strat, out_path=outdir / "plots_stratified.png")

    # Within-group attribution + sensitivity (recommended: randomized/treat group only)
    attrib_group = _safe_str(args.attrib_group) or str(treat)
    attrib_group_safe = _sanitize_filename_component(attrib_group) or "attrib"
    if attrib_group not in set(df[group_col].astype(str).unique().tolist()):
        print(f"[warn] attrib group '{attrib_group}' not found in column '{group_col}'; skipping attribution")
    else:
        df_attrib = df[df[group_col].astype(str) == str(attrib_group)].copy()
        outcome = _safe_str(args.attrib_outcome) or "collision"
        outcome_safe = _sanitize_filename_component(outcome) or "outcome"
        param_cols = [s.strip() for s in str(args.attrib_param_cols).split(",") if s.strip()]
        control_cols = [s.strip() for s in str(args.attrib_control_cols).split(",") if s.strip()]

        # Logit attribution
        try:
            coef = attribution_logit_within_group(
                df_attrib,
                outcome_col=outcome,
                param_cols=param_cols,
                control_cols=control_cols,
                use_log_params=bool(args.attrib_log_params),
                out_txt_path=outdir / f"attribution_logit_{attrib_group_safe}_{outcome_safe}.txt",
            )
            coef.to_csv(outdir / f"attribution_logit_{attrib_group_safe}_{outcome_safe}_coef.csv", index=False)
        except RuntimeError as e:
            (outdir / f"attribution_logit_{attrib_group_safe}_{outcome_safe}.txt").write_text(
                f"Attribution logit skipped: {e}\n",
                encoding="utf-8",
            )

        # Binned sensitivity tables (shape sanity check)
        if bool(args.bins_params):
            metrics = [
                "success_rate",
                "collision_rate",
                "time_to_goal_sec_success_mean",
                "path_efficiency_mean",
                "action_saturation_rate_mean",
            ]
            all_bins: List[pd.DataFrame] = []
            for pc in param_cols:
                tab = binned_sensitivity_table(
                    df_attrib,
                    param_col=pc,
                    n_bins=int(args.bins),
                    metrics=metrics,
                )
                if not tab.empty:
                    tab.insert(0, "group", attrib_group)
                    all_bins.append(tab)
            if all_bins:
                pd.concat(all_bins, axis=0, ignore_index=True).to_csv(
                    outdir / f"sensitivity_bins_{attrib_group_safe}.csv",
                    index=False,
                )

    # Console summary
    treat_rate = (
        float(df[df[group_col] == str(treat)]["success"].mean())
        if "success" in df.columns
        else float("nan")
    )
    control_rate = (
        float(df[df[group_col] == str(control)]["success"].mean())
        if "success" in df.columns
        else float("nan")
    )

    print("== QC ==")
    print(f"rows: total={qc.n_total} {treat}={qc.n_treat} {control}={qc.n_control}")
    print(
        "obstacles_hash: treat_unique={} control_unique={} intersection={} pair_ub={}".format(
            qc.obstacles_hash_treat_unique,
            qc.obstacles_hash_control_unique,
            qc.obstacles_hash_intersection,
            qc.obstacles_hash_pair_upper_bound,
        )
    )
    print("success rate: {}={} {}={}".format(treat, _pct(treat_rate), control, _pct(control_rate)))
    print(
        "stratified bins: requested={} used={} min_per_bin={} suggested_max={} suggested_bins={}{}".format(
            int(args.bins),
            int(bins_used),
            int(args.min_per_bin),
            int(bins_hint["max_bins"]),
            int(bins_hint["bins_suggested"]),
            " (auto)" if bool(args.auto_bins) else "",
        )
    )
    if int(bins_used) > int(bins_hint["max_bins"]) and int(bins_hint["max_bins"]) >= 2:
        print(
            "[warn] --bins may be too fine for --min-per-bin; consider: --bins {} (or use --auto-bins)".format(
                int(bins_hint["bins_suggested"])
            )
        )
    if reg_info:
        print(f"Layer-0+ success OR ({treat} vs {control}):", reg_info.get("success_or_treat"))
    print("Outputs written to:", outdir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
