#!/usr/bin/env python3
"""Build a deterministic USV scenario table (record/replay) without touching IsaacSim.

This script *replays* the current sampling logic used in:
- CaptureXYTask.get_goals(): target_xy ~ U([-goal_random_position, +goal_random_position]^2)
- CaptureXYTask.get_spawns():
  - spawn_xy = [r cos(theta), r sin(theta)], r ~ U([min_spawn_dist, max_spawn_dist])
  - obs_centers ~ U([target_xy-15, target_xy+15]^2), per-obstacle resampling if too close

It outputs:
- candidates file (optional): all unique candidates with metrics
- preselected scenario table: 300 scenarios by default (teacher input)
- (optional) selected scenario table: if you also want a geometry-only final selection

Design goals:
- Independent from IsaacSim/omni (pure torch/numpy)
- Matches the distribution of the current task code closely
- Provides geometry-only difficulty metrics for selection

NOTE:
- This does NOT run a policy; it uses continuous geometric metrics instead of success_rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# torch is optional: IsaacSim's python usually bundles it, but many venvs won't.
try:  # pragma: no cover
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# ----------------------------- Config helpers -----------------------------

def _load_task_cfg_value(task_yaml: str, dotted_key: str, default: float) -> float:
    """Best-effort load a numeric scalar from a task YAML.

    This intentionally stays dependency-free (no OmegaConf / PyYAML).
    It uses a simple regex to find the *last* key segment in the YAML text.

    Limitations:
    - If the same key name appears multiple times in the YAML, it may pick the first match.
    - If the value is an interpolation like ${...}, it will fall back to default.
    """

    import re

    key = dotted_key.split(".")[-1]
    try:
        with open(task_yaml, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return float(default)

    # YAML scalar number regex.
    num_re = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    pat = re.compile(rf"^\s*{re.escape(key)}\s*:\s*{num_re}\s*$", re.MULTILINE)
    m = pat.search(text)
    if not m:
        return float(default)
    try:
        return float(m.group(1))
    except Exception:
        return float(default)


# ----------------------------- Geometry core -----------------------------


def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point p to segment ab in 2D."""

    ab = b - a
    ap = p - a
    ab2 = float(np.dot(ab, ab))
    if ab2 <= 1e-12:
        return float(np.linalg.norm(ap))
    t = float(np.dot(ap, ab) / ab2)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _pairwise_min_distance(points: np.ndarray) -> float:
    """Min pairwise distance among points (N,2). Returns +inf if N<2."""

    n = int(points.shape[0])
    if n < 2:
        return float("inf")
    # O(N^2) is fine for N=16.
    dmin = float("inf")
    for i in range(n):
        di = points[i]
        for j in range(i + 1, n):
            dj = points[j]
            d = float(np.linalg.norm(di - dj))
            if d < dmin:
                dmin = d
    return dmin


# ----------------------------- Hashing / IO -----------------------------


def _obstacles_hash(obstacles_xy: np.ndarray, *, quant_m: float) -> str:
    """Hash obstacles in a stable way (order-sensitive), similar spirit to eval logging."""

    q = np.round(obstacles_xy / float(quant_m)).astype(np.int32)
    # Include shape and quant in the hashed payload.
    payload = q.tobytes() + (f"|shape={q.shape}|quant={quant_m}").encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _np_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


# ----------------------------- Scenario model -----------------------------


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    obstacles_xy: np.ndarray  # (big,2) float32
    spawn_root_pos: np.ndarray  # (3,) float32
    spawn_root_rot: np.ndarray  # (4,) float32
    target_pos: np.ndarray  # (3,) float32
    target_rot: np.ndarray  # (4,) float32
    init_root_vel_xy: np.ndarray  # (2,) float32
    obstacles_hash: str

    # Geometry-only difficulty metrics (all in meters):
    d_spawn: float
    d_goal: float
    d_corr: float
    d_oo: float
    density_score: float


# ----------------------------- Sampling logic -----------------------------


def _sample_target_xy_np(*, rng: np.random.Generator, goal_random_position: float) -> np.ndarray:
    # Matches: torch.rand((2,))*goal_random_position*2 - goal_random_position
    return rng.uniform(
        low=-float(goal_random_position),
        high=float(goal_random_position),
        size=(2,),
    ).astype(np.float32)


def _sample_spawn_xy_np(
    *, rng: np.random.Generator, min_spawn_dist: float, max_spawn_dist: float
) -> np.ndarray:
    r = float(rng.uniform(low=float(min_spawn_dist), high=float(max_spawn_dist)))
    theta = float(rng.uniform(low=0.0, high=2.0 * math.pi))
    return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=np.float32)


def _sample_spawn_yaw_quat_np(*, rng: np.random.Generator) -> np.ndarray:
    yaw = float(rng.uniform(low=0.0, high=math.pi))
    return np.array([math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)], dtype=np.float32)


def _sample_obstacles_xy_np(
    *,
    rng: np.random.Generator,
    big: int,
    target_xy: np.ndarray,  # (2,)
    spawn_xy: np.ndarray,  # (2,)
    box_half_range: float,
    min_dist_safe: float,
    max_iterations: int,
) -> np.ndarray:
    target_xy = np.asarray(target_xy, dtype=np.float32).reshape(2)
    spawn_xy = np.asarray(spawn_xy, dtype=np.float32).reshape(2)

    min_coords = target_xy - float(box_half_range)
    max_coords = target_xy + float(box_half_range)

    obs = rng.uniform(low=min_coords, high=max_coords, size=(int(big), 2)).astype(np.float32)

    for _ in range(int(max_iterations)):
        d_start = np.linalg.norm(obs - spawn_xy[None, :], axis=1)
        d_target = np.linalg.norm(obs - target_xy[None, :], axis=1)
        mask_invalid = (d_start < float(min_dist_safe)) | (d_target < float(min_dist_safe))
        if not bool(mask_invalid.any()):
            break
        obs[mask_invalid] = rng.uniform(
            low=min_coords, high=max_coords, size=(int(mask_invalid.sum()), 2)
        ).astype(np.float32)

    d_start = np.linalg.norm(obs - spawn_xy[None, :], axis=1)
    d_target = np.linalg.norm(obs - target_xy[None, :], axis=1)
    mask_invalid = (d_start < float(min_dist_safe)) | (d_target < float(min_dist_safe))
    if bool(mask_invalid.any()):
        obs[mask_invalid] = np.array([999.0, 999.0], dtype=np.float32)

    return obs


# ----------------------------- Metrics -----------------------------


def _compute_metrics(
    *,
    obstacles_xy: np.ndarray,
    spawn_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacle_radius: float,
    limbo_threshold: float = 100.0,
) -> Dict[str, float]:
    """Compute geometry-only difficulty metrics.

    All distances are *surface distances* (center distance minus obstacle radius),
    except d_oo which is obstacle-to-obstacle surface gap (minus 2R).
    """

    obs = np.asarray(obstacles_xy, dtype=np.float32)
    # Drop limbo obstacles (999,999).
    valid = np.all(np.abs(obs) < float(limbo_threshold), axis=1)
    obs_v = obs[valid]

    s = np.asarray(spawn_xy, dtype=np.float32).reshape(2)
    g = np.asarray(goal_xy, dtype=np.float32).reshape(2)

    if obs_v.shape[0] == 0:
        d_spawn = float("inf")
        d_goal = float("inf")
        d_corr = float("inf")
        d_oo = float("inf")
    else:
        d_spawn = float(np.min(np.linalg.norm(obs_v - s[None, :], axis=1)) - obstacle_radius)
        d_goal = float(np.min(np.linalg.norm(obs_v - g[None, :], axis=1)) - obstacle_radius)
        # Corridor clearance (min dist to segment minus radius)
        d_corr = float(
            min(_point_to_segment_distance(p, s, g) for p in obs_v) - obstacle_radius
        )
        d_oo = float(_pairwise_min_distance(obs_v) - 2.0 * obstacle_radius)

    density_score = float(min(d_spawn, d_goal, d_corr))
    return {
        "d_spawn": float(d_spawn),
        "d_goal": float(d_goal),
        "d_corr": float(d_corr),
        "d_oo": float(d_oo),
        "density_score": float(density_score),
    }


# ----------------------------- Build pipeline -----------------------------


def build_candidates(
    *,
    n_candidates: int,
    big: int,
    seed: int,
    device: str,
    goal_random_position: float,
    min_spawn_dist: float,
    max_spawn_dist: float,
    box_half_range: float,
    min_dist_safe: float,
    obstacle_radius: float,
    quant_m: float,
    max_iterations: int,
    init_vel_mode: str,
) -> List[Scenario]:
    """Generate unique candidate scenarios (dedup by obstacles_hash)."""

    # Primary RNG: numpy (no dependency on torch).
    rng = np.random.default_rng(int(seed))

    scenarios: List[Scenario] = []
    seen_hashes: set[str] = set()

    # We may need to oversample due to dedup; use a conservative loop cap.
    attempts = 0
    max_attempts = max(int(n_candidates) * 50, 1000)

    while len(scenarios) < int(n_candidates) and attempts < max_attempts:
        attempts += 1

        # NOTE: `device` is kept in the API to align with other scripts, but numpy sampling
        # runs on CPU by design (fast enough for big<=16, n<=500).
        _ = device

        target_xy = _sample_target_xy_np(
            rng=rng, goal_random_position=goal_random_position
        )  # (2,)
        spawn_xy = _sample_spawn_xy_np(
            rng=rng, min_spawn_dist=min_spawn_dist, max_spawn_dist=max_spawn_dist
        )  # (2,)
        spawn_quat = _sample_spawn_yaw_quat_np(rng=rng)  # (4,)
        obstacles_xy_np = _sample_obstacles_xy_np(
            rng=rng,
            big=big,
            target_xy=target_xy,
            spawn_xy=spawn_xy,
            box_half_range=box_half_range,
            min_dist_safe=min_dist_safe,
            max_iterations=max_iterations,
        )  # (big,2)

        # init_root_vel_xy: fixed zero by your chosen route.
        if init_vel_mode == "zero":
            init_vel_xy = np.array([0.0, 0.0], dtype=np.float32)
        elif init_vel_mode == "rand":
            init_vel_xy = rng.uniform(low=-1.5, high=1.5, size=(2,)).astype(np.float32)
        else:
            raise ValueError(f"Unknown init_vel_mode: {init_vel_mode}")
        h = _obstacles_hash(obstacles_xy_np, quant_m=quant_m)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        spawn_pos = np.array([float(spawn_xy[0]), float(spawn_xy[1]), 0.0], dtype=np.float32)
        spawn_rot = np.asarray(spawn_quat, dtype=np.float32)
        target_pos = np.array([float(target_xy[0]), float(target_xy[1]), 0.0], dtype=np.float32)
        target_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # task keeps target_orientation unchanged

        metrics = _compute_metrics(
            obstacles_xy=obstacles_xy_np,
            spawn_xy=spawn_pos[:2],
            goal_xy=target_pos[:2],
            obstacle_radius=obstacle_radius,
        )

        scenario = Scenario(
            scenario_id=len(scenarios),
            obstacles_xy=_np_float32(obstacles_xy_np),
            spawn_root_pos=_np_float32(spawn_pos),
            spawn_root_rot=_np_float32(spawn_rot),
            target_pos=_np_float32(target_pos),
            target_rot=_np_float32(target_rot),
            init_root_vel_xy=_np_float32(init_vel_xy),
            obstacles_hash=h,
            d_spawn=float(metrics["d_spawn"]),
            d_goal=float(metrics["d_goal"]),
            d_corr=float(metrics["d_corr"]),
            d_oo=float(metrics["d_oo"]),
            density_score=float(metrics["density_score"]),
        )
        scenarios.append(scenario)

    if len(scenarios) < int(n_candidates):
        raise RuntimeError(
            f"Could only generate {len(scenarios)}/{n_candidates} unique scenarios after {attempts} attempts. "
            "Try increasing max_attempts or loosening dedup/constraints."
        )

    return scenarios


def preselect_scenarios(
    *,
    candidates: List[Scenario],
    n_preselect: int,
    ratio_sparse_mid_dense: Tuple[int, int, int],
    seed: int,
) -> Tuple[List[Scenario], np.ndarray, Dict[str, object]]:
    """Preselect scenarios using geometry-only metrics only.

    This stage is intentionally minimal: dedup has already happened, and here we only do
    1:3:1 bin sampling based on density_score quantiles.

    Returns:
    - preselected scenarios (with scenario_id re-assigned to 0..N-1)
    - source_candidate_id array mapping back to the original candidates list
    - summary dict
    """

    rng = np.random.default_rng(int(seed))

    scores = np.array([s.density_score for s in candidates], dtype=np.float32)
    q20 = float(np.quantile(scores, 0.2))
    q80 = float(np.quantile(scores, 0.8))

    sparse = [s for s in candidates if s.density_score >= q80]
    dense = [s for s in candidates if s.density_score <= q20]
    mid = [s for s in candidates if (s.density_score > q20 and s.density_score < q80)]

    # Target preselect counts by ratio.
    r0, r1, r2 = ratio_sparse_mid_dense
    rsum = max(r0 + r1 + r2, 1)

    def _take(pool: List[Scenario], k: int) -> List[Scenario]:
        if k <= 0:
            return []
        if len(pool) <= k:
            return list(pool)
        idx = rng.choice(len(pool), size=k, replace=False)
        return [pool[int(i)] for i in idx]

    k_sparse = int(round(n_preselect * (r0 / rsum)))
    k_mid = int(round(n_preselect * (r1 / rsum)))
    k_dense = int(round(n_preselect * (r2 / rsum)))

    pre = _take(sparse, k_sparse) + _take(mid, k_mid) + _take(dense, k_dense)
    # If rounding left us short, fill from mid then the rest.
    if len(pre) < n_preselect:
        remaining = [s for s in candidates if s not in pre]
        fill = _take(remaining, n_preselect - len(pre))
        pre += fill

    # Map preselected scenarios back to candidates.
    # (We use obstacles_hash as the stable join key.)
    hash_to_candidate_id: Dict[str, int] = {c.obstacles_hash: int(i) for i, c in enumerate(candidates)}
    source_candidate_id = np.array([hash_to_candidate_id.get(s.obstacles_hash, -1) for s in pre], dtype=np.int64)

    # Reassign scenario_id to be dense 0..n_preselect-1 for table usability.
    pre_final: List[Scenario] = []
    for new_id, s in enumerate(list(pre)):
        pre_final.append(
            Scenario(
                scenario_id=int(new_id),
                obstacles_xy=s.obstacles_xy,
                spawn_root_pos=s.spawn_root_pos,
                spawn_root_rot=s.spawn_root_rot,
                target_pos=s.target_pos,
                target_rot=s.target_rot,
                init_root_vel_xy=s.init_root_vel_xy,
                obstacles_hash=s.obstacles_hash,
                d_spawn=s.d_spawn,
                d_goal=s.d_goal,
                d_corr=s.d_corr,
                d_oo=s.d_oo,
                density_score=s.density_score,
            )
        )

    summary: Dict[str, object] = {
        "n_candidates": int(len(candidates)),
        "n_preselect": int(len(pre)),
        "density_score_quantiles": {"q20": q20, "q80": q80},
        "bin_counts": {"sparse": len(sparse), "mid": len(mid), "dense": len(dense)},
        "ratio_sparse_mid_dense": list(ratio_sparse_mid_dense),
    }
    return pre_final, source_candidate_id, summary


def select_scenarios_from_pre(
    *,
    preselected: List[Scenario],
    n_selected: int,
    seed: int,
) -> Tuple[List[Scenario], Dict[str, object]]:
    """Optional geometry-only final selection.

    This is NOT used in the teacher-driven pipeline by default.
    It keeps a simple preference toward the mid-range of density_score.
    """

    if n_selected <= 0:
        raise ValueError("n_selected must be > 0")
    if n_selected > len(preselected):
        raise ValueError(f"n_selected ({n_selected}) must be <= preselected ({len(preselected)})")

    rng = np.random.default_rng(int(seed))

    pre_scores = np.array([s.density_score for s in preselected], dtype=np.float32)
    q25 = float(np.quantile(pre_scores, 0.25))
    q75 = float(np.quantile(pre_scores, 0.75))
    preferred = [s for s in preselected if q25 <= s.density_score <= q75]
    nonpreferred = [s for s in preselected if s not in preferred]

    def _take(pool: List[Scenario], k: int) -> List[Scenario]:
        if k <= 0:
            return []
        if len(pool) <= k:
            return list(pool)
        idx = rng.choice(len(pool), size=k, replace=False)
        return [pool[int(i)] for i in idx]

    selected = _take(preferred, min(n_selected, len(preferred)))
    if len(selected) < n_selected:
        selected += _take(nonpreferred, n_selected - len(selected))

    selected_final: List[Scenario] = []
    for new_id, s in enumerate(list(selected)):
        selected_final.append(
            Scenario(
                scenario_id=int(new_id),
                obstacles_xy=s.obstacles_xy,
                spawn_root_pos=s.spawn_root_pos,
                spawn_root_rot=s.spawn_root_rot,
                target_pos=s.target_pos,
                target_rot=s.target_rot,
                init_root_vel_xy=s.init_root_vel_xy,
                obstacles_hash=s.obstacles_hash,
                d_spawn=s.d_spawn,
                d_goal=s.d_goal,
                d_corr=s.d_corr,
                d_oo=s.d_oo,
                density_score=s.density_score,
            )
        )

    summary = {
        "n_preselected": int(len(preselected)),
        "n_selected": int(len(selected_final)),
        "pre_density_score_quantiles": {"q25": q25, "q75": q75},
    }
    return selected_final, summary


def _save_npz(
    path: str,
    scenarios: List[Scenario],
    meta: Dict[str, object],
    *,
    extra_arrays: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = dict(
        scenario_id=np.array([s.scenario_id for s in scenarios], dtype=np.int64),
        obstacles_xy=np.stack([s.obstacles_xy for s in scenarios]).astype(np.float32),
        spawn_root_pos=np.stack([s.spawn_root_pos for s in scenarios]).astype(np.float32),
        spawn_root_rot=np.stack([s.spawn_root_rot for s in scenarios]).astype(np.float32),
        target_pos=np.stack([s.target_pos for s in scenarios]).astype(np.float32),
        target_rot=np.stack([s.target_rot for s in scenarios]).astype(np.float32),
        init_root_vel_xy=np.stack([s.init_root_vel_xy for s in scenarios]).astype(np.float32),
        obstacles_hash=np.array([s.obstacles_hash for s in scenarios], dtype=object),
        d_spawn=np.array([s.d_spawn for s in scenarios], dtype=np.float32),
        d_goal=np.array([s.d_goal for s in scenarios], dtype=np.float32),
        d_corr=np.array([s.d_corr for s in scenarios], dtype=np.float32),
        d_oo=np.array([s.d_oo for s in scenarios], dtype=np.float32),
        density_score=np.array([s.density_score for s in scenarios], dtype=np.float32),
        meta=json.dumps(meta, ensure_ascii=False),
    )
    if extra_arrays:
        for k, v in extra_arrays.items():
            payload[str(k)] = v

    np.savez_compressed(
        path,
        **payload,
    )


def _save_meta_json(path: str, meta: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--task-yaml", type=str, default="", help="Optional task YAML path to read scalar params from")
    p.add_argument("--out-dir", type=str, default="runs/scenarios", help="Output directory")

    p.add_argument("--n-candidates", type=int, default=500)
    p.add_argument("--n-preselect", type=int, default=300, help="Teacher input size")
    p.add_argument(
        "--n-selected",
        type=int,
        default=0,
        help="Optional geometry-only final selection size (0 disables; teacher will do it)",
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")

    # Task-ish parameters (defaults match current code reasonably).
    p.add_argument("--big", type=int, default=16)
    p.add_argument("--goal-random-position", type=float, default=None)
    p.add_argument("--min-spawn-dist", type=float, default=None)
    p.add_argument("--max-spawn-dist", type=float, default=None)

    p.add_argument("--box-half-range", type=float, default=15.0)
    p.add_argument("--min-dist-safe", type=float, default=2.0)
    p.add_argument("--max-iterations", type=int, default=20)

    p.add_argument("--obstacle-radius", type=float, default=0.5)
    p.add_argument("--quant-m", type=float, default=0.05)

    p.add_argument("--init-vel-mode", type=str, default="zero", choices=["zero", "rand"], help="zero = fixed (0,0); rand = mimic env reset")

    # Selection: 1:3:1 => sparse:mid:dense
    p.add_argument("--ratio", type=str, default="1:3:1", help="sparse:mid:dense ratio")

    p.add_argument("--save-candidates", action="store_true", help="Also save candidates file")

    args = p.parse_args()

    # Load from task yaml if provided.
    goal_random_position = args.goal_random_position
    min_spawn_dist = args.min_spawn_dist
    max_spawn_dist = args.max_spawn_dist

    if args.task_yaml:
        if goal_random_position is None:
            goal_random_position = _load_task_cfg_value(
                args.task_yaml, "env.task_parameters.goal_random_position", default=5.0
            )
        if min_spawn_dist is None:
            min_spawn_dist = _load_task_cfg_value(
                args.task_yaml, "env.task_parameters.min_spawn_dist", default=0.5
            )
        if max_spawn_dist is None:
            max_spawn_dist = _load_task_cfg_value(
                args.task_yaml, "env.task_parameters.max_spawn_dist", default=5.0
            )

    # Final defaults if still None.
    if goal_random_position is None:
        goal_random_position = 5.0
    if min_spawn_dist is None:
        min_spawn_dist = 0.5
    if max_spawn_dist is None:
        max_spawn_dist = 5.0

    # Parse ratio.
    try:
        r_parts = [int(x) for x in str(args.ratio).split(":")]
        if len(r_parts) != 3:
            raise ValueError
        ratio = (r_parts[0], r_parts[1], r_parts[2])
    except Exception:
        raise ValueError(f"Invalid --ratio '{args.ratio}', expected like 1:3:1")

    meta = {
        "script": os.path.relpath(__file__),
        "seed": int(args.seed),
        "sampling": {
            "big": int(args.big),
            "goal_random_position": float(goal_random_position),
            "min_spawn_dist": float(min_spawn_dist),
            "max_spawn_dist": float(max_spawn_dist),
            "box_half_range": float(args.box_half_range),
            "min_dist_safe": float(args.min_dist_safe),
            "max_iterations": int(args.max_iterations),
            "obstacle_radius": float(args.obstacle_radius),
            "quant_m": float(args.quant_m),
            "init_vel_mode": str(args.init_vel_mode),
        },
        "selection": {
            "n_candidates": int(args.n_candidates),
            "n_preselect": int(args.n_preselect),
            "n_selected": int(args.n_selected),
            "ratio_sparse_mid_dense": list(ratio),
        },
        "notes": {
            "frame": "local (env origin)",
            "z_in_positions": "0.0 (to be set by sim reset if needed)",
            "target_rot": "identity (task returns unchanged targets_orientation)",
        },
    }

    candidates = build_candidates(
        n_candidates=int(args.n_candidates),
        big=int(args.big),
        seed=int(args.seed),
        device=str(args.device),
        goal_random_position=float(goal_random_position),
        min_spawn_dist=float(min_spawn_dist),
        max_spawn_dist=float(max_spawn_dist),
        box_half_range=float(args.box_half_range),
        min_dist_safe=float(args.min_dist_safe),
        obstacle_radius=float(args.obstacle_radius),
        quant_m=float(args.quant_m),
        max_iterations=int(args.max_iterations),
        init_vel_mode=str(args.init_vel_mode),
    )

    preselected, pre_source_ids, pre_summary = preselect_scenarios(
        candidates=candidates,
        n_preselect=int(args.n_preselect),
        ratio_sparse_mid_dense=ratio,
        seed=int(args.seed) + 1,
    )

    meta_out = dict(meta)
    meta_out["summary"] = {"preselect": pre_summary}

    selected: List[Scenario] = []
    if int(args.n_selected) > 0:
        selected, sel_summary = select_scenarios_from_pre(
            preselected=preselected,
            n_selected=int(args.n_selected),
            seed=int(args.seed) + 2,
        )
        meta_out["summary"]["selected"] = sel_summary

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Save.
    if args.save_candidates:
        _save_npz(os.path.join(out_dir, "candidates_500_unique.npz"), candidates, meta_out)

    _save_npz(
        os.path.join(out_dir, f"preselect_{len(preselected)}.npz"),
        preselected,
        meta_out,
        extra_arrays={"source_candidate_id": pre_source_ids},
    )

    if int(args.n_selected) > 0:
        _save_npz(os.path.join(out_dir, f"selected_{len(selected)}.npz"), selected, meta_out)

    _save_meta_json(os.path.join(out_dir, "scenario_table.meta.json"), meta_out)

    print("[scenario_table] done")
    print(f"  candidates: {len(candidates)}")
    print(f"  preselect:  {len(preselected)}")
    if int(args.n_selected) > 0:
        print(f"  selected:   {len(selected)}")
    print(f"  out_dir:    {out_dir}")
    print("  files:")
    if args.save_candidates:
        print("    - candidates_500_unique.npz")
    print(f"    - preselect_{len(preselected)}.npz")
    if int(args.n_selected) > 0:
        print(f"    - selected_{len(selected)}.npz")
    print("    - scenario_table.meta.json")


if __name__ == "__main__":
    main()
