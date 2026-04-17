"""Trajectory plotting utilities for play scripts.

Design goals
- Headless-friendly (matplotlib Agg backend).
- No dependency on Isaac Sim / omni.*. Accepts pure numpy inputs.
- Stable, comparable figures via explicit axis limits.
- Minimal, fixed info-box fields (per user spec):
  1) result + reason
  2) steps + T
  3) return_scaled
  4) path efficiency

This module is intended to be imported from multiple play scripts.
"""

from __future__ import annotations

import os
from collections.abc import Mapping as _Mapping
from collections.abc import Sequence as _Sequence
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

Axis4 = Tuple[float, float, float, float]  # xmin, xmax, ymin, ymax


@dataclass(frozen=True)
class TrajPlotStyle:
    fig_size: Tuple[float, float] = (6.5, 6.5)
    dpi: int = 160
    traj_color: str = "#1f77b4"  # matplotlib default blue
    traj_lw: float = 1.6
    start_color: str = "#2ca02c"  # green
    end_color: str = "#d62728"  # red
    goal_color: str = "#ff7f0e"  # orange
    obstacle_color: str = "#000000"
    obstacle_size: float = 10.0
    obstacle_body_color: str = "#666666"  # physical obstacle disk
    obstacle_body_edge_color: str = "#444444"
    obstacle_body_alpha: float = 0.35
    obstacle_body_lw: float = 0.6

    usv_start_face_color: str = "#2ca02c"  # green
    usv_end_face_color: str = "#d62728"  # red
    usv_edge_color: str = "#222222"
    usv_alpha: float = 0.22
    usv_lw: float = 0.9
    usv_nose_frac: float = 0.25  # fraction of half-length for the shoulder
    d0_edge_color: str = "#ff0000"
    d0_alpha: float = 0.10
    d0_lw: float = 0.8
    title_fontsize: int = 11
    textbox_fontsize: int = 9


def _maybe_xy(seq: Optional[Sequence[float]]) -> Optional[Tuple[float, float]]:
    if seq is None:
        return None
    try:
        x, y = float(seq[0]), float(seq[1])
    except Exception:
        return None
    if not np.isfinite([x, y]).all():
        return None
    return (x, y)


def _maybe_yaw(yaw_like: Any) -> Optional[float]:
    if yaw_like is None:
        return None
    try:
        yaw = float(yaw_like)
    except Exception:
        return None
    if not np.isfinite(yaw):
        return None
    return yaw


def _usv_footprint_local_points(*, length_m: float, width_m: float, nose_frac: float) -> np.ndarray:
    """Return a pointy USV footprint polygon in local frame.

    Local frame:
    - origin at geometric center
    - +x points to bow (nose)
    """

    L = float(length_m)
    W = float(width_m)
    nf = float(nose_frac)
    if not np.isfinite([L, W, nf]).all() or L <= 0 or W <= 0:
        raise ValueError(f"Invalid USV size: L={L}, W={W}")
    if nf <= 0:
        nf = 0.25
    nf = min(max(nf, 0.05), 0.95)

    a = 0.5 * L
    b = 0.5 * W
    shoulder_x = nf * a

    # Bow is a point, stern is flat.
    pts = np.array(
        [
            [a, 0.0],
            [shoulder_x, b],
            [-a, b],
            [-a, -b],
            [shoulder_x, -b],
        ],
        dtype=np.float64,
    )
    return pts


def _transform_points_xy_yaw(pts_local: np.ndarray, *, center_xy: Tuple[float, float], yaw_rad: float) -> np.ndarray:
    x0, y0 = float(center_xy[0]), float(center_xy[1])
    yaw = float(yaw_rad)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    pts_w = (pts_local @ R.T) + np.array([x0, y0], dtype=np.float64)
    return pts_w


def _is_finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def parse_axis(axis_like: Any) -> Optional[Axis4]:
    """Parse an explicit axis spec into (xmin, xmax, ymin, ymax).

    Accepts:
    - sequence of 4 numbers
    - string "xmin,xmax,ymin,ymax" (spaces ok)
    """

    if axis_like is None:
        return None

    if isinstance(axis_like, str):
        s = axis_like.strip()
        if not s:
            return None
        parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
        if len(parts) != 4:
            return None
        try:
            vals = [float(p) for p in parts]
        except Exception:
            return None
        return _validate_axis_tuple(tuple(vals))

    # Accept any non-string sequence (e.g., OmegaConf ListConfig).
    if isinstance(axis_like, np.ndarray):
        try:
            vals = [float(x) for x in axis_like.reshape(-1).tolist()]
        except Exception:
            return None
        if len(vals) != 4:
            return None
        return _validate_axis_tuple(tuple(vals))

    if isinstance(axis_like, _Mapping):
        return None

    if isinstance(axis_like, _Sequence) and not isinstance(axis_like, (str, bytes, bytearray)):
        try:
            vals = [float(x) for x in list(axis_like)]
        except Exception:
            return None
        if len(vals) != 4:
            return None
        return _validate_axis_tuple(tuple(vals))

    return None


def _validate_axis_tuple(axis: Axis4) -> Optional[Axis4]:
    try:
        xmin, xmax, ymin, ymax = (float(axis[0]), float(axis[1]), float(axis[2]), float(axis[3]))
    except Exception:
        return None
    if not (np.isfinite([xmin, xmax, ymin, ymax]).all()):
        return None
    if xmax <= xmin or ymax <= ymin:
        return None
    return (xmin, xmax, ymin, ymax)


def axis_from_env(var_name: str = "LOOPZ_PLAY_TRAJ_AXIS") -> Optional[Axis4]:
    """Read axis from environment variable."""

    return parse_axis(os.getenv(var_name, ""))


def axis_from_kill_dist(kill_dist: Any) -> Optional[Axis4]:
    """Fallback axis from a scalar kill_dist: [-k, k, -k, k]."""

    try:
        k = float(kill_dist)
    except Exception:
        return None
    if not np.isfinite(k) or k <= 0:
        return None
    return (-k, k, -k, k)


def ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def default_traj_out_dir(*, repo_root: str, run_id: str) -> str:
    """Return runs/play_traj/<run_id> under repo_root."""

    return os.path.join(str(repo_root), "runs", "play_traj", str(run_id))


def _fmt_num(value: Any, *, fmt: str) -> str:
    """Format scalar with missing-value rules.

    - None / missing: "--"
    - non-finite (NaN/Inf): "nan"
    """

    if value is None:
        return "--"
    try:
        v = float(value)
    except Exception:
        return "--"
    if not np.isfinite(v):
        return "nan"
    try:
        return format(v, fmt)
    except Exception:
        return str(v)


def _fmt_int(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return str(int(value))
    except Exception:
        return "--"


def build_info_box_text(
    *,
    success: Any,
    reason: Any,
    steps: Any,
    control_dt: Any,
    return_scaled: Any,
    path_efficiency: Any,
) -> str:
    """Return 4-line info box string with fixed formatting."""

    succ = None
    try:
        succ = int(success)
    except Exception:
        succ = None

    if succ is None:
        result = "--"
    else:
        result = "SUCCESS" if succ == 1 else "FAIL"

    reason_s = "--" if reason is None else str(reason)

    steps_s = _fmt_int(steps)

    # T missing rule: show "--" when control_dt missing/invalid.
    T_s = "--"
    try:
        dt = float(control_dt)
        st = int(steps) if steps is not None else None
        if st is not None and np.isfinite(dt) and dt > 0:
            T_s = format(float(st) * float(dt), ".2f")
    except Exception:
        T_s = "--"

    ret_s = _fmt_num(return_scaled, fmt=".3f")
    eff_s = _fmt_num(path_efficiency, fmt=".3f")

    lines = [
        f"result={result}  reason={reason_s}",
        f"steps={steps_s}  T={T_s}s",
        f"return={ret_s}",
        f"eff={eff_s}",
    ]
    return "\n".join(lines)


def save_episode_trajectory_png(
    *,
    out_path: str,
    axis: Axis4,
    traj_xy: np.ndarray,
    start_xy: Optional[Sequence[float]] = None,
    goal_xy: Optional[Sequence[float]] = None,
    obstacles_xy: Optional[np.ndarray] = None,
    obstacle_radius_m: Optional[float] = None,
    usv_length_m: Optional[float] = None,
    usv_width_m: Optional[float] = None,
    start_yaw_rad: Optional[float] = None,
    end_yaw_rad: Optional[float] = None,
    draw_usv_footprint: bool = False,
    d0_m: Optional[float] = None,
    title: str = "",
    success: Any = None,
    reason: Any = None,
    steps: Any = None,
    control_dt: Any = None,
    return_scaled: Any = None,
    path_efficiency: Any = None,
    style: Optional[TrajPlotStyle] = None,
) -> bool:
    """Save a single episode trajectory figure.

    Returns True on success, False if plotting is skipped/fails.
    """

    axis_v = parse_axis(axis)
    if axis_v is None:
        raise ValueError(f"Invalid axis spec: {axis}")

    style = style or TrajPlotStyle()

    # Lazy import matplotlib + force Agg backend.
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from matplotlib.patches import Polygon
    except Exception as exc:
        print(f"[traj_plot] matplotlib unavailable, skip plot (exc={type(exc).__name__}: {exc})")
        return False

    try:
        ensure_dir(os.path.dirname(out_path) or ".")

        xy = np.asarray(traj_xy, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] < 2 or xy.shape[0] < 1:
            print("[traj_plot] empty traj_xy, skip plot")
            return False

        x = xy[:, 0]
        y = xy[:, 1]

        fig, ax = plt.subplots(1, 1, figsize=style.fig_size, dpi=int(style.dpi))

        # USV footprint (optional): draw behind markers, above obstacle disks.
        if bool(draw_usv_footprint) and usv_length_m is not None and usv_width_m is not None:
            try:
                L = float(usv_length_m)
                W = float(usv_width_m)
                start_xy_v = _maybe_xy(start_xy) or (float(x[0]), float(y[0]))
                end_xy_v = (float(x[-1]), float(y[-1]))
                yaw0 = _maybe_yaw(start_yaw_rad)
                yaw1 = _maybe_yaw(end_yaw_rad)

                pts_local = _usv_footprint_local_points(length_m=L, width_m=W, nose_frac=float(style.usv_nose_frac))

                if yaw0 is not None:
                    pts0 = _transform_points_xy_yaw(pts_local, center_xy=start_xy_v, yaw_rad=yaw0)
                    ax.add_patch(
                        Polygon(
                            pts0,
                            closed=True,
                            facecolor=style.usv_start_face_color,
                            edgecolor=style.usv_edge_color,
                            alpha=float(style.usv_alpha),
                            lw=float(style.usv_lw),
                            zorder=2.4,
                        )
                    )
                if yaw1 is not None:
                    pts1 = _transform_points_xy_yaw(pts_local, center_xy=end_xy_v, yaw_rad=yaw1)
                    ax.add_patch(
                        Polygon(
                            pts1,
                            closed=True,
                            facecolor=style.usv_end_face_color,
                            edgecolor=style.usv_edge_color,
                            alpha=float(style.usv_alpha),
                            lw=float(style.usv_lw),
                            zorder=2.5,
                        )
                    )
            except Exception:
                # footprint is optional; never fail the whole plot
                pass

        # Obstacles
        if obstacles_xy is not None:
            obs = np.asarray(obstacles_xy, dtype=np.float64)
            if obs.ndim == 2 and obs.shape[1] >= 2 and obs.shape[0] > 0:
                # 1) Optional safety/collision radius (d0)
                if d0_m is not None and _is_finite(d0_m) and float(d0_m) > 0:
                    r = float(d0_m)
                    for i in range(obs.shape[0]):
                        cx, cy = float(obs[i, 0]), float(obs[i, 1])
                        if not np.isfinite([cx, cy]).all():
                            continue
                        ax.add_patch(
                            Circle(
                                (cx, cy),
                                radius=r,
                                edgecolor=style.d0_edge_color,
                                facecolor=style.d0_edge_color,
                                alpha=float(style.d0_alpha),
                                lw=float(style.d0_lw),
                                zorder=1,
                            )
                        )

                # 2) Physical obstacle disk (radius in meters)
                if obstacle_radius_m is not None and _is_finite(obstacle_radius_m) and float(obstacle_radius_m) > 0:
                    r_obs = float(obstacle_radius_m)
                    for i in range(obs.shape[0]):
                        cx, cy = float(obs[i, 0]), float(obs[i, 1])
                        if not np.isfinite([cx, cy]).all():
                            continue
                        ax.add_patch(
                            Circle(
                                (cx, cy),
                                radius=r_obs,
                                edgecolor=style.obstacle_body_edge_color,
                                facecolor=style.obstacle_body_color,
                                alpha=float(style.obstacle_body_alpha),
                                lw=float(style.obstacle_body_lw),
                                zorder=1.8,
                            )
                        )

                # 3) Obstacle centers
                ax.scatter(obs[:, 0], obs[:, 1], s=float(style.obstacle_size), c=style.obstacle_color, zorder=2.2)

        # Trajectory
        ax.plot(x, y, color=style.traj_color, lw=float(style.traj_lw), zorder=3)

        # Start/end markers
        ax.scatter([float(x[0])], [float(y[0])], s=35, c=style.start_color, marker="o", zorder=4, label="start")
        ax.scatter([float(x[-1])], [float(y[-1])], s=35, c=style.end_color, marker="o", zorder=4, label="end")

        # Goal marker
        if goal_xy is not None:
            try:
                gx, gy = float(goal_xy[0]), float(goal_xy[1])
                if np.isfinite([gx, gy]).all():
                    ax.scatter([gx], [gy], s=55, c=style.goal_color, marker="*", zorder=4, label="goal")
            except Exception:
                pass

        # Enforce axis limits for comparability
        xmin, xmax, ymin, ymax = axis_v
        ax.set_xlim(float(xmin), float(xmax))
        ax.set_ylim(float(ymin), float(ymax))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        # Title
        if title:
            ax.set_title(str(title), fontsize=int(style.title_fontsize))

        # Fixed 4-line info box
        info = build_info_box_text(
            success=success,
            reason=reason,
            steps=steps,
            control_dt=control_dt,
            return_scaled=return_scaled,
            path_efficiency=path_efficiency,
        )
        ax.text(
            0.98,
            0.98,
            info,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=int(style.textbox_fontsize),
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.6"),
            zorder=10,
        )

        # Tiny legend: keep minimal; can be disabled later if you prefer.
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return True

    except Exception as exc:
        print(f"[traj_plot] failed to save '{out_path}' (exc={type(exc).__name__}: {exc})")
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception:
            pass
        return False
