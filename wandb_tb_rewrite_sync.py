#!/usr/bin/env python3
"""Rewrite TensorBoard scalar tags and stream to Weights & Biases.

Use case:
- Training (Isaac Sim) writes TensorBoard event files (tfevents).
- A separate Python environment (e.g., conda env `wandb_sync`) reads the event files
  and logs them to W&B in (near) real time.

This script focuses on scalars and provides:
- Prefix filtering (Episode/, Loss/, rewards/, shaped_rewards/, episode_lengths/, Diagnostics/, Policy/ by default)
- Optional strip-prefix so metrics become `Episode/x` instead of `<run>/Episode/x`
- Persistent state JSON for resumable, non-duplicating sync

Example:
  python tools/wandb_tb_rewrite_sync.py \
    --tb-logdir /abs/path/to/runs/USV/Mar05_16-07-46/Mar05_16-08-18 \
    --entity loopzhang7-zhejiang-university \
    --project OmniIsaacGymEnvs \
    --run-name Mar05_16-08-18 \
    --strip-prefix auto \
    --poll-interval 10

"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _atomic_write_json(path: str, data: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _default_state_path(tb_logdir: str, run_name: str) -> str:
    safe = run_name.replace("/", "_")
    return os.path.join(tb_logdir, f".wandb_tb_rewrite_state.{safe}.json")


def _normalize_strip_prefix(strip_prefix: str, tb_logdir: str) -> Optional[str]:
    if not strip_prefix:
        return None
    if strip_prefix.lower() == "none":
        return None
    if strip_prefix.lower() == "auto":
        base = os.path.basename(os.path.normpath(tb_logdir))
        return base + "/"
    # ensure it ends with '/'
    return strip_prefix if strip_prefix.endswith("/") else (strip_prefix + "/")


def _tag_matches_prefixes(tag: str, prefixes: Iterable[str]) -> bool:
    for p in prefixes:
        if tag.startswith(p):
            return True
    return False


def _strip_prefix(tag: str, prefix: Optional[str]) -> str:
    if prefix and tag.startswith(prefix):
        return tag[len(prefix) :]
    return tag


def _try_generate_run_id() -> str:
    # Try a couple of wandb-internal helpers without depending on a specific version.
    try:
        from wandb.util import generate_id  # type: ignore

        return generate_id()
    except Exception:
        pass
    try:
        from wandb.sdk.lib import runid  # type: ignore

        return runid.generate_id()
    except Exception:
        pass
    # Fallback: time-based id, not ideal but stable enough.
    return f"{int(time.time())}"  # seconds


@dataclass
class SyncState:
    run_id: str
    last_step_by_tag: Dict[str, int]

    @staticmethod
    def from_json(obj: Mapping[str, Any]) -> "SyncState":
        run_id = str(obj.get("run_id") or "")
        last = obj.get("last_step_by_tag") or {}
        if not isinstance(last, dict):
            last = {}
        last_step_by_tag: Dict[str, int] = {}
        for k, v in last.items():
            try:
                last_step_by_tag[str(k)] = int(v)
            except Exception:
                continue
        return SyncState(run_id=run_id, last_step_by_tag=last_step_by_tag)

    def to_json(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "last_step_by_tag": self.last_step_by_tag,
            "updated_at": _now(),
        }


class GracefulKiller:
    def __init__(self) -> None:
        self.kill_now = False
        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, self._handler)

    def _handler(self, signum: int, frame: Any) -> None:  # noqa: ARG002
        self.kill_now = True


def _build_event_accumulator(tb_logdir: str):
    # Lazy import to keep error messages clear.
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import tensorboard's event_accumulator. "
            "Install with: pip install tensorboard"
        ) from e

    size_guidance = {
        # keep more scalars in memory; TB defaults can be too small.
        event_accumulator.SCALARS: 0,
        event_accumulator.COMPRESSED_HISTOGRAMS: 0,
        event_accumulator.IMAGES: 0,
        event_accumulator.AUDIO: 0,
        event_accumulator.HISTOGRAMS: 0,
        event_accumulator.TENSORS: 0,
    }
    return event_accumulator.EventAccumulator(tb_logdir, size_guidance=size_guidance)


def _collect_new_scalars(
    ea,
    include_prefixes: List[str],
    strip_prefix: Optional[str],
    last_step_by_tag: Mapping[str, int],
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, int], Dict[str, int]]:
    """Return (by_step, new_last_step_by_tag, counts_by_tag)."""

    # TensorBoard API: ea.Tags() returns dict, with key 'scalars'
    tags = ea.Tags().get("scalars", [])
    tags = [t for t in tags if _tag_matches_prefixes(t, include_prefixes)]

    by_step: Dict[int, Dict[str, float]] = {}
    new_last: Dict[str, int] = dict(last_step_by_tag)
    counts: Dict[str, int] = {}

    for tag in tags:
        # Scalars(tag) returns list of ScalarEvent(wall_time, step, value)
        try:
            events = ea.Scalars(tag)
        except Exception:
            continue

        last_step = int(last_step_by_tag.get(tag, -1))
        n = 0
        for ev in events:
            step = int(getattr(ev, "step"))
            if step <= last_step:
                continue
            value = float(getattr(ev, "value"))
            norm_tag = _strip_prefix(tag, strip_prefix)
            by_step.setdefault(step, {})[norm_tag] = value
            if step > new_last.get(tag, -1):
                new_last[tag] = step
            n += 1
        if n:
            counts[tag] = n

    # quick stats: min/max step per tag (optional)
    return by_step, new_last, counts


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Rewrite TB scalar tags and stream to W&B")
    ap.add_argument("--tb-logdir", required=True, help="TensorBoard log directory containing tfevents")
    ap.add_argument("--entity", required=True, help="W&B entity (user or team)")
    ap.add_argument("--project", required=True, help="W&B project")
    ap.add_argument("--run-name", required=True, help="W&B run name shown in UI")
    ap.add_argument(
        "--run-id",
        default="",
        help="Optional fixed W&B run id. If omitted, stored in state file for reuse.",
    )
    ap.add_argument(
        "--state-path",
        default="",
        help="Path to state JSON for resumable sync (default: inside tb-logdir)",
    )
    ap.add_argument(
        "--strip-prefix",
        default="auto",
        help="Prefix to strip from TB tags (e.g. 'Mar05_16-08-18/'), 'auto' uses basename(tb-logdir)/, 'none' disables.",
    )
    ap.add_argument(
        "--include-prefix",
        action="append",
        default=[],
        help="Repeatable: only sync tags starting with this prefix (default: Episode/, Loss/, rewards/, shaped_rewards/, episode_lengths/, Diagnostics/, Policy/)",
    )
    ap.add_argument("--poll-interval", type=float, default=10.0, help="Seconds between Reload polls")
    ap.add_argument(
        "--max-steps-per-commit",
        type=int,
        default=200,
        help="Max distinct steps to send per poll before committing state (avoid huge bursts)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Do not write to W&B; just print what would be sent")
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output (still prints errors)",
    )

    args = ap.parse_args(argv)

    tb_logdir = os.path.abspath(args.tb_logdir)
    if not os.path.isdir(tb_logdir):
        _eprint(f"[{_now()}] ERROR: --tb-logdir is not a directory: {tb_logdir}")
        return 2

    include_prefixes = args.include_prefix or [
        "Episode/",
        "Loss/",
        "rewards/",
        "shaped_rewards/",
        "episode_lengths/",
        "Diagnostics/",
        "Policy/",
    ]
    strip_prefix = _normalize_strip_prefix(args.strip_prefix, tb_logdir)

    state_path = args.state_path or _default_state_path(tb_logdir, args.run_name)
    state_obj = _load_json(state_path)
    if state_obj:
        state = SyncState.from_json(state_obj)
    else:
        state = SyncState(run_id="", last_step_by_tag={})

    # Determine / persist run id
    run_id = (args.run_id or "").strip() or (state.run_id or "").strip()
    if not run_id:
        run_id = _try_generate_run_id()
    state.run_id = run_id

    if not args.quiet:
        print(f"[{_now()}] TB logdir: {tb_logdir}")
        print(f"[{_now()}] W&B: entity={args.entity} project={args.project} name={args.run_name} id={run_id}")
        print(f"[{_now()}] include_prefixes={include_prefixes}")
        print(f"[{_now()}] strip_prefix={strip_prefix!r}")
        print(f"[{_now()}] state_path={state_path}")

    # Init W&B
    if not args.dry_run:
        try:
            import wandb
        except Exception as e:
            _eprint(f"[{_now()}] ERROR: cannot import wandb: {e}")
            return 2

        run = wandb.init(
            entity=args.entity,
            project=args.project,
            name=args.run_name,
            id=run_id,
            resume="allow",
            job_type="tb-rewrite-sync",
            config={
                "tb_logdir": tb_logdir,
                "include_prefixes": include_prefixes,
                "strip_prefix": strip_prefix,
            },
        )
    else:
        run = None

    ea = _build_event_accumulator(tb_logdir)
    killer = GracefulKiller()

    # Warm-up: reload once so Tags() works reliably
    try:
        ea.Reload()
    except Exception as e:
        _eprint(f"[{_now()}] ERROR: failed to Reload() TB logdir: {e}")
        if run is not None:
            try:
                run.finish()
            except Exception:
                pass
        return 2

    # Persist initial state
    _atomic_write_json(state_path, state.to_json())

    last_heartbeat = time.time()

    try:
        while not killer.kill_now:
            t0 = time.time()
            try:
                ea.Reload()
            except Exception as e:
                _eprint(f"[{_now()}] WARN: Reload() failed: {e}")
                time.sleep(args.poll_interval)
                continue

            by_step, new_last, counts = _collect_new_scalars(
                ea,
                include_prefixes=include_prefixes,
                strip_prefix=strip_prefix,
                last_step_by_tag=state.last_step_by_tag,
            )

            if by_step:
                steps = sorted(by_step.keys())
                if args.max_steps_per_commit > 0:
                    steps = steps[: args.max_steps_per_commit]

                sent_points = 0
                for step in steps:
                    payload = by_step[step]
                    if not payload:
                        continue
                    sent_points += len(payload)
                    if args.dry_run:
                        if not args.quiet:
                            keys_preview = list(payload.keys())[:6]
                            more = "" if len(payload) <= 6 else f" (+{len(payload)-6} more)"
                            print(f"[{_now()}] DRY-RUN step={step} keys={keys_preview}{more}")
                    else:
                        # One log call per step: stable step axis
                        run.log(payload, step=step)  # type: ignore[union-attr]

                # Advance state conservatively when we truncate steps.
                # new_last stores the maximum step observed per original TB tag, which may be beyond what we sent.
                # We only advance each tag up to the max step we actually emitted in this poll.
                max_sent_step = max(steps) if steps else None
                if max_sent_step is not None:
                    for tag, last_step in new_last.items():
                        prev = int(state.last_step_by_tag.get(tag, -1))
                        observed = int(last_step)
                        if observed > prev:
                            state.last_step_by_tag[tag] = max(prev, min(observed, int(max_sent_step)))

                _atomic_write_json(state_path, state.to_json())

                if not args.quiet:
                    unique_tags = len(counts)
                    print(
                        f"[{_now()}] synced steps={len(steps)} points={sent_points} tags={unique_tags} "
                        f"dt={time.time()-t0:.2f}s"
                    )
            else:
                # Heartbeat every ~60s
                if (time.time() - last_heartbeat) > 60 and not args.quiet:
                    print(f"[{_now()}] idle (no new scalars)")
                    last_heartbeat = time.time()

            time.sleep(max(0.0, args.poll_interval - (time.time() - t0)))

    finally:
        try:
            _atomic_write_json(state_path, state.to_json())
        except Exception:
            pass
        if run is not None:
            try:
                run.finish()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
