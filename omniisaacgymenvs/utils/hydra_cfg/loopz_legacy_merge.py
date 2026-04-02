from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from omegaconf import DictConfig, OmegaConf, open_dict


def _try_get_task_choice() -> Optional[str]:
    """Best-effort get Hydra runtime task choice string.

    Examples:
      - "USV/IROS2024/USV_Virtual_CaptureXY_SysID-TEST"
    """

    try:
        from hydra.core.hydra_config import HydraConfig

        hc = HydraConfig.get()
        choices = getattr(getattr(hc, "runtime", None), "choices", None)
        if isinstance(choices, dict):
            v = choices.get("task")
            return str(v) if v is not None else None
    except Exception:
        return None
    return None


def _split_domain_suite(task_choice: str) -> Optional[Tuple[str, str]]:
    parts = [p for p in str(task_choice).split("/") if p]
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def _cfg_root_dir() -> str:
    # This file lives in omniisaacgymenvs/utils/hydra_cfg; cfg is at omniisaacgymenvs/cfg.
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "cfg"))


def _default_override_path_from_task(task_choice: str) -> Optional[str]:
    ds = _split_domain_suite(task_choice)
    if ds is None:
        return None
    domain, suite = ds
    path = os.path.join(_cfg_root_dir(), "task", domain, suite, "cfg.yaml")
    return path if os.path.isfile(path) else None


def merge_legacy_env_arch(
    cfg: DictConfig,
    *,
    override_path: Optional[str] = None,
    allow_fallback_iros2024: bool = True,
    verbose: bool = True,
) -> Optional[str]:
    """Merge legacy loopz keys `environment` + `architecture` into Hydra cfg.

    Intent:
      - Fill missing keys without overriding explicit CLI overrides.
      - Auto-locate legacy cfg.yaml using Hydra runtime task choice.

    Returns:
      The override path used if merge succeeded, else None.
    """

    # Optional explicit path from cfg (lets users avoid code changes later).
    for key in ("legacy_cfg_path", "loopz_legacy_cfg_path"):
        try:
            v = getattr(cfg, key)
            if v:
                override_path = str(v)
                break
        except Exception:
            pass

    if override_path is None:
        task_choice = _try_get_task_choice()
        if task_choice:
            override_path = _default_override_path_from_task(task_choice)

    if override_path is None and allow_fallback_iros2024:
        # Conservative fallback: keep current project convention.
        candidate = os.path.join(_cfg_root_dir(), "task", "USV", "IROS2024", "cfg.yaml")
        if os.path.isfile(candidate):
            override_path = candidate

    if override_path is None:
        return None

    try:
        override_cfg = OmegaConf.load(override_path)
        env_override: Dict[str, Any] = (
            OmegaConf.to_container(getattr(override_cfg, "environment", {}), resolve=False) or {}
        )
        arch_override: Dict[str, Any] = (
            OmegaConf.to_container(getattr(override_cfg, "architecture", {}), resolve=False) or {}
        )

        # Avoid stomping over the real VecEnv/task parallelism settings.
        env_override.pop("num_envs", None)
        env_override.pop("num_threads", None)

        legacy_env_cfg = OmegaConf.create(env_override)
        legacy_arch_cfg = OmegaConf.create(arch_override)

        with open_dict(cfg):
            # Existing cfg (including CLI overrides) must win.
            current_env = getattr(cfg, "environment", OmegaConf.create({}))
            current_arch = getattr(cfg, "architecture", OmegaConf.create({}))
            cfg.environment = OmegaConf.merge(legacy_env_cfg, current_env)
            cfg.architecture = OmegaConf.merge(legacy_arch_cfg, current_arch)

        if verbose:
            print(f"[loopz-legacy-merge] merged legacy overrides: {override_path}")
        return override_path

    except Exception as exc:
        if verbose:
            print(f"[loopz-legacy-merge] WARN: failed to merge '{override_path}': {exc}")
        return None
