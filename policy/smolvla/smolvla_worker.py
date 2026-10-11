#!/usr/bin/env python

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _seed_everything(seed: int) -> None:
    """PATCH[seedfix]: SmolVLA 的 flow-matching 采样 (sample_noise) 走全局 torch RNG。

    eval_policy.py 的 prepare_episode_reset() 只 seed 了**仿真进程**的 RNG；
    策略 worker 是独立进程、从未被 seed ⇒ 同一集每次跑动作都不一样
    （实测逐集翻转率 20~35%）。按集 seed 之后同一 seed 的评测才可复现，
    A/B 两臂也才真正逐集配对。
    """
    import random as _random
    _random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--lerobot-root", default="")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_runtime(args: argparse.Namespace):
    if args.lerobot_root:
        lerobot_src = Path(args.lerobot_root).expanduser().resolve() / "src"
        if lerobot_src.exists():
            sys.path.insert(0, str(lerobot_src))
        else:
            raise FileNotFoundError(
                f"LeRobot source path does not exist: {lerobot_src}. "
                "Install lerobot in this Python environment or set SMOLVLA_LEROBOT_ROOT."
            )

    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor import (
            PolicyProcessorPipeline,
            batch_to_transition,
            transition_to_batch,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cannot import lerobot. Install LeRobot in the SmolVLA Python env, "
            "or run policy/smolvla/setup_lerobot.sh and set SMOLVLA_LEROBOT_ROOT."
        ) from exc

    # `policy_action_to_transition` / `transition_to_policy_action` are re-exported by
    # `lerobot.processor` from LeRobot 0.6 on, but on older releases (e.g. 0.4.x) they
    # only live in `lerobot.processor.converters`. Accept both paths so the worker does
    # not depend on the exact release the evaluation host happens to provide.
    try:
        from lerobot.processor import policy_action_to_transition, transition_to_policy_action
    except ImportError:
        from lerobot.processor.converters import (
            policy_action_to_transition,
            transition_to_policy_action,
        )

    checkpoint_dir = str(Path(args.checkpoint_dir).resolve())
    policy = SmolVLAPolicy.from_pretrained(checkpoint_dir)
    policy.to(args.device)
    policy.eval()

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_dir,
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": args.device}},
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_dir,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return policy, preprocessor, postprocessor


def _load_obs(obs_path: str) -> tuple[dict[str, np.ndarray], str]:
    with np.load(obs_path, allow_pickle=False) as data:
        obs = {key: data[key] for key in data.files if key != "task"}
        task = str(data["task"].item()) if "task" in data else ""
    return obs, task


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    args = _parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        policy, preprocessor, postprocessor = _load_runtime(args)

    _boot = os.environ.get("SMOLVLA_SEED")
    if _boot is not None and str(_boot).strip() != "":
        _seed_everything(int(_boot))
        print(f"[seedfix] worker RNG seeded with SMOLVLA_SEED={_boot}", file=sys.stderr)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        request = json.loads(line)
        cmd = request.get("cmd")

        try:
            if cmd == "reset":
                _seed = request.get("seed")
                if _seed is not None:
                    _seed_everything(int(_seed))
                policy.reset()
                _send({"ok": True})
                continue

            if cmd != "infer":
                raise ValueError(f"Unsupported command: {cmd}")

            with contextlib.redirect_stdout(sys.stderr):
                obs, task = _load_obs(request["obs_path"])
                obs = {key: torch.from_numpy(value) for key, value in obs.items()}
                # PATCH: images arrive as uint8 HWC; preprocessor VISUAL=IDENTITY expects float [0,1]
                for _k, _v in obs.items():
                    if _v.dtype == torch.uint8:
                        obs[_k] = _v.float().div_(255.0)
                obs["task"] = task
                proc_obs = preprocessor(obs)

                num_actions = int(request.get("n_action_steps", 1))
                actions = []
                for _ in range(num_actions):
                    action = policy.select_action(proc_obs)
                    action = postprocessor(action)
                    actions.append(action.squeeze(0).cpu().numpy().astype(np.float32).tolist())

            _send({"ok": True, "actions": actions})
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc()
            _send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

