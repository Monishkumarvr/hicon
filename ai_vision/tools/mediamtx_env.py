#!/usr/bin/env python3
"""Emit MediaMTX path overrides derived from the current environment."""
from __future__ import annotations

import os
import shlex
from typing import Mapping


DEFAULT_OVERLAY_SCRIPT = (
    "/home/hicon/hicon/ai_vision/tools/stream0_overlay_remote_relay.sh"
)


def _is_true(value: str | None) -> bool:
    return str(value or "").strip().lower() == "true"


def build_mediamtx_env(
    env: Mapping[str, str] | None = None,
    *,
    overlay_script_path: str = DEFAULT_OVERLAY_SCRIPT,
) -> dict[str, str]:
    env = dict(os.environ if env is None else env)
    overrides: dict[str, str] = {}

    stream0_source = env.get("HICON_CPPLUS_SOURCE_STREAM_0", "").strip()
    if stream0_source:
        overrides["MTX_PATHS_STREAM0_SOURCE"] = stream0_source
        overrides["MTX_PATHS_STREAM0_SOURCEONDEMAND"] = (
            "yes" if _is_true(env.get("HICON_USE_SEGMENT_BUFFER_0")) else "no"
        )

    remote_relay_url = env.get("HICON_STREAM0_REMOTE_RELAY_URL", "").strip()
    if remote_relay_url:
        overrides["MTX_PATHS_STREAM0_OVERLAY_RUNONREADY"] = overlay_script_path
        overrides["MTX_PATHS_STREAM0_OVERLAY_RUNONREADYRESTART"] = "yes"

    return overrides


def emit_shell_exports(
    env: Mapping[str, str] | None = None,
    *,
    overlay_script_path: str = DEFAULT_OVERLAY_SCRIPT,
) -> str:
    overrides = build_mediamtx_env(env, overlay_script_path=overlay_script_path)
    return "\n".join(
        f"export {key}={shlex.quote(value)}"
        for key, value in sorted(overrides.items())
    )


def main() -> int:
    output = emit_shell_exports()
    if output:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
