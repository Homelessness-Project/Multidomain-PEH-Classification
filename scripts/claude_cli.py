from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


class ClaudeNotFoundError(RuntimeError):
    pass


class ClaudeAuthError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClaudeResult:
    stdout: str
    stderr: str
    exit_code: int


def _detect_claude_cmd() -> str:
    cmd = os.environ.get("CLAUDE_CMD", "").strip()
    if cmd:
        return cmd

    found = shutil.which("claude")
    if found:
        return found

    raise ClaudeNotFoundError(
        "Could not find `claude` CLI. Install Claude Code so `claude` is on PATH, "
        "or set env var CLAUDE_CMD to the full path of the executable."
    )


def auth_status_text() -> str:
    claude_cmd = _detect_claude_cmd()
    proc = subprocess.run(
        [claude_cmd, "auth", "status", "--text"],
        text=True,
        capture_output=True,
        check=False,
    )
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    return "\n".join([s for s in [out, err] if s]).strip()


def run_claude(prompt: str, *, timeout_s: int | None = None, model: str | None = None) -> ClaudeResult:
    """
    Runs the local `claude` CLI in non-interactive mode by piping the prompt on stdin.
    """
    claude_cmd = _detect_claude_cmd()

    status = auth_status_text().lower()
    if "not logged in" in status:
        raise ClaudeAuthError("Not logged in. Run `claude auth login` once, then re-run.")

    argv = [claude_cmd, "-p", "Follow the instructions in stdin exactly. Output only what is requested."]
    if model:
        argv.extend(["--model", model])

    proc = subprocess.run(
        argv,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    return ClaudeResult(stdout=proc.stdout, stderr=proc.stderr, exit_code=proc.returncode)


def claude_code_version_text() -> str | None:
    try:
        claude_cmd = _detect_claude_cmd()
    except ClaudeNotFoundError:
        return None
    proc = subprocess.run([claude_cmd, "--version"], text=True, capture_output=True, check=False)
    out = (proc.stdout or "").strip()
    return out or None

