"""One engine a leased GPU: `node.sh` starts an engine on each GPU CUDA_VISIBLE_DEVICES names, and runs this first,
which refuses, with the reason, unless those GPUs are exactly the ones its gpu-lease holds. They must be distinct and as
many as the engines and as GPU_LEASE_UUID's UUIDs. Where gpu-lease is on the PATH, they must also be exactly the GPUs
`gpu-lease status` shows held by a lease this process runs under, its holder's pid an ancestor's. A GPU another process
held never joins the lease, so 8 engines under a lease of GPUs 1-7 are refused before any starts.

    python lease.py ENGINES        exit 0, or 3 with the reason
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path


def ancestors(pid: int | None = None) -> set[int]:
    """`pid` (this process by default) and every process above it."""
    pid, out = pid or os.getpid(), set()
    while pid > 1 and pid not in out:
        out.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
    return out


def held(status: str, holders: set[int]) -> dict[str, str]:
    """The GPUs `gpu-lease status` shows held by a lease whose pid is in `holders`: index to UUID."""
    out = {}
    for line in status.splitlines():
        words = line.split()
        if len(words) < 2 or words[1] != "held":
            continue
        kv = dict(w.split("=", 1) for w in words[2:] if "=" in w)
        if kv.get("pid", "").isdigit() and int(kv["pid"]) in holders:
            out[words[0]] = kv.get("uuid", "-")
    return out


def refusal(engines: int, env: Mapping[str, str], status: str | None, holders: set[int]) -> str | None:
    """Why `engines` engines, one a GPU CUDA_VISIBLE_DEVICES names, may not start under this lease, or None: `status`
    is `gpu-lease status`'s output (None: no gpu-lease here), `holders` this process and its ancestors."""
    named = env.get("CUDA_VISIBLE_DEVICES", "")
    visible = [d for d in named.split(",") if d]
    if len(visible) != engines:
        return f"{engines} engines, but CUDA_VISIBLE_DEVICES={named!r} names {len(visible)} GPUs"
    if len(set(visible)) != len(visible):
        return f"CUDA_VISIBLE_DEVICES={named!r} names a GPU twice"
    uuids = [u for u in env.get("GPU_LEASE_UUID", "").split(",") if u]
    if env.get("GPU_LEASE_UUID") and len(uuids) != engines:
        return f"the lease holds {len(uuids)} GPUs (GPU_LEASE_UUID), not the {engines} CUDA_VISIBLE_DEVICES names"
    if status is None:
        return None
    mine = held(status, holders)
    ours = set(mine.values()) if all(d.startswith("GPU-") for d in visible) else set(mine)
    if ours != set(visible):
        return (f"gpu-lease status shows this lease holding GPUs {','.join(sorted(mine, key=int)) or 'none'}, not "
                f"the {engines} CUDA_VISIBLE_DEVICES={named!r} names: a GPU another process held never joined it")
    return None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or not argv[0].isdigit() or int(argv[0]) < 1:
        print("usage: lease.py ENGINES (1 or more)", file=sys.stderr)
        return 2
    engines, status = int(argv[0]), None
    if (exe := shutil.which("gpu-lease")) is not None:
        p = subprocess.run([exe, "status"], capture_output=True, text=True)
        if p.returncode:
            print(f"lease: refused before any engine started: `gpu-lease status` exited {p.returncode} "
                  f"({p.stderr.strip()[-200:]}), so the GPUs this lease holds are unknown", flush=True)
            return 3
        status = p.stdout
    why = refusal(engines, os.environ, status, ancestors())
    if why is not None:
        print(f"lease: refused before any engine started: {why}", flush=True)
        return 3
    print(f"lease: {engines} engines, one on each GPU the lease holds ({os.environ['CUDA_VISIBLE_DEVICES']})",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
