"""A served row's compile artifact, pinned.  Inductor autotunes a kernel in the process that compiles it, timing its candidate configs
into that process's cache dir, and another process can time another winner: another config, another Program (served compiled
numerics vary across processes at fixed code, about one process in seven).  The Build and the Commit must run one artifact.

The row's artifact is what its Build read (`record`, carried by `served.build`'s Build as `artifact`): each kernel the Build bound
(its source's sha256 -> the config the process launched it with) and each autotune winner of the Inductor dir (`<dir>/<kernel>.best_config`,
by its path, the config without its timing), digested (SHA-512).  The wrappers are not in it as files: they name their kernels by
absolute path, and the Program digest holds what they compute.  A Commit process seeds its fresh Inductor dir with the Build's winners
at their paths (`served-pin seed`, `ops/served_commit.sh`), as a served process restarted on its own cache reads them, and the treatment
refuses to serve when the artifact its own Build read is not the Build of record's (`ServedCommit.rebuild`, `differences`).  A
process that serves the workload without the source's hooks (the control arm, a bench) has no launch record and checks its winners
(`served-pin check`).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

from verity_vllm.config import UsageError, option

#: a winner's fields that are not its config (Inductor drops them on load)
TIMING = ("time_taken_ms",)
PARTS = ("kernels", "winners")


def winners(inductor_dir: str) -> dict[str, dict[str, Any]]:
    """Each autotune winner of an Inductor dir by its path under the dir: the config its `.best_config` records, without its timing."""
    out = {}
    for path in sorted(glob.glob(os.path.join(inductor_dir, "*", "*.best_config"))):
        with open(path) as f:
            out[os.path.relpath(path, inductor_dir)] = {k: v for k, v in sorted(json.load(f).items()) if k not in TIMING}
    return out


def record(inductor_dir: str, kernels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The artifact a Build read: its kernels' launched configs (`kernels`: source sha256 -> the Build's kernel record), the dir's
    winners, and their digest."""
    return artifact_of(kernels, winners(inductor_dir))


def artifact_of(kernels: dict[str, dict[str, Any]], won: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """`record` of kernels and winners `won` (`winners`' form)."""
    body = {"kernels": {sha: kernels[sha]["config"] for sha in sorted(kernels)}, "winners": won}
    return {"digest": hashlib.sha512(json.dumps(body, sort_keys=True).encode()).hexdigest(), **body}


def differences(pinned: dict[str, Any], loaded: dict[str, Any], parts: tuple[str, ...] = PARTS) -> list[str]:
    """Each kernel or winner whose config `loaded` does not share with `pinned`, by name."""
    out = []
    for part in parts:
        a, b = pinned.get(part) or {}, loaded.get(part) or {}
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                out.append(f"{part} {k}: pinned {a.get(k, 'absent')}, loaded {b.get(k, 'absent')}")
    return out


def pinned(build: dict[str, Any]) -> dict[str, Any]:
    """The Build's artifact; a Build without one (built before the pin) cannot be shown to be what a process loads."""
    art = build.get("artifact")
    if not isinstance(art, dict) or not all(isinstance(art.get(p), dict) for p in PARTS):
        raise UsageError(f"the Build (program {str(build.get('program_digest'))[:16]}) records no compile artifact: build it again "
                         "(served-match) so that its artifact is pinned")
    return art


def seed(artifact: dict[str, Any], inductor_dir: str) -> int:
    """Write the artifact's winners into a fresh Inductor dir at their paths; the number written.  A dir with a winner of its own
    is not fresh."""
    if winners(inductor_dir):
        raise UsageError(f"{inductor_dir} already holds autotune winners: seed a fresh Inductor dir")
    for rel, cfg in artifact["winners"].items():
        path = os.path.join(inductor_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(cfg, f)
    return len(artifact["winners"])


@dataclass(frozen=True, kw_only=True)
class SeedOptions:
    build: str = option("--build", help="the Build of record (served-match's build.json): its pinned artifact")
    inductor_dir: str = option("--inductor-dir", help="the fresh Inductor dir a process is about to compile into (TORCHINDUCTOR_CACHE_DIR)")
    cmd: str = field(default="seed", init=False)


@dataclass(frozen=True, kw_only=True)
class CheckOptions:
    build: str = option("--build", help="the Build of record (served-match's build.json): its pinned artifact")
    inductor_dir: str = option("--inductor-dir", help="the Inductor dir a process served from")
    cmd: str = field(default="check", init=False)


COMMANDS = {"seed": SeedOptions, "check": CheckOptions}


def main(a: SeedOptions | CheckOptions) -> int:
    with open(a.build) as f:
        art = pinned(json.load(f))
    if a.cmd == "seed":
        n = seed(art, a.inductor_dir)
        print(f"[served-pin] seeded {n} autotune winners of artifact {art['digest'][:16]} -> {a.inductor_dir}")
        return 0
    diff = differences(art, {"winners": winners(a.inductor_dir)}, ("winners",))
    print(f"[served-pin] {a.inductor_dir}: winners {'equal to' if not diff else 'not'} artifact {art['digest'][:16]}'s"
          + "".join(f"\n  {d}" for d in diff[:20]))
    return 3 if diff else 0
