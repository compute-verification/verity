"""The landing's policy over what the runners report (`tools/check/policy.toml`)."""
from __future__ import annotations

import re
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PATH = "tools/check/policy.toml"
SCHEMA = "policy/v0"
ID_RE = re.compile(r"suite [^\s:]+(::\S.*)?|circuit-check \S+/\S+ [^\s{]+")
ALLOW_FIELDS = ("id", "owner", "expires", "reason")


class PolicyError(ValueError):
    pass


def _utc(text: str) -> float:
    if not text.endswith("Z"):
        raise ValueError(f"{text!r} doesn't end in Z")
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S%z").timestamp()


@dataclass(frozen=True)
class Allowance:
    id: str
    owner: str
    expires: str
    reason: str

    def live(self, now: float) -> bool:
        return now < _utc(self.expires)


@dataclass(frozen=True)
class Coverage:
    covered: bool
    expired: tuple[Allowance, ...] = ()


def circuit_keys(log: str) -> list[str]:
    out = []
    for line in log.splitlines():
        if line.startswith("FAIL ") and (parts := line[5:].split()):
            out.append(parts[0] + (f" {parts[1].split('{')[0]}" if len(parts) > 1 else ""))
    return out


@dataclass(frozen=True)
class Policy:
    off_gate: tuple[str, ...] = ()
    allow: tuple[Allowance, ...] = ()

    def _judge(self, ids: set[str], covered, now: float) -> Coverage:
        live = {a.id for a in self.allow if a.live(now)}
        if covered(live):
            return Coverage(True)
        return Coverage(False, tuple(a for a in self.allow if a.id in ids and not a.live(now)))

    def suite(self, name: str, row: dict, now: float) -> Coverage:
        tests = [str(t) for t in row.get("failed_tests") or ()]
        whole, each = f"suite {name}", [f"suite {name}::{t}" for t in tests]
        return self._judge({whole, *each}, lambda live: whole in live or (
            bool(each) and row.get("beyond_tests") is False and all(t in live for t in each)), now)

    def circuit(self, log: str, now: float) -> Coverage:
        each = [f"circuit-check {k}" for k in circuit_keys(log)]
        return self._judge(set(each), lambda live: bool(each) and all(k in live for k in each), now)


def parse(text: str, source: str = PATH, handles: set[str] | None = None) -> Policy:
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise PolicyError(f"{source}: not TOML: {e}") from None
    if doc.get("schema") != SCHEMA:
        raise PolicyError(f"{source}: schema {doc.get('schema')!r}, expected {SCHEMA!r}")
    if unknown := set(doc) - {"schema", "off_gate", "allow"}:
        raise PolicyError(f"{source}: unknown keys {sorted(unknown)}")
    off = doc.get("off_gate", [])
    if not (isinstance(off, list) and all(isinstance(x, str) and re.fullmatch(r"[\w.-]+", x) for x in off)):
        raise PolicyError(f"{source}: off_gate must be a list of job names")
    allow: list[Allowance] = []
    for i, t in enumerate(doc.get("allow", [])):
        where = f"{source}: allow {i + 1}"
        if not isinstance(t, dict) or set(t) != set(ALLOW_FIELDS) or not all(isinstance(v, str) and v for v in t.values()):
            raise PolicyError(f"{where}: an allowance is {', '.join(ALLOW_FIELDS)}, each a non-empty string")
        a = Allowance(**t)
        if not ID_RE.fullmatch(a.id):
            raise PolicyError(f"{where}: id {a.id!r} is none of `suite NAME`, `suite NAME::TEST` or `circuit-check CHECK/CODE "
                              "FAMILY`")
        if a.id in {b.id for b in allow}:
            raise PolicyError(f"{where}: id {a.id!r} is listed twice")
        try:
            _utc(a.expires)
        except ValueError:
            raise PolicyError(f"{where}: expires {a.expires!r} isn't a UTC time as YYYY-MM-DDTHH:MM:SSZ") from None
        if handles is not None and a.owner not in handles:
            raise PolicyError(f"{where}: owner {a.owner!r} isn't a handle in the registry ({', '.join(sorted(handles))})")
        allow.append(a)
    return Policy(tuple(off), tuple(allow))


def at(repo: Path, commit: str) -> Policy | str:
    r = subprocess.run(["git", "show", f"{commit}:{PATH}"], cwd=repo, capture_output=True, text=True)
    if r.returncode:
        exists = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo, capture_output=True).returncode == 0
        return Policy() if exists else f"{commit[:12]} isn't a commit here, so its {PATH} can't be read"
    try:
        return parse(r.stdout, f"{PATH} at {commit[:12]}")
    except PolicyError as e:
        return str(e)


__all__ = ["ALLOW_FIELDS", "Allowance", "ID_RE", "PATH", "Policy", "PolicyError", "SCHEMA", "Coverage", "at", "circuit_keys",
           "parse"]
