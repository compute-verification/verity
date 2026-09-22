"""φ — the base context known before execution.

    Phi.config    the static model configuration C (b1 keys; the profile module's `PROFILE.config`, BN included)
    Phi.profile   {module, id, pins}: the pinned vLLM profile (module name under `verity_capture.profiles`, the RunHeader's
                  12-hex `profile_id` that binds `Captured_v1{PROFILE}`, and the kernel pins the patterns accept)
    Phi.engine    scheduler facts: block_size, max_num_seqs, max_num_batched_tokens, enable_chunked_prefill, enable_prefix_caching
    Phi.workload  the EXACT `RunHeader.workload` dict (requests with request_id / prompt_token_ids / max_tokens / seed / n /
                  arrive_step, sampling, source, sha256): its canonical sha256 is `Captured_v1{WORKLOAD}`

`phi_from_log(path)` derives φ from a capture log's RunHeader (encode side; reads the header line only).  `phi.json` is the
declared input of the decode side; `Phi.digest()` is what the harness records as the input hash.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PHI_SCHEMA = "verity-aot/phi/v1"

_ENGINE_KEYS = ("block_size", "max_num_seqs", "max_num_batched_tokens", "enable_chunked_prefill", "enable_prefix_caching")


@dataclass
class Phi:
    schema: str = PHI_SCHEMA
    config: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)      # {"module": str, "id": str, "pins": dict}
    engine: dict[str, Any] = field(default_factory=dict)
    workload: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)        # repo / revision / config sha (informational; C is what binds)

    def to_json(self) -> dict:
        return {"schema": self.schema, "config": self.config, "profile": self.profile, "engine": self.engine,
                "workload": self.workload, "model": self.model}

    @classmethod
    def from_json(cls, d: dict) -> "Phi":
        if d.get("schema") != PHI_SCHEMA:
            raise ValueError(f"phi schema {d.get('schema')!r} != {PHI_SCHEMA!r}")
        return cls(config=dict(d["config"]), profile=dict(d["profile"]), engine=dict(d["engine"]),
                   workload=dict(d["workload"]), model=dict(d.get("model") or {}))

    def canonical(self) -> bytes:
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()

    def digest(self) -> str:
        return hashlib.sha256(self.canonical()).hexdigest()

    @property
    def requests(self) -> list[dict]:
        """The header's requests (one entry per submitted request; `n` may be > 1)."""
        return list(self.workload.get("requests") or [])

    def logical_requests(self) -> list[dict]:
        """The requests the ENGINE schedules, derived from φ: a request with n > 1 is forked into n children, added in index order
        right after each other, child id `f"{index}_{request_id}"`, child seed `seed + index` (None stays None), n = 1 — SOURCE-
        INSPECTED vllm/v1/engine/parallel_sampling.py:80,92 (`_get_child_sampling_params`, `get_child_info`) and
        llm_engine.py:289-290 (`for idx in range(n): ... add_request(child)`).  Every entry carries `_parent` (index into
        `requests`; the child's prompt IS the parent's prompt parameter, as in the fold) and `_child` (index within n, 0 for n = 1).
        The Program's outputs, the hints' `request` indices and the predictor's requests all index THIS list; for a workload
        without forks it is `requests` itself."""
        out: list[dict] = []
        for pi, r in enumerate(self.requests):
            n = int(r.get("n") or 1)
            if n <= 1:
                out.append({**r, "_parent": pi, "_child": 0})
                continue
            for i in range(n):
                seed = r.get("seed")
                out.append({**r, "request_id": f"{i}_{r['request_id']}", "n": 1, "seed": (None if seed is None else int(seed) + i),
                            "_parent": pi, "_child": i})
        return out

    @property
    def profile_id(self) -> str:
        return str(self.profile["id"])

    def workload_digest(self) -> str:
        from ...fold import workload_digest
        return workload_digest(self.workload)


def profile_config(module: str) -> dict[str, Any]:
    """`PROFILE.config` of `verity_capture.profiles.<module>` (BN included): the C the fold binds under that profile."""
    mod = importlib.import_module(f"verity_capture.profiles.{module}")
    prof = getattr(mod, "PROFILE")
    return dict(prof.config)


def _header_dict(path: str | Path) -> dict:
    p = Path(path)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("$e") == "RunHeader":
                return d
            break
    raise ValueError(f"{path}: first event is not a RunHeader")


def phi_from_log(log_path: str | Path, profile_module: str) -> Phi:
    """φ from a capture log's RunHeader + the named profile module (encode side)."""
    h = _header_dict(log_path)
    prof = h.get("profile") or {}
    cache = dict(prof.get("cache") or {})
    engine = {k: cache.get(k) for k in _ENGINE_KEYS}
    C = profile_config(profile_module)
    model = dict(h.get("model") or {})
    model.pop("local_path", None)
    return Phi(config=_jsonable(C), profile={"module": profile_module, "id": h["profile_id"], "pins": _profile_pins(profile_module)},
               engine=engine, workload=dict(h["workload"]), model={"repo": model.get("repo"), "revision": model.get("revision"),
                                                                    "config": model.get("config")})


def _profile_pins(module: str) -> dict:
    mod = importlib.import_module(f"verity_capture.profiles.{module}")
    prof = getattr(mod, "PROFILE")
    notes = dict(getattr(prof, "notes", {}) or {})
    return _jsonable({"profile_id_string": prof.id, "pins": notes.get("pins"), "kappa": notes.get("kappa")})


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def load_phi(path: str | Path) -> Phi:
    return Phi.from_json(json.loads(Path(path).read_text()))


def save_phi(phi: Phi, path: str | Path) -> None:
    Path(path).write_text(json.dumps(phi.to_json(), indent=1, sort_keys=True) + "\n")
