"""E1: tools/cell_gate.py admits only exchange-mode, gated session records with the cell's Σ, an accepted verdict, a
complete link context, the proofs kept beside the record, and a non-producer verifier."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import cell_gate as G  # noqa: E402

M = 3
SIGMA_TXT = f"verity/link-L/v2\nscheme blake3-keyed/row/v2\nm {M}\n"
SIGMA = hashlib.sha256(SIGMA_TXT.encode()).hexdigest()
PROOFS = {"chain/rep0": b"proof zero", "chain/rep1": b"proof one"}


def honest() -> dict:
    return {
        "kind": "flock-live-session/v1",
        "hello": json.dumps({"profile": "fast100", "reps": 2, "link": {"sigma": SIGMA, "m": M, "points": 2}}),
        "link_mode": "exchange",
        "link": {"mode": "exchange", "sigma": SIGMA, "root_f_sha256": "aa" * 32, "root_b_sha256": {"chain": "bb" * 32},
                 "points": [["01" * 16] * M, ["02" * 16] * M], "y": ["03" * 16, "04" * 16]},
        "config": {"require_link": True, "reps": 2},
        "proof_sha256": [{"stream": s, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)} for s, b in PROOFS.items()],
        "aborted": None,
        "verdict": {"accepted": True, "reps": [{"stream": s, "ok": True} for s in PROOFS]},
    }


def write(tmp: Path, rec: dict, operator: str = "red-team-verifier", proofs: dict | None = None) -> tuple[Path, Path]:
    s = tmp / "session"
    s.mkdir(parents=True, exist_ok=True)
    (s / "session.json").write_text(json.dumps(rec))
    for name, b in (PROOFS if proofs is None else proofs).items():
        (s / f"{name.replace('/', '.')}.bin").write_bytes(b)
    (s / "verifier.json").write_text(json.dumps({"operator": operator, "runpod_pod_id": "verpod"}))
    (tmp / "sigma.txt").write_text(SIGMA_TXT)
    return s, tmp / "sigma.txt"


def run(tmp: Path, rec: dict, **kw) -> dict:
    s, sig = write(tmp, rec, **kw)
    checks, _ = G.check(s, sig, ["agkr-flock-cell"], ["prodpod"])
    return checks


def test_honest_record_admitted(tmp_path):
    checks = run(tmp_path, honest())
    assert all(v["ok"] for v in checks.values()), checks
    text = G.link_txt(honest())
    assert text.startswith(G.LINK_TXT) and text.count("\npoint ") == 2 and f"sigma {SIGMA}" in text


def mutate(f):
    r = copy.deepcopy(honest())
    f(r)
    return r


@pytest.mark.parametrize("name,rec,kw,fails", [
    ("stub", mutate(lambda r: (r.update(link_mode="stub"), r.update(link=None))), {}, "exchange"),
    ("ungated", mutate(lambda r: r["config"].update(require_link=False)), {}, "gated"),
    ("other_sigma", mutate(lambda r: r["link"].update(sigma="cc" * 32)), {}, "sigma"),
    ("rejected", mutate(lambda r: r["verdict"]["reps"][1].update(ok=False)), {}, "accepted"),
    ("one_point", mutate(lambda r: r["link"].update(points=r["link"]["points"][:1])), {}, "context"),
    ("proof_missing", honest(), {"proofs": {"chain/rep0": PROOFS["chain/rep0"]}}, "preserved"),
    ("proof_altered", honest(), {"proofs": {**PROOFS, "chain/rep1": b"another proof"}}, "preserved"),
    ("producer_verifier", honest(), {"operator": "agkr-flock-cell"}, "non_producer"),
    ("anonymous_verifier", honest(), {"operator": ""}, "non_producer"),
])
def test_refused(tmp_path, name, rec, kw, fails):
    checks = run(tmp_path, rec, **kw)
    assert not checks[fails]["ok"], (name, checks[fails])
