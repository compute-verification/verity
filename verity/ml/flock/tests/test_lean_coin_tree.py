"""The keyed coin tree, `coin-tree/hm96-sha512/v2` (`docs/coin-tree-v2.md` §3 and §7), on the Lean verifier's side
(`Flock.CoinTree`, `flock-verify coin-tree-check`): R7 takes the prover's nonce from its `Hello` and requires the verifier's
own `Hello` with that nonce, byte for byte; the record's `coin_tree` has the nonce, a key with bit 2047 zero, a root and the
verifier's spec; and no two sessions verified together share a key. The values are the spec's §9 vectors."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"

#: coin-tree-v2.md §9
NONCE = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
KEY = ("53d43b82d13a8d6cd586807a53763a5864e185bca221ce92e9d427b309dee61dba712538f39a2df44ab848082f9399912409587ef73bec0e93b"
       "b890a30618c786071e1daec3262a1a4172b6133844b9d6095bfc7e1c10907c13f346242ba55f6da7ce6933f81e691d1e5c423a13ecee129e4c48c"
       "3ce4accf0203a6629c9b2973d922a4ed7d7ac9a477ecca699155231c4558ca66b6291f43b23e3df6f2d43a80f69b5cdc1961ae302bbf547bdac9"
       "9657384b447ddc8cff866d6429b101e6dfca9ec72e302c96d55bba508c16ed1abe3d1057c421c2ffbf1266c6485e90f8e1e22081b72242fe8396"
       "aca78cc5908947d6ebde3c7def16cc0dbb65ed499807e625")
ROOT = ("468d899b6bde51c72568b9cd72b6d35c809454fe82b991cd46647fec631224cc37af0b859a1133ce849b690e6583b7a12ddb90b2126611e6dcb"
        "aad8daee35740")
SPEC = {"block": 3, "rounds_log": 2, "scheme": "coin-tree/hm96-sha512/v2", "streams": ["session", "t/rep0", "t/rep1"]}
HELLO_COINS = ('{"block":3,"nonce":"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f","rounds_log":2,'
               '"scheme":"coin-tree/hm96-sha512/v2","streams":["session","t/rep0","t/rep1"]}')
#: the verifier's own Hello: its session parameters, the tree's coins with the nonce empty (R7 puts the prover's in)
OWN = {"digests": "sha512", "flavor": "rs", "link": {"m": 27, "points": 2, "sigma": "ab" * 64}, "profile": "fast100",
       "reps": 2, "rounds": "bytes retained", "tables": ["nl"], "coins": {**SPEC, "nonce": ""}}

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@pytest.fixture(scope="module", autouse=True)
def built():
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)


def _canon(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def _hello(nonce: str = NONCE, **change) -> str:
    return _canon({**OWN, **change, "coins": {**SPEC, "nonce": nonce}})


def _tree(**change) -> dict:
    return {"key": KEY, "nonce": NONCE, "root": ROOT, "spec": SPEC, **change}


def _check(tmp_path: Path, sessions: list[dict]) -> list[dict]:
    (tmp_path / "case.json").write_text(json.dumps({"hello": OWN, "sessions": sessions}))
    p = subprocess.run([str(BIN), "coin-tree-check", str(tmp_path / "case.json")], capture_output=True, text=True)
    return [json.loads(ln.split(" ", 1)[1]) for ln in p.stdout.splitlines() if ln.startswith("COINTREE ")]


def test_r7_takes_the_provers_nonce_and_the_record_checks(tmp_path):
    (r,) = _check(tmp_path, [{"hello": _hello(), "coin_tree": _tree()}])
    assert r == {"ok": True, "nonce": NONCE, "coins": HELLO_COINS}


@pytest.mark.parametrize("name,session,error", [
    ("nonce case", {"hello": _hello(NONCE.upper())}, "64 lowercase hex"),
    ("nonce length", {"hello": _hello(NONCE[:-2])}, "64 lowercase hex"),
    ("parameters", {"hello": _hello(reps=3)}, "session parameters are not the verifier's"),
    ("not canonical", {"hello": json.dumps({**OWN, "coins": {**SPEC, "nonce": NONCE}})}, "session parameters"),
    ("record nonce", {"coin_tree": _tree(nonce="11" * 32)}, "coin_tree.nonce is not Hello's"),
    ("key bit 2047", {"coin_tree": _tree(key=KEY[:-2] + "a5")}, "bit 2047"),
    ("key case", {"coin_tree": _tree(key=KEY.upper())}, "coin_tree.key is not"),
    ("root length", {"coin_tree": _tree(root=ROOT[:-2])}, "coin_tree.root"),
    ("spec", {"coin_tree": _tree(spec={**SPEC, "block": 4})}, "coin_tree.spec"),
    ("extra field", {"coin_tree": _tree(answer="00")}, "{key, nonce, root, spec}"),
    ("no coin_tree", {"coin_tree": None}, "{key, nonce, root, spec}"),
])
def test_a_session_that_does_not_check_is_refused(tmp_path, name, session, error):
    (r,) = _check(tmp_path, [{"hello": _hello(), "coin_tree": _tree(), **session}])
    assert not r["ok"] and error in r["error"], (name, r)


def test_a_session_key_is_never_reused(tmp_path):
    other = KEY[:-4] + "0000"
    rs = _check(tmp_path, [{"hello": _hello(), "coin_tree": _tree()}, {"hello": _hello(), "coin_tree": _tree(key=other)},
                           {"hello": _hello(), "coin_tree": _tree()}])
    assert [r["ok"] for r in rs] == [True, True, False] and "never reused" in rs[2]["error"], rs
