"""n2_advised_zk's period draws on the auditor's challenger's coins (`ChallengerCoins`, ``--challenger local``): one local
session for every member, under live_challenger's stub of the Lean verifier's coins (as `test_a5_reserve` runs it), so
nothing builds `sampled-proofs-verifier` and nothing is proved. Each member's draw is `LinkWindows.period_draw`'s on the
session's coins: committed to its period's commitment, then `subset:K` read, then its turn ended (`drawn`)."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from verity.core.protocols.sampled_proofs import service as S

HERE = Path(__file__).resolve().parents[1]
TREE = HERE.parents[1]
STUB = TREE / "benchmarks" / "live_challenger" / "tests" / "stub_serve.py"


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_n2_advised_zk", HERE / "n2_advised_zk.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


n2 = _load()
N, K = 26, 8


def _root(name: str, w: int) -> bytes:
    return hashlib.sha512(f"{name}/{w}".encode()).digest()


def _session(tmp_path: Path, monkeypatch, names: list[str]):
    sys.path.append(str(TREE / "benchmarks" / "one_stage"))
    import a5_coins as AC

    A = AC._module("a5_auditor", AC.TREE / "benchmarks" / "live_challenger" / "auditor.py")
    real = A.serve_argv
    monkeypatch.setattr(A, "serve_argv", lambda *a, **k: [sys.executable, str(STUB), *real(*a, **k)[2:]])
    monkeypatch.setenv("COINS_LOG", str(tmp_path / "coins.log"))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    main = tmp_path / "main.txt"
    main.write_text(AC.on_main([]) + "\n")
    a = SimpleNamespace(challenger="local", main=str(main), gate_key=None, peer=None, challenger_key=None, draw=K)
    rounds = {n: SimpleNamespace(windows=[SimpleNamespace(root=_root(n, w)) for w in range(N)], advice=None)
              for n in names}
    return n2.ChallengerCoins(a, rounds, tmp_path / "challenger"), rounds


def _draw(ch, name: str, rounds) -> dict:
    roots = [x.root.hex() for x in rounds[name].windows]
    commitment = hashlib.sha512(("period/" + name).encode()).hexdigest()
    coins, read = ch.ranks[name], bytearray()
    coins.commit(bytes.fromhex(commitment))
    rule = S.Law(f"subset:{K}")
    draw = S.select_draw(rule, S.selection_law(rule, N), N, lambda m: read.extend(b := coins(m)) or b)
    return {"period": {"roots": roots, "advice": None, "k": K, "commitment": commitment, "units": list(draw["units"]),
                       "read": read.hex()},
            "segment": coins.drawn()}


def test_every_members_draw_holds_on_the_one_session_and_a_draw_after_another_commitment_does_not(tmp_path, monkeypatch):
    """Both members plan before the session opens (26 windows, `subset:8`), each reads its coins in its own turn after its
    period's commitment, and each draw holds against the session's evidence (`coins.coins_reason`) on the root the
    verifier recomputes from what it received, every `a5_coins.negatives` refused, and the coins the period records it
    read (``read``, which the network verdict's Lean draws on) are the ones the session released. The same coins held to
    another commitment, or to a root over a period whose roots changed, and a period recording other coins than those
    released, are refused."""
    names = ["honest", "tampered_all"]
    ch, rounds = _session(tmp_path, monkeypatch, names)
    try:
        done = {n: _draw(ch, n, rounds) for n in names}
    finally:
        ch.close()
    for n in names:
        (tmp_path / "out" / n / "verifier").mkdir(parents=True)
    got = ch.check(done, tmp_path / "out")
    assert (tmp_path / "coins.log").read_text(), "the session's coins came from the stub, not a Lean build"
    assert ch.published.get("error") is None, ch.published.get("error")
    for n in names:
        assert got[n]["ok"] and got[n]["root_ok"] and got[n]["coins_reason"] is None, got[n]
        assert got[n]["read_is_released"] and got[n]["read_bytes"] == len(done[n]["period"]["read"]) // 2 > 0
        assert got[n]["negatives_refused"] and all(got[n]["negatives_refused"].values()), got[n]["negatives_refused"]
        assert len(done[n]["period"]["units"]) == K and (tmp_path / "out" / n / "verifier" / "coins.json").exists()
    assert ch.published["plan"]["needs"] == {"0": n2.coin_need(K, N, 64), "1": n2.coin_need(K, N, 64)}

    read = done["honest"]["period"]["read"]
    flipped = {**done, "honest": {**done["honest"], "period": {**done["honest"]["period"],
                                                               "read": f"{int(read[:2], 16) ^ 1:02x}" + read[2:]}}}
    refused = ch.check(flipped, tmp_path / "out")["honest"]
    assert not refused["ok"] and not refused["read_is_released"], refused

    other = {**done, "honest": {**done["honest"], "period": {**done["honest"]["period"], "commitment": "ee" * 64}}}
    refused = ch.check(other, tmp_path / "out")["honest"]
    assert not refused["ok"] and refused["coins_reason"], refused
    moved = {**done, "tampered_all": {**done["tampered_all"], "period": {
        **done["tampered_all"]["period"], "roots": ["dd" * 64] + done["tampered_all"]["period"]["roots"][1:]}}}
    assert not any(v["ok"] or v["root_ok"] for v in ch.check(moved, tmp_path / "out").values())


def test_a_draw_that_never_ran_leaves_its_member_refused_and_the_session_closed(tmp_path, monkeypatch):
    """A member whose draw raised before its commitment (its turn never taken) is skipped when the session closes: the
    other member's draw still holds, and the one with no segment is refused by name."""
    names = ["honest", "tampered_all"]
    ch, rounds = _session(tmp_path, monkeypatch, names)
    try:
        done = {"honest": _draw(ch, "honest", rounds)}
    finally:
        ch.close()
    roots = [x.root.hex() for x in rounds["tampered_all"].windows]
    done["tampered_all"] = {"period": {"roots": roots, "advice": None, "k": K, "commitment": "aa" * 64, "read": ""}}
    for n in names:
        (tmp_path / "out" / n / "verifier").mkdir(parents=True)
    got = ch.check(done, tmp_path / "out")
    assert got["honest"]["ok"], got["honest"]
    assert not got["tampered_all"]["ok"] and got["tampered_all"]["coins_reason"] == "the draw released no segment"
