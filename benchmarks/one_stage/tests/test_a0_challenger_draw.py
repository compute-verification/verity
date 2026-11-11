"""a0's Lean sample on a development challenger's coins (`a0.challenger_draw`): the verifier has no randomness of its
own, so `flock-verify draw` of fewer than every unit reads `--stream-file F`, F holding a challenger's coins opened
after the registration's digest. F holds exactly the bytes the sampling design reads in Lean's `Draw.drawOn` layout (no
extension), and the record beside it replays them from the challenger's commitment, under no other registration digest
or challenger. `flock-verify` is a stand-in that echoes its argv and F; the challenger's entropy is seeded."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

from verity.protocols.spot.device import challenger as Ch, coins as CC
from verity.protocols.spot.verifier.one_stage import draw as D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a0

#: a stand-in for flock-verify: ``draw ... --stream-file F`` prints a DRAW line holding its argv and F's bytes
FAKE = '''#!{python}
import json, pathlib, sys
args = sys.argv[1:]
stream = pathlib.Path(args[args.index("--stream-file") + 1]).read_bytes()
print("DRAW " + json.dumps({{"argv": args, "stream": stream.hex()}}))
'''


def stratified(k: int, sizes: list[int]) -> tuple[dict, int]:
    n, at, strata = sum(sizes), 0, []
    for i, size in enumerate(sizes):
        strata.append({"template": f"t{i}", "units": [[at, at + size]], "k": D.stratum_k(k, size, n)})
        at += size
    return {"law": "stratified", "k": k, "strata": strata}, n


LAWS = {"subset": ({"law": "subset", "k": 5}, 40), "bernoulli": ({"law": "bernoulli", "p": "1/3"}, 12),
        "stratified": stratified(6, [7, 20, 3])}


@pytest.fixture
def lean(tmp_path, monkeypatch):
    p = tmp_path / "flock-verify"
    p.write_text(FAKE.format(python=sys.executable))
    p.chmod(0o755)
    monkeypatch.setattr(a0.secrets, "token_bytes", random.Random("a0-challenger-draw").randbytes)
    return str(p)


def exactly(stream: bytes):
    """A source of `stream`'s bytes in order that refuses to read past them, as Lean's `drawOn` does."""
    at = [0]

    def read(n: int) -> bytes:
        if at[0] + n > len(stream):
            raise AssertionError("the draw read past the stream it was given")
        at[0] += n
        return stream[at[0] - n:at[0]]
    return read


@pytest.mark.parametrize("name", sorted(LAWS))
def test_the_draw_reads_a_challengers_coins_opened_after_the_registration(lean, tmp_path, name):
    law, n = LAWS[name]
    after = hashlib.sha512(name.encode()).digest()
    args = ["--population", str(n), "--law", name]
    p = a0.challenger_draw(lean, args, law, n, after.hex(), tmp_path / "draw.log")
    assert p.returncode == 0, p.stderr
    out = json.loads(next(x for x in p.stdout.splitlines() if x.startswith("DRAW ")).split(" ", 1)[1])
    stream_file = tmp_path / "draw.coins.bin"
    assert out["argv"] == ["draw", *args, "--stream-file", str(stream_file)]
    stream = bytes.fromhex(out["stream"])
    assert stream == stream_file.read_bytes()

    record = json.loads((tmp_path / "draw.coins.json").read_text())
    assert record["need"] == len(stream) > 0
    assert record["segment"]["after"] == after.hex()
    replay = CC.challenger_stream(Ch.LiveCommitment.from_json(record["commitment"]), record["segment"], after)
    assert replay[:len(stream)] == stream
    assert len(replay) == -(-len(stream) // Ch.COIN_BYTES) * Ch.COIN_BYTES

    drawn = CC.draw_first(CC.OwnRandomness(exactly(stream)), law, n)
    assert drawn["population"] == n and all(0 <= u < n for u in drawn["units"])


def test_the_registration_digest_may_come_as_bytes(lean, tmp_path):
    law, n = LAWS["subset"]
    after = hashlib.sha512(b"bytes").digest()
    p = a0.challenger_draw(lean, ["--population", str(n), "--k", "5"], law, n, after, tmp_path / "draw.log")
    assert p.returncode == 0, p.stderr
    assert json.loads((tmp_path / "draw.coins.json").read_text())["segment"]["after"] == after.hex()


@pytest.mark.parametrize("other, why", [
    ("registration", "the draw's coins were not opened after the prover's commitment the verifier holds"),
    ("challenger", "coin does not open against the epoch's coin commitment"),
])
def test_the_record_replays_under_no_other_registration_or_challenger(lean, tmp_path, other, why):
    law, n = LAWS["stratified"]
    after = hashlib.sha512(b"registration").digest()
    p = a0.challenger_draw(lean, ["--population", str(n)], law, n, after, tmp_path / "draw.log")
    assert p.returncode == 0, p.stderr
    record = json.loads((tmp_path / "draw.coins.json").read_text())
    commitment = Ch.LiveCommitment.from_json(record["commitment"])
    if other == "registration":
        after = hashlib.sha512(b"another registration").digest()
    else:
        commitment = Ch.LiveChallenger(a0.DEV_SCHEME, 0, commitment.families,
                                       random.Random("another challenger").randbytes).commitment
    with pytest.raises(Ch.CoinError, match=why):
        CC.challenger_stream(commitment, record["segment"], after)
