"""With ``FC_FIREWALL_VALUES`` unset, every path the value service's switch touches commits and writes what it wrote without the
switch, byte for byte: the SHA-512 of each output is pinned, with the operating system's salts drawn from a counter. The served
rows' record, ``b ‖ c`` and window file (`serve_commit.commit_served`, `CommitWindows._served_rows`), `registered_rows.commit`'s
entries and salts, a statement staged under a salt key (`UnitStager.stage`, through `circuit.write`, per instance and with
shared rows), and `predict node-registration`'s record, trusted input, provenance, salts files and summary."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from verity.ml.circuits.boolean import BIT
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import bind
from verity.ml.circuits.types import Array
from verity.ml.commitments import hm96
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock import class_statement as CS
from verity_vllm.commit import registered_rows as RR
from verity_vllm.pipeline import serve_commit as sc
from verity_vllm.program.predict import node_registration as NR

import tests.program.predict.test_node_registration as TNR
from tests.program.predict.test_node_registration import commit  # noqa: F401  -- fixture

ENV = "FC_FIREWALL_VALUES"
ENGINE = {"max_num_seqs": 8, "max_num_batched_tokens": 2048, "max_model_len": 512, "num_gpu_blocks": 64, "block_size": 16}
UNIT = bind(definition("UnsetPinsUnit", 1, lambda C, a, b, w: [C.AND(a[k], b[k]) ^ C.AND(w[k], w[2047 - k]) for k in range(16)],
                       (("a", Array(16, BIT)), ("b", Array(16, BIT)), ("w", Array(2048, BIT))), Array(16, BIT), register=False))

#: the SHA-512 of what each path writes without the switch
PINS = {
    "served_rows": {
        "record": "2c3e25f3ec17e917348c2f9920d252dba1167d7585ceea92a4993fb2dab754b875a11dac522b7303108d1c6b127ec4bacc51dc795e791d73c60d6db49238630d",
        "bcs": "d45ad1e8605fc781a10f7f37dcdeedacecb950ee148da70de50423bdee6e7327cd02f51ca34b313a043904c4771d31d6356fc75111983a1d0996d9016501f7d4",
        "line": "e017b0fe69a6c0d2a36415961ed298cb961403a07b38daa4fe3463c8f63db4c4b9b97e69ffefcd720b15700fcbf881ea214f6ed21fc2d467009c0b1c5526deb0",
        "file": "3de20f30092f84b4c8eaa5b9c18cca9487e8b0c18d49bc34161274185604dc5331e3238ed72ee0ca4579f5e1cb7ada339c54280f395f1695fc9874f3e3470228",
    },
    "registered_rows": {
        "u16": "f8206e062fd040e3ff8732da677d391038d8151763357ec6fda3d951e1821c7f223b05fd9eda73a0a2c86d91185902dfa822415d6ad052535b4f6a273438ed2d",
        "u32": "5ce859e8bbf88f1fdb2750647db1b44e7ce9da59591a4ee89dbf311738823b66c40716ef4e3a2f0d63637971e2c5f8d90592f073cf89a2368f762c3ebd0465b3",
    },
    "staged": {
        "per-instance": {
            "inst.bin": "ba8945cdbbfd172e2bb1427afcc312bc2fc3bbb7d1c6dce5f6a8f739e434fae85aac286c246c213ae7929022b9da80a624597ba5ea802dbf3bcc906699350c03",
            "pub.bin": "b0510e69ba7eccd796166a288f0ffaa54eaca2bc9eec06bfd31d3ae034f79de4f86e9a290b8ab84f38b97af9d66fbba7f08e37bd0700f9db386f8d5dd9ec1879",
        },
        "shared-rows": {
            "inst.bin": "574a24588f672f6a69a47306005d084af02fd39d5c8cdfac15f3276776c0c97612031862f6e884c26e543737754608016685483ca781cdb5bef99ca037dfe89f",
            "pub.bin": "78b7932954b1b7b7a9cf5dc3d9130e5577dbbcb72c772aac80ea6f39ebf97a268757d42132ef7773d049616558fcae1ee636e6e9f392676cb9af8bbd5d4b0407",
        },
    },
    "node_registration": {
        "record": "1acd11d55f6d65f937f71fe7cfb9ee6c637f294c24165a1f60d5c6801f5ca0e6cfe2419aa68ee8653640c7f9ac0f9600c27ff070169f2320b563936bd2d06504",
        "trusted": "bd0631af4a73b7ad004e3090342f5ba1a6d900043c1e1b8ba0597dc1326262336a8be46831701901cc28e6e9538929db899b345b3594ff319270f14fd002f2d3",
        "provenance": "5b7390a72a0f8a55b7138d71194b3b6d9ec0ca7fd9e57e402a679c0285f66786837bbbf68893ea67939630326b9c163d46c15357ac06eeafd2561835f05ca9ea",
        "salts": "ba9df6c3501d4ac3a80a4bd44633414bd9c0cb33a18e53af7b7f71e345e395d439d26b33f41c8a369832008da76e9199a7ef4219b95b9ad7640700c6166f3108",
        "registration": "95517ea2aecab70d660b6ed198da7bfa5486a4d0d27c02f585a27305321b399faee043152ea13f8cd8376927d15ccd01dd2902812bf2387d03c0f1b53ae75044",
        "keys": ["schema", "program_digest", "record_digest", "row", "wrapper", "shape", "commit_out", "pair",
                 "window", "theta_parameters", "theta_nodes", "rows", "checkpoint", "ranks", "salts"],
    },
}


def _sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


def _json(x) -> bytes:
    return json.dumps(x, sort_keys=True).encode()


@pytest.fixture
def counted(monkeypatch):
    """The switch unset, and every fresh salt (`hm96.Instance.fresh_salt`, `registered_rows`' OS bytes) drawn from a counter."""
    monkeypatch.delenv(ENV, raising=False)
    n = itertools.count()

    def draw(k: int) -> bytes:
        out = b""
        while len(out) < k:
            out += hashlib.sha512(b"unset-pins" + next(n).to_bytes(8, "big")).digest()
        return out[:k]
    monkeypatch.setattr(hm96.Instance, "fresh_salt", lambda self: draw(self.salt_bytes))
    monkeypatch.setattr(RR, "os", SimpleNamespace(urandom=draw))


def test_served_rows_commit_and_write_what_they_did(counted, tmp_path, monkeypatch):
    statics = sc.served_statics(ENGINE, {**dict.fromkeys(sc.SERVED_CLASS, 1), "R": 8, "P": 3, "IW": 6})
    by_step = {0: [("a-0", 0, 4, 0, 4), ("b-0", 4, 7, 0, 3)], 1: [("a-0", 0, 1, 4, 4), ("b-0", 1, 2, 3, 3)], 3: [("a-0", 0, 1, 5, 4)]}
    run_root = hashlib.sha512(b"unset-pins/run").digest()
    served, _session = sc.commit_served(statics, by_step, {"a-0": 0, "b-0": 1}, 5, run_root)
    rec = served.record()
    monkeypatch.setattr(sc, "served_rows", lambda com: by_step)
    me = SimpleNamespace(served=statics, arrivals={"a-0": 0, "b-0": 1}, known=None, out=tmp_path)
    line = sc.CommitWindows._served_rows(me, {"window": 2, "steps": 5, "run_root": run_root.hex()}, None)
    got = {"record": _sha(_json(rec)), "bcs": _sha(b"".join(served.bc(i) for i in range(rec["count"]))), "line": _sha(_json(line)),
           "file": _sha((tmp_path / "served_rows" / "w2.json").read_bytes())}
    assert got == PINS.get("served_rows"), json.dumps(got)


def test_registered_rows_commit_what_they_did(counted):
    rng = np.random.default_rng(11)
    got = {}
    for name, leaves, bits in (("u16", rng.integers(0, 1 << 16, 4096, dtype=np.uint16), 16),
                               ("u32", rng.integers(0, 1 << 32, 1536, dtype=np.uint32), 32)):
        c = RR.commit("ab" * 64, f"pins.{name}", leaves, bits)
        got[name] = _sha(_json(c.entry) + c.salts)
    assert got == PINS.get("registered_rows"), json.dumps(got)


@pytest.mark.slow
def test_a_statement_stages_under_its_salt_key_as_it_did(counted, tmp_path):
    st = CS.UnitStager(UNIT, "unit/unset-pins", [("ab", V2, 32, ["a", "b"]), ("w", V2, 2048, ["w"])], [(V2, (0,))])
    rng = np.random.default_rng(13)
    ab, w = rng.integers(0, 1 << 16, (5, 2), dtype=np.uint16), rng.integers(0, 1 << 16, (5, 128), dtype=np.uint16)
    table, refs = rng.integers(0, 1 << 16, (3, 2), dtype=np.uint16), np.array([2, 0, 1, 1, 2])
    got = {}
    for mode, rows, kw in (("per-instance", {"ab": ab, "w": w}, {}),
                           ("shared-rows", {"ab": table[refs], "w": w}, {"tables": {"ab": table, "w": w}, "refs": {"ab": refs, "w": np.arange(5)}})):
        st.stage(rows, tmp_path / mode, b"k" * 32, **kw)
        got[mode] = {f: _sha((tmp_path / mode / f).read_bytes()) for f in ("inst.bin", "pub.bin")}
    assert got == PINS.get("staged"), json.dumps(got)


@pytest.fixture
def fresh(monkeypatch):
    """`test_node_registration`'s draws from a generator of their own, so the bundle's tensors are the same in any test order."""
    monkeypatch.setattr(TNR, "RNG", np.random.default_rng(17))


def test_node_registration_writes_what_it_did(counted, fresh, commit, tmp_path):
    o = NR.Options(**{**commit["o"].__dict__, "checkpoints_manifest": TNR._manifest(commit)})
    assert NR.main(o) == 0
    log, salts = Path(o.out), Path(o.salts)
    doc = json.loads((log / "registration.json").read_text())
    doc.pop("timings")
    for rk in doc["ranks"]:
        del rk["wall_s"], rk["manifest_sha256"]     # the bundle's manifest hashes the fixture's paths
    got = {**{n: _sha((log / f"{n}.json").read_bytes()) for n in ("record", "trusted", "provenance")},
           "salts": _sha(b"".join(p.name.encode() + p.read_bytes() for p in sorted(salts.glob("*.salts")))),
           "registration": _sha(_json(doc).replace(str(tmp_path).encode(), b"<tmp>")), "keys": list(doc)}
    assert got == PINS.get("node_registration"), json.dumps(got)
