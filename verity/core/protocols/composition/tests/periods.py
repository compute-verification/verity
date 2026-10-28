"""Onsite's real components' cases for the composition's difftest, each with the answer its Python verifier gives it
(`reference.Recorded`), all on one store: the committed 9-block `C*` of the remote bandwidth audit's fixture
(`remote_bandwidth/tests/fixtures/p2-store.json`), registered for PoUB-P2 on its PoUS setup. Space's
(`pous.period.audit` of that `C*` on node `n0`, signed by a key at S1's deployed parameters, `mc_cases.deployed_signer`)
honest, under a prover that answers every other round with zeros, the honest audit's case committed to a setup with
another salt (refused by the setup tie), and an honest run on another store (`OTHER_STORE`, accepted, which onsite's
store tie refuses beside PoUB-P2's case); PoUB-P2's (`p2_case`, its sessions over a socket pair against
`remote-audit-verifier`, as `remote_bandwidth/lean/tests/test_p2_party.py` runs them) honest, under a prover whose
receipts skip a unit's reads, and the honest session's log stopped before its last window is verified; and of each, its
honest case with params or evidence that don't decode. A P2 case is recorded as the composition hands it, without its
`store_root`: onsite reads the store off its commitment's registration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import reference as R
from verity_difftest import protocol as P
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period as pous
from verity.core.protocols.pous import run as RUN
from verity.core.protocols.pous import setup
from verity.core.protocols.pous.schemes import p2 as P2
from verity.core.protocols.remote_bandwidth import p2, p2_case, p2_lean
from verity.ml.commitments.identity import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[5]
BANDWIDTH = ROOT / "verity/core/protocols/remote_bandwidth"
POUS_K = 8
"""Rounds of PoUS's audit: the composition reads only the component's answer, so its case is kept small (the hook's
own point is `pous_check`'s difftest's)."""
OTHER_POUS_STORE = hashlib.shake_256(b"another tiny MLP's weights").digest(500)
"""Four blocks of P2's 126 payload bytes at the small point's width: a store PoUS sets up, never the committed `C*`."""
COINS = hashlib.sha512(b"the period's live coins").digest()
OTHER_STORE = "other-store"
"""PoUS's honest run on `OTHER_POUS_STORE`, accepted: the one that meets PoUB-P2's case on two stores."""


def _committed(T) -> p2.Store:
    """The committed `C*` (`fixtures/p2-store.json`), registered for P2 on its PoUS setup at `T`'s node."""
    doc = json.loads((BANDWIDTH / "tests/fixtures/p2-store.json").read_text())
    data = (BANDWIDTH / "tests/fixtures/p2-c-star.bin").read_bytes()
    blocks = [data[p2.BLOCK_BYTES * j:p2.BLOCK_BYTES * (j + 1)] for j in range(len(data) // p2.BLOCK_BYTES)]
    m = doc["params"]
    params = P2.P2Params(m["w"], m["c0"], m["blocks_per_segment"], m["segments"], m["expander"])
    commitment, salt, n = bytes.fromhex(doc["commitment"]), bytes.fromhex(doc["salt"]), params.blocks_per_segment
    vk = setup.verifier_key_from_encoding(params, None, commitment, salt,
                                          [blocks[n * s:n * (s + 1)] for s in range(params.segments)])
    return p2.Store(p2.register(T.SOURCE, T.NODE, commitment, vk), blocks)


def _held(store: p2.Store) -> pous.Store:
    """PoUS's store for the `C*` that P2's `store` registers: its setup published at that `C*`'s commitment, salt and
    root."""
    reg, q = store.registration, store.registration.params
    n = q.blocks_per_segment
    return pous.Store(RUN.Point(q, POUS_K, pous.LIMIT_NS), RUN.Setup(q.blocks, reg.commitment, reg.root, reg.salt),
                      store.blocks, b"".join(mc.block_digest(b) for b in store.blocks),
                      pous.setup_case(q, reg.commitment, reg.salt,
                                      [list(store.blocks[n * s:n * (s + 1)]) for s in range(q.segments)]))


def _with_setup(case: dict, setup_of: dict, trusted: tuple[bytes, ...]) -> tuple[dict, dict]:
    """Space's `case` with the setup case `setup_of`'s commitment and evidence in its evidence, and the answer PoUS's
    Python verifier gives it (`pous.answer` on the case's own run)."""
    ev = json.loads(bytes.fromhex(case["evidence"]))
    ev["setup"] = {"commitment": setup_of["commitment"], "evidence": setup_of["evidence"]}
    p, q = case["params"], case["params"]["point"]["params"]
    point = RUN.Point(P2.P2Params(q["w"], q["c0"], q["blocks_per_segment"], q["segments"]),
                      p["point"]["k"], p["point"]["limit_ns"])
    answer = pous.answer(point, mc.SCHEMES, trusted, RUN.Setup.from_json(json.loads(bytes.fromhex(case["commitment"]))),
                         bytes.fromhex(ev["digests"]), RUN.Session.from_json(ev["session"]),
                         bytes.fromhex(p["session_id"]), setup_inputs=pous.setup_inputs(setup_of))
    return case | {"evidence": canonical_json_bytes(ev).hex()}, answer


def _other_salt(case: dict) -> dict:
    """Space's `case` committed to a setup with another salt, on the same root: a run its setup's tie refuses, and one
    onsite's store tie, which reads the root, doesn't."""
    published = json.loads(bytes.fromhex(case["commitment"]))
    salt = bytes.fromhex(published["salt"])
    published["salt"] = (bytes([salt[0] ^ 1]) + salt[1:]).hex()
    return case | {"commitment": canonical_json_bytes(published).hex()}


def _undecodable(cid: str, case: dict, digest: str) -> dict:
    return {"broken-params": ({**case, "params": {}}, P.outcome(cid, R.REJECT, "params: not its params")),
            "broken-evidence": ({**case, "evidence": "00"},
                                P.outcome(cid, R.REJECT, "evidence: not JSON", bytes.fromhex(digest)))}


def _served(case: dict) -> dict:
    """A P2 case as the composition hands it: its served keys."""
    return {k: case[k] for k in ("commitment", "evidence", "params")}


def build(tmp: Path) -> None:
    """Fill `reference.POUS` and `reference.BANDWIDTH` with their cases, the memory challenger's session logs in
    `tmp`."""
    M = R.module("mc_cases", ROOT / "verity/core/devices/memory_challenger/lean/tests/mc_cases.py")
    T = R.module("p2_party_sessions", BANDWIDTH / "lean/tests/test_p2_party.py")
    signer = M.deployed_signer()
    store = _committed(T)

    def space(name: str, run=None, **kw) -> pous.Run:
        mc.SessionLog.create(tmp / name)
        kw = {"signer": signer, "log": mc.SessionLog(tmp / name), "clock": M.Clock(10 ** 9), **kw}
        if run is not None:
            return pous.run("n0", run, COINS, point=pous.point_for(len(run), pous.W, pous.C0, k=POUS_K), **kw)
        return pous.audit("n0", _held(store), COINS, **kw)

    honest = space("honest")
    zeros = space("zeros", prover=lambda blocks: lambda i: bytes(len(blocks[i])) if i % 2 else blocks[i])
    other = space("other", run=OTHER_POUS_STORE)
    R.POUS.cases.update({R.ACCEPT: (honest.case, honest.answer), R.REJECT: (zeros.case, zeros.answer),
                         "other-setup": _with_setup(_other_salt(honest.case), honest.setup_case, (signer.public_key,)),
                         OTHER_STORE: (other.case, other.answer)}
                        | _undecodable(R.POUS.id, honest.case, honest.answer["commitment"]))
    why = R.POUS.cases["other-setup"][1]["reason"]
    assert why == "the run's commitment is not what its setup published", why
    def root(case: dict) -> bytes:
        return RUN.Setup.from_json(json.loads(bytes.fromhex(case["commitment"]))).root

    assert root(honest.case) == store.registration.root != root(other.case)

    exe = p2_lean.build()

    def reads(**kw) -> tuple[dict, dict]:
        kept = []

        class Keeping(T.Prover):
            def case(self, doc) -> None:
                kept.append(doc)

        proved, answer = T.run(exe, store, lambda c: Keeping(store, c, **kw), T.request(store))
        assert proved == answer and len(kept) == 1, answer
        assert p2_case.same_store(kept[0], honest.case) is None
        return _served(kept[0]), p2_case.answer(kept[0], answer)

    accepted, rejected = reads(), reads(skip=1)
    log = json.loads(bytes.fromhex(accepted[0]["evidence"]))["log"]
    last = max(k for k, x in enumerate(log) if x["op"] == "p2_openings")
    unverified = accepted[0] | {"evidence": canonical_json_bytes({"log": log[:last]}).hex()}
    R.BANDWIDTH.cases.update({R.ACCEPT: accepted, R.REJECT: rejected,
                              R.UNDECIDED: (unverified, p2_case.answer(unverified, {}))}
                             | _undecodable(R.BANDWIDTH.id, accepted[0], accepted[1]["commitment"]))
    for C in (R.POUS, R.BANDWIDTH):
        got = {k: a["verdict"] for k, (_, a) in C.cases.items()}
        want = {OTHER_STORE: R.ACCEPT} | {k: k for k in (R.ACCEPT, R.REJECT, R.UNDECIDED)}
        assert all(v == want.get(k, R.REJECT) for k, v in got.items()), (C.id, got)
