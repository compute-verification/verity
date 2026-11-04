"""scan_tie's statements of the egress tie (`scan_service.EgressTie`) at the certifier tests' class: one registration a
round, every proof block sampled, the four statements (``first``, ``ingress`` with the phase's `IngressScanBucket`
blocks but the last, ``last``, ``egress``) staged on the verifier's own registered reads, every output the row its read
opens, but the tampered round's egress ``o1`` at the one bucket whose token the re-signed phase changed, and the
honest round's on committed outputs one token off the egress at the buckets they don't render; and Lean's
statement setup (`flock-verify statement`, the verifier's checks before any proof) accepts every statement but the
tampered egress, which it refuses at that port, and refuses an ingress statement whose S_t chain is broken in its public
file. Nothing here proves (`scan_tie.py prove` does, on a node)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tests"))
from test_warden_zk import needs_lean, statement  # noqa: E402

from verity.ml.commitments import hm96  # noqa: E402
from verity.ml.commitments.indexed import RangeIndexedDomain  # noqa: E402
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels  # noqa: E402
from verity.ml.commitments.rowleaf import row_leaf  # noqa: E402


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_scan_tie", HERE / "scan_tie.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


st = _load()
SC, SZ, WZ = st.SC, st.SZ, st.WZ


@pytest.fixture(scope="module")
def rounds(tmp_path_factory):
    """Both rounds staged as `scan_tie.prove --dry` stages them (lowering the three templates takes about 30 s)."""
    out = tmp_path_factory.mktemp("scan-tie")
    t = st.tie_of(out, SZ.TEST)
    M = WZ._statement()
    sgs = st.stagers(M, t.tie, out / "stagers")
    return t, {name: st.stage_round(M, t.tie, t.at, t.tr, sgs, name, out / name, 2.0 ** -20) for name in st.ROUNDS}


@pytest.mark.slow
def test_each_round_is_one_registration_every_block_sampled_each_statement_as_its_verdict_needs(rounds):
    """The Spec's 2T proof blocks in four statements, each sampled; honest's every output the row its read opens and
    tampered's off only at the egress o1 of the bucket `_tamper` changed."""
    t, got = rounds
    tie, T = t.tie, t.tie.T
    assert [(m.name, m.base, len(m.units)) for m in tie.members] == [("first", 0, 1), ("ingress", 1, T - 2),
                                                                       ("last", T - 1, 1), ("egress", T, T)]
    assert tie.N == 2 * T
    for name, r in got.items():
        assert r.draw.units == list(range(tie.N)), name
    assert all(not m["off"] for m in got["honest"].staged.values())
    tampered = got["tampered"].staged
    assert tampered["egress"]["off"] == {"o1": [T + t.at.tampered.bucket]}
    assert not any(tampered[k]["off"] for k in ("first", "ingress", "last"))
    assert got["honest"].heads.ingress == got["tampered"].heads.ingress
    assert got["honest"].heads.egress.root != got["tampered"].heads.egress.root


@pytest.mark.slow
def test_the_honest_round_on_committed_outputs_one_token_off_is_off_only_at_the_egress_o1(rounds, tmp_path):
    """With the compute's committed outputs one id other than the network certifier's egress carries (``--outputs``, a
    relay that rewrote a token), the trace renders the committed outputs: the honest round is expected refused at the
    egress statement (`expected`), and its staging is off only at the egress o1 of the buckets they don't render."""
    t, _ = rounds
    q, xs = next((q, xs) for q, xs in sorted(t.at.outputs.items()) if xs)
    off = st.tie_of(tmp_path, SZ.TEST, outputs={**t.at.outputs, q: (xs[0] ^ 1, *xs[1:])})
    assert off.at.unrendered and st.expected(off.at, "honest") == (False, ("egress",))
    M = WZ._statement()
    got = st.stage_round(M, off.tie, off.at, off.tr, st.stagers(M, off.tie, tmp_path / "stagers"), "honest",
                         tmp_path / "honest", 2.0 ** -20).staged
    assert got["egress"]["off"] == {"o1": [off.tie.T + b for b in off.at.unrendered]}
    assert not any(got[k]["off"] for k in ("first", "ingress", "last"))


@pytest.mark.slow
@needs_lean
def test_lean_accepts_every_statement_but_the_tampered_egress_refused_at_its_o1(rounds, tmp_path):
    """Lean's statement setup, `Registered.checkComplete` with it, on each staged statement under the verifier's own
    registered file: what the host reads above only mirrors Lean, and a staging that drifts from the verifier's reads of
    the network certifier's rows, the trace's trees or the prover's ``scan/view`` and ``scan/last``, or a tamper Lean
    doesn't refuse where the Spec says, fails here and in no cheaper test."""
    t, got = rounds
    for name, r in got.items():
        for k, m in r.staged.items():
            rc, line = statement(m, tmp_path)
            if name == "tampered" and k == "egress":
                assert rc == 1 and line.startswith(f"setup: registered reads: port o1: table row "
                                                   f"{t.at.tampered.bucket} does not open"), line
                continue
            assert rc == 0, (name, k, line)
            reads = json.loads(m["registered_file"].read_text())["ports"]
            assert {p: f["root"] for p, f in json.loads(line)["registered"].items()} == \
                {p: e["root"] for p, e in reads.items()}, (name, k)




def _tables(m: dict) -> tuple[bytes, bytes, dict[str, int], dict[str, list[int]]]:
    """Member m's staged public file as its header line, its body, each port's first table row and each instance's ref
    into the port's table: the tables port-major in the circuit's port order (its META ``ports``), then each instance's
    refs (u32 little-endian, instance-major)."""
    with open(m["dir"] / "circuit.txt", "rb") as f:
        f.readline()
        names = [p[0] for p in json.loads(f.readline().decode().removeprefix("META "))["ports"]]
    head, body = (m["dir"] / f"pub-{m['n']}.bin").read_bytes().split(b"\n", 1)
    header = json.loads(head)
    n, shared, k = int(header["instances"]), header["shared_rows"], len(names)
    at = sum(shared[p] for p in names) * hm96.SHA512.proven_bytes
    start = {p: sum(shared[x] for x in names[:q]) for q, p in enumerate(names)}
    refs = {p: [int.from_bytes(body[at + 4 * (i * k + q):at + 4 * (i * k + q + 1)], "little") for i in range(n)]
            for q, p in enumerate(names)}
    return head, body, start, refs


def _rerooted(head: bytes, body: bytes, start: dict[str, int], port: str) -> bytes:
    """The header line with `port`'s frame-v3-sha512 root over the body's table rows, as the prover that wrote the
    public file computes it."""
    fv, count = (h := json.loads(head))["frame_v3"], h["shared_rows"][port]
    dom = CommitmentDomain(bytes.fromhex(fv["bindings"][port]), fv["owner"], RangeIndexedDomain(0, count), hash="sha512")
    assert dom.domain_id.hex() == fv["domain_ids"][port]
    bc, inst = hm96.SHA512.proven_bytes, hm96.SHA512
    leaves = [row_leaf(dom, r, inst.tree_leaf(inst.default_key, body[(start[port] + r) * bc:(start[port] + r + 1) * bc]),
                       fv["schemas"][port]) for r in range(count)]
    old, new = fv["roots"][port].encode(), tree_levels(dom, leaves)[-1][0].hex().encode()
    assert head.count(old) == 1
    return head.replace(old, new)


@pytest.mark.slow
@needs_lean
def test_lean_refuses_an_ingress_statement_whose_s_t_chain_is_broken_in_its_public_file(rounds, tmp_path):
    """The honest ingress statement's public file with block t's ``o1`` (S_t, which reads state leaf t + 1) replaced by
    its ``sin`` (S_{t−1}, state leaf t), a block that claims the state it read as the state it computed, and the
    header's ``o1`` root recomputed over the edited rows, as a prover that writes its own public file would: Lean's
    statement setup refuses it at that ``o1`` row's registered read, so the S_t chain is the registered reads' alone.
    Honestly staged, each block's ``o1`` is the next block's ``sin``, the one leaf both read."""
    m = rounds[1]["honest"].staged["ingress"]
    head, body, start, refs = _tables(m)
    bc, n = hm96.SHA512.proven_bytes, m["n"]

    def at(port: str, i: int) -> int:
        return (start[port] + refs[port][i]) * bc

    def bcs(port: str, i: int) -> bytes:
        return body[at(port, i):at(port, i) + bc]

    assert all(bcs("o1", i) == bcs("sin", i + 1) for i in range(n - 1))
    i = n // 2
    broken = tmp_path / "chain-break"
    broken.mkdir()
    os.symlink(m["dir"] / "circuit.txt", broken / "circuit.txt")
    edited = body[:at("o1", i)] + bcs("sin", i) + body[at("o1", i) + bc:]
    (broken / f"pub-{n}.bin").write_bytes(_rerooted(head, edited, start, "o1") + b"\n" + edited)
    rc, line = statement({**m, "dir": broken}, tmp_path)
    assert rc == 1 and line.startswith(f"setup: registered reads: port o1: table row {refs['o1'][i]} does not "
                                       "open"), line
