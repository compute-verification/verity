"""scan_zk.py --v3c: the scan's state and slot entries' trees are the ones V3c's window committed before its coins, at
that registration's roots and domains, so the honest round's ``scan/state`` and ``scan/prog<j>`` are the roots the
consistency check (`NetworkCheck.Period`'s rows form: `stateTied`, `wired`) holds to V3c's audit, and no block round
is proved. The window here is `prove_path.py window-commit`'s layout with roots made by the implementations the window
commit uses (the `SCAN_STATE` by `registered.Registered`, each `SelectEntry_v1` chunk's o0 by
`windows.rows.commit_ports`), so a held tree's root is theirs; a window whose rows, roots or chunks aren't the scan's
is refused by name."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from test_scan_zk import sz

from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.windows import rows as RWS

S = sz.W63K4
V2 = sz.REG.SCHEMA_V2
RUN, PROGRAM = "ab" * 64, "cd" * 64
TID = f"{sz.SELECT_ENTRY}{{KO={S.cls.KO}}}"
#: a SelectEntry_v1 member's ports: i0 wired (not in its body), i1 in its body before o0
PORTS = (("i0", 3), ("i1", 2))


def _salts(n: int) -> list[bytes]:
    return [os.urandom(sz.ADV.INST.salt_bytes) for _ in range(n)]


def v3c_window(d: Path, states: list[list[int]], entries: list[list[list[int]]]) -> Path:
    """A V3c window directory as `scan_zk.v3c_trees` reads it, committing ``states`` as its `SCAN_STATE` and lane j's
    ``entries`` as its `SelectEntry_v1` chunk j's o0."""
    d.mkdir(parents=True)
    words = [sz._words(r) for r in states]
    st = sz.REG.Registered.commit(PROGRAM, sz.IS.SCAN_STATE, words, schema=V2, window=RUN)
    (d / sz.SCAN_STATE_FILE).write_bytes(np.asarray(words, "<u2").tobytes() + b"".join(st.salts))
    members, roots = [{"k": 0, "name": "ScanSlots_v2{x}/c0", "template": "ScanSlots_v2{x}", "n": 1}], []
    for j, lane in enumerate(entries):
        n, w0 = len(lane), sz.IS.entry_bits(S.cls) // 16
        o0, i1 = np.asarray([sz._words(e) for e in lane], "<u2"), np.arange(2 * n, dtype="<u2").reshape(n, 2)
        ys = {"i1": _salts(n), "o0": _salts(n)}
        dom = CommitmentDomain(hashlib.sha512(b"o0 %d" % j).digest(), RWS.OWNER, RangeIndexedDomain(0, n),
                               hash="sha512")
        got, _ = RWS.commit_ports({"o0": dom}, {"o0": o0}, {}, {"o0": b"".join(ys["o0"])}, schemas={"o0": V2})
        k, name = 1 + j, f"{TID}/c{j}"
        body = np.concatenate([i1, o0], axis=1).astype("<u2").tobytes()
        (d / f"m{k}-private.bin").write_bytes(body + b"".join(ys["i1"][i] + ys["o0"][i] for i in range(n)))
        members.append({"k": k, "name": name, "template": TID, "n": n, "ports": [*map(list, PORTS), ["o0", w0]],
                        "wired_rows": ["i0"], "frame_v3": {
                            "roots": {"o0": got["o0"].hex()}, "domain_ids": {"o0": dom.domain_id.hex()},
                            "bindings": {"o0": dom.binding.hex()}, "owner": RWS.OWNER, "schemas": {"o0": V2}}})
        roots.append({"port": f"{name}/o0", "root": got["o0"].hex(), "leaves": n, "domain": dom.domain_id.hex()})
    (d / "window.json").write_text(json.dumps({"members": members}))
    (d / "registration.json").write_text(json.dumps({"roots": roots, "window": {
        "kind": "served", "run": RUN, "activations": {"program": PROGRAM, "values": {sz.IS.SCAN_STATE: st.entry()}}}}))
    return d


@pytest.fixture(scope="module")
def scanned(tmp_path_factory):
    """The synthetic w63k4 run's scan, its state rows and lane slot entries, and a V3c window committing them."""
    d = tmp_path_factory.mktemp("v3c")
    at = sz.link(d / "plain", setup=S)
    states = sz.IS.window_state(S.cls, at.chain.states[1:], 0, at.T)
    entries = sz.slot_entries(S.cls, at.chain.progs)
    return {"states": states, "entries": entries, "window": v3c_window(d / "window", states, entries),
            "rounds": set(sz.rounds_of(at))}


def test_the_scan_reads_v3cs_trees_at_its_roots_and_proves_no_block_round(tmp_path, scanned):
    window = scanned["window"]
    rec = json.loads((window / "registration.json").read_text())
    state, roots = rec["window"]["activations"]["values"][sz.IS.SCAN_STATE], {r["port"]: r for r in rec["roots"]}
    h = sz.host(tmp_path / "host", setup=S, v3c=window)
    assert set(h.rounds) == scanned["rounds"] - {"block"} and "block" in scanned["rounds"]
    trees = h.at.trees
    assert all(isinstance(trees[k], sz.Held) for k in ("state", *sz.lane_kinds(S.cls)))
    assert (trees["state"].root.hex(), sz.tree_domain(trees["state"]).domain_id.hex()) == (state["root"],
                                                                                          state["domain"])
    for j, p in enumerate(sz.lane_kinds(S.cls)):
        r = roots[f"{TID}/c{j}/o0"]
        assert (trees[p].root.hex(), sz.tree_domain(trees[p]).domain_id.hex(), len(trees[p].rows)) == (
            r["root"], r["domain"], r["leaves"])
    honest = h.rounds["honest"]
    assert all(m.refused is None for m in honest) and not sz.wired(honest)
    port = sz.tree_port("state", trees["state"], [0, 1], 0)
    entry = sz.tree_entry("state", trees["state"], [0, 1])
    assert sz.tree_refusal(port, trees["state"], entry, [sz.tree_bc(trees["state"], i) for i in (0, 1)]) is None
    out = tmp_path / "dry"
    assert sz.main(["prove", "--dry", "--out", str(out), "--class", S.name, "--v3c", str(window)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["v3c"] == {"window": str(window), "trees": sorted(["state", *sz.lane_kinds(S.cls)])}
    assert summary["roots"]["state"] == state["root"] and "block" not in summary["members"]


def _rewrite(window: Path, f: str, change) -> None:
    doc = json.loads((window / f).read_text())
    change(doc)
    (window / f).write_text(json.dumps(doc))


@pytest.mark.parametrize("case", ["state row", "state root", "entry row", "entry root", "chunks", "body"])
def test_a_v3c_window_that_isnt_the_scans_is_refused_by_name(tmp_path, scanned, case):
    states, entries = [list(r) for r in scanned["states"]], [[list(e) for e in lane] for lane in scanned["entries"]]
    if case == "state row":
        states[3][0] ^= 1
    if case == "entry row":
        entries[1][1][-1] ^= 1
    w = v3c_window(tmp_path / "window", states, entries)
    if case == "state root":
        _rewrite(w, "registration.json",
                 lambda d: d["window"]["activations"]["values"][sz.IS.SCAN_STATE].update(root="ee" * 64))
    if case == "entry root":
        _rewrite(w, "registration.json", lambda d: d["roots"][2].update(root="ee" * 64))
    if case == "chunks":
        _rewrite(w, "window.json", lambda d: d["members"].pop())
    if case == "body":
        (w / "m2-private.bin").write_bytes((w / "m2-private.bin").read_bytes()[:-1])
    why = {"state row": f"its {sz.IS.SCAN_STATE} rows [3] aren't the scan's",
           "state root": f"isn't its registered {sz.IS.SCAN_STATE}'s root and domain",
           "entry row": f"{TID}/c1's o0 rows [1] aren't the scan's lane 1 slot entries",
           "entry root": f"isn't the root and domain its registration names for {TID}/c2/o0",
           "chunks": "not one template's 4 chunks of 2 proof blocks, a lane each",
           "body": "m2-private.bin is"}[case]
    with pytest.raises(SystemExit) as e:
        sz.v3c_trees(w, S.cls, scanned["states"], scanned["entries"])
    assert str(e.value).startswith(f"--v3c {w}: ") and why in str(e.value), str(e.value)


def test_v3c_is_one_link_window_from_link_window_0(tmp_path, scanned):
    with pytest.raises(SystemExit, match="--v3c: V3c's window is link-window 0's, a group of 1, not 2 from 0"):
        sz.link(tmp_path, setup=S, group=2, v3c=scanned["window"])

