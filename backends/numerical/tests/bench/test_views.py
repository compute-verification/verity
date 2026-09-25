"""The approved spec's views (``bench.views``): the switch's parity gate against the frozen tables, and the spec's rules --
the full committed relation with SHA-256 / BLAKE3 commitments, end-to-end throughput, the security filter, the computed
headline, ``archived``, ``n/s`` and the provisional marker -- over an isolated store root."""

from __future__ import annotations

import dataclasses
import json
import random
from typing import Any

import pytest

from verity.verification.target import BF16_HOPPER, FIRST, FP8_ADA_MMA, FP8_HOPPER_WGMMA, NVFP4_SM120, TARGETS
from verity_numerical.bench import tables as T
from verity_numerical.bench import views as V

pytest.importorskip("research.store.local", reason="research.store (tools/research) is not importable")
from research.store.local import LocalStore  # noqa: E402
from test_tables import (  # noqa: E402, F401  (store: the fixture)
    A100, H100, LANE, REDTEAM, RTX4090, VERIFIER, attempt, equiv_doc, hashed, put, put_equiv, repacked, result, store,
)

GPU = {FIRST.name: A100, BF16_HOPPER.name: H100, FP8_HOPPER_WGMMA.name: H100, FP8_ADA_MMA.name: RTX4090, NVFP4_SM120.name: "NVIDIA GeForce RTX 5090"}
PROTOCOL = {"warm": True, "runs": 5, "statistic": "median", "contended": False}
SWEEP = {"id": "sweep-1", "point": 4, "plateau": True}


# --- parity: the new views under legacy rules select exactly today's cells ---------------------------------------------------------


def _legacy_rejected(views: V.Views) -> set[str]:
    return {r["artifact"] for r in views.rejected}


def _assert_parity(store: LocalStore) -> None:
    frozen = T.build(store)
    assert V.parity(store, frozen) == []
    legacy = V.build(store, V.LEGACY)
    assert {(c.line, c.column, c.artifact) for c in legacy.cells} == {
        (c.target, c.candidate + T.AUTH_COLUMN_SUFFIX.get(c.authentication, ""), c.artifact) for c in frozen.cells}
    # every result the frozen predicate rejects, the legacy rules reject, and the other way round (valid-but-not-best aside)
    assert _legacy_rejected(legacy) == {r["artifact"] for r in frozen.rejected}
    for c in legacy.cells:
        want = next(f for f in frozen.cells if f.artifact == c.artifact)
        assert T.fmt_overhead(c.overhead) == T.fmt_overhead(want.overhead) and c.overhead == pytest.approx(want.overhead, rel=1e-12)


def test_parity_over_every_branch_of_the_frozen_predicate(store: LocalStore) -> None:
    """The switch's gate 2 (kb/TABLES.md "The switch"): rendered with today's parameters -- 2^-128, B = 4096, K = 1536, the bare
    and + in-proof hash columns, proving time only -- the new views select the frozen tables' cells, artifact and ratio."""

    put(store, result(t_total=4.0))
    put(store, result(t_total=3.0))                                                               # best of two
    put(store, result(t_total=1.0, gpu=H100))                                                     # K: another SKU
    put(store, result(t_total=0.5, proof_class="ARITHMETIC_DIAGNOSTIC"))                          # Z
    put(store, result(t_total=0.6), verified_by=None)                                             # U
    put(store, result(t_total=0.7), verified_by=LANE)                                             # self-verified
    put(store, result(t_total=0.8, security={"target": -100, "achieved_log2": -100}))             # below 2^-128
    put(store, result(t_total=0.9, security={"target": -128, "achieved_log2": -120}))             # achieved above
    put(store, result(t_total=0.91, security={"target": -128, "achieved_log2": "?"}))              # achieved not a number
    put(store, result(t_total=1.1, instances=None))                                               # I
    small = result(t_total=0.2)
    small["workload_fingerprint"].update(B=64, instances={**T.FROZEN_INSTANCES[FIRST.name], "range": [0, 64]})
    put(store, small)                                                                             # legacy B
    put(store, result(t_total=1.2, buckets={"witness": 3.0, "encoding_commitment": 2.0}))         # P
    down = put(store, result(t_total=0.3, proof_class="NON_ZK_PROOF", target=BF16_HOPPER, gpu=H100),
               labels=[("proof_class", "NON_ZK_PROOF_DIAGNOSTIC", REDTEAM, "2026-09-23T00:00:00Z")])  # capped
    assert down
    put(store, result(candidate="B-Ligero", target=FP8_ADA_MMA, gpu=RTX4090, proof_class="COMPLETE_ZK_BACKEND", t_total=1.0))
    put(store, result(candidate="B-Ligero", target=FP8_ADA_MMA, gpu=RTX4090, proof_class="COMPLETE_HVZK_BACKEND", t_total=0.4))
    for leaf, sharing, auth, t in (("poseidon2", "none", "included-hash", 0.69), (None, "none", "included-hash", 0.70),
                                   ("ajtai-n64", "none", "included-hash", 0.35), ("blake3", "none", "included-hash", 0.40),
                                   ("poseidon2", "tile64x64", "included-hash", 0.30), ("poseidon2", "tile64x64", "included-hash-shared", 0.2),
                                   ("poseidon2", None, "included-hash", 0.5)):
        put(store, hashed(leaf, sharing=sharing, authentication=auth, t_total=t))
    auth = result(candidate="B-Ligero", target=FP8_ADA_MMA, gpu=RTX4090, proof_class="COMPLETE_ZK_BACKEND", t_total=0.1)
    auth["workload_fingerprint"]["authentication"] = "included"
    put(store, auth)
    put(store, result(candidate="SP1", t_total=5.0, security={"target": -100, "achieved_log2": -94.5}))
    put(store, result(candidate="SP1", t_total=6.0))
    rep = result(candidate="B-Ligero", target=FP8_ADA_MMA, gpu=RTX4090, proof_class="COMPLETE_ZK_BACKEND", t_total=0.25, instances=repacked(7))
    put(store, rep)
    put_equiv(store, equiv_doc(repacked(7)))
    big = result(t_total=0.06, gpu=RTX4090, target=FP8_ADA_MMA)
    big["workload_fingerprint"]["hardware"]["gpu"]["memory_total_bytes"] = 48 * 2**30
    put(store, big)                                                                               # a re-memoried board
    comp = result(t_total=0.01)
    comp["workload_fingerprint"]["software"]["backend"]["name"] = "verity-direct (plonky3 uni-stark)"
    put(store, comp)                                                                              # no family
    noprof = result(t_total=0.02)
    noprof["workload_fingerprint"]["profile"] = "not-a-target"
    put(store, noprof)
    assert len(T.build(store).cells) >= 4
    _assert_parity(store)


def test_parity_holds_on_a_seeded_random_store(store: LocalStore) -> None:
    """The same gate over 240 results whose every predicate input is drawn at random (candidate, target, SKU, class, batch, K,
    security, instances, authentication and leaf, verification, red-team verdicts)."""

    rng = random.Random(20260925)
    targets = list(TARGETS.values())
    for i in range(240):
        cand = rng.choice(("A-GKR", "B-Ligero", "SP1"))
        target = rng.choice(targets)
        pc = rng.choice(("NON_ZK_PROOF", "NON_ZK_PROOF_DIAGNOSTIC", "COMPLETE_ZK_BACKEND", "COMPLETE_HVZK_BACKEND", "ARITHMETIC_DIAGNOSTIC"))
        gpu = GPU[target.name] if rng.random() < 0.85 else rng.choice(list(GPU.values()))
        sec = rng.choice(({"target": -128, "achieved_log2": -128.3}, {"target": -128}, {"target": -100, "achieved_log2": -94.5},
                          {"target": -128, "achieved_log2": -127.7}, {"target": None}))
        doc = result(candidate=cand, target=target, gpu=gpu, proof_class=pc, t_total=round(rng.uniform(0.05, 5.0), 3), security=sec)
        fp = doc["workload_fingerprint"]
        if rng.random() < 0.1:
            fp["instances"] = None if rng.random() < 0.5 else {**fp["instances"], "range": [0, 2048]}
        if rng.random() < 0.08:
            fp["B"] = 2048
        if rng.random() < 0.05:
            fp["K"] = 768
        if cand == "B-Ligero" and rng.random() < 0.5:
            leaf = rng.choice(("poseidon2", "blake3", "ajtai-n64", None))
            auth = rng.choice(("included-hash", "included-hash", "included-hash-shared", "included"))
            fp.update(authentication=auth, hash={"poseidon2": "poseidon2-babybear-w24", "blake3": "blake3-keyed", "ajtai-n64": "ajtai-babybear-n64-b1-r0/v1",
                                                 None: "poseidon2-babybear-w24"}[leaf], sharing=rng.choice(("none", "none", "tile64x64", None)))
        labels = []
        if rng.random() < 0.15:
            labels.append(("proof_class", rng.choice(("NON_ZK_PROOF_DIAGNOSTIC", "COMPLETE_HVZK_BACKEND", "COMPLETE_ZK_BACKEND")),
                           f"red-team-{i % 3}", f"2026-09-23T0{i % 10}:00:00Z"))
        put(store, doc, verified_by=rng.choice((VERIFIER, VERIFIER, VERIFIER, None, LANE)), labels=labels)
    assert len(T.build(store).cells) >= 8
    _assert_parity(store)


def test_parity_reports_a_difference(store: LocalStore) -> None:
    a = put(store, result(t_total=4.0))
    frozen = T.build(store)
    assert V.parity(store, frozen, members=[a]) == []
    assert V.parity(store, frozen, members=[]) == [f"legacy-rules cell {a} is not in the snapshot"]
    moved = dataclasses.replace(frozen.cells[0], artifact="art:" + "0" * 64)
    frozen.cells[0] = moved
    assert V.parity(store, frozen) == [f"{FIRST.name} · A-GKR: tables art:{'0' * 64}, views {a}"]


# --- the spec's rules ----------------------------------------------------------------------------------------------------------------


def committed(store: LocalStore, leaf: str = "blake3", *, t_total: float = 0.4, commit: float = 0.6, protocol: bool = True,
              custody: bool = True, security: dict | None = None, verified_by: str | None = VERIFIER, labels: list[tuple] = (),
              B: int = 4096) -> str:
    """A B-Ligero in-proof-hash result on the 4090 line, measured under the protocol, with its commitment time and preserved dumps."""

    doc = hashed(leaf, t_total=t_total, security=security)
    doc["measurements"].append({"name": V.COMMIT, "value": commit, "unit": "s"})
    fp = doc["workload_fingerprint"]
    if B != 4096:
        fp.update(B=B, instances={**fp["instances"], "range": [0, B]})
        doc["measurements"] = [m for m in doc["measurements"] if not m["name"].startswith(("rate.", "overhead.", "vu."))]
    if protocol:
        doc["protocol"], doc["sweep"] = dict(PROTOCOL), dict(SWEEP)
    if not custody:
        return put(store, doc, verified_by=verified_by, labels=labels)
    files = store.put_artifact("run-files/v1", {"run_id": doc["run_id"], "dump": "proofs"})
    store.index.set_replica(files, "remote", "2026-09-25T00:00:00Z")
    art = store.put_artifact(T.RESULT_KIND, doc, refs={"run_files": files})
    store.put_attempt(attempt(doc["run_id"], art, LANE))
    store.add_label(art, "candidate", "X", LANE, ts="2026-09-22T10:00:00Z")
    if verified_by:
        store.add_label(art, "independently_verified", True, verified_by, ref=f"{doc['run_id']}:verify", ts="2026-09-22T10:01:00Z")
    for key, value, by, ts in labels:
        store.add_label(art, key, value, by, ts=ts)
    return art


BLAKE3_ROW = "frame-v3/blake3-row"


def line(views: V.Views, target=FP8_ADA_MMA, scheme: str | None = BLAKE3_ROW) -> dict[str, Any]:
    return next(ln for ln in views.lines if ln["line"] == target.name and ln["scheme"] == scheme)


def codes(views: V.Views, art: str) -> list[str]:
    return [k for k, _ in next(r for r in views.rejected if r["artifact"] == art)["codes"]]


def test_native_is_the_compute_bound_datasheet_estimate() -> None:
    for target in TARGETS.values():
        n = V.native(target)
        assert n["ops_per_instance"] == 3072 and n["steps"] == 96 and n["unit"] == "tensor cores"
        assert n["instances_per_second"] == pytest.approx(1 / target.native_peak.seconds_per_vu(T.K))
    assert V._native_text(V.native(FIRST)) == "1.0e11/s · tensor cores, BF16 312 T"


def test_a_full_relation_blake3_result_fills_its_family_cell_end_to_end(store: LocalStore) -> None:
    art = committed(store, t_total=0.4, commit=0.6)
    views = V.build(store)
    cell = line(views)["cells"]["B-Ligero"]
    assert cell is not None and cell.artifact == art and cell.configuration == "B-Ligero + BLAKE3 in circuit"
    assert cell.throughput["end_to_end_s"] == pytest.approx(1.0) and cell.throughput["instances_per_second"] == pytest.approx(4096)
    assert cell.overhead == pytest.approx(V.native(FP8_ADA_MMA)["instances_per_second"] / 4096)
    # proving time alone is the legacy ratio; end to end it is (commit + proving) / proving times that
    assert cell.overhead == pytest.approx(cell.legacy.overhead * 1.0 / 0.4)
    assert cell.provisional and "no red-team clearance of B-Ligero + BLAKE3 in circuit" in cell.provisional
    md = V.render_markdown(views)
    assert V.PREVIEW in md and "## Table 1" in md and "## Table 2" in md and "## Table 3" in md
    assert f"{T.fmt_overhead(cell.overhead)} (prov.)" in md
    # Table 3: commitment and the proving buckets per instance, with shares of end to end
    row3 = next(r for r in md.splitlines() if r.startswith(f"| {FP8_ADA_MMA.name} · frame-v3 · keyed-BLAKE3 rows · B-Ligero"))
    assert "146 µs (60%)" in row3 and "244 µs (100%)" in row3
    assert "Encoding + witness commitment" in md and "Serving overhead" in md
    serving, proving = V.overheads(cell)
    N = V.native(FP8_ADA_MMA)["instances_per_second"]
    assert serving == pytest.approx(N * 0.6 / 4096) and proving == pytest.approx(N * 0.4 / 4096)
    assert serving + proving == pytest.approx(cell.overhead)
    assert f"| {T.fmt_overhead(serving)} | {T.fmt_overhead(proving)} |" in row3


def test_a_red_team_clearance_of_the_statement_lifts_the_marker(store: LocalStore) -> None:
    a = committed(store, t_total=0.4)
    other = committed(store, t_total=0.9, labels=[("proof_class", "COMPLETE_ZK_BACKEND", REDTEAM, "2026-09-25T01:00:00Z")])
    cell = line(V.build(store))["cells"]["B-Ligero"]
    assert cell.artifact == a and cell.provisional is None and other
    # a later downgrade of the same statement caps every result of it (D), and stands
    committed(store, t_total=2.0, labels=[("proof_class", "COMPLETE_HVZK_BACKEND", "red-team-2", "2026-09-25T02:00:00Z")])
    views = V.build(store)
    assert line(views)["cells"]["B-Ligero"] is None and codes(views, a) == ["D"]


POSEIDON2_ROW = "frame-v3/poseidon2-row"


def test_table2_has_one_line_per_subcircuit_hardware_and_admissible_scheme(store: LocalStore) -> None:
    views = V.build(store)
    order = ["frame-v3/sha256", BLAKE3_ROW, POSEIDON2_ROW]
    assert {ln["scheme"] for ln in views.lines} == set(order) and len(views.lines) == 3 * len(TARGETS)
    assert [ln["scheme"] for ln in views.lines[:3]] == order                    # one subcircuit's schemes sit together
    assert not V._SCHEME["vllm-v1"].admissible and not V._SCHEME["frame-v3/ajtai-row"].admissible
    assert V._SCHEME[POSEIDON2_ROW].admissible and not V._SCHEME[POSEIDON2_ROW].standard      # core-defined, algebraic
    assert {ln["scheme"] for ln in V.build(store, V.LEGACY).lines} == {None}


def test_algebraic_hashes_are_admitted_flagged_and_optionally_filtered(store: LocalStore) -> None:
    """Daniel, 2026-09-25: results of algebraic-hash backends and schemes count like any other; Table 1 flags them and their
    Table 2 cells carry ``(alg.)``.  The standard-hashes filter is optional, off in the published view."""

    pos = committed(store, "poseidon2", t_total=0.1)
    views = V.build(store)
    cell = line(views, scheme=POSEIDON2_ROW)["cells"]["B-Ligero"]
    assert cell is not None and cell.artifact == pos and cell.algebraic == [
        "Poseidon2-BabyBear (statement commitment; collision resistance (row digests); algebraic)"]
    md = V.render_markdown(views)
    assert f"{T.fmt_overhead(cell.overhead)} (prov.) (alg.)" in md and f"{V.ALGEBRAIC_FLAG} (Table 1, B-Ligero + Poseidon2 per row" in md
    rows = dict((r[0], dict(zip(V.TABLE1_COLUMNS, r))) for _, r in V.table1_rows(views))
    assert rows["B-Ligero + Poseidon2 per row"]["Flag"] == f"**{V.ALGEBRAIC_FLAG}** (Poseidon2-BabyBear)"
    assert rows["SP1 stock (relation-only / bare guest)"]["Flag"] == f"**{V.ALGEBRAIC_FLAG}** (Poseidon2-KoalaBear)"
    assert rows["B-Ligero + BLAKE3 in circuit"]["Flag"] == ""
    strict = V.build(store, dataclasses.replace(V.SPEC, standard_hashes=True))
    assert codes(strict, pos) == ["L"] and line(strict, scheme=POSEIDON2_ROW)["empty"]["B-Ligero"]["state"] == "below-filter"
    # an algebraic hash inside the proof system alone (SP1's challenger) is admitted and flagged the same way
    art = committed(store, t_total=0.4)
    sp1ish = tuple(dataclasses.replace(c, hashes=c.hashes + V._SP1_HASHES) if c.id == "B-Ligero + BLAKE3 in circuit" else c
                   for c in V.CONFIGURATIONS)
    got = line(V.build(store, configurations=sp1ish))["cells"]["B-Ligero"]
    assert got.artifact == art and [h.split(" (")[0] for h in got.algebraic] == ["Poseidon2-KoalaBear", "Poseidon2-KoalaBear"]


def test_undefined_schemes_and_weaker_statements_leave_table2(store: LocalStore) -> None:
    ajtai = committed(store, "ajtai-n64", t_total=0.1)
    bare = put(store, result(candidate="B-Ligero", target=FP8_ADA_MMA, gpu=RTX4090, proof_class="COMPLETE_ZK_BACKEND", t_total=0.05))
    agkr = put(store, result(target=FP8_ADA_MMA, gpu=RTX4090, t_total=0.05))
    views = V.build(store)
    assert codes(views, ajtai) == ["S"] and "which verity.commitments does not define yet" in next(
        r for r in views.rejected if r["artifact"] == ajtai)["codes"][0][1]
    assert codes(views, bare)[0] == "R" and codes(views, agkr)[0] == "R"
    ln = line(views)
    assert ln["cells"]["B-Ligero"] is None and ln["empty"]["B-Ligero"]["state"] == "weaker"
    assert "a scheme verity.commitments does not define (B-Ligero + Ajtai n=64)" in ln["empty"]["B-Ligero"]["text"]
    assert ln["empty"]["A-GKR"]["text"] == "results only for a weaker statement (A-GKR CPU (Rust))"
    # the rejected list previews what such a result costs end to end
    pre = next(r for r in views.rejected if r["artifact"] == ajtai)["end_to_end_preview"]
    assert pre["commit_warm"] and pre["overhead"] == pytest.approx(V.native(FP8_ADA_MMA)["instances_per_second"] * 0.7 / 4096)


def test_the_protocol_custody_and_security_are_checked(store: LocalStore) -> None:
    v1 = committed(store, t_total=0.1, protocol=False)
    lost = committed(store, t_total=0.2, custody=False)
    weak = committed(store, t_total=0.3, security={"target": -100, "achieved_log2": -99.0})
    views = V.build(store)
    assert codes(views, v1) == ["M"] and codes(views, lost) == ["F"] and codes(views, weak) == ["L"]
    assert line(views)["cells"]["B-Ligero"] is None
    assert line(views)["empty"]["B-Ligero"]["state"] == "below-filter"
    # the unfiltered view takes the sub-128-bit result; the filtered cell names the unfiltered pick when both exist
    assert line(V.build(store, V.UNFILTERED))["cells"]["B-Ligero"].artifact == weak
    ok = committed(store, t_total=0.9)
    cell = line(V.build(store))["cells"]["B-Ligero"]
    assert cell.artifact == ok and cell.unfiltered["artifact"] == weak and cell.unfiltered["bound_log2"] == -99.0


def test_the_batch_is_the_backends_and_the_best_p_wins(store: LocalStore) -> None:
    committed(store, t_total=0.4, commit=0.1)                                                     # 8.2e3/s
    big = committed(store, t_total=3.0, commit=0.8, B=16384)                                      # 4.3e3/s
    cell = line(V.build(store))["cells"]["B-Ligero"]
    assert cell.throughput["instances_per_second"] == max(4096 / 0.5, 16384 / 3.8)
    assert cell.artifact != big and cell.throughput["instances"] == 4096
    faster = committed(store, t_total=1.5, commit=0.3, B=16384)                                   # 9.1e3/s: the larger batch wins
    cell = line(V.build(store))["cells"]["B-Ligero"]
    assert cell.artifact == faster and cell.throughput["instances"] == 16384 and cell.batching["total"] == 16384


def test_reasons_come_in_the_specs_order(store: LocalStore) -> None:
    art = put(store, result(target=FP8_ADA_MMA, gpu=A100, proof_class="ARITHMETIC_DIAGNOSTIC", t_total=0.1), verified_by=None)
    got = codes(V.build(store), art)
    assert got == sorted(got, key=list(V.REASON_LETTERS).index) and got[:2] == ["R", "Z"] and "U" in got and "K" in got


def test_archived_and_unsupported_configurations(store: LocalStore) -> None:
    art = committed(store, t_total=0.4)
    cfgs = tuple(dataclasses.replace(c, archived=True) if c.id == "B-Ligero + BLAKE3 in circuit" else c for c in V.CONFIGURATIONS)
    views = V.build(store, configurations=cfgs)
    assert line(views)["cells"]["B-Ligero"] is None and "archived" in next(r for r in views.rejected if r["artifact"] == art)["codes"][0][1]
    assert line(V.build(store, dataclasses.replace(V.SPEC, include_archived=True), cfgs))["cells"]["B-Ligero"].artifact == art
    ns = tuple(dataclasses.replace(c, unsupported={NVFP4_SM120.name: "no FP4 gadget"}) if c.family == "SP1" else c for c in V.CONFIGURATIONS)
    views = V.build(store, configurations=ns)
    assert line(views, NVFP4_SM120)["empty"]["SP1"] == {"state": "n/s", "text": "no FP4 gadget"}
    assert "| n/s [" in V.render_markdown(views)


def test_table1_carries_the_security_properties(store: LocalStore) -> None:
    assert {c.id for c in V.CONFIGURATIONS} == {v.id for v in V.D.VARIANTS}
    full = {c.id for c in V.CONFIGURATIONS if c.full_relation}
    assert full == {"B-Ligero + Poseidon2 per row", "B-Ligero + Poseidon2 shared 64x64 tile", "B-Ligero + BLAKE3 in circuit",
                    "B-Ligero + Ajtai n=64", "B-Ligero + Ajtai n=128"}
    assert [c.id for c in V.CONFIGURATIONS if c.full_relation and not c.scheme_problem()] == [
        "B-Ligero + Poseidon2 per row", "B-Ligero + BLAKE3 in circuit"]
    assert all(x.basis.startswith(("proven", "conjectured")) for c in V.CONFIGURATIONS for x in c.proximity)
    assert all(c.proximity and c.hashes and c.dominant_term for c in V.CONFIGURATIONS)
    committed(store, t_total=0.4)
    rows = dict((r[0], r) for _, r in V.table1_rows(V.build(store)))
    blake = dict(zip(V.TABLE1_COLUMNS, rows["B-Ligero + BLAKE3 in circuit"]))
    where = f"{FP8_ADA_MMA.name} · frame-v3 · keyed-BLAKE3 rows · B-Ligero"
    assert blake["Soundness (whole-proof bound at its cells; dominant term)"].startswith(f"2^-128.3 at {where}")
    assert "dominant term: proximity-gap term n/|F|" in blake["Soundness (whole-proof bound at its cells; dominant term)"]
    assert "BLAKE3 (keyed) (statement commitment; collision resistance (row digests); standard)" in blake["Hashes (role; property; class)"]
    assert blake["Commitment scheme"].startswith("frame-v3 · keyed-BLAKE3 rows (frame-v3/blake3-row; admissible)")
    assert blake["Proximity bounds"].startswith("Ligero column tests") and "unique decoding, proven" in blake["Proximity bounds"]
    assert blake["Cells"] == where
    sp1 = dict(zip(V.TABLE1_COLUMNS, rows["SP1 stock (relation-only / bare guest)"]))
    assert "Poseidon2-KoalaBear (proof-internal commitment; collision resistance (Merkle trees); algebraic)" in sp1["Hashes (role; property; class)"]


def test_cli_json_round_trips_and_parity_exit_codes(store: LocalStore, capsys: pytest.CaptureFixture[str]) -> None:
    committed(store, t_total=0.4)
    put(store, result(t_total=4.0))
    assert V.main(["--root", str(store.root), "--format", "json"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert js["schema"] == "verity/tables-views/v1" and js["table2"]["columns"] == ["A-GKR", "B-Ligero", "SP1"]
    assert len(js["table3"]) == 1 and js["table3"][0]["scheme"] == BLAKE3_ROW
    assert [x["id"] for x in js["schemes"] if x["admissible"]] == ["frame-v3/sha256", BLAKE3_ROW, POSEIDON2_ROW]
    assert V.main(["--root", str(store.root), "--rules", "legacy", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["table2"]["columns"] == ["A-GKR", "B-Ligero", "SP1"]
    assert V.main(["--root", str(store.root), "--parity"]) == 0
    assert capsys.readouterr().out.startswith("parity: ok")
    assert V.main(["--root", str(store.root), "--parity", "--snapshot", "tables-none"]) == 2


def test_finding_labels_mark_or_pull_cells_in_both_renderers(store: LocalStore) -> None:
    open_ = "UNDER RE-VERIFICATION: soundness issue R1/R2"
    art = committed(store, t_total=0.4, labels=[("finding", open_, "coordinator", "2026-09-25T07:45:00Z")])
    views = V.build(store)
    cell = line(views)["cells"]["B-Ligero"]
    assert cell.artifact == art and cell.legacy.finding.startswith(open_)
    assert "(under re-verification) [" in V.render_markdown(views)
    store.add_label(art, "finding", "PULLED R1/R2", "coordinator", ts="2026-09-25T09:00:00Z")
    views = V.build(store)
    assert line(views)["cells"]["B-Ligero"] is None and codes(views, art)[-1] == "D"
    # the legacy rules pull the same frozen cells, so parity holds with marks and pulls
    put(store, result(t_total=4.0), labels=[("finding", open_, "coordinator", "2026-09-25T07:45:00Z")])
    pulled = put(store, result(t_total=2.0), labels=[("finding", "PULLED R1/R2", "coordinator", "2026-09-25T09:00:00Z")])
    assert V.parity(store) == [] and pulled not in {c.artifact for c in V.build(store, V.LEGACY).cells}
