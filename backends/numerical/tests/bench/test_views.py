"""The approved spec's views (``bench.views``): the switch's parity gate against the frozen tables, and the spec's rules --
the full committed relation with SHA-256 / BLAKE3 commitments, end-to-end throughput, the security filter, the computed
headline, ``archived``, ``n/s`` and the provisional marker -- over an isolated store root."""

from __future__ import annotations

import dataclasses
import json
import math
import random
from typing import Any

import pytest

from verity.proofs.target import BF16_HOPPER, FIRST, FP8_ADA_MMA, FP8_HOPPER_WGMMA, NVFP4_SM120, TARGETS
from verity_numerical.bench import census, contract
from verity_numerical.bench import tables as T
from verity_numerical.bench import views as V
from verity_numerical.bench.views import slug

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
              B: int = 4096, manifest: str | None = None, setup: float | None = None, interaction: dict | None = None,
              backend: str | None = None, proof_class: str | None = None, joints: dict | None = None) -> str:
    """A B-Ligero in-proof-hash result on the 4090 line, measured under the protocol, with its commitment time and preserved dumps."""

    doc = hashed(leaf, t_total=t_total, security=security, **({"joints": joints, "buckets": {"lookup": 0.1 * t_total}} if joints else {}))
    doc["measurements"].append({"name": V.COMMIT, "value": commit, "unit": "s"})
    if setup is not None:
        doc["measurements"].append({"name": "fixed.setup_seconds", "value": setup, "unit": "s"})
    ix = {"rounds.sequential_depth": 0} if interaction is None else interaction
    for name, value in ix.items():
        if name in ("rtt_method", "t_total_includes_wait", "loopback_method"):
            doc.setdefault("interaction", {})[name] = value
            continue
        unit = {"rounds.sequential_depth": "count", "rounds.protocol": "count", "net.bytes_out": "B", "net.bytes_in": "B", "net.rtt_ms": "ms",
                "proof_bytes": "B", "relation.statement_bytes_total": "B"}.get(name, "s")
        doc["measurements"].append({"name": name, "value": value, "unit": unit})
    fp = doc["workload_fingerprint"]
    if backend is not None:
        fp["software"]["backend"]["name"] = backend
    if proof_class is not None:
        fp["proof_class"] = proof_class
        if proof_class.startswith("NON_ZK"):
            fp["zk_mode"] = "none"
    if B != 4096:
        fp.update(B=B, instances={**fp["instances"], "range": [0, B], "manifest_sha256": V.STREAMS[FP8_ADA_MMA.name](B)})
        doc["measurements"] = [m for m in doc["measurements"] if not m["name"].startswith(("rate.", "overhead.", "vu."))]
    if manifest is not None:
        fp["instances"] = {**fp["instances"], "manifest_sha256": manifest, "seed": 20260922, "recipe": "fp8-ada synthetic (x4 fold)"}
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
        assert n["instances_per_second"] == pytest.approx(1 / census.native_seconds_per_instance(contract.native_peak_of(target), T.K))
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
    rows = [r for r in md.splitlines() if r.startswith(f"| {FP8_ADA_MMA.name} · frame-v3 · keyed-BLAKE3 rows · B-Ligero")]
    row3, drill, _ = rows                                # Table 3, its seconds-per-instance drill-down, the interaction drill-down
    serving, proving = V.overheads(cell)
    N = V.native(FP8_ADA_MMA)["instances_per_second"]
    assert serving == pytest.approx(N * 0.6 / 4096) and proving == pytest.approx(N * 0.4 / 4096)
    assert serving + proving == pytest.approx(cell.overhead)
    # Table 3 in overhead units: each stage x N with its share; the stages sum to the cell
    assert row3.startswith(f"| {FP8_ADA_MMA.name} · frame-v3 · keyed-BLAKE3 rows · B-Ligero | {T.fmt_overhead(serving)} (60%) |")
    assert f"| {T.fmt_overhead(cell.overhead)} (100%) | — |" in row3 and "µs" not in row3
    st = V.stages(cell)
    assert st["commitment"] + st["interaction"] + sum(v for v in st["buckets"].values() if v) + st["other"] == pytest.approx(cell.overhead)
    assert "146 µs (60%)" in drill and "244 µs (100%)" in drill
    assert "Encoding + witness commitment" in md and "Setup, one-time (not in the total)" in md and "seconds per instance" in md


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
    order = ["frame-v3/sha256", BLAKE3_ROW, "vllm-v1", POSEIDON2_ROW]
    assert {ln["scheme"] for ln in views.lines} == set(order) and len(views.lines) == 4 * len(TARGETS)
    assert [ln["scheme"] for ln in views.lines[:4]] == order                    # one subcircuit's schemes sit together
    assert V._SCHEME["vllm-v1"].admissible and V._SCHEME["vllm-v1"].standard and not V._SCHEME["frame-v3/ajtai-row"].admissible
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


def test_synthetic_sets_are_streams_identified_by_generator_seed_and_n(store: LocalStore) -> None:
    """A sweep point above 4,096 counts when its instances are a prefix-consistent extension of the declared synthetic set: the
    same generator and seed over [0, n) (the digest hashes n, instance i does not depend on it)."""

    for name, recipe in V.STREAMS.items():
        assert recipe(4096) == T.FROZEN_INSTANCES[name]["manifest_sha256"], name
    big = committed(store, t_total=1.5, commit=0.3, B=16384)
    cell = line(V.build(store))["cells"]["B-Ligero"]
    assert cell.artifact == big and "instances [0, 16384) of vu-k1536-fp8-ada (a prefix-consistent extension" in V.render_markdown(V.build(store))
    ref = {**T.FROZEN_INSTANCES[FP8_ADA_MMA.name], "range": [0, 16384]}
    assert V.instance_range(FP8_ADA_MMA, {**ref, "manifest_sha256": V.STREAMS[FP8_ADA_MMA.name](16384)}) == ("stream", None)
    assert V.instance_range(FP8_ADA_MMA, {**ref, "manifest_sha256": V.STREAMS[FP8_ADA_MMA.name](16384), "seed": 1}) == (None, None)
    assert V.instance_range(FP8_ADA_MMA, {**ref, "manifest_sha256": V.STREAMS[FP8_ADA_MMA.name](8192)}) == (None, None)
    assert V.instance_range(FP8_ADA_MMA, {**ref, "range": [0, 2048], "manifest_sha256": V.STREAMS[FP8_ADA_MMA.name](2048)}) == ("prefix", None)
    assert V.instance_range(NVFP4_SM120, {**T.FROZEN_INSTANCES[NVFP4_SM120.name], "range": [0, 65536],
                                          "manifest_sha256": V.STREAMS[NVFP4_SM120.name](65536)}) == ("stream", None)
    # the legacy (parity) rules keep today's frozen range
    assert V.build(store, V.LEGACY).cells == []


def test_the_a100_fixture_has_no_stream_so_a_larger_batch_repeats_instances(store: LocalStore) -> None:
    ref = T.FROZEN_INSTANCES[FIRST.name]
    how, why = V.instance_range(FIRST, {**ref, "range": [0, 16384]})
    assert how is None and "repeats instances" in why and "4096 instances" in why
    assert V.instance_range(FIRST, {**ref, "range": [0, 64]}) == ("prefix", None)
    doc = result(t_total=4.0)
    doc["workload_fingerprint"].update(B=16384, instances={**ref, "range": [0, 16384]})
    doc["measurements"] = [m for m in doc["measurements"] if not m["name"].startswith(("rate.", "overhead.", "vu."))]
    art = put(store, doc)
    views = V.build(store)
    got = next(r for r in views.rejected if r["artifact"] == art)["codes"]
    assert ("I", why) in got


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
    assert {c.id for c in V.CONFIGURATIONS} == {v.id for v in V.D.VARIANTS} | {V.FLOCK_CONFIG.id}
    full = {c.id for c in V.CONFIGURATIONS if c.full_relation}
    assert full == {"B-Ligero + Poseidon2 per row", "B-Ligero + Poseidon2 shared 64x64 tile", "B-Ligero + BLAKE3 in circuit",
                    "B-Ligero + SHA-256 in circuit", "B-Ligero + vllm-v1 SHA-256 in circuit", "B-Ligero + Ajtai n=64",
                    "B-Ligero + Ajtai n=128", "SP1 stock, committed (frame-v3 SHA-256 rows)",
                    "SP1 stock, committed (vllm-v1 SHA-256 position leaves)", "Flock (flock-128-r2, live coins)", V.ROUTE_A}
    assert [c.id for c in V.CONFIGURATIONS if c.full_relation and not c.scheme_problem()] == [
        "B-Ligero + Poseidon2 per row", "B-Ligero + BLAKE3 in circuit", "B-Ligero + SHA-256 in circuit",
        "B-Ligero + vllm-v1 SHA-256 in circuit", V.ROUTE_A, "SP1 stock, committed (frame-v3 SHA-256 rows)",
        "SP1 stock, committed (vllm-v1 SHA-256 position leaves)", "Flock (flock-128-r2, live coins)"]
    assert all(c.interaction for c in V.CONFIGURATIONS)
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
    assert js["schema"] == "verity/tables-views/v1" and js["table2"]["columns"] == ["A-GKR", "B-Ligero", "SP1", "Flock"]
    assert len(js["table3"]) == 1 and js["table3"][0]["scheme"] == BLAKE3_ROW
    assert [x["id"] for x in js["schemes"] if x["admissible"]] == ["frame-v3/sha256", BLAKE3_ROW, "vllm-v1", POSEIDON2_ROW]
    assert V.main(["--root", str(store.root), "--rules", "legacy", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["table2"]["columns"] == ["A-GKR", "B-Ligero", "SP1", "Flock"]
    assert V.main(["--root", str(store.root), "--parity"]) == 0
    assert capsys.readouterr().out.startswith("parity: ok")
    assert V.main(["--root", str(store.root), "--parity", "--snapshot", "tables-none"]) == 2


def test_the_parity_render_takes_its_snapshot_and_is_recorded_as_an_artifact(store: LocalStore, capsys: pytest.CaptureFixture[str]) -> None:
    """The switch's gate 2 as one command: `--parity --take-snapshot --record` freezes the frozen cells as `tables --snapshot` does,
    compares, and stores the report (every cell's artifact and both ratios) as a tables-parity/v1 artifact."""
    a = put(store, result(t_total=4.0))
    committed(store, t_total=0.4)
    assert V.main(["--root", str(store.root), "--parity", "--take-snapshot", "--record", "--format", "json"]) == 0
    out = capsys.readouterr()
    rec = json.loads(out.out)
    snap = T.take_snapshot(store, T.build(store))
    assert rec["ok"] and rec["differences"] == [] and rec["snapshot"] == T.snapshot_name() and rec["snapshot_artifact"] == snap
    assert [(c["column"], c["artifact"]) for c in rec["cells"]] == [("A-GKR", a)]
    assert rec["cells"][0]["views_overhead"] == pytest.approx(rec["cells"][0]["tables_overhead"], rel=1e-12)
    m = store.get_manifest(rec["artifact"])
    assert m.kind == V.PARITY_KIND and m.refs == {"snapshot": snap, "cells": [a]} and m.meta["ok"] is True
    assert f"parity recorded as {rec['artifact']}" in out.err
    # the same store names the same snapshot and records the same parity artifact (content-addressed)
    assert V.main(["--root", str(store.root), "--parity", "--snapshot", T.snapshot_name(), "--record"]) == 0
    assert f"parity recorded as {rec['artifact']}" in capsys.readouterr().err
    # a difference is recorded too, and exits 1
    store.snapshot("tables-empty", [], note="empty")
    assert V.main(["--root", str(store.root), "--parity", "--snapshot", "tables-empty"]) == 1
    assert capsys.readouterr().out.startswith(f"parity: DIFFERENT\nlegacy-rules cell {a} is not in the snapshot")


def test_the_published_render_drops_the_preview_banner(store: LocalStore, capsys: pytest.CaptureFixture[str]) -> None:
    committed(store, t_total=0.4)
    views = V.build(store)
    assert V.render_markdown(views).startswith(f"> {V.PREVIEW}")
    pub = V.render_markdown(views, published=True)
    assert pub.startswith(f"> {V.PUBLISHED}") and V.PREVIEW not in pub
    assert pub.split("\n", 1)[1] == V.render_markdown(views).split("\n", 1)[1]          # only the banner changes
    assert V.main(["--root", str(store.root), "--published"]) == 0 and capsys.readouterr().out == pub
    js = V.render_json(views, published=True)
    assert js["published"] == V.PUBLISHED and "preview" not in js


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


def _stream_ref(n: int, **over: Any) -> dict[str, Any]:
    return {**T.FROZEN_INSTANCES[FP8_ADA_MMA.name], "range": [0, n], "manifest_sha256": V.STREAMS[FP8_ADA_MMA.name](n),
            "seed": 20260922, "recipe": "fp8-ada synthetic (uniform finite E4M3 bytes, model-recorded accumulators)", **over}


def test_an_equivalence_to_the_stream_admits_a_repacked_sweep_point(store: LocalStore) -> None:
    """The x4 fold's 8,192 plateau: its manifest names the fold, so it counts only through an instance-equiv/v1 whose frozen
    side is the declared set over [0, 8192) -- the synthetic stream by generator, seed and n, as the runner writes the ref."""

    art = committed(store, t_total=1.5, commit=0.3, B=8192, manifest="ab" * 32)
    cand = store.get_manifest(art).meta["workload_fingerprint"]["instances"]
    views = V.build(store)
    assert codes(views, art) == ["I"]
    eq = put_equiv(store, equiv_doc(cand, frozen=_stream_ref(8192)))
    views = V.build(store)
    cell = line(views)["cells"]["B-Ligero"]
    assert cell is not None and cell.artifact == art and cell.legacy.instances_equiv == eq
    assert f"{T.INSTANCE_EQUIV_NOTE} ({eq})" in V.render_markdown(views)
    # the frozen tables keep today's 4,096 rule
    assert art in {r["artifact"] for r in T.build(store).rejected}


@pytest.mark.parametrize(("frozen", "why"), [
    (_stream_ref(8192, seed=1), "neither FROZEN_INSTANCES[target] nor the declared set's stream"),
    (_stream_ref(8192, manifest_sha256=V.STREAMS[FP8_ADA_MMA.name](4096)), "neither FROZEN_INSTANCES[target] nor the declared set's stream"),
    (dict(T.FROZEN_INSTANCES[FP8_ADA_MMA.name]), "do not hold the same number of instances"),
])
def test_a_stream_equivalence_must_name_the_declared_set_over_the_same_range(store: LocalStore, frozen: dict, why: str) -> None:
    art = committed(store, t_total=1.5, commit=0.3, B=8192, manifest="ab" * 32)
    cand = store.get_manifest(art).meta["workload_fingerprint"]["instances"]
    put_equiv(store, equiv_doc(cand, frozen=frozen))
    got = next(r for r in V.build(store).rejected if r["artifact"] == art)["codes"]
    assert [k for k, _ in got] == ["I"] and why in got[0][1]


def test_an_equivalence_cannot_launder_the_a100_fixtures_repeats(store: LocalStore) -> None:
    ref = T.FROZEN_INSTANCES[FIRST.name]
    cand = {**ref, "range": [0, 8192], "manifest_sha256": "cd" * 32}
    doc = result(t_total=4.0)
    doc["workload_fingerprint"].update(B=8192, instances=cand)
    doc["measurements"] = [m for m in doc["measurements"] if not m["name"].startswith(("rate.", "overhead.", "vu."))]
    art = put(store, doc)
    put_equiv(store, equiv_doc(cand, target=FIRST, frozen={**ref, "range": [0, 8192]}))
    got = next(r for r in V.build(store).rejected if r["artifact"] == art)["codes"]
    assert ("I" in [k for k, _ in got]) and any("repeats instances" in w for k, w in got if k == "I")


def test_an_equivalence_whose_payload_is_not_local_is_reported_not_silently_read(store: LocalStore, tmp_path) -> None:
    art = committed(store, t_total=1.5, commit=0.3, B=8192, manifest="ab" * 32)
    cand = store.get_manifest(art).meta["workload_fingerprint"]["instances"]
    eq = put_equiv(store, equiv_doc(cand, frozen=_stream_ref(8192)), payload=tmp_path / "equiv.json")
    assert line(V.build(store))["cells"]["B-Ligero"].artifact == art           # readable: it counts
    store.object_path(store.get_manifest(eq).payload.sha256).unlink()
    views = V.build(store)
    assert views.unreadable == [f"{eq}: payload not local (blob {store.get_manifest(eq).payload.sha256[:12]}… is not in this store: "
                                "`research data fetch`)"]
    got = next(r for r in views.rejected if r["artifact"] == art)["codes"]
    assert got[0][0] == "I" and "payload not local" in got[0][1]
    md = V.render_markdown(views)
    assert "Unreadable here (1;" in md and V.render_json(views)["unreadable"] == views.unreadable
    # the frozen tables name it too, instead of a bare-meta document with a null candidate
    assert T.instance_equivalence(cand, FP8_ADA_MMA, T.instance_equivs(store), set())[1].startswith(f"{eq}: payload not local")


def test_setup_is_one_time_shown_beside_the_cell_amortized_over_the_batch(store: LocalStore) -> None:
    """Setup and key generation (SP1 ~16 s) get their own Table 3 column: seconds, and x native amortized over the measured batch
    (setup x N / instances); they never enter the Table 2 cell or the stage sum."""
    committed(store, t_total=0.4, commit=0.6, setup=16.0)
    views = V.build(store)
    cell = line(views)["cells"]["B-Ligero"]
    N = V.native(FP8_ADA_MMA)["instances_per_second"]
    assert cell.setup_s == 16.0 and cell.overhead == pytest.approx(N * 1.0 / 4096)          # the cell is commitment + proving only
    st = V.stages(cell)
    assert st["setup"] == {"seconds": 16.0, "amortized": pytest.approx(16.0 * N / 4096), "instances": 4096}
    assert st["commitment"] + sum(v for v in st["buckets"].values() if v) + st["other"] == pytest.approx(cell.overhead)
    assert f"| 16 s · {T.fmt_overhead(16.0 * N / 4096)} over 4096 |" in V.render_markdown(views)
    assert V.render_json(views)["table3"][0]["stages_overhead"]["setup"]["seconds"] == 16.0


LIVE = {"rounds.sequential_depth": 3, "net.bytes_out": 1.25e9, "net.bytes_in": 384.0, "net.rtt_ms": 1.0, "rtt_method": "ping median of 20, same pod pair",
        "net.wait_seconds": 0.103}


def test_the_cell_is_the_interaction_formula_at_the_reference_network(store: LocalStore) -> None:
    """t = compute + rounds x RTT + bytes / bandwidth at 1 ms and 100 Gb/s; Table 3's interaction stage is that network time x N; the
    drill-down gives the formula at 20 ms and 100 ms, and at 10 Gb/s."""
    committed(store, t_total=0.4, commit=0.6, interaction=LIVE)
    views = V.build(store)
    cell = line(views)["cells"]["B-Ligero"]
    N = V.native(FP8_ADA_MMA)["instances_per_second"]
    net = 3 * 0.001 + (1.25e9 + 384) * 8 / 100e9                                # 0.103 s at the reference profile
    assert cell.throughput["interaction_s"] == pytest.approx(net)
    assert cell.overhead == pytest.approx(N * (0.6 + 0.4 + net) / 4096)
    st = V.stages(cell)
    assert st["interaction"] == pytest.approx(N * net / 4096)
    assert st["commitment"] + st["interaction"] + sum(v for v in st["buckets"].values() if v) + st["other"] == pytest.approx(cell.overhead)
    md = V.render_markdown(views)
    at20 = N * (0.6 + 0.4 + 3 * 0.020 + (1.25e9 + 384) * 8 / 100e9) / 4096
    at10g = N * (0.6 + 0.4 + 3 * 0.001 + (1.25e9 + 384) * 8 / 10e9) / 4096
    assert "### Interaction drill-down" in md and T.fmt_overhead(at20) in md and "ping median of 20" in md
    assert "At 1 ms, 10 Gb/s" in md and T.fmt_overhead(at10g) in md and V.REF_BANDWIDTH_BPS == 100e9


def test_a_record_whose_t_total_includes_the_wait_counts_its_compute_once(store: LocalStore) -> None:
    """Flock's live records time the coin waits inside t.total (``interaction.t_total_includes_wait``): the cell takes prover compute
    from live.prover_compute_seconds (else t.total minus the wait) and adds the reference network once, not on top of the wait."""
    committed(store, t_total=0.4 + 0.103, commit=0.6, interaction={**LIVE, "t_total_includes_wait": True})
    cell = line(V.build(store))["cells"]["B-Ligero"]
    N = V.native(FP8_ADA_MMA)["instances_per_second"]
    net = 3 * 0.001 + (1.25e9 + 384) * 8 / 100e9
    assert cell.overhead == pytest.approx(N * (0.6 + 0.4 + net) / 4096)


def test_live_coin_handling_measured_on_loopback_is_prover_compute_not_network(store: LocalStore) -> None:
    """rounds x live.loopback_round_seconds (at most the measured wait) moves from the network wait into prover compute; the
    reference network (1 ms per round) is then added on top of it, and the tolerance check compares against that."""
    art = committed(store, t_total=0.4, commit=0.6, interaction={**LIVE, "live.loopback_round_seconds": 0.01})
    cell = line(V.build(store))["cells"]["B-Ligero"]
    N = V.native(FP8_ADA_MMA)["instances_per_second"]
    net = 3 * 0.001 + (1.25e9 + 384) * 8 / 100e9
    assert cell.artifact == art and cell.overhead == pytest.approx(N * (0.6 + 0.4 + 3 * 0.01 + net) / 4096)
    ix = V.interaction(store.get_manifest(art).meta)
    assert ix["loopback_s"] == pytest.approx(0.03) and ix["wait_s"] == pytest.approx(0.103 - 0.03)
    assert ix["prover_s"] == pytest.approx(0.43)


def test_the_tolerance_check_uses_the_runs_own_rtt_but_the_cell_uses_the_reference(store: LocalStore) -> None:
    """A same-DC session at 0.2 ms RTT: measured wall = compute + rounds x 0.2 ms + bytes; that passes the check (it matches the
    run), and the published cell is still the formula at 1 ms / 100 Gb/s."""
    rounds, rtt = 400, 0.0002
    wait = rounds * rtt + (1.25e9 + 384) * 8 / 100e9
    art = committed(store, t_total=0.4, commit=0.6, interaction={**LIVE, "rounds.sequential_depth": rounds, "net.rtt_ms": rtt * 1e3,
                                                                 "net.wait_seconds": wait})
    cell = line(V.build(store))["cells"]["B-Ligero"]
    N = V.native(FP8_ADA_MMA)["instances_per_second"]
    assert cell.artifact == art and cell.overhead == pytest.approx(N * (0.6 + 0.4 + rounds * 0.001 + (1.25e9 + 384) * 8 / 100e9) / 4096)


def test_a_borrowed_loopback_figure_is_named_in_the_cell_footnote(store: LocalStore) -> None:
    committed(store, t_total=0.4, commit=0.6, interaction={**LIVE, "live.loopback_round_seconds": 0.01,
                                                           "loopback_method": "BORROWED from r20260925-000000-0000"})
    md = V.render_markdown(V.build(store))
    assert "live-coin handling 10 ms per round, measured on loopback, counted as prover compute (BORROWED from r20260925-000000-0000)" in md


@pytest.mark.parametrize(("over", "why"), [
    ({"rounds.sequential_depth": None}, "a live-verifier result without its rounds"),
    ({"net.rtt_ms": None}, "no measured RTT"),
    ({"rtt_method": None}, "no how the RTT was measured"),
    ({"net.wait_seconds": None}, "no network wait"),
    ({"net.wait_seconds": 2.5}, "% from the model's"),
])
def test_an_interactive_result_needs_its_record_and_the_tolerance(store: LocalStore, over: dict, why: str) -> None:
    ix = {k: v for k, v in {**LIVE, **over}.items() if v is not None}
    art = committed(store, t_total=0.4, commit=0.6, interaction=ix)
    got = next(r for r in V.build(store).rejected if r["artifact"] == art)["codes"]
    assert [k for k, _ in got] == ["M"] and why in got[0][1]
    assert line(V.build(store, dataclasses.replace(V.SPEC, interaction_validated=False)))["cells"]["B-Ligero"].artifact == art


def test_the_legacy_rules_ignore_interaction_so_parity_holds(store: LocalStore) -> None:
    put(store, result(t_total=4.0))
    assert V.parity(store) == []
    assert all(c.throughput["interaction_s"] == 0.0 for c in V.build(store, V.LEGACY).cells)


def test_flock_is_its_own_family_and_column(store: LocalStore) -> None:
    assert V.candidate_for("flock union (backends/flock)") is V.FLOCK and V.candidate_for("verity-gkr gpu ... flock") .id == "A-GKR"
    doc = hashed("blake3", t_total=0.4)
    doc["workload_fingerprint"]["software"]["backend"]["name"] = "flock union, flock-128-r2 (backends/flock)"
    doc["workload_fingerprint"]["proof_class"] = "NON_ZK_PROOF"
    assert V.configuration_of(doc) is V.FLOCK_CONFIG
    views = V.build(store)
    assert V.render_json(views)["table2"]["columns"][-1] == "Flock" and line(views)["cells"]["Flock"] is None
    rows = dict((r[0], dict(zip(V.TABLE1_COLUMNS, r))) for _, r in V.table1_rows(views))
    assert rows["Flock (flock-128-r2, live coins)"]["Interaction model"].startswith("live verifier")
    assert rows["B-Ligero Fiat–Shamir (HVZK)"]["Interaction model"].startswith("non-interactive")


def test_route_a_stays_in_the_a_gkr_column_with_a_note() -> None:
    route_a = ("verity-gkr gpu (backends/gkr/gpu: checker v2 + LogUp + Ligero over BabyBear^6, SHA-512 Merkle; torch/Triton) + Flock "
               "flock-128-r2 (backends/flock/live flock-link, CPU) + the sigma link (route (a))")
    assert V.candidate_for(route_a).id == "A-GKR"
    md = V.render_markdown(V.Views(rules=V.SPEC))
    assert f"| **A-GKR** |  | " in md and V.FAMILY_NOTES["A-GKR"] in md and V.FAMILY_NOTES["Flock"] in md


FLOCK_LIVE = {"rounds.sequential_depth": 1070, "net.bytes_out": 922_000.0, "net.bytes_in": 34_240.0, "net.rtt_ms": 0.9,
              "rtt_method": "median of the session's coin round trips (same datacenter)", "net.wait_seconds": 1.071}


def test_a_pure_flock_record_fills_the_flock_column_without_renderer_changes(store: LocalStore) -> None:
    """The record contract agreed with lane flock-backend: backend name starting ``flock``, a complete class, the line's GPU SKU,
    whole-proof bound, protocol + plateau sweep, commitment timed, the interaction record, preserved dumps, a non-producer's
    verification label.  It lands in the Flock column on the keyed-BLAKE3 row line, provisional until red-team-flock grants it."""
    art = committed(store, t_total=0.4, commit=0.6, interaction=FLOCK_LIVE, backend="flock union, flock-128-r2 (backends/flock)",
                    proof_class="NON_ZK_PROOF", security={"target": -128, "achieved_log2": -195.5})
    views = V.build(store)
    cell = line(views)["cells"]["Flock"]
    assert cell is not None and cell.artifact == art and cell.configuration == V.FLOCK_CONFIG.id and cell.provisional
    assert line(views)["cells"]["B-Ligero"] is None
    assert cell.throughput["interaction_s"] == pytest.approx(1070 * 0.001 + (922_000 + 34_240) * 8 / 100e9)


def test_a_file_verified_interactive_result_pays_its_protocols_sequential_depth_and_its_bytes(store: LocalStore) -> None:
    """Every B-Ligero cell today is verified from its files: no measured RTT is needed. Its record is derived -- rounds = the
    protocol's sequential depth (3 per proof x the sub-batch proofs run one after another: ceil(N_subbatches / pipeline) waves),
    bytes = proof + statement bytes, network time = rounds x 1 ms + bytes / 100 Gb/s -- and Table 3 keeps it as the interaction
    stage, footnoted (root's ruling 2026-09-26)."""
    art = committed(store, t_total=0.4, commit=0.6, interaction={"rounds.sequential_depth": 3, "rounds.protocol": 3,
                                                                 "proof_bytes": 1.25e9, "relation.statement_bytes_total": 5e6})
    fp = store.get_manifest(art).meta["workload_fingerprint"]
    waves = math.ceil(fp.get("N_subbatches", 1) / (fp["software"]["backend"].get("pipeline") or fp.get("N_subbatches", 1)))
    views = V.build(store)
    cell = line(views)["cells"]["B-Ligero"]
    assert cell is not None and cell.artifact == art
    ix = cell.throughput["interaction"]
    assert ix["derived"] and not ix["live"] and ix["rounds"] == 3 * waves and ix["rounds_source"] == "derived: file-verified"
    assert ix["rtt_ms"] is None and ix["bytes_up"] == 1.255e9
    assert cell.throughput["interaction_s"] == pytest.approx(3 * waves * 0.001 + 1.255e9 * 8 / 100e9)
    md = V.render_markdown(views)
    share = 100 * V.stages(cell)["interaction"] / cell.overhead
    assert f"That interaction is {share:.0f}–{share:.0f}% of those cells" in md and f"{3 * waves} (derived)" in md and "— (file-verified)" in md


def test_file_verified_rounds_multiply_by_the_sequential_sub_batch_waves() -> None:
    fp = {"B": 32768, "N_subbatches": 193, "software": {"backend": {"name": "ligero", "pipeline": 4}}}
    meta = {"workload_fingerprint": fp, "measurements": [{"name": "t.total", "value": 7.0, "unit": "s"},
                                                         {"name": "rounds.sequential_depth", "value": 3, "unit": "count"}]}
    assert V.interaction(meta, interactive=True)["rounds"] == 3 * 49
    fp["software"]["backend"].pop("pipeline")
    assert V.interaction(meta, interactive=True)["rounds"] == 3 * 193
    fs = V.interaction(meta)
    assert fs["rounds"] == 0 and fs["rounds_source"] == "fiat-shamir" and fs["derived"]


def test_flocks_declared_class_is_non_zk_proof() -> None:
    assert V.FLOCK.proof_classes == ("NON_ZK_PROOF",) and "may change" in V.FLOCK_CONFIG.zk


ROUTE_A_NAME = ("verity-gkr gpu (backends/gkr/gpu: checker v2 + LogUp + Ligero over BabyBear^6, SHA-512 Merkle; torch/Triton) + Flock "
                "flock-128-r2 (backends/flock/live flock-link, CPU) + the sigma link (route (a))")
PLAIN_GKR_GPU = "verity-gkr gpu (backends/gkr/gpu: checker v2 + LogUp + Ligero over BabyBear^6, SHA-512 Merkle; torch/Triton)"
ROUTE_A_LIVE = {"rounds.sequential_depth": 4076, "net.bytes_out": 2_058_029.0, "net.bytes_in": 165_825.0, "net.rtt_ms": 0.356,
                "rtt_method": "median of 20 TCP connects to the verifier pod", "net.wait_seconds": 4076 * 0.000356 + (2_058_029 + 165_825) * 8 / 100e9}


def test_route_a_is_a_full_relation_a_gkr_configuration_and_plain_a_gkr_gpu_stays_weaker(store: LocalStore) -> None:
    """Route (a) -- A-GKR's prime proof, the operands as keyed-BLAKE3 rows proved in Flock, the sigma link -- has its own Table 1
    configuration (full relation, frame-v3/blake3-row, NON_ZK_PROOF, 2^-130.19) and fills the A-GKR column of its line; a plain
    A-GKR GPU result on the same line still reads "weaker statement"."""
    assert V.D.classify(V.D.features({"workload_fingerprint": {"software": {"backend": {"name": ROUTE_A_NAME}}}})).id == V.ROUTE_A
    assert V.D.classify(V.D.features({"workload_fingerprint": {"software": {"backend": {"name": PLAIN_GKR_GPU}}}})).id == "A-GKR GPU (torch/Triton)"
    cfg = V._CONFIG[V.ROUTE_A]
    assert cfg.family == "A-GKR" and cfg.full_relation and cfg.scheme == "frame-v3/blake3-row" and cfg.scheme_problem() is None
    assert "2^-130.19" in cfg.soundness and "decision 57" in cfg.assumptions and cfg.interaction.startswith("live verifier")
    plain = committed(store, t_total=0.5, commit=0.6, backend=PLAIN_GKR_GPU, proof_class="NON_ZK_PROOF")
    views = V.build(store)
    assert codes(views, plain)[0] == "R" and line(views)["cells"]["A-GKR"] is None
    assert line(views)["empty"]["A-GKR"]["text"] == "results only for a weaker statement (A-GKR GPU (torch/Triton))"
    art = committed(store, t_total=0.4, commit=0.6, backend=ROUTE_A_NAME, proof_class="NON_ZK_PROOF", interaction=ROUTE_A_LIVE,
                    security={"target": -128, "achieved_log2": -130.19})
    views = V.build(store)
    cell = line(views)["cells"]["A-GKR"]
    assert cell is not None and cell.artifact == art and cell.configuration == V.ROUTE_A and cell.bound_log2 == -130.19
    assert line(views)["cells"]["B-Ligero"] is None and codes(views, plain)[0] == "R"
    rows = dict((r[0], dict(zip(V.TABLE1_COLUMNS, r))) for _, r in V.table1_rows(views))
    assert rows[V.ROUTE_A]["Cells"].endswith("keyed-BLAKE3 rows · A-GKR")


def test_a_result_registered_with_its_envelope_as_payload_counts_in_the_spec_views(store: LocalStore, tmp_path) -> None:
    """Route (a)'s cell art:4b52879f is registered like cell.py rederive writes it: a lane summary as meta, the result envelope as
    the JSON payload.  The spec views read the envelope (as the drill-down does); the legacy rules read the meta only."""
    doc = hashed("blake3", t_total=0.4, security={"target": -128, "achieved_log2": -130.19})
    fp = doc["workload_fingerprint"]
    fp["software"]["backend"]["name"], fp["proof_class"], fp["zk_mode"] = ROUTE_A_NAME, "NON_ZK_PROOF", "none"
    doc["measurements"].append({"name": V.COMMIT, "value": 0.6, "unit": "s"})
    for name, value in ROUTE_A_LIVE.items():
        if name == "rtt_method":
            doc["interaction"] = {"rtt_method": value}
        else:
            doc["measurements"].append({"name": name, "value": value, "unit": {"rounds.sequential_depth": "count", "net.rtt_ms": "ms"}.get(
                name, "B" if name.startswith("net.bytes") else "s")})
    doc["protocol"], doc["sweep"] = dict(PROTOCOL), dict(SWEEP)
    files = store.put_artifact("run-files/v1", {"run_id": doc["run_id"], "dump": "proofs"})
    store.index.set_replica(files, "remote", "2026-09-25T00:00:00Z")
    (tmp_path / "result.json").write_text(json.dumps(doc))
    art = store.put_artifact(T.RESULT_KIND, {"lane": "route-a-live", "derived_by": "backends/gkr/tools/cell.py rederive"},
                             refs={"run_files": files}, file=tmp_path / "result.json")
    store.add_label(art, "independently_verified", True, VERIFIER, ts="2026-09-25T10:01:00Z")
    views = V.build(store)
    cell = line(views)["cells"]["A-GKR"]
    assert cell is not None, next((r["codes"] for r in views.rejected if r["artifact"] == art), "not seen")
    assert cell.artifact == art and cell.configuration == V.ROUTE_A
    assert art not in {c.artifact for c in V.build(store, V.LEGACY).cells} and V.parity(store) == []


# --- entities: the docs site's render -------------------------------------------------------------------------------------------------


def _entities_store(store: LocalStore) -> dict[str, str]:
    return {"blake3": committed(store, "blake3", t_total=0.4, commit=0.6, setup=2.0),
            "poseidon2": committed(store, "poseidon2", t_total=0.2, commit=0.1,
                                   labels=[("finding", "UNDER RE-VERIFICATION: R1", "coordinator", "2026-09-25T07:45:00Z")]),
            "a-gkr": put(store, result(t_total=4.0))}


def test_entities_print_every_table2_cell_and_footnote_exactly_as_the_markdown(store: LocalStore) -> None:
    _entities_store(store)
    views = V.build(store)
    md, ent = V.render_markdown(views), V.render_entities(views, rendered_at="2026-09-25T23:00:00Z", commit="abc")
    notes = dict(line.split("] ", 1) for line in md.splitlines() if line.startswith("[") and "] " in line)
    t2 = ent["tables"]["table2"]
    assert [f["text"] for f in t2["footnotes"]] == [notes[f"[{f['number']}"] for f in t2["footnotes"]]
    assert len(t2["footnotes"]) == len(notes)
    for ln in t2["lines"]:
        assert f"| {ln['label']} | {ln['native']['display']} [{ln['native']['footnote']}] |" in md
        for col in t2["columns"][1:]:
            cell = ln["cells"][col]
            assert f" {cell['display']} [{cell['footnote']}] |" in md
            assert (cell["value"] is None) == (cell["result"] is None) == (cell["state"] != "cell")
    assert ent["axes"]["backend_families"] == t2["columns"][1:] == ["A-GKR", "B-Ligero", "C-Flock", "D-SP1"]
    assert t2["column_labels"] == {"A-GKR": "A-GKR", "B-Ligero": "B-Ligero", "C-Flock": "Flock", "D-SP1": "SP1"}
    assert ent["axes"]["subcircuits"] == [c["id"] for c in ent["subcircuits"]]
    assert [c["legacy_target"] for c in ent["subcircuits"]] == list(TARGETS)


def test_entities_results_split_into_phases_that_sum_to_the_overhead_with_setup_apart(store: LocalStore) -> None:
    arts = _entities_store(store)
    ent = V.render_entities(V.build(store), rendered_at="2026-09-25T23:00:00Z", commit="abc")
    by_art = {r["artifact"]: r for r in ent["results"]}
    b3, p2 = by_art[arts["blake3"]], by_art[arts["poseidon2"]]
    for r in ent["results"]:
        assert sum(p["value"] for p in r["phases"]) == pytest.approx(r["overhead"]["value"], rel=1e-12)
        assert r["serving_overhead"] + r["proving_overhead"] + next(p["value"] for p in r["phases"] if p["id"] == "interaction") \
            == pytest.approx(r["overhead"]["value"], rel=1e-9)
        assert r["rendered_at"] == "2026-09-25T23:00:00Z" and r["network"] == "dc-1ms-100gbps" and r["subcircuit"].startswith("gemm-coordinate/k1536/")
        assert r["overhead_at"]["dc-1ms-100gbps"] == pytest.approx(r["overhead"]["value"])
    assert b3["id"] == "B-interactive@gemm-coordinate/k1536/sm89-mma-e4m3+frame-v3/blake3-keyed@bench-instances-fp8-ada-v1@rtx-4090/e4m3"
    assert b3["key"] == {"backend": "B-interactive", "statement": "gemm-coordinate/k1536/sm89-mma-e4m3+frame-v3/blake3-keyed",
                         "input_set": "bench-instances-fp8-ada-v1", "prover_hardware": "rtx-4090/e4m3"}
    assert b3["implementation"] == "B-Ligero + BLAKE3 in circuit" and b3["backend"] == "B-interactive" and b3["best"]
    assert "configuration" not in b3 and "configurations" not in ent
    assert b3["setup"]["seconds"] == 2.0 and b3["setup"]["amortized_overhead"] == pytest.approx(2.0 * b3["N"]["value"] / 4096)
    assert all(p["id"] != "setup" for p in b3["phases"])
    assert b3["state"] == {"provisional": True, "provisional_reason": b3["state"]["provisional_reason"], "algebraic": False, "algebraic_hashes": [],
                           "verified": True, "verified_by": VERIFIER, "under_reverification": None}
    assert b3["marks"] == ["provisional"] and b3["zk_class"] == "malicious-verifier-zk"
    b3cfg = next(c for c in V.configuration_records(V.build(store)) if c["id"] == "b-ligero-blake3-in-circuit")
    assert b3cfg["params"] == {"statistical-soundness": f"2^{{-{b3['soundness']['cell_bits']:.1f}}}"}
    assert p2["marks"] == ["provisional", "algebraic", "under-reverification"] and p2["state"]["algebraic"]
    assert p2["state"]["under_reverification"].startswith("UNDER RE-VERIFICATION")
    rows3 = {r["result"]: r["cells"] for r in ent["tables"]["table3"]["rows"]}
    assert rows3[b3["id"]]["setup"] == b3["setup"]["display"] and rows3[b3["id"]]["commitment"] == b3["phases"][0]["display"]


def test_entity_ids_are_stable_across_renders(store: LocalStore) -> None:
    _entities_store(store)
    views = V.build(store)
    a = V.render_entities(views, rendered_at="2026-09-25T23:00:00Z", commit="abc")
    b = V.render_entities(V.build(store), rendered_at="2026-09-26T13:00:00Z", commit="def")
    strip = lambda d: json.loads(json.dumps(d).replace("2026-09-26T13:00:00Z", "2026-09-25T23:00:00Z").replace('"def"', '"abc"'))  # noqa: E731
    assert strip(b) == json.loads(json.dumps(a))
    ids = [c["id"] for c in V.configuration_records(views)]
    assert len(ids) == len(set(ids)) == len(V.CONFIGURATIONS)
    assert len({r["id"] for r in a["results"]}) == len(a["results"])
    assert set(a["census"]["hardware"]) == {c["hardware"] for c in a["subcircuits"]}


def test_configuration_records_carry_lists_and_a_short_zk_class_and_table1_lists_backends() -> None:
    views = V.Views(rules=V.SPEC)
    ent = V.render_entities(views, rendered_at="2026-09-25T23:00:00Z", commit="abc")
    cfgs = {c["variant"]: c for c in V.configuration_records(views)}
    assert set(cfgs) == {c.id for c in V.CONFIGURATIONS}
    assert cfgs[V.FLOCK_CONFIG.id]["backend_family"] == "C-Flock" and cfgs[V.ROUTE_A]["backend_family"] == "A-GKR"
    for cfg in V.CONFIGURATIONS:
        e = cfgs[cfg.id]
        assert e["zk"] == {"class": V.zk_class(cfg), "text": cfg.zk} and e["zk"]["class"] in {c for _, c in V.ZK_CLASSES}
        assert "; ".join(e["interaction"]) == cfg.interaction and "; ".join(e["assumptions_text"]) == cfg.assumptions
        assert e["hash_class"] == ("algebraic" if cfg.algebraic() else "standard") and (e["algebraic_flag"] is not None) == bool(cfg.algebraic())
        assert e["display"]["Configuration"] == cfg.id and list(e["display"]) == list(V.TABLE1_COLUMNS)
    assert cfgs["B-Ligero Fiat–Shamir (HVZK)"]["zk"]["class"] == "hvzk" and cfgs[V.FLOCK_CONFIG.id]["zk"]["class"] == "none"
    assert cfgs["SP1 precompile (TC_DOT chip)"]["zk"]["class"] == "unknown"
    assert cfgs["B-Ligero v1"]["assumptions_text"] == ["BabyBear with its extension for challenges",
                                                   "interactive: statistical given the committed coins (HM96)"]
    assert V.split_list("a (b; c); d") == ["a (b; c)", "d"] and V.split_list("") == []
    t1 = ent["tables"]["table1"]
    assert [r["backend"] for r in t1["rows"]] == [b["id"] for b in ent["backends"]] == ent["axes"]["backends"]
    b = next(r for r in t1["rows"] if r["backend"] == "B-fs")["cells"]
    assert b["Guarantees"] == "Computational soundness, HVZK, Non-interactive, Transparent setup" and "Random oracle" in b["Assumptions"]


def test_cli_entities(store: LocalStore, capsys: pytest.CaptureFixture[str]) -> None:
    _entities_store(store)
    assert V.main(["--root", str(store.root), "--format", "entities", "--published"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert js["schema"] == V.ENTITIES_SCHEMA and js["provenance"]["published"] and js["provenance"]["banner"] == V.PUBLISHED
    assert js["provenance"]["rendered_at"].endswith("Z") and len(js["results"]) == 2


def test_entities_table1_claims_are_properties_with_the_assumptions_they_rest_on_and_their_declared_roles() -> None:
    from verity import claims

    ent = V.render_entities(V.Views(rules=V.SPEC), rendered_at="2026-09-25T23:00:00Z", commit="abc")
    cfgs = {c["variant"]: c for c in V.configuration_records(V.Views(rules=V.SPEC))}
    for e in cfgs.values():
        assert [c["property"]["id"] for c in e["claims"] if c["property"]["id"]] == e["properties"]
        for c in e["claims"]:
            p = c["property"]
            assert p["text"] and p["unmapped"] == (p["id"] is None) and (p["id"] is None or claims.get(p["id"]).kind == "guarantee")
            for a in c["assumptions"]:
                assert a["text"] and isinstance(a["role"], str) and a["unmapped"] == (a["id"] is None)
                assert a["id"] is None or claims.get(a["id"]).kind == "assumption"
        assert e["assumptions"] == list(dict.fromkeys(a["id"] for c in e["claims"] for a in c["assumptions"] if a["id"]))
        assert e["models"] in (["fiat-shamir"], ["live-verifier"])
    by = lambda variant: {c["property"]["id"]: c["assumptions"] for c in cfgs[variant]["claims"]}  # noqa: E731
    # interactive B-Ligero: statistical soundness on its hashes and bounds; ZK on the coin commitment; designated verifier
    v1 = by("B-Ligero v1")
    assert list(v1) == ["statistical-soundness", "malicious-verifier-zk", "designated-verifier", "transparent-setup"]
    assert [(a["id"], a["role"]) for a in v1["malicious-verifier-zk"]] == [("cr/blake2b", "Fiat–Shamir / coins (coin commitments)")]
    roles = {a["id"]: a["role"] for a in v1["statistical-soundness"] if a["id"]}
    assert roles["cr/blake3"] == "proof-internal commitment" and roles["xof/shake-256"] == "Fiat–Shamir / coins (PRG of the verifier's coins)"
    assert roles["rs-proximity-udr"] == "Ligero column tests (interleaved RS, rate 1/4)"
    assert all(a["role"] == "" for a in v1["statistical-soundness"] if a["id"] is None and "BabyBear" in a["text"])
    assert v1["designated-verifier"] == [] and v1["transparent-setup"] == []
    # its Fiat–Shamir sibling is a separate entry: computational soundness, HVZK on an honest verifier, non-interactive on the oracle
    fs = by("B-Ligero Fiat–Shamir (HVZK)")
    assert list(fs) == ["computational-soundness", "hvzk", "non-interactive", "transparent-setup"]
    assert fs["hvzk"] == [{"id": "honest-verifier", "role": "", "text": cfgs["B-Ligero Fiat–Shamir (HVZK)"]["zk"]["text"], "unmapped": False}]
    assert {a["id"] for a in fs["non-interactive"]} == {"random-oracle"} and len(fs["non-interactive"]) == 2
    assert [a["role"] for a in fs["non-interactive"]] == ["Fiat–Shamir / coins (Fiat–Shamir)", "Fiat–Shamir / coins"]
    assert fs["non-interactive"] == [a for a in fs["computational-soundness"] if a["id"] == "random-oracle"]
    # an in-circuit configuration whose ZK text names no coin commitment rests its ZK on nothing the code declares
    assert by("B-Ligero + BLAKE3 in circuit")["malicious-verifier-zk"] == []
    rt = cfgs[V.ROUTE_A]
    assert {"cr/sha-256", "cr/blake3", "cr/sha-512", "rs-proximity-udr", "rs-proximity-johnson"} <= set(rt["assumptions"])
    assert rt["properties"] == ["statistical-soundness", "no-zk", "designated-verifier", "transparent-setup"]
    assert "SHA-256 (Fiat–Shamir / coins; live transcript and root_F binding; standard)" in rt["unmapped"]
    sp1 = cfgs["SP1 stock (relation-only / bare guest)"]
    assert sp1["properties"] == ["computational-soundness", "non-interactive", "transparent-setup"] and sp1["claims"][1]["property"]["unmapped"]
    assert sp1["assumptions"][:1] == ["cr/poseidon2-koalabear"] and "random-oracle" in sp1["assumptions"]
    assert "logup" in cfgs["A-GKR CPU (Rust)"]["assumptions"]
    assert set(ent["claims"]) == {*(i for b in ent["backends"] for k in ("properties", "assumptions") for i in b[k]),
                                  *(i for st in ent["statements"] for i in st["assumptions"])}
    assert ent["claims"]["cr/sha-512"]["label"] == "CR · SHA-512"                   # A-GKR's proof-internal Merkle trees
    assert not any(i.startswith("ro/") for e in cfgs.values() for i in e["assumptions"])
    retired = ("beacon", "auditor", "challenger")
    assert not any(r in i for i in ent["claims"] for r in retired)


def test_entities_backends_are_backend_families_by_interaction_mode_with_their_claims_in_full() -> None:
    ent = V.render_entities(V.Views(rules=V.SPEC), rendered_at="2026-09-25T23:00:00Z", commit="abc")
    cfgs = {c["variant"]: c for c in V.configuration_records(V.Views(rules=V.SPEC))}
    inst = {i["id"]: i for i in ent["backends"]}
    assert list(inst) == ["A-route-a", "A-fs", "B-interactive", "B-fs", "C-interactive", "D-fs"]
    assert [f["id"] for f in ent["backend_families"]] == ["A-GKR", "B-Ligero", "C-Flock", "D-SP1"] == ent["axes"]["backend_families"]
    assert {f["id"]: f["column"] for f in ent["backend_families"]} == {"A-GKR": "A-GKR", "B-Ligero": "B-Ligero", "C-Flock": "Flock", "D-SP1": "SP1"}
    for i in inst.values():
        assert i["display"] == i["id"] and (i["id"] == f"{i['backend_family'][0]}-{i['mode']}" or i["composed"])
        members = [cfgs[c] for c in i["implementations"]]
        assert members and all(m["backend_family"] == i["backend_family"] and m["backend"] == i["id"] for m in members)
        for slot, claim in enumerate(i["claims"]):
            ids = {m["claims"][slot]["property"]["id"] for m in members}
            assert claim["property"]["id"] == (ids.pop() if len(ids) == 1 else None)
            union = {a["id"] or a["text"] for m in members for a in m["claims"][slot]["assumptions"] if not a["role"].startswith("statement commitment")}
            own = {a["id"] or a["texts"][0] for a in claim["assumptions"] if a["from"] in (None, "link") and a["implementations"][0] in i["implementations"]}
            assert own == union or (i["composed"] and own >= union)
            assert all(set(a["implementations"]) <= set(i["implementations"]) for a in claim["assumptions"] if a["from"] != "C-Flock")
            assert not any(r.startswith("statement commitment") for a in claim["assumptions"] for r in a["roles"])
    assert {c["variant"] for c in cfgs.values() if c["backend_family"]} == {c for i in inst.values() for c in i["implementations"]}
    b, bfs = inst["B-interactive"], inst["B-fs"]
    assert b["properties"] == ["statistical-soundness", "malicious-verifier-zk", "designated-verifier", "transparent-setup"]
    assert bfs["properties"] == ["computational-soundness", "hvzk", "non-interactive", "transparent-setup"] and bfs["implementations"] == ["B-Ligero Fiat–Shamir (HVZK)"]
    zk = next(a for a in b["claims"][1]["assumptions"] if a["id"] == "cr/blake2b")
    assert zk["roles"] == ["Fiat–Shamir / coins (coin commitments)"] and "B-Ligero v1" in zk["implementations"]
    assert inst["A-route-a"]["implementations"] == [V.ROUTE_A] and len(inst["A-fs"]["implementations"]) == 2
    assert "cr/blake3" not in inst["C-interactive"]["assumptions"]                 # C-Flock's BLAKE3 rows are its statement's
    ra = inst["A-route-a"]
    assert ra["composed"] and ra["backend_family"] == "A-GKR" and ra["column"] == "A-GKR" and ra["components"] == ["C-Flock", "link"]
    froms = {a["from"] for a in ra["claims"][0]["assumptions"]}
    assert {"C-Flock", "link"} <= froms
    link = [a for a in ra["claims"][0]["assumptions"] if a["from"] == "link"]
    assert any("root_F" in a["texts"][0] for a in link) and any(a["texts"][0].startswith("sigma link 2^-243.9") for a in link)
    assert inst["D-fs"]["claims"][1]["property"]["unmapped"] and len(inst["D-fs"]["claims"][1]["property"]["texts"]) >= 2
    c0 = next(c for c in ent["subcircuits"] if c["legacy_target"] == NVFP4_SM120.name)
    assert (c0["hardware_name"], c0["dtype_display"], c0["hardware_display"]) == ("RTX 5090", "FP4 (NVFP4, E2M1)", "RTX 5090 · FP4 (NVFP4, E2M1)")
    assert c0["peak"] == {"value": 1676e12, "unit": "FLOP/s", "display": "1676 T"}
    assert next(c for c in ent["subcircuits"] if c["legacy_target"] == BF16_HOPPER.name)["hardware_display"] == "H100 SXM5 · BF16"
    h8 = next(c for c in ent["subcircuits"] if c["legacy_target"] == FP8_HOPPER_WGMMA.name)
    assert h8["id"] == "gemm-coordinate/k1536/sm90-wgmma-e4m3" and h8["subcircuit_template"] == "gemm-coordinate"
    assert h8["display"] == "GemmCoordinate<1536> · sm90 wgmma · FP8 (E4M3)"
    assert h8["parameters"] == {"K": 1536, "datatype": "e4m3", "semantics": "sm90.wgmma.m64n8k32.e4m3"} and h8["accumulate"] == "fp32"
    assert not {"subcircuit_kind", "silicon_target", "target", "instruction"} & set(h8)
    assert all(c["parameters"]["semantics"] == TARGETS[c["legacy_target"]].instruction for c in ent["subcircuits"])


def test_entities_best_standard_hash_cell_per_backend_family_and_subcircuit(store: LocalStore) -> None:
    arts = _entities_store(store)
    fast = committed(store, "blake3", t_total=0.1, commit=0.1)
    ent = V.render_entities(V.build(store), rendered_at="2026-09-25T23:00:00Z", commit="abc")
    best = {(b["backend_family"], b["subcircuit"]): b for b in ent["best_standard_hash"]}
    ada = "gemm-coordinate/k1536/sm89-mma-e4m3"
    by_art = {r["artifact"]: r for r in ent["results"]}
    pick = best[("B-Ligero", ada)]
    assert pick["result"] == by_art[fast]["id"] and pick["scheme"] == "frame-v3/blake3-keyed" and pick["backend"] == "B-interactive"
    assert by_art[arts["poseidon2"]]["id"] not in {b["result"] for b in best.values()}                   # algebraic: never a standard-hash pick
    assert arts["blake3"] not in by_art and not by_art[fast]["best"] is False                             # one result per backend: its best
    assert all(by_art_r["overhead"]["value"] >= pick["overhead"]["value"] for by_art_r in ent["results"]
               if by_art_r["backend_family"] == "B-Ligero" and by_art_r["subcircuit"] == ada and not by_art_r["state"]["algebraic"])
    assert [r["backend_family"] for r in ent["results"] if r["artifact"] == arts["a-gkr"]] in ([], ["A-GKR"])


def test_entities_schemes_carry_the_hash_assumptions_their_configurations_declare(store: LocalStore) -> None:
    _entities_store(store)
    ent = V.render_entities(V.build(store), rendered_at="2026-09-25T23:00:00Z", commit="abc")
    sch = {s["id"]: s for s in ent["schemes"]}
    assert [x["legacy_id"] for x in sch.values()] == [s.id for s in V.SCHEMES]
    assert list(sch) == ["frame-v3/sha-256", "frame-v3/blake3-keyed", "vllm-v1/sha-256", "frame-v3/poseidon2-babybear",
                         "frame-v3/poseidon2-babybear-tile64", "frame-v3/ajtai"]
    assert sch["frame-v3/blake3-keyed"]["assumptions"] == ["cr/sha-256", "cr/blake3"]           # frame-v3's trees are SHA-256
    assert sch["frame-v3/sha-256"]["assumptions"] == ["cr/sha-256"] and sch["vllm-v1/sha-256"]["assumptions"] == ["cr/sha-256"]
    assert sch["frame-v3/poseidon2-babybear"]["assumptions"] == ["cr/sha-256", "cr/poseidon2-babybear"]
    assert all(x["tree_hash"] == "sha-256" for x in sch.values())
    assert {k: (x["leaf_hashes"], x["algebraic"]) for k, x in sch.items() if k != "frame-v3/ajtai"} == {
        "frame-v3/sha-256": (["sha-256"], False), "frame-v3/blake3-keyed": (["blake3"], False), "vllm-v1/sha-256": (["sha-256"], False),
        "frame-v3/poseidon2-babybear": (["poseidon2-babybear"], True), "frame-v3/poseidon2-babybear-tile64": (["poseidon2-babybear"], True)}
    b3 = sch["frame-v3/blake3-keyed"]
    assert b3["name"] == "frame-v3 · keyed-BLAKE3 rows" and b3["description"].startswith("frame-v3 SHA-256 trees over") and ";" not in b3["description"]
    assert "statement commitment (frame-v3 trees)" in b3["assumption_details"][0]["roles"]
    assert {"B-Ligero + BLAKE3 in circuit", V.ROUTE_A, V.FLOCK_CONFIG.id} <= set(b3["implementations"])
    assert all(r["scheme"] in sch for r in ent["results"])
    from verity import claims
    assert all(claims.get(i).kind == "assumption" for s in sch.values() for i in s["assumptions"])


def test_entities_results_are_every_admissible_backends_best_with_keys_input_sets_soundness_and_phases(store: LocalStore) -> None:
    arts = _entities_store(store)
    ent = V.render_entities(V.build(store), rendered_at="2026-09-25T23:00:00Z", commit="abc")
    by_art = {r["artifact"]: r for r in ent["results"]}
    b3 = by_art[arts["blake3"]]
    stmt = "gemm-coordinate/k1536/sm89-mma-e4m3+frame-v3/blake3-keyed"
    assert b3["key"] == {"backend": "B-interactive", "statement": stmt, "input_set": "bench-instances-fp8-ada-v1", "prover_hardware": "rtx-4090/e4m3"}
    st = {x["id"]: x for x in ent["statements"]}
    assert st[stmt]["subcircuit"] == "gemm-coordinate/k1536/sm89-mma-e4m3" and st[stmt]["scheme"] == "frame-v3/blake3-keyed"
    assert st[stmt]["assumptions"] == ["cr/sha-256", "cr/blake3"] and b3["assumptions"]["statement"] == st[stmt]["assumptions"]
    backend = next(x for x in ent["backends"] if x["id"] == "B-interactive")
    assert b3["assumptions"]["backend"] == backend["assumptions"]
    assert b3["assumptions"]["all"] == list(dict.fromkeys([*backend["assumptions"], "cr/sha-256", "cr/blake3"]))
    assert set(ent["axes"]["statements"]) == set(st) and all(x["id"] == f"{x['subcircuit']}+{x['scheme']}" for x in st.values())
    assert b3["hardware"] == b3["prover_hardware"] == "rtx-4090/e4m3" and b3["N"]["peak"] == "rtx-4090/e4m3"
    assert b3["N"]["work_model"] == "work-model/v0" and b3["P"]["value"] == b3["throughput"]["instances_per_second"]
    assert b3["overhead"]["value"] == pytest.approx(b3["N"]["value"] / b3["P"]["value"])
    assert b3["input_set"] == {"id": "bench-instances-fp8-ada-v1", "provenance": "synthetic", "range": [0, 4096], "how": "frozen"}
    assert b3["soundness"]["cell_log2"] == b3["soundness"]["cell_bits"] * -1 and b3["soundness"]["proofs"] == b3["batching"]["proofs"]
    assert "per_proof_log2" in b3["soundness"]
    kinds = {p["id"]: p["kind"] for p in b3["phases"]}
    assert kinds["commitment"] == "stage" and "setup" not in kinds and b3["setup"]["seconds"] == 2.0
    other = next(p for p in b3["phases"] if p["id"] == "other")
    assert other["raw"] == b3["other_raw"] and not other["flagged"]
    assert sum(p.get("raw", p["value"]) for p in b3["phases"]) == pytest.approx(b3["overhead"]["value"], rel=1e-12)
    # every admissible result appears once per backend, and the Table 2 cell names its backend and result
    p2 = by_art[arts["poseidon2"]]
    assert p2["backend"] == "B-interactive" and p2["scheme"] == "frame-v3/poseidon2-babybear" and p2["best"]
    t2 = {(ln["subcircuit"], ln["scheme"]): ln for ln in ent["tables"]["table2"]["lines"]}
    cell = t2[("gemm-coordinate/k1536/sm89-mma-e4m3", "frame-v3/blake3-keyed")]["cells"]["B-Ligero"]
    assert cell["backend"] == "B-interactive" and cell["result"] == b3["id"]
    assert t2[("gemm-coordinate/k1536/sm89-mma-e4m3", "frame-v3/blake3-keyed")]["native"]["peak"] == "rtx-4090/e4m3"
    rows3 = {r["result"]: r for r in ent["tables"]["table3"]["rows"]}
    assert rows3[b3["id"]]["key"]["statement"] == stmt and rows3[b3["id"]]["key"]["prover_hardware"] == "rtx-4090/e4m3"
    assert rows3[b3["id"]]["backend"] == "B-interactive"
    assert t2[("gemm-coordinate/k1536/sm89-mma-e4m3", "frame-v3/blake3-keyed")]["statement"] == stmt
    sub = {s["id"]: s for s in ent["subcircuits"]}
    assert sub["gemm-coordinate/k1536/sm90-mma-bf16"]["fidelity_notes"][0]["note"].startswith("H100 BF16 models mma")
    assert sub["gemm-coordinate/k1536/sm80-mma-bf16"]["input_set_provenance"] == "captured"


def test_a_second_admissible_result_of_another_backend_is_emitted_beside_the_familys_best(store: LocalStore) -> None:
    fast = committed(store, "blake3", t_total=0.1, commit=0.1)
    slow = committed(store, "blake3", t_total=0.5, commit=0.5)
    ent = V.render_entities(V.build(store), rendered_at="2026-09-25T23:00:00Z", commit="abc")
    arts = {r["artifact"] for r in ent["results"]}
    assert fast in arts and slow not in arts                                    # same backend (B-interactive): only its best
    views = V.build(store)
    assert {c.artifact for c in views.admissible} >= {fast, slow} and [c.artifact for c in views.cells] == [fast]


def test_blake3_xob_commits_keyed_blake3_bytes_so_it_is_a_circuit_of_one_scheme() -> None:
    """Decided from the code: ligero-verify's ``BLAKE3_XOB`` leaf scheme has ``BLAKE3``'s schema and params digest (the
    committed object), so ``frame-v3/blake3-keyed`` lists both circuits and no scheme id is added for xob."""
    import pathlib
    import re as _re

    src = (pathlib.Path(V.__file__).resolve().parents[4] / "ligero-verify" / "src" / "leaf.rs").read_text(encoding="utf-8")
    leaf = {m[0]: (m[1], m[2]) for m in _re.findall(r'name: "([^"]+)",\n(?:.*\n)*?    schema: "([^"]+)",\n    params_sha256: "([0-9a-f]+)"', src)}
    assert leaf["blake3-xob"] == leaf["blake3"]
    ent = V.render_entities(V.Views(rules=V.SPEC), rendered_at="x", commit="x")
    sch = {s["id"]: s for s in ent["schemes"]}
    assert sch["frame-v3/blake3-keyed"]["leaf_circuits"] == ["blake3", "blake3-xob"] and not any("xob" in k for k in sch)
    for s in sch.values():                                                     # each scheme's circuits commit one object
        assert len({leaf[c] for c in s["leaf_circuits"]}) <= (2 if s["id"] == "frame-v3/ajtai" else 1)


def test_a_fused_stage_is_one_span_and_other_keeps_its_raw_value(store: LocalStore) -> None:
    art = committed(store, "blake3", t_total=0.4, commit=0.6, joints={("witness", "encoding_commitment"): 0.2})
    ent = V.render_entities(V.build(store), rendered_at="2026-09-25T23:00:00Z", commit="abc")
    r = next(x for x in ent["results"] if x["artifact"] == art)
    spans = [p for p in r["phases"] if p["kind"] == "span"]
    assert len(spans) == 1 and spans[0]["covers"] == ["witness", "encoding_commitment"]
    assert (spans[0]["from"], spans[0]["to"]) == ("witness", "encoding_commitment")
    assert not {"witness", "encoding_commitment"} & {p["id"] for p in r["phases"] if p["kind"] == "stage"}
    assert r["other_raw"] is not None and sum(p.get("raw", p["value"]) for p in r["phases"]) == pytest.approx(r["overhead"]["value"], rel=1e-12)
    rows3 = {x["result"]: x["cells"] for x in ent["tables"]["table3"]["rows"]}
    assert rows3[r["id"]]["witness"].startswith("witness+encoding_commitment: ") and rows3[r["id"]]["encoding_commitment"] == T.JOINT_CONTINUATION


def test_every_backends_best_is_kept_not_only_the_familys(store: LocalStore) -> None:
    """Two backends of one family on one statement: both results are emitted; Table 2's cell stays the family's best."""
    fast = committed(store, "blake3", t_total=0.1, commit=0.1)
    slow = committed(store, "blake3", t_total=0.5, commit=0.5)
    views = V.build(store)
    assert [c.artifact for c in views.cells] == [fast]
    one = V.best_per_key(views.admissible, lambda c: "B-interactive")
    two = V.best_per_key(views.admissible, lambda c: "B-interactive" if c.artifact == fast else "B-other")
    assert [c.artifact for c in one] == [fast] and sorted(c.artifact for c in two) == sorted([fast, slow])


def test_a_results_interaction_record_recomputes_its_overhead_at_any_network(store: LocalStore) -> None:
    rtt, rounds = 0.000356, 4076
    live = committed(store, "blake3", t_total=16.0, commit=0.6, proof_class="NON_ZK_PROOF", backend="flock (flock-128-r2)",
                     interaction={"rounds.sequential_depth": rounds, "rounds.protocol": 2 * rounds, "net.rtt_ms": rtt * 1e3,
                                  "rtt_method": "median of 20 TCP connects", "net.bytes_out": 2_058_029, "net.bytes_in": 165_825,
                                  "net.wait_seconds": rounds * rtt + (2_058_029 + 165_825) * 8 / 100e9 + rounds * 0.0001,
                                  "live.loopback_round_seconds": 0.0001, "loopback_method": "measured on loopback, this run"})
    filed = committed(store, "poseidon2", t_total=0.5, commit=0.3)
    views = V.build(store, dataclasses.replace(V.SPEC, interaction_validated=False))
    ent = V.render_entities(views, rendered_at="2026-09-25T23:00:00Z", commit="abc")
    by_art = {r["artifact"]: r for r in ent["results"]}
    for art in (live, filed):
        r = by_art[art]
        ix = r["interaction"]
        at = lambda rtt_s, bw: r["N"]["value"] * (ix["commit_s"] + ix["prover_compute_s"] + ix["verifier_compute_s"] + ix["rounds"] * rtt_s  # noqa: E731
                                                   + (ix["bytes_up"] + ix["bytes_down"]) * 8 / bw) / ix["inputs"]
        assert at(0.001, 100e9) == pytest.approx(r["overhead"]["value"], rel=1e-9)
        for n in ("rtt-20ms-100gbps", "dc-1ms-10gbps"):
            prof = ent["census"]["networks"][n]
            assert at(prof["rtt_s"], prof["bandwidth_bps"]) == pytest.approx(r["overhead_at"][n], rel=1e-9)
    ix = by_art[live]["interaction"]
    assert ix["live"] and ix["rounds"] == ix["sequential_depth"] == rounds and ix["rounds_protocol"] == 2 * rounds
    assert (ix["bytes_up"], ix["bytes_down"]) == (2_058_029, 165_825)
    assert ix["measured_rtt"] == {"ms": pytest.approx(rtt * 1e3), "method": "median of 20 TCP connects"}
    assert ix["loopback_per_round"] == {"s": 0.0001, "method": "measured on loopback, this run"}
    assert ix["loopback_moved_s"] == pytest.approx(rounds * 0.0001)
    assert ix["prover_compute_s"] == pytest.approx(16.0 + rounds * 0.0001) and ix["verifier_compute_s"] == 0.0
    assert "instances" not in ix and ix["inputs"] == 4096 and ix["formula"].endswith("/ inputs")
    fx = by_art[filed]["interaction"]
    assert fx["derived"] and fx["rounds_source"] == "derived: file-verified" and fx["measured_rtt"] is None and fx["loopback_per_round"] is None
