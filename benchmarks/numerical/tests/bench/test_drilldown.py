"""Drill-downs D1-D3 (non-canonical; layout fixed 2026-09-24, campaign morning-tables BRIEF §1): the VARIANTS registry recognises
the real backend names and never two variants at once, D2 takes its ✓ / letter from tables.py's predicate, D3 reads only the cell's
result envelope -- over an isolated store root."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from verity.ml.circuits.targets.target import FIRST, FP8_ADA_MMA
from verity_numerical.bench import contract
from verity_numerical.bench import drilldown as D
from verity_numerical.bench import tables as T

pytest.importorskip("research.store.local", reason="research.store (tools/research) is not importable")
from research.store.local import LocalStore  # noqa: E402
from test_tables import RTX4090, VERIFIER, cell, hashed, put, put_registered, result, store  # noqa: E402, F401  (store: the fixture)

SEEDS = ("B-Ligero v1", "B-Ligero v3 (private selection)", "B-Ligero v3 fused hints", "B-Ligero v3x4 fused (l=4096)", "B-Ligero v1x4 (fold)",
         "B-Ligero v3x4 (fold)", "B-Ligero v2 (public selection)", "B-Ligero v2x4 (public selection)", "B-Ligero Fiat–Shamir (HVZK)",
         "B-Ligero + Poseidon2 per row", "B-Ligero + Poseidon2 shared 64x64 tile", "B-Ligero + BLAKE3 in circuit",
         "B-Ligero + SHA-256 in circuit", "B-Ligero + vllm-v1 SHA-256 in circuit", "B-Ligero + Ajtai n=64", "B-Ligero + Ajtai n=128", "B-Ligero native external authentication",
         "Ligerito", "A-GKR CPU (Rust)", "A-GKR GPU (torch/Triton)", "A-GKR route (a): prime + Flock rows + sigma link",
         "SP1 stock (relation-only / bare guest)", "SP1 stock, FRI query count raised (otherwise unmodified)",
         "SP1 stock, committed (frame-v3 SHA-256 rows)", "SP1 stock, committed (vllm-v1 SHA-256 position leaves)", "SP1 precompile (TC_DOT chip)")
FOLDED = "b-ligero chain, fp8-ada relation (archive/direct/ligero/fp8); folded: 4 instruction steps per column at FP8Params.from_model(ada_e4m3_m16n8k32x4)"
V3 = "b-ligero chain, bf16-ampere-v3 relation (archive/direct/ligero/privsel/relation.py at params.from_model(ampere_bf16_m16n8k16))"
V3X4 = "b-ligero chain, fp8-ada-v3x4 relation (archive/direct/ligero/privsel/relation.py at fp8params.from_model(ada_e4m3_m16n8k32x4))"
V2 = "b-ligero chain, fp8-ada-v2 relation (archive/direct/ligero/pubsel/relation.py at fp8params.from_model(ada_e4m3_m16n8k32))"
V2X4 = "b-ligero chain, bf16-hopper-v2x4 relation (archive/direct/ligero/pubsel/relation.py at params.from_model(hopper_bf16_m16n8k16x4))"
FP8 = "b-ligero chain, fp8-ada relation (archive/direct/ligero/fp8)"


def env(name: str, *, auth: str | None = "excluded", proof_class: str = "COMPLETE_ZK_BACKEND", bound: str = "interactive: statistical",
        l: int = 16384, **fp: Any) -> dict[str, Any]:
    return {"workload_fingerprint": {"software": {"backend": {"name": name}}, "proof_class": proof_class, "authentication": auth,
                                     "security": {"bound_model": bound, "rs_l": l}, **fp}}


def leaf(scheme: str, h: str, schema: str) -> dict[str, Any]:
    return {"hash": h, "leaf": {"scheme": scheme, "schema": schema}}


P2 = leaf("poseidon2", "poseidon2-babybear-w24", "poseidon2-babybear-w24/row/v2h")
CASES = [
    (env(FP8), {}, "B-Ligero v1"),
    (env("b-ligero chain (archive/direct/ligero)", auth=None), {}, "B-Ligero v1"),
    (env(FOLDED), {}, "B-Ligero v1x4 (fold)"),
    (env(V3), {}, "B-Ligero v3 (private selection)"),
    (env(V3.replace("bf16-ampere-v3", "fp8-ada-v3")), {"hints": "fused"}, "B-Ligero v3 fused hints"),
    (env(V3X4, l=4096), {"hints": "fused"}, "B-Ligero v3x4 fused (l=4096)"),
    (env(V3X4, l=4096, software={"backend": {"name": V3X4, "hints": "fused device kernel"}}), {}, "B-Ligero v3x4 fused (l=4096)"),
    (env(V3X4, l=16384), {"hints": "fused"}, "B-Ligero v3x4 (fold)"),
    (env(V3X4, l=4096), {}, "B-Ligero v3x4 (fold)"),
    (env(V2), {}, "B-Ligero v2 (public selection)"),
    (env(V2X4), {}, "B-Ligero v2x4 (public selection)"),
    (env(FP8, bound="non-interactive: Fiat–Shamir"), {}, "B-Ligero Fiat–Shamir (HVZK)"),
    (env(FP8, proof_class="COMPLETE_HVZK_BACKEND", bound="x"), {}, "B-Ligero Fiat–Shamir (HVZK)"),
    (env(FP8, auth="included-hash", sharing="none", **P2), {}, "B-Ligero + Poseidon2 per row"),
    (env(FP8, auth="included-hash", sharing="none", hash="poseidon2-babybear-w24"), {}, "B-Ligero + Poseidon2 per row"),
    (env(FP8, auth="included-hash-shared", sharing="tile64x64", **P2), {}, "B-Ligero + Poseidon2 shared 64x64 tile"),
    (env(FP8, auth="included-hash", sharing="tile64x64", **P2), {}, "B-Ligero + Poseidon2 shared 64x64 tile"),
    (env(FP8, auth="included-hash", sharing="none", **leaf("blake3", "blake3-keyed", "blake3-keyed/row/v2")), {}, "B-Ligero + BLAKE3 in circuit"),
    (env(FP8, auth="included-hash", sharing="none", **leaf("sha256", "sha256", "sha256/row/v1")), {}, "B-Ligero + SHA-256 in circuit"),
    (env(FP8, auth="included-hash", sharing="none", **leaf("vllm-v1", "sha256", "vllm-v1/pos-leaf/v0")), {}, "B-Ligero + vllm-v1 SHA-256 in circuit"),
    (env(FP8, auth="included-hash", sharing="none", **leaf("ajtai-n64", "ajtai-babybear-n64-b1-r0/v1", "ajtai-babybear-n64-b1-r0/row/v1")), {},
     "B-Ligero + Ajtai n=64"),
    (env(FP8, auth="included-hash", sharing="none", **leaf("ajtai-n128", "ajtai-babybear-n128-b1-r0/v1", "ajtai-babybear-n128-b1-r0/row/v1")), {},
     "B-Ligero + Ajtai n=128"),
    (env("b-ligero chain (archive/direct/ligero)", auth="included"), {}, "B-Ligero native external authentication"),
    (env("ligerito", proof_class="NON_ZK_PROOF_DIAGNOSTIC"), {}, "Ligerito"),
    (env("verity-gkr", proof_class="NON_ZK_PROOF_DIAGNOSTIC", software={"backend": {"name": "verity-gkr"}, "toolchain": {"rustc": "1.8"}}), {},
     "A-GKR CPU (Rust)"),
    (env("verity-gkr-gpu (archive/gkr/gpu, torch + triton)", proof_class="NON_ZK_PROOF_DIAGNOSTIC"), {}, "A-GKR GPU (torch/Triton)"),
    (env("verity-gkr", proof_class="NON_ZK_PROOF_DIAGNOSTIC",
         software={"backend": {"name": "verity-gkr"}, "toolchain": {"python": "3.12", "torch": "2.5", "triton": "3.1"}}), {}, "A-GKR GPU (torch/Triton)"),
    (env("sp1", proof_class="NON_ZK_PROOF"), {}, "SP1 stock (relation-only / bare guest)"),
    (env("sp1 tc-dot (sp1 6.4.0 + TC_DOT chip)", proof_class="NON_ZK_PROOF"), {}, "SP1 precompile (TC_DOT chip)"),
    (env("sp1-tcdot", proof_class="NON_ZK_PROOF"), {}, "SP1 precompile (TC_DOT chip)"),
    (env("sp1 relation-bare/v2 (SP1, FRI query count raised to a 134-bit target (otherwise unmodified); core STARK, CUDA prover)",
         proof_class="NON_ZK_PROOF"), {}, "SP1 stock, FRI query count raised (otherwise unmodified)"),
    (env("sp1 relation-committed/v1 (stock SP1 6.4.0, frame-v3 SHA-256 rows; core STARK, CUDA prover)", proof_class="NON_ZK_PROOF",
         auth="included-hash", statement="relation-committed/v1"), {}, "SP1 stock, committed (frame-v3 SHA-256 rows)"),
    (env("sp1 relation-committed-vllm/v1 (stock SP1 6.4.0, vllm-v1 SHA-256 position leaves; core STARK, CUDA prover)",
         proof_class="NON_ZK_PROOF", auth="included-hash", statement="relation-committed-vllm/v1"), {},
     "SP1 stock, committed (vllm-v1 SHA-256 position leaves)"),
]


# --- D1: the registry -------------------------------------------------------------------------------------------------------------------


def test_variants_registry_holds_the_seeds_with_every_d1_column() -> None:
    assert tuple(v.id for v in D.VARIANTS) == SEEDS
    for v in D.VARIANTS:
        row = v.d1_row()
        assert len(row) == len(D.D1_COLUMNS) and all(isinstance(x, str) and x.strip() for x in row), v.id
        assert v.recognise and v.sources and v.candidate in (None, *(c.id for c in T.CANDIDATES)), v.id


@pytest.mark.parametrize(("doc", "meta", "want"), CASES, ids=[c[2] + f"-{i}" for i, c in enumerate(CASES)])
def test_variants_recognise_the_real_backend_names(doc: dict[str, Any], meta: dict[str, Any], want: str) -> None:
    f = D.features(doc, meta)
    got = D.classify(f)
    assert got is not None and got.id == want, (f, got)
    cand = T.candidate_for(doc["workload_fingerprint"]["software"]["backend"]["name"])
    assert (cand.id if cand else None) == got.candidate          # A-GKR's and SP1's variants reach their Table 2 columns


def test_variant_recognisers_are_mutually_exclusive() -> None:
    keys = ("family", "fold", "fused", "l", "mode", "proof_class", "authentication", "leaf", "sharing")
    b = itertools.product(("v1", "v2", "v3", None), (1, 4), (False, True), (4096, 16384), ("interactive", "fiat-shamir", None),
                          ("COMPLETE_ZK_BACKEND", "COMPLETE_HVZK_BACKEND"), (None, "excluded", "included", "included-hash", "included-hash-shared"),
                          (None, "poseidon2", "blake3", "ajtai-n64", "ajtai-n128"), (None, "none", "tile64x64"))
    grid = [{"candidate": "B-Ligero", "device": None, "tcdot": False, **dict(zip(keys, c))} for c in b]
    grid += [{**dict.fromkeys(keys), "fold": 1, "fused": False, "candidate": cand, "device": dev, "tcdot": tc, "authentication": auth,
              "sec_raised": sr}
             for cand, dev, tc, auth, sr in itertools.product(("A-GKR", "SP1", "Ligerito", None), ("cpu", "gpu", None), (False, True),
                                                              (None, "excluded", "included"), (False, True))]
    for f in grid:
        hits = [v.id for v in D.VARIANTS if v.match(f)]
        assert len(hits) <= 1, (f, hits)


# --- D2: fastest per variant x target, marked by tables.py's predicate -------------------------------------------------------------------

LIGERO_4090 = dict(candidate="B-Ligero", target=FP8_ADA_MMA, gpu=RTX4090, proof_class="COMPLETE_ZK_BACKEND")


def d2_cell(md: str, variant: str, column: str) -> str:
    d2 = md.split("## D2", 1)[1].split("## D3", 1)[0].splitlines()
    head = next(line for line in d2 if line.startswith("| Variant |"))
    row = next(line for line in d2 if line.startswith(f"| {variant} |"))
    cols = [c.strip() for c in head.strip("|").split("|")]
    return [c.strip() for c in row.strip("|").split("|")][cols.index(column)]


def footnote(md: str, cell: str) -> str:
    n = cell.rsplit("[", 1)[1].rstrip("]")
    return next(line for line in md.splitlines() if line.startswith(f"[{n}] "))


def test_d2_marks_the_fastest_result_with_the_predicates_first_reason(store: LocalStore) -> None:
    valid = put(store, result(**LIGERO_4090, t_total=0.3))
    put(store, result(**LIGERO_4090, t_total=0.4))                                                       # slower: not shown
    selfie = put_registered(store, result(**LIGERO_4090, t_total=0.2), {"lane": "wave-4090-2"},
                            labels=[("verified", "accepted", "wave-4090-2", "2026-09-24T04:12:37Z")])
    hvzk = put(store, result(**{**LIGERO_4090, "proof_class": "COMPLETE_HVZK_BACKEND"}, t_total=0.5))
    ajtai = put(store, hashed("ajtai-n64", t_total=0.6))
    shared = put(store, hashed("poseidon2", sharing="tile64x64", authentication="included-hash-shared", t_total=0.7))
    p2 = put(store, hashed("poseidon2", t_total=0.8))

    dd = D.build(store)
    md = D.render_markdown(dd)
    tab = T.build(store)
    assert cell(tab, FP8_ADA_MMA, "B-Ligero").artifact == valid
    c = d2_cell(md, "B-Ligero v1", "4090 FP8")                  # the self-verified 0.2 s run is faster: D2 shows the verified one
    assert c.startswith("0.3 s · ") and f" {D.PASS} {D.VERIFIED_MARK} [" in c, c
    note = footnote(md, c)
    assert valid in note and f"independently verified by {VERIFIER}" in note and f"faster, unverified: {selfie} 0.2 s" in note, note
    for variant, art, mark in (("B-Ligero Fiat–Shamir (HVZK)", hvzk, "Z"), ("B-Ligero + Ajtai n=64", ajtai, "X"),
                               ("B-Ligero + Poseidon2 shared 64x64 tile", shared, "S"), ("B-Ligero + Poseidon2 per row", p2, D.PASS)):
        c = d2_cell(md, variant, "4090 FP8")
        assert f" {mark} {D.VERIFIED_MARK} [" in c and art in footnote(md, c), (variant, c)
    selfie_only = D.fastest(dd, "B-Ligero v1", FP8_ADA_MMA.name)[0]
    assert selfie_only.artifact == valid and D.fastest_unverified(dd, "B-Ligero v1", FP8_ADA_MMA.name).artifact == selfie
    assert d2_cell(md, "B-Ligero v1", "A100 BF16") == T.EMPTY
    # the marks are tables.py's predicate: ✓ exactly where its build accepts, and the reasons it rejects with, in its order
    rej = {r["artifact"]: r["reasons"] for r in tab.rejected}
    assert len(dd.measured) == 7 and set(rej) == {selfie, hvzk, ajtai, shared}
    for m in dd.measured:
        assert [why for _, why in m.codes] == rej.get(m.artifact, []) and all(ch in T.REASON_LETTERS for ch, _ in m.codes), m.artifact


def test_d2_counts_a_payload_envelope_but_never_as_valid(store: LocalStore, tmp_path: Path) -> None:
    """A lane that registered its summary as meta and the result envelope as the file: D2 shows the run (with an ``X`` saying
    why), tables.py -- which reads meta -- never makes it a cell."""

    doc = result(**LIGERO_4090, t_total=0.1)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    art = store.put_artifact(T.RESULT_KIND, {"lane": "wave-4090-2", "summary": "fused 4090 run"}, file=path)
    store.add_label(art, "verified", "accepted", "verify-night", ts="2026-09-24T11:00:00Z")
    assert list(T.build(store).cells) == []
    dd = D.build(store)
    (m,) = dd.measured
    assert m.artifact == art and m.payload and m.codes == [("X", D.PAYLOAD_ONLY)] and m.variant == "B-Ligero v1"
    md = D.render_markdown(dd)
    c = d2_cell(md, "B-Ligero v1", "4090 FP8")
    assert " X iv [" in c and "(payload envelope)" in footnote(md, c) and "register the result.json as --meta" in footnote(md, c)


def test_d2_shows_the_fastest_unverified_result_only_when_nothing_is_verified(store: LocalStore) -> None:
    a = put(store, result(**LIGERO_4090, t_total=0.25), verified_by=None)
    b = put(store, result(**LIGERO_4090, t_total=0.2), verified_by=None)
    dd = D.build(store)
    shown, ok = D.fastest(dd, "B-Ligero v1", FP8_ADA_MMA.name)
    assert shown.artifact == b and shown.verified_by is None and ok is None and D.fastest_unverified(dd, "B-Ligero v1", FP8_ADA_MMA.name) is None
    c = d2_cell(D.render_markdown(dd), "B-Ligero v1", "4090 FP8")
    assert c.startswith("0.2 s · ") and f" U {D.UNVERIFIED_MARK} [" in c and a not in c, c


def test_d2_ranks_by_overhead_so_a_small_batch_does_not_win(store: LocalStore) -> None:
    full = put(store, result(**LIGERO_4090, t_total=0.3))
    small = result(**LIGERO_4090, t_total=0.05)
    small["workload_fingerprint"].update(B=64)
    small["workload_fingerprint"]["instances"]["range"] = [0, 64]
    put(store, small)
    best, ok = D.fastest(D.build(store), "B-Ligero v1", FP8_ADA_MMA.name)
    assert best.artifact == full and ok.artifact == full


# --- D3: communication and verifier cost from the cell's envelope ---------------------------------------------------------------------


def with_d3(doc: dict[str, Any], *, cpu: str = "verify.cpu_s") -> dict[str, Any]:
    doc["measurements"] += [contract.measurement("proof_bytes", 66e6, "B"), contract.measurement("net.bytes_in", 4 * 5740),
                            contract.measurement("rounds.protocol", 3), contract.measurement(cpu, 3.6), contract.measurement("t.total_live", 0.33)]
    doc["validation"]["evidence"] = {"live_verifier": {"sessions": 4}}
    return doc


def test_d3_reads_only_the_cells_envelope_and_marks_what_is_absent(store: LocalStore) -> None:
    ligero = put(store, with_d3(result(**LIGERO_4090, t_total=0.3)))
    agkr = put(store, result(t_total=4.0))                                               # A100 A-GKR cell: no D3 field at all
    dd = D.build(store)
    rows = {r["artifact"]: r for r in dd.d3}
    assert set(rows) == {ligero, agkr} == {c.artifact for c in dd.tables.cells}
    r = rows[ligero]
    assert r["proof_bytes"] == 66e6 and r["proof_bytes_from"] == "proof_bytes"
    assert r["gbit_per_s"] == pytest.approx(66e6 * 8 / 0.3 / 1e9) and r["coins_bytes"] == pytest.approx(5740)
    assert r["rounds"] == 3 and r["verifier_cpu_s"] == 3.6 and r["verifier_cpu_from"] == "verify.cpu_s"
    assert r["verifier_cores"] == pytest.approx(3.6 / 0.3) and r["live_tax"] == pytest.approx(0.33 / 0.3)
    assert r["variant"] == "B-Ligero v1" and "transcript.bytes" in r["fields_absent"] and "proof_bytes" in r["fields_present"]
    a = rows[agkr]
    assert a["fields_present"] == [] and a["fields_absent"] == list(D.D3_FIELDS)
    assert all(a[k] is None for k in ("proof_bytes", "gbit_per_s", "coins_bytes", "rounds", "verifier_cpu_s", "verifier_cores", "live_tax"))
    md = D.render_markdown(dd)
    line = next(x for x in md.splitlines() if x.startswith("| A100 BF16 · A-GKR |"))
    assert line.count(T.UNMEASURED) == 7, line
    line = next(x for x in md.splitlines() if x.startswith("| 4090 FP8 · B-Ligero |"))
    assert "66 MB" in line and "5.74 kB" in line and "12" in line and "1.1×" in line and T.UNMEASURED not in line, line


def test_d3_falls_back_to_the_emitters_verifier_seconds_and_says_so(store: LocalStore) -> None:
    put(store, with_d3(result(**LIGERO_4090, t_total=0.3), cpu="verifier.seconds"))
    dd = D.build(store)
    (r,) = dd.d3
    assert r["verifier_cpu_s"] == 3.6 and r["verifier_cpu_from"] == "verifier.seconds"
    line = next(x for x in D.render_markdown(dd).splitlines() if x.startswith("| 4090 FP8 · B-Ligero |"))
    assert "3.6 s†" in line and "12†" in line, line


def live(doc: dict[str, Any], *, cpu: float, t_live: float, prover_dc: str = "EU-RO-1", verifier_dc: str = "EU-RO-1",
         url: str = "tcp://213.173.105.95:48843", sessions: int = 5) -> dict[str, Any]:
    """``doc`` as a live run: the live verifier's clock, both prover clocks, and its evidence (DCs, sessions, bytes each way)."""

    doc["measurements"] += [contract.measurement("verify.cpu_s", cpu), contract.measurement("t.total_live", t_live),
                            contract.measurement("net.bytes_in", sessions * 5860), contract.measurement("rounds.protocol", 3)]
    doc["validation"]["evidence"] = {"live_verifier": {"accepted": True, "prover_dc": prover_dc, "verifier_dc": verifier_dc, "url": url,
                                                       "sessions": sessions, "bytes_in": sessions * 5860, "bytes_out": sessions * 1.03e8,
                                                       "rtt_ms_median": 0.96}}
    return doc


def test_d3_takes_the_median_same_config_same_dc_live_run(store: LocalStore) -> None:
    """The cell (a local run) has no live clock: D3 takes the median-verifier-CPU live run of the same config in one DC, never a
    run of another l, another DC pair or below 2^-128, and reads the tax from that run's own two clocks."""

    cell_art = put(store, with_d3(result(**LIGERO_4090, t_total=0.3), cpu="verifier.seconds"))
    arts = {c: put(store, live(result(**LIGERO_4090, t_total=0.32), cpu=c, t_live=0.33)) for c in (2.6, 2.2, 2.4)}
    put(store, live(result(**LIGERO_4090, t_total=0.34, security={"target": -128, "achieved_log2": -128.3, "rs_l": 4096}), cpu=1.0, t_live=0.35))
    put(store, live(result(**LIGERO_4090, t_total=0.34), cpu=1.1, t_live=0.35, verifier_dc="EU-CZ-1"))
    put(store, live(result(**LIGERO_4090, t_total=0.34, security={"target": -100, "achieved_log2": -100.2}), cpu=1.2, t_live=0.35))
    dd = D.build(store)
    (r,) = dd.d3
    assert r["artifact"] == cell_art and r["live_from"] == "companion" and r["live_artifact"] == arts[2.4]
    assert sorted(r["live_pool"]) == sorted(arts.values()) and r["live_pool_cpu_s"] == [2.2, 2.4, 2.6]
    assert r["verifier_cpu_s"] == 2.4 and r["verifier_cpu_from"] == "verify.cpu_s" and r["verifier_cores"] == pytest.approx(2.4 / 0.3)
    assert r["proof_bytes"] == 66e6 and r["coins_bytes"] == pytest.approx(5860) and r["wire_bytes_up"] == pytest.approx(1.03e8)
    assert r["live_tax"] == pytest.approx(0.33 / 0.32) and r["live_tax_s"] == pytest.approx(0.01) and r["loopback"] is False
    md = D.render_markdown(dd)
    line = next(x for x in md.splitlines() if x.startswith("| 4090 FP8 · B-Ligero |"))
    assert "2.4 s |" in line and "†" not in line and "(+10 ms)" in line and "°" not in line, line
    assert f"- 4090 FP8 · B-Ligero: companion {arts[2.4]} (EU-RO-1, verifier tcp://213.173.105.95:48843" in md
    assert "3 same-config live run(s), verifier CPU 2.2–2.6 s" in md


def test_d3_companion_prefers_a_verifier_on_its_own_host_over_loopback_runs(store: LocalStore) -> None:
    cell_art = put(store, with_d3(result(**LIGERO_4090, t_total=0.3), cpu="verifier.seconds"))
    for c in (2.0, 2.1, 2.2):
        put(store, live(result(**LIGERO_4090, t_total=0.31), cpu=c, t_live=0.32, url="tcp://127.0.0.1:7000"))
    own = {c: put(store, live(result(**LIGERO_4090, t_total=0.31), cpu=c, t_live=0.33)) for c in (3.0, 3.4)}
    (r,) = D.build(store).d3
    assert r["artifact"] == cell_art and r["live_from"] == "companion" and r["live_artifact"] == own[3.0]
    assert r["verifier_cpu_s"] == 3.0 and r["loopback"] is False and len(r["live_pool"]) == 5


def test_d3_prefers_the_cells_own_live_run_and_marks_a_loopback_verifier(store: LocalStore) -> None:
    cell_art = put(store, live(result(**LIGERO_4090, t_total=0.3), cpu=7.0, t_live=0.301, url="tcp://127.0.0.1:7000"))
    put(store, live(result(**LIGERO_4090, t_total=0.31), cpu=6.0, t_live=0.32))
    dd = D.build(store)
    (r,) = dd.d3
    assert r["live_from"] == "cell" and r["live_artifact"] == cell_art and r["verifier_cpu_s"] == 7.0 and r["loopback"] is True
    line = next(x for x in D.render_markdown(dd).splitlines() if x.startswith("| 4090 FP8 · B-Ligero |"))
    assert "1×° (+1 ms)" in line, line


def test_d3_without_a_live_run_takes_a_reverification_verdict_of_the_cells_proof(store: LocalStore) -> None:
    agkr = put(store, result(t_total=1.35))
    other = put(store, result(t_total=9.0, gpu="NVIDIA H100 80GB HBM3"))
    ms = [contract.measurement("verify.cpu_s", 15.2), contract.measurement("verify.wall_s", 2.1)]
    store.put_artifact(D.VERDICT_KIND, {"result": "FAIL", "verifier": "x", "detail": "", "measurements": ms}, refs={"result": agkr})
    ok = store.put_artifact(D.VERDICT_KIND, {"result": "PASS", "verifier": "verity-gkr-verify", "detail": "", "measurements": ms},
                            refs={"result": agkr})
    dd = D.build(store)
    rows = {r["artifact"]: r for r in dd.d3}
    r = rows[agkr]
    assert other not in rows and r["verifier_cpu_s"] == 15.2 and r["verifier_cpu_from"] == "verdict" and r["verdict_artifact"] == ok
    assert r["verifier_cores"] == pytest.approx(15.2 / 1.35) and r["live_tax"] is None and r["live_artifact"] is None
    md = D.render_markdown(dd)
    line = next(x for x in md.splitlines() if x.startswith("| A100 BF16 · A-GKR |"))
    assert "15.2 s‡" in line and line.endswith("| … |"), line
    assert f"- A100 BF16 · A-GKR: no live record; verifier CPU from {ok}" in md


# --- the CLI --------------------------------------------------------------------------------------------------------------------------


def test_cli_renders_all_three_drilldowns(store: LocalStore, capsys: pytest.CaptureFixture[str]) -> None:
    put(store, with_d3(result(**LIGERO_4090, t_total=0.3)))
    put(store, hashed("blake3", t_total=0.9))
    assert D.main(["--root", str(store.root), "--format", "md"]) == 0
    md = capsys.readouterr().out
    assert D.LAYOUT in md and "## D1" in md and "## D2" in md and "## D3" in md
    assert all(f"| {v} |" in md for v in SEEDS)
    assert D.main(["--root", str(store.root), "--format", "json"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert js["canonical"] is False and js["layout"] == D.LAYOUT
    assert [v["id"] for v in js["d1"]["variants"]] == list(SEEDS) and [r["variant"] for r in js["d2"]["rows"]] == list(SEEDS)
    assert js["d2"]["rows"][SEEDS.index("B-Ligero + BLAKE3 in circuit")]["cells"]["4090 FP8"]["mark"] == "X"
    assert len(js["d3"]["rows"]) == 1 and js["d3"]["rows"][0]["cell"] == "4090 FP8 · B-Ligero"
    assert js["d1"]["variants"][0]["holds"] == ["4090 FP8 · B-Ligero"]


def test_sp1_soundness_leads_with_the_whole_proof_figure() -> None:
    for vid, whole in (("SP1 stock (relation-only / bare guest)", "about 2^-94.5 per proof"),
                       ("SP1 stock, FRI query count raised (otherwise unmodified)", "2^-95.46 per proof"),
                       ("SP1 precompile (TC_DOT chip)", "about 2^-96.0 per proof")):
        text = D._BY_ID[vid].soundness
        assert text.startswith(whole) and text.index("per proof") < text.index("Per shard proof"), vid
    assert "SP1_TARGET_BITS_OF_SECURITY stays 100" in D._BY_ID["SP1 stock, FRI query count raised (otherwise unmodified)"].binding
