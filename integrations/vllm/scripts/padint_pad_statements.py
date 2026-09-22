#!/usr/bin/env python3
"""[PAD-INT, stage 2] Padded-path statements over a REAL early-terminating B0 record: the live and inactive relations
of the padded serving program, with the realized liveness derived from the committed tokens, each instance addressed
to a call of the padded program (L3 address map), emitted as `veritor.pad-serve@1` SP1 batches.

Compared with `zk/bench/pad_relations.py::build_b0` (PROOF-COV / INT-PAD-3), whose inactive instances were CONSTRUCTED
twins (`live = 0` imposed on committed live words because no request in `INTPAD2-B0-20260909T013519Z` sampled EOS),
every instance here carries the liveness the record REALIZED:

* `live_in[g]` for sampling event `g` (event 0 = prefill `Final`, event `s ≥ 1` = decode step `s`) is the flag chain
  `live_0 = 1`, `live_{g+1} = live_g ∧ (y_g ≠ EOS_ID)` evaluated on the committed boundary `output_token_ids`; the chain
  is cross-checked against the committed `finish_reason` and `present` bits (a request that stopped by EOS has its
  EOS as the last committed token — vLLM V1 keeps the stop token in `token_ids`).
* live events (`g < n`, `n` = produced tokens): `PadGuard_v1{EOS=0}(1, y_g)` with `y_g` the committed word (interior
  `sampler.select` leaf when the request has committed interiors, else the boundary `output_token` leaf) — the live
  relation reproduces the committed token (`tok_out = y_g`) and computes the flag (`live_out = [y_g ≠ 0]`); the event
  `g = n-1` of an EOS-terminated request is the STOP EVIDENCE instance (`live_in = 1, y = EOS → live_out = 0`).
* inactive events (`g ≥ n`): `PadGuard_v1{EOS=0}(0, ·)` → `(0, 0)`; `live_in = 0` is a value of the committed flag
  chain, not a construction. The raw sample `y_g` of an inactive event was never materialized by production
  (spec §12 `canonical, not captured`): the statement opens the canonical placeholder `0` for it and says so —
  the step-boundary relation of spec §6 (`live_in = 0 ⟹ (tok_out, live_out) = (0, 0)`) holds for every `y`.
* row muxes (`PadRowMux{N=192}`, the (c) region of `PadDecodeStep_v1` at the B0 shape): live steps `s ≤ n-1`
  select the committed appended k row (`layers.L.self_attn.rotary_emb.rope` port 1 at `t = LP-1+s`); inactive
  steps `s ≥ n` select against the zero row from a `canonical, not captured` placeholder row (the computed row of an
  inactive step is never materialized by production; the relation's output is the zero row for every computed row).

The FULL inactive-step BODY (spec §12 materialization with the reference evaluator, INACT-BASE's cost item) is NOT
done here and is stated as such in every artifact.  Program identity: `--program hand-built` binds
`serve3.serve_v3({**B0, "EOS_ID": 0}, LP, D_MAX=STEPS)` (label HAND-BUILT FALLBACK — not the converged path) or
`--program derived:<descriptor.json.gz>` decodes a DERIVED padded program; either way the digest is recorded beside
the certified record's `Serve_v1` digest and every instance is located in the padded program by named call.

    PYTHONPATH=src:zk/bench:vllm-poc python scripts/padint_pad_statements.py \
        --record RECORD_DIR --bundle EXTRACTED_BUNDLE_ROOT --program hand-built --out BATCHES_DIR
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
for entry in (str(REPO), str(REPO / "src"), str(REPO / "zk" / "bench"), str(REPO / "vllm-poc")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from pad_relations import Bundle, PadBatcher, PadTranslated, emit, make_row_mux  # noqa: E402
from pb_common import write_manifest  # noqa: E402
from veritor.core.pad_serve import make_pad_serve_gate_set  # noqa: E402

EOS = 0  # B0's EOS_ID (SmolLM2-135M <|endoftext|>) = PAD_TOKEN_ID (spec §4/§6: no constraint needed)
HAND_BUILT_LABEL = "HAND-BUILT FALLBACK — not the converged path (Serve_v3 from verity_vllm/registry/serve3.py; the derived padded program was not available)"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# realized liveness from the committed boundary
# ---------------------------------------------------------------------------


def flag_chain(output_token_ids: list[int], slots: int) -> tuple[list[int], list[int]]:
    """`(live_in, live_out)` per sampling event `g = 0..slots-1` from the committed tokens: `live_in[0] = 1`,
    `live_out[g] = live_in[g] ∧ (y_g ≠ EOS)`, `live_in[g+1] = live_out[g]`; events without a committed token
    (`g ≥ n`) are inactive with `live_in = 0` (the chain has already absorbed)."""
    live_in, live_out = [], []
    live = 1
    for g in range(slots):
        live_in.append(live)
        if g < len(output_token_ids):
            y = output_token_ids[g]
            live = live & (0 if y == EOS else 1)
        else:
            # no committed token: production never ran this event; the chain must already be 0, unless the request
            # ended for a reason other than EOS/length (recorded by the caller as an anomaly)
            live = 0 if live == 0 else live
        live_out.append(live)
    return live_in, live_out


def request_liveness(boundary: dict, slots: int) -> dict[str, Any]:
    outs = [int(x) for x in boundary["output_token_ids"]]
    n = len(outs)
    live_in, live_out = flag_chain(outs, slots)
    finish = boundary.get("finish_reason")
    stopped_by_eos = n >= 1 and outs[-1] == EOS
    anomalies = []
    if finish == "stop" and not stopped_by_eos:
        anomalies.append("finish_reason=stop but the last committed token is not EOS (stop string / stop_token_ids?)")
    if finish == "length" and n != slots:
        anomalies.append(f"finish_reason=length but {n} tokens committed for {slots} slots")
    if stopped_by_eos and n < slots and finish != "stop":
        anomalies.append(f"EOS committed at slot {n-1} but finish_reason={finish!r}")
    present = [bool(r["present"]) for r in boundary["output_records"]]
    if present != [g < n for g in range(slots)]:
        anomalies.append("present bits != [g < n]")
    # live_in[g] == 1 iff g < n  must hold when the chain and the engine agree (EOS-inclusive token_ids)
    if [1 if g < n else 0 for g in range(slots)] != live_in:
        anomalies.append("flag chain live_in != committed presence (an EOS inside the transcript?)")
    return {"request_id": boundary["request_id"], "prompt_len": len(boundary["prompt_token_ids"]), "n_present": n,
            "finish_reason": finish, "stopped_by_eos": stopped_by_eos, "output_token_ids": outs,
            "live_in": live_in, "live_out": live_out,
            "live_events": [g for g in range(slots) if live_in[g] == 1], "inactive_events": [g for g in range(slots) if live_in[g] == 0],
            "live_steps": [s for s in range(1, slots) if live_in[s] == 1], "inactive_steps": [s for s in range(1, slots) if live_in[s] == 0],
            "stop_event": (n - 1) if stopped_by_eos else None, "anomalies": anomalies}


# ---------------------------------------------------------------------------
# the padded program and its named calls (L3 addresses)
# ---------------------------------------------------------------------------


def padded_program(spec: str, LP: int, D_MAX: int):
    """`(Program, digest, label, function_id)` for `hand-built` or `derived:<descriptor.json.gz>`."""
    from verity.ir import Program, encode_program, program_digest
    from verity_vllm.registry import b1, serve3

    if spec == "hand-built":
        C3 = {**b1.B0_CONFIG, "EOS_ID": EOS}
        prog = Program(serve3.serve_v3(C3, LP, D_MAX))
        return prog, program_digest(encode_program(prog)), HAND_BUILT_LABEL, prog.fn.id
    if spec.startswith("derived:"):
        from verity.ir.codec import decode_program

        from verity.ir.defs import REGISTRY

        path = Path(spec.split(":", 1)[1])
        with gzip.open(path, "rb") as f:
            desc = json.loads(f.read())
        prog = decode_program(desc, REGISTRY)  # DecodedProgram: .circuit, .gates, .root (SpecializedDefinition), .annotations
        derive = (prog.annotations or {}).get("derive") or {}
        root_id = getattr(prog.root, "id", "") or ""
        # the program function's id: the root is the `Root[<fn id>]_v1` wrapper `verity.ir.program._lower_root` builds
        fn_id = root_id[len("Root["):-len("]_v1")] if root_id.startswith("Root[") and root_id.endswith("]_v1") else (root_id or prog.program_fn_id)
        label = (f"DERIVED padded program ({path.name}; verity_vllm.frontend.derive_cli serve --padded; mode={derive.get('mode')}, "
                 f"profile={derive.get('profile')}, family claim={(derive.get('family') or {}).get('family') if isinstance(derive.get('family'), dict) else derive.get('family')})")
        return prog, program_digest(desc), label, fn_id
    raise SystemExit(f"--program must be hand-built or derived:<descriptor.json.gz>, got {spec!r}")


class _BatchNode:
    """A named primitive `batch` node (the row muxes `t=<t>/kmux.<L>` are `batch(SelectBf16_v1, …)`): its whole gate
    range `[lo, hi)` = `n` members × the primitive's gates, addressed as one unit."""

    def __init__(self, sc, k: int, node) -> None:
        self.lo = sc.offset + sc.spec.body.offsets[k]
        self.hi = self.lo + node.n * node.fn.gates
        self.id = f"batch[{node.n}]({node.fn.id})"
        self.node = node


def _scope_named(sc, prefix: str, named: dict[str, Any], circuit) -> None:
    from verity.ir.layout import FunctionCall

    for k, node in enumerate(sc.spec.body.nodes):
        nm = getattr(node, "name", None)
        if not nm:
            continue
        if node.form in ("primitive", "call"):
            named[f"{prefix}{nm}"] = FunctionCall(circuit, sc, k, 0)
        else:  # a batch / scan node: primitive members are gates, composite members are calls — address the node
            named[f"{prefix}{nm}"] = _BatchNode(sc, k, node)


def named_calls(prog) -> dict[str, Any]:
    """Root-level named nodes plus the named nodes nested one level inside every `s=<s>/step` (`PadDecodeStep`):
    `t=<t>/guard`, `t=<t>/final`, `t=<t>/kmux.<L>` / `vmux.<L>` (primitive batch nodes), `t=<t>/layers.<L>/{pre,attn,post}`.
    Node names are hand-built metadata (`serve3.py` names its nodes; the descriptor codec does not carry names), so this
    only works on the hand-built `Program`; for a decoded descriptor use `structural_named_calls`."""
    named: dict[str, Any] = {}
    circuit = prog.circuit
    _scope_named(circuit.root_scope, "", named, circuit)
    for c in list(circuit.root_calls()):
        nm = c.node.name
        if not c.is_primitive and nm and nm.endswith("/step"):
            _scope_named(c.scope(), f"{nm}/", named, circuit)
    return named


def _name_paths(ref) -> dict[str, tuple[int, ...]]:
    """name -> node-index path (root index, or (step root index, inner index)) in the hand-built reference."""
    paths: dict[str, tuple[int, ...]] = {}
    root = ref.circuit.root_scope
    for k, node in enumerate(root.spec.body.nodes):
        nm = getattr(node, "name", None)
        if nm:
            paths[nm] = (k,)
    for c in list(ref.circuit.root_calls()):
        nm = c.node.name
        if not c.is_primitive and nm and nm.endswith("/step"):
            sc = c.scope()
            for k2, node in enumerate(sc.spec.body.nodes):
                nm2 = getattr(node, "name", None)
                if nm2:
                    paths[f"{nm}/{nm2}"] = (c.node_idx, k2)
    return paths


def structural_named_calls(prog, ref) -> tuple[dict[str, Any], dict[str, Any]]:
    """L3 addresses in a DECODED program (no node names): the hand-built reference `ref` — same dims, same digest — supplies
    name -> node-index path; each path is resolved in `prog`'s own circuit and cross-checked call-by-call against the
    reference (callee id and gate interval `[lo, hi)` must agree, else it is a mismatch). Returns `(named, report)`."""
    from verity.ir.layout import FunctionCall

    ref_named = named_calls(ref)
    circuit = prog.circuit
    root = circuit.root_scope
    named: dict[str, Any] = {}
    mismatches: list[str] = []
    for key, path in _name_paths(ref).items():
        sc = root
        if len(path) == 2:
            sc = FunctionCall(circuit, root, path[0], 0).scope()
        k = path[-1]
        node = sc.spec.body.nodes[k]
        got = FunctionCall(circuit, sc, k, 0) if node.form in ("primitive", "call") else _BatchNode(sc, k, node)
        want = ref_named[key]
        if (got.id, got.lo, got.hi) != (want.id, want.lo, want.hi):
            mismatches.append(f"{key}: derived {got.id}[{got.lo},{got.hi}) vs hand-built {want.id}[{want.lo},{want.hi})")
        named[key] = got
    report = {"source": "structural node-index paths resolved in the DERIVED program's circuit (the descriptor codec carries no node "
                        "names; names come from the same-dimension hand-built Serve_v3 and are labels only)",
              "cross_checked_calls": len(named), "mismatches": mismatches[:20], "n_mismatches": len(mismatches)}
    return named, report


def address_of(named: dict[str, Any], key: str) -> dict[str, Any]:
    c = named[key]
    return {"call": key, "fn": c.id, "gates": [c.lo, c.hi], "n_gates": c.hi - c.lo}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", required=True, help="run_composed record dir (program/program.json, served_tokens.json, manifest.json)")
    ap.add_argument("--bundle", required=True, help="extracted bundle root (contains ru/, commitments.json)")
    ap.add_argument("--program", default="hand-built", help="hand-built | derived:<descriptor.json.gz>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mux-layers", default="0,15,29")
    ap.add_argument("--horizon-mux-steps", default="1,8,15", help="row-mux steps for requests that ran to the horizon (all steps for early-terminating requests)")
    ap.add_argument("--label", default=None, help="record id for provenance (default: record dir name)")
    args = ap.parse_args()

    record = Path(args.record)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    label = args.label or record.name
    t_start = time.perf_counter()

    program_json = json.loads((record / "program" / "program.json").read_text())
    manifest = json.loads((record / "manifest.json").read_text())
    LP = int(program_json["program"]["LP"])
    STEPS = int(program_json["program"]["STEPS"])
    D_MAX = STEPS  # lead's decision on serve3-spec §15 Q1: D_MAX := STEPS = max_tokens - 1; designated = D_MAX+1 token slots
    slots = D_MAX + 1
    certified_digest = program_json["program"]["digest"]
    print(f"record {label}: certified program {program_json['program']['function']} digest {certified_digest[:16]}; LP={LP} STEPS={STEPS} -> D_MAX={D_MAX}, {slots} designated slots", flush=True)

    prog, pad_digest, prog_label, prog_fn = padded_program(args.program, LP, D_MAX)
    address_report: dict[str, Any]
    if args.program == "hand-built":
        named = named_calls(prog)
        address_report = {"source": "hand-built node names (serve3.py) resolved in the hand-built program's circuit"}
    else:
        ref, ref_digest, _, _ = padded_program("hand-built", LP, D_MAX)
        named, address_report = structural_named_calls(prog, ref)
        address_report["hand_built_reference_digest"] = ref_digest
        address_report["same_digest_as_derived"] = ref_digest == pad_digest
        if address_report["n_mismatches"]:
            raise SystemExit(f"address map: {address_report['n_mismatches']} call(s) differ between the derived program and the "
                             f"hand-built reference: {address_report['mismatches'][:3]}")
    print(f"padded program {prog_fn}: digest {pad_digest[:16]}; {prog.gates:,} gates; {len(named)} named calls ({prog_label[:40]}…)", flush=True)
    # structural sanity of the address space: every event has a guard, every step has muxes for every layer
    NL = 30
    for g in range(slots):
        key = f"t={LP - 1}/guard" if g == 0 else f"s={g}/step/t={LP - 1 + g}/guard"
        assert key in named, key
    for s in range(1, slots):
        for L in range(NL):
            assert f"s={s}/step/t={LP - 1 + s}/kmux.{L}" in named, (s, L)
    n_guard_calls = sum(1 for k in named if k.endswith("/guard"))
    assert n_guard_calls == slots, (n_guard_calls, slots)

    bundle = Bundle(Path(args.bundle))
    boundaries: dict[str, dict] = {}
    for req in bundle.requests:
        bpath = Path(args.bundle) / "ru" / req / "boundary.json"
        boundaries[req] = json.loads(bpath.read_text())
    served = json.loads((record / "served_tokens.json").read_text()) if (record / "served_tokens.json").exists() else None

    liveness = {req: request_liveness(b, slots) for req, b in boundaries.items()}
    interior = {req: bool(bundle.positions(req, "sampler.select")) for req in bundle.requests}
    for req, lv in liveness.items():
        print(f"  {req[:8]}: n={lv['n_present']:2d} finish={lv['finish_reason']:6s} live events {len(lv['live_events'])} inactive {len(lv['inactive_events'])} "
              f"stop_event={lv['stop_event']} interiors={'yes' if interior[req] else 'no'} anomalies={lv['anomalies'] or 'none'}", flush=True)
    early = [r for r, lv in liveness.items() if lv["inactive_events"]]
    horizon = [r for r, lv in liveness.items() if not lv["inactive_events"]]
    if not early:
        print("WARNING: no request in this record has an inactive event — an all-live record is NOT inactive evidence (GUIDANCE §44.2)", flush=True)

    entries: list[dict[str, Any]] = []
    prep: dict[str, dict] = {}
    census: dict[str, Any] = {"guard": [], "rowmux": []}

    # ---- guards: every sampling event of every request, at the realized liveness -------------------------------
    translated = PadTranslated(__import__("verity.ir.defs", fromlist=["bind"]).bind(
        __import__("verity_vllm.registry.serve3", fromlist=["PadGuard"]).PadGuard, EOS=EOS), f"PadGuard<EOS={EOS}>", [1, 32])
    batcher = PadBatcher(translated, f"padint-{label}", [32, 1])
    # early-terminating requests first so that any prefix batch already covers live AND inactive instances
    order = early + horizon
    for req in order:
        lv = liveness[req]
        b = boundaries[req]
        for g in range(slots):
            t = LP - 1 + g
            live = lv["live_in"][g]
            if live:
                if interior[req]:
                    y = int(bundle.value(req, t, "sampler.select")) & 0xFFFFFFFF
                    src = "captured: interior sampler.select leaf"
                else:
                    y = int(b["output_records"][g]["token_id"]) & 0xFFFFFFFF
                    src = "captured: boundary output_token leaf"
                assert y == (lv["output_token_ids"][g] & 0xFFFFFFFF), (req, g)
            else:
                y = 0
                src = "canonical, not captured (inactive event: production never sampled here; placeholder 0; relation output independent of y)"
            outs = translated.oracle_outputs([live, y])
            expect = [y if live else 0, live & (0 if y == EOS else 1)]
            assert outs == expect, (req, g, live, y, outs)
            assert outs[1] == lv["live_out"][g], (req, g)
            kind = "inactive" if not live else ("stop-evidence" if (live and y == EOS) else "live")
            batcher.add(f"{req[:8]}-g{g}-t{t}-{kind}", [live, y], outs)
            census["guard"].append({"request_id": req, "event": g, "t": t, "kind": kind, "live_in": live, "y": y, "y_source": src,
                                    "tok_out": outs[0], "live_out": outs[1], "address": address_of(named, f"t={t}/guard" if g == 0 else f"s={g}/step/t={t}/guard")})
    prep_key = f"pad-guard/eos{EOS}-padint"
    prep[prep_key] = {"obligations": len(batcher.prepared), "cpu_s": batcher.prep_s, "positions": batcher.prep_positions}
    n_live = sum(1 for c in census["guard"] if c["kind"] == "live")
    n_stop = sum(1 for c in census["guard"] if c["kind"] == "stop-evidence")
    n_inact = sum(1 for c in census["guard"] if c["kind"] == "inactive")
    emit(out, entries, prep_key, batcher, [len(batcher.prepared)],
         {"relation_id": f"PadGuard_v1{{EOS={EOS}}}", "record": label, "inactive_units": n_inact, "live_units": n_live, "stop_evidence_units": n_stop,
          "program_digest": pad_digest, "program": prog_fn, "program_label": prog_label, "certified_program_digest": certified_digest,
          "provenance": f"committed record {label}: liveness = the flag chain on the committed boundary output_token_ids (EOS-inclusive), "
                        f"cross-checked against finish_reason/present bits; live y = committed sampler.select / output_token words; "
                        f"inactive y = canonical placeholder 0 (canonical, not captured); {n_stop} stop-evidence instance(s) (live_in=1, y=EOS -> live_out=0); "
                        f"every instance addressed to a PadGuard call of {prog_fn} (census.json)"})

    # ---- row muxes at the B0 shape, at the realized liveness ----------------------------------------------------
    N = 192  # KVH*D
    mux = PadTranslated(make_row_mux(N), f"PadRowMux<N={N}>", [1] + [16] * N)
    mbatcher = PadBatcher(mux, f"padint-{label}", [16] * N)
    layers = [int(x) for x in args.mux_layers.split(",")]
    hsteps = [int(x) for x in args.horizon_mux_steps.split(",")]
    n_mlive = n_minact = words_checked = 0
    for req in order:
        if not interior[req]:
            continue
        lv = liveness[req]
        steps = list(range(1, slots)) if lv["inactive_steps"] else hsteps
        for s in steps:
            t = LP - 1 + s
            live = lv["live_in"][s]
            for L in layers:
                key = (req, t, f"layers.{L}.self_attn.rotary_emb.rope", "1")
                if live:
                    if key not in bundle.entries:
                        raise SystemExit(f"{req[:8]} t={t} L={L}: live step but no committed rope row — a captured-leaf gap, not a padding fact")
                    row = [int(v) for v in bundle.value(req, t, f"layers.{L}.self_attn.rotary_emb.rope", "1")]
                    src = "captured: appended k row (rotary_emb.rope port 1)"
                else:
                    if key in bundle.entries:
                        raise SystemExit(f"{req[:8]} t={t} L={L}: inactive step but a committed rope row exists — liveness/record disagreement")
                    row = [0] * N
                    src = "canonical, not captured (inactive step: production never computed this row; placeholder zero row; relation output = zero row for every computed row)"
                assert len(row) == N
                outs = mux.oracle_outputs([live] + row)
                if live:
                    assert outs == row
                    words_checked += N
                    n_mlive += 1
                else:
                    assert all(v == 0 for v in outs)
                    n_minact += 1
                mbatcher.add(f"{req[:8]}-s{s}-t{t}-L{L}-{'live' if live else 'inactive'}", [live] + row, outs)
                census["rowmux"].append({"request_id": req, "step": s, "t": t, "layer": L, "kind": "live" if live else "inactive", "live_in": live,
                                         "row_source": src, "address": address_of(named, f"s={s}/step/t={t}/kmux.{L}")})
    prep_key = f"pad-rowmux/n{N}-padint"
    prep[prep_key] = {"obligations": len(mbatcher.prepared), "cpu_s": mbatcher.prep_s, "positions": mbatcher.prep_positions}
    if mbatcher.prepared:
        emit(out, entries, prep_key, mbatcher, [len(mbatcher.prepared)],
             {"relation_id": f"PadRowMux{{N={N}}} (the (c) region of PadDecodeStep_v1 at the B0 shape)", "record": label,
              "inactive_units": n_minact, "live_units": n_mlive, "captured_words_checked": words_checked,
              "program_digest": pad_digest, "program": prog_fn, "program_label": prog_label, "certified_program_digest": certified_digest,
              "provenance": f"committed record {label}: live steps select the committed appended k row (rope port 1, layers {layers}); inactive steps "
                            f"(realized: the request stopped by EOS before them) select against the zero row from a canonical, not captured placeholder row; "
                            f"every instance addressed to a kmux call of {prog_fn} (census.json)"})

    gs = make_pad_serve_gate_set()
    total_ob = sum(v["obligations"] for v in prep.values())
    total_s = sum(v["cpu_s"] for v in prep.values())
    total_pos = sum(v["positions"] for v in prep.values())
    write_manifest(out / "manifest.json", entries,
                   {"workload": "padint-pad", "gate_set": str(gs.id), "gate_set_digest": str(gs.digest),
                    "semantics": "pad-serve@1 computed-interior (typed selects, eq_i32 stop test, absorbing boolean chain); ordinary per-VU authenticated statements",
                    "leaf_layout": "inputs/outputs one word per leaf (u1/u16/u32 by port width, frame/v3); constants pinned with expected",
                    "witness_prep": {"per_relation": prep, "obligations": total_ob, "cpu_s": total_s, "positions": total_pos,
                                     "cpu_s_per_obligation": (total_s / total_ob) if total_ob else None,
                                     "cpu_s_per_opened_position": (total_s / total_pos) if total_pos else None}})

    # ---- the PAD-INT census: program identities, liveness, address map, evidence labels -----------------------
    profile = manifest.get("inputs", {}).get("semantic_profile")
    doc = {
        "schema": "veritor/padint-pad-statements/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "record": label,
        "evidence_label": prog_label,
        "address_map": address_report,
        "program": {"padded": {"function": prog_fn, "digest": pad_digest, "gates": prog.gates, "source": args.program, "LP": LP, "D_MAX": D_MAX, "EOS_ID": EOS},
                    "certified": {"function": program_json["program"]["function"], "digest": certified_digest},
                    "linkage": (("SINGLE-DIGEST: the record's program IS the padded program (run_composed --padded-program); the certificate and every "
                                 "pad statement name the same digest")
                                if certified_digest == pad_digest else
                                ("TWO-DIGEST: the padded program is a DIFFERENT program from the certified Serve_v1 (guards + muxes added, Serve_v3 identity); "
                                 "its live region equals Serve_v2{LP, STEPS=ℓ} gate-for-gate by construction and Serve_v1 via CF1 (spec §13 prefix property); "
                                 "the record's certificate names the Serve_v1 digest — this census links every pad statement to the padded digest and records both")),
                    "single_digest": certified_digest == pad_digest,
                    "d_max_convention": "D_MAX := STEPS = max_tokens - 1 (lead decision, spec §15 Q1); designated outputs = the D_MAX+1 = "
                                        f"{slots} token slots out[0..{D_MAX}] at positions t = LP-1+k = {LP-1}..{LP-1+D_MAX}, one-to-one with the "
                                        "certified record's output manifest (ℓ_out = STEPS+1 = max_tokens: slot 0 = the prefill-produced token, "
                                        "slots 1..STEPS = decode steps 1..STEPS); liveness is internal, not designated (GUIDANCE §32.6)"},
        "profile": {"semantic_profile": profile if isinstance(profile, (str, type(None))) else "see manifest.inputs.semantic_profile",
                    "semantic_profile_sha256": manifest.get("inputs", {}).get("semantic_profile_sha256"),
                    "profile_id": (manifest.get("inputs", {}).get("production_manifest") or {}).get("adapter", {}).get("profile_id") if isinstance(manifest.get("inputs", {}).get("production_manifest"), dict) else None},
        "sampling_policy_sha256": manifest.get("inputs", {}).get("sampling_policy_sha256"),
        "output_manifest": {"designated_slots_per_request": slots, "positions": list(range(LP - 1, LP + D_MAX)), "declared_domain": 16,
                            "designation": "token slots only (GUIDANCE §32.6; serve3-spec errata 2026-09-08)"},
        "requests": liveness, "interiors_committed": interior,
        "early_terminating_requests": early, "horizon_requests": horizon,
        "counts": {"guard": {"live": n_live, "stop_evidence": n_stop, "inactive": n_inact, "total": len(census["guard"])},
                   "rowmux": {"live": n_mlive, "inactive": n_minact, "total": len(census["rowmux"]), "captured_words_checked": words_checked}},
        "not_done": ["full inactive-step BODY materialization with the reference evaluator (spec §12; INACT-BASE cost item): inactive raw samples and "
                     "computed rows are canonical placeholders here, admissible only because the step-boundary relations are independent of them",
                     "L2 correspondence of live interiors against the padded program (CAPCORR's derive_corr covers Serve_v1; by the prefix property the "
                     "live region is gate-identical, but that run was not repeated under the padded digest)"],
        "served_tokens_ok": served.get("ok") if served else None,
        "census": census,
        "batches": entries,
        "seconds": round(time.perf_counter() - t_start, 1),
    }
    (out / "padint_statements.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["counts"]), flush=True)
    print(f"padded program digest {pad_digest}; certified Serve_v1 digest {certified_digest}; label: {prog_label}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
