"""Entry point of the derived path: an ordinary (unannotated) serving module -> derived Verity program artefacts.

    python -m verity_vllm.frontend.derive_cli serve --config B0 --lp 32 --steps 15 --out OUT
    python -m verity_vllm.frontend.derive_cli layer --config B0 --t 4 --out OUT
    python -m verity_vllm.frontend.derive_cli mlp   --out OUT                         (reference mode, S1)
    python -m verity_vllm.frontend.derive_cli negatives                                (every refusal, one line each)

`serve` writes what `e2e/run_composed.py`'s stage 1 writes for the hand-built program, from the DERIVED one:

    OUT/program/descriptor.json.gz   the derived program's codec descriptor (annotations carry the derivation provenance)
    OUT/program/program.json         schema verity-ir/program-summary/v1 + a `derived` block (source module, mode, profile,
                                     rule set, family identity claim, flat digest, weights binding)
    OUT/derive-report.json           the full DeriveReport (provenance incl. every rule application)

so the plan / correspondence / proof stages can name the derived program's digest (GUIDANCE §44.1). Memory: the B0
LP=32 STEPS=15 derivation needs ~0.5 GB and ~10 s — run it on the pod, not the laptop (§44.4).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import torch


def _config(name: str) -> dict:
    from verity_vllm.registry import b1
    try:
        return dict(getattr(b1, f"{name.upper()}_CONFIG"))
    except AttributeError:
        raise SystemExit(f"no static configuration {name!r} in verity_vllm.registry.b1 (B0 | B1)")


def _write_program(out: Path, rep, *, extra: dict) -> dict:
    from verity.ir.codec import descriptor_size, encode_program, program_digest
    program = rep.program
    desc = encode_program(program)
    digest = program_digest(desc)
    assert digest == rep.digest
    pdir = out / "program"
    pdir.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(desc, sort_keys=True).encode()
    with gzip.open(pdir / "descriptor.json.gz", "wb") as f:
        f.write(raw)
    doc = {
        "schema": "verity-ir/program-summary/v1",
        "program": {"function": program.fn.id, "digest": digest, "gates": program.gates, "input_gates": program.input_gates,
                    "descriptor_bytes": descriptor_size(desc), "descriptor_sha256": hashlib.sha256(raw).hexdigest(),
                    "descriptor": "descriptor.json.gz", **extra},
        "derived": {
            "padded": bool(extra.get("padded", False)),
            "source": rep.provenance["source"], "mode": rep.provenance["mode"], "profile": rep.provenance["profile"],
            "vocabulary": rep.provenance["vocabulary"], "ruleset": rep.provenance["ruleset"]["name"],
            "rules_applied": rep.provenance["rules_applied"],
            "flat_root": rep.flat_root, "flat_digest": rep.flat_digest,
            "family": None if not rep.family else {k: v for k, v in rep.family.items() if k not in ("node_bijection",)},
            "identity_claim": ("verified isomorphism with the registered family (node-for-node, references equal, weights bound injectively)"
                               if rep.family else "the derived flat program's own identity (no registered family matched)"),
            "weights": rep.provenance["weights"], "mutation_effects": rep.provenance["mutation_effects"],
            "returned_inputs": rep.provenance["returned_inputs"], "dead_gates": rep.dead.to_json() if rep.dead else None,
            "report": "../derive-report.json",
        },
        "vocabulary": "Program / FunctionCall / Subcircuit / Obligation (GUIDANCE §13.3); the descriptor is TA1's verity.ir codec",
    }
    (pdir / "program.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    (out / "derive-report.json").write_text(json.dumps(rep.to_json(), indent=1, sort_keys=True, default=str))
    return doc


def cmd_serve(a) -> int:
    from verity_vllm.frontend.derive import derive_report
    from verity_vllm.frontend import derive_examples as ex
    C = _config(a.config)
    t0 = time.time()
    padded_extra: dict = {}
    if a.padded:
        # the padded circuit (serve3-spec; PAD-INT stage 1): C3 = {**C, "EOS_ID"} at the binding site, D_MAX := STEPS
        C = {**C, "EOS_ID": a.eos_id}
        profile = a.profile if a.profile != "b1-eager" else "b1-eager-v2"  # Serve_v3 is built on the v2 skeleton
        with torch.device("meta"):
            m = ex.PaddedGreedyServe(C, a.steps, eos_id=a.eos_id)
        rep = derive_report(m, (torch.empty(a.lp, dtype=torch.int32, device="meta"),), mode=a.mode, profile=profile, config=C, name="PaddedServe")
        padded_extra = {"padded": True, "D_MAX": a.steps, "EOS_ID": a.eos_id, "designated_slots": a.steps + 1,
                        "slot_positions": list(range(a.lp - 1, a.lp + a.steps)), "declared_domain": 16,
                        "designation": "token slots only (GUIDANCE §32.6); the live vector is an internal value, not designated",
                        "d_max_convention": "D_MAX = STEPS = max_tokens - 1; designated = D_MAX+1 token slots (lead's decision on spec §15 Q1)"}
    else:
        with torch.device("meta"):
            m = ex.GreedyServe(C, a.steps, vllm_rows=(a.rows == "vllm"), final_norm_all_rows=(a.rows == "all"))
        rep = derive_report(m, (torch.empty(a.lp, dtype=torch.int32, device="meta"),), mode=a.mode, profile=a.profile, config=C,
                            name="Serve" if a.rows == "sampled" else f"Serve[{a.rows}-rows]")
    dt = time.time() - t0
    if not rep.ok:
        print(f"UNSUPPORTED({rep.unsupported.reason}) {rep.unsupported.node} @ {rep.unsupported.module_path}: {rep.unsupported.hint}")
        return 2
    if a.padded:
        rep.provenance["padded"] = {**padded_extra, "family_claim": rep.family["family"] if rep.family else None,
                                    "checked_against": "serve3.serve_v3(C3, LP, D_MAX) expanded to leaf kinds (verified isomorphism)" if rep.family else None}
    out = Path(a.out)
    doc = _write_program(out, rep, extra={"static_configuration": {k: v for k, v in C.items() if k != "profile"}, "profile": C.get("profile"),
                                          "LP": a.lp, "STEPS": a.steps, "row_policy": a.rows, **padded_extra})
    if a.map_to_family and not rep.family:
        _write_family_map(out, rep, C, a.lp, a.steps)
    fam = rep.family["family"] if rep.family else None
    print(f"derived {rep.root}")
    print(f"  digest       {rep.digest}")
    print(f"  flat digest  {rep.flat_digest}  ({rep.flat_root[:60]}...)")
    print(f"  identity     {'FAMILY ' + fam if fam else 'no registered family matched: ' + json.dumps(rep.provenance.get('families_not_matched', []))[:400]}")
    print(f"  gates        {rep.gates:,} ({rep.program.input_gates:,} Input); dead {rep.dead.total if rep.dead else 0}")
    print(f"  rules        {', '.join(rep.provenance['rules_applied'])}")
    print(f"  wrote        {out / 'program' / 'program.json'} ({dt:.1f}s)")
    if a.expect_digest and rep.digest != a.expect_digest:
        print(f"  MISMATCH expected {a.expect_digest}")
        return 3
    return 0


def _write_family_map(out: Path, rep, C: dict, lp: int, steps: int) -> None:
    """`OUT/serve_v1-map.json`: the derived program as a documented superset of `Serve_v1{C,LP,STEPS}` — node_map (derived node
    index -> Serve_v1 leaf path) for every family leaf, and the extra derived nodes with their argument reference runs."""
    from verity.ir.defs import bind as ir_bind
    from verity_vllm.registry import b1
    from verity_vllm.frontend.rules.family import match_program
    spec = ir_bind(b1.Serve, C=C, LP=lp, STEPS=steps)
    fam = match_program(rep.flat_definition.body, spec, flat_weights_param=len(rep.flat_definition.body.params) - 1, allow_extra=True, with_map=True)
    fam["derived_digest"] = rep.digest
    fam["derived_node_names"] = [n.name for n in rep.flat_definition.body.nodes]
    (out / "serve_v1-map.json").write_text(json.dumps(fam, indent=1, sort_keys=True, default=str))
    print(f"  family map   Serve_v1 superset={fam.get('superset')} matched_leaves={fam.get('node_bijection')} extra={len(fam.get('extra_nodes', []))} "
          f"-> {out / 'serve_v1-map.json'}" + (f" ({fam['reason'][:120]})" if not fam.get("superset") else ""))


def cmd_layer(a) -> int:
    from verity_vllm.frontend.derive import derive_report
    from verity_vllm.frontend import derive_examples as ex
    C = _config(a.config)
    with torch.device("meta"):
        m = ex.SingleLayer(C, layer_idx=0 if a.first else 1, pos0=a.pos0, first=a.first)
    H = C["H"]
    inputs = (torch.empty(a.t, H, dtype=torch.bfloat16, device="meta"),) * (1 if a.first else 2)
    rep = derive_report(m, inputs, mode="profile", profile=a.profile, config=C, name="Layer")
    if not rep.ok:
        print(f"UNSUPPORTED({rep.unsupported.reason}) {rep.unsupported.node} @ {rep.unsupported.module_path}: {rep.unsupported.hint}")
        return 2
    if a.out:
        _write_program(Path(a.out), rep, extra={"static_configuration": {k: v for k, v in C.items() if k != "profile"}, "T": a.t, "L0": a.first})
    print(f"derived {rep.root}\n  digest {rep.digest}\n  identity {'FAMILY ' + rep.family['family'] if rep.family else 'no family matched'}")
    if rep.family:
        print(f"  weights {json.dumps(rep.family['module_weights_by_role'])}")
    return 0


def cmd_mlp(a) -> int:
    from verity_vllm.frontend.derive import derive_report
    from verity_vllm.frontend import derive_examples as ex
    torch.manual_seed(0)
    m = ex.MLP2(16, 32, 8)
    rep = derive_report(m, (torch.empty(4, 16),), mode="reference", name="MLP2")
    if not rep.ok:
        print(f"UNSUPPORTED({rep.unsupported.reason}): {rep.unsupported.hint}")
        return 2
    if a.out:
        _write_program(Path(a.out), rep, extra={"reference": True})
    print(f"derived {rep.root}\n  digest {rep.digest}\n  vocabulary {rep.provenance['vocabulary']['name']} {rep.provenance['vocabulary'].get('pin', '')}")
    print(f"  rules {', '.join(rep.provenance['rules_applied'])}")
    return 0


def cmd_hf(a) -> int:
    """HF (transformers-shaped) classes in REFERENCE mode: a Llama-style layer with RoPE, the LlamaForCausalLM shape, or a bounded
    greedy request. No hand-built oracle exists for these (they are the un-fused circuit); the evidence is REFERENCE-MODE + the
    derivation record (rules, derived tables). Reports the derived tables the verifier must supply."""
    from verity_vllm.frontend.derive import derive_report
    from verity_vllm.frontend import derive_examples as ex
    torch.manual_seed(0)
    dt = {"bf16": torch.bfloat16, "f32": torch.float32}[a.dtype]
    C = dict(H=a.h, NH=a.nh, KVH=a.kvh, D=a.d, I=a.i, EPS=1e-5, BIAS=a.bias, V=a.v, NL=a.nl)
    if a.what == "layer":
        m, ex_in, name = ex.HFDecoderLayer(C, dt, rope=True), (torch.empty(a.t, a.h, dtype=dt),), "HFLayerRoPE"
    elif a.what == "lm":
        m, ex_in, name = ex.HFTinyLM(C, dt), (torch.empty(a.t, dtype=torch.int64),), "HFTinyLM"
    else:
        m, ex_in, name = ex.HFGreedy(C, a.steps, dt), (torch.empty(a.t, dtype=torch.int64),), "HFGreedy"
    rep = derive_report(m, ex_in, mode="reference", name=name, config={**C, "dtype": a.dtype})  # SRC hashes the dims + dtype
    if not rep.ok:
        print(f"UNSUPPORTED({rep.unsupported.reason}): {rep.unsupported.hint}")
        return 2
    if a.out:
        _write_program(Path(a.out), rep, extra={"reference": True, "hf": a.what, "config": C, "dtype": a.dtype})
    tables = [w for w in rep.provenance["weights"] if w["kind"] == "derived_table"]
    print(f"derived {rep.root}\n  digest {rep.digest}\n  vocabulary {rep.provenance['vocabulary']['name']} {rep.provenance['vocabulary'].get('pin', '')}")
    print(f"  rules {', '.join(rep.provenance['rules_applied'])}")
    print(f"  weights {len(rep.provenance['weights'])} fields ({sum(1 for w in rep.provenance['weights'] if w['aliases'])} tied); "
          f"derived tables {[(w['name'], w['shape']) for w in tables]} — supplied by the verifier per the recorded derivation")
    print(f"  program: {rep.program.input_gates} input gates; no hand-built oracle (HF classes are the un-fused circuit): evidence REFERENCE-MODE")
    return 0


def cmd_negatives(a) -> int:
    from verity_vllm.frontend.derive import derive
    from verity_vllm.frontend import derive_examples as ex
    from verity_vllm.registry import b1
    C = dict(b1.B0_CONFIG)
    C.update(NL=2, V=64, MAX_POS=64)
    meta = lambda *s, dt=torch.bfloat16: torch.empty(*s, dtype=dt, device="meta")  # noqa: E731
    # small reference-mode modules are built on CPU (a lifted constant must be seen on the module's device); the
    # profile-mode ones on meta
    cpu = [
        ("unknown op (tanh)", ex.NegUnknownOp(), (torch.empty(2, 8),), dict(mode="reference"), "numerics_unregistered"),
        ("custom op without binding", ex.NegCustomOp(), (torch.empty(2, 8),), dict(mode="reference"), "custom_op_without_binding"),
        ("hidden state (lifted constant)", ex.NegHiddenConstant(), (torch.empty(2, 8),), dict(mode="reference"), "hidden_state"),
        ("in-place without rule (add_)", ex.NegInPlace(), (torch.empty(2, 8),), dict(mode="reference"), "mutation"),
        ("torch.cond", ex.NegCond(), (torch.empty(2, 8),), dict(mode="reference"), "dynamic_control_flow"),
        ("wrong shape", torch.nn.Linear(16, 8), (torch.empty(4, 15),), dict(mode="reference"), "export"),
        ("f32 linear in the bf16 profile", torch.nn.Linear(16, 8), (torch.empty(4, 16),), dict(mode="profile", profile="b1-eager"),
         ("numerics_unregistered", "decomposition")),
    ]
    with torch.device("meta"):
        cases = cpu + [
            ("data-dependent gather", ex.NegDataDependent(), (meta(2, 8),), dict(mode="profile", profile="b1-eager"), "dynamic_control_flow"),
            ("forward writes a buffer", ex.NegMutatesWeight(), (meta(2, 8),), dict(mode="profile", profile="b1-eager"), "mutation"),
            ("stale alias after in-place RoPE", ex.NegStaleAlias(C), (meta(3, C["H"]),), dict(mode="profile", profile="b1-eager", config=C), "mutation"),
            ("HF eager layer (decomposed norm) in the fused profile", ex.HFDecoderLayer(b1.B0_CONFIG), (meta(4, b1.B0_CONFIG["H"]),),
             dict(mode="profile", profile="b1-eager", config=b1.B0_CONFIG), ("decomposition", "numerics_unregistered")),
            ("allowed_dims T={4} with T=5", ex.SingleLayer(b1.B0_CONFIG, 0), (meta(5, b1.B0_CONFIG["H"]),),
             dict(mode="profile", profile="b1-eager", config=b1.B0_CONFIG, allowed_dims={"T": {4}}), "shape"),
        ]
    bad = 0
    for label, mod, inputs, kw, want in cases:
        r = derive(mod, inputs, **kw)
        from verity_vllm.frontend.derive import Unsupported
        if not isinstance(r, Unsupported):
            print(f"FAIL  {label}: derived a program {r.fn.id[:60]} (must be REFUSED)")
            bad += 1
            continue
        ok = r.reason in (want if isinstance(want, tuple) else (want,))
        print(f"{'REFUSED' if ok else 'FAIL'}  {label}: Unsupported({r.reason}) @ {r.module_path}: {r.hint[:110]}")
        bad += 0 if ok else 1
    # lost sharing: an untied LM head derives (it is a real program) but cannot claim Serve_v1 (embed is one role)
    from verity_vllm.frontend.derive import derive_report
    Cu = dict(C, tie_word_embeddings=False)
    with torch.device("meta"):
        rep = derive_report(ex.GreedyServe(Cu, 2), (torch.empty(3, dtype=torch.int32, device="meta"),), mode="profile", profile="b1-eager", config=Cu)
    ok = rep.ok and not rep.family
    print(f"{'REFUSED' if ok else 'FAIL'}  lost sharing (untied lm_head) claiming Serve_v1: family {'not matched' if ok else 'MATCHED'}; flat program derived: {rep.ok}")
    bad += 0 if ok else 1
    # padded: a mux INSIDE the step body (spec §6 forbids it) derives as its own program but cannot claim Serve_v3
    C3 = dict(C, EOS_ID=0)
    with torch.device("meta"):
        rep = derive_report(ex.PaddedGreedyServe(C3, 2, eos_id=0, guard_inside=True), (torch.empty(3, dtype=torch.int32, device="meta"),),
                            mode="profile", profile="b1-eager-v2", config=C3)
    ok = rep.ok and not rep.family
    print(f"{'REFUSED' if ok else 'FAIL'}  guard inside the step body claiming Serve_v3: family {'not matched' if ok else 'MATCHED'}; flat program derived: {rep.ok}")
    bad += 0 if ok else 1
    print(f"{len(cases) + 2 - bad}/{len(cases) + 2} negatives refused")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="derive the greedy serving loop (prefill LP + STEPS decode) and write program artefacts")
    s.add_argument("--config", default="B0")
    s.add_argument("--lp", type=int, default=32)
    s.add_argument("--steps", type=int, default=15)
    s.add_argument("--mode", default="profile", choices=("profile", "reference"))
    s.add_argument("--profile", default="b1-eager")
    s.add_argument("--out", required=True)
    s.add_argument("--rows", default="sampled", choices=("sampled", "vllm", "all"),
                   help="final-norm / LM-head row policy: sampled = last row only (Serve_v1); vllm = final norm over every prefill row, "
                        "LM head over the sampled row (vLLM's model forward + LogitsProcessor); all = norm and LM head over every row")
    s.add_argument("--map-to-family", action="store_true", help="when no family matches, write serve_v1-map.json (superset map to Serve_v1)")
    s.add_argument("--expect-digest", default=None, help="exit 3 unless the derived digest equals this")
    s.add_argument("--padded", action="store_true", help="derive the padded circuit (PaddedGreedyServe; oracle Serve_v3{C3,LP,D_MAX=STEPS})")
    s.add_argument("--eos-id", type=int, default=0, help="EOS id of C3 (B0: 0)")
    s.set_defaults(fn=cmd_serve)
    l = sub.add_parser("layer", help="derive one decoder layer at T positions")
    l.add_argument("--config", default="B0")
    l.add_argument("--t", type=int, default=4)
    l.add_argument("--pos0", type=int, default=0)
    l.add_argument("--not-first", dest="first", action="store_false")
    l.add_argument("--profile", default="b1-eager")
    l.add_argument("--out", default=None)
    l.set_defaults(fn=cmd_layer)
    m = sub.add_parser("mlp", help="reference-mode 2-layer MLP")
    m.add_argument("--out", default=None)
    m.set_defaults(fn=cmd_mlp)
    h = sub.add_parser("hf", help="reference-mode HF (transformers-shaped) classes: layer with RoPE | LlamaForCausalLM shape | greedy request")
    h.add_argument("what", choices=["layer", "lm", "greedy"])
    h.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    h.add_argument("--t", type=int, default=3, help="positions (layer) / prompt tokens (lm, greedy)")
    h.add_argument("--steps", type=int, default=2)
    h.add_argument("--h", type=int, default=16)
    h.add_argument("--nh", type=int, default=2)
    h.add_argument("--kvh", type=int, default=1)
    h.add_argument("--d", type=int, default=8)
    h.add_argument("--i", type=int, default=32)
    h.add_argument("--v", type=int, default=32)
    h.add_argument("--nl", type=int, default=2)
    h.add_argument("--bias", action="store_true", help="q/k/v bias (Qwen2-style)")
    h.add_argument("--out", default=None)
    h.set_defaults(fn=cmd_hf)
    n = sub.add_parser("negatives", help="run every negative case; exit 1 unless all are refused")
    n.set_defaults(fn=cmd_negatives)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
