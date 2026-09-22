"""Correspondence with the Program as a parameter (lane CAPCORR, GUIDANCE §43.1 / §43.5 S4–S5).

    Gate -> exactly one Value<w>.
    Composite interfaces -> ordered collections of value references.
    Commitment positions -> physical authentication locations for those values,
                            not semantic components of gate outputs.

`verity_vllm.correspondence.core.run` locates every committed capture leaf in *a* program and re-evaluates the located
call gate by gate; historically the program came from a profile port's `static_config.py` (a hand-built registry
`Serve`). This module makes the program an explicit argument so that an AUTOMATICALLY DERIVED program (lane DERIVE's
importer) is a one-argument swap, and adds the three things the derived path needs that the port path did not have:

* `load_program(spec)` — one string names how the program is obtained:
    `registry:<port dir>` / `registry:B0` / `registry:B1`   the hand-built registry program (harness validation
                                                             baseline; NOT the deliverable, mandate boundary 6);
    `python:<module>:<callable>`                             any importable factory returning a `Program`
                                                             (DERIVE's importer entry point; called with LP=, STEPS=
                                                             when its signature accepts them);
    `descriptor:<path.json[.gz]>`                            a serialised descriptor, decoded through the codec.
* `program_identity(program)` — digest + primitive-vocabulary version (`frontend.provenance.registry_version`) +
  gate counts, the record every downstream artefact pins.
* `binding_check(identity, record_dir, ...)` — REFUSES (a list of reasons, nonzero exit in the CLI) when the
  supplied program's digest is not the record's pin, when the kind-vocabulary version differs from the expected one,
  or when the semantic profile differs. Nothing heavy runs before this check passes.
* `StructuralCircuitMap` — capture identity -> (call, gates) that does NOT rely on the hand-built call-name
  annotations (`t=<t>/layers.<L>/pre`, ...): names are not part of the descriptor/digest and a derived program may
  carry other hints. The map is rebuilt from the program's canonical root-call order and function identities and
  every assignment is checked against the dataflow (Attention's `T`, the Embedding token operand, post's `attn`
  operand, Final's inputs), so a mislabeled call is an error, never a silently wrong address.
* `lower_call_to_subcircuit(call)` — the program's gates of one `GemmCoordinate` call rewritten as a
  `typed-obligation/v0` subcircuit over the `veritor.tc-ampere-bf16` gate set (rule table `TC_LOWERING_RULES`),
  so an SP1 statement is built FROM the program's gates and its subcircuit digest can be compared with what the
  certificate's proofs were stated over (`veritor.interface.programs.gemm_coordinate`).
"""
from __future__ import annotations

import gzip
import hashlib
import importlib
import importlib.util
import inspect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity.ir.codec import decode_program, program_digest
from verity_vllm.correspondence.core import CaptureIdentity, CircuitMap, NotInProgram, _returns_component, _select_child
from verity.ir.defs import REGISTRY, bind
from verity.ir.layout import FunctionCall
from verity.ir.program import Program

# ---------------------------------------------------------------------------------------------------------
# program sources
# ---------------------------------------------------------------------------------------------------------


@dataclass
class LoadedProgram:
    """A program plus how it was obtained (recorded in every artefact)."""
    program: Any  # Program (or a DecodedProgram wrapped by `_DecodedAsProgram`)
    spec: str
    source_kind: str  # registry | python | descriptor
    source: dict[str, Any] = field(default_factory=dict)
    # the static configuration record when the source exposes one (registry ports do; a derived program may
    # expose it via `program.annotations["static_configuration"]`)
    config: dict[str, Any] | None = None
    LP: int | None = None
    STEPS: int | None = None


class _DecodedAsProgram:
    """Adapter giving a codec-decoded program the surface `verity_vllm.correspondence.core.run` uses."""

    def __init__(self, decoded, desc: dict) -> None:
        self.decoded = decoded
        self.root = decoded.root
        self.circuit = decoded.circuit
        self.annotations = dict(decoded.annotations)
        self.desc = desc

    @property
    def fn(self):
        class _F:  # the program function identity (the descriptor's `program_fn` string)
            id = self.decoded.program_fn_id
        return _F()

    @property
    def gates(self) -> int:
        return self.root.gates

    @property
    def input_gates(self) -> int:
        return sum(n.gates for n in self.root.body.nodes if getattr(n.fn, "family", None) == "Input")

    def input_coll(self, pi: int):
        raise NotImplementedError("a descriptor-decoded program carries no parameter collections (names are not "
                                  "encoded); use `python:` for a derived program with typed parameters")


def _parse_lp_steps(s: str | None) -> tuple[int | None, int | None]:
    if not s:
        return None, None
    m = re.search(r"LP=(\d+)", s)
    n = re.search(r"STEPS=(\d+)", s)
    return (int(m.group(1)) if m else None), (int(n.group(1)) if n else None)


def load_program(spec: str, *, LP: int | None = None, STEPS: int | None = None, repo_root: Path | None = None) -> LoadedProgram:
    """See the module docstring for the spec grammar. `LP`/`STEPS` are passed to factories that accept them."""
    if ":" not in spec:
        raise ValueError(f"program spec {spec!r} needs a `kind:` prefix (registry: | python: | descriptor:)")
    kind, _, rest = spec.partition(":")
    if kind == "registry":
        from verity_vllm.registry import b1
        if rest in ("B0", "B1"):
            C = {"B0": b1.B0_CONFIG, "B1": b1.B1_CONFIG}[rest]
            if LP is None or STEPS is None:
                raise ValueError("registry:B0/B1 needs LP and STEPS")
            p = Program(bind(b1.Serve, C=C, LP=LP, STEPS=STEPS))
            return LoadedProgram(p, spec, "registry", {"definition": "verity_vllm.registry.b1.Serve", "config": rest}, dict(C), LP, STEPS)
        port = Path(rest)
        if repo_root is not None and not port.is_absolute() and not port.exists():
            port = repo_root / port
        file = port if port.is_file() else port / "static_config.py"
        if not file.exists():
            raise FileNotFoundError(f"registry port {rest!r}: no static_config.py at {file}")
        mspec = importlib.util.spec_from_file_location(f"capcorr_port_{port.stem}", file)
        mod = importlib.util.module_from_spec(mspec)
        assert mspec.loader is not None
        mspec.loader.exec_module(mod)
        if LP is None or STEPS is None:
            raise ValueError("registry:<port> needs LP and STEPS")
        p = mod.program(LP=LP, STEPS=STEPS)
        return LoadedProgram(p, spec, "registry", {"port": str(file), "port_id": getattr(mod, "PORT_ID", port.name)},
                             dict(getattr(mod, "CONFIG", {})) or None, LP, STEPS)
    if kind == "python":
        modname, _, attr = rest.rpartition(":")
        if not modname or not attr:
            raise ValueError(f"python spec {spec!r} must be python:<module>:<callable>")
        mod = importlib.import_module(modname)
        fn = getattr(mod, attr)
        kwargs: dict[str, Any] = {}
        try:
            params = inspect.signature(fn).parameters
            if "LP" in params and LP is not None:
                kwargs["LP"] = LP
            if "STEPS" in params and STEPS is not None:
                kwargs["STEPS"] = STEPS
        except (TypeError, ValueError):
            pass
        p = fn(**kwargs)
        cfg = None
        if isinstance(getattr(p, "annotations", None), dict):
            cfg = p.annotations.get("static_configuration") or p.annotations.get("config")
        if cfg is None:
            # the root call's bound statics ARE the program's static configuration (a derived program carries no port file):
            # Serve_v1{C=<config>, LP, STEPS} -> C; the binding stage compares it with the record's pinned configuration
            st = getattr(getattr(p, "fn", None), "statics", None)
            if isinstance(st, dict) and isinstance(st.get("C"), dict):
                cfg = dict(st["C"])
        lp2, st2 = _parse_lp_steps(getattr(getattr(p, "fn", None), "id", None))
        return LoadedProgram(p, spec, "python", {"module": modname, "callable": attr, "kwargs": kwargs,
                                                 "source_file": getattr(mod, "__file__", None)}, cfg, LP or lp2, STEPS or st2)
    if kind == "descriptor":
        path = Path(rest)
        raw = gzip.open(path, "rb").read() if path.suffix == ".gz" else path.read_bytes()
        desc = json.loads(raw)
        dec = decode_program(desc, REGISTRY)
        p = _DecodedAsProgram(dec, desc)
        lp2, st2 = _parse_lp_steps(desc.get("program_fn"))
        return LoadedProgram(p, spec, "descriptor", {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}, None, LP or lp2, STEPS or st2)
    raise ValueError(f"unknown program source kind {kind!r} in {spec!r}")


# ---------------------------------------------------------------------------------------------------------
# identity + binding
# ---------------------------------------------------------------------------------------------------------


def vocabulary_version() -> dict[str, Any]:
    """The kind-vocabulary version: `verity_vllm.frontend.provenance.registry_version`'s record computed over the DECLARED
    vocabulary (the module-level `PrimitiveDefinition`s of `registry.prims` and `CompositeDefinition`s of `registry.b1`)
    rather than over whatever `REGISTRY.defs` holds at call time. The two agree in a fresh process; after a program is
    built, `REGISTRY.defs` also holds shape-instantiated primitives (`GatherBf16x49152_v1`, `Const32[…]`, `Input16_v1`)
    and `registry_version()`'s digest changes with the process state (`33ee156d…` fresh vs `5d3d3d3f…` after a B0 build),
    which makes it unusable as a pin. Same record schema, so a digest recorded by DERIVE's fresh call compares equal."""
    import inspect as _inspect
    from verity.ir.codec import canonical_json
    from verity.ir.defs import CompositeDefinition, PrimitiveDefinition
    from verity_vllm.registry import b1, prims
    prim_rows = []
    comp_rows = []
    seen: set[str] = set()
    for mod, cls in ((prims, PrimitiveDefinition), (b1, CompositeDefinition)):
        for _n, d in _inspect.getmembers(mod):
            if not isinstance(d, cls) or d.id in seen:
                continue
            seen.add(d.id)
            if isinstance(d, PrimitiveDefinition):
                if getattr(d, "family", None) in ("Const", "Input"):
                    continue
                prim_rows.append([d.id, [[n, t.to_json()] for n, t in d.params], d.ret.to_json(), d.conformance])
            else:
                comp_rows.append([d.id, list(d.statics), d.conformance])
    prim_rows.sort(key=lambda r: r[0])
    comp_rows.sort(key=lambda r: r[0])
    sources = {mod.__name__: hashlib.sha256(_inspect.getsource(mod).encode()).hexdigest() for mod in (prims, b1)}
    rec = {"schema": "verity-ir/registry-version/v0", "primitives": prim_rows, "composites": comp_rows, "sources": sources}
    digest = hashlib.sha256(canonical_json(rec)).hexdigest()
    return {"schema": rec["schema"], "digest": digest, "primitives": len(prim_rows), "composites": len(comp_rows), "sources": sources,
            "computed_over": "declared vocabulary (module-level definitions of registry.prims + registry.b1)"}


def program_identity(lp: LoadedProgram) -> dict[str, Any]:
    """Digest, vocabulary version and counts — what every artefact pins about the program."""
    from verity_vllm.frontend.provenance import registry_version
    p = lp.program
    digest = program_digest(p.desc) if isinstance(p, _DecodedAsProgram) else program_digest(p)
    rv = vocabulary_version()
    rv["registry_version_at_call"] = registry_version()["digest"]  # informative: the state-dependent value
    ident = {"function": p.fn.id, "digest": digest, "gates": int(p.gates), "input_gates": int(p.input_gates),
             "vocabulary": rv,
             "source": {"spec": lp.spec, "kind": lp.source_kind, **lp.source}, "LP": lp.LP, "STEPS": lp.STEPS}
    if lp.config is not None:
        ident["static_configuration"] = {k: v for k, v in lp.config.items() if k != "profile"}
        ident["profile"] = lp.config.get("profile")
    ann = getattr(p, "annotations", None)
    if isinstance(ann, dict):
        if "capcorr_derive_summary" in ann:
            ident["derive"] = ann["capcorr_derive_summary"]
        dv = ann.get("derive")
        if isinstance(dv, dict):  # DERIVE's provenance record: keep the identity-bearing parts, not the per-node application log
            ident.setdefault("derive", {})
            ident["derive"]["provenance"] = {k: dv.get(k) for k in ("schema", "frontend", "derive", "torch", "mode", "profile", "rules_applied", "result")
                                             if k in dv}
            ident["derive"]["provenance"]["vocabulary"] = (dv.get("vocabulary") or {}).get("id") if isinstance(dv.get("vocabulary"), dict) else dv.get("vocabulary")
            ident["derive"]["provenance"]["ruleset"] = (dv.get("ruleset") or {}).get("id") if isinstance(dv.get("ruleset"), dict) else None
            ident["derive"]["provenance"]["registry"] = (dv.get("registry") or {}).get("digest") if isinstance(dv.get("registry"), dict) else None
            src = dv.get("source") or {}
            ident["derive"]["provenance"]["source"] = {k: src.get(k) for k in ("kind", "class", "name", "example_inputs", "mode", "profile") if k in src}
            fam = dv.get("family") or {}
            ident["derive"]["family"] = {k: fam.get(k) for k in ("family", "matched", "flat_nodes", "family_leaves") if k in fam}
    return ident


def record_pins(record_dir: Path) -> dict[str, Any]:
    """What the certified record pins about its program, profile and prover identity."""
    rd = Path(record_dir)
    pins: dict[str, Any] = {"record": rd.name}
    pj = json.loads((rd / "program" / "program.json").read_text())
    pins["program_digest"] = pj["program"]["digest"]
    pins["program_function"] = pj["program"]["function"]
    pins["program_profile"] = pj["program"].get("profile")
    pins["static_configuration"] = pj["program"].get("static_configuration")
    pins["LP"], pins["STEPS"] = pj["program"].get("LP"), pj["program"].get("STEPS")
    cert_path = rd / "certificate.json"
    if cert_path.exists():
        cert = json.loads(cert_path.read_text())
        pins["certificate_program_digest"] = cert["program"]["program_digest"]
        pins["certificate_sha256"] = hashlib.sha256(cert_path.read_bytes()).hexdigest()
        pins["certificate_schema"] = cert.get("schema")
        kc = cert.get("kind_coverage") or {}
        pins["proof_kind_families"] = {"tc": (cert.get("proofs") or {}).get("families_proved"),
                                       "coverage": {k: v.get("gate_set") for k, v in (kc.get("families") or {}).items()},
                                       "kinds_proved": kc.get("kinds_proved")}
        inst = cert.get("instance") or {}
        sp = inst.get("semantic_profile") or {}
        pins["semantic_profile_sha256"] = sp.get("sha256")
        pins["semantic_profile"] = Path(sp["path"]).stem if sp.get("path") else None
        pins["trusted_replay_equivalence_class"] = inst.get("trusted_replay_equivalence_class")
    man = json.loads((rd / "manifest.json").read_text())
    inp = man.get("inputs", {})
    pins["manifest_program_digest"] = inp.get("program_digest")
    pins["elf_sha256"], pins["vk_hash"], pins["host_sha256"] = inp.get("elf_sha256"), inp.get("vk_hash"), inp.get("host_sha256")
    # the rest of the ONE-system pins (GUIDANCE §44.1): output manifest, sampling policy, commitments, plan, challenge
    for k in ("production_record_sha256", "served_tokens_sha256", "sampling_policy_sha256", "commitments_sha256", "plan_digest", "challenge_digest",
              "description_sha256", "workload_sha256"):
        pins[k] = inp.get(k)
    spj = record_dir / "sampling_policy.json"
    if spj.exists():
        pins["sampling_policy"] = json.loads(spj.read_text())
        pins["sampling_policy_file_sha256"] = hashlib.sha256(spj.read_bytes()).hexdigest()
    summ_path = rd / "summary.json"
    if summ_path.exists():
        summ = json.loads(summ_path.read_text())
        arch = (((summ.get("stages") or {}).get("verify") or {}).get("bindings_detail") or {}).get("architecture") or {}
        pins["semantic_profile"] = pins.get("semantic_profile") or arch.get("semantic_profile")
        pins["semantic_profile_sha256"] = pins.get("semantic_profile_sha256") or arch.get("semantic_profile_sha256")
    prod = inp.get("production_manifest") or {}
    # capture changes execution relative to stock vLLM (enforce_eager, batch-invariant kernels, dispatch-mode interception,
    # recorder patches): the record states that configuration; downstream artefacts must carry it, never claim stock execution
    pins["production_configuration"] = {"compilation": prod.get("compilation"), "adapter": prod.get("adapter"),
                                        "engine_flags": prod.get("engine_flags") or prod.get("env"),
                                        "semantic_profile_requires": None}
    bj = rd / "bundle" / "bundle.json"
    pins["capture_profile_id"] = json.loads(bj.read_text()).get("profile_id") if bj.exists() else None
    return pins


def binding_check(identity: dict[str, Any], pins: dict[str, Any], *, expect_vocabulary: str | None = None,
                  expect_profile_sha256: str | None = None) -> list[str]:
    """The refusal conditions. Empty list = bound. Every reason names both sides."""
    problems: list[str] = []
    d = identity["digest"]
    for k in ("program_digest", "certificate_program_digest", "manifest_program_digest"):
        v = pins.get(k)
        if v is not None and v != d:
            problems.append(f"program digest mismatch: supplied program {identity['function']} digest {d[:16]}… != record {k} {v[:16]}…")
    if pins.get("LP") is not None and identity.get("LP") is not None and pins["LP"] != identity["LP"]:
        problems.append(f"LP mismatch: program {identity['LP']} != record {pins['LP']}")
    if pins.get("STEPS") is not None and identity.get("STEPS") is not None and pins["STEPS"] != identity["STEPS"]:
        problems.append(f"STEPS mismatch: program {identity['STEPS']} != record {pins['STEPS']}")
    dreg = ((identity.get("derive") or {}).get("provenance") or {}).get("registry")
    if dreg is not None and dreg != identity["vocabulary"]["digest"]:
        problems.append(f"kind-vocabulary version mismatch: program was DERIVED under registry-version {dreg[:16]}… but is evaluated under "
                        f"{identity['vocabulary']['digest'][:16]}…")
    if expect_vocabulary is not None and identity["vocabulary"]["digest"] != expect_vocabulary:
        problems.append(f"kind-vocabulary version mismatch: program evaluated under registry-version {identity['vocabulary']['digest'][:16]}… "
                        f"!= expected {expect_vocabulary[:16]}…")
    prof = identity.get("profile")
    if prof is not None and pins.get("program_profile") is not None and prof != pins["program_profile"]:
        problems.append(f"profile mismatch: program static profile {json.dumps(prof, sort_keys=True)[:80]}… != record program.json profile "
                        f"{json.dumps(pins['program_profile'], sort_keys=True)[:80]}…")
    sc = identity.get("static_configuration")
    if sc is not None and pins.get("static_configuration") is not None and sc != pins["static_configuration"]:
        diff = sorted(set(sc) ^ set(pins["static_configuration"])) + [k for k in sc if k in pins["static_configuration"] and sc[k] != pins["static_configuration"][k]]
        problems.append(f"static configuration mismatch on {diff}")
    if expect_profile_sha256 is not None and pins.get("semantic_profile_sha256") not in (None, expect_profile_sha256):
        problems.append(f"semantic profile mismatch: record captured under {pins.get('semantic_profile')} {str(pins.get('semantic_profile_sha256'))[:16]}… "
                        f"!= expected {expect_profile_sha256[:16]}…")
    return problems


# ---------------------------------------------------------------------------------------------------------
# structural circuit map (no reliance on hand-built call-name annotations)
# ---------------------------------------------------------------------------------------------------------


class StructuralCircuitMap(CircuitMap):
    """`CircuitMap` whose `named` table is rebuilt from the program's canonical root-call order and function
    identities, then checked against the dataflow. Works for any program whose root calls are the `Serve_v1`
    circuit's (`Embedding`, `LayerPre`, `Attention`, `LayerPost`, `Final` at every position), whatever their names.

    Naming reconstructed: `t=<t>/embed`, `t=<t>/layers.<L>/{pre,attn,post}`, `t=<t>/final`. Checks (all raise
    `ValueError` on failure): counts per function; `Attention_v1{T}` static T == t+1; `post.args[0:NH*D]` are the
    `attn` call's returns; Embedding at a decode position reads the previous Final's return; the prefill Embedding
    at position i reads prompt Input gate i (when the program exposes `input_coll`)."""

    def __init__(self, program: Any, names: dict[str, str] | None = None, *, check: bool = True) -> None:
        self.program = program
        self.circuit = program.circuit
        self.names = names or {}
        self.named: dict[str, FunctionCall] = {}
        self.structural_names: dict[tuple[int, int], str] = {}  # (lo, hi) -> reconstructed name
        self.annotation_agreement: dict[str, int] = {"agree": 0, "differ": 0, "unnamed": 0}
        roots = list(self.circuit.root_calls())
        by_fn: dict[str, list[FunctionCall]] = {}
        for c in roots:
            by_fn.setdefault(c.name, []).append(c)
        embeds, pres, attns, posts, finals = (by_fn.get(k, []) for k in ("Embedding", "LayerPre", "Attention", "LayerPost", "Final"))
        n_pos = len(embeds)
        n_fin = len(finals)
        if n_pos == 0 or n_fin == 0:
            raise ValueError("structural map: no Embedding/Final root calls — not a Serve-shaped program")
        steps = n_fin - 1
        lp = n_pos - steps
        if lp <= 0:
            raise ValueError(f"structural map: {n_pos} positions but {n_fin} Final calls")
        if not (len(pres) == len(attns) == len(posts)) or len(pres) % n_pos:
            raise ValueError(f"structural map: pre/attn/post counts {len(pres)}/{len(attns)}/{len(posts)} vs {n_pos} positions")
        nl = len(pres) // n_pos
        self.LP, self.STEPS, self.NL = lp, steps, nl
        # canonical emission order (`_forward_positions`): prefill — embeds for i<LP; per layer: LP pres, then (attn, post) per i;
        # then Final(LP-1); decode step s — embed, per layer pre/attn/post, Final.  Rather than assume, read the order from the
        # roots themselves and label by counting.
        pos_of_embed = list(range(n_pos))
        for i, c in enumerate(embeds):
            self._name(c, f"t={pos_of_embed[i]}/embed")
        # pres: positions cycle inside layers for prefill, then one per layer per decode step
        # prefill block: layer-major (L outer, i inner); decode blocks: one position each
        k = 0
        for L in range(nl):
            for i in range(lp):
                self._name(pres[k], f"t={i}/layers.{L}/pre"); k += 1
        for s in range(steps):
            for L in range(nl):
                self._name(pres[k], f"t={lp + s}/layers.{L}/pre"); k += 1
        k = 0
        for L in range(nl):
            for i in range(lp):
                self._name(attns[k], f"t={i}/layers.{L}/attn"); self._name(posts[k], f"t={i}/layers.{L}/post"); k += 1
        for s in range(steps):
            for L in range(nl):
                self._name(attns[k], f"t={lp + s}/layers.{L}/attn"); self._name(posts[k], f"t={lp + s}/layers.{L}/post"); k += 1
        for j, c in enumerate(finals):
            self._name(c, f"t={lp - 1 + j}/final")
        if check:
            self._check_dataflow()

    def _name(self, c: FunctionCall, name: str) -> None:
        self.named[name] = c
        self.structural_names[(c.lo, c.hi)] = name
        ann = c.node.name
        if ann is None:
            self.annotation_agreement["unnamed"] += 1
        elif ann == name:
            self.annotation_agreement["agree"] += 1
        else:
            self.annotation_agreement["differ"] += 1

    @staticmethod
    def _first_args(c: FunctionCall, n: int) -> tuple[int, ...]:
        """The first `n` leaves of parameter 0 (never `c.args`: a layer's weights record has 3.5 M leaves)."""
        from verity.ir.layout import param_leaf
        sc = c.scope()
        return tuple(param_leaf(sc, 0, j) for j in range(n))

    def _check_dataflow(self) -> None:
        errs: list[str] = []
        for name, c in self.named.items():
            m = re.match(r"^t=(\d+)/layers\.(\d+)/(pre|attn|post)$", name)
            if m and m.group(3) == "attn":
                t = int(m.group(1))
                T = c.fn.bindings.get("T") if hasattr(c.fn, "bindings") else None
                if T is not None and T != t + 1:
                    errs.append(f"{name}: Attention static T={T} but position t={t}")
            if m and m.group(3) == "post":
                t, L = int(m.group(1)), int(m.group(2))
                attn = self.named.get(f"t={t}/layers.{L}/attn")
                if attn is not None:
                    rets = tuple(attn.returns)
                    if self._first_args(c, len(rets)) != rets:
                        errs.append(f"{name}: arg[0:{len(rets)}] is not the returns of {attn!r}")
        # Embedding of a decode position reads the previous Final's token
        for s in range(self.STEPS):
            t = self.LP + s
            emb, fin = self.named.get(f"t={t}/embed"), self.named.get(f"t={t - 1}/final")
            if emb is not None and fin is not None and self._first_args(emb, 1) != tuple(fin.returns[:1]):
                errs.append(f"t={t}/embed: token operand {self._first_args(emb, 1)} is not Final(t={t - 1}) return {fin.returns[:1]}")
        try:
            pc = self.program.input_coll(0)
            from verity.ir.layout import resolve
            for i in range(self.LP):
                g = resolve(self.circuit.root_scope, pc.refs[i])
                emb = self.named[f"t={i}/embed"]
                if self._first_args(emb, 1)[0] != g:
                    errs.append(f"t={i}/embed: token operand {self._first_args(emb, 1)[0]} is not prompt Input gate {g}")
        except (NotImplementedError, AttributeError):
            pass
        if errs:
            raise ValueError("structural map dataflow check failed: " + "; ".join(errs[:6]) + (f" (+{len(errs) - 6} more)" if len(errs) > 6 else ""))


def canonical_reference(call: FunctionCall, gates: tuple[int, ...] | None = None) -> dict[str, Any]:
    """The program-side address of a value: function identity, statics, call interval, returns range."""
    fn = call.fn
    statics = {}
    if hasattr(fn, "bindings"):
        for k, v in fn.bindings.items():
            statics[k] = v if isinstance(v, (int, float, str, bool)) or v is None else str(v)[:64]
    gates = call.returns if gates is None else gates
    contiguous = all(gates[i + 1] == gates[i] + 1 for i in range(len(gates) - 1))
    return {"function": fn.id, "kind": fn.name, "version": getattr(getattr(fn, "defn", fn), "version", None), "statics": statics,
            "call_interval": [call.lo, call.hi], "value_gates": len(gates),
            "returns_gates": [gates[0], gates[-1] + 1] if gates and contiguous else None, "contiguous": contiguous,
            "value_gate_span": [gates[0], gates[-1] + 1] if gates else None}


# ---------------------------------------------------------------------------------------------------------
# program gates -> typed-obligation subcircuit (the tc-ampere-bf16 lowering rule)
# ---------------------------------------------------------------------------------------------------------

#: How a `verity.ir` primitive inside a `GemmCoordinate_v1{K}` call is spelled in the `veritor.tc-ampere-bf16@2`
#: gate set. `AmpereBF16TcDot16(acc, a[16], b[16])` with `acc` a `Const<32>` zero gate is the zero-accumulator
#: step `tc_dot16_0(a, b)` (the constant is folded: the gate set has no constant gate); otherwise `tc_dot16(acc, a,
#: b)`; `F2fpBf16(x)` is `f32_to_bf16(x)`. Anything else in the call body is a lowering error (the call is not the
#: registered GemmCoordinate body), never silently dropped.
TC_LOWERING_RULES = {
    "id": "capcorr.tc-lowering", "version": 1, "gate_set": "veritor.tc-ampere-bf16@2",
    "rules": {"AmpereBF16TcDot16_v1[acc=Const 0]": "tc_dot16_0(a, b)", "AmpereBF16TcDot16_v1": "tc_dot16(acc, a, b)",
              "F2fpBf16_v1": "f32_to_bf16(x)", "Const<32>(0)": "folded into tc_dot16_0"},
}


class LoweringError(ValueError):
    pass


def lower_call_to_subcircuit(call: FunctionCall):
    """A `GemmCoordinate` call's gates as a `typed-obligation/v0` `Subcircuit` (`veritor.interface`). Incoming values
    are the call's `args` in signature order (x[0..K], wrow[0..K]); the one outgoing value is the call's return."""
    from veritor.interface.programs import make_subcircuit
    from veritor.interface.typed_obligation import GATE, IN, Definition

    args = list(call.args)
    pos: dict[int, int] = {}
    for i, g in enumerate(args):
        pos.setdefault(g, i)
    gates: list[tuple[str, list[tuple[str, int]]]] = []
    idx: dict[int, int] = {}
    consts: dict[int, int] = {}

    def ref(o: int) -> tuple[str, int]:
        if o in idx:
            return (GATE, idx[o])
        if o in pos:
            return (IN, pos[o])
        raise LoweringError(f"gate {o} is neither inside the call nor one of its arguments")

    for gv in call:
        fam = getattr(gv.prim, "family", None)
        if fam == "Const":
            consts[gv.index] = gv.prim.evaluate()
            continue
        if gv.prim.name == "AmpereBF16TcDot16":
            acc = gv.operands[0]
            if acc in consts:
                if consts[acc] != 0:
                    raise LoweringError(f"gate {gv.index}: accumulator constant {consts[acc]} is not zero")
                gates.append(("tc_dot16_0", [ref(o) for o in gv.operands[1:]]))
            else:
                gates.append(("tc_dot16", [ref(o) for o in gv.operands]))
        elif gv.prim.name == "F2fpBf16":
            gates.append(("f32_to_bf16", [ref(o) for o in gv.operands]))
        else:
            raise LoweringError(f"gate {gv.index}: primitive {gv.prim.id} has no rule in {TC_LOWERING_RULES['id']}")
        idx[gv.index] = len(gates) - 1
    out = []
    for g in call.returns:
        if g not in idx:
            raise LoweringError(f"return gate {g} was not lowered")
        out.append(idx[g])
    K = len(args) // 2
    if len(args) != 2 * K or K % 16:
        raise LoweringError(f"{len(args)} incoming values is not 2*K with K a multiple of 16")
    return make_subcircuit([16] * len(args), gates, out, Definition("GemmCoordinate", (("K", K),), "Gemm"))


def locate_coordinate(cmap: CircuitMap, capture_id: str, j: int) -> tuple[FunctionCall, FunctionCall]:
    """The `Gemm` call a capture id addresses and its j-th `GemmCoordinate` member."""
    call, _ = cmap.locate(CaptureIdentity.parse(capture_id))
    if call.name not in ("Gemm",) and not call.name.startswith("Gemm"):
        raise NotInProgram(f"{capture_id}: located call {call.id} is not a Gemm")
    kids = list(call.children())
    if not (0 <= j < len(kids)):
        raise NotInProgram(f"{capture_id}: coordinate j={j} outside the Gemm's {len(kids)} members")
    return call, kids[j]


__all__ = ["LoadedProgram", "load_program", "program_identity", "record_pins", "binding_check", "StructuralCircuitMap",
           "canonical_reference", "lower_call_to_subcircuit", "locate_coordinate", "TC_LOWERING_RULES", "LoweringError",
           "_returns_component", "_select_child"]
