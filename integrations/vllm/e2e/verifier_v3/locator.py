"""The located unit of a selected VU, derived by the VERIFIER from the fixed circuit (integrator 19:50Z, R4 M1 / R9 residual EA).

``mapping.units.jsonl`` is written by the prover's ``rec_p1_obligations.py``.  A verifier that reads it to learn WHERE a selected VU
lives (request, t, layer, Gemm, member j / weights leaf + offset) builds its expected statement at the prover's location: relabel one unit
to a consistent decoy and prove the decoy, and the record passes.  So the verifier derives the location itself, here:

* :func:`build_pinned_program` -- the program of the instance's ``program.identity`` (``case``/``LP``/``STEPS``) built in the verifier
  process (``verity.ir``), refused unless its specialised id, gate count and ``program_digest`` equal what the instance / population pin;
* :class:`ProgramLocator` -- gate interval -> ``(structural root call, Gemm call, member j)``; Input gate -> ``(weights leaf id, word
  offset)`` | ``(prompt, t)``.  Moved here from ``scripts/rec_p1_obligations.py`` (which now imports it) so both sides run the
  same code -- and the verifier RUNS it, never reads its output from the prover;
* :func:`locate_selected` -- every row of the verifier's own ``chronology/selected.json`` (V2-re-derived; ``gate_interval`` and ``kind``
  checked against the population by ``partition_sample.verify``) -> a unit in the shape of a ``mapping.units.jsonl`` row;
* :func:`compare_units` -- the prover's file, when present, is compared per unit on every field the deriver uses; any disagreement is
  a refusal (the prover's file is never the source).

Cost on REC-P1-B0 (30 172 selected VUs, 518.8 M gates): build 0.2 s, digest 0.25 s, locator 1.4 s, locate 26 093 units 2.8 s;
~100 MB RSS for the program.  Recorded in ``timings.json`` under the verifier's derivation of expected statements.
"""

from __future__ import annotations

import bisect
import os
import time
from collections.abc import Mapping
from typing import Any

__all__ = ["GEMM_OPS", "LOCATE_FIELDS", "ProgramLocator", "build_pinned_program", "make_locator", "locate_selected", "compare_units",
           # [E3 2026-09-11] positions of every gate (input | boundary | interior) and the expected statement as positions
           "GatePosition", "PositionRefusal", "Region", "Regions", "PopulationRows", "BoundaryIndex", "GateResolver",
           "call_at", "output_gates_of", "operand_gates_of", "gate_width", "serve_regions", "serve_calls", "calls_from_intervals", "vllm_row_gates",
           "inventory", "interior_layout_rule",
           # [E3, integrator D1] registered (non-Serve_v1) programs through the same verifier path
           "build_registered_program", "GenericLocator", "generic_regions", "position_adapter"]

#: [LZ] ``build_pinned_program`` builds the program lazily (``verity_vllm.ir.lazy.LazyProgram``: root body arithmetic, streamed digest -- memory
#: bounded by the query, not by the program's runs) unless this variable is ``0``; the digest and every ``locate()`` are identical to the
#: full ``Program(sp)`` / ``ProgramLocator`` (``tests/contract/lz_lazy_program_test.py``; `data/contract/lz-lazy-program/report.md`).
LAZY_PROGRAM_ENV = "VERIFIER_V3_LAZY_PROGRAM"

# Gemm child index within a structurally named root call -> the production op suffix and the description's op_type
GEMM_OPS = {("pre", 0): ("self_attn.qkv_proj.gemm", "gemm"), ("post", 0): ("self_attn.o_proj.gemm", "gemm"),
            ("post", 1): ("mlp.gate_up_proj.gemm", "gemm"), ("post", 2): ("mlp.down_proj.gemm", "gemm"), ("final", 0): (None, "lm_head")}

#: [A5/6 patch] kinds located in the attention body (coverage_v1.interior_layout.INTERIOR_KINDS + the head unit, base id)
def _attention_kinds() -> frozenset:
    try:
        from e2e.coverage_v1.attn_head_builder import HEAD_KIND
        from e2e.coverage_v1.interior_layout import INTERIOR_KINDS

        return frozenset(INTERIOR_KINDS) | {HEAD_KIND}
    except ImportError:  # pragma: no cover
        return frozenset()


ATTENTION_KINDS = _attention_kinds()


def _base_kind(kind: str) -> str:
    import re

    return re.sub(r"\{[^}]*\}", "", kind)


#: The unit fields the expectation deriver (``e2e.verifier_v3.expected.Deriver``) reads; a prover file must agree on all of them.
LOCATE_FIELDS = ("vu_index", "ru", "request_id", "kind", "class", "gate_interval", "locate")


class ProgramLocator:
    """Gate -> (structural root-call name, Gemm call, member j) and Input gate -> (weights leaf id, offset) | (prompt, t)."""

    def __init__(self, program, C: dict[str, Any]) -> None:
        from verity_vllm.correspondence.core import _LAYER_FIELDS, _WEIGHT_FIELDS, _coll_range
        from verity_vllm.correspondence.program import StructuralCircuitMap

        self.program = program
        self.C = dict(C)  # the pinned program's config (NH / KVH / D ...): builders' RecordHandle reads it
        self.cmap = StructuralCircuitMap(program)
        roots = sorted(self.cmap.named.items(), key=lambda kv: kv[1].lo)
        self.root_names = [n for n, _ in roots]
        self.root_calls = [c for _, c in roots]
        self.root_lo = [c.lo for c in self.root_calls]
        self._gemms: dict[str, list] = {}
        self._members: dict[tuple[str, int], tuple[dict[tuple[int, int], int], int]] = {}
        self._cmap = None
        self._xcands: dict[tuple[int, int | None], list] = {}
        circuit = program.circuit
        # weights: field collections -> [start, end) gate ranges with the weights leaf canonical id
        W = program.input_coll(1)
        ranges: list[tuple[int, int, str, str]] = []  # (start, end, canonical_id, field)
        s, e = _coll_range(circuit, W["embed"])
        ranges.append((s, e, _WEIGHT_FIELDS["embed"][0], "weights.embed"))
        for L in range(C["NL"]):
            w = W["layers"][L]
            for fname, (cid_t, _srcs) in _LAYER_FIELDS.items():
                if fname == "qkv_b" and not C.get("BIAS", True):
                    continue
                s, e = _coll_range(circuit, w[fname])
                ranges.append((s, e, cid_t.format(L=L), f"weights.layers[{L}].{fname}"))
        s, e = _coll_range(circuit, W["norm"])
        ranges.append((s, e, _WEIGHT_FIELDS["norm"][0], "weights.norm"))
        s, e = _coll_range(circuit, W["cos_sin"])
        ranges.append((s, e, _WEIGHT_FIELDS["cos_sin"][0], "weights.cos_sin"))
        ranges.sort()
        self.w_ranges = ranges
        self.w_starts = [r[0] for r in ranges]
        P = program.input_coll(0)
        self.prompt_range = _coll_range(circuit, P)
        self.input_gates = program.input_gates

    def root_of(self, g: int):
        i = bisect.bisect_right(self.root_lo, g) - 1
        if i < 0 or not (self.root_calls[i].lo <= g < self.root_calls[i].hi):
            return None, None
        return self.root_names[i], self.root_calls[i]

    def gemms_of(self, name: str, root) -> list:
        if name not in self._gemms:
            self._gemms[name] = [c for c in root.children() if c.name.startswith("Gemm")]
        return self._gemms[name]

    def locate_gemm_coordinate(self, lo: int, hi: int) -> dict[str, Any]:
        name, root = self.root_of(lo)
        if root is None:
            return {"refused": f"gate {lo} lies in no root call"}
        t_s, _, rest = name.partition("/")
        t = int(t_s[2:])
        gemms = self.gemms_of(name, root)
        gi = next((k for k, c in enumerate(gemms) if c.lo <= lo < c.hi), None)
        if gi is None:
            return {"refused": f"gate {lo} in root {name} lies in no Gemm call"}
        gemm = gemms[gi]
        key = (name, gi)
        if key not in self._members:
            kids = list(gemm.children())
            self._members[key] = {(c.lo, c.hi): k for k, c in enumerate(kids) if c.name.startswith("GemmCoordinate")}, len(kids)
        members, n_kids = self._members[key]
        j = members.get((lo, hi))
        if j is None:
            return {"refused": f"[{lo},{hi}) is not exactly one GemmCoordinate member of Gemm {name}#{gi} ({n_kids} members)"}
        kids = range(n_kids)
        if rest == "final":
            role, L = "final", None
        elif rest == "embed":
            return {"refused": f"Gemm inside {name}?"}
        else:
            layer_s, role = rest.split("/")
            L = int(layer_s.split(".")[1])
        op_suffix, op_type = GEMM_OPS[(role, gi)]
        return {"root": name, "t": t, "layer": L, "role": role, "gemm_index": gi, "j": j, "N": len(kids), "op_suffix": op_suffix, "op_type": op_type}

    # -- [A2 21:00Z, R9] the operands of a located GemmCoordinate member, from the PROGRAM's dataflow --------------------------------------
    # The member's declared interface (``FunctionCall.args``: K gates of ``x`` then K gates of ``wrow``) names the gates it reads; the x
    # gates are matched against the program gates of every capture-v2 value of the same (t, layer) (``verity_vllm.correspondence.core.CircuitMap``,
    # the verifier's own identity table), the wrow gates against the Input gates of the weights collections (:meth:`locate_input`).  So the
    # producer op / port / column slice and the weight tensor + row of every expectation come from the pinned program, and the prover's
    # ``description.json`` can only be COMPARED against them (``Deriver.coordinate``: V5.description).

    def _x_candidates(self, t: int, L: int | None) -> list[tuple[str, str, int, tuple[int, ...], dict[int, int]]]:
        key = (t, L)
        xc = self.__dict__.setdefault("_xcands", {})  # [LZ] LazyProgramLocator does not run ProgramLocator.__init__: create the caches on first use
        if key not in xc:
            from verity_vllm.correspondence.core import _LAYER_OPS, CaptureIdentity, CircuitMap

            if self.__dict__.get("_cmap") is None:
                # CircuitMap.locate only reads `.named` / `.names`; build it over the locator's own `cmap.named` (StructuralCircuitMap for a full
                # program, `_LazyNamed` arithmetic for a lazy one) instead of CircuitMap.__init__'s walk over every root call (O(runs) -- would
                # materialise the lazy program's body)
                cm = CircuitMap.__new__(CircuitMap)
                cm.program, cm.circuit, cm.names, cm.named = self.program, self.program.circuit, {}, self.cmap.named
                self._cmap = cm
            idents = [("norm", "norm.rmsnorm", p) for p in (0, 1)] if L is None else [(str(L), f"layers.{L}.{s}", p) for (s, p) in _LAYER_OPS]
            out = []
            for layer_s, op, port in idents:
                try:
                    _, gates = self._cmap.locate(CaptureIdentity("", "", t, "", layer_s, op, port))
                except (KeyError, IndexError, ValueError):
                    continue
                out.append((layer_s, op, port, tuple(gates), {g: i for i, g in enumerate(gates)}))
            xc[key] = out
        return xc[key]

    def gemm_operands(self, lo: int, hi: int) -> dict[str, Any]:
        """``{"K", "x": {"layer", "op", "port", "cols"}, "w": {"canonical_id", "row", "field"}}`` of the GemmCoordinate member ``[lo, hi)``,
        or ``{"refused": why}``.  ``cols`` is ``[a, b)`` when the member reads a slice of the producer's row, else None (the whole row)."""
        L = self.locate_gemm_coordinate(lo, hi)
        if "refused" in L:
            return L
        name, root = self.root_of(lo)
        gemm = self.gemms_of(name, root)[L["gemm_index"]]
        member = list(gemm.children())[L["j"]]
        K = int(member.arg_types()[0].leaves)
        args = tuple(member.args)
        if len(args) != 2 * K:
            return {"refused": f"GemmCoordinate member reads {len(args)} gates, not 2K={2 * K}"}
        xg, wg = args[:K], args[K:]
        hits = []
        for layer_s, op, port, gates, pos in self._x_candidates(L["t"], L["layer"]):
            i0 = pos.get(xg[0])
            if i0 is not None and gates[i0:i0 + K] == xg:
                hits.append({"layer": layer_s, "op": op, "port": port, "cols": None if (i0 == 0 and K == len(gates)) else [i0, i0 + K]})
        if len(hits) != 1:
            return {"refused": f"x operand of {name} Gemm#{L['gemm_index']}[{L['j']}] matches {len(hits)} captured values of (t={L['t']}, layer={L['layer']})"}
        w0 = self.locate_input(wg[0])
        if "refused" in w0 or w0.get("kind") != "weight":
            return {"refused": f"wrow operand gate {wg[0]} is not a weights Input gate: {w0}"}
        if wg != tuple(range(wg[0], wg[0] + K)) or w0["offset"] % K:
            return {"refused": f"wrow operand of {name} Gemm#{L['gemm_index']}[{L['j']}] is not one contiguous K-aligned row of {w0['canonical_id']}"}
        return {"K": K, "x": hits[0], "w": {"canonical_id": w0["canonical_id"], "row": w0["offset"] // K, "field": w0["field"]}}

    def locate_input(self, g: int) -> dict[str, Any]:
        if g >= self.input_gates:
            return {"refused": f"gate {g} is not an Input gate (input gates are [0, {self.input_gates}))"}
        ps, pe = self.prompt_range
        if ps <= g < pe:
            return {"kind": "prompt_token", "t": g - ps}
        i = bisect.bisect_right(self.w_starts, g) - 1
        if i >= 0 and self.w_ranges[i][0] <= g < self.w_ranges[i][1]:
            s, e, cid, field = self.w_ranges[i]
            return {"kind": "weight", "canonical_id": cid, "offset": g - s, "field": field, "words": e - s}
        return {"refused": f"Input gate {g} is in neither the prompt nor any weights collection"}


def build_pinned_program(identity: dict[str, Any], *, expected_digest: str | None, expected_gates: int | None, timings: dict[str, float] | None = None,
                         lazy: bool | None = None, builder: Mapping[str, Any] | None = None):
    """Build the instance's program in THIS process and check it against the pins.  Returns ``(program, C, problems)``; ``program`` is
    ``None`` when the identity cannot be built.  ``identity`` = ``protocol_instance.json#program.identity`` (``case``, ``LP``, ``STEPS``,
    ``function``); ``expected_digest`` = the instance's / population's ``program_digest``; ``expected_gates`` = the pinned gate count.

    ``lazy`` (default: ``$VERIFIER_V3_LAZY_PROGRAM != "0"``): the program is a ``verity_vllm.ir.lazy.LazyProgram`` -- the root body is never
    materialised (A3: ``Program(sp)`` needs 17.9 GB at 1024x511, RecursionError at 2048x1023, OOM at 6500x1599) and ``program_digest`` is
    the sha256 of the streamed descriptor, byte-identical to ``program_digest(encode_program(Program(sp)))``.  A program function the lazy
    layout does not know (anything but ``Serve_v1``) falls back to the full build, recorded in ``timings["program_build_mode"]``."""

    problems: list[str] = []
    t0 = time.perf_counter()
    if lazy is None:
        lazy = os.environ.get(LAZY_PROGRAM_ENV, "1") != "0"
    try:
        from verity.ir import Program, bind, encode_program, program_digest
        from verity_vllm.registry import b1
    except ImportError as e:  # pragma: no cover
        return None, None, [f"verity.ir not importable in the verifier process: {e}"]
    if builder is None and isinstance(identity.get("builder"), Mapping):
        builder = identity["builder"]
    if builder is not None:
        # [E3, integrator D1 2026-09-11] a REGISTERED verity.ir program (E8's CORE ...): the instance pins `program.builder = {module, symbol,
        # kwargs?, program_digest}`; the verifier imports that callable in THIS process, builds, and checks the digest fail-closed
        return build_registered_program(builder, identity=identity, expected_digest=expected_digest, expected_gates=expected_gates, timings=timings)
    try:
        # == scripts/compact_whole_b0.build_config(case, 0, []) without the scripts import: the registry's named config, no overrides.
        # `case` may be a run label rather than a registry name (A9: "TINY-B0-4x8"); then the config is the registry entry whose bound
        # id equals identity.function (the config hash is part of the id), refused when none does.
        lp, steps = int(identity["LP"]), int(identity["STEPS"])
        # [E7 D54/D57 2026-09-13] the identity binds the program: `family` Serve_v1 (every pre-D54 instance; the default when absent;
        # the legacy kBlockN = 64 geometry is implicit) or Serve_v4 with `identity.attention` = {kernel, kBlockM, kBlockN, num_splits}
        # -- the DECLARED kernel geometry E6 pinned, threaded to AttnBlock_v3/AttentionHead_v3 as the static BN.  The verifier trusts
        # the identity, not a derivation from the head dim; an unknown family / a missing or malformed geometry is a refusal.
        fam, geom = identity.get("family"), identity.get("attention")

        def _bind(cfg):
            return b1.serve_program(cfg, lp, steps, family=fam, attention=geom)

        try:
            b1.serve_family(fam)
        except KeyError as e:
            return None, None, [f"program identity names an unknown serving family: {e}"]
        cfg_name = f"{identity.get('case')}_CONFIG"
        if hasattr(b1, cfg_name):
            C = dict(getattr(b1, cfg_name))
            sp = _bind(C)
        else:
            want = identity.get("function")
            match = None
            for name in sorted(n for n in dir(b1) if n.endswith("_CONFIG")):
                cand = dict(getattr(b1, name))
                spc = _bind(cand)
                if spc.id == want:
                    match = (cand, spc)
                    break
            if match is None:
                return None, None, [f"no registry config binds to identity.function {want!r} (case {identity.get('case')!r} is not a registry name)"]
            C, sp = match
        mode = "full"
        if lazy:
            from verity_vllm.ir.lazy import LazyProgram, LazyUnsupported, lazy_program_digest
            try:
                program = LazyProgram(sp)
                mode = "lazy"
            except LazyUnsupported as e:
                if timings is not None:
                    timings["program_lazy_unsupported"] = str(e)[:200]
        if mode == "full":
            program = Program(sp)
    except Exception as e:  # noqa: BLE001
        return None, None, [f"program of identity {identity!r} cannot be built: {type(e).__name__}: {e}"]
    if timings is not None:
        timings["program_build_s"] = time.perf_counter() - t0
        timings["program_build_mode"] = mode
    if identity.get("function") not in (None, sp.id):
        problems.append(f"built program id {sp.id} != instance identity.function {identity.get('function')}")
    if isinstance(expected_gates, bool) or not isinstance(expected_gates, int):
        # A9's instances at shapes never built before pin gates as "UNAVAILABLE" (MODELED): the built count is then the first measurement, recorded
        if timings is not None:
            timings["program_gates_built"] = program.gates
    elif program.gates != int(expected_gates):
        problems.append(f"built program has {program.gates} gates, the pin says {expected_gates}")
    t1 = time.perf_counter()
    d = lazy_program_digest(program) if mode == "lazy" else program_digest(encode_program(program))
    if timings is not None:
        timings["program_digest_s"] = time.perf_counter() - t1
    if expected_digest is not None and d != expected_digest:
        problems.append(f"built program digest {d[:16]} != pinned {str(expected_digest)[:16]}")
    return program, C, problems


def build_registered_program(builder: Mapping[str, Any], *, identity: Mapping[str, Any] | None = None, expected_digest: str | None,
                             expected_gates: int | None, timings: dict[str, float] | None = None):
    """[E3, integrator D1] Build a registered ``verity.ir`` program from the instance pin ``program.builder = {"module": <py module>,
    "symbol": <callable>, "params": {...}?, "program_digest": <hex>}`` (E6's ``protocol_instance.ProgramBuilderPin``; ``kwargs`` accepted as
    an alias of ``params``) and check it fail-closed: the callable (``symbol(**params)``) must return a ``Program`` or ``(Program, digest)``;
    the recomputed ``program_digest(encode_program(program))`` must equal the pin's ``program_digest`` AND ``expected_digest`` (the
    population's) when given; ``expected_gates`` likewise.  When E6's reference ``e2e.protocol_instance.build_pinned_program`` is importable
    the program is built THROUGH it (one code path; its ``InstanceError`` becomes a problem).  Returns ``(program, C, problems)`` with
    ``C = {**identity, "builder": <the pin>, "registered": True}`` (the config the locator / RecordHandle carry).  Any import / build error
    is a problem, never an exception (the verifier refuses by name)."""
    import importlib

    problems: list[str] = []
    t0 = time.perf_counter()
    mod_name, sym = str(builder.get("module", "")), str(builder.get("symbol", ""))
    if not mod_name or not sym:
        return None, None, [f"program.builder pin lacks module / symbol: {dict(builder)!r}"]
    params = dict(builder.get("params") if builder.get("params") is not None else (builder.get("kwargs") or {}))
    declared = None
    try:
        from verity.ir import Program, encode_program, program_digest

        try:
            from e2e.protocol_instance import InstanceError, build_pinned_program as _e6_build  # type: ignore[import-not-found]
        except ImportError:
            _e6_build = None
        if _e6_build is not None and "." not in sym and builder.get("program_digest"):
            pin = {"module": mod_name, "symbol": sym, "params": params or None, "program_digest": str(builder["program_digest"]), "source": builder.get("source")}
            try:
                program, declared = _e6_build(pin, expected_digest=expected_digest)
            except InstanceError as e:
                return None, None, [f"program.builder (protocol_instance.build_pinned_program): {e}"]
        else:
            fn = importlib.import_module(mod_name)
            for part in sym.split("."):
                fn = getattr(fn, part)
            built = fn(**params)
            program, declared = (built if isinstance(built, tuple) else (built, None))
        if not isinstance(program, Program):
            return None, None, [f"program.builder {mod_name}.{sym} returned {type(program).__name__}, not a verity.ir Program"]
    except Exception as e:  # noqa: BLE001
        return None, None, [f"program.builder {mod_name}.{sym} cannot build the program: {type(e).__name__}: {e}"]
    if timings is not None:
        timings["program_build_s"] = time.perf_counter() - t0
        timings["program_build_mode"] = "registered"
    t1 = time.perf_counter()
    d = program_digest(encode_program(program))
    if timings is not None:
        timings["program_digest_s"] = time.perf_counter() - t1
    pinned = builder.get("program_digest")
    if pinned is not None and d != str(pinned):
        problems.append(f"built program digest {d[:16]} != program.builder.program_digest {str(pinned)[:16]}")
    if declared is not None and str(declared) != d:
        problems.append(f"program.builder returned digest {str(declared)[:16]} != recomputed {d[:16]}")
    if expected_digest is not None and d != expected_digest:
        problems.append(f"built program digest {d[:16]} != pinned {str(expected_digest)[:16]}")
    if isinstance(expected_gates, int) and not isinstance(expected_gates, bool) and program.gates != int(expected_gates):
        problems.append(f"built program has {program.gates} gates, the pin says {expected_gates}")
    if identity and identity.get("function") not in (None, program.fn.id):
        problems.append(f"built program id {program.fn.id} != instance identity.function {identity.get('function')}")
    C = {**dict(identity or {}), "builder": dict(builder), "registered": True}
    return program, C, problems


class GenericLocator:
    """[E3, D1] The locator of a registered (non-``Serve_v1``) program: ``input_gates``, ``root_of(g)`` (the root call holding ``g``,
    named by its node name), ``locate_input(g)`` (``{"kind": "typed_input", "gate": g}`` -- the position itself comes from the adapter's
    ``weight_word``), ``calls()`` (the root calls as E2's call order), ``program`` / ``C``.  No weights fields, no prompt, no Gemm members."""

    def __init__(self, program, C: dict[str, Any] | None = None) -> None:
        self.program, self.C = program, dict(C or {})
        self.input_gates = int(program.input_gates)
        roots = sorted(program.circuit.root_calls(), key=lambda c: c.lo)
        self.root_calls = roots
        self.root_names = [(c.node.name or c.name) for c in roots]
        self.root_lo = [c.lo for c in roots]

    def root_of(self, g: int):
        i = bisect.bisect_right(self.root_lo, int(g)) - 1
        if i < 0 or not (self.root_calls[i].lo <= g < self.root_calls[i].hi):
            return None, None
        return self.root_names[i], self.root_calls[i]

    def locate_input(self, g: int) -> dict[str, Any]:
        if not (0 <= int(g) < self.input_gates):
            return {"refused": f"gate {g} is not an Input gate (input_gates = {self.input_gates})"}
        return {"kind": "typed_input", "gate": int(g)}

    def calls(self) -> list[tuple[Any, int, int]]:
        """``(call id, lo, hi)`` of every root call in gate order -- E2 rule item 2's call order for a registered program.  The call id is
        the root call's node NAME (``stage<r>``; the same string :meth:`root_of` returns and E8's ``demo_core.record`` commits as
        ``CallSpan(str(ru.name))`` -- its ACCEPTED live record `CORE-live-exhaustive-02` carries ``call=stage<r>``, integrator 00:29Z), and
        its NODE INDEX as a decimal string only when the node carries no name (``Node.name`` is an annotation, not part of the program
        digest -- a program whose root calls are unnamed is spelled by index).  Serve_v1 uses the structural name too (E2's ``serve_calls``).
        A registered program that wants another spelling gives ``calls`` in its adapter (E8's does, with the same names)."""
        return [(str(c.node.name) if getattr(c.node, "name", None) else str(int(c.node_idx)), int(c.lo), int(c.hi)) for c in self.root_calls]


def generic_regions(loc, layout, request_id: str, *, committed_trees: Mapping[str, Any] | None = None) -> "Regions":
    """[E3, D1 default for a registered program] one RU region per ROOT call, ``step`` = the call's ordinal (E8's CORE: one RU per stage
    call), tree ``layout.tree_id(request_id, step)``.  A program whose RUs are not its root calls passes its own ``regions`` adapter."""

    return Regions([Region(layout.tree_id(request_id, k), int(c.lo), int(c.hi), request_id, k) for k, c in enumerate(loc.root_calls)])


def position_adapter(C: Mapping[str, Any] | None, program, locator) -> dict[str, Any] | None:
    """[E3, D1] The :class:`~e2e.verifier_v3.expected.PositionOracle` adapter of a registered program: ``program.builder.adapter``
    (``"<module>:<symbol>"`` or a symbol of the builder module; default ``position_adapter``) is called as ``fn(program, locator, C)`` and
    must return the adapter dict (``regions``, ``row_gates``, ``calls``, ``prefix``, ``weight_word``, ``trees``; see ``PositionOracle``).
    ``None`` for a ``Serve_v1`` program (the vLLM defaults apply).  For a registered program without ``row_gates`` the adapter REFUSES
    (``NO_ROW_GATES``) when a resolver is built -- boundary positions are never guessed."""
    import importlib

    b = (C or {}).get("builder") if isinstance(C, Mapping) else None
    if not isinstance(b, Mapping):
        return None
    spec = str(b.get("adapter") or "position_adapter")
    mod_name, _, sym = spec.rpartition(":") if ":" in spec else (str(b.get("module", "")), "", spec)
    try:
        fn = importlib.import_module(mod_name)
        for part in sym.split("."):
            fn = getattr(fn, part)
    except (ImportError, AttributeError):
        out = {}
    else:
        out = dict(fn(program, locator, C) or {})
    if isinstance(locator, GenericLocator):
        out.setdefault("calls", lambda loc: calls_from_intervals(loc.calls()))
        out.setdefault("regions", generic_regions)
        if "row_gates" not in out:  # the registered program's step-1 rows must be mapped by its own adapter -- refuse by name, never guess
            def _no_row_gates(_loc):
                raise PositionRefusal("NO_ROW_GATES", f"registered program {getattr(program.fn, 'id', '?')}: program.builder.adapter gives no row_gates "
                                                      f"(committed step-1 leaf id -> (call lo, hi, gates)); boundary positions cannot be derived")
            out["row_gates"] = _no_row_gates
    return out or None


#: PROTOCOL.md vocabulary for a selected VU whose check the implementation cannot establish (W-A9-6): never the legacy "exempt (charged)"
UNSUPPORTED_CLASS = "unsupported (no checker: completion risk)"
LEGACY_EXEMPT_CLASS = "exempt (charged)"


def contract_class(cls: str | None) -> str:
    """The class vocabulary a contract record carries: ``provable`` / ``input unit`` rows keep their class; anything else (the legacy
    ``exempt (charged)`` family label, a kind absent from the table) is :data:`UNSUPPORTED_CLASS`."""

    c = str(cls or "")
    if c.startswith("provable") or c.startswith("input unit"):
        return c
    return UNSUPPORTED_CLASS


def _expectation_builders() -> dict[str, Any]:
    """``e2e.coverage_v1.builders.expectation_builders()`` (lazy: coverage_v1 is imported by the verifier after the locator) -- the ONE
    registry the dispatch reads; a kind in it is a proof-covered unit whatever the legacy expressibility class says."""
    from e2e.coverage_v1.builders import expectation_builders

    return expectation_builders()


def make_locator(program, C: dict[str, Any]):
    """The locator for a program from :func:`build_pinned_program`: ``LazyProgramLocator`` over a lazy program (``root_of`` arithmetic,
    no ``StructuralCircuitMap`` over every root), ``ProgramLocator`` over a full ``Program`` -- same ``locate()`` results."""
    if isinstance(C, Mapping) and (C.get("registered") or isinstance(C.get("builder"), Mapping)):
        return GenericLocator(program, C)  # [E3, D1] a registered program: no Serve_v1 weights fields / prompt / structural names
    try:
        from verity_vllm.ir.lazy import LazyProgram
    except ImportError:  # pragma: no cover
        return ProgramLocator(program, C)
    if isinstance(program, LazyProgram):
        from e2e.lazy_locator import LazyProgramLocator

        loc = LazyProgramLocator(program, C)
        if "C" not in loc.__dict__:  # ProgramLocator keeps the registry config for the expectation builders (RecordHandle.config); LZ's subclass skips __init__
            loc.C = C
        return loc
    return ProgramLocator(program, C)


def locate_selected(loc: ProgramLocator, rows: list[dict[str, Any]], *, klass: dict[str, str], ru_request: dict[int, str], n_v: int) -> list[dict[str, Any]]:
    """The verifier's located unit for every selected row (``chronology/selected.json#selected``: ``vu_index``, ``ru_index``, ``kind``,
    ``gate_interval``), in the shape of a ``mapping.units.jsonl`` row.  ``klass`` = the expressibility table's class per ``fn|form`` kind;
    the table's class is kept verbatim in ``expressibility_class`` and ``class`` carries the contract vocabulary (:func:`contract_class`;
    W-A9-6) -- the table itself (pinned in the partition digest) is not rewritten.  A kind with a registered expectation-capable builder
    (``coverage_v1.builders.expectation_builders()``: CK-ELEM's four elementwise kinds) is ``provable (BUILDERS: <family>)`` with
    ``locate = {builder, gate, family}``; ``n_v`` = VUs per RU (``base_vu`` = the VU's index within its RU)."""

    builders = _expectation_builders()
    # [E3 D25, 2026-09-12 00:30Z] dispatch by CAPABILITY, not by locator class name: a Serve_v1 locator (`ProgramLocator` / `LazyProgramLocator`)
    # has `locate_gemm_coordinate` and its path below is byte-identical to before (E9's honest TINY gate).  A registered program's
    # `GenericLocator` has no GEMM coordinates: there `provable` (E9's `steps.registered_kind_classes`: "the ONE builder registry can build it")
    # names a BUILDER unit -- located as `{builder, gate, family}`, the shape `RecordPartition.locate` -> `dispatch_expectation` consumes and
    # E8's runner produced -- and a `provable` kind with no builder is refused BY NAME in `locate`, never an AttributeError.
    gemm = getattr(loc, "locate_gemm_coordinate", None)
    units: list[dict[str, Any]] = []
    for row in rows:
        kind = f"{row['kind']['fn']}|{row['kind']['form']}"
        cls = klass.get(kind)
        ru = int(row["ru_index"])
        lo, hi = row["gate_interval"]
        u: dict[str, Any] = {"vu_index": int(row["vu_index"]), "ru": ru, "request_id": ru_request[ru], "base_vu": int(row["vu_index"]) - ru * n_v,
                             "kind": kind, "class": cls, "gate_interval": [int(lo), int(hi)]}
        if cls is None:
            u["note"] = "kind absent from the expressibility table: no checker"
        u["expressibility_class"] = cls  # the pinned table's row (legacy vocabulary allowed here), for the integrator
        u["class"] = cls = contract_class(cls)
        if kind in builders and (not cls.startswith("provable") or gemm is None):
            # [CK-ELEM dispatch, re-targeted 22:20Z] the builder resolves the operand/output words in the committed rows (prover) and
            # derives the expectation (verifier) from the VU's one gate in the pinned program.  (Serve_v1: only when the pinned table does
            # not call the kind `provable` -- its GEMM kinds keep the coordinate path below; a registered program: every builder kind.)
            b = builders[kind]
            u["class"] = cls = f"provable (BUILDERS: {b.family})"
            u["locate"] = {"builder": kind, "gate": int(lo), "family": str(b.family)}
            units.append(u)
            continue
        if cls.startswith("provable"):
            if gemm is None:
                u["locate"] = {"refused": f"{kind}: class {cls!r} but the registered program has no GEMM coordinates and no builder derives an "
                                          f"expectation for the kind (registered: {sorted(builders)})"}
                units.append(u)
                continue
            u["locate"] = gemm(int(lo), int(hi))
            u["operands"] = loc.gemm_operands(int(lo), int(hi))  # verifier-only (not a mapping.units.jsonl field): the expectation's x / W
        elif cls.startswith("input unit"):
            u["locate"] = loc.locate_input(int(lo))
        elif kind in ATTENTION_KINDS or _base_kind(kind) in ATTENTION_KINDS:
            # [A5/6 patch] attention-body kinds and the (flagged, PROPOSED) head unit: (t, layer, head) from the program's
            # structural map -- coverage_v1.attn_locator; the located unit carries `unit` = "head" | "interior" and the call chain
            from e2e.coverage_v1.attn_locator import locate_attention

            u["locate"] = locate_attention(loc, int(lo), int(hi))
        else:
            u["locate"] = None
        units.append(u)
    return units


def compare_units(derived: list[dict[str, Any]], prover: list[dict[str, Any]], *, fields: tuple[str, ...] = LOCATE_FIELDS) -> list[str]:
    """Per-unit disagreements between the verifier's derivation and the prover's ``mapping.units.jsonl`` on ``fields`` (and on the VU
    set).  Empty list == the prover's file says exactly what the verifier derives.  The derivation is used either way."""

    d = {u["vu_index"]: u for u in derived}
    p = {u["vu_index"]: u for u in prover}
    out: list[str] = []
    if set(d) != set(p):
        out.append(f"VU sets differ: {len(set(d) - set(p))} derived-only, {len(set(p) - set(d))} prover-only")
    for vu in sorted(set(d) & set(p)):
        du = d[vu]
        builder_unit = isinstance(du.get("locate"), dict) and "builder" in du["locate"]
        for f in fields:
            mine, theirs = du.get(f), p[vu].get(f)
            if f == "class":
                # [W-A9-12] the derived class of a builder unit comes from the ONE registry (`provable (BUILDERS: <family>)`, locate_selected);
                # the prover's file may carry exactly that (a fresh record: its obligations stage calls locate_selected) OR the pinned table's
                # class (a record written before the registry knew the kind: `no-checker` / `exempt (charged)`).  Both agree with the
                # derivation; any other relabel of a unit's class still disagrees.  Non-builder units: the table's class, contract vocabulary
                theirs = contract_class(theirs)
                if builder_unit and theirs in (contract_class(du.get("class")), contract_class(du.get("expressibility_class"))):
                    continue
                mine = contract_class(mine)
            elif f == "locate" and theirs is None and not (contract_class(du.get("expressibility_class")).startswith("provable") or
                                                          contract_class(du.get("expressibility_class")).startswith("input unit")):
                # the prover's file locates nothing for a unit its table called unsupported (legacy `exempt (charged)` / A9 `no-checker`): the
                # verifier now locates it (registered builder / attention-body locator) -- more knowledge on the verifier side is not a prover
                # relabel; for a provable / input unit a missing prover location IS a disagreement
                continue
            if mine != theirs:
                out.append(f"VU {vu}: {f} derived {mine!r} != prover {theirs!r}")
                break
    return out


# =====================================================================================================================================
# [E3, EOD 2026-09-11] The position of EVERY gate -- input | boundary | interior -- and a VU's expected statement as positions
# (brief interfaces 1, 3, 4, 5).  Everything below is derived from the pinned program, the pinned partition (population rows), the
# committed id lists of step 1 (bound by ``ids_digest`` before J) and the pinned ``interior_layout``; the prover's ``mapping.units.jsonl``,
# ``description.json`` and the interior ``index.json`` entries are never read for a position.  Nothing here is vLLM-specific except the
# two adapters named so: :func:`serve_regions` (RU step regions of a ``Serve_v1`` program) and :func:`vllm_row_gates` (capture-v2 ids
# -> the producing call's return gates).  E8's CORE circuit passes its own ``regions`` and ``row_gates``.
# =====================================================================================================================================

import bisect as _bisect
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

ORIGIN_INPUT, ORIGIN_BOUNDARY, ORIGIN_INTERIOR = "input", "boundary", "interior"
#: [E5 `const32-consequence.md` §4, E6 pin `kind=literal`] a zero-arity ``Const<w>[0x..]`` gate that is the output of NO VU of the pinned
#: population: its value is the program text, not a committed word -- ``GatePosition(origin="literal", tree="literal", leaf=<prim id>, value=bits)``
ORIGIN_LITERAL = "literal"
LITERAL_TREE = "literal"
WEIGHTS_TREE = "weights"


class PositionRefusal(ValueError):
    """The verifier cannot name the position of a gate / the statement of a VU from its own sources.  ``code`` names the refusal:
    ``NOT_A_CALL`` (a multi-gate VU interval that is no call of the program), ``NO_REGION`` (a gate in no declared RU region),
    ``INTERIOR_LAYOUT_UNPINNED`` (an interior word with no pinned layout), ``DUPLICATE_BOUNDARY_POSITION`` (two committed rows are
    the producing row of one gate), ``NO_ROWS`` (no partition rows given), ``OUTSIDE_PARTITION`` (a gate no VU produces)."""

    def __init__(self, code: str, msg: str) -> None:
        super().__init__(f"{code}: {msg}")
        self.code = code


@dataclass(frozen=True, slots=True)
class GatePosition:
    """Where one gate's committed word lives: ``origin`` ∈ {input, boundary, interior, literal}; ``(tree, leaf, flat)`` with the leaf's
    ``shape`` (row-major; the reference offset is ``unravel(flat, shape)``) and the word's ``width`` in bits.  A ``literal`` (E5 §4) is no
    committed word: ``tree = "literal"``, ``leaf`` = the primitive's id (``Const32[0x00000001]``), ``value`` = its bits from the program text."""

    gate: int
    origin: str
    tree: str
    leaf: str
    shape: tuple[int, ...]
    flat: int
    width: int
    value: int | None = None

    @property
    def offset(self) -> tuple[int, ...]:
        return tuple(int(i) for i in np.unravel_index(int(self.flat), self.shape)) if len(self.shape) > 1 else (int(self.flat),)

    def expected(self):
        from veritor.interface.expected import ExpectedValue

        return ExpectedValue(self.tree, self.leaf, self.offset)

    def to_json(self) -> dict[str, Any]:
        d = {"gate": self.gate, "origin": self.origin, "tree": self.tree, "leaf": self.leaf, "shape": list(self.shape), "flat": self.flat, "width": self.width}
        if self.value is not None:
            d["kind"], d["value"] = ORIGIN_LITERAL, int(self.value)
        return d


def literal_of(program, g: int) -> GatePosition | None:
    """The literal a zero-arity ``Const<w>[0x..]`` gate carries (E5 `const32-consequence.md` §4: ``(kind="literal", width=w, value=bits)``
    from ``program.circuit.gate(g).prim``), ``None`` for any other gate.  The caller decides whether the gate is a VU output of the pinned
    population (then it is a committed word, not a literal)."""
    gate = program.circuit.gate(int(g))
    prim = gate.prim
    if getattr(prim, "family", None) != "Const" and not str(getattr(prim, "name", "")).startswith("Const"):
        return None
    if len(tuple(gate.operands)) or len(prim.params):
        return None
    w = gate_width(program, int(g))
    bits = int(prim.evaluate())  # a zero-arity primitive: its value IS the program text (`const(w, bits)`: `lambda: bits`)
    if "[0x" in str(prim.name) and bits != int(str(prim.name).split("[0x", 1)[1].rstrip("]"), 16):
        raise PositionRefusal("LITERAL_MISMATCH", f"gate {g}: {prim.id} evaluates to 0x{bits:x}, not the literal in its name")
    return GatePosition(int(g), ORIGIN_LITERAL, LITERAL_TREE, str(prim.id), (1,), 0, int(w), bits)


def literal_ids_of_program(program) -> dict[str, GatePosition]:
    """[E3 night, for E9's ``LiteralTable.from_program``] Every distinct ``Const<w>[0x..]`` primitive of the pinned program, ``{prim id:
    literal_of(one instance gate)}``, from the PROGRAM TEXT: walk each distinct function DEFINITION once (root function ids -- on a lazy
    ``Serve_v1`` layout: embed, final, pre0/pre1, post, and ``attn_spec(T)`` for every T -- then their bodies recursively, one member per
    batch / scan since every member is the same function), and hand ONE absolute gate of each distinct Const primitive to :func:`literal_of`
    (the id, width and value spelling stay that one derivation).  Exact: a Const gate of the circuit is an instance of a Const node of some
    definition on the walk, and every definition reachable from a root is walked.  A gate-by-gate scan of TINY is 242,675,685 ``literal_of``
    calls (E10 08:10Z: the driver sat in it); this is O(distinct definitions) -- milliseconds on TINY, seconds at 6500x1600."""
    from verity.ir.defs import PrimitiveDefinition

    circ = program.circuit
    lay = getattr(program, "layout", None)
    roots: list[tuple[Any, int]] = []  # (fn, absolute lo) of one instance per distinct root function id
    if lay is not None and hasattr(lay, "index") and hasattr(lay, "fn") and hasattr(lay, "offset"):
        seen_root: set[str] = set()
        roles: list[tuple] = [("embed", 0), ("final", int(lay.LP) - 1)]
        for L_ in range(int(lay.NL)):
            roles += [("pre", 0, L_), ("post", 0, L_)]
        roles += [("attn", t, 0) for t in range(int(lay.P))]
        for role in roles:
            k = lay.index(role)
            fn = lay.fn(k)
            if str(fn.id) in seen_root:
                continue
            seen_root.add(str(fn.id))
            roots.append((fn, int(circ.root_scope.offset) + int(lay.offset(k))))
    else:
        seen_root = set()
        for c in circ.root_calls():
            if str(c.fn.id) in seen_root:
                continue
            seen_root.add(str(c.fn.id))
            roots.append((c.fn, int(c.lo)))
    out: dict[str, GatePosition] = {}
    seen_fn: set[str] = set()

    def walk(fn, lo: int) -> None:
        if isinstance(fn, PrimitiveDefinition):
            if str(fn.id) in out:
                return
            lp = literal_of(program, lo)
            if lp is not None:
                out[str(fn.id)] = lp
            return
        if str(fn.id) in seen_fn:
            return
        seen_fn.add(str(fn.id))
        body = fn.body
        for j, node in enumerate(body.nodes):
            walk(node.fn, lo + int(body.offsets[j]))  # member 0 of a batch / iteration 0 of a scan: the same function as every member

    for fn, lo in roots:
        walk(fn, lo)
    return out


# --- the program: the call at an interval, a VU's output and operand gates -------------------------------------------------------------


def call_at(circuit, lo: int, hi: int):
    """The ``FunctionCall`` whose interval is exactly ``[lo, hi)`` (descending the canonical layout by ``node_at`` / division -- no
    enumeration of siblings), or ``None`` when no call has that interval.  Works on a full ``Program`` and on a ``LazyProgram``
    (``LazyRootBody`` implements ``node_at`` / ``nodes`` / ``offsets``)."""
    from verity.ir.defs import PrimitiveDefinition
    from verity.ir.layout import FunctionCall, Scope

    if not (0 <= lo < hi <= circuit.size):
        return None
    scope = circuit.root_scope
    while True:
        body = scope.spec.body
        k = body.node_at(lo - scope.offset)
        node = body.nodes[k]
        node_off = scope.offset + body.offsets[k]
        m = 0 if node.form in ("primitive", "call") else (lo - node_off) // node.fn.gates
        c_lo = node_off + m * node.fn.gates
        if (c_lo, c_lo + node.fn.gates) == (lo, hi):
            return FunctionCall(circuit, scope, k, m)
        if isinstance(node.fn, PrimitiveDefinition) or c_lo + node.fn.gates < hi or lo < c_lo:
            return None
        scope = Scope(node.fn, c_lo, scope, k, m)


@dataclass(frozen=True, slots=True)
class UnitSpan:
    """A multi-gate VU interval as the program sees it.  ``call`` when ``[lo, hi)`` is exactly one call.  Otherwise a **regrouped unit**
    (E5's ruling / E6's ``population_v3.regroup_rows``: ``TokenSelect_v1|whole-scan`` = the 49,151 ``ArgmaxStep`` iterations of one scan
    as ONE VU): ``members`` = the ``m_hi - m_lo`` consecutive member calls of ONE combinator node (``scan`` / ``batch`` / ``map``) of
    ``enclosing``'s body that tile ``[lo, hi)`` exactly, ``enclosing`` = the composite call whose body holds them."""

    lo: int
    hi: int
    call: Any = None
    enclosing: Any = None
    scope: Any = None
    node_idx: int = -1
    m_lo: int = 0
    m_hi: int = 0

    @property
    def regrouped(self) -> bool:
        return self.call is None

    def members(self) -> Iterator[Any]:
        from verity.ir.layout import FunctionCall

        for m in range(self.m_lo, self.m_hi):
            yield FunctionCall(self.enclosing.circuit, self.scope, self.node_idx, m)


def unit_span(circuit, lo: int, hi: int) -> UnitSpan | None:
    """``[lo, hi)`` as one call (``UnitSpan.call``) or as a run of ``>= 2`` consecutive members of one combinator node inside one
    composite call (``UnitSpan.regrouped``); ``None`` when it is neither (a run of ROOT calls, a partial member, a primitive's interior).
    Descends the canonical layout by ``node_at`` / division -- no enumeration of siblings."""
    from verity.ir.defs import PrimitiveDefinition
    from verity.ir.layout import FunctionCall, Scope

    if not (0 <= lo < hi <= circuit.size):
        return None
    scope = circuit.root_scope
    while True:
        body = scope.spec.body
        k = body.node_at(lo - scope.offset)
        node = body.nodes[k]
        node_off = scope.offset + body.offsets[k]
        g = int(node.fn.gates)
        single = node.form in ("primitive", "call")
        m_lo = 0 if single else (lo - node_off) // g
        m_hi = 1 if single else -(-(hi - node_off) // g)
        c_lo, c_hi = node_off + m_lo * g, node_off + m_hi * g
        if (c_lo, c_hi) == (lo, hi):
            if m_hi - m_lo == 1:
                return UnitSpan(lo, hi, call=FunctionCall(circuit, scope, k, m_lo))
            if scope.parent is None:
                return None  # a run of root-level members is no unit of any call
            enclosing = FunctionCall(circuit, scope.parent, scope.node_idx, scope.member)
            return UnitSpan(lo, hi, enclosing=enclosing, scope=scope, node_idx=k, m_lo=m_lo, m_hi=m_hi)
        if isinstance(node.fn, PrimitiveDefinition) or c_hi < hi or lo < c_lo or m_hi - m_lo != 1:
            return None
        scope = Scope(node.fn, c_lo, scope, k, m_lo)


def output_gates_of(program, lo: int, hi: int) -> tuple[int, ...]:
    """The output gates of the VU ``[lo, hi)`` (PROTOCOL step 3: "each output gate of V"), ascending: its one gate (a primitive /
    batch-of-primitive unit); the distinct declared returns of its call that lie inside the interval; or, for a **regrouped unit** (a run
    of member calls inside one composite call -- E5's ``TokenSelect_v1|whole-scan`` over the ``ArgmaxStep`` scan), **the declared returns
    of the ENCLOSING call that lie inside the interval** -- the values the unit exposes (the scan's final ``ni`` = the ``TokenSelect``
    return = the sampler's token word; E5's ``SelectStatement`` binds exactly that word).  The member calls' own returns that are not
    returns of the enclosing call (every per-step ``best`` / ``best_idx`` / ``i+1``) are NOT output gates of the unit -- they are internal to
    it and are required interior words of no VU.  A multi-gate interval that is neither is refused (``NOT_A_CALL``)."""

    if hi - lo == 1:
        return (int(lo),)
    u = unit_span(program.circuit, lo, hi)
    if u is None:
        raise PositionRefusal("NOT_A_CALL", f"[{lo},{hi}) is no call of the pinned program (nor a run of member calls inside one call)")
    c = u.call if not u.regrouped else u.enclosing
    return tuple(sorted({int(g) for g in c.returns if lo <= g < hi}))


def operand_gates_of(program, lo: int, hi: int) -> tuple[int, ...]:
    """The operand gates of the VU ``[lo, hi)`` in signature order, repeats kept: ``Gate.operands`` of its one gate; ``call.args`` of its
    call; for a **regrouped unit**, the args of its member calls, in member order, that lie OUTSIDE ``[lo, hi)`` (what the unit reads from
    the rest of the program: for the whole scan ``logits[0]``, the two init literals ``Const32[0]`` / ``Const32[1]``, then ``logits[1..V)``;
    a member's args produced by an earlier member -- the carry -- are internal)."""

    if hi - lo == 1:
        return tuple(int(g) for g in program.circuit.gate(lo).operands)
    u = unit_span(program.circuit, lo, hi)
    if u is None:
        raise PositionRefusal("NOT_A_CALL", f"[{lo},{hi}) is no call of the pinned program (nor a run of member calls inside one call)")
    if not u.regrouped:
        return tuple(int(g) for g in u.call.args)
    return tuple(int(a) for m in u.members() for a in m.args if not (lo <= int(a) < hi))


def gate_width(program, g: int) -> int:
    """Declared width of gate ``g``'s value (``Gate.prim.ret``)."""

    ret = program.circuit.gate(g).prim.ret
    w = ret.homogeneous_width() if hasattr(ret, "homogeneous_width") else getattr(ret, "w", None)
    if w is None:
        raise PositionRefusal("NO_WIDTH", f"gate {g}: return type {ret!r} has no single width")
    return int(w)


# --- regions: the RU (step) trees a program's gates commit into ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Region:
    """One retained RU region: the interior tree it commits into and its gate range ``[lo, hi)``.  ``ru_tree`` is the step-1 RU tree
    of the same step when the program has one (``ru/<req>/<step>``); ``step`` is diagnostic."""

    tree: str
    lo: int
    hi: int
    request_id: str = ""
    step: int | None = None
    ru_tree: str | None = None


class Regions:
    """An ordered, non-overlapping list of :class:`Region`; ``at(g)`` by bisection."""

    def __init__(self, regions: Sequence[Region]) -> None:
        rs = sorted(regions, key=lambda r: r.lo)
        for a, b in zip(rs, rs[1:]):
            if a.hi > b.lo:
                raise ValueError(f"regions overlap: {a.tree} [{a.lo},{a.hi}) and {b.tree} [{b.lo},{b.hi})")
        self.regions = tuple(rs)
        self._lo = [r.lo for r in rs]

    def __iter__(self) -> Iterator[Region]:
        return iter(self.regions)

    def __len__(self) -> int:
        return len(self.regions)

    def at(self, g: int) -> Region | None:
        i = _bisect.bisect_right(self._lo, g) - 1
        if i < 0 or not (self.regions[i].lo <= g < self.regions[i].hi):
            return None
        return self.regions[i]

    def by_tree(self, tree: str) -> Region | None:
        return next((r for r in self.regions if r.tree == tree), None)


def serve_regions(loc, layout, request_id: str, *, committed_trees: Mapping[str, Any] | None = None) -> Regions:
    """[vLLM adapter] The RU step regions of a ``Serve_v1`` program from the program alone: step 0 = the prefill root calls
    ``[input_gates, lo(t=LP/embed))``; step s >= 1 = ``[lo(t=LP+s-1/embed), lo(t=LP+s/embed))`` (the last to ``gates``).  ``layout`` is
    the pinned :class:`~veritor.interface.expected.InteriorLayout` (tree id spelling).  With ``committed_trees`` (``commitments.json#trees``)
    each region's positions are cross-checked against the committed RU tree's ``positions`` -- a disagreement is a refusal."""

    named = loc.cmap.named
    LP, STEPS = int(loc.cmap.LP), int(loc.cmap.STEPS)
    gates = int(loc.program.gates)
    starts = [int(loc.input_gates)] + [int(named[f"t={t}/embed"].lo) for t in range(LP, LP + STEPS)] + [gates]
    out: list[Region] = []
    for step in range(STEPS + 1):
        positions = list(range(LP)) if step == 0 else [LP + step - 1]
        ru_tree = f"ru/{request_id}/{step:06d}"
        if committed_trees is not None:
            c = committed_trees.get(ru_tree)
            if c is not None and c.get("positions") is not None and [int(p) for p in c["positions"]] != positions:
                raise PositionRefusal("REGION_POSITIONS", f"{ru_tree}: committed positions {c['positions']} != the program's step {step} positions {positions}")
        out.append(Region(layout.tree_id(request_id, step), starts[step], starts[step + 1], request_id, step, ru_tree))
    return Regions(out)


# --- partition rows: the VUs (kind, gate interval) of the pinned population ---------------------------------------------------------


def _rows_from_vus(vus: Sequence[Mapping[str, Any]], kind_table: Any = None) -> tuple[list[list[int]], list[tuple[str, int]]]:
    """Rows ``[first_vu, count, ru, kind, lo, stride, unit_size]`` from a VU list (``{vu_index?, ru_index?, kind, gate_interval | gate}``),
    coalescing runs of one kind / one RU / one unit size at a constant stride; kind ids in first-seen order (``out_bits`` from a kind table's
    ``declared_bits`` / ``out_bits`` when given, else 0 -- widths always come from the program)."""
    bits: dict[str, int] = {}
    if isinstance(kind_table, list):
        for e in kind_table:
            if isinstance(e, Mapping):
                name = e.get("kind") or (f"{e.get('fn')}|{e.get('form')}" if e.get("fn") else None)
                if name:
                    bits[str(name)] = int(e.get("declared_bits", e.get("out_bits", 0)) or 0)
    kid: dict[str, int] = {}
    kinds: list[tuple[str, int]] = []
    rows: list[list[int]] = []
    for i, v in enumerate(vus):
        vu = int(v.get("vu_index", i))
        gi = v.get("gate_interval")
        lo, hi = (int(gi[0]), int(gi[1])) if gi is not None else (int(v["gate"]), int(v["gate"]) + 1)
        k = str(v["kind"])
        if k not in kid:
            kid[k] = len(kinds)
            kinds.append((k, bits.get(k, 0)))
        ru = int(v.get("ru_index", v.get("ru", 0)))
        size = hi - lo
        if rows:
            r = rows[-1]
            if r[2] == ru and r[3] == kid[k] and r[6] == size and r[0] + r[1] == vu:
                if r[1] == 1 and lo >= r[4] + size:
                    r[5] = lo - r[4]
                    r[1] = 2
                    continue
                if r[1] > 1 and lo == r[4] + r[1] * r[5]:
                    r[1] += 1
                    continue
        rows.append([vu, 1, ru, kid[k], lo, size, size])
    return rows, kinds


class PopulationRows:
    """The pinned partition as rows ``[first_vu, count, ru, kind, lo, stride, unit_size]`` (``population/v2`` / ``a9/population-v5``) of
    ONE program instance, sorted by ``lo``; ``kinds[kid]`` -> ``(name, out_bits)``.  Accepts the JSON document, a ``CompactPopulation``,
    a ``partition_sample.CompactView`` (``.pop``) or ``ReplicatedView`` (``.base``)."""

    def __init__(self, population: Any) -> None:
        doc = population
        while not isinstance(doc, Mapping) and not (hasattr(doc, "rows") and hasattr(doc, "kinds")):
            nxt = getattr(doc, "base", None) or getattr(doc, "pop", None)  # partition_sample.ReplicatedView / CompactView
            if nxt is None or nxt is doc:
                break
            doc = nxt
        if hasattr(doc, "rows") and hasattr(doc, "kinds"):
            rows = list(doc.rows)
            kinds = [(k.name, int(k.out_bits)) for k in doc.kinds]
            self.vu_count = int(getattr(doc, "vu_count", sum(r[1] for r in rows)))
        elif isinstance(doc, Mapping) and "rows" not in doc and isinstance(doc.get("vus"), list):
            # a generic partition document listing its VUs (E8's `demo-core/stage-ru-gate-vu/v1`: `vus[] = {vu_index, ru_index, kind, gate_interval}`)
            # -> rows by coalescing runs of one kind with a constant stride; widths come from the program, so out_bits = the declared bits or 0
            rows, kinds = _rows_from_vus(doc["vus"], doc.get("kinds"))
            self.vu_count = len(doc["vus"])
        elif isinstance(doc, Mapping):
            if doc.get("row_fields") not in (None, ["first_vu", "count", "ru", "kind", "lo", "stride", "unit_size"]):
                raise PositionRefusal("NO_ROWS", f"population row_fields {doc.get('row_fields')!r} unknown")
            rows = list(doc["rows"])
            entries = doc["kinds"]["entries"] if isinstance(doc["kinds"], Mapping) else doc["kinds"]
            kinds = [(f"{e['fn']}|{e['form']}", int(e["out_bits"])) for e in sorted(entries, key=lambda e: int(e["id"]))]
            self.vu_count = int(doc.get("vu_count", sum(r[1] for r in rows)))
        else:
            raise PositionRefusal("NO_ROWS", f"not a population document / view: {type(population).__name__}")
        if not rows:
            raise PositionRefusal("NO_ROWS", "the population has no rows")
        self.rows = sorted(([int(x) for x in r] for r in rows), key=lambda r: r[4])
        self.kinds = kinds
        self._lo = [r[4] for r in self.rows]
        by_first = sorted(range(len(self.rows)), key=lambda i: self.rows[i][0])
        self._first_sorted = [self.rows[i][0] for i in by_first]
        self._first_rows = by_first
        for a, b in zip(self.rows, self.rows[1:]):
            if a[1] > 0 and a[4] + (a[1] - 1) * a[5] + a[6] > b[4]:
                raise PositionRefusal("NO_ROWS", f"population rows overlap in gate order: {a} and {b}")

    def kind_name(self, kid: int) -> str:
        return self.kinds[kid][0]

    def kind_of_row(self, row: Sequence[int]) -> str:
        return self.kinds[int(row[3])][0]

    def row_of_gate(self, g: int) -> tuple[list[int], int] | None:
        """``(row, member j)`` of the VU whose gate interval contains ``g``, or None (an Input gate / a gate of no VU)."""
        i = _bisect.bisect_right(self._lo, g) - 1
        if i < 0:
            return None
        r = self.rows[i]
        if r[1] == 0:
            return None
        j = (g - r[4]) // r[5] if r[5] > 0 else 0
        if j >= r[1]:
            j = r[1] - 1
        lo = r[4] + j * r[5]
        if lo <= g < lo + r[6]:
            return r, j
        return None

    def row_of_vu(self, vu_in_instance: int) -> tuple[list[int], int]:
        """``(row, member j)`` of VU ``vu_in_instance`` (instance-relative index; rows sorted by ``lo`` need not be sorted by ``first_vu``)."""
        i = _bisect.bisect_right(self._first_sorted, int(vu_in_instance)) - 1
        if i < 0:
            raise PositionRefusal("OUTSIDE_PARTITION", f"VU {vu_in_instance} is covered by no row")
        r = self.rows[self._first_rows[i]]
        j = int(vu_in_instance) - r[0]
        if not 0 <= j < r[1]:
            raise PositionRefusal("OUTSIDE_PARTITION", f"VU {vu_in_instance} is covered by no row")
        return r, j

    def interval_of_vu(self, vu_in_instance: int) -> tuple[str, int, int]:
        """``(kind, lo, hi)`` of VU ``vu_in_instance``."""
        r, j = self.row_of_vu(vu_in_instance)
        lo = r[4] + j * r[5]
        return self.kinds[r[3]][0], lo, lo + r[6]

    def rows_in(self, lo: int, hi: int) -> Iterator[list[int]]:
        """The members of every row that lie within ``[lo, hi)``, as rows of the same shape ``[first_vu, count, ru, kind, lo, stride, unit_size]``
        **clipped to the bound** (a batch row of the legacy population spans several root calls -- e.g. one ``|batch`` row over the 4 per-token
        ``embed`` calls -- so a call's words are the row's members inside the call).  A single MEMBER (one VU) straddling the bound is refused:
        ``REGION_VU_STRADDLE`` (a VU belongs to one call / one RU region)."""
        i = max(0, _bisect.bisect_left(self._lo, lo) - 1)
        # [E1 SCALE 2026-09-13] index from i instead of `self.rows[i:]`: the slice copied the tail of the row list (11.7 M rows on 128x64) on
        # EVERY call -> 0.5 s per `call_layout` -> the required-set derivation / step 4 took hours instead of seconds (same rows, same order).
        rows = self.rows
        for k in range(i, len(rows)):
            r = rows[k]
            if r[1] == 0:
                continue
            r_lo, r_hi = r[4], r[4] + (r[1] - 1) * r[5] + r[6]
            if r_hi <= lo:
                continue
            if r_lo >= hi:
                break
            if r_lo >= lo and r_hi <= hi:
                yield r
                continue
            stride, size = int(r[5]), int(r[6])
            j_lo = 0 if r_lo >= lo else (lo - r_lo + stride - 1) // stride  # first member starting at or after lo
            j_hi = int(r[1]) if r_hi <= hi else max(j_lo, (hi - r_lo - size) // stride + 1)  # members ending at or before hi
            if j_lo > 0 and r_lo + (j_lo - 1) * stride + size > lo:
                raise PositionRefusal("REGION_VU_STRADDLE", f"VU {r[0] + j_lo - 1} of row {r} straddles the bound {lo} of [{lo},{hi})")
            if j_hi < int(r[1]) and r_lo + j_hi * stride < hi:
                raise PositionRefusal("REGION_VU_STRADDLE", f"VU {r[0] + j_hi} of row {r} straddles the bound {hi} of [{lo},{hi})")
            if j_hi > j_lo:
                yield [int(r[0]) + j_lo, j_hi - j_lo, r[2], r[3], r_lo + j_lo * stride, stride, size]


# --- boundary index: committed operator rows -> original positions of their gates -------------------------------------------------------


def vllm_row_gates(loc) -> Callable[[str], tuple[int, int, tuple[int, ...]] | None]:
    """[vLLM adapter] ``canonical id -> (producing call lo, hi, return gates in row order)`` through the program's own identity table
    (``verity_vllm.correspondence.core.CircuitMap.locate`` over the locator's ``cmap.named``: a ``StructuralCircuitMap`` for a full program,
    ``_LazyNamed`` arithmetic for a lazy one -- no walk over every root call).  ``None`` for an id that is not a capture-v2 value id or
    has no counterpart in the program (``NotInProgram``: production's per-position final ops at unsampled prefill positions)."""
    from verity_vllm.correspondence.core import CaptureIdentity, CircuitMap, NotInProgram

    from verity_vllm.correspondence.core import _LAYER_OPS

    cm = CircuitMap.__new__(CircuitMap)
    cm.program, cm.circuit, cm.names, cm.named = loc.program, loc.program.circuit, {}, loc.cmap.named
    named = cm.named
    # [E3 night, item 3 -- long context] ROW TEMPLATES.  `CircuitMap.locate` descends the lazy tree per id and materialises the producing
    # call's returns per word (0.96 ms/id at 128x64 on the laptop; 2.9 M ids at 6500x1600 = 47 min, 2.6 G words).  A verity.ir function
    # has ONE body definition: for two root calls of the SAME function id, the producing child's [lo, hi) and its component's return gates
    # RELATIVE to the root's `lo` are identical.  Memoise, per (root function id, op, output), the child's relative span and the RELATIVE
    # RUNS of its gates (start, count, stride, flat0) -- one `cm.locate` per template, O(#runs) per id afterwards; `locate` stays the one
    # derivation (the first id of every template goes through it, the root name comes from correspondence's own `_LAYER_OPS` table).
    # Exact by construction; asserted equal to the direct `locate` on every TINY id (tests/contract/e3_positions_test.py).
    templates: dict[tuple[str, str, int], tuple[int, int, tuple[tuple[int, int, int, int], ...], int] | None] = {}
    # [E3 night, item 3 -- the QUADRATIC term, by name] the attention root call at position t has a KV argument of t+1 parts: E2's lazy root
    # body (`verity_vllm.ir.lazy.ServeLayout.arg_refs`, `kv_part`) rebuilds that Concat every time the node is MATERIALISED (`named[...]` ->
    # `root_call` -> `FunctionCall.__init__` -> `body.nodes[k]`), so walking every attention row costs sum_t (t+1) x NL x 2 -- 12.3 M
    # `kv_part` calls at 512x128 (MEASURED, cProfile), ~2 G at 6500x1600.  The attention output row (`self_attn.attn.attn`, root call itself,
    # component 0) needs only the root's activation SCOPE (its specialization + gate offset), both ARITHMETIC in the lazy layout
    # (`layout.fn(k)`, `layout.offset(k)`): build the `Scope` directly -- `Scope.child(k)` computes the very same (spec, offset, parent, k, 0)
    # -- and resolve the body's returns in it, never touching the node's arguments.  Lazy programs only; a full program falls through.
    lay = getattr(loc, "layout", None)
    lazy_named = lay is not None and hasattr(named, "_index")
    attn_fn: dict[int, Any] = {}
    attn_runs: dict[tuple[str, int], tuple[tuple[tuple[int, int, int, int], ...], int]] = {}

    def attn_row(ident, rn: str):
        """(lo, hi, gates) of the attention output row at (t, L) without materialising the root node; None when not arithmetic."""
        from verity.ir.layout import Scope, resolve

        k = named._index(rn)
        if k is None:
            return None
        T = int(ident.t) + 1
        fn = attn_fn.get(T)
        if fn is None:
            fn = attn_fn[T] = lay.fn(k)
        root_scope = loc.program.circuit.root_scope
        off = int(root_scope.offset) + int(lay.offset(k))
        sc = Scope(fn, off, root_scope, k, 0)
        key = (str(fn.id), int(ident.output))
        tm = attn_runs.get(key)
        if tm is None:
            rel = _RETURNS_REL.get(key[0])
            if rel is None:
                rel = _RETURNS_REL[key[0]] = tuple(int(resolve(sc, r)) - off for r in sc.spec.body.ret)
            rt = fn.signature()[1]
            if type(rt).__name__ == "Tuple":  # component `ident.output` of a Tuple return (correspondence._returns_component)
                o = 0
                for i, t_ in enumerate(rt.fields):
                    n = t_.leaves
                    if i == int(ident.output):
                        rel = rel[o:o + n]
                        break
                    o += n
                else:
                    return None
            tm = attn_runs[key] = (tuple(_runs_of(rel)), len(rel))
        return off, off + int(fn.gates), RunGates(off, tm[0], tm[1])

    def root_name(ident) -> str | None:
        if ident.layer == "embed":
            return f"t={ident.t}/embed"
        if ident.layer in ("norm", "lm_head", "sampler"):
            return f"t={ident.t}/final"
        try:
            L_ = int(ident.layer)
            role = _LAYER_OPS[(ident.op[len(f"layers.{L_}."):], ident.output)][0]
        except (ValueError, KeyError):
            return None
        return f"t={ident.t}/layers.{L_}/{role}"

    def row_gates(cid: str):
        try:
            ident = CaptureIdentity.parse(cid)
        except ValueError:
            return None
        rn = root_name(ident)
        if lazy_named and rn is not None and rn.endswith("/attn"):
            try:
                sel = _LAYER_OPS[(ident.op[len(f"layers.{int(ident.layer)}."):], ident.output)][1]
            except (ValueError, KeyError):
                sel = "?"
            if sel is None:  # the root call's own output component: arithmetic scope, no node materialisation
                r = attn_row(ident, rn)
                if r is not None:
                    return r
        root = None
        if rn is not None:
            try:
                root = named[rn]
            except KeyError:
                root = None
        if root is None or root.is_primitive:
            try:
                call, gates = cm.locate(ident)
            except (NotInProgram, KeyError, IndexError):
                return None
            return int(call.lo), int(call.hi), tuple(int(g) for g in gates)
        base = int(root.lo)
        key = (str(root.fn.id), ident.op, int(ident.output))
        tm = templates.get(key, False)
        if tm is False:
            try:
                call, gates = cm.locate(ident)
            except (NotInProgram, KeyError, IndexError):
                templates[key] = None
                return None
            rel = [int(g) - base for g in gates]
            templates[key] = tm = (int(call.lo) - base, int(call.hi) - base, tuple(_runs_of(rel)), len(rel))
        if tm is None:
            return None
        rel_lo, rel_hi, runs, n = tm
        return base + rel_lo, base + rel_hi, RunGates(base, runs, n)

    return row_gates


def _runs_of(rel: Sequence[int]) -> Iterator[tuple[int, int, int, int]]:
    """Maximal arithmetic runs (start, count, stride, flat0) over a gate list in row order (stride > 0 within a run)."""
    n = len(rel)
    k0 = 0
    while k0 < n:
        if k0 + 1 >= n:
            yield int(rel[k0]), 1, 1, k0
            k0 += 1
            continue
        d = int(rel[k0 + 1]) - int(rel[k0])
        if d <= 0:
            yield int(rel[k0]), 1, 1, k0
            k0 += 1
            continue
        k1 = k0 + 1
        while k1 + 1 < n and int(rel[k1 + 1]) - int(rel[k1]) == d:
            k1 += 1
        yield int(rel[k0]), k1 - k0 + 1, d, k0
        k0 = k1 + 1


class RunGates(Sequence):
    """The return gates of a committed row as arithmetic RUNS re-based on the producing root call's ``lo`` -- a read-only sequence of ints
    (``len``, indexing, iteration, equality with a tuple) plus ``.runs`` = ``[(start, count, stride, flat0), ...]`` in row order for callers
    that can consume runs directly (:class:`BoundaryIndex`)."""

    __slots__ = ("base", "rel_runs", "n")

    def __init__(self, base: int, rel_runs: tuple[tuple[int, int, int, int], ...], n: int) -> None:
        self.base, self.rel_runs, self.n = int(base), rel_runs, int(n)

    @property
    def runs(self) -> list[tuple[int, int, int, int]]:
        return [(self.base + s, c, d, f) for s, c, d, f in self.rel_runs]

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        b = self.base
        for s, c, d, _f in self.rel_runs:
            yield from range(b + s, b + s + c * d, d)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return tuple(self)[i]
        if i < 0:
            i += self.n
        if not 0 <= i < self.n:
            raise IndexError(i)
        for s, c, d, f in self.rel_runs:
            if f <= i < f + c:
                return self.base + s + (i - f) * d
        raise IndexError(i)

    def __eq__(self, other) -> bool:
        return tuple(self) == tuple(other)

    def __hash__(self) -> int:
        return hash(tuple(self))


_RETURNS_REL: dict[str, tuple[int, ...]] = {}
_RETURNS_ORIG = None


def memoize_returns(enable: bool = True) -> None:
    """[E3 night, item 3 -- long context] ``FunctionCall.returns`` resolves every return leaf of a call one by one (``layout.resolve`` per word:
    5 us/word; a boundary row is 576..49,152 words) -- ``row_gates`` cost 2.16 ms/id on TINY and the step-1 index of 6500x1600 (2.9 M ids,
    2.6 G words) would take hours.  A verity.ir function has ONE body definition, so a call's returns RELATIVE to its own ``lo`` are the same
    for every call of that function id: memoise ``returns - lo`` per ``fn.id`` and re-base.  Exact by construction (MEASURED equal on every
    committed row of TINY, 05:58Z); opt-in (the census and the no-record oracle), process-wide."""
    from verity.ir import layout as _layout

    global _RETURNS_ORIG
    FC = _layout.FunctionCall
    if enable and _RETURNS_ORIG is None:
        _RETURNS_ORIG = FC.__dict__["returns"]
        orig_fget = _RETURNS_ORIG.fget

        def returns(self):
            if self.is_primitive:
                return (self.lo,)
            key = str(self.fn.id)
            rel = _RETURNS_REL.get(key)
            lo = int(self.lo)
            if rel is None:
                r = orig_fget(self)
                rel = _RETURNS_REL[key] = tuple(int(g) - lo for g in r)
                return r
            return tuple(lo + d for d in rel)

        FC.returns = property(returns)
    elif not enable and _RETURNS_ORIG is not None:
        FC.returns = _RETURNS_ORIG
        _RETURNS_ORIG = None
        _RETURNS_REL.clear()


class BoundaryIndex:
    """The step-1 positions of one request: every committed operator row (the committed id lists of the RU trees, checked against
    ``ids_digest`` by the caller) mapped through ``row_gates`` to the gates it holds.  A row word whose gate lies INSIDE the producing
    call is that gate's ORIGINAL boundary position; a word whose gate lies outside (a returned argument, a re-committed K/V row, an
    Input gate copied into a row) is a physical COPY -- recorded in ``copies`` for the duplicate check, never returned as a position
    (PROTOCOL: "every shared gate has one value ... original values are used").  Two rows producing one gate is a refusal
    (``DUPLICATE_BOUNDARY_POSITION``)."""

    def __init__(self, committed_ids: Mapping[str, Sequence[str]], row_gates: Callable[[str], tuple[int, int, tuple[int, ...]] | None], *,
                 shape_of: Callable[[str, str, int], tuple[int, ...]] | None = None, width_of: Callable[[str, str], int] | None = None) -> None:
        # [E3 night, item 3] the index is stored as ARITHMETIC RUNS (start, count, stride, row, flat0), not per word: a committed row's inside gates
        # are one arithmetic progression (3,987 of 4,362 TINY rows) or a few (the attention output: 9 runs of 64) -- 16 B per word (41 GB for
        # the 2.6 G boundary words of 6500x1600) becomes ~40 B per run (~150 MB).  Lookups bisect on run starts; interleaved runs (a later start
        # inside an earlier run's span) are handled by the prefix-max of run ends.  MEASURED equal to the per-word index on TINY (06:05Z).
        starts: list[int] = []
        counts: list[int] = []
        strides: list[int] = []
        rows_i: list[int] = []
        flat0: list[int] = []
        self.rows: list[tuple[str, str, tuple[int, ...], int]] = []  # (tree, cid, shape, width)
        self.copies: dict[int, list[tuple[str, str, int]]] = {}
        self.unmapped: list[tuple[str, str]] = []
        self.n_words = 0
        for tree in sorted(committed_ids):
            for cid in committed_ids[tree]:
                rg = row_gates(cid)
                if rg is None:
                    self.unmapped.append((tree, cid))
                    continue
                c_lo, c_hi, gates = rg
                if not gates:
                    continue
                if isinstance(gates, RunGates):
                    # run fast path: the row's gates arrive as arithmetic runs (row templates) -- classify each run against the producing
                    # call's span without materialising words; a run straddling the span is split at the boundary (exact: same runs the
                    # per-word path would find, since a run is one progression and the span is one interval)
                    ridx = len(self.rows)
                    shape = shape_of(tree, cid, len(gates)) if shape_of is not None else (len(gates),)
                    width = width_of(tree, cid) if width_of is not None else 16
                    self.rows.append((tree, cid, tuple(int(d) for d in shape), int(width)))
                    for s, c, d, f in gates.runs:
                        last = s + (c - 1) * d
                        if s >= c_lo and last < c_hi:
                            starts.append(s); counts.append(c); strides.append(d); rows_i.append(ridx); flat0.append(f)
                            self.n_words += c
                        elif last < c_lo or s >= c_hi:
                            for m in range(c):
                                self.copies.setdefault(s + m * d, []).append((tree, cid, f + m))
                        else:
                            m_lo = max(0, -((s - c_lo) // d)) if s < c_lo else 0  # first member >= c_lo
                            m_hi = min(c, (c_hi - 1 - s) // d + 1)  # one past the last member < c_hi
                            for m in range(0, m_lo):
                                self.copies.setdefault(s + m * d, []).append((tree, cid, f + m))
                            if m_hi > m_lo:
                                starts.append(s + m_lo * d); counts.append(m_hi - m_lo); strides.append(d); rows_i.append(ridx); flat0.append(f + m_lo)
                                self.n_words += m_hi - m_lo
                            for m in range(m_hi, c):
                                self.copies.setdefault(s + m * d, []).append((tree, cid, f + m))
                    continue
                g = np.asarray(gates, dtype=np.int64)
                inside = (g >= c_lo) & (g < c_hi)
                ridx = len(self.rows)
                shape = shape_of(tree, cid, len(gates)) if shape_of is not None else (len(gates),)
                width = width_of(tree, cid) if width_of is not None else 16
                self.rows.append((tree, cid, tuple(int(d) for d in shape), int(width)))
                if inside.any():
                    idx = np.nonzero(inside)[0]
                    gi = g[idx]
                    self.n_words += int(len(gi))
                    # maximal arithmetic runs over (gate, flat) in row order (fast path: the whole row is one progression -- 92 % of TINY's rows)
                    k0 = 0
                    n = len(gi)
                    if n >= 2:
                        dd = np.diff(gi)
                        if dd[0] > 0 and bool((dd == dd[0]).all()) and bool((np.diff(idx) == 1).all()):
                            starts.append(int(gi[0])); counts.append(n); strides.append(int(dd[0])); rows_i.append(ridx); flat0.append(int(idx[0]))
                            k0 = n
                    while k0 < n:
                        if k0 + 1 >= n:
                            starts.append(int(gi[k0])); counts.append(1); strides.append(1); rows_i.append(ridx); flat0.append(int(idx[k0]))
                            k0 += 1
                            continue
                        d = int(gi[k0 + 1] - gi[k0])
                        if d <= 0 or idx[k0 + 1] != idx[k0] + 1:
                            starts.append(int(gi[k0])); counts.append(1); strides.append(1); rows_i.append(ridx); flat0.append(int(idx[k0]))
                            k0 += 1
                            continue
                        k1 = k0 + 1
                        while k1 + 1 < n and int(gi[k1 + 1] - gi[k1]) == d and idx[k1 + 1] == idx[k1] + 1:
                            k1 += 1
                        starts.append(int(gi[k0])); counts.append(k1 - k0 + 1); strides.append(d); rows_i.append(ridx); flat0.append(int(idx[k0]))
                        k0 = k1 + 1
                for k in np.nonzero(~inside)[0]:
                    self.copies.setdefault(int(g[k]), []).append((tree, cid, int(k)))
        if starts:
            st = np.asarray(starts, dtype=np.int64)
            order = np.argsort(st, kind="stable")
            self.starts = st[order]
            self.counts = np.asarray(counts, dtype=np.int64)[order]
            self.strides = np.asarray(strides, dtype=np.int64)[order]
            self.row_idx = np.asarray(rows_i, dtype=np.int32)[order]
            self.flat0 = np.asarray(flat0, dtype=np.int64)[order]
            self.ends = self.starts + (self.counts - 1) * self.strides  # last gate of each run
            self.prefix_max_end = np.maximum.accumulate(self.ends)
            self.interleaved = bool((self.ends[:-1] >= self.starts[1:]).any())
            # duplicates: a gate in two runs -- exact check where spans overlap (rare: interleaved runs only), cheap check otherwise
            if self.interleaved:
                seen = None
                for k in np.nonzero(self.ends[:-1] >= self.starts[1:])[0]:
                    a = self._run_gates(k)
                    b = self._run_gates(k + 1)
                    common = np.intersect1d(a, b)
                    if len(common):
                        g0 = int(common[0])
                        ra, rb = self.rows[int(self.row_idx[k])], self.rows[int(self.row_idx[k + 1])]
                        raise PositionRefusal("DUPLICATE_BOUNDARY_POSITION", f"gate {g0} is produced by two committed rows: {ra[1]} and {rb[1]}")
                    seen = k
                _ = seen
        else:
            self.starts = self.counts = self.strides = self.ends = self.prefix_max_end = self.flat0 = np.zeros(0, dtype=np.int64)
            self.row_idx = np.zeros(0, dtype=np.int32)
            self.interleaved = False

    def _run_gates(self, k: int) -> np.ndarray:
        return self.starts[k] + self.strides[k] * np.arange(self.counts[k], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.n_words)

    def _find(self, g: int) -> tuple[int, int] | None:
        """``(run k, member m)`` holding gate ``g``, else None."""
        k = int(np.searchsorted(self.starts, g, side="right")) - 1
        while k >= 0 and int(self.prefix_max_end[k]) >= g:
            off = g - int(self.starts[k])
            st = int(self.strides[k])
            if off >= 0 and off % st == 0 and off // st < int(self.counts[k]):
                return k, off // st
            if not self.interleaved:
                return None
            k -= 1
        return None

    def word(self, g: int) -> GatePosition | None:
        f = self._find(int(g))
        if f is None:
            return None
        k, m = f
        tree, cid, shape, width = self.rows[int(self.row_idx[k])]
        return GatePosition(int(g), ORIGIN_BOUNDARY, tree, cid, shape, int(self.flat0[k]) + m, width)

    def mask(self, gates: np.ndarray) -> np.ndarray:
        gates = np.asarray(gates, dtype=np.int64)
        if not len(self.starts) or not len(gates):
            return np.zeros(len(gates), dtype=bool)
        k = np.searchsorted(self.starts, gates, side="right") - 1
        ok = k >= 0
        kk = np.maximum(k, 0)
        off = gates - self.starts[kk]
        hit = ok & (off >= 0) & (off % self.strides[kk] == 0) & (off // self.strides[kk] < self.counts[kk])
        if self.interleaved:
            # a miss whose gate lies under an earlier run's span: the exact scalar path
            cand = np.nonzero(~hit & ok & (self.prefix_max_end[kk] >= gates))[0]
            for i in cand:
                hit[i] = self._find(int(gates[i])) is not None
        return hit

    def count_below(self, gates: np.ndarray) -> int:
        """How many of ``gates`` (any order) have an original boundary position."""
        return int(np.count_nonzero(self.mask(np.asarray(gates, dtype=np.int64)))) if len(gates) else 0


# --- the resolver ---------------------------------------------------------------------------------------------------------------------


def interior_layout_rule(pin: Mapping[str, Any] | None):
    """The pinned :class:`~veritor.interface.expected.InteriorLayout` **under the rule the pin NAMES** (``interior_layout.rule``; absent =
    ``interior-layout/chunks-4096/v1`` for records written before D13, ``.../v2`` for new pins), with the leaf-id spelling taken from E2's
    layout module (``e2e.interior.layout.leaf_id_rule(pin)``) when it exists in this checkout -- so the verifier and the writer spell one id --
    else the rule as E2 stated it (v2 ``<prefix>call=<c>/w=<w>/chunk=<k>``, v1 ``<prefix>call=<c>/chunk=<k>``).  A pin E2's ``check_pin``
    refuses under its own rule (framing, word order, tree template, digest) is ``InteriorLayoutUnpinned`` here (``V3.interior_layout``)."""
    from veritor.interface.expected import InteriorLayout, InteriorLayoutUnpinned

    leaf_rule = None
    try:  # E2's module; optional (the rule is stated in the interface module either way)
        from e2e.interior import layout as _e2  # type: ignore[import-not-found]

        if isinstance(pin, Mapping) and pin:
            rule_of, check_pin = getattr(_e2, "rule_of", None), getattr(_e2, "check_pin", None)
            if rule_of is not None and check_pin is not None:
                try:
                    framing = str(pin.get("framing", ""))
                    words = int(framing[len("chunks-"):]) if framing.startswith("chunks-") and framing[len("chunks-"):].isdigit() else 4096
                    problems = list(check_pin(pin, chunk_words=words, rule=rule_of(pin)))
                    if pin.get("digest") is None:  # a pin without a digest derives by the rule it NAMES; the digest itself is E9's V3 check
                        problems = [x for x in problems if "interior_layout.digest" not in x]
                except ValueError as e:
                    problems = [str(e)]
                if problems:
                    raise InteriorLayoutUnpinned("V3.interior_layout: pinned interior_layout refused by E2's layout module under the rule it names: " + "; ".join(problems))
            leaf_id_rule = getattr(_e2, "leaf_id_rule", None)
            if leaf_id_rule is not None:
                leaf_rule = leaf_id_rule(pin)
            else:  # an older module: a single spelling
                leaf_rule = getattr(_e2, "leaf_id", None)
    except ImportError:
        pass
    return InteriorLayout.from_pin(pin, leaf_id_rule=leaf_rule)


def serve_calls(loc) -> Callable[[int], tuple[Any, int, int] | None]:
    """[vLLM adapter] ``gate -> (call id, lo, hi)`` of the ROOT call holding the gate (``ProgramLocator.root_of`` / ``LazyProgramLocator.root_of``):
    the call id is the root call's STRUCTURAL NAME (``ServeLayout.name(k)``, e.g. ``t=4/layers.0/pre``) -- what E2's ``CallSpan.call`` /
    ``serve_calls`` spell into the leaf id ``<prefix>call=<name>/chunk=<k>``; call ORDER is still the root-node index (rule item 2)."""

    lay = getattr(loc, "layout", None)
    if lay is not None and hasattr(lay, "node_at") and hasattr(lay, "offset") and hasattr(lay, "name"):
        # [E3 night, item 3 -- the quadratic term, by name] a lazy program: (name, lo, hi) are ARITHMETIC in E2's layout (`node_at`, `offset`,
        # `name`); `root_of` would materialise the root FunctionCall, and an attention node's KV argument is a Concat of t+1 parts
        # (`ServeLayout.arg_refs` / `kv_part`) rebuilt per materialisation -- sum over positions = O(T^2): `inventory` 12.9 s at 512x128,
        # 239 s at 2048x256 (MEASURED on the pod).  Same triple `LazyProgramLocator.root_of` returns (`lay.name(k)`, `root_call(k).lo/hi`
        # = `offset(k)`, `offset(k+1)`), no node touched.
        gates = int(lay.gates)

        def calls(g: int):
            g = int(g)
            if not (0 <= g < gates):
                return None
            k = lay.node_at(g)
            if lay.role(k)[0] == "input":
                return None
            return str(lay.name(k)), int(lay.offset(k)), int(lay.offset(k + 1))

        return calls

    def calls(g: int):
        name, root = loc.root_of(int(g))
        if root is None:
            return None
        return str(name if name is not None else root.node_idx), int(root.lo), int(root.hi)

    return calls


def calls_from_intervals(intervals: Sequence[tuple[Any, int, int]]) -> Callable[[int], tuple[Any, int, int] | None]:
    """[generic] ``gate -> (call id, lo, hi)`` from the caller's ordered, disjoint ``(call_id, lo, hi)`` list (E2 rule item 2, CORE)."""

    iv = sorted(((c, int(lo), int(hi)) for c, lo, hi in intervals), key=lambda x: x[1])
    los = [x[1] for x in iv]
    for a, b in zip(iv, iv[1:]):
        if a[2] > b[1]:
            raise ValueError(f"call intervals overlap: {a} and {b}")

    def calls(g: int):
        i = _bisect.bisect_right(los, int(g)) - 1
        if i < 0 or not (iv[i][1] <= g < iv[i][2]):
            return None
        return iv[i]

    return calls


class GateResolver:
    """Gate -> :class:`GatePosition` and VU -> :class:`~veritor.interface.expected.PositionStatement` for ONE request (program instance),
    from: the pinned ``program`` (+ ``loc``: its locator for ``locate_input`` / ``input_gates``), the partition ``rows``, the step-1
    :class:`BoundaryIndex`, the RU ``regions`` (gate -> interior tree), the layout's ``calls`` (gate -> ``(call id, lo, hi)``), the leaf-id
    ``prefix`` and the pinned ``layout`` (``None`` == unpinned: any interior word refuses by name).

    Resolution rule (E2 ``interior-layout/chunks-4096/v1``, in code): an Input gate -> its weights leaf word / prompt boundary leaf
    (``input``); else its ORIGINAL committed row word of step 1 (``boundary``); else, with ``c`` the call holding the gate and ``G(c)`` the
    output gates of every VU of the partition in ``[c.lo, c.hi)`` that have no boundary position, ascending gate index, with widths
    ``w(g)``: chunk the sequence (a chunk closes at ``chunk_words`` words, at a width change, or at the call's end) -> the word is
    ``(tree = interior/<req>/<step> of the region holding g, leaf = <prefix>call=<c>/chunk=<k>, flat = rank of g in chunk k)``
    (``interior``)."""

    def __init__(self, program, loc, *, rows: PopulationRows, boundary: BoundaryIndex, regions: Regions | None, layout,
                 calls: Callable[[int], tuple[Any, int, int] | None] | None = None, prefix: str = "", request_id: str = "",
                 weight_word: Callable[[int], GatePosition | None] | None = None, prompt_word: Callable[[int], GatePosition | None] | None = None,
                 input_gates: int | None = None) -> None:
        self.program, self.loc, self.rows, self.boundary, self.regions, self.layout = program, loc, rows, boundary, regions, layout
        self.calls = calls if calls is not None else (serve_calls(loc) if loc is not None else None)
        self.prefix, self.request_id = prefix, request_id
        self._weight_word, self._prompt_word = weight_word, prompt_word
        self.input_gates = int(input_gates if input_gates is not None else (loc.input_gates if loc is not None else program.input_gates))
        self._rel: dict[tuple[int, int, int], tuple[int, ...]] = {}  # (kid, stride, unit_size) -> relative output gate offsets
        self._rel_w: dict[tuple[int, int, int], tuple[int, ...]] = {}  # (kid, stride, unit_size) -> widths of those outputs
        self._call_layout: dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]] = {}  # call id -> (gates, chunk, flat, sizes)
        self.timings: dict[str, float] = {}

    # -- per-kind structure ---------------------------------------------------------------------------------------------------------
    def rel_outputs(self, row: Sequence[int]) -> tuple[int, ...]:
        """Output gate offsets relative to a member's ``lo`` for every member of ``row`` (one specialization per kind -> one layout)."""
        key = (int(row[3]), int(row[5]), int(row[6]))
        rel = self._rel.get(key)
        if rel is None:
            lo, size = int(row[4]), int(row[6])
            outs = output_gates_of(self.program, lo, lo + size)
            rel = self._rel[key] = tuple(g - lo for g in outs)
            self._rel_w[key] = tuple(gate_width(self.program, g) for g in outs)
        return rel

    def rel_widths(self, row: Sequence[int]) -> tuple[int, ...]:
        self.rel_outputs(row)
        return self._rel_w[(int(row[3]), int(row[5]), int(row[6]))]

    def output_gates_of_row(self, row: Sequence[int], j_lo: int = 0, j_hi: int | None = None) -> np.ndarray:
        """Output gates of members ``j_lo .. j_hi`` of ``row``, ascending."""
        rel = np.asarray(self.rel_outputs(row), dtype=np.int64)
        n = int(row[1]) if j_hi is None else int(j_hi)
        if n <= j_lo or not len(rel):
            return np.zeros(0, dtype=np.int64)
        js = np.arange(j_lo, n, dtype=np.int64) * int(row[5]) + int(row[4])
        return (js[:, None] + rel[None, :]).reshape(-1)

    def output_widths_of_row(self, row: Sequence[int], j_lo: int = 0, j_hi: int | None = None) -> np.ndarray:
        w = np.asarray(self.rel_widths(row), dtype=np.int64)
        n = int(row[1]) if j_hi is None else int(j_hi)
        if n <= j_lo or not len(w):
            return np.zeros(0, dtype=np.int64)
        return np.tile(w, n - j_lo)

    # -- positions -----------------------------------------------------------------------------------------------------------------
    def input_word(self, g: int) -> GatePosition | None:
        if g >= self.input_gates:
            return None
        L = self.loc.locate_input(int(g)) if (self.loc is not None and hasattr(self.loc, "locate_input")) else {"kind": "typed_input", "gate": int(g)}
        if L.get("kind") not in ("weight", "prompt_token"):
            # a registered program's typed input (GenericLocator) / no locator: the adapter's `weight_word` names the Input position
            w = self._weight_word(int(g)) if self._weight_word is not None else None
            if w is None:
                raise PositionRefusal("INPUT_UNLOCATED", f"Input gate {g}: no input layout names its committed leaf ({L.get('refused', L.get('kind'))})")
            return w
        if L.get("kind") == "weight":
            if self._weight_word is None:
                raise PositionRefusal("NO_WEIGHTS_LAYOUT", f"gate {g}: weights word {L['canonical_id']}[{L['offset']}] but no weights layout to name its leaf")
            w = self._weight_word(int(g))
            if w is None:
                raise PositionRefusal("NO_WEIGHTS_LAYOUT", f"gate {g}: weights word {L['canonical_id']}[{L['offset']}] has no leaf in the registered weights index")
            return w
        if L.get("kind") == "prompt_token":
            if self._prompt_word is None:
                raise PositionRefusal("NO_PROMPT_LAYOUT", f"gate {g}: prompt token {L['t']} but no boundary layout to name its leaf")
            return self._prompt_word(int(g))
        raise PositionRefusal("INPUT_UNLOCATED", f"Input gate {g}: {L.get('refused', L)}")

    def boundary_word(self, g: int) -> GatePosition | None:
        w = self.boundary.word(g)
        if w is None:
            return None
        return GatePosition(w.gate, w.origin, w.tree, w.leaf, w.shape, w.flat, gate_width(self.program, g))

    def literal_word(self, g: int) -> GatePosition | None:
        """[E5 §4 / E6 `kind=literal`] A zero-arity ``Const<w>`` gate that is the output gate of NO VU of the pinned population is a program
        literal, not a committed word: ``GatePosition(origin="literal", value=bits)``.  Under a population that still lists the literal kinds as
        VUs (the v1 rows) the gate IS a VU output and resolves to its committed (interior) position instead -- the population decides."""
        rj = self.rows.row_of_gate(g)
        if rj is not None:
            row, j = rj
            if (g - (int(row[4]) + j * int(row[5]))) in self.rel_outputs(row):
                return None
        return literal_of(self.program, g)

    def resolve(self, g: int) -> GatePosition:
        g = int(g)
        w = self.input_word(g)
        if w is not None:
            return w
        w = self.boundary_word(g)
        if w is not None:
            return w
        w = self.literal_word(g)
        if w is not None:
            return w
        return self.interior_word(g)

    def oracle(self) -> Callable[[str, int], Any]:
        """E7's ``word(request_id, g) -> CommittedWord | None`` oracle (``ck_elem.RecordHandle(oracles=(...))``): the position of any committed
        gate of this request, ``None`` for a gate that is no committed word (an operand that is not a VU output ...); refusals propagate."""
        try:
            from e2e.coverage_v1.ck_elem import CommittedWord
        except ImportError:  # pragma: no cover
            CommittedWord = None  # type: ignore[assignment]

        def word(request_id: str, g: int):
            if self.request_id and str(request_id) != self.request_id:
                return None
            try:
                p = self.resolve(int(g))
            except PositionRefusal as e:
                if e.code in ("OUTSIDE_PARTITION", "NOT_AN_OUTPUT", "NO_REGION"):
                    return None
                raise
            if p.origin == ORIGIN_LITERAL:
                return None  # a program literal is no committed word: the builder compiles it from the program text (E5 §4)
            return CommittedWord(p.tree, p.leaf, p.shape, p.flat, p.width) if CommittedWord is not None else p

        return word

    # -- the interior word: E2's per-call chunk rule -----------------------------------------------------------------------------------
    def call_of(self, g: int) -> tuple[Any, int, int]:
        if self.calls is None:
            raise PositionRefusal("NO_CALLS", f"gate {g}: no call order declared for the interior layout")
        c = self.calls(int(g))
        if c is None:
            raise PositionRefusal("NO_CALLS", f"gate {g} lies in no call of the interior layout")
        return c

    def call_layout(self, call: tuple[Any, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
        """The interior word sequence of one call: ``(gates ascending, leaf rank within the call per gate, flat per gate, [(n words, width,
        k)] per leaf in LAYOUT order)`` -- ``k`` the chunk index the leaf id carries (v1: = the rank; v2: restarting per width group) --
        one numpy batch per partition row of the call; cached per call id (memory = the working set of calls, not the program)."""
        cid, lo, hi = call
        cl = self._call_layout.get(cid)
        if cl is None:
            t0 = time.perf_counter()
            gs: list[np.ndarray] = []
            ws: list[np.ndarray] = []
            for row in self.rows.rows_in(lo, hi):
                outs = self.output_gates_of_row(row)
                if not len(outs):
                    continue
                m = ~self.boundary.mask(outs)
                if not m.any():
                    continue
                gs.append(outs[m])
                ws.append(self.output_widths_of_row(row)[m])
            if gs:
                gates = np.concatenate(gs)
                widths = np.concatenate(ws)
                order = np.argsort(gates, kind="stable")
                gates, widths = gates[order], widths[order]
                if self.layout is None:
                    chunk = flat = np.zeros(len(gates), dtype=np.int64)
                    sizes: list[tuple[int, int, int]] = []
                else:
                    c_l, f_l, sizes = self.layout.chunks(widths.tolist())
                    chunk, flat = np.asarray(c_l, dtype=np.int64), np.asarray(f_l, dtype=np.int64)
            else:
                gates = chunk = flat = np.zeros(0, dtype=np.int64)
                sizes = []
            cl = self._call_layout[cid] = (gates, chunk, flat, sizes)
            self.timings[f"call_layout_s[{cid}]"] = time.perf_counter() - t0
        return cl

    def interior_word(self, g: int) -> GatePosition:
        if self.regions is None:
            raise PositionRefusal("NO_REGION", f"gate {g}: no RU regions declared for request {self.request_id!r}")
        region = self.regions.at(g)
        if region is None:
            raise PositionRefusal("NO_REGION", f"gate {g} lies in no declared RU region of request {self.request_id!r}")
        rj = self.rows.row_of_gate(g)
        if rj is None:
            raise PositionRefusal("OUTSIDE_PARTITION", f"gate {g} is a gate of no VU of the pinned partition")
        row, j = rj
        member_lo = int(row[4]) + j * int(row[5])
        if (g - member_lo) not in self.rel_outputs(row):
            raise PositionRefusal("NOT_AN_OUTPUT", f"gate {g} is not an output gate of its VU [{member_lo},{member_lo + row[6]}) ({self.rows.kind_name(row[3])}); "
                                                   f"only output gates are committed (PROTOCOL step 3)")
        call = self.call_of(g)
        gates, chunk, flat, sizes = self.call_layout(call)
        k = int(np.searchsorted(gates, g))
        if k >= len(gates) or int(gates[k]) != g:
            raise PositionRefusal("NOT_AN_OUTPUT", f"gate {g} is not an interior word of call {call[0]} [{call[1]},{call[2]})")
        if self.layout is None:
            raise PositionRefusal("INTERIOR_LAYOUT_UNPINNED",
                                  f"gate {g} ({self.rows.kind_name(row[3])}) is an interior word (#{k} of call {call[0]} in {region.tree}) and "
                                  f"protocol_instance.json#commitments.interior_layout is absent")
        ck = int(chunk[k])
        n_words, width, kk = sizes[ck]
        return GatePosition(int(g), ORIGIN_INTERIOR, region.tree, self.layout.leaf_id(self.prefix, call[0], kk, width), (int(n_words),), int(flat[k]), int(width))

    # -- the region's leaves in rank order (E2 item 5: rank = call order, then [v2: increasing width, then] chunk index) -----------------
    def region_calls(self, region: Region) -> Iterator[tuple[Any, int, int]]:
        """The calls of ``region`` in pinned call order (ascending lo), derived by walking the region's gates call by call."""
        g = region.lo
        while g < region.hi:
            c = self.calls(g) if self.calls is not None else None
            if c is None:
                g += 1  # a gate of no call (should not happen inside a region; skip rather than loop forever)
                continue
            if c[1] < region.lo or c[2] > region.hi:
                raise PositionRefusal("REGION_CALL_STRADDLE", f"call {c[0]} [{c[1]},{c[2]}) straddles region {region.tree} [{region.lo},{region.hi})")
            yield c
            g = c[2]

    def region_leaves(self, region: Region) -> Iterator[tuple[str, int, int, int]]:
        """Stream ``(leaf id, n words, width, first gate)`` of every interior leaf of ``region`` in rank order (first gate = the leaf's
        lowest gate index; under v2 a leaf's gates are not contiguous in the call)."""
        for c in self.region_calls(region):
            gates, chunk, _flat, sizes = self.call_layout(c)
            if not len(gates):
                continue
            first = np.full(len(sizes), np.iinfo(np.int64).max, dtype=np.int64)
            np.minimum.at(first, chunk, gates)
            for r, (n, w, k) in enumerate(sizes):
                yield self.layout.leaf_id(self.prefix, c[0], k, w), int(n), int(w), int(first[r])

    def region_words(self, region: Region) -> int:
        return sum(len(self.call_layout(c)[0]) for c in self.region_calls(region))

    def required_gates(self, region: Region) -> Iterator[tuple[int, str, int]]:
        """Stream ``(gate, kind, width bits)`` of every interior word of ``region`` in layout order -- E1's required-value inventory and
        E2's leaf sequence, without materialising the population (one numpy batch per row)."""
        for c in self.region_calls(region):
            gates, _chunk, _flat, _sizes = self.call_layout(c)
            for row in self.rows.rows_in(c[1], c[2]):
                outs = self.output_gates_of_row(row)
                if not len(outs):
                    continue
                m = ~self.boundary.mask(outs)
                if not m.any():
                    continue
                kind = self.rows.kind_name(row[3])
                for g, w in zip(outs[m].tolist(), self.output_widths_of_row(row)[m].tolist()):
                    yield int(g), kind, int(w)

    # -- the expected statement as positions ---------------------------------------------------------------------------------------
    def statement(self, vu_index: int, kind: str, lo: int, hi: int, *, label: str = ""):
        """``(vu_index, kind, operands -> positions, outputs -> positions)`` of the VU ``[lo, hi)``.  Raises :class:`PositionRefusal`
        (by code) or :class:`~veritor.interface.expected.InteriorLayoutUnpinned` -- never a statement over invented words."""
        from veritor.interface.expected import InteriorLayoutUnpinned, PositionStatement

        operands = operand_gates_of(self.program, lo, hi)
        outputs = output_gates_of(self.program, lo, hi)
        if not outputs:
            raise PositionRefusal("NO_OUTPUT", f"vu {vu_index}: [{lo},{hi}) ({kind}) has no output gate inside its interval")
        try:
            ops = tuple(self.resolve(g) for g in operands)
            outs = tuple(self.resolve(g) for g in outputs)
        except PositionRefusal as e:
            if e.code == "INTERIOR_LAYOUT_UNPINNED":
                raise InteriorLayoutUnpinned(f"V3.interior_layout: vu {vu_index} ({kind}): {e}") from e
            raise
        return PositionStatement(int(vu_index), str(kind), tuple(p.expected() for p in ops), tuple(p.expected() for p in outs),
                                 operands, outputs, tuple(p.origin for p in ops) + tuple(p.origin for p in outs), label or f"vu={vu_index} {kind} [{lo},{hi})")


def e2_layout(resolver: GateResolver, region: Region):
    """E2's ``e2e.interior.layout.InteriorLayout`` of ``region`` built from THIS resolver's required gates (rule item 3: the partition's VU
    output gates with no boundary position, one word per gate, widths from the population's kind table) over the region's calls in pinned
    order -- what E9 passes to ``validate_step4`` / ``required_regions`` as the verifier's ``required`` layout, and what
    ``tests/contract/e3_positions_test.py`` asserts gate-by-gate against :meth:`GateResolver.interior_word`.  Raises
    :class:`~veritor.interface.expected.InteriorLayoutUnpinned` without the pin; ``ImportError`` when E2's module is not in the checkout."""
    from e2e.interior import layout as _e2  # type: ignore[import-not-found]
    from veritor.interface.expected import InteriorLayoutUnpinned

    if resolver.layout is None:
        raise InteriorLayoutUnpinned(f"V3.interior_layout: {region.tree}: no pinned interior_layout -- no required layout can be stated")
    spans, by_call = [], {}
    for c in resolver.region_calls(region):
        spans.append(_e2.CallSpan(str(c[0]), int(c[1]), int(c[2])))
        by_call[spans[-1].call] = c

    def call_gw(span: Any) -> tuple[np.ndarray, np.ndarray]:  # per call: gates ascending; the width of gate i = the width of its leaf
        g, chunk, _flat, sizes = resolver.call_layout(by_call[span.call])
        if not len(g):
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        return np.asarray(g, dtype=np.int64), np.asarray([w for _n, w, _k in sizes], dtype=np.int64)[chunk]

    # under the rule the pin NAMES (v1 for records written before D13, v2 for new pins) -- E2's layout accepts both by name.
    # [NIGHT D38, E2 on E3's behalf] LazyInteriorLayout: one call materialised at a time (memory ~ the largest call, not the RU), the same
    # ids / ranks / locate() as the eager InteriorLayout.build (asserted in tests/contract/e2_interior_test.py)
    return _e2.LazyInteriorLayout(region.tree, spans, call_gw, prefix=resolver.prefix, chunk_words=resolver.layout.chunk_words,
                                  rule=resolver.layout.rule)


# --- inventory: per kind {boundary | interior, count}, streaming over rows ---------------------------------------------------------------


def inventory(resolver: GateResolver, *, regions: Iterable[Region] | None = None) -> dict[str, Any]:
    """Per kind ``{"boundary": n, "interior": n, "vus": n, "out_gates_per_vu": k, "producer": boundary | interior-interpreter | mixed}`` over the
    output gates of every VU of the given regions (default: every region of the resolver), plus per region ``{"words", "leaves", "calls"}``
    -- what E1 (required values) and E6 (family inventory / leaf counts) need, computed one row at a time (never the population in Python)."""

    per_kind: dict[str, dict[str, Any]] = {}
    per_region: dict[str, dict[str, Any]] = {}
    total_b = total_i = 0
    for region in (regions if regions is not None else (resolver.regions or ())):
        words = leaves = ncalls = 0
        for c in resolver.region_calls(region):
            ncalls += 1
            for row in resolver.rows.rows_in(c[1], c[2]):
                outs = resolver.output_gates_of_row(row)
                if not len(outs):
                    continue
                nb = resolver.boundary.count_below(outs)
                ni = int(len(outs) - nb)
                k = per_kind.setdefault(resolver.rows.kind_name(row[3]), {"boundary": 0, "interior": 0, "vus": 0, "out_gates_per_vu": len(resolver.rel_outputs(row)),
                                                                          "widths": sorted(set(resolver.rel_widths(row)))})
                k["boundary"] += int(nb)
                k["interior"] += ni
                k["vus"] += int(row[1])
                words += ni
            if resolver.layout is not None:
                leaves += len(resolver.call_layout(c)[3])
        per_region[region.tree] = {"words": words, "leaves": leaves if resolver.layout is not None else None, "calls": ncalls, "lo": region.lo, "hi": region.hi}
        total_i += words
    for k in per_kind.values():
        total_b += k["boundary"]
        k["producer"] = "boundary" if k["interior"] == 0 else ("interior-interpreter" if k["boundary"] == 0 else "mixed")
    return {"kinds": dict(sorted(per_kind.items())), "regions": per_region, "boundary_words": total_b, "interior_words": total_i,
            "rule": "interior-layout/chunks-4096/v1: per call, the output gates of every VU of the partition in the call with no original step-1 "
                    "position, ascending gate index, chunked at 4096 words / a width change / the call's end"}
