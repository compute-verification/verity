"""[E3, D32 — user's ruling 18:41 PT] The expressibility CENSUS of a declared population: for EVERY VU, the verifier's own expected statement
(``GateResolver.statement``: operand references — committed position / prescribed literal / table reference —, the aliasing pattern BY IDENTITY of the
committed word (never by value), padding literals, output positions, the registered family) and the builder side's answer ``can_express(statement)``
(E7's predicate; until it lands the call is stubbed behind :func:`can_express` and recorded ``PENDING``).  A VU the oracle itself refuses is a census
verdict too (``E_EXPECTATION_<code>``), never a crash.

    PYTHONPATH=. python e2e/verifier_v3/census.py --instance out/eod/e6/tiny/protocol_instance.json \
        --population <population.v5.json (v2)> --record ~/veritor-archive/A9-TINY-04137cde/V5-TINY-bex0-inc [--limit 10000] [--out ...]

Output ``out/eod/e3/census/<instance_id>.json``: per-(kind, operand shape, verdict) counts with one example VU each, the exact uncovered VU list (or its
sha256 + a sample when > 10k), totals (VUs, expressible, not expressible by reason, pending), and the MEASURED rate (VUs/s) so the full run can be
scheduled (laptop if < 90 min, else the job file ``out/eod/jobs/e3-census.json``).

Operand shape (one string per VU; the aggregation key with the kind):
  ``<origins>|alias=<pattern>|pad0=<n>|lit=<ids>|out=<origins>`` where origins are per operand ``B`` boundary (step-1 row word), ``I`` interior (step-3
  word), ``W`` weights table word (a TABLE reference), ``P`` prompt / boundary-tree input, ``L`` literal; the alias pattern labels each committed operand
  position by the first-occurrence letter of its committed IDENTITY ``(tree, leaf, offset)`` (``a,a`` = x·x on one word; ``a,b`` = two words even when
  their bits agree; ``L`` = literal, ``.`` = table word, distinct by gate); ``pad0`` counts zero literals (``Const16[0x0000]`` / ``Const32[0x00000000]``);
  wide fan-in (> 256 operands, the Gather / whole-scan units) is summarised as ``W×n`` / ``B×n`` with the distinct-identity count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

WIDE = 256  # operands above this: identities counted vectorised (a Gather row / the whole-scan logits row), never per-word objects

# --- the builder side's predicate (E7) -------------------------------------------------------------------------------------------------


def can_express(statement: Any, unit: Mapping[str, Any] | None = None, *, program: Any = None, tables: Mapping[str, Any] | None = None) -> str:
    """E7's builder-side predicate (``coverage_v1.builders.can_express(statement, unit=None, *, program=None, tables=None) -> "ok" | "E_<reason>"``,
    D32.2, merged `99dfe222`): with ``unit`` + ``program`` it runs the registered builder's OWN ``build`` over a phantom record (identities, zero bits
    -- the exact prover path: resolvers, aliasing, padding literals, prescribed constants, table rows, family resolution, validate, lowering);
    ``tables`` = the instance's ``commitments.tables`` pin (the MUFU kinds).  A wide unit is handed as E3's summary dict (``kind`` key).  My side
    adapted to E7's signature (the 02:12Z stub called ``fn(statement)`` / ``fn(kind, statement)``)."""
    try:
        from e2e.coverage_v1.builders import can_express as _ce
    except ImportError:
        return "PENDING(E7.can_express)"
    try:
        r = _ce(statement, unit, program=program, tables=tables)
    except Exception as e:  # noqa: BLE001 -- the predicate names crashes itself; this is the belt
        return f"E_PREDICATE_{type(e).__name__}:{str(e)[:80]}"
    return "ok" if r in (True, None, "ok") else ("ok" if isinstance(r, str) and r.split("(")[0].split(":")[0].strip() == "ok" else str(r))


#: The NAMED rules a statement's expressibility can rely on, by the shape feature that invokes them (G1 names the new ones; the census attributes a
#: verdict to a rule by FEATURE, never by an old E_ code): aliasing -> `aliased-body/v1` (D30/D32); a program literal at an operand position (PV
#: `ZERO16` pad, `Const32[...]`) -> `prescribed-literal/v1` (G1(a), LANDED: E9 `steps.LITERAL_RULE` / E7 `elem_builders.LITERAL_RULE`);
#: the embedding Gather -> `gather-equality/v1` (G1(b), LANDED: E9 `steps.GATHER_RULE`).  The names are READ from the verifier's module (one
#: derivation); the literals here are only the fallback when the module predates them.
def rule_names() -> dict[str, str]:
    names = {"aliased": "aliased-body/v1", "literal": "prescribed-literal/v1", "gather": "gather-equality/v1", "composed": "composed-body/v1"}
    for modname, pairs in (("e2e.verifier_v3.steps", (("LITERAL_RULE", "literal"), ("GATHER_RULE", "gather"))),
                           ("e2e.coverage_v1.elem_builders", (("LITERAL_RULE", "literal"), ("ALIAS_RULE", "aliased"), ("COMPOSED_RULE", "composed"))),
                           ("e2e.coverage_v1.builders", (("PRESCRIBED_LITERAL_RULE", "literal"), ("ALIASED_BODY_RULE", "aliased"), ("GATHER_FAMILY", "gather")))):
        try:
            import importlib

            M = importlib.import_module(modname)
        except Exception:  # noqa: BLE001
            continue
        for attr, key in pairs:
            v = getattr(M, attr, None)
            if isinstance(v, str) and v:
                names[key] = v
    return names


def rules_of(kind: str, pos: list, pat: list[str], names: Mapping[str, str]) -> list[str]:
    out: list[str] = []
    if any(p.origin == "literal" for p in pos):
        out.append(names["literal"])
    if len(set(x for x in pat if x != "L" and x != ".")) < sum(1 for x in pat if x != "L" and x != "."):
        out.append(names["aliased"])
    if kind.startswith("Gather"):
        out.append(names["gather"])
    if kind in composed_kinds():
        out.append(names["composed"])  # E7's `composed-body/v1` registration (`elem_builders.COMPOSED_KINDS`): Qwen2's `Bf16Add_v1|batch` QKV bias add
    return out


_COMPOSED: frozenset[str] | None = None


def composed_kinds() -> frozenset[str]:
    global _COMPOSED
    if _COMPOSED is None:
        try:
            from e2e.coverage_v1.elem_builders import COMPOSED_KINDS
            _COMPOSED = frozenset(str(k) for k in COMPOSED_KINDS)
        except Exception:  # noqa: BLE001
            _COMPOSED = frozenset()
    return _COMPOSED


ISOLATE = ("Mufu", "Rsqrt", "InvSum", "Ftz", "F32Max", "GuardNegInfZero")  # E4's / E7's MUFU-table builders: `TableWords` for the 4 pinned tables


def maxrss_gb() -> float:
    """Peak RSS of this process in GB -- `ru_maxrss` is BYTES on macOS and KILOBYTES on Linux (the pod runs read 0.0017 for 1.67 GB before 08:55Z)."""
    import resource
    import sys as _sys

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if _sys.platform == "darwin" else r * 1024 / 1e9


def isolate_kind(kind: str) -> bool:
    """E7's predicate on a MUFU-table kind builds E4's `TableWords` for the pinned tables and keeps them cached for the process: +1.2 GB (MufuSqrtFtz),
    +0.5 GB (MufuEx2Ftz), +0.7 GB (RsqrtApprox), +0.4 GB (Fa2InvSum) -- MEASURED 03:20Z, 3.12 GB maxrss after four calls, which is what the laptop's
    `mem_guardian` (per-process cap 2.5 GB) killed pass 2 for twice.  Those keys are evaluated in a SUBPROCESS (one per key: fresh oracle, one
    predicate call, exit) so the census process stays at ~0.33 GB.  E7's cost, not the census's."""
    from e2e.coverage_v1 import builders as B

    if os.environ.get("CENSUS_ISOLATE", "1") == "0":
        # [E3 2026-09-14 00:10Z, D59 census of 256x63] the isolation is a LAPTOP artifact (mem_guardian's per-process cap); on a pod each isolated
        # call re-parses the population file (1,013 MB for 256x63 -> ~12 GB RSS per call, ~20 s) beside the shard's own copy: 8 of 12 shards
        # OOM-killed at 128 GB.  CENSUS_ISOLATE=0 keeps E7's predicate in-process (+~3 GB cached TableWords per shard, the number above).
        return False
    b = B.BUILDERS.get(kind)
    return getattr(b, "owner", "") == "ck-ftz" or any(t in kind for t in ISOLATE)


def can_express_isolated(instance_path: Path, population_path: Path, record: Path | None, vu: int, kind: str, lo: int, hi: int, request_id: str | None = None) -> str:
    """:func:`can_express` for ONE VU in a fresh interpreter (see :func:`isolate_kind`); the verdict comes back on stdout as JSON."""
    import os
    import subprocess

    args = json.dumps({"instance": str(instance_path), "population": str(population_path), "record": str(record) if record else None, "vu": vu, "kind": kind, "lo": lo, "hi": hi, "request_id": request_id})
    code = "import sys, json; from e2e.verifier_v3.census import _one_vu; print(json.dumps(_one_vu(**json.loads(sys.argv[1]))))"
    try:
        r = subprocess.run([sys.executable, "-c", code, args], capture_output=True, text=True, timeout=600, env=dict(os.environ))
    except subprocess.TimeoutExpired:
        return "E_PREDICATE_TIMEOUT(600s, isolated)"
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-1:] or [f"exit {r.returncode}"]
        return f"E_PREDICATE_SUBPROCESS:{tail[0][:100]}"
    try:
        return str(json.loads(r.stdout.strip().splitlines()[-1]))
    except Exception:  # noqa: BLE001
        return f"E_PREDICATE_SUBPROCESS:unparsable:{r.stdout[-100:]}"


def _one_vu(instance: str, population: str, record: str | None, vu: int, kind: str, lo: int, hi: int, request_id: str | None = None) -> str:
    inst = json.loads(Path(instance).read_text())
    pop = json.loads(Path(population).read_text())
    oracle, program, rid, _C = build_oracle(inst, pop, Path(record) if record else None, request_id=request_id)
    res = oracle.resolver(rid)
    st = res.statement(int(vu), kind, int(lo), int(hi))
    return can_express(st, {"vu_index": int(vu), "kind": kind, "request_id": rid, "gate_interval": [int(lo), int(hi)]}, program=program,
                       tables=(inst.get("commitments") or {}).get("tables"))


# --- one VU ------------------------------------------------------------------------------------------------------------------------------


def _letter(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(97 + r) + s
    return s


def _origin_code(p) -> str:
    if p.origin == "literal":
        return "L"
    if p.origin == "input":
        return "W" if p.tree == "weights" else "P"
    return "B" if p.origin == "boundary" else "I"


def _token(p) -> str:
    """One operand / output position as the predicate sees it: origin letter + word width, or the literal's primitive id (value and width)."""
    if p.origin == "literal":
        return f"L:{p.leaf}"  # the Const primitive id spells width AND value; the literal tree's position is a function of the program's id set
    t = str(getattr(p, "tree", "") or "")
    if p.origin == "input" and t.startswith("table/"):
        return f"T{int(p.width)}:{t}"  # a TABLE reference (D23 MUFU tables, `table/mufu/<name>`): the table's NAME is part of the shape
    return f"{_origin_code(p)}{int(p.width)}"


def _pad0(p) -> bool:
    return p.origin == "literal" and int(p.value or 0) == 0


def shape_key(kind: str, pos: list, outs: list, pat: list[str]) -> str:
    """[03:06Z integrator rule] The SHAPE KEY of a statement -- everything E7's identity-structural ``can_express`` can depend on: kind (its family and
    statement format follow from it), the operand positions in order as ``<origin><width>`` / ``L:<Const id>`` tokens (padding and prescribed constants
    by POSITION and value, table references as ``W``), the alias wiring (``a,a,b``: which positions name one committed identity), the output tokens.
    Never a committed word's name or bits.  One predicate call per distinct key answers for every VU of the key."""
    return f"{kind}|{','.join(_token(p) for p in pos)}|alias={','.join(pat)}|out={','.join(_token(p) for p in outs)}"


def shape_summary(pos: list, outs: list, pat: list[str]) -> str:
    """The readable shape (pass-1 spelling): origin letters, alias pattern, zero-literal count, other literal ids, output origins."""
    codes = "".join(_origin_code(p) for p in pos)
    pad0 = sum(1 for p in pos if _pad0(p))
    lits = sorted({p.leaf for p in pos if p.origin == "literal" and not _pad0(p)})
    return f"{codes}|alias={','.join(pat)}|pad0={pad0}|lit={','.join(lits) or '-'}|out={''.join(_origin_code(p) for p in outs)}"


RULES = rule_names()


def vu_verdict(resolver, program, vu: int, kind: str, lo: int, hi: int, family: str | None) -> dict[str, Any]:
    """The census row of one VU: ``{"vu", "kind", "shape", "verdict", "express", "n_ops", "n_out"}``.  Narrow units go through the verifier's
    :meth:`GateResolver.statement` verbatim; wide units resolve every NON-input operand and count input identities vectorised (an Input gate is
    its own committed word: distinct gates are distinct identities)."""
    from e2e.verifier_v3.locator import PositionRefusal, operand_gates_of, output_gates_of
    from veritor.interface.expected import InteriorLayoutUnpinned

    row: dict[str, Any] = {"vu": int(vu), "kind": kind}
    if family is not None and family.startswith("verifier ("):
        # E7 night (G1): a kind verified BY NAME by the verifier (builders.VERIFIER_COVERED, e.g. embed-gather = V6 row opening + equality) has no
        # builder statement to classify -- its row is the rule; the 49k-operand gather resolve (~0.6 s/VU x 6,912) is skipped
        row.update(n_ops=int(hi - lo), n_out=0, shape=f"verifier-covered|{family}", key=f"{kind}|{family}", family=family, verdict="ok", aliased=False)
        row["_statement"] = {"vu": int(vu), "kind": kind, "verifier_rule": family}  # the predicate (can_express) answers the kind by name
        return row
    try:
        gemm = getattr(resolver.loc, "locate_gemm_coordinate", None) if kind.startswith("GemmCoordinate_v1{") else None
        if gemm is not None:
            return _gemm_verdict(resolver, program, row, kind, lo, hi, family, gemm)
        ops = operand_gates_of(program, lo, hi)
        outs = output_gates_of(program, lo, hi)
        if not outs:
            raise PositionRefusal("NO_OUTPUT", f"[{lo},{hi}) has no output gate inside its interval")
        row["n_ops"], row["n_out"] = len(ops), len(outs)
        if len(ops) <= WIDE:
            st = resolver.statement(vu, kind, lo, hi)
            pos = [resolver.resolve(g) for g in ops]  # GatePosition (with origin / value) -- the same objects `statement` built
            codes = [_origin_code(p) for p in pos]
            ids: dict[tuple, str] = {}
            pat = []
            for p, c in zip(pos, codes):
                if c == "L":
                    pat.append("L")
                    continue
                key = (p.tree, p.leaf, int(p.flat))
                pat.append(ids.setdefault(key, _letter(len(ids))))
            outp = [resolver.resolve(g) for g in outs]
            row["shape"] = shape_summary(pos, outp, pat)
            row["key"] = shape_key(kind, pos, outp, pat)
            row["aliased"] = len(ids) < sum(1 for c in codes if c != "L")
            row["rules"] = rules_of(kind, pos, pat, RULES)
            statement: Any = st
        else:
            g = np.asarray(ops, dtype=np.int64)
            is_in = g < int(resolver.input_gates)
            n_in = int(is_in.sum())
            distinct_in = int(len(np.unique(g[is_in])))
            # every input operand is a committed word only if the weights / prompt layout names it: resolve the first and last as a check
            in_codes = Counter()
            for gg in ([int(g[is_in][0]), int(g[is_in][-1])] if n_in else []):
                in_codes[_origin_code(resolver.resolve(gg))] += 1
            rest = [resolver.resolve(int(x)) for x in g[~is_in]]
            ids = {}
            pat = []
            for p in rest:
                if p.origin == "literal":
                    pat.append("L")
                    continue
                pat.append(ids.setdefault((p.tree, p.leaf, int(p.flat)), _letter(len(ids))))
            pad0 = sum(1 for p in rest if _pad0(p))
            lits = sorted({p.leaf for p in rest if p.origin == "literal" and not _pad0(p)})
            out_codes = [_origin_code(p) for p in (resolver.resolve(g0) for g0 in outs)]
            wide = f"{'/'.join(sorted(in_codes)) or 'W'}×{n_in}(distinct {distinct_in})" if n_in else ""
            narrow = "".join(_origin_code(p) for p in rest)
            if len(narrow) > 24:
                narrow = f"{Counter(narrow).most_common(1)[0][0]}×{len(narrow)}(distinct {len(ids)})"
                pat = [f"{len(ids)} identities"]
            row["shape"] = f"{wide}{'+' if wide and narrow else ''}{narrow}|alias={','.join(pat) or '-'}|pad0={pad0}|lit={','.join(lits) or '-'}|out={''.join(out_codes)}"
            row["key"] = f"{kind}|WIDE:{'/'.join(sorted(in_codes)) or 'W'}x{n_in}:distinct={distinct_in}|{','.join(_token(p) for p in rest[:64])}{'...' if len(rest) > 64 else ''}|alias={','.join(pat) or '-'}|out={','.join(_token(resolver.resolve(g0)) for g0 in outs)}"
            row["aliased"] = (distinct_in < n_in) or (len(ids) < sum(1 for p in rest if p.origin != "literal"))
            row["rules"] = rules_of(kind, rest, pat, RULES)
            if distinct_in < n_in and RULES["aliased"] not in row["rules"]:
                row["rules"].append(RULES["aliased"])
            statement = {"vu": vu, "kind": kind, "operands": len(ops), "input_operands": n_in, "distinct_input_identities": distinct_in,
                         "other_operands": [p.to_json() for p in rest[:64]], "outputs": [resolver.resolve(g0).to_json() for g0 in outs]}
        row["family"] = family
        row["verdict"] = "ok" if family is not None else "E_NO_BUILDER"
        row["_statement"] = statement  # the runner hands it to E7's predicate (per class sample / every VU of a small class)
    except InteriorLayoutUnpinned as e:
        row["verdict"] = "E_EXPECTATION_INTERIOR_LAYOUT_UNPINNED"
        row["error"] = str(e)[:200]
    except PositionRefusal as e:
        row["verdict"] = f"E_EXPECTATION_{e.code}"
        row["error"] = str(e)[:200]
    except Exception as e:  # noqa: BLE001 -- a defect in the oracle is a census verdict, not a crash (D31.2's AttributeError class)
        row["verdict"] = f"E_EXPECTATION_{type(e).__name__}"
        row["error"] = str(e)[:200]
    row.setdefault("shape", "-")
    row.setdefault("key", f"{kind}|{row['verdict']}")
    return row


_GEMM_ROWS: dict[tuple, dict[str, Any]] = {}


def _gemm_verdict(resolver, program, row: dict[str, Any], kind: str, lo: int, hi: int, family: str | None, gemm) -> dict[str, Any]:
    """A `GemmCoordinate_v1{K}` member: the verifier's path is Serve_v1's structural coordinate (`locate_gemm_coordinate` + `gemm_operands`, the
    fixed circuit `programs.gemm_coordinate(K)` of the instance's epoch family) -- per member the coordinate `(root, j, N)`; per ROW (one
    coordinate row = one x row, N w rows) the operands are spelled ONCE through the oracle's full statement (`statement` over the first member:
    x = K boundary words of one producer row, w = K weights-table words, out = one boundary word) and `gemm_operands` is checked on the first
    and last member (same x, w.row = j).  The operand shape is uniform over the row by construction of the coordinate."""
    from e2e.verifier_v3.locator import PositionRefusal, operand_gates_of, output_gates_of

    L = gemm(lo, hi)
    if "refused" in L:
        raise PositionRefusal("GEMM_COORDINATE", str(L["refused"]))
    key = (str(L.get("root")), int(L.get("gemm_index", 0)), int(L.get("N", 0)))
    j, N = int(L["j"]), int(L["N"])
    info = _GEMM_ROWS.get(key)
    if info is None:
        if len(_GEMM_ROWS) > 8192:  # rows are visited in population order; bound the per-row memo at long context (1M coordinate rows at 6500x1600)
            for k in list(_GEMM_ROWS)[:4096]:
                del _GEMM_ROWS[k]
        ops = operand_gates_of(program, lo, hi)
        outs = output_gates_of(program, lo, hi)
        pos = [resolver.resolve(g) for g in ops]
        codes = Counter(_origin_code(p) for p in pos)
        ids = {(p.tree, p.leaf, int(p.flat)) for p in pos if p.origin != "literal"}
        go = resolver.loc.gemm_operands(lo, hi)
        if "refused" in go:
            raise PositionRefusal("GEMM_OPERANDS", str(go["refused"]))
        x_leaf = sorted({p.leaf for p in pos if p.origin in ("boundary", "interior")})
        w_leaf = sorted({p.tree for p in pos if p.origin == "input"})
        out_codes = "".join(_origin_code(resolver.resolve(g)) for g in outs)
        shape = (f"{'+'.join(f'{c}×{n}' for c, n in sorted(codes.items()))}(distinct {len(ids)})|alias={'-' if len(ids) == len(pos) else 'REPEAT'}|pad0=0|lit=-"
                 f"|x={len(x_leaf)} row leaf|w={','.join(w_leaf)}|K={go['K']}|out={out_codes}")
        info = _GEMM_ROWS[key] = {"shape": shape, "n_ops": len(ops), "n_out": len(outs), "K": int(go["K"]), "N": N, "x": go["x"], "w0": go["w"], "aliased": len(ids) < len(pos)}
    if j == N - 1:  # last member of the row: the coordinate's w row must be j (the operands follow the coordinate, not the prover)
        go = resolver.loc.gemm_operands(lo, hi)
        if "refused" in go or int(go["w"]["row"]) != j or go["x"] != info["x"]:
            raise PositionRefusal("GEMM_ROW_SHAPE", f"member j={j}: gemm_operands {go} disagree with the row's first member {info['x']} / {info['w0']}")
    row.update(n_ops=info["n_ops"], n_out=info["n_out"], shape=info["shape"], aliased=info["aliased"], family=family, key=f"{kind}|GEMM:K={info['K']}|{info['shape']}")
    row["verdict"] = "ok" if family is not None else "E_NO_BUILDER"
    row["_statement"] = {"kind": kind, "vu": row["vu"], "coordinate": L, "K": info["K"], "operands": info["n_ops"], "x": info["x"], "w_row": j}  # E7: approved checker kind -> ok
    row["fixed_circuit"] = family is not None
    return row


# --- the population ------------------------------------------------------------------------------------------------------------------------


def iter_units(pop: dict[str, Any], limit: int | None = None):
    """``(vu, kind, lo, hi)`` for every VU of the population in VU order (rows are ``[first_vu, count, ru, kind, lo, stride, unit_size]``)."""
    names = {int(e["id"]): f"{e['fn']}|{e['form']}" for e in pop["kinds"]["entries"]}
    n = 0
    for r in pop["rows"]:
        first, count, _ru, k, lo, stride, usz = (int(x) for x in r)
        kind = names[k]
        for j in range(count):
            g = lo + j * stride
            yield first + j, kind, g, g + usz
            n += 1
            if limit is not None and n >= limit:
                return


def row_picks(count: int, picks: int) -> list[int]:
    """[rows=sampled] The member indices of one population row the census EVALUATES: every member when ``count <= picks``, else the first, the
    last and ``picks - 2`` evenly spread members.  The row's other members are ATTRIBUTED the row's verdict iff every evaluated member agrees on
    (shape key, verdict); a row whose evaluated members disagree is re-evaluated member by member (exact)."""
    if count <= picks:
        return list(range(count))
    inner = max(picks - 2, 0)
    step = count / (inner + 1)
    idx = {0, count - 1} | {int(round(step * (i + 1))) for i in range(inner)}
    return sorted(i for i in idx if 0 <= i < count)


def iter_rows(pop: dict[str, Any], *, shard: tuple[int, int] | None = None):
    """``(first_vu, count, kind, lo, stride, unit_size)`` per population row (``shard = (i, n)``: rows ``i::n`` -- one process per shard)."""
    names = {int(e["id"]): f"{e['fn']}|{e['form']}" for e in pop["kinds"]["entries"]}
    for ri, r in enumerate(pop["rows"]):
        if shard is not None and ri % shard[1] != shard[0]:
            continue
        first, count, _ru, k, lo, stride, usz = (int(x) for x in r)
        yield first, count, names[k], lo, stride, usz


def _ranges(vus: list[int]) -> list[list[int]]:
    """Sorted VU ints -> ``[[first, count], ...]`` runs."""
    out: list[list[int]] = []
    for v in vus:
        if out and out[-1][0] + out[-1][1] == v:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return out


def _merge_ranges(rs: list[list[int]]) -> list[list[int]]:
    rs = sorted(rs)
    out: list[list[int]] = []
    for f, c in rs:
        if out and out[-1][0] + out[-1][1] >= f:
            out[-1][1] = max(out[-1][1], f + c - out[-1][0])
        else:
            out.append([f, c])
    return out


def families() -> dict[str, str | None]:
    from e2e.coverage_v1 import builders as B

    B.ensure_families() if hasattr(B, "ensure_families") else None
    out: dict[str, str | None] = {}
    for k in list(B.BUILDERS):
        out[k] = str(getattr(B.BUILDERS[k], "family", None))
    return out


def family_of(kind: str, fams: dict[str, str | None], epoch_family: str | None = None) -> str | None:
    """The family that expresses ``kind``: the ONE builder registry's (``BUILDERS[kind].family``), else -- for ``GemmCoordinate_v1{K}`` on a Serve_v1
    program -- the instance's ``epoch.family`` fixed circuit (``programs.gemm_coordinate(K)``, the path every TINY GEMM VU was proved through)."""
    from e2e.coverage_v1 import builders as B

    if kind in fams:
        return fams[kind]
    if kind.startswith("GemmCoordinate_v1{") and epoch_family:
        return f"{epoch_family} (fixed circuit gemm_coordinate)"
    try:
        b = B.BUILDERS[kind]  # base-kind lookup (`{...}` parameter block stripped)
    except Exception:  # noqa: BLE001
        rule = B.verifier_covered_rule(kind) if hasattr(B, "verifier_covered_rule") else None
        # [E7, NIGHT G1 -- a two-line out-of-lane edit flagged to E3/the integrator] a kind the VERIFIER checks by name without a proof
        # (`gather-equality/v1`: opened weights word == committed output); E9's entry point does the check, this only classifies it
        return f"verifier ({rule})" if rule else None
    return str(getattr(b, "family", None))


# --- the capture profile of verity_vllm's step-1 boundary (what the prover commits per position), read from the D7 record 05:30Z ------------
#: Per position t of a Serve_v1 request: the op outputs committed in step 1 (`p=<profile>/r=<rid>/t=<t>/ph=<prefill|decode>/l=<l>/op=<op>/port=<k>`).
#: MEASURED on the D7 record `V5-TINY-bex0-inc` (profile `65d138443752`, 30 layers): 362 ids per prefill position, 364 per token-producing position.
PROFILE_TOP = (("embed", "embed_tokens.embedding", 0), ("norm", "norm.rmsnorm", 0), ("norm", "norm.rmsnorm", 1))
PROFILE_TOKEN = (("lm_head", "lm_head.gemm", 0), ("sampler", "sampler.select", 0))  # only where a token is produced (last prefill position, every decode step)
PROFILE_LAYER = (("input_layernorm.rmsnorm", 0), ("input_layernorm.rmsnorm", 1), ("self_attn.qkv_proj.gemm", 0), ("self_attn.rotary_emb.rope", 0),
                 ("self_attn.rotary_emb.rope", 1), ("self_attn.attn.attn", 0), ("self_attn.o_proj.gemm", 0), ("post_attention_layernorm.rmsnorm", 0),
                 ("post_attention_layernorm.rmsnorm", 1), ("mlp.gate_up_proj.gemm", 0), ("mlp.act_fn.act_mul", 0), ("mlp.down_proj.gemm", 0))
PROFILE_LAYER0_EXCLUDED = (("input_layernorm.rmsnorm", 1),)  # layer 0's residual input IS the embedding row (committed as l=embed), not a second copy
DEFAULT_PROFILE_ID = "65d138443752"


def synth_boundary_ids(profile_id: str, rid: str, LP: int, STEPS: int, n_layers: int, row_gates=None) -> dict[str, list[str]]:
    """[PROVISIONAL until a record / E6's pin spells it] The step-1 committed id lists of one request from the capture profile TEMPLATE above:
    ``ru/<rid>/000000`` = prefill positions t = 0..LP-1, ``ru/<rid>/<k>`` (k = 1..STEPS) = decode position t = LP+k-1.  With ``row_gates`` (the
    program's identity table) an id the program has no counterpart for is DROPPED and counted -- the program decides, the template proposes."""
    out: dict[str, list[str]] = {}
    dropped = 0
    for step in range(STEPS + 1):
        ts = list(range(LP)) if step == 0 else [LP + step - 1]
        ids: list[str] = []
        for t in ts:
            ph = "prefill" if step == 0 else "decode"
            pre = f"p={profile_id}/r={rid}/t={t}/ph={ph}"
            cands = [f"{pre}/l={l}/op={op}/port={k}" for l, op, k in PROFILE_TOP]
            for li in range(n_layers):
                for op, k in PROFILE_LAYER:
                    if li == 0 and (op, k) in PROFILE_LAYER0_EXCLUDED:
                        continue
                    cands.append(f"{pre}/l={li}/op=layers.{li}.{op}/port={k}")
            if step > 0 or t == LP - 1:
                cands += [f"{pre}/l={l}/op={op}/port={k}" for l, op, k in PROFILE_TOKEN]
            for cid in cands:
                # production commits the per-position final norm at unsampled prefill positions too (6 ids on TINY, `NotInProgram`): they are in the
                # list as the record spells them; `BoundaryIndex` maps what the program holds and drops the rest, exactly as with a record's list
                if row_gates is not None and row_gates(cid) is None:
                    dropped += 1
                ids.append(cid)
        out[f"ru/{rid}/{step:06d}"] = ids
    out["_not_in_program"] = [str(dropped)]  # type: ignore[assignment]
    return out


def build_oracle(instance: dict[str, Any], pop: dict[str, Any], record: Path | None, *, request_id: str | None = None):
    """The verifier's position oracle over the pinned program + declared population + the record's committed id lists (E2 layout by the pin).
    Without a record (a NEW instance nobody served yet: G3's / G4's): the step-1 id lists are SYNTHESISED from the capture-profile template
    (:func:`synth_boundary_ids`, PROVISIONAL, validated against the D7 record on TINY) and weights words are spelled by tensor id + offset."""
    from e2e.verifier_v3 import expected as X
    from e2e.verifier_v3 import locator as L

    ident = instance["program"]["identity"]
    program, C, problems = L.build_pinned_program(ident, expected_digest=pop["program_digest"], expected_gates=pop["gates"])
    if problems:
        raise SystemExit(f"REFUSED: pinned program: {problems}")
    if instance["partition"]["partition_digest"] != pop["partition_digest"]:
        raise SystemExit(f"REFUSED: population partition_digest {pop['partition_digest'][:12]} != instance {instance['partition']['partition_digest'][:12]}")
    loc = L.make_locator(program, C)
    L.memoize_returns(True)  # [item 3] per-function-id return offsets (exact; 10x on the step-1 index build)
    ids: dict[str, list[str]] = {}
    synth: dict[str, Any] | None = None
    if record is not None:
        sel = record / "adapter/selective"
        comm = json.loads((sel / "commitments.json").read_text())
        pi = json.loads((record / "protocol_instance.json").read_text())
        if ident != pi["program"]["identity"]:
            raise SystemExit("REFUSED: the record's program identity is not the instance's")
        rids = sorted({t.split("/")[1] for t in comm["trees"] if t.startswith("ru/")})
        rid = request_id or rids[0]
        for t in comm["trees"]:
            if t.startswith(f"ru/{rid}/"):
                _, r, step = t.split("/")
                ids[t] = [e["canonical_id"] for e in json.loads((sel / "ru" / r / step / "index.json").read_text())["entries"]]
        # the record's capture reader names the weights leaves (`Deriver.reader` -- the same object `coverage_inputs` hands the oracle); without a weights
        # store every table reference refuses NO_WEIGHTS_LAYOUT by name (a census verdict, not a stand-in)
        reader = X.Deriver(sel, comm, recompute_leaves=False).reader
        oracle = X.PositionOracle(program, loc, population=pop, committed_ids=ids, commitments=comm, interior_layout=instance["commitments"]["interior_layout"], reader=reader)
    else:
        profile_id = str((instance.get("commitments") or {}).get("profile_id") or instance.get("profile_id") or DEFAULT_PROFILE_ID)
        rid = request_id or "00000000-0000-5000-8000-000000000000"
        n_layers = int(getattr(getattr(program, "layout", None), "NL", 0) or C.get("num_hidden_layers") or len(C.get("layers", [])) or 0)
        if not n_layers:
            raise SystemExit("REFUSED: cannot read the layer count (layout.NL / C.num_hidden_layers) for the boundary template")
        ids = synth_boundary_ids(profile_id, rid, int(ident["LP"]), int(ident["STEPS"]), n_layers, L.vllm_row_gates(loc))
        dropped = int(ids.pop("_not_in_program")[0])
        synth = {"label": "PROVISIONAL", "profile_id": profile_id, "n_layers": n_layers, "ids_per_step": {t: len(v) for t, v in ids.items()}, "not_in_program": dropped,
                 "rule": "synth_boundary_ids: capture-profile template MEASURED on the D7 record (id lists byte-equal per step on TINY, 05:40Z); `BoundaryIndex` maps what the program holds"}

        def weight_word(g: int):
            Lw = loc.locate_input(int(g))
            return L.GatePosition(int(g), "input", "weights", str(Lw["canonical_id"]), (1,), int(Lw["offset"]), 16)

        oracle = X.PositionOracle(program, loc, population=pop, committed_ids=ids, commitments={"profile_id": profile_id, "trees": {}},
                                  interior_layout=instance["commitments"]["interior_layout"], profile_id=profile_id, adapter={"weight_word": weight_word})
    oracle.synth_boundary = synth  # type: ignore[attr-defined]
    return oracle, program, rid, C


def _pick(vu: int, p: float) -> bool:
    """Deterministic stratified pick: the VU's hash falls below ``p`` (reproducible across runs, spread over the whole population)."""
    return p >= 1.0 or ((vu * 2654435761) & 0xFFFFFFFF) / 4294967296.0 < p


def run(instance_path: Path, population_path: Path, record: Path | None, *, limit: int | None, out: Path, budget_min: float | None, offset: int = 0,
        predicate: str = "shape", prior: Path | None = None, sample_per_class: int = 500, full_class_max: int = 3000, slow_class_cap: int = 120,
        predicate_budget_min: float | None = None, verify_per_shape: int = 3, rows: str = "all", picks: int = 6, shard: tuple[int, int] | None = None,
        request_id: str | None = None) -> dict[str, Any]:
    """The census.  ``predicate``: ``shape`` (DEFAULT, integrator 03:06Z: E7's ``can_express`` is identity-structural -- ONE call per distinct
    :func:`shape_key`, memoised, attributed to every VU of the key; ``verify_per_shape`` further stratified members of each key are also evaluated and a
    disagreement makes the key ``MIXED``), ``none`` (statement half only), ``sampled`` (E7's exact-path predicate on every VU of a class with ≤ ``full_class_max``
    members and on a deterministic stratified sample of ``sample_per_class`` VUs of every larger class -- class = (kind, operand shape); the prior
    pass-1 census gives the class counts; a class whose predicate costs > 100 ms/VU (E4's table kinds) is capped at ``slow_class_cap`` samples), or
    ``all`` (every VU -- ~15 ms/VU = 20 h for TINY; a pod job).  The predicate is E7's exact prover path and IDENTITY-ONLY: its verdict is a function of
    the statement's structure -- kind, operand count and order, origins, literal ids, aliasing -- which is what the class key spells, so a class's
    verdict is inferred for its unsampled members and labelled ``inferred``; a class whose samples disagree is ``MIXED`` and every member is uncovered."""
    instance = json.loads(instance_path.read_text())
    pop = json.loads(population_path.read_text())
    tables = (instance.get("commitments") or {}).get("tables")
    t0 = time.perf_counter()
    oracle, program, rid, _C = build_oracle(instance, pop, record, request_id=request_id)
    res = oracle.resolver(rid)
    fams = families()
    t_build = time.perf_counter() - t0
    declared: dict[str, Any] = {}
    ct = instance_path.parent / "coverage_table.json"
    if ct.exists():
        for k, v in (json.loads(ct.read_text()).get("kinds") or {}).items():
            declared[k] = {"status": v.get("status"), "position": v.get("position")}
    prior_counts: dict[tuple[str, str], int] = {}
    if prior is not None and prior.exists():
        for a in json.loads(prior.read_text()).get("by_kind_shape_verdict") or []:
            prior_counts[(a["kind"], a["shape"])] = prior_counts.get((a["kind"], a["shape"]), 0) + int(a["count"])
    agg: dict[tuple[str, str, str], dict[str, Any]] = {}
    cls: dict[tuple[str, str], dict[str, Any]] = {}  # (kind, shape) -> predicate sampling state
    verdicts: Counter = Counter()
    per_kind: dict[str, Counter] = defaultdict(Counter)
    aliased: Counter = Counter()
    members: dict[tuple[str, str], list[list[int]]] = defaultdict(list)  # class -> VU ranges [first, count] (exact membership, compact at any N_V)
    stmt_refused: list[list[int]] = []
    n = 0  # VUs accounted (evaluated + attributed)
    n_eval = 0  # VUs whose statement the oracle derived in this process
    n_rows = 0
    nonuniform_rows = 0
    n_pred = 0
    t_pred = 0.0
    t1 = time.perf_counter()
    last = t1
    stop = False
    epoch_family = (instance.get("epoch") or {}).get("family")

    def account(vu: int, kind: str, lo: int, hi: int, row: dict[str, Any], weight: int) -> None:
        """Book one evaluated VU (`weight` = 1) or a row's attributed run of `weight` members starting at `vu` with this member's verdicts."""
        nonlocal n, n_pred, t_pred
        ck = (kind, row["key"]) if predicate == "shape" else (kind, row["shape"])
        c = cls.get(ck)
        if c is None:
            pc = prior_counts.get((kind, row["shape"]))
            c = cls[ck] = {"kind": kind, "shape": row["shape"], "key": row["key"], "count": 0, "sampled": 0, "verdicts": Counter(), "examples": {}, "secs": 0.0,
                           "prior_count": pc, "p": (1.0 if (pc is None or pc <= full_class_max) else min(1.0, sample_per_class / pc)), "seen_first": 0, "rules": row.get("rules", []),
                           "stmt_verdict": row["verdict"]}
        c["count"] += weight
        members[ck].append([vu, weight])
        st = row.pop("_statement", None)
        if row["verdict"].startswith("E_EXPECTATION_"):
            stmt_refused.append([vu, weight])
        elif predicate == "shape" and st is not None:
            # one call per distinct key (the first VU of the key), plus `verify_per_shape` stratified members as the guard against a dependence
            # outside the key (p = 1e-4 -> ~ every 10,000th member; a key with < 10k members gets its first few)
            want = c["sampled"] == 0 or (c["sampled"] < 1 + verify_per_shape and (_pick(vu, 1e-4) or (c["prior_count"] or 10**9) < 4 * 10**4 and c["count"] % 97 == 0))
            if want:
                unit = {"vu_index": vu, "kind": kind, "request_id": rid, "gate_interval": [lo, hi]}
                tp = time.perf_counter()
                if isolate_kind(kind):
                    v = can_express_isolated(instance_path, population_path, record, vu, kind, lo, hi, rid)
                    c["isolated"] = True
                else:
                    v = can_express(st, unit, program=program, tables=tables)
                dt = time.perf_counter() - tp
                c["secs"] += dt
                t_pred += dt
                n_pred += 1
                c["sampled"] += 1
                c["verdicts"][v] += 1
                c["examples"].setdefault(v, vu)
                row["express"] = v
        elif predicate not in ("none", "shape") and st is not None:
            want = predicate == "all" or c["seen_first"] < 20 or (_pick(vu, c["p"]) and (c["prior_count"] is not None or c["sampled"] < max(full_class_max, sample_per_class)))
            if want and c["sampled"] >= slow_class_cap and c["secs"] / max(c["sampled"], 1) > 0.1:
                want = False  # a slow class (E4's table builders, 0.2-0.4 s each): capped
            if want and predicate_budget_min is not None and t_pred > predicate_budget_min * 60:
                want = False
            if want:
                c["seen_first"] += 1
                unit = {"vu_index": vu, "kind": kind, "request_id": rid, "gate_interval": [lo, hi]}
                tp = time.perf_counter()
                v = can_express(st, unit, program=program, tables=tables)
                dt = time.perf_counter() - tp
                c["secs"] += dt
                t_pred += dt
                n_pred += 1
                c["sampled"] += 1
                c["verdicts"][v] += 1
                c["examples"].setdefault(v, vu)
                row["express"] = v
        key = (kind, row["shape"], row["verdict"])
        a = agg.get(key)
        if a is None:
            a = agg[key] = {"kind": kind, "shape": row["shape"], "verdict": row["verdict"], "family": row.get("family"), "count": 0, "example_vu": vu,
                            "example_interval": [lo, hi]}
            if "error" in row:
                a["example_error"] = row["error"]
        a["count"] += weight
        verdicts[row["verdict"]] += weight
        per_kind[kind][row["verdict"]] += weight
        if row.get("aliased"):
            aliased[kind] += weight
        n += weight

    def tick(vu: int) -> bool:
        nonlocal last
        now = time.perf_counter()
        if now - last > 30:
            last = now
            import resource

            print(f"  {n} VUs ({n_eval} evaluated, rows {n_rows}, non-uniform {nonuniform_rows})  {n_eval / (now - t1):.0f} eval/s {n / (now - t1):.0f} VU/s  "
                  f"predicate {n_pred} calls {t_pred:.0f}s  classes {len(cls)}  maxrss {maxrss_gb():.2f} GB", file=sys.stderr, flush=True)
        if budget_min is not None and (now - t1) > budget_min * 60:
            print(f"  time budget {budget_min} min reached after {n} VUs (vu {vu}); stopping", file=sys.stderr)
            return True
        return False

    def evaluate(vu: int, kind: str, lo: int, hi: int) -> dict[str, Any]:
        nonlocal n_eval
        n_eval += 1
        row = vu_verdict(res, program, vu, kind, lo, hi, family_of(kind, fams, epoch_family))
        if "rules" not in row:
            # a row that skipped the statement half (E7's verifier-covered fast path: `family = "verifier (<rule>)"`, Gather under
            # gather-equality/v1) is attributed to the NAMED rule the family spells, so the residue / inventory read by rule everywhere
            fam = str(row.get("family") or "")
            row["rules"] = [fam[len("verifier ("):-1]] if fam.startswith("verifier (") and fam.endswith(")") else rules_of(kind, [], [], RULES)
        return row

    for first, count, kind, lo, stride, usz in iter_rows(pop, shard=shard):
        if stop:
            break
        n_rows += 1
        if first + count <= offset:
            continue
        if limit is not None and first >= offset + limit:
            break
        if rows == "sampled" and first >= offset and (limit is None or first + count <= offset + limit):
            idx = row_picks(count, picks)
            got = [(j, evaluate(first + j, kind, lo + j * stride, lo + j * stride + usz)) for j in idx]
            sig = {(r["key"], r["verdict"], r["shape"]) for _j, r in got}
            if len(sig) == 1 and len(idx) < count:
                # uniform row: the evaluated members are booked as themselves, the gaps between them attributed to the row's verdict
                prev = -1
                for j, r in got:
                    if j > prev + 1:
                        account(first + prev + 1, kind, lo + (prev + 1) * stride, lo + (prev + 1) * stride + usz, dict(r, _statement=None), j - prev - 1)
                    account(first + j, kind, lo + j * stride, lo + j * stride + usz, r, 1)
                    prev = j
                stop = tick(first + count - 1)
                continue
            if len(sig) > 1:
                nonuniform_rows += 1
            done = {j: r for j, r in got} if len(sig) == 1 else {}
            for j in range(count):
                r = done.get(j) or evaluate(first + j, kind, lo + j * stride, lo + j * stride + usz)
                account(first + j, kind, lo + j * stride, lo + j * stride + usz, r, 1)
            stop = tick(first + count - 1)
            continue
        for j in range(count):
            vu = first + j
            if vu < offset:
                continue
            if limit is not None and vu >= offset + limit:
                stop = True
                break
            account(vu, kind, lo + j * stride, lo + j * stride + usz, evaluate(vu, kind, lo + j * stride, lo + j * stride + usz), 1)
            if tick(vu):
                stop = True
                break
    secs = time.perf_counter() - t1
    rate = n / secs if secs else None
    total = int(pop["vu_count"])
    # --- class verdicts: E7's predicate over the class's samples; inferred for the unsampled members ---------------------------------------
    classes: list[dict[str, Any]] = []
    final: Counter = Counter()  # every VU's final verdict (statement refusal | class predicate verdict)
    exact: Counter = Counter()  # the sampled VUs' exact verdicts
    uncovered_by_reason: dict[str, list[int]] = defaultdict(list)
    for ck, c in cls.items():
        vs = c["verdicts"]
        if c.get("stmt_verdict", "ok") != "ok":
            cv = str(c["stmt_verdict"])  # the statement half refused (E_NO_BUILDER / E_EXPECTATION_<code>): the class is uncovered by that name
        elif c["sampled"] == 0:
            cv = "PENDING(not sampled)" if predicate != "none" else "PENDING(E7.can_express not run)"
        elif len(vs) == 1:
            cv = next(iter(vs))
        else:
            cv = "MIXED(" + ",".join(f"{k}:{v}" for k, v in sorted(vs.items())) + ")"
        entry = {"kind": c["kind"], "shape": c["shape"], "key": c.get("key"), "rules": c.get("rules", []), "statement_verdict": c.get("stmt_verdict", "ok"), "count": c["count"], "prior_count": c["prior_count"], "sampled": c["sampled"],
                 "sampling": ("one call per shape key + verification members" if predicate == "shape" else "all members" if c["sampled"] >= c["count"] else f"stratified p={c['p']:.4g}" + (" (slow class, capped)" if c["sampled"] >= slow_class_cap and c["secs"] / max(c["sampled"], 1) > 0.1 else "")),
                 "predicate_verdicts": dict(vs), "class_verdict": cv, "isolated_subprocess": bool(c.get("isolated")), "inferred_members": c["count"] - c["sampled"], "example_vu_by_verdict": dict(c["examples"]),
                 "predicate_ms_per_vu": (1000 * c["secs"] / c["sampled"]) if c["sampled"] else None}
        classes.append(entry)
        for k, v in vs.items():
            exact[k] += v
        final[cv] += c["count"]
        if cv != "ok" and not (predicate == "none" and cv.startswith("PENDING")):
            uncovered_by_reason[cv].extend(members[ck])
    unc_ranges = _merge_ranges([r for rs in uncovered_by_reason.values() for r in rs] + stmt_refused)
    n_unc = sum(c for _f, c in unc_ranges)

    def _unc_entry(rs: list[list[int]]) -> dict[str, Any]:
        rs = _merge_ranges(rs)
        cnt = sum(c for _f, c in rs)
        e: dict[str, Any] = {"count": cnt, "ranges": rs if len(rs) <= 20_000 else rs[:200], "n_ranges": len(rs),
                             "sha256_of_ranges": hashlib.sha256(json.dumps(rs).encode()).hexdigest()}
        if cnt <= 2_000_000:  # the sorted-int digest of pass 2 (comparable across modes); an exact list when small
            lst = [f + i for f, c in rs for i in range(c)]
            e["sha256_of_sorted_list"] = hashlib.sha256(json.dumps(lst).encode()).hexdigest()
            if cnt <= 10_000:
                e["vus"] = lst
        return e

    unc_doc: dict[str, Any] = {**_unc_entry(unc_ranges), "by_reason": {k: _unc_entry(v) for k, v in uncovered_by_reason.items()}}
    unc_all = range(n_unc)  # count only from here on
    doc = {
        "schema": "veritor/e3/expressibility-census/v2", "instance_id": instance["instance_id"], "partition_digest": pop["partition_digest"],
        "tree": tree_stamp(),
        "population": {"file": str(population_path), "content_sha256": pop.get("content_sha256"), "vu_count": total, "kinds": len(pop["kinds"]["entries"])},
        "record_for_committed_ids": str(record) if record is not None else None, "boundary_ids": ("record" if record is not None else getattr(oracle, "synth_boundary", None)),
        "request_id": rid, "rows_mode": rows, "picks_per_row": picks if rows == "sampled" else None, "shard": list(shard) if shard else None,
        "rows_seen": n_rows, "nonuniform_rows": nonuniform_rows, "vus_evaluated": n_eval, "interior_layout_rule": instance["commitments"]["interior_layout"].get("rule"),
        "tables_pinned": sorted(tables) if tables else [],
        "coverage": {"vus_checked": n, "first_vu": offset, "last_vu": (offset + n - 1) if n else None, "of_population": total, "complete": n + offset >= total},
        "totals": {"vus": n, "expressible_statement": verdicts.get("ok", 0), "statement_not_derived": {k: v for k, v in verdicts.items() if k != "ok"},
                   "final_verdict_all_vus": dict(final), "predicate_exact_on_sampled_vus": dict(exact), "predicate_calls": n_pred, "predicate_seconds": t_pred,
                   "uncovered": len(unc_all), "aliased_vus_by_kind": dict(aliased)},
        "predicate": {"mode": predicate, "function": "e2e.coverage_v1.builders.can_express(statement, unit, program=pinned program, tables=instance commitments.tables)",
                      "distinct_shape_keys": len(cls), "verify_per_shape": verify_per_shape,
                      "shape_key": "kind | operand tokens in order (<origin><width> | L:<Const id>) | alias wiring | output tokens -- never a word's name or bits",
                      "sample_per_class": sample_per_class, "full_class_max": full_class_max, "slow_class_cap": slow_class_cap, "prior": str(prior) if prior else None},
        "per_kind": {k: dict(v) for k, v in sorted(per_kind.items())},
        "declared_by_e6": declared,
        "classes": sorted(classes, key=lambda a: (-a["count"], a["kind"])),
        "by_kind_shape_verdict": sorted(agg.values(), key=lambda a: (-a["count"], a["kind"])),
        "uncovered": unc_doc,
        "measured": {"vus_per_s": rate, "evaluated_per_s": (n_eval / secs) if secs else None, "seconds": secs, "oracle_build_s": t_build, "est_full_population_min": (total / rate / 60) if rate else None,
                     "maxrss_gb": maxrss_gb(),
                     "label": "MEASURED" if n else "NOT RUN"},
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": ["statement half: the verifier's GateResolver.statement per VU (E3); predicate half: E7's can_express, exact prover path, identity-only",
                  "class = (kind, operand shape); the predicate's verdict is inferred for a class's unsampled members (labelled inferred_members); MIXED classes are uncovered whole",
                  "aliasing is by committed IDENTITY (tree, leaf, offset) of each operand position, never by value (D32.1)",
                  f"wide units (> {WIDE} operands: Gather rows, the whole-scan logits row) count input identities vectorised; every non-input operand is resolved"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, default=str))
    return doc


#: The tree the census was RUN ON -- the gate's verdict line is "census 0 on <hash>" (E10's frozen tree): `--tree <label>` names it when the run
#: is on a synced copy without `.git`; the git HEAD of the census's own repository is recorded whenever one is reachable.  Set by `main`.
TREE_LABEL: str | None = None


def tree_stamp() -> dict[str, Any]:
    head = None
    try:
        r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        head = r.stdout.strip() or None
        st = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=20).stdout
        # code only: a modified report / table under out/ does not make the CODE tree dirty
        dirty = [ln for ln in st.splitlines() if ln.strip() and not ln[3:].startswith("out/")]
    except Exception:  # noqa: BLE001
        dirty = None
    return {"label": TREE_LABEL, "git_head": head, "dirty": (bool(dirty) if head else None), "dirty_paths": (dirty[:10] if dirty else [])}


def merge_shards(paths: list[Path], out: Path) -> dict[str, Any]:
    """Merge shard census JSONs (``--shard i/n``) into one document: counts add, classes / aggregates merge by key, uncovered ranges union."""
    docs = [json.loads(p.read_text()) for p in paths]
    base = docs[0]
    cls: dict[tuple[str, str], dict[str, Any]] = {}
    agg: dict[tuple[str, str, str], dict[str, Any]] = {}
    unc_by: dict[str, list[list[int]]] = defaultdict(list)
    tot: Counter = Counter()
    final: Counter = Counter()
    exact: Counter = Counter()
    per_kind: dict[str, Counter] = defaultdict(Counter)
    aliased: Counter = Counter()
    snd: Counter = Counter()
    n_pred = 0
    t_pred = 0.0
    secs = 0.0
    n_eval = 0
    rows_seen = 0
    nonuni = 0
    maxrss = 0.0
    for d in docs:
        for c in d["classes"]:
            k = (c["kind"], c.get("key") or c["shape"])
            e = cls.get(k)
            if e is None:
                cls[k] = e = dict(c)
                e["predicate_verdicts"] = Counter(c["predicate_verdicts"])
            else:
                e["count"] += c["count"]
                e["sampled"] += c["sampled"]
                e["inferred_members"] += c["inferred_members"]
                e["predicate_verdicts"].update(c["predicate_verdicts"])
                for v, vu in c["example_vu_by_verdict"].items():
                    e["example_vu_by_verdict"].setdefault(v, vu)
        for a in d["by_kind_shape_verdict"]:
            k = (a["kind"], a["shape"], a["verdict"])
            if k in agg:
                agg[k]["count"] += a["count"]
            else:
                agg[k] = dict(a)
        for r, e in d["uncovered"]["by_reason"].items():
            unc_by[r].extend(e.get("ranges") or [])
        t = d["totals"]
        tot["vus"] += t["vus"]
        tot["expressible_statement"] += t["expressible_statement"]
        snd.update(t["statement_not_derived"])
        exact.update(t["predicate_exact_on_sampled_vus"])
        n_pred += t["predicate_calls"]
        t_pred += t["predicate_seconds"]
        for k, v in d["per_kind"].items():
            per_kind[k].update(v)
        aliased.update(t["aliased_vus_by_kind"])
        secs = max(secs, d["measured"]["seconds"])
        n_eval += d.get("vus_evaluated", 0)
        rows_seen += d.get("rows_seen", 0)
        nonuni += d.get("nonuniform_rows", 0)
        maxrss = max(maxrss, d["measured"].get("maxrss_gb") or 0)
    classes = []
    for e in cls.values():
        vs = e["predicate_verdicts"]
        pend = "PENDING(E7.can_express not run)" if (base.get("predicate") or {}).get("mode") == "none" else "PENDING(not sampled)"
        cv = str(e["statement_verdict"]) if e.get("statement_verdict", "ok") != "ok" else pend if e["sampled"] == 0 else (next(iter(vs)) if len(vs) == 1 else "MIXED(" + ",".join(f"{k}:{v}" for k, v in sorted(vs.items())) + ")")
        e["class_verdict"] = cv
        e["predicate_verdicts"] = dict(vs)
        final[cv] += e["count"]
        classes.append(e)
    if (base.get("predicate") or {}).get("mode") == "none":
        unc_by = {k: v for k, v in unc_by.items() if not k.startswith("PENDING")}
    unc_ranges = _merge_ranges([r for rs in unc_by.values() for r in rs])
    n_unc = sum(c for _f, c in unc_ranges)
    doc = dict(base)
    doc["predicate"] = dict(base.get("predicate") or {}, distinct_shape_keys=len(cls))
    doc.update(shard=None, shards_merged=[str(p) for p in paths], rows_seen=rows_seen, nonuniform_rows=nonuni, vus_evaluated=n_eval,
               coverage={"vus_checked": tot["vus"], "of_population": base["coverage"]["of_population"], "complete": tot["vus"] >= base["coverage"]["of_population"]},
               totals={"vus": tot["vus"], "expressible_statement": tot["expressible_statement"], "statement_not_derived": dict(snd), "final_verdict_all_vus": dict(final),
                       "predicate_exact_on_sampled_vus": dict(exact), "predicate_calls": n_pred, "predicate_seconds": t_pred, "uncovered": n_unc, "aliased_vus_by_kind": dict(aliased)},
               per_kind={k: dict(v) for k, v in sorted(per_kind.items())}, classes=sorted(classes, key=lambda a: (-a["count"], a["kind"])),
               by_kind_shape_verdict=sorted(agg.values(), key=lambda a: (-a["count"], a["kind"])),
               uncovered={"count": n_unc, "n_ranges": len(unc_ranges), "ranges": unc_ranges if len(unc_ranges) <= 20_000 else unc_ranges[:200],
                          "sha256_of_ranges": hashlib.sha256(json.dumps(unc_ranges).encode()).hexdigest(),
                          "by_reason": {r: {"count": sum(c for _f, c in _merge_ranges(v)), "n_ranges": len(_merge_ranges(v)), "sha256_of_ranges": hashlib.sha256(json.dumps(_merge_ranges(v)).encode()).hexdigest()} for r, v in unc_by.items()}},
               measured={"vus_per_s": tot["vus"] / secs if secs else None, "evaluated_per_s": n_eval / secs if secs else None, "seconds": secs, "seconds_is": "wall of the slowest shard", "maxrss_gb": maxrss, "maxrss_is": "max over shards",
                         "label": "MEASURED", "shards": len(docs)},
               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, default=str))
    return doc


# --- CENSUS.md -----------------------------------------------------------------------------------------------------------------------------


def _aliased_shape(shape: str) -> bool:
    """The class's alias pattern repeats a committed identity letter (`a,a` / `a,a,b`); `L` (literal) and `.`/summaries never count."""
    if "|alias=" not in shape:
        return False
    pat = [x for x in shape.split("|alias=")[1].split("|")[0].split(",") if x and x != "L" and x != "-"]
    return len(pat) > 1 and len(set(pat)) < len(pat) and all(len(x) <= 2 and x.isalpha() for x in pat)


def _has_literal(shape: str) -> bool:
    """A literal at an operand position: `pad0>0` or a named `lit=Const…`."""
    if "|lit=Const" in shape:
        return True
    if "|pad0=" in shape:
        return not shape.split("|pad0=")[1].split("|")[0].startswith("0")
    return False


def write_md(doc: dict[str, Any], path: Path) -> None:
    """The short human census: residue first (0 uncovered or the exact residue by reason), then the classes the integrator asked for by name."""
    t = doc["totals"]
    unc = doc["uncovered"]
    cls = doc["classes"]
    L: list[str] = []
    L.append(f"# Expressibility census — `{doc['instance_id']}` (E3, {doc['utc']})\n")
    mode = (doc.get("predicate") or {}).get("mode")
    if mode == "none":
        # statement half only (a new instance, no builder predicate run): the residue that matters is what the ORACLE could not derive
        nd = t.get("statement_not_derived") or {}
        n_nd = sum(nd.values())
        L.append(f"**STATEMENT HALF ONLY (predicate not run): the verifier's expected statement derives for {t['expressible_statement']:,} of {t['vus']:,} VUs"
                 + (f"; NOT derived for {n_nd:,}: {dict(nd)}" if n_nd else "; 0 not derived") + ".** E7's `can_express` verdicts are PENDING for every class (run with `--predicate shape`).\n")
        L.append("Per (kind, operand shape) inventory below -- the by-shape answer to \"does this checkpoint reuse the implemented operations?\": a shape whose key "
                 "(kind, operand origins/widths, literal ids, alias wiring, output widths) also occurs in the TINY census is a REUSED shape; a kind absent from the builder registry is `E_NO_BUILDER`.\n")
    elif unc["count"] == 0:
        L.append("**RESIDUE: 0 uncovered VUs.** Every VU's expected statement derives and E7's `can_express` answers `ok` for its class.\n")
    else:
        L.append(f"**RESIDUE: the census does NOT read 0 uncovered — {unc['count']:,} of {t['vus']:,} VUs ({100 * unc['count'] / t['vus']:.2f} %) are not expressible today, by reason:**\n")
        for reason, r in sorted(unc["by_reason"].items(), key=lambda kv: -kv[1]["count"]):
            kinds, rules = Counter(), Counter()
            for c in cls:
                if c["class_verdict"] == reason:
                    kinds[c["kind"]] += c["count"]
                    rules[", ".join(f"`{x}`" for x in (c.get("rules") or [])) or "no named rule"] += c["count"]
            digest = (f" (sorted VU list sha256 `{r['sha256_of_sorted_list'][:16]}…`)" if r.get("sha256_of_sorted_list")
                      else f" (VU ranges sha256 `{r.get('sha256_of_ranges', '-')[:16]}…`, {r.get('n_ranges')} ranges)")
            by_rule = "; by the NAMED rule the shape invokes: " + ", ".join(f"{k} {v:,}" for k, v in rules.most_common())
            L.append(f"* `{reason}` — **{r['count']:,} VUs** — " + ", ".join(f"`{k}` {v:,}" for k, v in kinds.most_common()) + digest + by_rule)
        L.append("\nThe user decides on this residue before any regeneration (D32.2).\n")
    L.append("## Totals\n")
    L.append(f"* Population: {doc['population']['vu_count']:,} VUs, {doc['population']['kinds']} kinds, `partition_digest {doc['partition_digest'][:12]}…`; checked {doc['coverage']['vus_checked']:,} (complete: {doc['coverage']['complete']}).")
    L.append(f"* Statement half (E3's oracle): derived for {t['expressible_statement']:,}; not derived: {t['statement_not_derived'] or 'none'}.")
    L.append(f"* Predicate half (E7's `can_express`, exact prover path, identity-only): {t['predicate_calls']:,} VUs evaluated in {t['predicate_seconds']:.0f} s; exact verdicts on them: {dict(t['predicate_exact_on_sampled_vus'])}.")
    L.append(f"* Final verdict over ALL VUs (a class's verdict inferred for its unsampled members): " + ", ".join(f"`{k}` {v:,}" for k, v in sorted(t["final_verdict_all_vus"].items(), key=lambda kv: -kv[1])) + ".")
    ok_rules = Counter()
    for c in cls:
        if c["class_verdict"] == "ok":
            ok_rules[", ".join(f"`{x}`" for x in (c.get("rules") or [])) or "plain (no named rule)"] += c["count"]
    if ok_rules:
        L.append("* `ok` by the NAMED rule the shape invokes: " + ", ".join(f"{k} {v:,}" for k, v in ok_rules.most_common()) + ".")
    L.append(f"* Predicate mode `{doc['predicate']['mode']}`: {doc['predicate'].get('distinct_shape_keys')} distinct shape keys (kind | operand tokens `<origin><width>` / `L:<Const id>` in order | alias wiring | output tokens); ONE `can_express` call per key + up to {doc['predicate'].get('verify_per_shape')} stratified verification members; a disagreement = `MIXED`. Rate {doc['measured']['vus_per_s']:.0f} VU/s, {doc['measured']['seconds'] / 60:.1f} min.\n")
    L.append("## Not expressible, by (kind, operand shape, reason)\n")
    L.append("| count | kind | operand shape | reason | rule | sampled | ms/VU |\n|---:|---|---|---|---|---:|---:|")
    for c in cls:
        if c["class_verdict"] != "ok":
            rules = ", ".join(f"`{x}`" for x in (c.get("rules") or [])) or "-"
            L.append(f"| {c['count']:,} | `{c['kind']}` | `{c['shape'][:90]}` | `{c['class_verdict']}` | {rules} | {c['sampled']} | {c['predicate_ms_per_vu'] or 0:.0f} |")
    def section(title: str, pred, note: str = "") -> None:
        L.append(f"\n## {title}\n")
        if note:
            L.append(note + "\n")
        L.append("| count | kind | operand shape | class verdict | sampled | inferred |\n|---:|---|---|---|---:|---:|")
        tot = Counter()
        for c in cls:
            if pred(c):
                L.append(f"| {c['count']:,} | `{c['kind']}` | `{c['shape'][:90]}` | `{c['class_verdict']}` | {c['sampled']} | {c['inferred_members']:,} |")
                tot[c["class_verdict"]] += c["count"]
        L.append("\nTotal: " + ", ".join(f"`{k}` {v:,}" for k, v in tot.most_common()))
    n_alias = sum(c["count"] for c in cls if _aliased_shape(c["shape"]))
    section("Aliased classes (D30 `aliased-body/v1`; identity, never value)", lambda c: _aliased_shape(c["shape"]), f"All {n_alias:,} must be `ok` or the reason is named.")
    section("PV padding units (`AmpereBF16TcDot16_v1|batch`; `prescribed-literal/v1` pads since wave 2, `E_INPUT_NOT_PROTECTED` before)", lambda c: c["kind"].startswith("AmpereBF16TcDot16"))
    section("FTZ / MUFU table kinds (E4's builders; `commitments.tables` pinned: " + ", ".join(doc.get("tables_pinned") or []) + ")",
            lambda c: c["kind"].split("_v1")[0] in ("MufuEx2Ftz", "MufuSqrtFtz", "Fa2InvSum", "F32AddFtz", "F32MulFtz", "F32FmaSubFtz", "GuardNegInfZero", "DivFullRcp", "DivFullScaleA", "F32Max"))
    section("`RsqrtApprox_v1|primitive`", lambda c: c["kind"].startswith("RsqrtApprox"))
    section("Gather (`GatherBf16x49152_v1|batch`, D31.3)", lambda c: c["kind"].startswith("GatherBf16x49152"))
    section("Prescribed-constant operands (a program literal at an operand position; E5 §4)", lambda c: _has_literal(c["shape"]) and not c["kind"].startswith("AmpereBF16TcDot16"))
    section("Everything else", lambda c: not c["kind"].startswith(("AmpereBF16TcDot16", "GatherBf16x49152", "RsqrtApprox")) and not _has_literal(c["shape"]) and not _aliased_shape(c["shape"]))
    L.append(f"\nSource: `{path.with_suffix('.json').name}` (pass 2), `…pass1.json` (statement half only). Tool: `e2e/verifier_v3/census.py`.\n")
    path.write_text("\n".join(L))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--instance", required=True)
    ap.add_argument("--population", required=True, help="the declared population file (v5 rows, a9/legacy-minus-inputs/v2)")
    ap.add_argument("--record", default=None, help="an archived record of this program (committed step-1 id lists for the positions); absent: the boundary ids are synthesised from the capture-profile template (PROVISIONAL)")
    ap.add_argument("--request-id", default=None)
    ap.add_argument("--rows", choices=("all", "sampled"), default="all", help="sampled: evaluate `--picks` members per population row, attribute uniform rows (exact fallback otherwise)")
    ap.add_argument("--picks", type=int, default=6)
    ap.add_argument("--shard", default=None, help="i/n: this process takes rows i::n (merge the shard JSONs with --merge)")
    ap.add_argument("--merge", nargs="*", default=None, help="merge shard census JSONs into --out (no evaluation)")
    ap.add_argument("--jobs", type=int, default=None, help="run N shard processes (rows i::N) in parallel and merge them into --out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--budget-min", type=float, default=None, help="stop after this many minutes (partial census, coverage.complete=false)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tree", default=None, help="the label of the tree this census runs on (E10's frozen hash) -- recorded in the output as `tree.label`")
    ap.add_argument("--predicate", choices=("shape", "none", "sampled", "all"), default="shape")
    ap.add_argument("--verify-per-shape", type=int, default=3)
    ap.add_argument("--prior", default=None, help="a pass-1 census JSON: class counts for the stratified sample")
    ap.add_argument("--sample-per-class", type=int, default=500)
    ap.add_argument("--full-class-max", type=int, default=3000)
    ap.add_argument("--slow-class-cap", type=int, default=120)
    ap.add_argument("--predicate-budget-min", type=float, default=None)
    ap.add_argument("--md", default=None, help="also write the short CENSUS.md here")
    a = ap.parse_args(argv)
    global TREE_LABEL
    TREE_LABEL = a.tree
    inst = json.loads(Path(a.instance).read_text())
    if a.merge is not None:
        doc = merge_shards([Path(x) for x in a.merge], Path(a.out))
        if a.md:
            write_md(doc, Path(a.md))
        print(json.dumps({"instance": doc["instance_id"], "vus": doc["totals"]["vus"], "final_verdict_all_vus": doc["totals"]["final_verdict_all_vus"], "uncovered": doc["totals"]["uncovered"]}, indent=1))
        return 0
    shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
    if a.jobs and a.jobs > 1:
        import subprocess

        out = Path(a.out) if a.out else Path(__file__).resolve().parents[3] / "out/night/e3/census" / inst["instance_id"] / "census.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        raw = list(argv if argv is not None else sys.argv[1:])
        base: list[str] = []
        skip = False
        for x in raw:  # drop --jobs/--out/--md and their values
            if skip:
                skip = False
                continue
            if x in ("--jobs", "--out", "--md"):
                skip = True
                continue
            if x.startswith(("--jobs=", "--out=", "--md=")):
                continue
            base.append(x)
        procs = []
        shards = []
        for i in range(a.jobs):
            so = out.with_name(out.stem + f".shard{i}of{a.jobs}.json")
            shards.append(so)
            log = open(so.with_suffix(".log"), "w")
            procs.append(subprocess.Popen([sys.executable, __file__, *base, "--shard", f"{i}/{a.jobs}", "--out", str(so)], stdout=log, stderr=subprocess.STDOUT))
        rc = [p.wait() for p in procs]
        if any(rc):
            print(f"shards failed: {rc}", file=sys.stderr)
            return 1
        doc = merge_shards(shards, out)
        if a.md:
            write_md(doc, Path(a.md))
        print(json.dumps({"instance": doc["instance_id"], "vus": doc["totals"]["vus"], "final_verdict_all_vus": doc["totals"]["final_verdict_all_vus"], "uncovered": doc["totals"]["uncovered"], "measured": doc["measured"], "out": str(out)}, indent=1))
        return 0
    out = Path(a.out) if a.out else Path(__file__).resolve().parents[3] / "out/eod/e3/census" / (inst["instance_id"] + ("" if a.limit is None else f".first{a.limit}") + (f".shard{shard[0]}of{shard[1]}" if shard else "") + ".json")
    doc = run(Path(a.instance), Path(a.population), Path(a.record) if a.record else None, limit=a.limit, out=out, budget_min=a.budget_min, offset=a.offset, predicate=a.predicate,
              prior=Path(a.prior) if a.prior else None, sample_per_class=a.sample_per_class, full_class_max=a.full_class_max, slow_class_cap=a.slow_class_cap,
              predicate_budget_min=a.predicate_budget_min, verify_per_shape=a.verify_per_shape, rows=a.rows, picks=a.picks, shard=shard, request_id=a.request_id)
    if a.md:
        write_md(doc, Path(a.md))
    t = doc["totals"]
    print(json.dumps({"instance": doc["instance_id"], "vus_checked": doc["coverage"]["vus_checked"], "expressible_statement": t["expressible_statement"],
                      "final_verdict_all_vus": t["final_verdict_all_vus"], "predicate_exact_on_sampled_vus": t["predicate_exact_on_sampled_vus"],
                      "uncovered": t["uncovered"], "measured": doc["measured"], "out": str(out)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
