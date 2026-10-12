"""Program correspondence across reference encodings (frontend rulings, §8.5).

    Gate -> exactly one Value<w>.
    Composite interfaces -> ordered collections of value references.
    Commitment positions -> physical authentication locations for those values,
                            not semantic components of gate outputs.

The program digest hashes the serialised descriptor, so two programs that denote the *same circuit* can have
different digests when a reference sequence is written differently (`Concat` of 128 one-leaf runs vs a
zero-stride `Strided`; an `Explicit` list vs the `Strided` it is). "Never demand digest equality across
different encodings" — this module is the test to use instead: definition by definition it checks the same
gates (intrinsic gate counts), the same producers (node forms, callee ids, argument/output types, batch axes
and counts) and the same **ordered reference sequences** (leaf by leaf when the encodings differ), and at the
root the same `S.out` (the root outputs). It never merges distinct produced values: `[z0, z1, z2]` and
`[z, z, z]` compare unequal leaf by leaf.

Verdicts per definition: `identical` (byte-equal encodings), `same-references` (different encodings, equal
leaf sequences), `mismatch`, `unverified` (a differing encoding too large to compare leaf by leaf under
`leaf_limit`). A program *corresponds* when nothing is `mismatch`/`unverified` and the definition sets agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verity.ml.circuits.codec import _spec_id, program_digest
from verity.ml.circuits.defs import PrimitiveDefinition, SpecializedDefinition
from verity.ml.circuits.refs import Refs


@dataclass
class DefinitionVerdict:
    id: str
    verdict: str  # identical | same-references | mismatch | unverified | missing-left | missing-right
    gates: int | None = None
    detail: str = ""

    def to_json(self) -> dict:
        return {"id": self.id, "verdict": self.verdict, "gates": self.gates, "detail": self.detail}


@dataclass
class CorrespondenceReport:
    corresponds: bool
    digest_left: str
    digest_right: str
    definitions: int
    identical: int
    same_references: int
    mismatches: list[DefinitionVerdict]
    unverified: list[DefinitionVerdict]
    same_reference_ids: list[str] = field(default_factory=list)
    root_out: str = ""  # identical | same-references | mismatch
    name_differences: list[str] = field(default_factory=list)  # call-name annotations that differ (not circuit)
    gates_left: int = 0
    gates_right: int = 0
    # `correspond_rebound` only: the definition pairing (left id, right id, verdict, the statics that differ) and the
    # definitions reachable on one side only (bound as statics but never called: part of the identity, not of the circuit)
    pairs: list[dict] = field(default_factory=list)
    unpaired_left: list[str] = field(default_factory=list)
    unpaired_right: list[str] = field(default_factory=list)

    @property
    def digest_equal(self) -> bool:
        return self.digest_left == self.digest_right

    @property
    def renamed(self) -> list[dict]:
        """The pairs whose two ids differ (a rebinding: name, version or binding record)."""
        return [p for p in self.pairs if p["left"] != p["right"]]

    def to_json(self) -> dict:
        out = {"corresponds": self.corresponds, "digest_equal": self.digest_equal, "digest_left": self.digest_left,
               "digest_right": self.digest_right, "gates_left": self.gates_left, "gates_right": self.gates_right,
               "definitions": self.definitions, "identical": self.identical, "same_references": self.same_references,
               "same_reference_ids": list(self.same_reference_ids), "root_out": self.root_out,
               "mismatches": [m.to_json() for m in self.mismatches], "unverified": [u.to_json() for u in self.unverified],
               "name_differences": list(self.name_differences)}
        if self.pairs or self.unpaired_left or self.unpaired_right:
            out.update({"pairs": list(self.pairs), "unpaired_left": list(self.unpaired_left), "unpaired_right": list(self.unpaired_right)})
        return out


def _reachable(root: SpecializedDefinition) -> dict[str, object]:
    """Definitions reachable from the root, by descriptor id (bodies are built on demand)."""
    out: dict[str, object] = {}

    def visit(fn) -> None:
        fid = _spec_id(fn)
        if fid in out:
            return
        out[fid] = fn
        if isinstance(fn, PrimitiveDefinition):
            return
        for n in fn.body.nodes:
            visit(n.fn)
        for v in fn.bindings.values():
            _visit_static(v, visit)

    visit(root)
    return out


def _visit_static(v, visit) -> None:
    if isinstance(v, (PrimitiveDefinition, SpecializedDefinition)):
        visit(v)
    elif isinstance(v, (list, tuple)):
        for x in v:
            _visit_static(x, visit)
    elif isinstance(v, dict):
        for x in v.values():
            _visit_static(x, visit)


def _refs_equal(a: Refs, b: Refs, leaf_limit: int) -> str:
    """identical | same-references | mismatch | unverified."""
    if a.count != b.count:
        return "mismatch"
    if a.to_json() == b.to_json():
        return "identical"
    if a.count > leaf_limit:
        return "unverified"
    for j in range(a.count):
        if a[j] != b[j]:
            return "mismatch"
    return "same-references"


def compare_bodies(ba, bb, *, leaf_limit: int = 1 << 22, fid: str = "", names: list[str] | None = None) -> DefinitionVerdict:
    """Same body? `identical` (byte-equal encodings), `same-references` (different encodings, same gates and
    reference sequences leaf by leaf), `mismatch`, or `unverified` (a differing encoding beyond `leaf_limit`).
    Call names are annotations: differences are collected into `names`, never a mismatch."""
    return _compare_bodies_(ba, bb, leaf_limit, names if names is not None else [], fid)


def _compare_bodies(fa: SpecializedDefinition, fb: SpecializedDefinition, leaf_limit: int, names: list[str], fid: str) -> DefinitionVerdict:
    return _compare_bodies_(fa.body, fb.body, leaf_limit, names, fid)


def _compare_bodies_(ba, bb, leaf_limit: int, names: list[str], fid: str, same_callee=None) -> DefinitionVerdict:
    """`same_callee(fa, fb) -> bool` replaces callee-id equality (default) — `correspond_rebound` passes the pairing."""
    if same_callee is None:
        same_callee = _same_id
    if ba.params != bb.params:
        return DefinitionVerdict(fid, "mismatch", None, f"params {ba.params!r} vs {bb.params!r}")
    if ba.ret_type != bb.ret_type:
        return DefinitionVerdict(fid, "mismatch", None, f"ret_type {ba.ret_type!r} vs {bb.ret_type!r}")
    if len(ba.nodes) != len(bb.nodes):
        return DefinitionVerdict(fid, "mismatch", None, f"{len(ba.nodes)} nodes vs {len(bb.nodes)}")
    if ba.gates != bb.gates:
        return DefinitionVerdict(fid, "mismatch", None, f"{ba.gates} gates vs {bb.gates}")
    worst = "identical"
    for k, (na, nb) in enumerate(zip(ba.nodes, bb.nodes)):
        if na.form != nb.form or not same_callee(na.fn, nb.fn) or na.out != nb.out or tuple(na.axes) != tuple(nb.axes) \
                or na.n != nb.n or na.n_xs != nb.n_xs or len(na.args) != len(nb.args):
            return DefinitionVerdict(fid, "mismatch", ba.gates,
                                     f"node {k}: {na.form} {_spec_id(na.fn)} n={na.n} vs {nb.form} {_spec_id(nb.fn)} n={nb.n}")
        if (na.name or None) != (nb.name or None):
            names.append(f"{fid} node {k}: {na.name!r} vs {nb.name!r}")
        for i, (aa, ab) in enumerate(zip(na.args, nb.args)):
            if aa.type != ab.type:
                return DefinitionVerdict(fid, "mismatch", ba.gates, f"node {k} arg {i}: type {aa.type!r} vs {ab.type!r}")
            v = _refs_equal(aa.refs, ab.refs, leaf_limit)
            if v == "mismatch":
                return DefinitionVerdict(fid, "mismatch", ba.gates, f"node {k} arg {i}: reference sequences differ ({aa.refs!r} vs {ab.refs!r})")
            worst = _worse(worst, v)
    v = _refs_equal(ba.ret, bb.ret, leaf_limit)
    if v == "mismatch":
        return DefinitionVerdict(fid, "mismatch", ba.gates, f"return references differ ({ba.ret!r} vs {bb.ret!r})")
    worst = _worse(worst, v)
    return DefinitionVerdict(fid, worst, ba.gates)


_RANK = {"identical": 0, "same-references": 1, "unverified": 2, "mismatch": 3}


def _worse(a: str, b: str) -> str:
    return a if _RANK[a] >= _RANK[b] else b


def _same_id(fa, fb) -> bool:
    return _spec_id(fa) == _spec_id(fb)


def _statics_diff(fa, fb) -> dict:
    """The binding records of two paired specialisations, as JSON: keys on one side only and keys whose values differ
    (function references by id, records/tuples expanded)."""
    from verity.ml.circuits.defs import _plain_static
    a = fa.bindings if isinstance(fa, SpecializedDefinition) else {}
    b = fb.bindings if isinstance(fb, SpecializedDefinition) else {}
    out: dict = {}
    left_only = {k: _plain_static(a[k]) for k in a if k not in b}
    right_only = {k: _plain_static(b[k]) for k in b if k not in a}
    differ = {k: [_plain_static(a[k]), _plain_static(b[k])] for k in a if k in b and _plain_static(a[k]) != _plain_static(b[k])}
    if left_only:
        out["left_only"] = left_only
    if right_only:
        out["right_only"] = right_only
    if differ:
        out["differ"] = differ
    return out


def correspond_rebound(left, right, *, leaf_limit: int = 1 << 22) -> CorrespondenceReport:
    """Correspondence of two programs whose definitions may differ in *identity* — name, version or binding record
    (a factored definition with new explicit statics against the definition whose body hard-coded them; a port's
    `NameHopper_v1` against `Name_v1`) — but must denote the same circuit.

    Definitions are paired by position: the two roots are walked in lockstep and the callee of node k on the left is
    paired with the callee of node k on the right (a pairing is a bijection: a definition paired twice, differently, is
    a mismatch). Each paired body is compared exactly as `correspond` compares same-id bodies (same gates, same
    producers up to the pairing, same ordered reference sequences leaf by leaf, call names as annotations), and the
    two roots' `S.out` leaf by leaf. Digests are reported, never required: with different identities they differ by
    construction, so the verdict that carries the claim is `corresponds` with every pair `identical` (byte-equal
    encodings up to callee ids) or `same-references`. `report.pairs` is the pairing with each pair's statics diff —
    the record of *which* statics the rebinding introduced or changed; `unpaired_*` lists definitions reachable on one
    side only through static bindings (bound but never called — part of an identity, not of the circuit)."""
    ra, rb = left.root, right.root
    pair_of: dict[str, str] = {}
    rev: dict[str, str] = {}
    work: list[tuple] = []

    def same_callee(fa, fb) -> bool:
        ia, ib = _spec_id(fa), _spec_id(fb)
        if ia in pair_of:
            return pair_of[ia] == ib
        if ib in rev:
            return False
        if isinstance(fa, PrimitiveDefinition) != isinstance(fb, PrimitiveDefinition):
            return False
        pair_of[ia] = ib
        rev[ib] = ia
        work.append((fa, fb))
        return True

    same_callee(ra, rb)
    names: list[str] = []
    mism: list[DefinitionVerdict] = []
    unver: list[DefinitionVerdict] = []
    same_ids: list[str] = []
    pairs: list[dict] = []
    n_ident = n_same = 0
    while work:
        fa, fb = work.pop(0)
        ia, ib = _spec_id(fa), _spec_id(fb)
        fid = ia if ia == ib else f"{ia} ~ {ib}"
        if isinstance(fa, PrimitiveDefinition):
            ok = fa.params == fb.params and fa.ret == fb.ret
            v = DefinitionVerdict(fid, "identical" if ok else "mismatch", 1, "" if ok else "primitive signatures differ")
        else:
            v = _compare_bodies_(fa.body, fb.body, leaf_limit, names, fid, same_callee=same_callee)
        rec = {"left": ia, "right": ib, "verdict": v.verdict, "gates": v.gates}
        if ia != ib:
            d = _statics_diff(fa, fb)
            if d:
                rec["statics"] = d
        pairs.append(rec)
        if v.verdict == "identical":
            n_ident += 1
        elif v.verdict == "same-references":
            n_same += 1
            same_ids.append(fid)
        elif v.verdict == "unverified":
            unver.append(v)
        else:
            mism.append(v)
    root_out = _refs_equal(ra.body.ret, rb.body.ret, leaf_limit) if ra.body.ret_type == rb.body.ret_type else "mismatch"
    unpaired_left = sorted(set(_reachable(ra)) - set(pair_of), key=lambda s: (len(s), s))
    unpaired_right = sorted(set(_reachable(rb)) - set(rev), key=lambda s: (len(s), s))
    ok = not mism and not unver and root_out in ("identical", "same-references")
    return CorrespondenceReport(ok, program_digest(left), program_digest(right), len(pairs), n_ident, n_same, mism, unver,
                                same_ids, root_out, names, ra.gates, rb.gates, pairs, unpaired_left, unpaired_right)


def correspond(left, right, *, leaf_limit: int = 1 << 22) -> CorrespondenceReport:
    """Graph/reference correspondence of two programs (`Program` or `DecodedProgram`): same definitions, same
    gates, same producers, same ordered reference sequences, same `S.out`. Digest equality is reported but
    never required."""
    da, db = _reachable(left.root), _reachable(right.root)
    ids = sorted(set(da) | set(db), key=lambda s: (len(s), s))
    mism: list[DefinitionVerdict] = []
    unver: list[DefinitionVerdict] = []
    same_ids: list[str] = []
    names: list[str] = []
    n_ident = n_same = 0
    for fid in ids:
        fa, fb = da.get(fid), db.get(fid)
        if fa is None or fb is None:
            mism.append(DefinitionVerdict(fid, "missing-left" if fa is None else "missing-right"))
            continue
        if isinstance(fa, PrimitiveDefinition) or isinstance(fb, PrimitiveDefinition):
            if not (isinstance(fa, PrimitiveDefinition) and isinstance(fb, PrimitiveDefinition)):
                mism.append(DefinitionVerdict(fid, "mismatch", None, "primitive on one side, composite on the other"))
            elif fa.params != fb.params or fa.ret != fb.ret:
                mism.append(DefinitionVerdict(fid, "mismatch", 1, "primitive signatures differ"))
            else:
                n_ident += 1
            continue
        # the root ids differ by name (`Root[...]`); everything else must match by id — compare bodies either way
        v = _compare_bodies(fa, fb, leaf_limit, names, fid)
        if v.verdict == "identical":
            n_ident += 1
        elif v.verdict == "same-references":
            n_same += 1
            same_ids.append(fid)
        elif v.verdict == "unverified":
            unver.append(v)
        else:
            mism.append(v)
    # roots: the two `Root[...]` ids differ when the program functions' ids differ; compare them directly
    ra, rb = left.root, right.root
    root_ids = {_spec_id(ra), _spec_id(rb)}
    if len(root_ids) == 2:
        mism = [m for m in mism if m.id not in root_ids]
        v = _compare_bodies(ra, rb, leaf_limit, names, f"{_spec_id(ra)} ~ {_spec_id(rb)}")
        if v.verdict == "identical":
            n_ident += 1
        elif v.verdict == "same-references":
            n_same += 1
            same_ids.append(v.id)
        elif v.verdict == "unverified":
            unver.append(v)
        else:
            mism.append(v)
    root_out = _refs_equal(ra.body.ret, rb.body.ret, leaf_limit) if ra.body.ret_type == rb.body.ret_type else "mismatch"
    ok = not mism and not unver and root_out in ("identical", "same-references")
    return CorrespondenceReport(ok, program_digest(left), program_digest(right), len(ids), n_ident, n_same, mism, unver,
                                same_ids, root_out, names, ra.gates, rb.gates)


__all__ = ["correspond", "correspond_rebound", "compare_bodies", "CorrespondenceReport", "DefinitionVerdict"]
