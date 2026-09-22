"""[A3 contract 2026-09-10] The verifier's own derivation of every selected unit's expected statement (PROTOCOL.md R4/R9;
conformance R4 M1) -- what replaces the legacy V5 (``scripts/rec_p1_verify.py:292-325``), which compared the located unit with
``record["provenance"]["units"]`` (prover-written metadata outside the bound object) and checked root hex only.  That V5 is NOT copied
into ``verifier_v3``.

Here every identity comes from the verifier's sources and nothing from the prover's object or provenance:

* the selected units and their located gate (request, t, layer, Gemm, member j / weight tensor + offset) from ``mapping.units.jsonl``,
  which ``rec_p1_obligations.ProgramLocator`` derives from the pinned program (V1.program_rebuilt pins its digest);
* the committed trees of J with their committed roots, leaf counts and id lists (``commitments.json``; each RU tree's ``index.json``
  whose recomputed ``ids_digest`` MUST equal the committed one -- an id list is used only after that check; the weights tree's row-leaf
  ids from the registered weights index).  Rank order is ``verity_vllm.identity.sort_key`` (``verity-vllm/leaf/v1``);
* for a GemmCoordinate unit the output row id ``(request, t, op) -> canonical id`` from the committed id list, the x producer (op, port,
  column slice) and the weight tensor + row from the located member's own operand gates in the pinned program (``locator.gemm_operands``,
  carried as ``unit["operands"]``), then ``coordinate_check.open_coordinate`` authenticates the three openings against the committed roots
  -- K, the x row shape and the weight row shape are read from those authenticated openings.  The prover's ``description.json`` is compared
  against the derivation (``V5.description``), never read into it [A2 21:00Z];
* for an Input unit the leaf and word from the located Input gate (``unit["locate"]``: weights tensor + offset, row layout from the weights
  index) and the pin from the verifier's own authenticated opening of the committed leaf under the weights root (== the auditor's
  registered root, V3); ``weights.cos_sin`` is a derived table -> no equation, checked by V6 alone.  The prover's
  ``obligations/input_checks.jsonl`` row is compared (leaf, word, committed, expected, recomputed ``ok``: ``V6.inputs``), never read.

Coverage (R9) is A5/6's ``e2e.coverage_v1`` (integrator note 19:20Z, the integration contract): :func:`coverage_inputs` builds once
per record ``trees`` (a ``CommittedTree`` per committed tree of the retained RUs + weights) and ``rank_of`` (A2's ``rank_oracle`` over the
committed id lists, ``identity.sort_key``), a :class:`RecordPartition` (``locate(vu_id)`` -> A2 ``ExpectedObligation`` via
``gemm_coordinate_expectation`` / ``input_equation_expectation``) and a :class:`RecordCircuit` (``kind_program`` -> family + canonical body
digest from ``veritor.interface.programs``).  ``reverify.py`` then calls ``expected_statements(circuit, partition, Q)`` (raising => the
record is refused), builds an ``AcceptedProof`` per proof ONLY after approved-guest verification + ``check_binding``, and
``check_coverage(expected, proofs, trees=, rank_of=)`` -- per VU covered exactly once / uncovered / duplicate / unexpected, with A2's
``check_expected_positions`` run per proof (a failing proof covers nothing).  No ``selected_vus`` / ``provenance.units`` / ``unit_matches``.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from veritor.interface.expected import (
    CommittedTree,
    ExpectedObligation,
    ExpectedValue,
    gemm_coordinate_expectation,
    input_equation_expectation,
    rank_oracle,
)
from veritor.interface.programs import gemm_coordinate

WEIGHTS_TREE = "weights"
INTERIOR_TREE_PREFIX = "interior/"


@dataclass
class CommittedTrees:
    """The committed trees the verifier lets a statement reference, with their id lists (checked against ``ids_digest``)."""

    trees: dict[str, CommittedTree] = field(default_factory=dict)
    ids: dict[str, list[str]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    #: [E9, NIGHT G1] trees whose id list is ALREADY in rank order (the verifier-derived public trees: D23 `table/…`, `literal/w=…`)
    ordered: set[str] = field(default_factory=set)

    def rank_of(self):
        """The committed rank oracle: code-point order (``identity.sort_key``) for the step-1 trees; for the step-3 ``interior/`` trees (and the
        verifier-derived public trees, ``ordered``) the committed list order itself (E2 rule item 5: rank = layout order -- call order, then chunk
        index -- never the id string's order)."""
        from verity_vllm import identity

        listed_ids = {t: ids for t, ids in self.ids.items() if t.startswith(INTERIOR_TREE_PREFIX) or t in self.ordered}
        sorted_ = rank_oracle({t: ids for t, ids in self.ids.items() if t not in listed_ids}, sort_key=identity.sort_key)
        listed = rank_oracle(listed_ids)

        def rank_of(tree: str, canonical_id: str) -> int | None:
            return listed(tree, canonical_id) if tree in listed_ids else sorted_(tree, canonical_id)

        return rank_of


def committed_trees(sel: Path, commitments: dict[str, Any], tree_ids: list[str], cache: Any) -> CommittedTrees:
    """Committed root / leaf count / id list for every tree in ``tree_ids`` -- RU trees from their ``index.json`` (the committed list:
    its recomputed ``ids_digest`` must equal ``commitments.trees[tid].ids_digest`` and its length the committed ``leaf_count``), the
    weights / boundary trees from the capture reader's tree (same check).  A tree that fails is NOT added (a statement referencing it
    fails in :func:`check_statement` as 'not a committed tree of J')."""
    from verity_vllm import hashing

    out = CommittedTrees()
    for tid in tree_ids:
        c = (commitments.get("trees") or {}).get(tid)
        if c is None and tid.startswith(INTERIOR_TREE_PREFIX) and tid.count("/") == 2:
            # E2 records the step-3 trees under `commitments.json#interior_roots["<req>/<step>"]` (`e2e.interior.tree.add_interior_roots`)
            c = (commitments.get("interior_roots") or {}).get(tid[len(INTERIOR_TREE_PREFIX):])
        if c is None:
            out.problems.append(f"{tid}: not in commitments.json")
            continue
        try:
            if tid.startswith("ru/"):
                _, rid, step = tid.split("/")
                index = json.loads((sel / "ru" / rid / step / "index.json").read_text())
                ids = [e["canonical_id"] for e in index["entries"]]
            elif tid.startswith(INTERIOR_TREE_PREFIX):
                # [E3 2026-09-11] E2's step-3 tree: `adapter/selective/interior/<req>/<step>/index.json` (`e2e.interior.tree.write_tree`):
                # `leaves[] = {rank, canonical_id, ...}` -- ids taken in RANK order (layout order), never re-sorted
                _, rid, step = tid.split("/")
                index = json.loads((sel / "interior" / rid / step / "index.json").read_text())
                rows_ = list(index.get("leaves") or index.get("entries") or index.get("ids") or [])
                if rows_ and isinstance(rows_[0], dict) and all("rank" in e for e in rows_):
                    rows_.sort(key=lambda e: int(e["rank"]))
                ids = [str(e["canonical_id"]) if isinstance(e, dict) else str(e) for e in rows_]
            elif tid.startswith("boundary/") and (sel / "boundary" / tid.split("/", 1)[1] / "index.json").exists():
                # [D17] record_v5 spelling (E8's CORE, E9's `step2_check_registered`): `boundary/<rid>/index.json#entries[].canonical_id`,
                # read before the capture cache (a registered program's record has no capture)
                index = json.loads((sel / "boundary" / tid.split("/", 1)[1] / "index.json").read_text())
                ids = [str(e["canonical_id"]) if isinstance(e, dict) else str(e) for e in index["entries"]]
            elif int(c.get("leaf_count", -1)) == 0 and (cache is None or not (sel / tid / "index.json").exists()):
                # [D17] an EMPTY committed tree with no store on disk (E8's CORE: "DemoCore has no prescribed weights: empty tree (0 leaves)"):
                # the id list is [] and its ids_digest / leaf_count are checked like any other
                ids = []
            else:
                ids = list(cache.tree(tid).ids)
        except Exception as e:  # noqa: BLE001
            out.problems.append(f"{tid}: id list unreadable: {type(e).__name__}: {str(e)[:120]}")
            continue
        digest = hashing.ids_digest(list(ids) if tid.startswith(INTERIOR_TREE_PREFIX) else sorted(ids)).hex()
        if digest != c.get("ids_digest"):
            out.problems.append(f"{tid}: recomputed ids_digest {digest[:16]} != committed {str(c.get('ids_digest'))[:16]}")
            continue
        if len(ids) != int(c["leaf_count"]):
            out.problems.append(f"{tid}: {len(ids)} ids, committed leaf_count {c['leaf_count']}")
            continue
        out.trees[tid] = CommittedTree(str(c["root"]), int(c["leaf_count"]))
        out.ids[tid] = ids
    return out


def output_row_index(ids_by_tree: dict[str, list[str]], lm_head_ops: set[str]) -> dict[tuple[str, int, str], str]:
    """``(request, t, op) -> canonical id`` of every committed GEMM / lm_head output row, from the COMMITTED id lists."""
    from verity_vllm import identity

    idx: dict[tuple[str, int, str], str] = {}
    for tid, ids in ids_by_tree.items():
        if not tid.startswith("ru/"):
            continue
        for cid_s in ids:
            c = identity.parse(cid_s)
            if c.kind == "value" and c.op and (c.op.endswith(".gemm") or c.op in lm_head_ops) and (c.port or 0) == 0 and c.coordinate is None:
                idx[(c.request_id, c.t, c.op)] = cid_s
    return idx


class Deriver:
    """Derives expectations for located units of one record (Description + committed id lists + authenticated openings)."""

    def __init__(self, sel: Path, commitments: dict[str, Any], *, recompute_leaves: bool = True) -> None:
        from verity_vllm.capture import CaptureReader
        from verity_vllm.commit import TreeCache
        from verity_vllm.description import Description

        self.sel = sel
        self.commitments = commitments
        self.reader = CaptureReader(sel)
        self.cache = TreeCache(self.reader, recompute_leaves=recompute_leaves)
        # [E3, D17] vLLM's `description.json` is compare-only and Serve_v1-only: a registered program's record (E8's CORE) has none.
        # `self.desc = None` then; every vLLM-only path that needs it refuses BY NAME (`DescriptionUnavailable`), never an unhandled exception.
        self.desc = None
        self.lm_head_ops: set[str] = set()
        if (sel / "description.json").exists():
            self.desc = Description(self.reader.description)
            self.lm_head_ops = {n.op for n in self.desc.nodes_of_type("lm_head")}
        self._committed: CommittedTrees | None = None
        self._row_index: dict[tuple[str, int, str], str] | None = None

    # [A5/6 patch] (request, t, op, port) -> (tree, canonical id, word count) of every committed operator row, from the COMMITTED id
    # lists and the RU index entries' shapes -- what `coverage_v1.attn_head_builder.head_expectation` reads (never a value)
    def attention_row(self, request_id: str, t: int, op: str, port: int):
        if not hasattr(self, "_attn_rows"):
            from verity_vllm import identity

            rows: dict[tuple[str, int, str, int], tuple[str, str, int]] = {}
            for tid, ids in (self._committed.ids if self._committed else {}).items():
                if not tid.startswith("ru/"):
                    continue
                _, rid, step = tid.split("/")
                shapes = {e["canonical_id"]: int(np.prod(e["shape"])) for e in json.loads((self.sel / "ru" / rid / step / "index.json").read_text())["entries"]}
                for cid_s in ids:
                    c = identity.parse(cid_s)
                    if c.kind == "value" and c.op and c.coordinate is None:
                        rows[(c.request_id, int(c.t), c.op, int(c.port or 0))] = (tid, cid_s, shapes.get(cid_s, 0))
            self._attn_rows = rows
        return self._attn_rows.get((request_id, int(t), op, int(port)))

    @property
    def config(self) -> dict[str, Any]:
        """NH / KVH / D for the head expectation's slice arithmetic: the pinned program's C (``build_pinned_program`` -> passed to
        ``coverage_inputs(config=)``), cross-checked against the Description's attention node consts (num_heads / num_kv_heads /
        head_size) when present -- the fixed circuit is authoritative, a disagreement is a refusal."""
        C = getattr(self, "_config", None)
        if C is None:
            raise ValueError("Deriver.config: the pinned program's C was not given to coverage_inputs(config=) -- head expectations refused")
        if self.desc is None:
            raise DescriptionUnavailable("Deriver.config: head expectations cross-check the Description's attention consts and this record has no "
                                         "adapter/selective/description.json (a registered program's record) -- vLLM head expectations refused by name")
        for n in self.desc.nodes_of_type("attention"):
            k = getattr(n, "consts", None) or {}
            if k and (int(k.get("num_heads", C["NH"])), int(k.get("num_kv_heads", C["KVH"])), int(k.get("head_size", C["D"]))) != (int(C["NH"]), int(C["KVH"]), int(C["D"])):
                raise ValueError(f"Description attention consts {k} disagree with the pinned program C (NH={C['NH']}, KVH={C['KVH']}, D={C['D']})")
            break
        return {"NH": int(C["NH"]), "KVH": int(C["KVH"]), "D": int(C["D"])}

    def committed(self, tree_ids: list[str]) -> CommittedTrees:
        """Committed trees for exactly ``tree_ids`` (each call re-checks; the id lists are cached per tree)."""
        if self._committed is None:
            self._committed = CommittedTrees()
        missing = [t for t in tree_ids if t not in self._committed.trees]
        if missing:
            more = committed_trees(self.sel, self.commitments, missing, self.cache)
            self._committed.trees.update(more.trees)
            self._committed.ids.update(more.ids)
            self._committed.problems.extend(more.problems)
        return CommittedTrees({t: self._committed.trees[t] for t in tree_ids if t in self._committed.trees},
                              {t: self._committed.ids[t] for t in tree_ids if t in self._committed.ids},
                              [p for p in self._committed.problems if p.split(":")[0] in set(tree_ids)])

    def row_index(self, ru_tree_ids: list[str]) -> dict[tuple[str, int, str], str]:
        c = self.committed(ru_tree_ids)
        return output_row_index(c.ids, self.lm_head_ops)

    def op_of(self, locate: dict[str, Any]) -> str | None:
        if locate.get("role") == "final":
            return next(iter(self.lm_head_ops)) if len(self.lm_head_ops) == 1 else None
        if locate.get("op_suffix") is None or locate.get("layer") is None:
            return None
        return f"layers.{locate['layer']}.{locate['op_suffix']}"

    # [A2 21:00Z, R9 EAs E4/E5 + E11 -- CLAIM granted on these two functions] Every identity of an expectation comes from the verifier's
    # program (``unit["locate"]`` / ``unit["operands"]``: ``locator.ProgramLocator`` over the pinned program), the committed id lists and the
    # verifier's own authenticated openings.  The prover's ``adapter/selective/description.json`` and ``obligations/input_checks.jsonl`` are
    # COMPARED against that derivation and never read into it: a disagreement is a named refusal (``V5.description`` / ``V6.inputs``).

    def coordinate(self, unit: dict[str, Any], row_index: dict[tuple[str, int, str], str]) -> ExpectedObligation:
        """The expectation for one provable (GemmCoordinate) located unit.  Raises ``ValueError`` with the reason when the verifier
        cannot derive it (no committed output row for the located gate, an opening fails to authenticate, ...) -- the caller records a FAIL;
        :class:`DescriptionDisagreement` (``V5.description``) when the prover's Description names other operands than the program."""
        from verity_vllm import identity
        from verity_vllm.capture import ru_tree_id
        from verity_vllm_adapter.coordinate_check import describe_coordinate, open_coordinate

        vu = unit.get("vu_index")
        L, ops = unit.get("locate") or {}, unit.get("operands") or {}
        if "refused" in L or L.get("j") is None:
            raise ValueError(f"vu {vu}: located unit refused or without a member index: {json.dumps(L)[:120]}")
        if "refused" in ops or not ops.get("x") or not ops.get("w"):
            raise ValueError(f"vu {vu}: located unit without verifier-derived operands (locator.gemm_operands): {json.dumps(ops)[:160]}")
        rid, op = unit["request_id"], self.op_of(L)
        if op is None:
            raise ValueError(f"vu {vu}: cannot name the GEMM op of {json.dumps(L)[:120]}")
        cid = row_index.get((rid, int(L["t"]), op))
        if cid is None:
            raise ValueError(f"vu {vu}: no committed output row for ({rid[:8]}, t={L['t']}, {op}) in the committed id lists of J")
        j, K = int(L["j"]), int(ops["K"])
        # the program's operands, named in the record's identity: x = the producer's committed row at the same (request, t, phase), W[row, :]
        # = the row leaf of the weights tensor the member's wrow gates lie in (weights index layout under the registered root)
        x_id = str(identity.parse(cid).with_(layer=str(ops["x"]["layer"]), op=str(ops["x"]["op"]), port=int(ops["x"]["port"])))
        w_leaf = self.reader.weights.leaf_id_for_row(str(ops["w"]["canonical_id"]), int(ops["w"]["row"]))
        if w_leaf == ops["w"]["canonical_id"]:
            raise ValueError(f"vu {vu}: {ops['w']['canonical_id']} is a whole-tensor leaf: no row-addressable opening for W[{ops['w']['row']}, :]")
        x_cols = None if ops["x"].get("cols") is None else list(range(int(ops["x"]["cols"][0]), int(ops["x"]["cols"][1])))
        # the prover's description.json: compared field by field, never used
        if self.desc is None:
            raise DescriptionUnavailable(f"V5.description: vu {vu}: GemmCoordinate expectations compare the prover's description.json and this record "
                                         f"has none (adapter/selective/description.json absent -- a registered program's record has no GEMM coordinates)")
        try:
            _inst, d_x, d_w, d_cols, d_N = describe_coordinate(self.desc, self.reader, cid, j)
        except Exception as e:  # noqa: BLE001
            raise DescriptionDisagreement(f"V5.description: vu {vu}: description.json cannot name the operands of {cid}[{j}]: {type(e).__name__}: {str(e)[:160]}") from e
        dis = [f"{f}: program {mine!r} != description {theirs!r}" for f, mine, theirs in
               (("x", x_id, d_x), ("w", w_leaf, d_w), ("x_cols", x_cols, d_cols), ("N", int(L["N"]), int(d_N))) if mine != theirs]
        if dis:
            raise DescriptionDisagreement(f"V5.description: vu {vu}: {cid}[{j}]: " + "; ".join(dis))
        lo = open_coordinate(self.reader, self.commitments, self.cache, cid, j, x_id=x_id, w_leaf=w_leaf, x_cols=x_cols)  # RelationError when an opening fails to authenticate
        if lo.K != K:
            raise ValueError(f"vu {vu}: committed weight row {w_leaf} has {lo.K} words, the program's GemmCoordinate has K={K}")
        loc_out, loc_x = self.reader.locate(cid), self.reader.locate(x_id)
        if loc_out is None or loc_x is None:
            raise ValueError(f"vu {vu}: {cid if loc_out is None else x_id} is in no replay unit of its request")
        out_tree, x_tree = ru_tree_id(*loc_out), ru_tree_id(*loc_x)
        x_shape = tuple(int(d) for d in lo.x.shape)
        cols = list(range(K)) if x_cols is None else x_cols
        x_offsets = [tuple(int(i) for i in np.unravel_index(k, x_shape)) for k in cols]
        w_offsets = [(0, k) if len(tuple(lo.w_row.shape)) == 2 else (k,) for k in range(K)]
        return gemm_coordinate_expectation(out=ExpectedValue(out_tree, cid, (j,)), x=(x_tree, x_id), x_offsets=x_offsets, w=(WEIGHTS_TREE, w_leaf),
                                           w_offsets=w_offsets, subcircuit=gemm_coordinate(K).digest, label=f"vu={vu} {cid}[{j}]")

    def input_equation(self, unit: dict[str, Any], check: dict[str, Any] | None) -> ExpectedObligation | None:
        """The expectation for one Input unit from the verifier's located gate (``unit["locate"]``: weights tensor + word offset) and its own
        authenticated opening of the committed leaf; ``None`` for a derived table (``weights.cos_sin``: not a checkpoint tensor -> no Input
        equation; V6 covers it by opening).  The pin is the committed word under the weights root, which V3 requires to equal the auditor's
        registered root (``weights_authorized.json#expected_root``, computed from the pinned checkpoint before any prover artefact) -- so the
        prover's ``input_checks.jsonl`` row, when given, is COMPARED (leaf, word, committed, expected, ok recomputed) and never read:
        :class:`InputCheckDisagreement` (``V6.inputs``).  Raises ``ValueError`` for a prompt-token input (its expectation derivation is
        A5/6's ``coverage_v1``; none in B0) or an unknown leaf."""
        from verity_vllm_adapter.coordinate_check import _open

        vu = unit.get("vu_index")
        L = unit.get("locate") or {}
        kind = L.get("kind")
        if kind == "prompt_token":
            raise ValueError(f"vu {vu}: prompt-token Input expectation derivation is not implemented in verifier_v3 (A5/6 coverage_v1) -- fail closed")
        if kind != "weight" or "refused" in L:
            raise ValueError(f"vu {vu}: located Input gate is not a weights word: {json.dumps(L)[:120]}")
        cid, off = str(L["canonical_id"]), int(L["offset"])
        entry = self.reader.weights.entry(cid)
        shape = tuple(int(d) for d in entry.shape)
        if entry.layout.kind == "rows":
            row_words = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            row, word = divmod(off, row_words)
            leaf = self.reader.weights.leaf_id_for_row(cid, row)
            offset = (0, *(int(i) for i in np.unravel_index(word, shape[1:]))) if len(shape) > 1 else (word,)
        else:
            leaf, word = cid, off
            offset = tuple(int(i) for i in np.unravel_index(word, shape))
        width = 32 if entry.dtype == "float32" else 16
        o, problem = _open(self.cache, self.commitments, leaf)
        if o is None or not hasattr(o, "value"):
            raise ValueError(f"vu {vu}: {leaf}: {problem or 'not a single-leaf opening'}")
        words = np.frombuffer(o.value, dtype="<u4" if width == 32 else "<u2").reshape(-1)
        if not (0 <= word < words.size):
            raise ValueError(f"vu {vu}: word {word} outside the {words.size}-word leaf {leaf}")
        committed = int(words[word])
        derived_table = str(L.get("field", "")) == "weights.cos_sin" or entry.dtype == "float32"
        expected = None if derived_table else committed
        registered = self._registered_leaf_table()
        leaf_ok = None if registered is None else (registered.get(cid, {}).get("tensor_sha256") == entry.sha256)
        ok = (leaf_ok is not False) and (expected is None or expected == committed)
        if check is not None:  # the prover's row, compared only
            mine = {"kind": "weight", "leaf": leaf, "word": word, "committed": committed, "expected": expected, "ok": ok}
            if leaf_ok is not None:
                mine["registered_leaf_sha256_match"] = leaf_ok
            dis = [f"{f}: verifier {v!r} != input_checks.jsonl {check.get(f)!r}" for f, v in mine.items() if check.get(f) != v]
            if dis:
                raise InputCheckDisagreement(f"V6.inputs: vu {vu}: {leaf}[{word}]: " + "; ".join(dis))
        if not ok:
            raise ValueError(f"vu {vu}: {leaf}: committed leaf sha256 != registered leaf table (weights_authorized.json)")
        if expected is None:
            return None
        return input_equation_expectation(ExpectedValue(WEIGHTS_TREE, leaf, offset), expected, width,
                                          label=f"vu={vu} {leaf}{''.join(f'[{i}]' for i in offset)}")

    def _registered_leaf_table(self) -> dict[str, dict[str, Any]] | None:
        """``weights_authorized.json#leaf_table`` by canonical id (the auditor's registration in the record root, digest-pinned by the
        instance -- a verifier source), or None when the record has none."""
        if not hasattr(self, "_reg_leaves"):
            p = self.sel.parents[1] / "weights_authorized.json"
            self._reg_leaves = {e["canonical_id"]: e for e in json.loads(p.read_text()).get("leaf_table", [])} if p.exists() else None
        return self._reg_leaves


class DescriptionUnavailable(ValueError):
    """[E3, D17] A vLLM-only path (GemmCoordinate description comparison, attention-head consts) was reached on a record with no
    ``adapter/selective/description.json`` -- a registered program's record.  Refused by name; positions / statements never need it."""

    check = "V5.description"


class DescriptionDisagreement(ValueError):
    """``V5.description``: the prover's ``description.json`` names other operands (x row, weight tensor / row, ``arg_cols``, N) for a located
    GemmCoordinate than the pinned program does.  The expectation is built from the program either way; this is a refusal of the record."""

    check = "V5.description"


class InputCheckDisagreement(ValueError):
    """``V6.inputs``: the prover's ``obligations/input_checks.jsonl`` row for a located Input unit disagrees with the verifier's own leaf /
    word / committed word / pin / recomputed ``ok``.  Never a source; a refusal of the record."""

    check = "V6.inputs"


# --- A5/6's coverage_v1 call boundary (integrator note 19:20Z): Partition / Circuit over this deriver -----------------------------------


class NotAProofCoveredUnit(ValueError):
    """The selected VU has no expected statement a proof could meet: no checker for its kind (an uncovered selected equation), or a
    derived-table input the verifier checks by its own opening (V6).  ``expected_statements`` turns it into a ``CoverageError`` (refusal)."""


class PositionOracle:
    """[E3, EOD 2026-09-11 -- brief interfaces 1, 4, 5] The expected statement of ANY VU of the declared population **as positions**
    (``(vu_index, kind, operands -> (tree, leaf id, flat), output -> (tree, leaf id, flat))``) from the pinned program, the pinned partition
    (population rows) and the pinned ``protocol_instance.json#commitments.interior_layout`` only.  The prover's ``mapping.units.jsonl`` /
    ``description.json`` / interior ``index.json`` are never read for a position; the step-1 id lists are used only after ``ids_digest``.

    Program-agnostic: one :class:`~e2e.verifier_v3.locator.GateResolver` per request, built from an ``adapter`` of callables --
    ``row_gates(loc)`` (committed id -> producing call + return gates), ``regions(loc, layout, rid, committed_trees)`` (gate -> interior
    tree), ``calls(loc)`` (gate -> the layout's call), ``prefix(rid)`` (leaf-id prefix), ``weight_word(g)`` / ``prompt_word(rid, g)``
    (Input gates).  The default adapter is ``Serve_v1``'s (:func:`~e2e.verifier_v3.locator.vllm_row_gates`, :func:`serve_regions`,
    :func:`serve_calls`, ``p=<profile>/r=<rid>/``); E8's CORE passes its own (``calls_from_intervals`` over its RU list, its boundary
    leaf -> return gates map, its prefix) and resolves through the same code."""

    def __init__(self, program: Any, locator: Any, *, population: Any, committed_ids: dict[str, list[str]], commitments: dict[str, Any],
                 interior_layout: dict[str, Any] | None, reader: Any = None, profile_id: str = "", shapes: dict[str, tuple[int, ...]] | None = None,
                 adapter: dict[str, Any] | None = None) -> None:
        from e2e.verifier_v3 import locator as L
        from veritor.interface.expected import InteriorLayoutUnpinned

        self.program, self.locator, self.commitments, self.reader = program, locator, commitments, reader
        self.profile_id = str(profile_id or commitments.get("profile_id", ""))
        self.committed_ids = committed_ids
        self.shapes = shapes or {}
        self.rows = L.PopulationRows(population)
        self.pin = interior_layout
        self.layout = None
        self.layout_problem: str | None = None
        try:
            self.layout = L.interior_layout_rule(interior_layout)
        except InteriorLayoutUnpinned as e:  # boundary kinds still resolve; any interior word refuses by name (V3.interior_layout)
            self.layout_problem = str(e)
        generic = isinstance(locator, L.GenericLocator)
        self.adapter = {"row_gates": L.vllm_row_gates, "regions": L.generic_regions if generic else L.serve_regions,
                        "calls": (lambda loc: L.calls_from_intervals(loc.calls())) if generic else L.serve_calls,
                        "prefix": lambda rid: f"p={self.profile_id}/r={rid}/", "weight_word": self._weight_word, "prompt_word": self._prompt_word,
                        # the committed step-1 trees of one request: Serve_v1's `ru/<rid>/*` + `boundary/<rid>`; a registered program's record: every tree
                        "trees": (lambda rid, ids: {t: list(v) for t, v in ids.items()}) if generic else
                                 (lambda rid, ids: {t: list(v) for t, v in ids.items() if t.startswith(f"ru/{rid}/") or t == f"boundary/{rid}"})}
        self.adapter.update(adapter or {})
        self._resolvers: dict[str, Any] = {}
        self.timings: dict[str, float] = {}

    # -- Input gates (vLLM adapter defaults; mirror coverage_v1.ck_elem.ElemLocator.weight_word so both sides spell one position) --------
    def _weight_word(self, g: int):
        from e2e.verifier_v3.locator import GatePosition, PositionRefusal

        L = self.locator.locate_input(int(g))
        cid, offset = str(L["canonical_id"]), int(L["offset"])
        reader = self.reader
        if reader is None or not reader.has_weights():
            raise PositionRefusal("NO_WEIGHTS_LAYOUT", f"gate {g}: weights word {cid}[{offset}] but no weights store names its leaf")
        entry = reader.weights.entry(cid)
        shape = tuple(int(d) for d in entry.shape)
        K = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        if entry.layout.kind == "rows":
            row, k = divmod(offset, K)
            start, end = entry.block_of(row)
            return GatePosition(int(g), "input", WEIGHTS_TREE, entry.block_id(start), tuple(int(d) for d in entry.block_shape(start, end)), (row - start) * K + k, 16)
        return GatePosition(int(g), "input", WEIGHTS_TREE, cid, shape, offset, 16)

    def _prompt_word(self, rid: str, g: int):
        from e2e.verifier_v3.locator import GatePosition
        from verity_vllm import identity as vid

        L = self.locator.locate_input(int(g))
        return GatePosition(int(g), "input", f"boundary/{rid}", str(vid.boundary_id(self.profile_id, rid, "prompt_token", int(L["t"]))), (1,), 0, 32)

    # -- per request -------------------------------------------------------------------------------------------------------------------
    def resolver(self, request_id: str):
        from e2e.verifier_v3 import locator as L

        r = self._resolvers.get(request_id)
        if r is None:
            import time as _t

            t0 = _t.perf_counter()
            ids = self.adapter["trees"](request_id, self.committed_ids)
            boundary = L.BoundaryIndex(ids, self.adapter["row_gates"](self.locator), shape_of=lambda _t_, c, n: self.shapes.get(c, (n,)))
            layout = self.layout
            if layout is None:  # regions still named (for the refusal text) with the interface's default spelling
                from veritor.interface.expected import InteriorLayout

                layout = InteriorLayout()
            regions = self.adapter["regions"](self.locator, layout, request_id, committed_trees=self.commitments.get("trees") or {})
            r = L.GateResolver(self.program, self.locator, rows=self.rows, boundary=boundary, regions=regions, layout=self.layout,
                               calls=self.adapter["calls"](self.locator), prefix=self.adapter["prefix"](request_id), request_id=request_id,
                               weight_word=self.adapter["weight_word"], prompt_word=lambda g: self.adapter["prompt_word"](request_id, g))
            self._resolvers[request_id] = r
            self.timings[f"resolver_s[{request_id[:8]}]"] = _t.perf_counter() - t0
        return r

    def statement(self, unit: Mapping[str, Any]):
        """The :class:`~veritor.interface.expected.PositionStatement` of a located unit (``vu_index``, ``kind``, ``request_id``,
        ``gate_interval``); raises ``PositionRefusal`` (by code) / ``InteriorLayoutUnpinned``."""

        lo, hi = (int(x) for x in unit["gate_interval"])
        return self.resolver(str(unit["request_id"])).statement(int(unit["vu_index"]), str(unit["kind"]), lo, hi)

    def word(self, request_id: str, g: int):
        """E7's ``word(request_id, g) -> CommittedWord | None`` oracle (``ck_elem.RecordHandle(oracles=(...))``)."""

        return self.resolver(str(request_id)).oracle()(request_id, g)

    def resolve_all(self, units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """MEASURED resolve rate over ``units``: per kind ``{resolved, refused: {code: n}}``, with one example statement / refusal per kind."""
        from e2e.verifier_v3.locator import PositionRefusal
        from veritor.interface.expected import InteriorLayoutUnpinned

        per: dict[str, dict[str, Any]] = {}
        n_ok = 0
        for u in units:
            k = per.setdefault(str(u["kind"]), {"resolved": 0, "refused": {}, "origins": {}})
            try:
                st = self.statement(u)
            except (PositionRefusal, InteriorLayoutUnpinned) as e:
                code = getattr(e, "code", "INTERIOR_LAYOUT_UNPINNED")
                k["refused"][code] = k["refused"].get(code, 0) + 1
                k.setdefault("example_refusal", str(e)[:240])
                continue
            n_ok += 1
            k["resolved"] += 1
            for o in st.origins:
                k["origins"][o] = k["origins"].get(o, 0) + 1
            k.setdefault("example", st.to_json())
        return {"units": len(units), "resolved": n_ok, "rate": (n_ok / len(units)) if units else None, "kinds": dict(sorted(per.items())),
                "interior_layout": self.pin, "layout_problem": self.layout_problem}


def _is_literal_tree(tree: Any) -> bool:
    """A word of the verifier-derived literal tree (G1 PROPOSED `prescribed-literal/v1`: `literal/w=<w>`; E3's own spelling `literal` for a compiled literal)."""
    t = str(tree)
    return t == "literal" or t.startswith("literal/")


class RecordPartition:
    """``coverage_v1.Partition``: ``locate(vu_id) -> ExpectedObligation`` for a selected VU of this record, from the verifier's located unit
    (``mapping.units.jsonl`` = ``ProgramLocator`` over the pinned program), the committed id lists and the Description (never provenance).

    [E3 2026-09-11] With ``positions`` (a :class:`PositionOracle`) every VU also has :meth:`statement` -- its expected statement as
    positions; a builder kind whose ``ck_elem`` path refuses an interior word takes its obligation from that statement + the registered
    builder's body digest (the same positions E7's oracle hook resolves), and a kind with no checker is refused as *resolved, uncovered*
    (positions named) -- never as "interior"."""

    def __init__(self, deriver: Deriver, *, units_by_vu: dict[int, dict[str, Any]], checks_by_vu: dict[int, dict[str, Any]],
                 row_index: dict[tuple[str, int, str], str], record_handle: Any = None, positions: PositionOracle | None = None,
                 literals: Any = None) -> None:
        self.deriver, self.units_by_vu, self.checks_by_vu, self.row_index = deriver, units_by_vu, checks_by_vu, row_index
        # [CK-ELEM dispatch, re-targeted 22:20Z] e2e.coverage_v1.ck_elem.RecordHandle (verifier-built program + ProgramLocator + commitments
        # + committed id lists checked against ids_digest): what an expectation-capable builder reads on the verifier side (never the prover's object)
        self.record_handle = record_handle
        self.positions = positions
        # [E9, NIGHT G1 `prescribed-literal/v1`] the verifier's literal table (steps.LiteralTable, derived from the pinned program; None = no
        # instance pin -> a statement with a literal operand refuses LITERAL_UNPINNED, fail closed)
        self.literals = literals

    def _has_literal(self, st: Any) -> bool:
        origins = tuple(st.origins or ())[:len(st.operands)]
        return any(o == "literal" for o in origins) or any(str(v.tree) == "literal" for v in st.operands)

    def statement(self, vu_id: str):
        """[E3] The VU's expected statement as positions (:class:`~veritor.interface.expected.PositionStatement`) -- E9's step-5 call."""
        u = self.units_by_vu.get(int(vu_id))
        if u is None:
            raise KeyError(f"VU {vu_id} is not a located unit of the verifier's derivation")
        if self.positions is None:
            raise ValueError(f"VU {vu_id}: the partition was built without a population (coverage_inputs(population=...)): no position oracle")
        return self.positions.statement(u)

    def _from_positions(self, vu_id: str, u: dict[str, Any], why: str) -> ExpectedObligation:
        """The obligation of a builder kind from the position statement + the registered builder's canonical body digest."""
        from e2e.coverage_v1.builders import BUILDERS

        if self.positions is None:
            raise ValueError(f"VU {vu_id}: {why}; and no position oracle (coverage_inputs(population=...)) to derive the positions")
        st = self.positions.statement(u)
        tp = self._table_port(vu_id, u, st)
        digest = getattr(BUILDERS[str(u["kind"])], "subcircuit_digest", lambda: None)()
        if len(set(st.operand_gates)) < len(st.operand_gates) or self._has_literal(st) or digest is None:
            # [E9, D32.1] ALIASED operands (one program value read at several operand positions, RMSNorm x*x): the expectation's
            # position -> port map comes from the pinned program's operand GATE identities, the committed word identities must agree,
            # and the body is the aliased one (E7's `aliased-body/v1`), derived here independently of the builder (steps.alias_identity_expectation)
            # [NIGHT G1 `prescribed-literal/v1`] a literal operand is an incoming word of the verifier's literal table (self.literals);
            # without a table the statement refuses LITERAL_UNPINNED by name
            # [E7 for E9, D40 19:20Z -- E8's n3 (prompt 32) refusal `VU 1244303 … AmpereBF16TcDot16_v1|batch: builder without a canonical body
            # digest`] a kind whose body is chosen by the UNIT (the PV step: `tc_dot16` over a committed accumulator, `tc_dot16_0` over the
            # ZERO32 literal, the aliased body when a word repeats -- `subcircuit_digest()` is None by design) takes the SAME path as its padded
            # siblings: the body is `steps.alias_body_digest(kind, wiring)` over the program's gate identities (identity wiring -> the family's
            # canonical `tc_dot16`), the pad literals prescribed-literal ports where present.  ONE expectation path for m = 0..15; the registry
            # is consulted for the kind's spec only, never for a digest it does not have.
            from e2e.verifier_v3 import steps as _steps

            e, _wiring = _steps.alias_identity_expectation(st, literals=self.literals)
            if tp is not None:
                e = ExpectedObligation((*e.incoming, tp), e.outgoing, e.subcircuit, e.pins, e.layouts, e.label)
            return e
        ports, wiring = st.ports()
        if st.aliased:
            # [D30 `aliased-body/v1`, D32.1] a word read at several operand positions: the body is E7's aliased body for THIS wiring (position -> port,
            # from the pinned program's operand gate identities), never the kind's canonical body; incoming lists each word once
            from e2e.coverage_v1.elem_builders import alias_digests

            match = [d for d, w in alias_digests(str(u["kind"]), incoming=len(ports)).items() if tuple(w) == tuple(p for p in wiring if p >= 0)]
            if len(match) != 1:
                raise ValueError(f"VU {vu_id}: kind {u.get('kind')!r}: aliased wiring {wiring} has {len(match)} `aliased-body/v1` bodies in E7's derivation")
            digest = match[0]
        ob = st.obligation(str(digest))
        return ExpectedObligation((*ports, tp) if tp is not None else tuple(ports), ob.outgoing, ob.subcircuit, ob.pins, ob.layouts, ob.label)

    @staticmethod
    def _structural_literals(kind: str) -> tuple[str, ...]:
        """[composed-body/v1] the literal ids the kind's pinned body reads besides its operands (E7's `elem_builders.structural_literals`); () otherwise."""
        try:
            from e2e.coverage_v1.elem_builders import structural_literals

            return tuple(structural_literals(str(kind)))
        except Exception:  # noqa: BLE001 -- no builder registry on this path: no composed kinds
            return ()

    def _table_port(self, vu_id: str, u: dict[str, Any], st: Any):
        """[E7 for E9, 2026-09-12 16:30Z -- E1's s3 refusal `V5.positions`: `MufuEx2Ftz_v1|primitive` obligation 13 "no selected unit expects that
        value"] A TABLE kind's statement (D23 `table-layout/v1`: E7's `RsqrtApprox` / `MufuSqrtFtz` / `DivFullRcp`, E4's `MufuEx2Ftz` / `Fa2InvSum`)
        reads ONE MORE incoming word than the program has operands: the pinned MUFU table's row `index_fn(operand value)` as `(index << 32) | T[index]`
        -- the guest checks the index half against the operand inside the body, so the row is a port of the statement, and an expectation without
        it is a DIFFERENT statement (s3: prover `1edf3e22…` vs verifier `18b206f5…`, same subcircuit `f8883f6f…`).  E3's position statement knows the
        program's operands only; the port is derived here from the record's AUTHENTICATED operand word (`record_handle.open_leaf`, the TreeCache
        V3.trees_rehashed verified) through E7's one spelling `elem_builders.expected_table_port` (the same `TableWords.table_row` the builder's
        resolver uses).  `None` for a kind without a table; a table kind with no record handle to read the value from refuses by name (fail closed:
        the verifier cannot name the row the prover's statement binds)."""
        from e2e.coverage_v1.elem_builders import expected_table_port, table_of_kind

        kind = str(u.get("kind"))
        if table_of_kind(kind) is None:
            return None
        if self.record_handle is None:
            raise ValueError(f"VU {vu_id}: kind {kind!r} reads a MUFU table row selected by its operand's committed value; the partition has no record "
                             f"handle to read it from -> TABLE_ROW_UNRESOLVED (fail closed)")
        return expected_table_port(kind, st, self.record_handle)

    def locate(self, vu_id: str) -> ExpectedObligation:
        u = self.units_by_vu.get(int(vu_id))
        if u is None:
            raise KeyError(f"VU {vu_id} is not a located unit of the verifier's derivation")
        cls = str(u.get("class", ""))
        if isinstance(u.get("locate"), dict) and "builder" in u["locate"]:
            # [CK-ELEM dispatch] coverage_v1.builders.dispatch_expectation over the ONE registry: positions from the program + committed id lists
            from e2e.coverage_v1.builders import dispatch_expectation

            if self.record_handle is None:
                if self.positions is not None:
                    return self._pinned(vu_id, self._from_positions(vu_id, u, "no record handle"))
                raise ValueError(f"VU {vu_id}: kind {u.get('kind')!r} has a builder but the partition was built without a record handle (program / locator)")
            try:
                e = dispatch_expectation(u, self.record_handle)
            except ValueError as ex:
                # [E3] the builder's own locator refuses an interior word ("interior value the record does not commit"): the positions come
                # from the program + partition + pinned layout instead (E7's oracle hook resolves the same words once wired)
                # [E9, NIGHT 14:40Z -- n2c VU 1365320 `F32MulFtz_v1|batch`] E7's FTZ builder (`ck_ftz.py:517`) spells the same situation
                # "operand gate … of gate …: not a committed word of the record" (its word index covers `commitments.json#trees`, not the
                # interior trees); the verifier's own positions (program + partition + pinned layout: `interior/<req>/<step>` offsets 166/209/234
                # there) are the derivation of record -- the builder's is the cross-check, absent by name for that unit
                if self.positions is None or not any(k in str(ex) for k in ("interior", "not a committed word")):
                    raise
                e = self._from_positions(vu_id, u, str(ex)[:120])
            if self.positions is not None and self.positions.layout is not None:
                # the builder's positions and the verifier's own statement must agree (one derivation, two spellings -> a disagreement is a refusal)
                st = self.positions.statement(u)
                # a program literal operand (E5 §4, `origin == "literal"`) is compiled into the builder's statement, never a position of it
                ops_mine = [v for v, o in zip(st.operands, st.origins) if o != "literal"]
                if getattr(e, "statement", None) is not None and getattr(e, "subcircuit", None) is None:
                    # a statement-bound family (E5's select guest, E7's `StatementExpectation`): positions at LEAF granularity (`offset` = the
                    # leaf's rank, the guest authenticates whole leaves) -- every leaf it binds must be a leaf my statement names, and its
                    # outputs' leaves must be exactly my outputs' leaves
                    mine_in, mine_out = {(v.tree, v.leaf) for v in ops_mine}, {(v.tree, v.leaf) for v in st.outputs}
                    theirs_in, theirs_out = {(v.tree, v.leaf) for v in e.incoming if not _is_literal_tree(v.tree)}, {(v.tree, v.leaf) for v in e.outgoing}
                    if not theirs_in <= mine_in or theirs_out != mine_out:
                        raise ValueError(f"VU {vu_id}: builder leaves in={sorted(theirs_in)} out={sorted(theirs_out)} != the verifier's position "
                                         f"statement (in {len(mine_in)} leaves, out={sorted(mine_out)})")
                else:
                    mine_in = tuple(v.key for v in ops_mine)
                    has_literal = self._has_literal(st)
                    # the literal ids the verifier's expectation lists as PORTS (a literal compiled into the body -- the PV step's ZERO32
                    # accumulator, `steps.PV_ACC_ZERO_KIND` -- is not one; E3's statement still names it at its operand position)
                    n_lit_mine = len({v.leaf for v, o in zip(st.operands, st.origins) if o == "literal"})
                    if len(set(st.operand_gates)) < len(st.operand_gates) or has_literal:
                        # [E9, D32.1] aliased operands: the verifier's statement is per operand position; the body lists each DISTINCT word
                        # once (aliased-body/v1) -- fold by the program's gate identities (steps.alias_identity_expectation) before comparing,
                        # and the builder's body must be the identity-derived aliased body, not the kind's canonical one.
                        # [NIGHT G1 `prescribed-literal/v1`] a literal operand is an incoming word at the verifier's literal-table position
                        from e2e.verifier_v3 import steps as _steps

                        e_mine, _w = _steps.alias_identity_expectation(st, literals=self.literals)
                        mine_in = tuple(v.key for v in e_mine.incoming)
                        n_lit_mine = len({v.key for v in e_mine.incoming if _is_literal_tree(v.tree)})
                        if str(getattr(e, "subcircuit", "") or "").lower() != e_mine.subcircuit:
                            raise ValueError(f"VU {vu_id}: builder body {str(getattr(e, 'subcircuit', ''))[:16]} != the verifier's identity-derived "
                                             f"{'aliased ' if len(set(_w)) < len(_w) else ''}body {e_mine.subcircuit[:16]} (wiring {_w})")
                        # [integrator merge, E3 d7858739] the statement's own identity-deduplicated PORTS (`PositionStatement.ports`) are the same fold
                        # by construction -- two spellings of one derivation; a disagreement is a refusal, never a silent choice (E3's ports leave
                        # literals out; compare the committed ports only)
                        ports, _m = st.ports()
                        committed_mine = tuple(k for k in mine_in if not str(k[0]).startswith("literal/w="))
                        if tuple(v.key for v in ports) != committed_mine:
                            raise ValueError(f"VU {vu_id}: statement ports {[v.key for v in ports]} != identity-folded incoming {list(committed_mine)}")
                    tp = self._table_port(vu_id, u, st)  # [table-layout/v1] the row port both derivations must name (see _table_port)
                    if tp is not None:
                        mine_in = (*mine_in, tp.key)
                    # [E7 SCALE 2026-09-12 22:55Z, E10 q2 `Bf16Add_v1|batch` -- named hunk, composed-body/v1] the body's STRUCTURAL literal ports:
                    # a function of the kind's pinned body spec (`elem_builders.structural_literals`, e.g. the `one` of Bf16Add) and the verifier's
                    # literal table -- not program operands (E3's statement cannot name them), appended after the operands like the table port;
                    # an unpinned instance refuses by name (the builder already did).
                    for lid in self._structural_literals(str(u.get("kind"))):
                        if self.literals is None:
                            raise ValueError(f"VU {vu_id}: {u.get('kind')}: LITERAL_UNPINNED: the body reads the structural literal {lid} and the instance pins no literal tree")
                        lp = self.literals.position(lid)
                        mine_in = (*mine_in, lp.key)
                        n_lit_mine += 1
                    mine = (mine_in, tuple(v.key for v in st.outputs))
                    theirs = (tuple(v.key for v in e.incoming), tuple(v.key for v in e.outgoing))
                    if mine != theirs:
                        raise ValueError(f"VU {vu_id}: builder positions {theirs} != the verifier's position statement {mine}")
                    n_lit_theirs = len({v.key for v in e.incoming if _is_literal_tree(v.tree)})
                    if n_lit_theirs and n_lit_theirs != n_lit_mine:
                        raise ValueError(f"VU {vu_id}: builder names {n_lit_theirs} distinct literal-tree words, the pinned program has {n_lit_mine} distinct literal ids at the operand positions")
            return self._pinned(vu_id, e)
        if cls.startswith("provable"):
            return self._pinned(vu_id, self.deriver.coordinate(u, self.row_index))
        if cls.startswith("input"):
            c = self.checks_by_vu.get(int(vu_id))
            if c is None:
                raise KeyError(f"VU {vu_id}: input unit without a verifier check row")
            e = self.deriver.input_equation(u, c)
            if e is None:
                raise NotAProofCoveredUnit(f"VU {vu_id}: derived-table input (no Input equation; covered by the verifier's opening, V6)")
            return self._pinned(vu_id, e)
        # [A5/6 patch] attention body: the head unit (flag COVERAGE_V1_HEAD_VU, PROPOSED partition change) gets its value-free
        # expectation from the committed id lists; a per-primitive attention kind reads an interior word no record commits -> the
        # selected equation is uncovered, with the interior leaf named (never "no checker" when a checker exists).
        from e2e.coverage_v1.builders import base_kind, head_vu_enabled
        from e2e.coverage_v1.interior_layout import INTERIOR_KINDS

        kind = str(u.get("kind"))
        L = u.get("locate") or {}
        if base_kind(kind) == "AttentionHead_v1|call" and head_vu_enabled():
            from e2e.coverage_v1.attn_head_builder import head_expectation
            from e2e.coverage_v1.builders import LocatedUnit

            if "refused" in L or L.get("unit") != "head":
                raise ValueError(f"VU {vu_id}: head unit not located as exactly one head call: {json.dumps(L)[:120]}")
            unit = LocatedUnit(str(vu_id), kind, u["request_id"], int(L["t"]), layer=int(L["layer"]), head=int(L["head"]), ru_index=int(u.get("ru", 0)))
            return self._pinned(vu_id, head_expectation(unit, self.deriver.attention_row, self.deriver.config))
        # [E3 2026-09-11] a kind without a checker: its positions still resolve (program + partition + pinned layout) -- the VU is
        # RESOLVED but UNCOVERED (no body to expect), named with its positions; never refused as "interior"
        resolved = ""
        if self.positions is not None:
            from e2e.verifier_v3.locator import PositionRefusal
            from veritor.interface.expected import InteriorLayoutUnpinned

            try:
                st = self.positions.statement(u)
                resolved = f"; positions RESOLVED ({', '.join(st.origins)}): out {[v.key for v in st.outputs]}"
            except InteriorLayoutUnpinned:
                raise
            except PositionRefusal as ex:
                resolved = f"; positions REFUSED {ex.code}: {str(ex)[:120]}"
        if kind in INTERIOR_KINDS:
            reads = INTERIOR_KINDS[kind]
            raise NotAProofCoveredUnit(f"VU {vu_id}: kind {kind!r} reads attention-interior words (PROPOSED attn-interior/v0 ports "
                                       f"{sorted({*reads.interior_in, *(() if reads.interior_out is None else (reads.interior_out,))})})"
                                       + ("" if reads.approved_checker else "; no checker in the approved guest") + " -- an uncovered selected equation" + resolved)
        raise NotAProofCoveredUnit(f"VU {vu_id}: kind {u.get('kind')!r} has no checker (legacy class {cls!r}) -- an uncovered selected equation" + resolved)

    @staticmethod
    def _pinned(vu_id: str, e: ExpectedObligation) -> ExpectedObligation:
        """[A10 wiring condition 2] every expectation the verifier hands to coverage pins the body: ``subcircuit`` is never None (the interface's
        default None would skip the body check, ``interface/expected.py:314``).  A None here is a programming error, refused loudly."""

        from e2e.coverage_v1.coverage import is_statement_expectation

        if is_statement_expectation(e):
            # [D31.2, E7's patch `out/eod/e7/patches/e3-statement-bound-locate.diff` adopted] a statement-bound family (E5's `veritor.select-argmax@1`,
            # E7's `StatementExpectation` / `ck_select.SelectStatement`): the expectation IS the guest's statement -- its body is the approved ELF
            # (gate V4), keyed by sha256(statement) (`coverage.expected_statements`) and checked by the family's `check_expected`; there is no
            # subcircuit digest to pin and reading one is the AttributeError that voided Q (E9 finding 23:15Z)
            return e
        if getattr(e, "subcircuit", None) is None:
            raise ValueError(f"VU {vu_id}: expectation without a subcircuit digest (the body check would be skipped) -- refused")
        return e


class RecordCircuit:
    """``coverage_v1.Circuit``: the fixed circuit's family and kind program.  The family is the DECLARED interface family of the audit
    (``--family``); the kind program is the expectation's own canonical body (``programs.gemm_coordinate(K)`` / ``programs.input_equation``),
    re-derived here from the expectation's shape so a disagreement with what the partition declared is a refusal."""

    def __init__(self, family: str, kinds_by_vu: dict[int, str] | None = None) -> None:
        self.family = family
        self.kinds_by_vu = kinds_by_vu or {}  # [CK-ELEM dispatch] vu -> kind, so a registered builder's kind names its own family + body

    def kind_program(self, vu_id: str, e: ExpectedObligation) -> tuple[str, str]:
        from veritor.interface.programs import input_equation

        kind = self.kinds_by_vu.get(int(vu_id))
        if kind is not None:
            from e2e.coverage_v1.builders import derives_expectation, dispatch_kind_program

            if derives_expectation(kind):
                return dispatch_kind_program(kind, e)
        if str(getattr(e, "label", "")).startswith("AttentionHead_v1{"):
            # [A5/6 patch] the head unit: fa2 family (gate set id = pinned MUFU tables), head digest re-derived from (T, D) = the label's
            # parameters through the fixed circuit's lowering -- a disagreement with the expectation's own digest is a refusal upstream
            import re

            from e2e.coverage_v1.attn_head_builder import fa2_gate_set, head_chain_kinds, head_subcircuit_digest

            m = re.match(r"AttentionHead_v1\{T=(\d+),D=(\d+)\}", str(e.label))
            T, D = int(m.group(1)), int(m.group(2))
            return fa2_gate_set().id, head_subcircuit_digest(T, D, head_chain_kinds(T, D))
        if e.incoming:  # GemmCoordinate<K>: K activation words then K weight words
            if len(e.incoming) % 2:
                raise ValueError(f"VU {vu_id}: {len(e.incoming)} incoming values is not 2K")
            return self.family, gemm_coordinate(len(e.incoming) // 2).digest
        (pin,) = e.pins
        if pin is None:
            raise ValueError(f"VU {vu_id}: an obligation without incoming values must be a prescribed Input")
        return self.family, input_equation(int(pin, 16), 4 * len(pin)).digest


def coverage_inputs(sel: Path, commitments: dict[str, Any], *, retained_request_ids: set[str], units: list[dict[str, Any]], checks: list[dict[str, Any]],
                    family: str, recompute_leaves: bool = True, config: dict[str, Any] | None = None, program: Any = None, locator: Any = None,
                    population: Any = None, interior_layout: dict[str, Any] | None = None, adapter: dict[str, Any] | None = None,
                    boundary_request_ids: set[str] | None = None, public_trees: Any = None, literals: Any = None) -> dict[str, Any]:
    """Everything ``coverage_v1`` needs, built once per record: ``trees`` (a ``CommittedTree`` per committed tree of the RETAINED RUs +
    the weights tree + [E3] the step-3 ``interior/`` trees, id lists checked against ``ids_digest``), ``rank_of``, ``partition``, ``circuit``,
    and the VU sets: ``Q_proof`` = the selected VUs a proof must cover (provable + Input-equation units), ``derived_only`` = input VUs of a
    derived table (V6 covers them), ``uncovered`` = selected VUs of retained RUs whose kind has no checker (refusal by itself).  ``problems``
    lists tree failures.

    [E3 2026-09-11] ``population`` (the pinned partition: ``population.v5.json`` document or ``partition_sample`` view) and
    ``interior_layout`` (``protocol_instance.json#commitments.interior_layout``, ``None`` == unpinned) build the :class:`PositionOracle`
    (``"positions"``): ``partition.statement(vu)`` for every VU, interior kinds included; ``positions.word`` is E7's oracle."""
    d = Deriver(sel, commitments, recompute_leaves=recompute_leaves)
    d._config = config  # [A5/6 patch] the pinned program's C (build_pinned_program), for head expectations
    # [D17] `boundary/<rid>` trees too: a registered program's record (E8's CORE) commits its step-1 rows there (record_v5 spelling)
    # [E9 2026-09-12, D32 finding on CORE-live-sampled-B1-03] `boundary/<rid>` trees are STEP-1 commitments of EVERY RU (their roots are
    # pinned before r_q; V3.no_recommit covers all of them), and a retained RU's VU may read a word another RU produced (DemoCore's stage
    # r reads stage r-1's boundary output).  Loading only the retained RUs' boundary trees made such a read fall through to the interior
    # path and refuse `NO_REGION` in the re-verifier while the live step 5 accepted the same statement.  `boundary_request_ids` names
    # the boundary trees to load (default: every committed `boundary/` tree); `ru/` and step-3 `interior/` trees stay retained-only.
    b_ids = set(boundary_request_ids) if boundary_request_ids is not None else None
    tree_ids = [WEIGHTS_TREE] + [t for t in (commitments.get("trees") or {})
                                 if ((t.startswith("ru/") or t.startswith(INTERIOR_TREE_PREFIX)) and t.split("/")[1] in retained_request_ids)
                                 or (t.startswith("boundary/") and (b_ids is None or t.split("/")[1] in b_ids))]
    # E2's step-3 trees live under `commitments.json#interior_roots["<req>/<step>"]` (tree id `interior/<req>/<step>`), not under `trees`
    tree_ids += [f"{INTERIOR_TREE_PREFIX}{k}" for k in (commitments.get("interior_roots") or {})
                 if k.split("/")[0] in retained_request_ids and f"{INTERIOR_TREE_PREFIX}{k}" not in tree_ids]
    committed = d.committed(tree_ids)
    # [E9, NIGHT G1] the VERIFIER-DERIVED public trees (D23 MUFU tables rebuilt against the pin, `prescribed-literal/v1` literal trees from the
    # program): `steps.public_trees(...)` -> {tree: (CommittedTree, ids in rank order)}; `literals` = the steps.LiteralTable (None = unpinned).
    # Without them a statement citing `table/mufu/<name>` or `literal/w=<w>` refused "root … is not a committed tree of J" (finding, 05:30Z).
    if public_trees is None and (literals is not None or (commitments.get("tables") if isinstance(commitments, dict) else None)):
        from e2e.verifier_v3 import steps as _steps

        public_trees, pt_problems = _steps.public_trees(None, commitments, program=program, literal_table=literals)
        committed.problems.extend(f"{t}: {p}" for t, p in ((p.split(":")[0], p) for p in pt_problems))
    for tid, (ct, ids_) in dict(public_trees or {}).items():
        committed.trees[tid] = ct
        committed.ids[tid] = list(ids_)
        committed.ordered.add(tid)
    row_index = output_row_index(committed.ids, d.lm_head_ops)
    units_by_vu = {int(u["vu_index"]): u for u in units}
    checks_by_vu = {int(c["vu_index"]): c for c in checks}
    q_proof, derived_only, uncovered = [], [], []
    for vu, u in units_by_vu.items():
        cls = str(u.get("class", ""))
        if cls.startswith("provable"):
            q_proof.append(vu)
        elif cls.startswith("input"):
            # [A2 ckpt 21:05Z §5.1] classified by the PROGRAM's field (the located Input gate lies in weights.cos_sin, the rope table derived
            # from C, not a checkpoint tensor), never by the prover's `expected: None`; Deriver.input_equation is the single authority
            (derived_only if str((u.get("locate") or {}).get("field", "")).endswith("cos_sin") else q_proof).append(vu)
        else:
            # [A5/6 patch] the head unit (flag) is proof-covered; every other non-provable kind is an uncovered selected equation
            from e2e.coverage_v1.builders import base_kind, head_vu_enabled

            (q_proof if (head_vu_enabled() and base_kind(str(u.get("kind"))) == "AttentionHead_v1|call") else uncovered).append(vu)
    # [CK-ELEM dispatch] the record as an expectation-capable builder reads it on the verifier side: the verifier's program build + locator,
    # the committed trees, the committed id lists (recomputed against ids_digest above) and the record's weights layout -- no prover object
    record_handle = None
    if program is not None and locator is not None:
        from e2e.coverage_v1.ck_elem import RecordHandle

        # [E9, NIGHT 09:50Z -- n2c's `V6.gather` refusal `GATHER_UNRESOLVED: … no capture reader: openings unavailable`] the handle carries the
        # deriver's authenticated TreeCache (the same openings V3.trees_rehashed / V6.inputs read), so a verifier-side check that must OPEN a
        # committed word (gather-equality/v1: the token word, the selected embedding word, the output) can; `RecordHandle.open` sets the
        # same field on the builder side.  Every opening is still authenticated against the committed roots by the check that uses it.
        record_handle = RecordHandle(program, dict(config or {}), locator, commitments, reader=d.reader,
                                     committed_ids={t: list(ids) for t, ids in committed.ids.items()}, profile_id=str(commitments.get("profile_id", "")),
                                     _cache=d.cache)
        # [E7 SCALE 2026-09-12 22:50Z, E10 q2 `Bf16Add_v1|batch: LITERAL_UNPINNED` -- named hunk in E9's file] the handle carries the verifier's
        # literal table from construction (not only once the position oracle exists, below): a composed-body kind's structural literal
        # (`Const32[0x3f800000]_v1`, the `one` of Bf16Add) is read from it by the builder's own expectation (`elem_builders._structural_words`
        # -> `record_literal_table`); `literals` None = unpinned instance -> LITERAL_UNPINNED by name, as before.
        record_handle.literals = literals
    # [E3 2026-09-11] the position oracle: program + partition + pinned layout -> every VU's statement as positions (interior kinds included)
    positions = None
    if population is not None and program is not None and locator is not None:
        shapes: dict[str, tuple[int, ...]] = {}
        for tid in committed.ids:
            if tid.startswith("ru/"):
                try:
                    _, rid, step = tid.split("/")
                    for e in json.loads((sel / "ru" / rid / step / "index.json").read_text())["entries"]:
                        if e.get("shape") is not None:
                            shapes[str(e["canonical_id"])] = tuple(int(x) for x in e["shape"])
                except Exception:  # noqa: BLE001 -- shapes only spell the offset; a missing shape is 1-D
                    pass
        if adapter is None:  # [D1] a registered program's adapter (program.builder.adapter) -- None for Serve_v1 (the vLLM defaults)
            from e2e.verifier_v3.locator import position_adapter

            adapter = position_adapter(config, program, locator)
        positions = PositionOracle(program, locator, population=population, committed_ids={t: list(ids) for t, ids in committed.ids.items()},
                                   commitments=commitments, interior_layout=interior_layout, reader=d.reader,
                                   profile_id=str(commitments.get("profile_id", "")), shapes=shapes, adapter=adapter)
        if record_handle is not None and hasattr(record_handle, "oracles"):  # E7's program-agnostic hook, once landed
            try:
                # [E7 SCALE 2026-09-12, oracles hook LANDED (ck_elem.RecordHandle.oracles / elem_builders.verifier_oracle_chain); named hunk in
                # E9's file] the handle's chain is now LiteralOracle -> ElemLocator -> E3's positions: the literal table it reads is the
                # verifier's (`literals`, steps.LiteralTable; None = unpinned -> a Const operand refuses LITERAL_UNPINNED by name, as before)
                record_handle.literals = literals
                record_handle.oracles = tuple(record_handle.oracles or ()) + (positions.word,)
            except Exception:  # noqa: BLE001
                pass
    kinds_by_vu = {vu: str(u.get("kind", "")) for vu, u in units_by_vu.items()}
    return {"deriver": d, "trees": committed.trees, "rank_of": committed.rank_of(), "tree_problems": committed.problems,
            "partition": RecordPartition(d, units_by_vu=units_by_vu, checks_by_vu=checks_by_vu, row_index=row_index, record_handle=record_handle, positions=positions,
                                         literals=literals),
            "literals": literals,
            "circuit": RecordCircuit(family, kinds_by_vu), "positions": positions,
            "Q_proof": sorted(q_proof), "derived_only": sorted(derived_only), "uncovered": sorted(uncovered)}


# --- [user directive 20:10Z R9 / integrator 22:15Z item 2] V6.inputs: ONE named check path over A2's Deriver.input_equation ------------


def input_checks_v6(deriver: "Deriver", *, units: list[dict[str, Any]], checks: list[dict[str, Any]], prompt_of_request: dict[str, list[int]] | None = None,
                    checkpoint_word=None) -> dict[str, Any]:
    """``V6.inputs`` for every selected Input unit of the retained RUs, in one pass, through A2's :meth:`Deriver.input_equation` (the single
    authority: leaf/word from the verifier's located gate, the committed word from the verifier's own authenticated opening under the
    registered root, ``ok`` recomputed; the prover's ``obligations/input_checks.jsonl`` row is COMPARED field by field -> ``V6.inputs``
    ``InputCheckDisagreement``).  Two verifier-side additions: (a) prompt-token units are pinned by the step-2-authenticated prompt
    (``epoch.inputs``); (b) ``checkpoint_word`` -- the verifier's own copy of the pinned checkpoint, when given -- is a cross-check of the
    committed word against the checkpoint word (the pin itself is the auditor's registered root, V3; the checkpoint is not required).
    Returns ``{"problems", "units", "derived_only", "checkpoint_checked", "extra_rows"}``; any problem is a refusal named ``V6.inputs``."""

    from verity_vllm import identity as vid

    by_vu = {int(c["vu_index"]): c for c in checks}
    problems: list[str] = []
    n_units = n_derived = n_ckpt = 0
    for u in units:
        L = u.get("locate") or {}
        cls = str(u.get("class", ""))
        if not cls.startswith("input") or L.get("kind") not in ("weight", "prompt_token"):
            continue
        vu = int(u["vu_index"])
        n_units += 1
        c = by_vu.get(vu)
        try:
            if L["kind"] == "prompt_token":
                rid, t = str(u["request_id"]), int(L["t"])
                cid_b = str(vid.boundary_id(deriver.reader.profile_id, rid, "prompt_token", t))
                o = deriver.cache.open(cid_b)
                committed = int.from_bytes(o.value, "little", signed=True) & 0xFFFFFFFF
                prompt = (prompt_of_request or {}).get(rid)
                if prompt is None or t >= len(prompt):
                    problems.append(f"V6.inputs: vu {vu}: prompt_token[{t}] of {rid[:8]}: no step-2-authenticated prompt (epoch.inputs) to pin it")
                elif (int(prompt[t]) & 0xFFFFFFFF) != committed:
                    problems.append(f"V6.inputs: vu {vu}: prompt_token[{t}] of {rid[:8]}: committed {committed} != declared {int(prompt[t])}")
                if c is not None and (c.get("kind") != "prompt_token" or c.get("t") != t or c.get("committed") != committed):
                    problems.append(f"V6.inputs: vu {vu}: input_checks.jsonl prompt row disagrees with the verifier's (kind/t/committed)")
                continue
            e = deriver.input_equation(u, c)  # A2: raises InputCheckDisagreement (V6.inputs) / ValueError; None for the derived table
            if e is None:
                n_derived += 1
            if checkpoint_word is not None and L["kind"] == "weight" and e is not None:
                word, _src = checkpoint_word(str(L["canonical_id"]), int(L["offset"]))
                if word is not None:
                    n_ckpt += 1
                    committed_word = int(e.pins[0], 16) if e.pins and e.pins[0] is not None else None  # the pin: big-endian hex of the committed word
                    if committed_word is not None and word != committed_word:
                        problems.append(f"V6.inputs: vu {vu}: verifier checkpoint word {word} != committed word {committed_word} under the registered root")
        except InputCheckDisagreement as ex_:
            problems.append(str(ex_))
        except Exception as ex_:  # noqa: BLE001
            problems.append(f"V6.inputs: vu {vu}: {type(ex_).__name__}: {str(ex_)[:160]}")
    extra = sorted(set(by_vu) - {int(u["vu_index"]) for u in units if str(u.get("class", "")).startswith("input")})
    if extra:
        problems.append(f"V6.inputs: input_checks.jsonl carries {len(extra)} rows for VUs the verifier did not select as Input units")
    return {"problems": problems, "units": n_units, "derived_only": n_derived, "checkpoint_checked": n_ckpt, "extra_rows": len(extra)}
