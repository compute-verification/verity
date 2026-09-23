"""[COORD R17-39 (c), Daniel / reqval] The deliberately dumb, independent reference for the alias RULE -- how the table is DERIVED from a
Program (the manifest-build half).  moe's `test_alias_reference_fixture.py` (M-0799) is the reference for the CHAIN that consumes the table
(binding numbering -> committed_reader / CommittedStore -> oracle_compare over a real MatchOracle) with expectations computed by counting;
this file is the other half: which Program modules become one instance at all, and that a different logical producer never does.

Daniel: the reader (`CommittedStore`) and the oracle (`oracle_compare`) sharing ONE alias mechanism (`alias_table` / `resolve_op_path`
over the manifest's `op_path_aliases`) prevents drift AND lets them share the same bug -> keep ONE deliberately dumb reference
fixture/enumerator for alias behaviour -- "different logical producer / version -> never merged solely because bits / offsets coincide"
-- that tests the COMMON MECHANISM, not every production record.

The enumerator below knows nothing of `required_manifest` / `required_values`: it reads Program rows as strings.  Its rule, in full:
  * the module of a row is the row name before the first '/'; the collector path drops the wrapper prefix ('model.' here);
  * a module is a candidate shared module iff its leaf name is DECLARED shared (`rotary_emb`: vLLM's `get_rope` cache) -- a declaration,
    never an inference from bytes or offsets;
  * the logical producer of a module is the SET of (member hint, full spec string incl. parameters) of its rows;
  * candidates with the SAME logical producer are one instance: canonical = the lowest layer's path, invocation = layer order within
    the group; candidates with DIFFERENT logical producers are different instances -- never merged, whatever their names look like.
The production mechanism (`build_manifest` on both engines -> `op_path_aliases`; `alias_table` + `resolve_op_path`, the one table the
reader AND the oracle resolve through) must agree with it, positive and negative."""
from __future__ import annotations

import copy
import re

import pytest

from verity_capture.commit import required_manifest as RM
from verity_capture.commit import sampled_replay as SR
from verity_capture.commit.tests.test_required_manifest import _program

SHARED_LEAVES = ("rotary_emb",)                                                    # the declaration (vLLM get_rope), not an inference
WRAPPER = "model."


def _layer(path: str) -> int:
    m = re.search(r"\.layers\.(\d+)\.", path)
    return int(m.group(1)) if m else -1


def dumb_alias_enumerator(rows: list[dict], *, shared_leaves=SHARED_LEAVES, wrapper: str = WRAPPER) -> dict[str, list]:
    """Program rows -> {aliased collector path: [canonical collector path, invocation]} by the rule in the module docstring."""
    producers: dict[str, set] = {}
    for r in rows:
        module = r["name"].split("/", 1)[0]
        if module.rsplit(".", 1)[-1] not in shared_leaves:
            continue
        hint = r["name"].split("/")[-1].split("[")[0]                            # '.../rotary_embedding/query[3]' -> 'query'
        producers.setdefault(module, set()).add((hint, r["spec"]))
    groups: dict[frozenset, list[str]] = {}
    for module, sig in producers.items():
        groups.setdefault(frozenset(sig), []).append(module)
    out: dict[str, list] = {}
    for members in groups.values():
        members = sorted(members, key=_layer)
        canon = members[0]
        for k, m in enumerate(members):
            if k:
                out[m[len(wrapper):] if m.startswith(wrapper) else m] = [canon[len(wrapper):] if canon.startswith(wrapper) else canon, k]
    return out


def _production_tables(prog: dict) -> dict[str, dict]:
    return {e: RM.build_manifest(prog, request_id="r0", engine=e)["op_path_aliases"] for e in ("inventory", "traversal")}


def test_positive_one_shared_instance_the_production_table_equals_the_dumb_enumerator_on_both_engines():
    prog = _program(LP=4, T=2, layers=4)
    ref = dumb_alias_enumerator(prog["instances"]["rows"])
    assert ref == {f"model.layers.{k}.self_attn.rotary_emb": ["model.layers.0.self_attn.rotary_emb", k] for k in (1, 2, 3)}
    for engine, table in _production_tables(prog).items():
        assert table == ref, engine
    # the common mechanism resolves exactly as the table says, and a path with no entry is invocation 0 of itself -- nothing else
    aliases = SR.alias_table({"op_path_aliases": ref})
    assert SR.resolve_op_path(aliases, "model.layers.3.self_attn.rotary_emb") == ("model.layers.0.self_attn.rotary_emb", 3)
    assert SR.resolve_op_path(aliases, "model.layers.0.self_attn.rotary_emb") == ("model.layers.0.self_attn.rotary_emb", 0)
    assert SR.resolve_op_path(aliases, "model.layers.2.mlp.down_proj") == ("model.layers.2.mlp.down_proj", 0)


def test_negative_a_different_logical_producer_is_never_merged_because_the_family_name_coincides():
    """Layer 1's rotary rows describe ANOTHER producer (D=32 where the others are D=64: a different RotaryEmbedding instance -- vLLM's
    get_rope caches on the parameters).  Before R17-39 (c) every RoPE* module was folded into one instance by the family name alone:
    layer 1 -> (layers.0, 1), layer 2 -> (layers.0, 2).  Now: layers 0 and 2 are one instance (layer 2 -> (layers.0, 1)); layer 1 is
    its own (no alias entry, invocation 0 at its own path) -- on both engines, equal to the dumb enumerator; its identities sit at its
    own path so a collector that had committed them under layers.0 would fail BY NAME (identities_without_rows), never be satisfied
    by a neighbour's words."""
    prog = _program(LP=4, T=2, layers=3)
    mutated = copy.deepcopy(prog)
    n = 0
    for r in mutated["instances"]["rows"]:
        if ".layers.1.self_attn.rotary_emb" in r["name"] and r["spec"].startswith("RoPE_v1"):
            r["spec"] = r["spec"].replace("D=64", "D=32")
            n += 1
    assert n > 0
    ref = dumb_alias_enumerator(mutated["instances"]["rows"])
    assert ref == {"model.layers.2.self_attn.rotary_emb": ["model.layers.0.self_attn.rotary_emb", 1]}
    for engine in ("inventory", "traversal"):
        man = RM.build_manifest(mutated, request_id="r0", engine=engine)
        assert man["op_path_aliases"] == ref, engine
        rot = [i for i in man["identities"] if "rotary_emb" in i["op_path"]]
        by_path = {}
        for i in rot:
            by_path.setdefault(i["op_path"], set()).add(int(i["invocation"]))
        assert by_path == {"model.layers.0.self_attn.rotary_emb": {0, 1}, "model.layers.1.self_attn.rotary_emb": {0}}, (engine, by_path)
        assert man["complete"] is True and not man["unmodelled"]               # a well-formed Program with two instances is complete
        # the F-r17b-38 reachability rule agrees with the two-group table: every invocation >= 1 identity is the image of an alias entry
        images = {(str(c), int(k)) for c, k in man["op_path_aliases"].values()}
        assert all((str(i["op_path"]), int(i["invocation"])) in images for i in man["identities"] if int(i["invocation"]) >= 1)
        # the widths tell the two producers apart on the identities themselves (D=32 rows are half the width)
        w0 = {int(i["width"]) for i in rot if i["op_path"].startswith("model.layers.0.")}
        w1 = {int(i["width"]) for i in rot if i["op_path"].startswith("model.layers.1.")}
        assert w0 and w1 and max(w1) < max(w0)
    # a second version of the family is a different producer too (RoPE_v2 rows on layer 2): three instances, no alias at all
    versioned = copy.deepcopy(prog)
    for r in versioned["instances"]["rows"]:
        if ".layers.2.self_attn.rotary_emb" in r["name"] and r["spec"].startswith("RoPE_v1"):
            r["spec"] = r["spec"].replace("RoPE_v1", "RoPE_v2")
        if ".layers.1.self_attn.rotary_emb" in r["name"] and r["spec"].startswith("RoPE_v1"):
            r["spec"] = r["spec"].replace("D=64", "D=32")
    assert dumb_alias_enumerator(versioned["instances"]["rows"]) == {}
    for engine in ("inventory", "traversal"):
        assert RM.build_manifest(versioned, request_id="r0", engine=engine)["op_path_aliases"] == {}, engine


def test_negative_the_declaration_off_means_no_alias_whatever_the_rows_look_like():
    prog = _program(LP=4, T=2, layers=3)
    assert dumb_alias_enumerator(prog["instances"]["rows"], shared_leaves=()) == {}
    for engine in ("inventory", "traversal"):
        assert RM.build_manifest(prog, request_id="r0", engine=engine, shared_rope=False)["op_path_aliases"] == {}, engine


def test_coinciding_bytes_and_offsets_never_make_an_alias_the_reader_reads_the_named_invocation_only():
    """[the A2 sentence itself] two entries of one canonical path at invocations 0 and 1 whose committed words are BYTE-IDENTICAL are two
    identities: the reader returns invocation k's words for op_path k and nothing collapses; a path with no alias entry and no binding at
    invocation 0 is missing by name even though a neighbour's bytes would 'fit'."""
    import numpy as np
    ROT = "model.layers.0.self_attn.rotary_emb"
    W = 8
    same = np.arange(W, dtype=np.uint16).tobytes()                              # identical bytes for both invocations
    entries = [{"name": f"{ROT}/0#{inv}", "stream_offset": 0, "element_offset": 0, "encoding": {"dtype": "bf16"},
                "identity": {"request_id": "r0", "step": 0, "op_path": ROT, "output_member": "0", "invocation": inv, "element_range": [0, W],
                             "family": "instance_outputs"}} for inv in (0, 1)]
    bmap = {"entries": entries}
    reads: list[str] = []

    def committed(step: int, name: str, lo: int, hi: int) -> bytes | None:
        reads.append(name)
        return same[lo:hi]

    aliases = {"model.layers.1.self_attn.rotary_emb": [ROT, 1]}
    store = SR.CommittedStore(bmap, committed, aliases)
    # layer 1's path -> invocation 1's entry BY NAME; layer 0's -> invocation 0's; the bytes being equal changes nothing about which was read
    w1, _ = store.words("r0", 0, "model.layers.1.self_attn.rotary_emb", "0", 0, W)
    assert w1 is not None and reads[-1] == f"{ROT}/0#1"
    w0, _ = store.words("r0", 0, ROT, "0", 0, W)
    assert w0 is not None and reads[-1] == f"{ROT}/0#0"
    assert w0.tolist() == w1.tolist()                                            # identical bytes -- two identities all the same
    # a path with no alias entry names invocation 0 of ITSELF: nothing bound there -> missing by name, never the neighbour's fitting bytes
    n_reads = len(reads)
    w2, why = store.words("r0", 0, "model.layers.2.self_attn.rotary_emb", "0", 0, W)
    assert w2 is None and "no committed identity for (r0, step 0, model.layers.2.self_attn.rotary_emb/0)" in why and len(reads) == n_reads
    # and an alias pointing at an invocation with no entry is missing by name too, with the bound set
    st2 = SR.CommittedStore(bmap, committed, {"model.layers.5.self_attn.rotary_emb": [ROT, 5]})
    w5, why5 = st2.words("r0", 0, "model.layers.5.self_attn.rotary_emb", "0", 0, W)
    assert w5 is None and "invocation 5" in why5 and "[0, 1]" in why5 and len(reads) == n_reads
