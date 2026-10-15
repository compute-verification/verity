"""Padding steps -- the INACTIVE rows of a padded Commit, synthesized by ENABLEMENT at finalize (R17-43 explicit enablement; LIFTING-SPEC
v0.7 A9.3 / B8 / B9; PADDED-TABLE candidates #3 / #4 `smollm2` earlystop / mixedstop).  One rule, two consumers:

  * the pod committer (`native_host.finalize`): after the run's executed steps are folded, the padding steps of every request are appended
    to the step list (their layouts published, their tree levels retained for openings) and the run root is the fold over ALL S steps of
    the padded Program -- one root, one binding map, every identity of RequiredValues(Lift(P)) bound (the record's `manifest.json.gz`);
  * the laptop conversion over a PULLED record (`padded_map_of_record`): the record's executed-step layouts + step roots, the same padding
    steps synthesized from its manifest, the same fold -> the padded map + coverage over the manifest, with the record's own step roots
    for the executed prefix.  (The run root of the conversion is NOT the record's: the pulled record carries no geometry digest and its
    run root was the fold over the executed steps only; the conversion's root binds S = cap steps.  The one-root claim of PADDED-TABLE is
    made by the pod consumer; the conversion demonstrates the population, the bytes and the map.)

The rule (SPEC A9.1 N2, A9.3, B9 R4'): request r's Program owns steps [arrival, arrival + cap); its EXECUTED PREFIX of record is
`executed_prefix_of_record` (moe's rule: declared LAG + served tokens + declared EOS, never inferred from the trace); every identity the
manifest requires of r at a step >= arrival + executed_prefix_len is ⊥ BY ENABLEMENT (the step's call has `a_c = Active(0)`): its committed
word is the canonical ⊥ of the lifted width, `1 || 0^w`, in the lifted representation (A11 / R17-6: the TRUE lifted width w+1, no split to
fit 32 bits) -- a bf16 element (w = 16) is the u32 word 0x0001_0000, a 32-bit element (u32 / i32 / f32) the u64 word 1 << 32, a 64-bit
element (i64 token id) the 16-byte word 1 << 64.  The synthesized stream of a padding step is tensor-major in the manifest's identity
order, chunked into `chunk`-byte `pos_leaf` leaves and folded with the production rule (`scheme.fold_levels`); the step root is
`scheme.step_root(program_digest, ctx_digest(run_id, step), geo_digest, layout_digest, N, merkle)` -- exactly the executed steps' rule, so a
padding step is opened and verified by the production verifier like any other.  Nothing here reads the trace to decide what is
inactive: the manifest (the Program's population) and the executed prefix of record decide.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from verity_vllm.commit.binding import HIDDEN_FAMILIES, coverage_check
from verity_vllm.check.executed_prefix import executed_prefix_of_record, request_steps_of_record
from verity_vllm.commit import scheme

PADDING_RULES_VERSION = "padded-v1"                 # LIFTING-SPEC v0.7 B8: the padded request Program's rules version
PADDED_MAP_SCHEMA = "padded-binding-map/1"
CHUNK = 256                                         # chunk-leaf-v1 with the GPU tree

# committed dtype -> (payload width w, lifted container bytes, lifted dtype label).  The lifted word is w+1 bits wide (A11), carried in
# the smallest power-of-two container that holds it; ⊥ = 1 << w (tag 1, zero payload: the canonical ⊥, A9.3 / R17-6).
LIFTED_CONTAINER: dict[str, tuple[int, int, str]] = {
    "bf16": (16, 4, "lifted17/u32"), "f16": (16, 4, "lifted17/u32"), "fp16": (16, 4, "lifted17/u32"),
    "u32": (32, 8, "lifted33/u64"), "i32": (32, 8, "lifted33/u64"), "f32": (32, 8, "lifted33/u64"), "int32": (32, 8, "lifted33/u64"),
    "float32": (32, 8, "lifted33/u64"), "uint32": (32, 8, "lifted33/u64"), "bfloat16": (16, 4, "lifted17/u32"), "float16": (16, 4, "lifted17/u32"),
    "i64": (64, 16, "lifted65/u128"), "u64": (64, 16, "lifted65/u128"), "int64": (64, 16, "lifted65/u128"),
    "u8": (8, 2, "lifted9/u16"), "i8": (8, 2, "lifted9/u16"), "fp8": (8, 2, "lifted9/u16"), "e4m3": (8, 2, "lifted9/u16"),
}


def bot_word(dtype: str) -> bytes:
    """The canonical ⊥ of one element of committed dtype `dtype`, in its lifted container (little-endian)."""
    w, nb, _ = LIFTED_CONTAINER[dtype]
    return (1 << w).to_bytes(nb, "little")


@dataclass
class PaddingMeta:
    """A published layout row of a padding step -- the fields `native_host.TensorMeta` / `binding.build_binding_map` read."""
    ordinal: int
    name: str
    dtype: str                  # the LIFTED container label (`lifted17/u32`, ...): the map's `encoding.dtype` says what the bytes are
    shape: tuple[int, ...]
    nbytes: int
    chunk: int
    first_leaf: int
    n_leaves: int
    stream_off: int
    committed_dtype: str = ""   # the identity's dtype in the executed steps (bf16, u32, i64): the payload width comes from it
    elements: int = 0
    block_kind: str = "padding"  # [revb S-r17b-59 (i)] the block's domain, READ from the layout row (never inferred from a label): executed | padding
    nb: int = 1                  # the CMT-2 chunk header's NB for this block (PADDING_BLOCK): 1 = padding block, 0 = executed block


@dataclass
class PaddingStep:
    """What the pod consumer appends: `root` / `n_leaves` / `n_bytes` like a `StepCommit`, `metas` for `_layouts[step]`, `levels` for
    `_levels[step]` (opening generation), `stream` = the bytes (kept by the conversion; the pod may drop them: every word is ⊥)."""
    step: int
    metas: list[PaddingMeta]
    stream: bytes
    levels: list[list[bytes]]
    merkle_root: bytes
    root: bytes
    n_leaves: int
    n_bytes: int
    identities: list[dict[str, Any]] = field(default_factory=list, repr=False)
    leaf_base: int = 0          # > 0: APPENDED to an executed step's tree (a mixed step, B > 1): its leaves rank after the executed ones
    stream_base: int = 0
    appended: bool = False      # True: `root` is the padding SUBTREE's bind (conversion witness); the step's root is the refold of
    #                             executed leaf digests + these leaves on the pod (`refold_step`), which a pulled record cannot do
    layout: str = "chunk-leaf-v1"   # the leaf rule of record (GPU tree chunk header) -- see `leaf_digests`


def tensor_name(op_path: str, member: str) -> str:
    """The committed tensor name `build_binding_map` splits back into (op_path, member) -- `split_name`'s inverse."""
    if op_path.startswith(HIDDEN_FAMILIES):
        return f"{op_path}/{member}"
    return op_path if member == "out" else f"{op_path}/{member}"


PADDING_NAME_SUFFIX = "/bot"


def padding_tensor_name(op_path: str, member: str, request_id: str) -> str:
    """The name of the PADDING tensor that materializes identity (request_id, op_path, member): the executed member's name + `/<rid>/bot`
    (a shared plane's member already reads `<rid>/<g...>/bot`, kept).  A padding tensor is its OWN tensor (revb F-r17b-66): at a MIXED
    step it sits beside the executed member of the same identity name, and a committed reader asked for the executed member by its bare
    name must find ONE meta (S-r17b-19 n3: two metas under one name -> None) -- on the first #4 pod run the padding tensor carried the
    executed name and the C2 boundary linkage read `model.embed_tokens/in` at every mixed step as 'not retained' (24/44 linked)."""
    base = tensor_name(op_path, member)
    if base.endswith(PADDING_NAME_SUFFIX):
        return base
    return f"{base}/{request_id}{PADDING_NAME_SUFFIX}"


def member_plane_identity(shared: dict[str, Any], rid: str, engine_step: int, arrival: int) -> dict[str, Any]:
    """[B9.1 (1) rule (a); revb F-r17b-64] The ⊥ per-sequence plane of a listed member INACTIVE at a B>1 shared hidden launch row.
    The launch row (`request_id "*"`, `requests` listed, `engine_step` e) is the manifest's PACKING of the members' own attention rows at
    e; the tap commits per-sequence planes and the requirement is op-level per member (>= 1 plane of that op at e bound to it).  A member
    the executed prefix of record places OUTSIDE e has that call disabled for its sequence (R17-43): its plane is a required value, ⊥ of
    the plane's committed dtype, SIZED BY THE MEMBER'S OWN single-sequence geometry (the Program's Attention row for that request alone),
    never by the executed launch (whose shape is the other members' business):
      `members_geometry[rid]` on the row (the manifest declares it)        -> `geometry_source: declared`
      else the row's launch geometry reduced to one sequence               -> `geometry_source: launch-share` (HB / n_seq; M, NB, BN, D
         of the launch: the union's T_max, a DECLARED quantity of the manifest at full cap, so Q(P) stays fixed across stopping points)
      else the row's own `element_range` width (a fixture without geometry) -> `geometry_source: row-width`
    Named `<op_path>/<rid>/<g...>/bot` so `split_name` gives (op_path, member) and the op-level requirement binds it."""
    from verity_vllm.query.manifest.format import mat_total_words
    geo = (shared.get("members_geometry") or {}).get(rid)
    src = "declared"
    if geo is None and isinstance(shared.get("geometry"), dict) and all(k in shared["geometry"] for k in ("HB", "M", "NB", "BN", "D")):
        g = shared["geometry"]
        n_seq = max(1, int(g.get("n_seq") or len(shared.get("requests") or [1])))
        hb = int(g["HB"]) // n_seq
        if hb * n_seq != int(g["HB"]):
            raise ValueError(f"{shared.get('op_path')} at engine step {engine_step}: launch HB {g['HB']} is not a multiple of n_seq {n_seq}")
        geo = {"HB": hb, "M": int(g["M"]), "NB": int(g["NB"]), "BN": int(g["BN"]), "D": int(g["D"]), "n_seq": 1,
               "seqlen_k": g.get("seqlen_k"), "rows": (int(g["rows"]) // n_seq) if g.get("rows") is not None else None, **({"ms": True} if g.get("ms") else {})}
        src = "launch-share"
    if geo is not None:
        words = mat_total_words(int(geo["HB"]), int(geo["M"]), int(geo["NB"]), int(geo["BN"]), int(geo["D"]), bool(geo.get("ms")))
        gname = f"g{geo['HB']}x{geo['M']}x{geo['NB']}" + ("" if (int(geo["BN"]), int(geo["D"])) == (128, 64) else f"x{geo['BN']}x{geo['D']}")
    else:
        lo, hi = shared["element_range"]
        if hi is None or int(hi) <= int(lo):
            raise ValueError(f"{shared.get('op_path')} at engine step {engine_step}: shared row without geometry or a bounded element range")
        words, gname, src = int(hi) - int(lo), str(shared.get("output_member") or "plane"), "row-width"
    return {"family": shared.get("family"), "program_digest": shared["program_digest"], "request_id": rid, "step": int(engine_step) - int(arrival),
            "engine_step": int(engine_step), "invocation": int(shared.get("invocation", 0)), "op_path": str(shared["op_path"]),
            "output_member": f"{rid}/{gname}/bot", "element_range": [0, int(words)], "rank": int(shared.get("rank", 0)),
            "dtype": str(shared.get("dtype") or "u32"), "shared_of": {"requests": list(shared.get("requests") or []), "output_member": shared.get("output_member")},
            "geometry": geo, "geometry_source": src}


def padding_steps_of_record(identities: Iterable[dict[str, Any]], epr: dict[str, Any], arrivals: dict[str, int]) -> dict[int, list[dict[str, Any]]]:
    """{engine_step: [manifest identity, ...]} of every required identity OUTSIDE its request's executed prefix of record: ⊥ by enablement.
    `epr` = `executed_prefix_of_record(...)` (must be ok: an unplaceable prefix -- undeclared LAG, unverifiable stop -- refuses by name,
    nothing is synthesized); `arrivals` = {r<i>: arrival engine step}.  Weights (registered input) are never padding.  Engine step of a
    per-request row = its request-relative `step` + the request's arrival (the loader's rule; every emitted identity carries `engine_step`).
    A B>1 SHARED hidden launch row (`request_id "*"`, `requests`, `engine_step`) becomes one ⊥ per-sequence plane PER LISTED MEMBER inactive
    at that engine step (`member_plane_identity`, rule (a)): at a MIXED step the launch ran over the others and binds to them; at an IDLE
    step nobody launched and every member's plane is ⊥.  Members are the row's `requests` (normalised to r<i> by the caller)."""
    if not epr.get("ok"):
        raise ValueError("executed prefix of record not ok: " + "; ".join(epr.get("problems") or ["(unnamed)"]))
    executed = {rid: set(steps) for rid, steps in request_steps_of_record(epr, arrivals).items()}
    out: dict[int, list[dict[str, Any]]] = {}
    for ident in identities:
        if ident.get("family") == "weights" or ident.get("op_path") == "weights":
            continue
        st = ident.get("step")
        if ident.get("request_id") == "*" and isinstance(ident.get("requests"), list) and ident.get("engine_step") is not None:
            es = int(ident["engine_step"])
            for rid in ident["requests"]:
                rid = str(rid)
                if rid not in executed:
                    raise ValueError(f"shared row {ident.get('op_path')} at engine step {es} lists request {rid} which the executed prefix of record does not name")
                if es in executed[rid]:
                    continue
                out.setdefault(es, []).append(member_plane_identity(ident, rid, es, int(arrivals.get(rid, 0))))
            continue
        if st == "*" or st is None:
            continue
        rid = str(ident["request_id"])
        if rid not in executed:
            raise ValueError(f"manifest identity of request {rid} which the executed prefix of record does not name")
        arrival = int(arrivals.get(rid, 0))
        es = int(st) + arrival
        if es in executed[rid]:
            continue
        out.setdefault(es, []).append(dict(ident, engine_step=es))
    return dict(sorted(out.items()))


def padding_layout(step_identities: list[dict[str, Any]], *, chunk: int = CHUNK, leaf_base: int = 0, ordinal_base: int = 0) -> tuple[list[PaddingMeta], bytes]:
    """The synthesized stream of one padding step: per identity one tensor of `hi - lo` canonical ⊥ words of the identity's lifted
    container, tensor-major in identity order (the manifest's order), each tensor at a chunk-aligned stream offset so a rank names
    exactly one (tensor, byte range) like the executed steps' packed layout.  `leaf_base` / `ordinal_base` > 0: the padding is appended
    to an executed step's layout (its first leaf ranks after the executed leaves; the returned stream is the padding part only)."""
    metas: list[PaddingMeta] = []
    parts: list[bytes] = []
    off = 0
    base = leaf_base * chunk
    for k, ident in enumerate(step_identities, start=ordinal_base):
        lo, hi = ident["element_range"]
        if hi is None or hi < lo:
            raise ValueError(f"identity {ident.get('op_path')}/{ident.get('output_member')} at step {ident.get('step')}: unbounded element range")
        n = int(hi) - int(lo)
        dtype = str(ident.get("dtype") or "")
        if dtype not in LIFTED_CONTAINER:
            raise ValueError(f"identity {ident.get('op_path')}/{ident.get('output_member')}: committed dtype {dtype!r} has no lifted container")
        _w, nb, label = LIFTED_CONTAINER[dtype]
        word = bot_word(dtype)
        data = word * n
        rows = ident.get("rows")
        shape = (int(rows), n // int(rows)) if rows and n % int(rows) == 0 and int(rows) > 0 else (n,)
        pad = (-off) % chunk
        if pad:
            parts.append(b"\0" * pad)
            off += pad
        nbytes = n * nb
        metas.append(PaddingMeta(ordinal=k, name=padding_tensor_name(str(ident["op_path"]), str(ident["output_member"]), str(ident["request_id"])), dtype=label, shape=shape,
                                 nbytes=nbytes, chunk=chunk, first_leaf=(base + off) // chunk, n_leaves=max(1, (nbytes + chunk - 1) // chunk) if nbytes else 0,
                                 stream_off=base + off, committed_dtype=dtype, elements=n))
        parts.append(data)
        off += nbytes
    return metas, b"".join(parts)


def layout_digest(metas: list[PaddingMeta]) -> bytes:
    """`native_host.NativeHostCommitter.layout_digest` verbatim: (ordinal, name, dtype, shape, nbytes, chunk, first_leaf, n_leaves, stream_off)."""
    doc = [[m.ordinal, m.name, m.dtype, list(m.shape), m.nbytes, m.chunk, m.first_leaf, m.n_leaves, m.stream_off] for m in metas]
    return hashlib.sha256(json.dumps(doc, separators=(",", ":")).encode()).digest()


def ctx_digest(run_id: str, step: int) -> bytes:
    """`native_host._ctx_digest` verbatim."""
    return hashlib.sha256(f"{run_id}/step={step}".encode()).digest()


PADDING_BLOCK = 1      # CMT-2 chunk-header field NB of a PADDING leaf (executed leaves of the GPU tree carry NB = 0): a padding leaf is in its own
#                        domain -- no executed leaf of any step, index or content hashes equal to a padding leaf, and vice versa


def leaf_digests(stream: bytes, *, step: int, chunk: int = CHUNK, layout: str = "chunk-leaf-v1", leaf_base: int = 0, block_kind: int = PADDING_BLOCK) -> list[bytes]:
    """The padding block's leaves over `stream`, by the committer's leaf rule of record.  `chunk-leaf-v1` / `-v2` = the GPU tree
    (`native_host.commit_block_offline` / `_gpu_layout`): the block zero-padded to a chunk multiple, leaf i = SHA-256(CMT-2 chunk header
    (HB = step, M = the block's padded word count (v1) or 0 (v2), NB = `block_kind`, launch_tag = step, chunk_index = leaf_base + i,
    chunk_words = chunk/4, src_mask = 0) || chunk i); anything else = the host path's `pos_leaf(chunk)` (last leaf shorter).  The
    executed block of a step is hashed with NB = 0 and its own M by the committer; the padding block (the whole step when nobody executed
    it, or appended after the executed leaves of a mixed step) with NB = PADDING_BLOCK and ITS OWN M -- each block's header binds that
    block's length, the layout digest + N bind the step's shape over both."""
    if layout.startswith("chunk-leaf-v"):
        v2 = layout == "chunk-leaf-v2"
        padded = -(-len(stream) // chunk) * chunk if stream else chunk
        buf = bytes(stream) + b"\0" * (padded - len(stream))
        words = 0 if v2 else padded // 4

        class _L:                                   # duck-typed StreamLayout, as native_host._gpu_layout builds it (NB marks the block kind)
            HB, M, NB = step, words, block_kind
        return [scheme.chunk_leaf(scheme.chunk_header(_L, step, leaf_base + i, chunk // 4, 0), buf[i * chunk:(i + 1) * chunk]) for i in range(padded // chunk)]
    return [scheme.pos_leaf(stream[o:o + chunk]) for o in range(0, len(stream), chunk)] or [scheme.pos_leaf(b"")]


def padding_step_commit(step: int, identities: list[dict[str, Any]], *, program_digest: bytes, run_id: str, geo_digest: bytes,
                        chunk: int = CHUNK, leaf_base: int = 0, ordinal_base: int = 0, layout: str = "chunk-leaf-v1") -> PaddingStep:
    """Synthesize + commit one padding step with the production rule: the record's leaf rule (`layout`) per chunk, fold_levels, step_root.
    `leaf_base` > 0 (a mixed step: other requests executed it): the padding leaves rank after the executed ones and `root` binds the
    padding SUBTREE (ctx `step=<s>/padding`) as the conversion's witness; the step's one root is `refold_step` on the pod."""
    metas, stream = padding_layout(identities, chunk=chunk, leaf_base=leaf_base, ordinal_base=ordinal_base)
    if layout.startswith("chunk-leaf-v") and stream:
        stream = stream + b"\0" * ((-len(stream)) % chunk)          # the GPU tree pads a block to a chunk multiple
    leaves = leaf_digests(stream, step=step, chunk=chunk, layout=layout, leaf_base=leaf_base)
    levels = scheme.fold_levels(list(leaves))
    merkle = levels[-1][0]
    ctx = ctx_digest(run_id, step) if leaf_base == 0 else hashlib.sha256(f"{run_id}/step={step}/padding".encode()).digest()
    root = scheme.step_root(program_digest, ctx, geo_digest, layout_digest(metas), len(leaves), merkle)
    return PaddingStep(step=step, metas=metas, stream=stream, levels=levels, merkle_root=merkle, root=root, n_leaves=len(leaves),
                       n_bytes=len(stream), identities=list(identities), leaf_base=leaf_base, stream_base=leaf_base * chunk, appended=leaf_base > 0,
                       layout=layout)


def refold_step(executed_leaf_digests: list[bytes], executed_metas: list[Any], ps: PaddingStep, *, program_digest: bytes, run_id: str,
                geo_digest: bytes) -> tuple[list[list[bytes]], bytes]:
    """The pod consumer's one root of a MIXED step: the executed step's retained leaf digests (`_levels[step][0]`) + the padding leaves,
    refolded, bound with the layout digest over executed metas + padding metas (the padding metas' first_leaf / stream_off already rank
    after the executed ones).  Returns (levels, root); the pod replaces `_levels[step]`, extends `_layouts[step]`, sets the StepCommit's root."""
    if ps.leaf_base != len(executed_leaf_digests):
        raise ValueError(f"padding of step {ps.step} was laid out after {ps.leaf_base} leaves but the executed step has {len(executed_leaf_digests)}")
    leaves = list(executed_leaf_digests) + list(ps.levels[0])      # executed block (NB = 0, its M) then the padding block (NB = 1, its M)
    levels = scheme.fold_levels(leaves)
    root = scheme.step_root(program_digest, ctx_digest(run_id, ps.step), geo_digest, layout_digest(list(executed_metas) + ps.metas), len(leaves), levels[-1][0])
    return levels, root


def opening(ps: PaddingStep, leaf_rank: int) -> tuple[bytes, list[bytes | None]]:
    """(leaf bytes, path) of leaf `leaf_rank` of a padding step, from the retained levels -- verified with `fold_path(pos_leaf(leaf), rank, path)`."""
    leaf = ps.stream[leaf_rank * ps.metas[0].chunk if ps.metas else 0:][: (ps.metas[0].chunk if ps.metas else 0)] if ps.stream else b""
    return leaf, list(scheme.path_in_levels(ps.levels, leaf_rank))


def leaf_digest_of(ps: PaddingStep, leaf: bytes, leaf_rank: int) -> bytes:
    """The digest of padding leaf `leaf_rank` (subtree-relative) under the step's leaf rule -- what a verifier recomputes from the opened bytes."""
    if ps.layout.startswith("chunk-leaf-v"):
        chunk = ps.metas[0].chunk if ps.metas else CHUNK
        v2 = ps.layout == "chunk-leaf-v2"

        class _L:
            HB, M, NB = ps.step, (0 if v2 else ps.n_bytes // 4), PADDING_BLOCK
        return scheme.chunk_leaf(scheme.chunk_header(_L, ps.step, ps.leaf_base + leaf_rank, chunk // 4, 0), leaf)
    return scheme.pos_leaf(leaf)


def verify_opening(ps: PaddingStep, leaf_rank: int) -> bool:
    leaf, path = opening(ps, leaf_rank)
    return scheme.fold_path(leaf_digest_of(ps, leaf, leaf_rank), leaf_rank, path) == ps.merkle_root


def padding_entries(ps: PaddingStep, *, layout_version: str = "chunk-leaf-v1", step_root: bytes | None = None) -> list[dict[str, Any]]:
    """The binding-map entries of a padding step, in `binding.build_binding_map`'s entry schema (+ `activity`): one per synthesized tensor,
    its identity the manifest identity it materialises (request, step, invocation, op_path, member, element_range, rank verbatim -- a B>1
    request's row is its own tensor, so no row split is derived).  Both consumers emit exactly these.  `step_root`: the pod consumer's ONE
    root of a mixed step (the refold); the conversion, which cannot refold, leaves it None and carries the padding subtree root."""
    out: list[dict[str, Any]] = []
    root_hex = step_root.hex() if step_root is not None else (None if ps.appended else ps.root.hex())
    for m, ident in zip(ps.metas, ps.identities):
        lo, hi = ident["element_range"]
        out.append({
            "identity": {"program_digest": ident["program_digest"], "request_id": ident["request_id"], "step": int(ident.get("engine_step", ident["step"])),
                         "invocation": int(ident.get("invocation", 0)), "op_path": ident["op_path"], "output_member": ident["output_member"],
                         "element_range": [int(lo), int(hi)], "rank": int(ident.get("rank", 0))},
            "kind": "leaf", "step_root": root_hex, "layout": layout_version, "chunk": int(m.chunk),
            **({"padding_subtree_root": ps.root.hex(), **({} if step_root is not None else {"step_root_note": "mixed step: the one step root is the pod's refold (refold_step)"})} if ps.appended else {}),
            "leaf_lo": int(m.first_leaf), "leaf_hi": int(m.first_leaf + m.n_leaves), "stream_offset": int(m.stream_off), "width_bytes": int(m.nbytes),
            "encoding": {"dtype": m.dtype, "shape": list(m.shape), "committed_dtype": m.committed_dtype, "bot_word_hex": bot_word(m.committed_dtype).hex()},
            "element_offset": 0, "tensor_ordinal": int(m.ordinal), "name": m.name, "activity": "bot-by-enablement",
            "block_kind": m.block_kind, "nb": int(m.nb), "family": ident.get("family"),
            **({"engine_step": int(ident["engine_step"]), "request_step": ident["step"]} if "engine_step" in ident else {}),
            **({"shared_of": ident["shared_of"], "geometry_source": ident.get("geometry_source")} if "shared_of" in ident else {}),
        })
    return out


def padding_for_committer(com: Any, *, manifest: dict[str, Any], requests: list[dict[str, Any]], sampling: dict[str, Any],
                          served_tokens: dict[str, list[int]] | None, stop: dict[str, Any] | None, arrivals: dict[str, int] | None = None) -> list[PaddingStep]:
    """The POD consumer's hook (`NativeHostCommitter.padding_hook`; run inside `finalize()` after every executed step is hashed, before the
    run-root fold): the padding steps of THIS run.  `manifest` = the required-value manifest document (cap shape; `identities` rows);
    `requests` / `sampling` = the workload's (request ids `r<index>`, caps, declared arrivals, as commit_delta.binding_record reads them);
    `served_tokens` = the run's served tokens per workload request id; `stop` = stop_rule_of_record's {eos, lag}.  The executed prefix is
    `executed_prefix_of_record` -- one rule for every consumer; a prefix it cannot place REFUSES by name and nothing is synthesized.
    A step the committer executed (B > 1: another request's rows) gets its padding APPENDED (leaf / ordinal base = the step's) for
    `finalize` to refold; a step nobody executed becomes a padding step of its own."""
    ids = manifest.get("entries") or manifest.get("rows") or manifest.get("identities") or []
    order = {str(r.get("request_id")): f"r{int(r.get('index', i))}" for i, r in enumerate(requests)}
    caps = {f"r{int(r.get('index', i))}": int(r.get("max_tokens") or sampling.get("max_tokens") or 1) for i, r in enumerate(requests)}
    arr = dict(arrivals) if arrivals else {f"r{int(r.get('index', i))}": int(r.get("arrive_step") or 0) for i, r in enumerate(requests)}

    def _logical(rid: str) -> str:
        rid = str(rid)
        if rid in order:
            return order[rid]
        base = rid.rsplit("-", 1)[0]
        if base in order:
            return order[base]
        hits = [v for k, v in order.items() if rid.startswith(k)]
        return hits[0] if len(hits) == 1 else rid

    served = None if served_tokens is None else {_logical(str(k)): [int(t) for t in v] for k, v in served_tokens.items()}
    epr = executed_prefix_of_record(sorted(caps, key=lambda k: int(k[1:])), served, caps=caps,
                                    eos_of_record=(stop or {}).get("eos"), lag=(stop or {}).get("lag"))
    if not epr.get("ok"):
        raise RuntimeError("padded finalize REFUSED: executed prefix of record not placeable: " + "; ".join(epr.get("problems") or ["(unnamed)"]))
    per_request = [d for d in ids if d.get("request_id") not in (None, "*") and d.get("family") != "weights" and d.get("step") != "*"
                   and d.get("element_range") and d["element_range"][1] is not None]
    # [rule (a), F-r17b-64] the B>1 shared hidden launch rows: their listed members normalised to r<i> (workload ids | indices | r<i>)
    shared = [dict(d, requests=[_logical(str(r)) if not str(r).startswith("r") or not str(r)[1:].isdigit() else str(r) for r in d["requests"]])
              for d in ids if d.get("request_id") == "*" and isinstance(d.get("requests"), list) and d.get("engine_step") is not None
              and d.get("family") != "weights"]
    by_step = padding_steps_of_record(per_request + shared, epr, arr)
    layout = com.layout if getattr(com, "gpu_tree", False) else "host-pos-leaf"
    out: list[PaddingStep] = []
    for s, sids in by_step.items():
        kw: dict[str, Any] = dict(program_digest=com.program_digest, run_id=com.run_id, geo_digest=com._geo_digest, chunk=com.chunk, layout=layout)
        if s in com._done:                                             # a MIXED step: appended after the executed leaves; finalize refolds
            kw.update(leaf_base=int(com._done[s].n_leaves), ordinal_base=len(com._layouts[s]))
        out.append(padding_step_commit(s, sids, **kw))
    if out:
        print(f"[padding] executed prefix of record {{" + ", ".join(f"{k}: {v['executed_prefix_len']}/{v['cap']}" for k, v in epr['requests'].items())
              + f"}} LAG {epr['lag']}; {len(out)} padding steps ({sum(1 for ps in out if ps.appended)} mixed), "
              f"{sum(ps.n_bytes for ps in out):,} bytes / {sum(ps.n_leaves for ps in out):,} leaves of bot", flush=True)
    return out


def padded_binding_map_of_committer(com: Any, run: Any, program_digest_hex: str, *, executed_prefix: dict[str, Any] | None = None,
                                    program_rules_version: str | None = None, **build_kw: Any) -> dict[str, Any]:
    """The POD consumer's padded map (`padded-binding-map/1`) after `finalize()` on a committer with padding steps registered
    (`NativeHostCommitter.add_padding_steps`): `binding.build_binding_map` over the EXECUTED layouts only (a padding tensor is never
    row-split by the request_rows heuristic -- its identity is the manifest's), then one entry per padding tensor with the step's ONE root
    (the refold, for a mixed step), `activity` on every entry, digest over the entries.  `build_kw` = build_binding_map's keyword arguments
    (request_rows, request_id, rank, layout_version, chunk, weight_registration, collector).  `program_rules_version`: the Program of
    record's (a Program whose digest carries the padding declaration is `padded-v1`; an ordinary Program's is None -- the padding rule is
    named separately as `padding_rules_version`, never stamped on the Program: R17-42)."""
    from verity_vllm.commit.binding import build_binding_map

    pads: dict[int, PaddingStep] = dict(getattr(com, "_padding", {}) or {})
    exec_layouts: dict[int, list] = {}
    for s, metas in com._layouts.items():
        ps = pads.get(s)
        if ps is None:
            exec_layouts[s] = list(metas)
        elif ps.appended:                                              # mixed: the executed metas are the ones before the padding block
            exec_layouts[s] = [m for m in metas if m.stream_off is None or m.stream_off < ps.stream_base]
    bmap = build_binding_map(exec_layouts, run.steps, run.run_root, program_digest_hex, **build_kw)
    entries: list[dict[str, Any]] = []
    for e in bmap["entries"]:
        entries.append(dict(e, activity=("active" if e["kind"] == "leaf" else "registered-input"), **({"block_kind": "executed", "nb": 0} if e["kind"] == "leaf" else {})))
    for s in sorted(pads):
        entries.extend(padding_entries(pads[s], layout_version=bmap["layout"], step_root=run.steps[s].root))
    pm: dict[str, Any] = {"schema": PADDED_MAP_SCHEMA, "rules_version": bmap.get("rules_version"),
                          "program_rules_version": program_rules_version, "padding_rules_version": PADDING_RULES_VERSION,
                          "consumer": "pod finalize (native_host.add_padding_steps; padding-steps/1, LIFTING-SPEC v0.7 B9.1 rule)",
                          "record_map_rules_version": bmap.get("rules_version"), "program_digest": program_digest_hex,
                          "run_root": run.run_root.hex(), "rank": int(bmap.get("rank", 0)), "layout": bmap["layout"],
                          "collector": dict(bmap.get("collector") or {}, padding="synthesized by enablement at finalize (padding_steps.py; pod consumer)"),
                          "entries": entries}
    pm["padding"] = {"rule": "LIFTING-SPEC v0.7 B9.1: every identity of a step outside its request's executed prefix of record is the canonical "
                             "bot of its lifted width, synthesized at finalize; the executed prefix is executed_prefix_of_record, never inferred",
                     "steps": sorted(pads), "mixed_steps": sorted(s for s, ps in pads.items() if ps.appended),
                     "S": len(run.steps), "consumer": "pod (native_host.finalize)", "executed_prefix": executed_prefix}
    canon = json.dumps(pm["entries"], sort_keys=True, separators=(",", ":"), default=str).encode()
    pm["digest"] = hashlib.sha256(canon).hexdigest()
    pm["entries_n"] = len(pm["entries"])
    return pm


# ---------------------------------------------------------------------- the laptop conversion over a pulled record

@dataclass
class _StepStub:
    root: bytes


def _identity_tuple(d: dict[str, Any]) -> tuple:
    lo, hi = d["element_range"]
    return (d["program_digest"], d["request_id"], d["step"], int(d.get("invocation", 0)), d["op_path"], d["output_member"],
            (int(lo), -1 if hi is None else int(hi)), int(d.get("rank", 0)))


def padded_map_of_record(record_dir: str | Path, *, run_id: str | None = None, geo_digest: bytes | None = None, chunk: int = CHUNK) -> dict[str, Any]:
    """Convert a pulled Commit record (manifest + binding map + declared workload Program) into the padded commit: executed steps kept
    with the record's own step roots, padding steps synthesized by enablement, one run root over cap steps, one padded binding map,
    coverage over the record's manifest.  Returns {"padded_map", "padding": [PaddingStep], "report", "executed_prefix"}."""
    rd = Path(record_dir)
    man = json.load(gzip.open(rd / "manifest.json.gz", "rt"))
    bmap = json.load(gzip.open(rd / "commit" / "binding_map_p0.json.gz", "rt"))
    wp = json.loads((rd / "build_workload" / "workload_program.json").read_text())
    reqs = wp.get("requests") or []
    ids = man["identities"]
    pdg = bmap["program_digest"]
    program_digest = scheme.program_field(pdg)
    run_id = run_id or f"padded-conversion/{rd.name}"
    geo = geo_digest or hashlib.sha256(b"cmt-integ/offline-geo/" + run_id.encode()).digest()

    # the executed prefix of record: declared LAG + declared EOS + the served tokens the record carries (never inferred from the trace)
    lag = (wp.get("lag") or {}).get("value")
    caps = {str(r.get("request_id", f"r{i}")): int(r.get("cap") or r.get("max_tokens") or 0) for i, r in enumerate(reqs)}
    arrivals = {str(r.get("request_id", f"r{i}")): int(r.get("arrive_step", 0) or 0) for i, r in enumerate(reqs)}
    eos = (wp.get("stop_rule") or {}).get("eos_token_ids")
    served = _served_tokens_of_record(rd, wp)
    epr = executed_prefix_of_record(list(caps), served, caps=caps, eos_of_record=eos, lag=lag)
    if not epr["ok"]:
        return {"ok": False, "why": "executed prefix of record not placeable: " + "; ".join(epr["problems"]), "executed_prefix": epr}

    # executed steps: the record's layouts + step roots; padding steps: synthesized
    leaf_entries = [e for e in bmap["entries"] if e["kind"] == "leaf"]
    by_step: dict[int, list[dict[str, Any]]] = {}
    for e in leaf_entries:
        by_step.setdefault(int(e["identity"]["step"]), []).append(e)
    executed_steps = sorted(by_step)
    exec_of_rule = request_steps_of_record(epr, arrivals)
    union_rule = sorted({s for v in exec_of_rule.values() for s in v})
    if executed_steps != union_rule:
        return {"ok": False, "why": f"the record committed steps {executed_steps[:8]}.. but the executed prefix of record is {union_rule[:8]}..: "
                                    f"execution and the declared rule disagree (never reconciled by inference)", "executed_prefix": epr}
    pad_ids = padding_steps_of_record(ids, epr, arrivals)
    padding: list[PaddingStep] = []
    for s, idents in pad_ids.items():
        if s in by_step:                         # a MIXED step (B > 1): other requests executed it; the padding is appended to its tree
            es = by_step[s]
            leaf_base = max(int(e["leaf_hi"]) for e in es)
            ordinal_base = max(int(e["tensor_ordinal"]) for e in es) + 1
            padding.append(padding_step_commit(s, idents, program_digest=program_digest, run_id=run_id, geo_digest=geo, chunk=chunk,
                                               leaf_base=leaf_base, ordinal_base=ordinal_base, layout=bmap["layout"]))
        else:                                    # a step nobody executed: the padding IS the step
            padding.append(padding_step_commit(s, idents, program_digest=program_digest, run_id=run_id, geo_digest=geo, chunk=chunk, layout=bmap["layout"]))
    steps: dict[int, _StepStub | None] = {s: _StepStub(bytes.fromhex(es[0]["step_root"])) for s, es in by_step.items()}
    mixed_steps = sorted(ps.step for ps in padding if ps.appended)
    for ps in padding:
        if ps.appended:
            steps[ps.step] = None                # the record's root covers the executed leaves only; the refold needs the retained leaf digests (pod)
        else:
            steps[ps.step] = _StepStub(ps.root)
    S = max(steps) + 1
    if sorted(steps) != list(range(S)):
        return {"ok": False, "why": f"steps {sorted(steps)[:8]}.. are not the contiguous range of the padded Program", "executed_prefix": epr}
    run_root: bytes | None = None
    if not mixed_steps:
        run_root = scheme.run_root(program_digest, geo, [steps[s].root for s in range(S)])   # type: ignore[union-attr]
    pad_steps = {ps.step for ps in padding}
    # the padded map: the record's leaf entries verbatim (their step roots are the record's; per-request rows already split by the
    # record's own rule) + one synthesized entry per padding tensor (its identity is the manifest's, so a B>1 request's row is its own
    # tensor: no row-split heuristic) + the registered input
    entries: list[dict[str, Any]] = []
    for e in leaf_entries:
        entries.append(dict(e, activity="active", block_kind="executed", nb=0))
    for ps in padding:
        entries.extend(padding_entries(ps, layout_version=bmap["layout"]))
    for e in bmap["entries"]:
        if e["kind"] != "leaf":
            entries.append(dict(e, activity="registered-input"))
    # [revb M-1335 CORRECTION 2 / R17-42] the map's program_digest is the RECORD's (unlifted) Program: its program_rules_version stays the
    # record's -- stamping `padded-v1` on a conversion would be the post-hoc re-label R17-42 forbids; the padding rule is named beside it
    pm: dict[str, Any] = {"schema": PADDED_MAP_SCHEMA, "rules_version": bmap.get("rules_version"),
                          "program_rules_version": bmap.get("program_rules_version"), "conversion": "padding-steps/1 (LIFTING-SPEC v0.7 B9.1 rule)",
                          "padding_rules_version": PADDING_RULES_VERSION,
                          "record_map_rules_version": bmap.get("rules_version"), "program_digest": pdg,
                          "run_root": (run_root.hex() if run_root is not None else None),
                          "run_root_note": (None if run_root is not None else f"mixed steps {mixed_steps[:6]}..: their one root is the refold of the executed "
                                            f"leaf digests + the padding leaves (refold_step), which the pulled record does not retain -- pod-only"),
                          "rank": int(bmap.get("rank", 0)), "layout": bmap["layout"],
                          "collector": dict(bmap.get("collector") or {}, padding="synthesized by enablement at finalize (padding_steps.py)"),
                          "entries": entries}
    pm["padding"] = {"rule": "LIFTING-SPEC v0.7 A9.1 N2 / A9.3 / B9: every identity of a step outside its request's executed prefix of record is "
                             "the canonical ⊥ of its lifted width, synthesized at finalize; the executed prefix is executed_prefix_of_record "
                             "(declared LAG + declared EOS + served tokens), never inferred from the trace",
                     "steps": sorted(pad_steps), "executed_steps": executed_steps, "S": S, "lag": lag, "run_id": run_id,
                     "geo_digest": geo.hex(), "geo_digest_is_the_records": geo_digest is not None,
                     "step_roots_of_record_kept": [s for s in executed_steps]}
    canon = json.dumps(pm["entries"], sort_keys=True, separators=(",", ":"), default=str).encode()
    pm["digest"] = hashlib.sha256(canon).hexdigest()
    pm["entries_n"] = len(pm["entries"])

    # coverage over the record's manifest, request-aware (the weights family = registered input, checked against the weights root
    # registration as in the record's rule; B>1 shared hidden planes matched by engine step over their listed requests)
    n_weights = sum(1 for d in ids if d.get("family") == "weights")
    cov = coverage_over_manifest(ids, pm)
    cov_before = coverage_over_manifest(ids, bmap)
    # [rule (a)] the padding may only ADD member planes where the executed prefix of record places a member outside the step: the
    # executed launch bindings are untouched (same count before and after), the bot planes are the difference
    cov["shared_executed_bindings_unchanged"] = cov["shared"]["members_bound_executed_n"] == cov_before["shared"]["members_bound_executed_n"]
    cov["ok"] = bool(cov["ok"]) and cov["shared_executed_bindings_unchanged"]
    bot_words = sum(m.elements for ps in padding for m in ps.metas)
    bot_bytes = sum(ps.n_bytes for ps in padding)
    active_bytes = sum(int(e["width_bytes"]) for e in leaf_entries)
    active_words = sum(int(e["identity"]["element_range"][1]) - int(e["identity"]["element_range"][0]) for e in leaf_entries)
    by_container = Counter(m.dtype for ps in padding for m in ps.metas)
    report = {
        "ok": bool(cov["ok"]),
        "record": rd.name, "program_digest": pdg, "run_root_padded": (run_root.hex() if run_root is not None else None), "run_root_of_record": bmap["run_root"],
        "mixed_steps": mixed_steps, "one_root_on_laptop": run_root is not None,
        "executed_prefix": epr["requests"], "lag": lag,
        "steps": {"executed": len(executed_steps), "padding": len(padding), "S": S},
        "coverage_over_manifest": {"before": {k: cov_before[k] for k in ("ok", "expected_n", "bound_n", "missing_n", "shared")},
                                   "padded": cov,
                                   "weights_identities_excluded": n_weights,
                                   "rule": "the record's (required_manifest.manifest_coverage_check): weights = registered input, checked "
                                           "against the weights root registration (weights_of_record), not the step layouts"},
        "words": {"active": active_words, "bot_by_enablement": bot_words},
        "bytes": {"active_committed": active_bytes, "bot_synthesized": bot_bytes, "padded_total": active_bytes + bot_bytes,
                  "overhead_vs_executed_prefix": (active_bytes + bot_bytes) / active_bytes if active_bytes else None},
        "leaves": {"active": sum(1 for _ in leaf_entries) and sum(max(int(e["leaf_hi"]) - int(e["leaf_lo"]), 0) for e in leaf_entries),
                   "padding": sum(ps.n_leaves for ps in padding)},
        "containers": dict(by_container),
        "padding_step_roots": {ps.step: (ps.root.hex() + ("  [padding subtree of a mixed step]" if ps.appended else "")) for ps in padding},
        "openings_checked": {ps.step: all(verify_opening(ps, r) for r in (0, ps.n_leaves // 2, ps.n_leaves - 1)) for ps in padding},
        "bot_words_canonical": all(_all_bot(ps) for ps in padding),
        "labels": ["lifted-padded", "CPU Commit-only conversion over the Build+Match of record: the executed steps' roots are the record's, "
                   "the padding steps are synthesized here, the run root binds S = cap steps and is the CONVERSION's (geo digest not in the pull)"],
    }
    return {"ok": report["ok"], "padded_map": pm, "padding": padding, "report": report, "executed_prefix": epr}


def _all_bot(ps: PaddingStep) -> bool:
    for m in ps.metas:
        word = bot_word(m.committed_dtype)
        o = m.stream_off - ps.stream_base
        seg = ps.stream[o:o + m.nbytes]
        if seg != word * m.elements:
            return False
    return True


# ------------------------------------------------------------------------------------------------------ the inactive stratum (C2 population)

INACTIVE_RULE = ("R17-42 / B9.1: on a PADDED record the population of record is the FULL cap -- RequiredValues(P, Q_module_v1) with every "
                 "identity outside a request's executed prefix of record a REQUIRED value assigned tau(e) = bot (R17-43: the call's enable is "
                 "Active(0)).  Those identities are the named stratum `inactive_by_enablement` beside reached / uncovered (accounted = reached + "
                 "uncovered + inactive == |Q(P)|); their check is the committed word == enc(bot) of the lifted width, entry by entry, word by "
                 "word (a bot-by-enablement entry whose leaf is NOT the bot word FAILS); no Program evaluation -- the relation is the enable.  "
                 "There is no not-executed class on a padded record (not_executed_by_rule.n 0 on both sides, renamed, never dropped).")

SIZING_RULE = ("S-r17b-67 C1 (revb M-1429): a bot plane's element count must equal the MEMBER'S OWN Program Attention row at e. "
               "`declared` (members_geometry[rid] on the shared row) IS that; `launch-share` (HB / n_seq with the LAUNCH's M = the union's "
               "T_max) equals it only when the members' kv geometry at e coincide -- a named fallback, not the rule; `row-width` is "
               "fixture-only.  Sign-against: every bot plane `declared`, or the record COUNTS the others and the cell carries the label.")


def bot_plane_sizing(pads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """[revb S-r17b-67 C1, M-1565 (4)] How the ⊥ planes of a shared-plane family were SIZED, counted on the record (the padded map is an
    excluded artefact; `geometry_source` lives per entry and never reached runs.jsonl / the README before this): the histogram of
    `geometry_source` over the family's bot-by-enablement entries, the per-sequence geometry names (`g<HB>x<M>x<NB>[x<BN>x<D>]`, from the
    entry's `identity.output_member`) and the words per plane, plus `all_declared` -- the C1 label the PADDED-TABLE cell must carry."""
    pads = list(pads)
    src = Counter(str(e.get("geometry_source") or "unstated") for e in pads)
    gname: Counter[str] = Counter()
    words: Counter[int] = Counter()
    for e in pads:
        ident = e.get("identity") if isinstance(e.get("identity"), dict) else {}
        om = str(ident.get("output_member") or e.get("name") or "")
        parts = om.split("/")
        gname[next((p for p in parts if p.startswith("g") and "x" in p), parts[-2] if len(parts) >= 2 else om)] += 1
        er = ident.get("element_range")
        if isinstance(er, (list, tuple)) and len(er) == 2 and er[1] is not None:
            words[int(er[1]) - int(er[0])] += 1
        else:
            words[-1] += 1
    return {"planes": len(pads), "geometry_source": dict(sorted(src.items())), "all_declared": bool(pads) and set(src) == {"declared"},
            "plane_geometry": dict(sorted(gname.items())), "words_per_plane": {str(k): v for k, v in sorted(words.items())}, "rule": SIZING_RULE}


def padded_population(sr: dict[str, Any], bmap: dict[str, Any], committed: Any, epr: dict[str, Any], *, max_report: int = 8) -> dict[str, Any]:
    """[revb M-1419 (1), sign-against for fix (a); reqval M-1318 third consumer] Rewrite a `sampled_replay` result of a PADDED Commit into the
    padded record's population: the identities the replay classified `not_executed_by_rule` (request-relative steps >= n_exec: they left the
    pool before any draw -- the same identities the padding materialised) become `population.inactive_by_enablement`, `not_executed_by_rule`
    reads n 0 with `renamed_to`; then (i) EVERY bot-by-enablement entry of the map is read back through `committed` (`<name>#ord<ordinal>`: a
    padding tensor beside an executed one of the same name at a mixed step) and checked word for word == bot_word(committed_dtype); (ii) the
    ONE-MECHANISM reconciliation, kept: the Commit's padding classification (entries by family) against the population's inactive bucket by
    family -- equal or a named problem.  Returns the record block (`ok`, the counts, the problems); mutates `sr` in place."""
    from verity_vllm.check import oracle_compare as _OC
    ordinal_sep = getattr(_OC, "ORDINAL_SEP", None)                 # moe F-moe-r17-06 (int-8): `<name>#ord<k>` names the k-th meta of a step exactly
    pop = sr.setdefault("population", {}) if isinstance(sr.get("population"), dict) else sr.setdefault("population", {})
    ne = dict(pop.get("not_executed_by_rule") or {})
    n_ne = int(ne.get("identities") or 0)
    # sampled_replay writes `not_executed_rule` / `not_executed_by_family` INSIDE `population` (the pod's sampled_replay_p0.json, #3 05:33Z);
    # an older shape had them at the top level -- both are read, and renamed where they are
    holder = pop if isinstance(pop.get("not_executed_rule"), dict) else sr
    ner = holder.get("not_executed_rule") if isinstance(holder.get("not_executed_rule"), dict) else None
    requests = (ner or {}).get("requests")
    if not requests and isinstance(ne.get("by_request"), dict):                      # the bucket's own per-request facts (a_r, lag, n_exec, cap)
        requests = {r: {k: v for k, v in d.items() if k in ("a_r", "lag", "n_exec", "cap", "arrive", "steps_not_executed", "identities")}
                    for r, d in ne["by_request"].items() if isinstance(d, dict)}
    inactive = {"rule": INACTIVE_RULE, "identities": n_ne, "by_family": dict(ne.get("by_family") or {}), "totals": ne.get("totals"),
                "by_request": ne.get("by_request"), "requests": requests,
                "from": "sampled_replay.population.not_executed_by_rule (the pool exclusion at n_exec), renamed"}
    pop["inactive_by_enablement"] = inactive
    pop["not_executed_by_rule"] = {"identities": 0, "n": 0, "applied": False, "renamed_to": "inactive_by_enablement", "rule": INACTIVE_RULE,
                                   "scope": ne.get("scope")}
    if ner is not None:
        holder["not_executed_rule"] = dict(ner, n=0, renamed_to="inactive_by_enablement")
    holder_f = pop if "not_executed_by_family" in pop else sr
    if "not_executed_by_family" in holder_f:
        holder_f["inactive_by_family"], holder_f["not_executed_by_family"] = holder_f["not_executed_by_family"], {}
    # (i) the word check over every padding entry of the map
    pads = [e for e in bmap.get("entries", []) if e.get("activity") == "bot-by-enablement"]
    names_at: Counter = Counter()                                         # same-name metas of a step (a mixed step's executed + padding tensor)
    for e in bmap.get("entries", []):
        st_e = (e.get("identity") or {}).get("step") if isinstance(e.get("identity"), dict) else None
        if e.get("name") is not None and isinstance(st_e, int):          # a registered input / weights entry has no per-step name
            names_at[(int(st_e), str(e["name"]))] += 1
    checked = words = 0
    mismatches: list[dict[str, Any]] = []
    unread: list[dict[str, Any]] = []
    # [revb M-1482 (ii)] the dtypes ARE on the record: per committed dtype, the lifted container the bot word was read back through, the
    # bot word's hex, entries checked and words read -- so 'router_ids i32 / router_w f32 -> lifted33/u64, bf16 -> lifted17/u32' is a
    # line of the record, not of the reader
    by_dtype: dict[str, dict[str, Any]] = {}
    for e in pads:
        dt = (e.get("encoding") or {}).get("committed_dtype")
        if dt not in LIFTED_CONTAINER:
            mismatches.append({"step": e["identity"]["step"], "name": e["name"], "why": f"no lifted container for {dt!r}"}); continue
        _w, nb, _label = LIFTED_CONTAINER[dt]
        n_words = int(e["width_bytes"]) // nb
        bd = by_dtype.setdefault(str(dt), {"container": _label, "lifted_width_bits": _w + 1, "container_bytes": nb, "bot_word_hex": bot_word(dt).hex(),
                                          "padding_entries": 0, "checked_entries": 0, "checked_words": 0, "members": set()})
        bd["padding_entries"] += 1
        bd["members"].add(str((e.get("identity") or {}).get("output_member") or "?"))
        st = int(e["identity"]["step"])
        if ordinal_sep is not None:
            got = committed(st, f"{e['name']}{ordinal_sep}{int(e['tensor_ordinal'])}", 0, int(e["width_bytes"]))
        elif names_at[(st, str(e["name"]))] == 1:                    # an older reader: the bare name is exact only when it names ONE meta of the step
            got = committed(st, str(e["name"]), 0, int(e["width_bytes"]))
        else:
            got = None                                               # two metas of the name (a mixed step) and no ordinal naming: unread, named
        if got is None:
            if len(unread) < max_report:
                unread.append({"step": e["identity"]["step"], "name": e["name"], "ordinal": e["tensor_ordinal"]})
            continue
        checked += 1
        words += n_words
        bd["checked_entries"] += 1
        bd["checked_words"] += n_words
        want = bot_word(dt) * n_words
        if bytes(got) != want:
            bad = next((i for i in range(n_words) if bytes(got[i * nb:(i + 1) * nb]) != bot_word(dt)), None)
            if len(mismatches) < max_report:
                mismatches.append({"step": e["identity"]["step"], "name": e["name"], "request_id": e["identity"]["request_id"], "first_bad_word": bad,
                                   "got_hex": bytes(got[bad * nb:(bad + 1) * nb]).hex() if bad is not None else None, "want_hex": bot_word(dt).hex()})
    # (ii) one mechanism: the Commit's padding classification by family vs the population's inactive bucket by family
    commit_by_family = Counter(str(e.get("family") or "unstated") for e in pads)
    pop_by_family = {str(k): int(v) for k, v in (inactive["by_family"] or {}).items()}
    problems: list[str] = []
    if unread:
        problems.append(f"{len(unread)}+ padding entries not readable through the committed reader (e.g. {unread[0]})")
    if mismatches:
        problems.append(f"{len(mismatches)}+ padding entries whose committed word is NOT the bot word (R17-42 s5): e.g. {mismatches[0]}")
    # a BATCH-LEVEL family (request_id '*', the shared hidden launch planes at B>1): the population counts a plane identity inactive only when
    # EVERY listed member is past its stop (idle step) and states the (plane, past member) CELLS beside it (`plane_cells_not_executed.cells`,
    # sampled_replay / moe M-0468); the Commit's rule (a) pads one bot plane per inactive listed member at MIXED and IDLE steps = the cells.
    # The cells are the number the padding reconciles against (revb M-1419 (3): F-r17b-64 surfaces here the moment (b) lands)
    tot_by_fam = ((inactive.get("totals") or {}).get("by_family") or {}) if isinstance(inactive.get("totals"), dict) else {}
    fams = sorted(set(commit_by_family) | set(pop_by_family))
    # [revb M-1482 (i)] the KIND of each family's padding on the Commit side, stated on the record: `shared-plane (rule a)` when its bot tensors
    # are per-member planes of a batch-level `*` row (`shared_of`), else `per-request` (one tensor per request-level identity)
    commit_kind = {f: ("shared-plane (rule a)" if any(e.get("shared_of") for e in pads if str(e.get("family") or "unstated") == f) else "per-request")
                   for f in fams if commit_by_family.get(f)}
    by_family = {}
    for f in fams:
        cells = ((tot_by_fam.get(f) or {}).get("plane_cells_not_executed") or {}) if isinstance(tot_by_fam.get(f), dict) else {}
        if "cells" in cells:
            by_family[f] = {"commit_padding_entries": int(commit_by_family.get(f, 0)), "population_inactive": int(cells["cells"]), "basis": "cells",
                            "population_inactive_plane_identities": int(pop_by_family.get(f, 0)), "population_planes_fully_inactive": int(cells.get("identities", 0))}
        else:
            by_family[f] = {"commit_padding_entries": int(commit_by_family.get(f, 0)), "population_inactive": int(pop_by_family.get(f, 0)), "basis": "identities"}
        if f in commit_kind:
            by_family[f]["commit_kind"] = commit_kind[f]
        if commit_kind.get(f, "").startswith("shared-plane"):
            by_family[f]["bot_plane_sizing"] = bot_plane_sizing([e for e in pads if str(e.get("family") or "unstated") == f])
    if inactive["identities"] or pads:
        for f, d in by_family.items():
            if d["commit_padding_entries"] != d["population_inactive"]:
                problems.append(f"family {f}: the Commit padded {d['commit_padding_entries']} entries, the population names {d['population_inactive']} inactive "
                                f"{'(plane, member) cells' if d['basis'] == 'cells' else 'identities'}")
    n_exec = {r: {"commit": (epr.get("requests") or {}).get(r, {}).get("executed_prefix_len"), "population": (v or {}).get("n_exec")}
              for r, v in ((inactive.get("requests") or {}).items())}
    for r, d in n_exec.items():
        if d["commit"] is not None and d["population"] is not None and int(d["commit"]) != int(d["population"]):
            problems.append(f"{r}: n_exec {d['commit']} (Commit) != {d['population']} (population)")
    for bd in by_dtype.values():
        bd["members"] = sorted(bd["members"])[:32]
    out = {"ok": not problems, "rule": INACTIVE_RULE, "padding_entries": len(pads), "checked_entries": checked, "checked_words": words,
           "unread_n": len(unread), "mismatches_n": len(mismatches), "mismatches": mismatches[:max_report], "unread": unread[:max_report],
           "by_dtype": by_dtype, "by_family": by_family, "n_exec": n_exec, "population_inactive_identities": inactive["identities"],
           "problems": problems[:max_report]}
    sr["inactive_by_enablement_check"] = out
    return out


def padding_vs_oracle(pp: dict[str, Any], oc: dict[str, Any] | None, bmap: dict[str, Any]) -> dict[str, Any]:
    """[revb S-r17b-70 C3, M-1495] the TWO documents of a padded record agree on ONE number: `oracle_compare.padding` (the entries the oracle
    stated apart as padding, per Q(P) family, with their (step, request) pairs) == `padded_population`'s inactive stratum (the Commit's
    padding entries per family, read back as bot words) == the padded map's own `bot-by-enablement` entries -- and the padding pairs are
    exactly the (step, request) pairs with NO executed entry (a request is active or inactive at a step, never both).  Agreement is what
    makes 'stated apart' a statement rather than a suppression; a disagreement is named per family / per pair."""
    pads = [e for e in bmap.get("entries", []) if e.get("activity") == "bot-by-enablement"]
    map_by_family = Counter(str(e.get("family") or "unstated") for e in pads)
    map_pairs = sorted({(int(e["identity"]["step"]), str(e["identity"].get("request_id", "r0"))) for e in pads})
    exec_pairs = {(int(e["identity"]["step"]), str(e["identity"].get("request_id", "r0"))) for e in bmap.get("entries", [])
                  if e.get("activity") != "bot-by-enablement" and isinstance(e.get("identity"), dict) and isinstance(e["identity"].get("step"), int)
                  and str(e["identity"].get("request_id", "r0")) != "*" and e.get("family") != "weights"}
    both = sorted(set(map_pairs) & exec_pairs)
    pop_by_family = {f: int(d.get("commit_padding_entries", 0)) for f, d in (pp.get("by_family") or {}).items() if int(d.get("commit_padding_entries", 0))}
    problems: list[str] = []
    if both:
        problems.append(f"{len(both)} (step, request) pairs carry BOTH an executed and a padding entry, e.g. {both[:3]}")
    if dict(map_by_family) != pop_by_family:
        problems.append(f"the map's padding by family {dict(map_by_family)} != the inactive stratum's {pop_by_family}")
    opad = (oc or {}).get("padding") if isinstance(oc, dict) else None
    if not isinstance(oc, dict) or oc.get("error") is not None or ("compared" not in oc and not opad):   # no oracle ran (no --match-dir / unreadable)
        out = {"ok": not problems, "oracle": "absent (no --match-dir oracle ran on this record" + (f": {str(oc.get('why'))[:120]}" if isinstance(oc, dict) else "") + ")",
               "map_padding_entries": len(pads), "map_pairs": len(map_pairs),
               "map_by_family": dict(sorted(map_by_family.items())), "stratum_by_family": pop_by_family, "problems": problems}
        return out
    if not isinstance(opad, dict):
        # an oracle that predates fix (d): its padding entries sit in the 'identity without a byte range' grade + the hidden_stream class
        nc = oc.get("not_compared") if isinstance(oc.get("not_compared"), dict) else {}
        out = {"ok": False, "oracle": "predates fix (d): no `padding` block; the padding entries are graded 'identity without a byte range' + 'no oracle family: hidden_stream'",
               "map_padding_entries": len(pads), "map_pairs": len(map_pairs), "map_by_family": dict(sorted(map_by_family.items())), "stratum_by_family": pop_by_family,
               "oracle_not_compared": nc, "problems": problems + ["oracle_compare has no padding block (record predates eac07aca4)"]}
        return out
    o_by_family = {str(k): int(v) for k, v in (opad.get("by_query_family") or {}).items()}
    o_pairs = [(int(s), str(r)) for s, r in (opad.get("pair_list") or [])] if opad.get("pair_list") is not None else None
    if o_by_family != dict(map_by_family):
        problems.append(f"the oracle stated apart {o_by_family} by family, the map carries {dict(map_by_family)}")
    if int(opad.get("entries_not_compared", -1)) != len(pads):
        problems.append(f"oracle padding.entries_not_compared {opad.get('entries_not_compared')} != the map's {len(pads)} padding entries")
    if int(opad.get("entries_not_compared", -1)) != int(pp.get("padding_entries", -2)):
        problems.append(f"oracle padding.entries_not_compared {opad.get('entries_not_compared')} != the inactive stratum's {pp.get('padding_entries')}")
    if o_pairs is not None:
        if sorted(o_pairs) != map_pairs:
            d1, d2 = sorted(set(o_pairs) - set(map_pairs)), sorted(set(map_pairs) - set(o_pairs))
            problems.append(f"oracle padding pairs != the map's inactive pairs: oracle-only {d1[:3]} ({len(d1)}), map-only {d2[:3]} ({len(d2)})")
    elif int(opad.get("pairs", -1)) != len(map_pairs):
        problems.append(f"oracle padding.pairs {opad.get('pairs')} != the map's {len(map_pairs)} inactive (step, request) pairs (older oracle: count only)")
    if sorted(int(x) for x in (opad.get("steps") or [])) != sorted({s for s, _ in map_pairs}):
        problems.append("oracle padding.steps != the map's padding steps")
    return {"ok": not problems, "oracle": "oracle_compare.padding (fix (d))", "entries": {"oracle": int(opad.get("entries_not_compared", 0)), "map": len(pads),
            "stratum": int(pp.get("padding_entries", 0))}, "by_family": {"oracle": dict(sorted(o_by_family.items())), "map": dict(sorted(map_by_family.items())), "stratum": pop_by_family},
            "pairs": {"oracle": int(opad.get("pairs", 0)), "map": len(map_pairs), "pair_list_compared": o_pairs is not None, "executed_and_padding_overlap": len(both)},
            "steps": sorted({s for s, _ in map_pairs}), "problems": problems,
            "rule": "S-r17b-70 C3: oracle_compare.padding by family + pairs == padded_population's inactive stratum by family == the map's bot-by-enablement entries; "
                    "the padding pairs are exactly the (step, request) pairs without an executed entry"}


def coverage_over_manifest(identities: list[dict[str, Any]], bmap: dict[str, Any]) -> dict[str, Any]:
    """Request-aware coverage of the manifest by a map (`binding.coverage_check`, exact request / step / rank; the weights family excluded
    as in the record's rule) + the B>1 SHARED identities (`request_id == "*"`, a hidden launch plane with `requests` and `engine_step`)
    under B9.1 (1) rule (a): a shared row is covered iff EVERY listed member has an entry of that (engine_step, invocation, op_path,
    member-prefix) -- the executed launch plane bound whole to the members it ran over (the record's hidden-launch rule), or the member's
    own ⊥ per-sequence plane (`member_plane_identity`) where the executed prefix of record places it outside the step.  Counted per
    member: bound by an executed plane / bound by a ⊥ plane / missing (named)."""
    from verity_vllm.commit.binding import map_index
    plain = [_identity_tuple(d) for d in identities if d.get("family") != "weights" and d.get("request_id") != "*"]
    cov = coverage_check(plain, bmap)
    idx = map_index(bmap)
    keys = list(idx)
    shared_n = shared_bound = members_n = members_exec = members_pad = 0
    shared_missing: list[dict[str, Any]] = []
    for d in identities:
        if d.get("family") == "weights" or d.get("request_id") != "*":
            continue
        shared_n += 1
        es = int(d.get("engine_step", -1))
        row_ok = True
        for rid in (d.get("requests") or []):
            members_n += 1
            hit = None
            # op-level, as the pod's loader reads a shared hidden row (`load_required_manifest`: member `*` -- the row names the PACKED
            # launch geometry `g<HB*n_seq>x..`, the tap commits per-sequence planes `s<i>/g<HB>x..`, a member alone launches `g<HB>x..`)
            for k in keys:
                if k[0] == rid and k[1] == es and k[2] == int(d.get("invocation", 0)) and k[3] == d["op_path"] and k[5] == int(d.get("rank", 0)):
                    hit = idx[k][0]
                    break
            if hit is None:
                row_ok = False
                if len(shared_missing) < 8:
                    shared_missing.append({"request_id": rid, "engine_step": es, "op_path": d["op_path"], "output_member": d.get("output_member")})
            elif hit.get("activity") == "bot-by-enablement":
                members_pad += 1
            else:
                members_exec += 1
        if row_ok:
            shared_bound += 1
    out = {k: cov[k] for k in ("expected_n", "bound_n", "missing_n", "ambiguous_n")}
    out["ok"] = bool(cov["ok"]) and shared_bound == shared_n
    out["shared"] = {"n": shared_n, "bound_n": shared_bound, "missing_n": shared_n - shared_bound, "missing": shared_missing[:4],
                     "members_n": members_n, "members_bound_executed_n": members_exec, "members_bound_bot_n": members_pad,
                     "members_missing_n": members_n - members_exec - members_pad,
                     "rule": "B9.1 (1) (a): a shared launch row is covered iff EVERY listed member has a plane of that op at the engine step -- "
                             "the executed launch (bound whole to the members it ran over) or the member's own bot per-sequence plane"}
    out["missing_examples"] = cov.get("missing", [])[:4]
    return out


def _served_tokens_of_record(rd: Path, wp: dict[str, Any]) -> dict[str, list[int]] | None:
    """The served tokens of record per request: the Match capture's `tokens.json` (`generated_token_ids` per request, the same source
    reqval's sampled replay names as `a_r_source`), mapped to `r<k>` by the workload Program's request-id convention.  The commit's
    `runs.jsonl` generated_tokens is a length-only witness and is NOT accepted: the stop rule needs the last token id."""
    p = rd / "match" / "capture" / "tokens.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    ids = ((wp.get("request_ids") or {}).get("workload_request_ids") or {})       # r<k> -> engine request id
    back = {v: k for k, v in ids.items()}
    out: dict[str, list[int]] = {}
    for i, r in enumerate(d.get("requests") or []):
        rid = back.get(r.get("request_id"), f"r{i}")
        out[rid] = [int(x) for x in (r.get("generated_token_ids") or [])]
    return out or None


def write_padded_record(res: dict[str, Any], out_dir: str | Path) -> None:
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    with gzip.open(od / "padded_binding_map_p0.json.gz", "wt") as f:
        json.dump(res["padded_map"], f)
    (od / "padded_report.json").write_text(json.dumps(res["report"], indent=1, default=str))
    (od / "padding_step_roots.json").write_text(json.dumps({"schema": "verity/padding-steps/v1", "rules_version": PADDING_RULES_VERSION,
                                                              "steps": [{"step": ps.step, "root": ps.root.hex(), "merkle_root": ps.merkle_root.hex(),
                                                                         "n_leaves": ps.n_leaves, "n_bytes": ps.n_bytes, "tensors": len(ps.metas),
                                                                         "layout_digest": layout_digest(ps.metas).hex()} for ps in res["padding"]]},
                                                             indent=1))
