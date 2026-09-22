"""[moe F-moe-r17-06] The record that failed: OLMoE B2 i32/o32 mixedstop on the evaluator train (tree_int5 f43dbaa3c / tree_int6 c5efbed29) --
the FIRST real record reading the aliased shared RotaryEmbedding through manifest.op_path_aliases: every sampled RoPE_v1 / Attention_v3 row
of layers 1..15 UNEQUAL (whole row), layer 0 equal in every sample.  Cause: the collector publishes the shared module's 16 invocations per
step under the SAME member name (`model.layers.0.self_attn.rotary_emb/0`, one meta each, distinct tensor_ordinal / stream offset); the
binding map binds them as 16 identities (invocation k); the store resolved layer L -> (canonical, invocation L) and picked invocation L's
ENTRY correctly -- then asked the committed reader by the entry's NAME alone, and `committed_reader` returns the FIRST meta of that name
(invocation 0 = layer 0's words) for every layer.  The alias unit's test worlds gave each invocation a distinct name, so they could not see it.

Fix: `entry_read_name` -- when more than one invocation is bound under a member name at a step, the store asks for `<name>#ord<tensor_ordinal>`
and `committed_reader` selects THAT meta exactly; non-shared members keep the bare name (every reader of record and test double unchanged)."""
from __future__ import annotations

import numpy as np

from verity_capture.commit import oracle_compare as OC
from verity_capture.commit import sampled_replay as SR

ROT = "model.layers.0.self_attn.rotary_emb"
W = 16                                                                         # words per (request) row


class _Host:
    """The retained host tensor's face the reader uses (`.numpy()` -> a C-contiguous array, sliced by BYTES)."""
    def __init__(self, a: np.ndarray):
        self._a = np.ascontiguousarray(a)

    def numpy(self) -> np.ndarray:
        return self._a


class _Meta:
    def __init__(self, name: str, ordinal: int, dev_off: int, host: np.ndarray):
        self.name, self.ordinal, self.dev_off, self.host = name, ordinal, dev_off, _Host(host)


class _Com:
    """A committer double with the collector's REAL shape: layouts[step] = metas in acquisition order, the shared module once per invocation
    under one name; host retention per meta."""
    def __init__(self, metas: list[_Meta]):
        self._layouts = {0: metas}
        self._gpu_blocks = {}


def _record(n_layers: int = 4, requests: int = 2):
    """Step 0 (decode-shaped: one row per request): per layer L one meta `ROT/0` with rows [r0, r1] filled with 1000*L + 10*r + i; a per-layer
    q_norm meta `model.layers.L.self_attn.q_norm/out` (one invocation).  The binding map as build_binding_map writes it: per (meta, request)
    one entry, name = the meta's name, invocation k = the k-th same-name meta of the step, tensor_ordinal = the meta's ordinal."""
    metas: list[_Meta] = []
    entries: list[dict] = []
    ordinal = 0
    inv: dict[str, int] = {}
    for L in range(n_layers):
        for name in (f"model.layers.{L}.self_attn.q_norm/out", f"{ROT}/0"):
            rows = np.stack([np.arange(W, dtype=np.uint16) + 1000 * L + 10 * r + (5000 if "q_norm" in name else 0) for r in range(requests)])
            m = _Meta(name, ordinal, 0, rows)
            metas.append(m)
            k = inv.get(name, 0)
            inv[name] = k + 1
            op, member = name.rsplit("/", 1)
            for r in range(requests):
                entries.append({"identity": {"request_id": f"r{r}", "step": 0, "invocation": k, "op_path": op, "output_member": member,
                                             "element_range": [0, W], "rank": 0, "program_digest": "p"},
                                "kind": "leaf", "name": name, "stream_offset": 0, "width_bytes": rows.nbytes, "encoding": {"dtype": "bf16", "shape": list(rows.shape)},
                                "element_offset": r * W, "tensor_ordinal": ordinal})
            ordinal += 1
    bmap = {"entries": entries}
    manifest = {"op_path_aliases": {f"model.layers.{L}.self_attn.rotary_emb": [ROT, L] for L in range(1, n_layers)}}
    return _Com(metas), bmap, manifest


def test_the_real_reader_returns_the_first_meta_of_a_name_so_the_store_must_ask_for_the_exact_invocation():
    com, bmap, manifest = _record()
    rd = OC.committed_reader(com)
    # [revb S-r17b-19 n3] the reader by BARE name of a shared member (4 metas under it at the step): None by name -- the first-meta answer
    # (invocation 0 = layer 0's words for every layer) was the alias class; a third caller cannot regress into it silently
    assert rd(0, f"{ROT}/0", 0, W * 2) is None
    # the exact meta by ordinal: invocation 2's meta (layer 2, ordinal 5) -- layer 2's words
    b2 = rd(0, f"{ROT}/0{OC.ORDINAL_SEP}5", 0, W * 2)
    assert np.frombuffer(b2, dtype=np.uint16).tolist() == (np.arange(W) + 2000).tolist()
    b0 = rd(0, f"{ROT}/0{OC.ORDINAL_SEP}1", 0, W * 2)                           # invocation 0 (ordinal 1) by ITS ordinal: layer 0's words
    assert np.frombuffer(b0, dtype=np.uint16).tolist() == (np.arange(W) + 0).tolist()
    assert rd(0, f"{ROT}/0{OC.ORDINAL_SEP}999", 0, 4) is None                 # no such meta: None, never a neighbour's bytes
    assert rd(0, "model.layers.3.self_attn.q_norm/out", 0, W * 2) is not None   # a bare non-shared name reads as before
    assert rd(0, "model.layers.3.self_attn.q_norm/out" + OC.ORDINAL_SEP + "6", 0, W * 2) is not None   # ... and by its own ordinal
    assert rd(0, "model.layers.3.self_attn.q_norm/out" + OC.ORDINAL_SEP + "5", 0, W * 2) is None       # a neighbour's ordinal: None


class _Blk:
    """[revb n2] the pod's retained DEVICE block face: one contiguous keep tensor per step, sliced by bytes at meta.dev_off (+ .numpy())."""
    def __init__(self, a: np.ndarray):
        self._a = np.ascontiguousarray(a).reshape(-1).view(np.uint8)

    def __getitem__(self, s):
        return _Host(self._a[s])


def test_the_device_block_path_reads_the_exact_invocation_at_its_dev_off_too():
    """The pod path (`_gpu_blocks[step] = (block, ...)`, `m.dev_off`) selects the meta the same way; the bytes come from the block at
    the SELECTED meta's dev_off -- layer L's words, not the first meta's; a bare shared name is None on this path too."""
    com, bmap, manifest = _record()
    metas = com._layouts[0]
    blob = np.concatenate([m.host.numpy().reshape(-1) for m in metas])          # the keep tensor: metas laid out in acquisition order
    off = 0
    for m in metas:
        m.dev_off, off = off, off + m.host.numpy().nbytes
        m.host = None                                                            # host retention absent: the device path is the only one
    com._gpu_blocks = {0: (_Blk(blob), None)}
    rd = OC.committed_reader(com)
    assert rd(0, f"{ROT}/0", 0, W * 2) is None                                   # bare shared name: None on this path too
    for m in metas:
        if m.name != f"{ROT}/0":
            continue
        L = [x for x in metas if x.name == m.name].index(m)
        b = rd(0, f"{ROT}/0{OC.ORDINAL_SEP}{m.ordinal}", W * 2, W * 4)           # request r1's row of invocation L
        assert np.frombuffer(b, dtype=np.uint16).tolist() == (np.arange(W) + 1000 * L + 10).tolist(), L
    st = SR.CommittedStore.of_record(bmap, rd, manifest)
    for L in range(4):
        op = ROT if L == 0 else f"model.layers.{L}.self_attn.rotary_emb"
        w, _ = st.words("r0", 0, op, "0", 0, W)
        assert w.tolist() == (np.arange(W) + 1000 * L).tolist(), L


def test_layer_l_reads_its_own_invocations_words_through_the_alias_table_on_a_real_shaped_binding_map():
    com, bmap, manifest = _record()
    st = SR.CommittedStore.of_record(bmap, OC.committed_reader(com), manifest)
    assert st.name_inv[(0, f"{ROT}/0")] == 4 and st.name_inv[(0, "model.layers.1.self_attn.q_norm/out")] == 1
    for L in range(4):
        op = ROT if L == 0 else f"model.layers.{L}.self_attn.rotary_emb"
        for r in range(2):
            w, dtype = st.words(f"r{r}", 0, op, "0", 0, W)
            assert w is not None and dtype == "bf16", (L, r)
            assert w.tolist() == (np.arange(W) + 1000 * L + 10 * r).tolist(), (L, r, w[:4])        # layer L's OWN words, request r's row
            q, _ = st.words(f"r{r}", 0, f"model.layers.{L}.self_attn.q_norm", "out", 0, W)
            assert q.tolist() == (np.arange(W) + 5000 + 1000 * L + 10 * r).tolist()               # the non-shared member: bare-name read


def test_entry_read_name_suffixes_only_shared_members():
    e = {"name": "m/out", "tensor_ordinal": 7}
    assert OC.entry_read_name(e, 1) == "m/out"
    assert OC.entry_read_name(e, 16) == f"m/out{OC.ORDINAL_SEP}7"
    assert OC.entry_read_name({"name": "m/out"}, 16) == "m/out"                                   # no ordinal on the entry: nothing to select by
    assert OC.entry_read_name({"name": f"m/out{OC.ORDINAL_SEP}3", "tensor_ordinal": 3}, 16) == f"m/out{OC.ORDINAL_SEP}3"
    _com, bmap, _m = _record()
    sni = OC.same_name_invocations(bmap)
    assert sni[(0, f"{ROT}/0")] == 4 and all(v == 1 for k, v in sni.items() if "q_norm" in k[1])


def test_the_oracle_compare_reads_each_invocation_exactly_too():
    """The C2 oracle compare walks the binding map with the same reader: a shared module's invocation-k entry must read meta k's bytes."""
    com, bmap, _m = _record()
    rd = OC.committed_reader(com)
    sni = OC.same_name_invocations(bmap)
    for e in bmap["entries"]:
        ident = e["identity"]
        if ident["op_path"] != ROT:
            continue
        eo = int(e["element_offset"])
        b = rd(0, OC.entry_read_name(e, sni[(0, e["name"])]), eo * 2, (eo + W) * 2)
        L, r = ident["invocation"], int(ident["request_id"][1:])
        assert np.frombuffer(b, dtype=np.uint16).tolist() == (np.arange(W) + 1000 * L + 10 * r).tolist(), (L, r)
