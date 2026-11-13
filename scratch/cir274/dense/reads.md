# CIR-274 DENSE: reads neither wire form takes, and rows a lowering refuses

Worker DENSE (piece 4a). Work branch: `cursor/circuits-cir274-dense-6f04`. Tags: [test] is
`test_node_window_dense.py` run locally at LP = 32 and 1 layer; [local] is a local call in this session; [run ID] is a
remote run's output; [inferred] is inferred.

A wire reads one writer's row whole. A port with several writers is cut k ways along its outer elements, or read
through `AllGather_v1` with cut parts.

**Count: 3 families, at 43 prompt ids and 4 layers 2 + 2 + 1 = 5 nodes [r20261113-015207-eb58].** All three are left unlowered, so their
readers' rows are free. None is cured by a Definition or by a hunk in vllm's files.

## 1. `EmbeddingShard_v1{VS=19360,H=6144,START=0}`: its SLICES unit reads a strided column of the weight

- **Nodes.** At LP = 32 and 1 layer, these are node 0 (the prefill, batch of 32) and node 20 (the decode) [local dump].
  At 43 ids and 4 layers there are 2 nodes [r20261113-015207-eb58].
- **Reader's port.** `table`, of the slice `EmbeddingShard_v1{VS=19360,H=1,START=0}`. `unit_templates` lists it 6144
  times; it is the only template besides the row itself [local].
- **Words read.** Each of the 6144 units reads 19360 words: column h of the [19360, 6144] table, at stride 6144.
  `SLICES` gives the table H on dimension 1, `(("H",), ((None,), (1,)))` [read serving_templates.py:46-50].
- **The writer's row.** This is the weights record (`pweights[0, 118947840)`), not a unit. So the read is a registered
  read, not a wire.
- **Why neither form takes it.** `_Builder.read` reads rows of the port's width (19360 words) from a base, at a
  constant step per digit [read window_builder.py:204-213], under the registration `weight_param(label, 19360)`: the
  weight as rows of 19360 contiguous words. Column h is 19360 words at stride 6144, not one such row. A read at
  base 19360·h would be accepted, which is the danger: `_Builder._weight` checks only `(lo - first) % words` and
  `n % words` [read window_builder.py:184-193], and it would read row h of the flat [6144, 19360] view, which is not
  column h. So the lowering must not be written that way.
- **The row itself.** It fails `check_proof_block`: its table row is 237,895,680 B, which needs a 2^38-bit block [local].
- **What is left free.** At LP = 32 and 1 layer, the all-reduce's own partial: row 0 of `AllReduceMnnvl` `parts`, cut
  8 ways. That is 1,056 unit ports of 192 words, 202,752 words [test]. At 43 ids and 4 layers it is the decode's
  all-reduce only (the prefill's is `AllReduceMultimem`, piece 2): 32 unit ports, 6,144 words [r20261113-015207-eb58].
- **What fits a shard (item 6), written, not applied.**
  - *How `window_circuit._Embedding` reads the embedding on main.*
    - One `EmbeddingRead_v1{V, H}` unit a position, whose `tok` is fed from the position's token.
    - Its table is `b.table(self.weight(refs, V·H), H, V)`: the weights record's [V, H] rows, registered as a table.
    - Its `root` and `domain` are two `IF.TableRead`s.
    - Its `row`, `salt` and `path` are an `IF.Opening` at the hidden key `tok`, pinned by its verdict `o1`.
    - Its `h` (`o0`) is the position's row.
  - *What #9375 changes there (head `34c5cbb3705c`, fetched, not merged).*
    - `IF.Opening` is renamed `IF.TableOpening` with no alias (interface.py), and so are `_Builder.openings`,
      `merged()` and `_Embedding`'s append.
    - `_Builder.value` gains `count` (`radix=(count,)`, `step=(words,)`).
    - The read itself, its table and its key are unchanged.
    - `node_window_sparse._table_reads` builds `IF.Opening` and breaks once #9375 lands (census-all.md).
  - *Why a shard can't take `EmbeddingRead` as `_Embedding` does.* The two differ off the rows they own:
    - `EmbeddingShard_v1` gives `table[tok - START]` on [START, START + VS) and **zeros** elsewhere;
    - `EmbeddingRead_v1{V, H}` gives its row on [0, V) and **0x7FFF** in every word elsewhere (`FILL`).

    So a table of the shard's VS rows keyed by `tok - START` gives NaN where the Program gives 0: on a token another
    rank owns, which is most tokens. The all-reduce then sums NaN.
  - *The form that fits, with existing Definitions.*
    - **Units:** per column block b of G = 192 words (`_cols(H, grain.norm)`, 32 blocks at H = 6144), one
      `EmbeddingRead_v1{V=154880, H=192}` unit a member, keyed by the token as it is (no `- START`).
    - **Key:** the token itself.
    - **Table:** a registered parameter of its own. It has V rows of 192 words: the shard's rows at START, **zero
      rows elsewhere**.
    - **Rows outside the shard:** they read 0, as the Program does.
    - **What the all-reduce reads:** its column block b reads block b's `o0` row (192 words) by wire, its own partial
      on row 0 of `parts`, which is today's free `EmbeddingShard` port: 32 unit ports, 6,144 words, at 43 ids and 4
      layers [r20261113-022750-b23e]. A 6144-word `h` would have to be cut, which `Piece.rows` refuses.
  - *Its costs.*
    - V - VS = 135,520 zero rows a rank: 832,634,880 zero words, 1.67 GB as u16, committed in the table's tree.
    - D = merkle_depth(V) = 18 against 15 for the shard.
    - A token ∉ [0, V) reads 0x7FFF against the Program's 0. That is completeness only: vLLM refuses such ids
      [inferred].
    - The tie between the registered table and the weights record is unproved, as with the experts' tables.
  - *Why not applied.*
    - V is not among `EmbeddingShard_v1`'s statics, so `serving_templates` must list `EmbeddingRead_v1{V, H=G}` for
      it.
    - The prover must commit the padded table (`records`, as `expert_tables` does).
    - Both files are on `main` (@vllm), so their hunk is `needs-vllm.md`, and this item stops there.
    - The hunk below also builds `IF.TableOpening`, which exists only after #9375, a path #9375 changes.
  - *The hunk:* `adapters/vllm/verity_vllm/program/predict/node_window_dense.py` (@circuits; not on `main`). A draft
    that has not been run; it needs `needs-vllm.md`'s `ST.shard_vocab` and `ST.shard_param`.

    ```python
    def _shard_reads(w: NodeWindow, tid: str, at: int, units: int, label: str, V: int, G: int, block: int) -> None:
        """Units ``[at, at + units)`` of the read ``tid`` read column block ``block`` of the embedding shard ``label`` as a
        table of V rows of G words, the shard's rows at its START and zero rows elsewhere: a registered parameter of its
        own (`serving_templates.shard_param`), its root and domain id table reads, and its row, salt and path an opening
        at the key ``tok`` that the verdict ``o1`` holds."""
        param = ST.shard_param(label, G, block)
        w.b.params[param] = V * G
        w.b.tables.setdefault(("shard", label, G, block), IF.Table(param, param, 0, G, V))
        w.b.roots += [IF.TableRead(tid, "root", param, "root", at, units),
                      IF.TableRead(tid, "domain", param, "domain", at, units)]
        w.b.openings.append(IF.TableOpening(tid, param, "tok", ("row", "salt", "path"), "o1", at, units))


    def embedding_shard(w: NodeWindow, j: int, node: Any) -> None:
        """`EmbeddingShard_v1{VS, H, START}` as H / G column blocks of `EmbeddingRead_v1{V, G}`, member m's block b at unit
        ``a + b M``, each keyed by the token itself over block b's table, which reads 0 off the shard as the node does;
        the all-reduce reads block b's ``o0`` by wire."""
        spec, fam, M = descriptor_id(node.fn), _family(node), w.members(node)
        VS, H = int(node.fn.bindings["VS"]), int(node.fn.bindings["H"])
        V, G = ST.shard_vocab(spec), _cols(H, w.b.grain.norm)
        A, label = H // G, w.label(node.args[1].refs, VS * H)
        t, a = w.unit(fam, _part(spec, ST.READ, V=V, H=G), A * M)
        tok = w.take(node, 0)
        w.feed(t, "tok", a, tok if isinstance(tok, _Free) else [tok[i % M] for i in range(A * M)], (A, M))
        for b in range(A):
            _shard_reads(w, t, a + b * M, M, label, V, G, b)
        w.out(j, node, [Piece(0, H, Row(t, a, "o0", G), M, 1)])
    ```

    `FAMILIES["EmbeddingShard"] = embedding_shard`. Owners (`research msg owners`): `node_window_dense.py` @circuits,
    `serving_templates.py` @vllm.

## 2. `TokenSelect_v4{V=154880}`: no proof block holds its unit

- **Nodes.** At LP = 32 and 1 layer, nodes 19 and 39 [local dump]. At 43 ids and 4 layers there are 2 nodes [r20261113-015207-eb58].
- **Reader's port.** `logits`, 154880 words, read from `AllGather_v1{WORLD=8,N=19360}`'s output. That output is now
  the 8 rank-major slices `AllGather_v1{WORLD=1,N=19360}`, each 19360 words, so a cut of 8 ways takes the read as a
  wire (rank 0's slice wired, the 7 peers' free).
- **Why it stays unlowered.** It is a refusal of the unit, not of the read. `STAGES` holds `TokenSelect_v1` only, so
  `unit_templates` lists the row itself alone, and `check_proof_block` refuses it at every cut of `logits`
  (1, 2, 4, 8, 16, 32, 64 and 121 ways need 2^28 bits; 128 and 160 ways need 2^29) [local]. The served select
  `GreedySelect_v1{V=154880,N=7744}` is piece 5.
- **What is left free.** Nothing, because no lowered unit reads it. `StopCheck` (3, below) is its only other reader.

## 3. `StopCheck_v1{T=2,STOP=(),...}`: no serving template holds it

- **Nodes.** At LP = 32 and 1 layer, node 40 [local dump]. At 43 ids and 4 layers there is 1 node [r20261113-015207-eb58].
- **Reader's port.** `ids`, 2 leaves of 32 bits: each step's token select's one u32 (`n19[0,1)`, `n39[0,1)`). Its
  output is 1 leaf of bits.
- **Why it stays unlowered.** `serving_templates.template` refuses it: "the serving rows commit u16 words, and its
  ports' words are [1, 32] bits" [local]. The read itself, two writers' rows into `ids` cut 2 ways along its 2 outer
  elements, is a form a wire takes. Its writers, though, are the token selects of 2, which are unlowered.
