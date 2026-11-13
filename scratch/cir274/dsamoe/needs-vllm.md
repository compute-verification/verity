# CIR-274 4b (DSAMOE): what the sparse lowering needs of serving's registrations

Two drafts, written and not applied. Lines on `main` are at `origin/main` `8c11a1d5ab`; lines of the frame and of
`node_window_sparse.py` are at the families head `b5c74a0683`. Tags: [read] read in the file at that line; [inferred] not
run.

## (a) BatchedGemm's `w` as its own registration in (B, N, K) order

**What it is.** `BatchedGemm_v1{B, K, N}` is MLA's latent absorption: `W_UK_T` at `{B=8, K=192, N=512}` (q_nope against
the latent) and `W_UV` at `{B=8, K=512, N=256}` (the value up-projection), per rank. `dsa.py:453-458` reads the field,
stored [B][K][N] (`W_UK_T` [NH][DN][L], `W_UV` [NH][L][DV], `check/fp8_ue8m0.py:16`), as [B][N][K]: one
`strided_view(w, b K N, K, 1, N, N K)` a head (`dsa.py:457`), so leaf j = n K + k of head b is the field's word
b K N + k N + n [read]. A `SLICES` unit (b, n) reads K words that are a column of the record, and `IF.Read`
(`interface.py:517-531`: rows of the port's width at `base + Σ d_k step_k` of one parameter's leaf order) reads only runs.
Today the port is a free input under `weights`, B·N·M units of K words a node (`node_window_sparse.py:84-87`).

**Name.** `f"{WR.weight_param(label, K)}/bnk"`, that is `weights/<label>@<K>/bnk`, with `label` the field's label in the
weights record (`node_window.py:153-156`, the Program's `weights` struct). It follows `expert_rows.table_param`'s form
(`pipeline/expert_rows.py:67-70`: `{WR.weight_param(label, words)}/b{block}/table`), a suffix on the field's
`weight_param` (`check/weights_of_record.py:63`), as `window_builder.table_param` (`window_builder.py:27-32`) is too. It
does not follow `window_builder`'s table name (`table/{weight_param(label, words)}+{off}`, `window_builder.py:230`),
since that names an `IF.Table`, and a read declares no table: `IF.Read` names only its parameter. The function would sit
beside `table_param` in `window_builder.py`, the module that names the window's weight parameters.

**Rows.** B·N rows of K u16 words (bf16), row (b, n) = the field's words b K N + k N + n for k < K:

| field | B·N rows | K words a row | leaves B·N·K |
|---|---|---|---|
| `W_UK_T` | 8 · 512 = 4,096 | 192 | 786,432 |
| `W_UV` | 8 · 256 = 2,048 | 512 | 1,048,576 |

Two registrations a layer (every layer has MLA), at the rank's TP shard as the record composes it.

**The `IF.Read` the lowering would declare.**
`IF.Read(t, "w", "weights/<label>@<K>/bnk", at=a, radix=(B, N, M), base=0, step=(N K, K, 0))`, made with
`w.b.read(t, "w", a, (B, N, M), 0, (N * K, K, 0), param=param)` (`window_builder.py:204-213`) and
`w.b.params[param] = B * N * K`. The same unit order as today's registered read (`node_window_sparse.py:85`).

**What the prover commits.** The parameter's rows as `window_weights.weight_value` commits a weight
(`program/predict/window_weights.py:114-125`): `H1R.commit(program, param, rows, key, nonce, schema=G.SCHEMA_V2)`, at
the weights record's program digest, with `hm96-sha512/row/v2` rows, under the committer's key, at the nonce
`weight_nonces` gives (`window_weights.py:128-131`). The rows are
`np.ascontiguousarray(words.reshape(B, K, N).transpose(0, 2, 1)).view("<u2").reshape(B * N, K)`. Two lines of
`window_weights` change for it, and nothing else in the commit path does:
- `_param` (`window_weights.py:52-60`) accepts `weights/<label>@<w>` and its `/table` only, and refuses anything else by
  name, so it would accept `/bnk` and return its kind;
- `_rows` (`window_weights.py:69-76`) cuts the record's words into rows of the width, so for `/bnk` it would transpose
  first. It needs the field's (B, K, N) shape, which `_fields` has (`window_weights.py:134-139`).

`weight_nonces` numbers every `weights/` parameter in sorted order, so adding parameters renumbers the nonces after them.
A registration made before the change is then stale, and a new one is made, as for any new parameter.

**What `check/weights_of_record.py` would have to derive.** Nothing new of record. The record keeps one digest a field,
in record order (`weights_of_record.py:1-60`). The `/bnk` parameter is a re-layout of a field of record, as `/table` and
the expert tables are. The verifier's entry for it is `window_weights._entry` (`window_weights.py:79-87`: value, leaves
B·N·K / K rows of K, words K, domain), made by `weights_record`'s `shape` callback (`window_weights.py:201-203`), which
needs only `_rows`' size. The size is unchanged, so it holds as is once `_param` accepts the name. That the root is the
transposed checkpoint words is the committer's own check, as it is for every weight (`window_weights.py:6-8`). For `W_UV`
the (B, N, K) order is kv_b_proj's dequantised rows before the record's transpose, `heads[:, DN:, :]`
(`check/fp8_ue8m0.py:92`): the record transposes kv_b_proj into `W_UV`, and the registration transposes it back. A
registration could be tied to kv_b_proj's rows directly instead, but that is a second derivation of one field, and this
draft does not take it.

**Does anything on `main` register a permuted weight? (searched by form).** The search covered `transpose(`, `swapaxes`,
`moveaxis`, `.permute(`, `.T` and `reshape(E…)[:, block…]` in every module that commits or registers a weight:
`window_weights`, `expert_rows`, `expert_tables`, `weight_rows`, `serving_rows`, `records/`, `pouw/h1/registered`,
`one_stage/registered`, `weights_of_record`, `derived_weights`, `fp8_moe_scales` and `fp8_ue8m0`. Then over every
`pipeline/`, `program/predict/`, `acquire/` and `commit/` module. The answer is no. What it found:
- **Registered, not permuted within a row:**
  - `window_weights._rows` reshapes the record's words, in order;
  - `/table` holds the record's rows, in order;
  - `expert_rows.table_words` (`expert_rows.py:82-83`) makes a table whose row e is expert e's rows [b n, b n + n), a
    contiguous run of the record, with its rows a gather across experts at a stride. Rows are reordered, but no row's
    words are.
- **Permuted, committed per window as a unit's rows, never registered:**
  - `pipeline/part_rows.py:189-190`: `W.reshape(E, U, c, K).transpose(1, 0, 2, 3)`;
  - `pipeline/kchunk_rows.py:112`: `W.reshape(E, N, -1, k)[:, :, P].transpose(1, 2, 0, 3)`;
  - `serving_rows.py:546` and `window_rows.py:518`, both on activations.
- **A permuted field of record:** `fp8_ue8m0.absorbed` (`check/fp8_ue8m0.py:92`) defines `W_UV` itself as kv_b_proj's
  rows transposed.

So `/bnk` would be the first registration whose rows are not runs of the record.

**The hunk in `node_window_sparse.py` that would use it** (not applied; `WB.bnk_param` is the name above, in
`window_builder.py`):

```diff
@@ def batched_gemm(w: NodeWindow, j: int, node: Any) -> None:
     refs = node.args[1].refs
     if len(w.segments(refs.slice(0, K))) == 1:
         w.b.read(t, "w", a, (B, N, M), w.weight(refs, B * N * K), (N * K, K, 0))
+    elif _bnk(refs, B, K, N):
+        param = WB.bnk_param(w.label(refs, B * N * K), K)
+        w.b.params[param] = B * N * K
+        w.b.read(t, "w", a, (B, N, M), 0, (N * K, K, 0), param=param)
     else:
         w.feed(t, "w", a, _Free([("param", WEIGHTS, _label_at(w, refs))]), (B, N, M))
     w.out(j, node, [Piece(0, B * N, Row(t, a, "o0", 1), M, 1)])
+
+
+def _bnk(refs: Refs, B: int, K: int, N: int) -> bool:
+    """Whether ``refs`` are a whole [B][K][N] field read as [B][N][K] (`dsa.py`'s absorb: one `strided_view` a head),
+    the order of its ``/bnk`` registration (`window_builder.bnk_param`)."""
+    parts = refs.parts if isinstance(refs, Concat) else (refs,)
+    return (len(parts) == B and all(isinstance(p, Strided) and (p.inner, p.outer_stride, p.inner_stride, p.count)
+                                    == (K, 1, N, N * K) and p.base == parts[0].base + b * K * N for b, p in enumerate(parts))
+            and parts[0].base == <the field's first leaf>)
```

The last condition holds the read to the whole field from its first word, which is what `base=0` assumes. The field's
first leaf is `w.weight`'s offset of zero, as `_Builder._weight` gives it (`window_builder.py:196-201`). The space and
index of each part must also be the weights record's, as `w.label` already requires. [inferred: not run, not built]

## (b) What my expert tables' declarations assume of serving's committer

My declarations are `_table_reads` (`node_window_sparse.py:151-160`): for each block b of each expert stack's weight
and scales, `IF.Table(param, param, 0, H, E)`, `w.b.params[param] = H * E`, the `root` and `domain` table reads of
`param`, and an `IF.Opening(tid, param, "tok", ("row", "salt", "path"), "o1", at, units)`. Serving's side is the
registration `verity-vllm tp-register-tables` (`pipeline/expert_tables.py`), the record `records/expert_tables.py`, and
a window committer under it (`expert_rows.committer`, `expert_rows.py:377`). Each assumption, and what on `main` it must
agree with:

| assumption | my side | `main` it must agree with |
|---|---|---|
| **parameter name** | `XR.table_param(label, H, block)` = `weights/<label>@<H>/b<block>/table`, `label` the weights record's field label (`node_window.py:153-156`) | `expert_rows.table_param` (`pipeline/expert_rows.py:67-70`), called by `expert_tables.tables_of` with the Program's weight field names (`expert_tables.py:75, 77`; fields from `weights_of_record.program_weight_fields`, `weights_of_record.py:320-326`), matched by `EXPERTS_RE` (`weights_of_record.py:90`) |
| **row width** | weights H = n K / 2 (e4m3 pairs in u16 words); scales H = 2 K / G (f32 in two u16 words), K the weight's | `tables_of`: `n * K // 2` (`expert_tables.py:75`) and `2 * K` of the scale field's last dim, which is K / G (`expert_tables.py:77`); `table_words`' `.reshape(E, -1).view("<u2")` (`expert_rows.py:83`) |
| **row count** | E rows, one an expert (`EmbeddingRead_v1{V=E, H}`), leaves H · E | `table_words` rows = E (`expert_rows.py:82-83`); `expert_rows.declarations`' `IF.Table(param, param, 0, leaves // E, E)` (`expert_rows.py:332`); the entry's `leaves` (`records/expert_tables.py:18`) |
| **block rows n** | `_block_rows` (`node_window_sparse.py:140-148`): w13 n = 1, w2 n = 32, from `unit_templates` and `BLOCK_WORDS` = 4096; the registration is not read | the registration's `blocks` (`records/expert_tables.py:30`), set by `--block w13=n --block w2=n` (`expert_tables.py:245, 276`), whose docstring example is `--block w13=1 --block w2=32`; refused unless a power of two dividing N (`expert_tables.py:72-74`) |
| **block order** | weight table b = each expert's rows [b n, (b + 1) n); scale table b' = G-block b', read by block b at b' = b n / G; blocks b < N / n, b' < N / G | `tables_of`' `range(N // n)` and `range(N)` of the scale field's NB (`expert_tables.py:75, 77`); `table_words`' `[:, block * n:(block + 1) * n, :]` and `[:, block, :]` (`expert_rows.py:82`) |
| **root** | a table read of `param`'s `root` into the read's `root` port: the verifier's pin | the entry's `root` (`records/expert_tables.py:18, 50`), from `H1R.commit(...).entry()` (`expert_tables.py:146`; `protocols/pouw/h1/registered.py:278`); a window's committer commits under the same key, program and nonces (`expert_rows.committer`, `expert_rows.py:377-382`) |
| **domain** | a table read of `param`'s `domain` into the read's `domain` port; my lowering assumes no program | `G.domain(program, value, w, n, WORD_BITS)` (`registered.py:305`) at `program = expert_tables.program(repo, revision, world, rank, prefix)` (`expert_tables.py:53-57`): the model's identity, not the request Program's digest nor the weights record's (`r.digest`, `window_weights.py:224`) |
| **row schema** | an opening of `row`, `salt`, `path` (`gates.embedding`) | `hm96-sha512/row-seg/v1` (`schema=G.SCHEMA_SEG`, `expert_tables.py:146`) |

**Where they could disagree:**
- **Labels.** My labels are the request Program's `weights` struct fields. A registration made from a rank Program of
  the same Build names the same fields. A registration made `served=` from a capture (`expert_tables.py:195-197`,
  prefix `""`) names them as vLLM's `params.json` does, and those names agree only if the Program's labels are vLLM's
  names. [inferred]
- **Block rows.** My n is computed, not read from the registration. A registration at other `--block` values has no
  parameter of my names, so its roots pin nothing in the window and my reads count free. That is a visible loss of
  credit, not a false claim. [inferred]
- **`window_weights` would refuse the names.** My table parameters begin with `weights/`. `window_weights.weight_params`
  (`window_weights.py:63-66`) takes every such parameter, and `_param` (`window_weights.py:52-60`) refuses
  `weights/<label>@<H>/b<b>/table` (its width `"<H>/b<b>"` is not a number). This doesn't bite today:
  `window_weights` lowers the class circuit (`window_weights.py:148`), not the node window. A registration of the node
  window's parameters would have to give the expert tables to `expert_tables` and the rest to `window_weights`.
  [inferred]
