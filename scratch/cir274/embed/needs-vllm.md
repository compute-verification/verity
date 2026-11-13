# CIR-274 EMBED: the hunk in vllm's files the embedding shard's lowering would need (not applied)

Worker EMBED. DENSE's draft (`scratch/cir274/dense/needs-vllm.md` on `cursor/circuits-cir274-dense-scratch-6f04`) restated
at one width, under VLLM-417's two conditions and its 08:04Z note. Owner @vllm. Nothing here is applied: the work branch
`cursor/circuits-cir274-embed-6f04` changes only `node_window_embed.py` and its test, where these names live tonight.

## Nothing refused the unit

Neither the frame nor the builder consults `unit_templates` when a lowering allocates a unit (`_Builder.unit` takes the
bound Definition; `settle` checks each template with `check_proof_block`). So no hunk was needed to lower the row: the
file's own `unit_templates` wraps `serving_templates.unit_templates` and adds the read, and the lowering binds the read
from it with `serving_templates.template`. What `serving_templates` would hold is below; lifting it means deleting the
four names from `node_window_embed.py` and importing them.

## `adapters/vllm/verity_vllm/pipeline/serving_templates.py`

As written in `node_window_embed.py` at the work head (lift as is):

```python
#: an ``EmbeddingShard_v1`` row's whole vocabulary V, by its shard's VS: its statics carry VS and START, not V
SHARD_VOCAB: dict[int, int] = {19360: 154880}
#: the one row width of a shard's padded tables, fixed at registration: a table's salts are its field's place and its
#: block, so two widths of one field would take the same salts
SHARD_WIDTH = 192


def shard_vocab(spec: str) -> int: ...          # V of the row; refused for a VS of no registered vocabulary


def shard_param(label: str, words: int, block: int) -> str:
    """``weights/<label>@<words>/b<block>/vocab``; refused at any width but SHARD_WIDTH."""
```

and in `unit_templates(spec)`, beside the `EXPERTS` case, the read at the one width only (DENSE's draft listed it at
every halving of H within 256 words; condition 1 makes that one):

```diff
+        if spec_base(spec) == "EmbeddingShard_v1":
+            from verity.commitments.gates.embedding import read
+
+            H = int(spec_statics(spec)["H"])
+            if H % SHARD_WIDTH == 0:
+                hit += ((descriptor_id(read(shard_vocab(spec), SHARD_WIDTH)), H // SHARD_WIDTH),)
```

## The tables' registration (`pipeline/expert_tables.py`, `records/expert_tables.py`)

Per VLLM-417 at 08:04Z the padded tables go to `expert_tables`' registration with nonces from `TABLES_NONCE`, and
nothing of `window_weights` is handed a table name. What that needs, not drafted as code:

- `tables_of` takes the embedding field too: a field `(VS, H)` of 16-bit words whose label is the shard's gives
  `[(b, shard_param(label, 192, b), "vocab", 0) for b in range(H // 192)]`. `node_window_embed.tables_of(params)` gives
  exactly this form from a window's parameters, and `expert_tables.nonces` takes it unchanged
  (`test_split_keeps_weight_nonces`: 32 nonces from `TABLES_NONCE`, all distinct, all above every plain weight's).
- The rows of block b: `node_window_embed.table_words(field[:, 192 b : 192 (b + 1)], START, V)`, committed as
  `hm96-sha512/row-seg/v1` rows, as an expert table's.
- Committed once a process, in the rank's one registration (condition 2), and opened by the window committer from the
  registration's held rows, not recommitted per window. The window itself declares the tables (`IF.Table`,
  `IF.TableRead`, `IF.TableOpening`, `params`) and commits nothing.

Cost of the padding [inferred from the shapes, not measured here]: V − VS = 135,520 zero rows a block, 32 blocks, i.e.
832,634,880 zero words (1.67 GB as u16) a rank beside the shard's 118,947,840. One block's commit on this VM took about
2 s (a probe, not a run).

## Not vllm's, and in the way of the shared values check

`node_window_values.Tables` refuses any table whose parameter isn't `/b<b>/table` ("is not an expert table's"), so
`node_window_values.run` can't check these units without a hunk there (a `/b<b>/vocab` case whose rows are
`table_words` of the shard's columns). That file is circuits' (held by REQBLOCK tonight); the work branch has a value
test of its own instead (`test_values`, `test_planted_errors_fail`, `test_values_boolean`).
