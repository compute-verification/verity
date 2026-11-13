# CIR-274 EMBED: notes

Work branch `cursor/circuits-cir274-embed-6f04` from the joint head `30dac672d2bb682dd3a81c80bcf4915f2577fd2e`; it
changes `adapters/vllm/verity_vllm/program/predict/node_window_embed.py` and
`adapters/vllm/tests/program/predict/test_node_window_embed.py` and nothing else (`git diff --stat 30dac672 HEAD`).

## The form

Per column block b of 192 words (32 at H = 6,144), one `EmbeddingRead_v1{V=154880,H=192}` unit a member, block b of
member t at unit `a + 32 b + t`, keyed by the token as it is. Block b's table is `weights/<label>@192/b<b>/vocab`, V rows
of 192 words: the shard's rows at START, zeros elsewhere (`table_words`). Its root and domain are two `IF.TableRead`s and
its row, salt and path an `IF.TableOpening` at `tok` with verdict `o1`, as `node_window_sparse._table_reads` declares an
expert table. The node's output is `Piece(0, H, Row(read, a, "o0", 192), step=M, member=1)`, so the all-reduce's
column block b reads block b's `o0` by wire.

## VLLM-417's two conditions

1. **One width a field, fixed at registration, a second refused.** Met in the file: `WIDTH = 192` is the one width;
   `shard_param` refuses any other; the lowering refuses an H not whole blocks of 192 and a window whose norm grain
   cuts H at another width (`_cols(H, grain.norm) != 192`); and it refuses a shard whose table is already declared at
   another width in the window (`_one_width`). `unit_templates` lists the read at 192 alone. Shown by
   `test_second_width_refused`. Not shown: a refusal across windows or processes (two windows each at one width, but
   different ones). The window can't see that; the registration (`expert_tables`) is where a second width would have
   to be refused for good, and it is vllm's.
2. **Committed once a process, not each window.** Met as far as the window goes: the lowering only declares (`params`,
   `IF.Table`, `IF.TableRead`, `IF.TableOpening`) and commits nothing; no code in either file commits a table but the
   test's own committer. That the prover commits them once a process is vllm's registration and committer, not
   written (needs-vllm.md). Not shown: anything about `commit_tables`' cost.

## VLLM-417 at 08:04Z (the split)

1. Tables go to the tables' registration, plain weights to `window_weights`: `split(params) -> (plain, tables)` keeps
   both kinds of table (`/b<b>/vocab` and the experts' `/b<b>/table`) apart; `tables_of` gives the shard's tables in
   the form `expert_tables.nonces` takes, and its nonces start at `TABLES_NONCE`. Nothing in the file hands a table name
   to `window_weights`.
2. Shown by `test_split_keeps_weight_nonces` (LP = 32, 1 dense layer), `test_split_beside_expert_tables` (LP = 32, 1 MoE
   layer, expert tables present: 19 plain weights, 18 would move unsplit) and run `r20261113-084147-26ee` (43 ids, 4
   layers: 61 plain, 60 would move unsplit; the same with and without the shard's tables). `window_weights._param`
   still refuses `/b<b>/vocab` (asserted in the test). No change to `window_weights.py`, `expert_tables.py`,
   `expert_rows.py` or `serving_templates.py`.

## Values (c)

`node_window_values` can't take these tables without a hunk (its `Tables` refuses a non-expert table), so the test file
has its own: made-up columns of blocks 0 and 1, padded with `table_words` and committed by `pouw.h1.registered.commit`
under a test key; each unit fed only from what the circuit declares for it (`_declared`: the one table its root, domain
and opening name); its outputs by `gates.embedding.reference`, against `EmbeddingShard_v1`'s words (the NumPy
transcription, held to the reference evaluator by `node_window_values.native_selftest`). Tokens 7 and 19,359 (inside)
and 19,360, 100,000 and 154,879 (outside, zeros). Plants: a block declared over block 0's table gives block 0's words;
a wrong row at the key fails its opening (verdict 0). `test_values_boolean` (slow, 68 s) evaluates the Boolean read on
two units and gets the reference's words and verdicts.

## What I'd write next / set aside

- The shared values check (`node_window_values.run`) over the whole window, once its `Tables` takes `/vocab` tables.
- The padded tables cost 832,634,880 zero words a rank [inferred from the shapes]. A smaller form (a table of VS rows
  and a select of zeros off the shard) needs a Definition that isn't registered; not pursued.
