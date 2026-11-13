# CIR-274 DENSE: hunks in files on `main` that the node window needs (not applied)

## `EmbeddingShard_v1` as `EmbeddingRead_v1` over a padded table (reads.md 1)

Owner @vllm (`research msg owners`). Two files, each on `main`. Drafts that have not been run.

`adapters/vllm/verity_vllm/pipeline/serving_templates.py`: the vocabulary V of a shard's row, its padded table's
parameter, and the read template listed for it.

```python
#: an ``EmbeddingShard_v1`` row's whole vocabulary V, by its shard's VS: its statics carry VS and START, not V
SHARD_VOCAB: dict[int, int] = {19360: 154880}


def shard_vocab(spec: str) -> int:
    """The vocabulary V of the ``EmbeddingShard_v1`` row ``spec``: what its padded table's rows (and the read's key) range
    over."""
    return SHARD_VOCAB[int(spec_statics(spec)["VS"])]


def shard_param(label: str, words: int, block: int) -> str:
    """The registered parameter of column block ``block`` of the embedding shard ``label`` as a table of V rows of ``words``
    u16 words: the shard's rows at its START, zero rows elsewhere."""
    return f"{WR.weight_param(label, words)}/b{int(block)}/vocab"
```

and in `unit_templates(spec)`, beside the `EXPERTS` case (the read at each halving of H within 256 words, H / g of
them):

```diff
+        if spec_base(spec) == "EmbeddingShard_v1":
+            from verity.commitments.gates.embedding import read
+
+            H, V = int(spec_statics(spec)["H"]), shard_vocab(spec)
+            hit += tuple((descriptor_id(read(V, H >> k)), 1 << k) for k in range(H.bit_length())
+                         if H % (1 << k) == 0 and H >> k <= 256)
```

`adapters/vllm/verity_vllm/records/`: the prover commits each block's padded table, V rows of G words, as
`expert_tables` commits an expert block's. Its root is then the registered one exactly when the committed rows are the
shard's at START and zeros elsewhere. Not drafted: `expert_tables`' committer is the model, and the padding is the
only new part.
