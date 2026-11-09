[2026-11-09T23:41Z] e44d4ad2c6d5

## PROVECUT, 23:50Z Mon 9 Nov: item 1 has a refusal, the sparse MLA blocks (the full table follows by 00:25Z)

- **Refused under `--zk`: the sparse MLA blocks.** [read] The served stager puts their EX2 reads in lookup slots
  (`chunk_rows.reads`, `integrations/vllm/verity_vllm/pipeline/chunk_rows.py:39-58`):
  - `SparseMla_v1`'s head blocks, `SparseMlaHeadBlock_v1` at a whole block (`CHAINS`, `serving_templates.py:86`);
  - `SparseMlaSplit_v1`'s parts with `T >= BN`, `SparseMlaSplitPart_v1` (`SPLITS`, `serving_templates.py:79`).
- **Why they are refused.** [read]
  - `read_slots` has the last stage return the outputs (`read_slots.py:433`), so META `out_net` is `stage{N}`
    (`verity_flock/circuit.py:838`).
  - `scopeOkH` refuses that (`ProvedScope.lean:150`, through `KsScope.unitWhy`, `KsScope.lean:90`).
  - The templated form gets no exemption: `scopeOkT` (`ProvedScope.lean:267`) has the same `outNet == unitNet`.
- **Not refused** [read], because they are inline: the chain end (`SparseMlaHeadEnd_v1`) and the split combine
  (`SparseMlaCombine_v1`).
- **My #8359 (open) would add two refused types:** FlashInfer's FP8 linear and its quantisation part, which it puts in
  slots (`SLOT_LINEARS`).
- Runs on small staged statements of each form, and the rest of the table, follow by 00:25Z.

