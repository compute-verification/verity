Lane circuits
Refs CIR-259

`benchmarks/served_units/stage.py` no longer stages its 18 `PARTS` types, the `Q_word` classes of word Definitions whose statements claim proof blocks the Lean verifier cannot derive. In their place each caller stages a whole Definition, bound by default to the Program of its instances and that Program's partition object, with `partition.json` and `program.json` written beside the statement. `stage.py` refuses a retired name and says what to stage instead, or that nothing does. `--bind` is gone, and `--unbound` stages the old way for the two callers that hand Lean neither file. This PR holds the `--bind` commits from `cursor/circuits-cir245-census-bind-6f04` (head `31b1b56aec2a`) and merges `origin/main` at `5f0c008aa040`.

## What `parse()` says for each retired name

__RETIRED_TABLE__

The census names, and the three with nothing in their place, are the ones CIR-259's first part staged and verified bound (CENSUSBIND's table on CIR-259). `sparse-mla-head`'s key block is the Boolean version `SparseMlaHeadBlock_v2`: the census's own v1 block at 64 keys reads its exponentials in lookup slots, a form `verify --zk` refuses, and `chunk_rows.zk_refusal` names the v2 block as the one to stage. A test holds each replacement to `parse()` and each census replacement to `zk_refusal` being empty.

## What each caller hands the Lean verifier now

__CALLERS__

## What moved

__MOVED__

## Checks

__CHECKS__

## Not done, or not established

__NOT_DONE__
