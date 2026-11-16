# verity_vllm.check

The adapter's own checks of a workload it served. The adapter has to show that what it compiled and committed is
right: that the Programs its Build derived are what vLLM executed, and that the values it committed are the ones those
Programs compute. This package runs those checks over the records the adapter's stages wrote, and gives their verdicts.

There are two verdicts. The Match's gates say whether a recorded execution agrees with the Programs. The Commit's
verdict says whether the committed values agree with what was served, as one document beside the files it was derived
from.

| Part | What it checks |
| --- | --- |
| `match` | A recorded execution's fold against the workload Program its Build declared. |
| `replay` | A Commit's committed values: sampled rows evaluated again and compared bit for bit. |
| The weights of record | That the served weights are the checkpoint of record's, from the checkpoint's own bytes. |
| The gates | A Match's frozen invariants over the records of one run directory. |
| The verdicts | A Commit's rules, folded into one verdict document. |

Each module's docstring says what it holds. [`DESIGN.md`](DESIGN.md) lists which names other code imports from each
module.
