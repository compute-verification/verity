# PoUW

PoUW, proof of useful work, credits a registered model's matrix products with the multiply-adds its check accepts:
each product is noised with a salt meant to open only after its weights are committed; today that order is checked on
the served node, by the Lean against the challenger's signed log, not by the auditor. The challenger's coins then
sample tiles, which the Lean check recomputes from their opened A and B rows. One scheme is served, Pearl+
(`pearl-c-h100-v1-h3`, Pearl's FP8 on the H100 and H200); PoUW's one deployed statement,
`Properties.Pouw.MonitoringPeriod.H1CheckedCreditMonitoringPeriod`, is for H1's tile unit, which nothing serves.

The guarantee: `Properties.Pouw.PearlPlus.WorkIsTheCredit` (`security/Properties/Pouw/PearlPlus.lean`), not deployed.

Not yet: a deployed Pearl+ statement (remote soundness assumes `Verity.Vocab.PouwSound`) or a priced one; a check that
hides A and B, or reads the circuit or the kernel binary. The sessions: `--draw-live` and `--salt-live` hold the sample
and each salt to their signed sessions under the pinned key; the served road passes both but runs that Lean on the node,
which supplies the gate's fetch (asked-once holds only over it), the pins, retention, the period rule and the compose.
So no coin of a Pearl+ verdict of record is held to the challenger against a prover that controls the node. Grinding (a
steer needing m noised coordinates succeeds with probability at most μ^m a try) is stated nowhere: Pearl+'s one seed is
`seedB` over the salt and B's root, and GRIND's priced premium is H1's, proved and in no lock. Deletable: nothing in
PoUW's Python, data or kernel sources (each thing that looks dead has one reader); the FP4, FP8 and INT8 roads, the
filler and Dither are kept paths. Of our 754 Lean modules, 139 are imported by no listed guarantee's module, 106 of them
on kept paths; of the other 33 none is deletable: 16 hold what no lock lists (honest twins of unwitnessed guarantees,
the grinding and steering theorems the composition's spec cites, single reductions such as `sakMatmul_of_residual`), 9
are empty roots lake or the Lean check needs, 8 are not established (read at `one-circuit` `38285e72851`).

## Pieces

| piece | what it is | where |
| --- | --- | --- |
| the scheme | how a product's operands become committed, checked values | `prover/schemes/` |
| the build | `build.sh`, refused unless `build_gate.py` passes its `sass_gate.json` | `prover/kernels/pearl_c_sm90/` |
| the device driver | not yet one: each kernel tree's own `run.py`, run by `run_device.py` | [`prover/kernels/`](prover/kernels/README.md) |
| the hardware seam | GPU arithmetic, Python and Lean | `verity/circuits/silicon/`, `verity/circuits/evaluate/` |
| the commitment seam | hm96 rows in frame-v3-sha512 trees; PoUW's tagged bindings | `verifier/lean/commit/` |
| the prover | serves beside the model, noises and commits each registered product | `prover/serving/` |
| the Lean verifier | Pearl+'s verdict of record and its credit (`pouw-pearlplus`) | `verifier/lean/pearlplus/` |
| the auditor's device | the challenger (`trusted-device`): signs its coin commitments and logs | `proof/spot/verifier/challenger/` |
| its circuits | each scheme's Definitions and an execution phase's Program | `prover/circuit/` |

## What varies

| implementation | kernels and build | Lean verifier | `verity run` |
| --- | --- | --- | --- |
| `pearl-c-h100-v1-h3` (FP8, sm_90a) | served | served | served |
| `pearl-c-sm120-v1-h3` (FP8, sm_120a) | exists | never written | never written |
| `pearl-c-nvfp4-v0` (NVFP4, sm_120a) | deleted | never written | never written |
| `ncp-v2` (int7) | exists | never written | never written |
| `dither` (int8) | exists | never written | never written |

A new implementation supplies: a builder in `prover/schemes/`, its Definitions, and a kernel directory with its build.
Fixed for every implementation: the commitments, the salt, the coins, the sample and the openings.

## Interfaces

- From the workload: the model's registered linear layers, weights committed before the salt opens.
- From `run`: the challenger's coins, in order: the salt session after the weights' commitment, the passes served and
  committed, then the draw session that samples the tiles; each coin opens once, and a retry never resamples.
- It gives: `pouw-pearlplus`'s verdict, accept or reject with the run's commitment and its work in MACs credited at
  Pearl's TT_OUT, a name it records (`Pouw.PearlC.Assumptions.TTOutTilePearlCH100Rev1`) that no statement ties to the
  credit, to the remote and onsite protocols (`verity/protocols/run/verifier/`).

## Try it

~~~sh
uv run python -c "
from verity.protocols.accounting.compute.pouw import scheme
from verity.protocols.accounting.compute.pouw.prover import schemes
print(schemes.of(scheme.PEARL_PLUS, 'h100', pinned=True, arch='sm_90a').name)"
~~~

It prints `pearl-c-h100-v1-h3`, the kernel scheme Pearl+ runs on the H100, and needs no GPU.
