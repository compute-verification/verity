# PoUW

PoUW, proof of useful work, tells the auditor how much GPU time went into the matrix products of a registered model:
each product runs under a scheme that noises it with a salt the prover learns only after its weights are committed,
and commits what a check needs; the challenger's coins then sample tiles, which the verifier checks. One scheme is
served, Pearl+ (`pearl-c-h100-v1-h3`, Pearl's FP8 on the H100 and H200), and its Lean check recomputes each sampled
tile's words from the opened A and B rows.

The guarantee: `Properties.Pouw.MonitoringPeriod.H1CheckedCreditMonitoringPeriod`
(`security/Properties/Pouw/MonitoringPeriod.lean`), deployed, for H1's tile unit, which is no longer served: except
with probability δ_s + n·(δ_bind + δ_ks), an accepted period's fixed words are at least (1 − ε)₊ of its live
words. It counts committed words and prices none. Pearl+'s properties (`Properties.Pouw.PearlPlus.AcceptMeans`,
`Properties.Pouw.PearlPlus.WorkIsTheCredit`, `security/Properties/Pouw/PearlPlus.lean`) are listed, not deployed: they
say what the check's accept means and that its published work is `creditAtAssumption`'s MACs, which no statement prices.

Trusted and assumed: the challenger on the auditor's device (`trusted-nebius-host`); the deployed statement's open
premises, among them `ServiceSound`, `SaltIsLiveCoin`, `StagedPins` and H1's lucks. Pearl+'s credit is computed at
Pearl's assumption TT_OUT, named in its code's prose; no statement ties that credit to it or to a device price.

Not yet: the deployed remote soundness takes PoUW's target (`Verity.Vocab.PouwSound`) as a hypothesis; MAC pricing is
only in two listed bridges, not deployed (`WorkOfH1CheckedCreditMonitoringPeriod`, `WorkOfCreditedWords`); the Pearl+
check verifies no proof and reads no circuit, its tie to the Definitions is proved and in no lock, and it opens A and
B values to the verifier; it reads no challenger key, so the node's Python alone checks the draw session's signatures,
and Python's verify must also accept for a verdict of record; nothing on the auditor's side checks the kernel binary.

## Pieces

| piece | what it is | where |
| --- | --- | --- |
| the scheme | how a product's operands become committed, checked values | `prover/schemes/` |
| the build | a kernel tree's `build.sh`, refused unless its `sass_gate.json` passes | `prover/kernels/pearl_c_sm90/` |
| the one driver | Pearl-C's pipeline on the device, either kernel tree | `adapters/torch/verity_torch/pouw/pearl_c/` |
| the hardware seam | what the GPU computes, bit for bit, as the check evaluates it | `verity/circuits/evaluate/` |
| the commitment seam | hm96 rows in frame-v3-sha512 trees; PoUW's tagged bindings | `verifier/lean/commit/` |
| the prover | serves beside the model, noises and commits each registered product | `prover/serving/` |
| the Lean verifier | Pearl+'s verdict and credit (`pouw-pearlplus`); H1's checks | `verifier/lean/pearlplus/` |
| the auditor's device | the challenger: it signs the coins | `verity/protocols/proof/spot/verifier/challenger/` |
| its circuits | each scheme's Definitions and an execution phase's Program | `prover/circuit/` |

## What varies

| implementation | kernels and build | Lean verifier | `verity run` |
| --- | --- | --- | --- |
| `pearl-c-h100-v1-h3` (FP8, sm_90a) | served | served | served |
| `pearl-c-sm120-v1-h3` (FP8, sm_120a) | exists | never written | never written |
| `pearl-c-nvfp4-v0` (NVFP4, sm_120a) | never written | never written | never written |

A new implementation supplies: a builder in `prover/schemes/`, its Definitions, and a kernel directory with its build.
Fixed for every implementation: the commitments, the salt, the coins, the sample and the openings.

## Interfaces

- From the workload: the model's registered linear layers, weights committed before the salt opens.
- From `run`: the challenger's coins, in order: the salt session after the weights' commitment, the passes served and
  committed, then the draw session that samples the tiles; each coin opens once and a failure is never redrawn.
- It gives: `pouw-pearlplus`'s answer, accept or reject with the run's commitment and its published work in credited
  MACs, to the remote and onsite protocols (`verity/protocols/run/verifier/`).

## Try it

~~~sh
uv run python -c "
from verity.protocols.accounting.compute.pouw import scheme
from verity.protocols.accounting.compute.pouw.prover import schemes
print(schemes.of(scheme.PEARL_PLUS, 'h100', pinned=True, arch='sm_90a').name)"
~~~

It prints `pearl-c-h100-v1-h3`, the kernel scheme Pearl+ runs on the H100, and needs no GPU.
