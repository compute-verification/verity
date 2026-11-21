# proof

The proof layer establishes that the proof blocks a live sample picks from one registered circuit compute, on their
committed values, what the circuit says, from one zero-knowledge Flock proof a sampled block that Lean checks. SPOT
registers the docket and samples proof blocks on the challenger's live coins; Flock proves each sampled block over its hm96
rows, and every byte the workers send the auditor leaves through the ZK server.

The guarantee: `Flock.Guarantees.SampledKSLiveV5` (`security/Proofs/Flock/Bridge/V5/Statement.lean`).

Not yet: the outer (recursive) proof, which the first version leaves out; a two-stage audit, which the verifier refuses by name (`Spot.Levels.TWO_LEVELS_REFUSED`); the guarantee as an instance of the sampling layer's `Spot.Sampling.Sound` rather than proved on its own; the cover form for subset designs, without which a window whose templates interleave (every real model's) is refused; the check reading the weights' registration of record, without which a weight several blocks read is refused.

## Pieces

| piece | what it is | where |
| --- | --- | --- |
| the prover | lowers a sampled block to a Flock statement over its committed rows and proves it: Python, the Rust crate `flock-live`, CUDA | `flock/prover/` |
| the proof server | one `flock-circuit prove --zk` a GPU, proving the sampled blocks | `flock/prover/python/verity_flock/proof_server.py` |
| the Lean verifier | one session's verdict, `verity-verify-proof` (`verity verify proof`) | `flock/verifier/` |
| the ZK server | the workers' only exit: each message held to the send check, then released | `flock/verifier/Flock/ZkServer/`, `recursion/verifier/zk_server/` |
| the docket | the registration record, fixed before any coin | `spot/prover/docket.py`, `spot/verifier/one_stage/docket.py` |
| the sample | the sampling designs and the sample on live coins; the levels | `spot/verifier/one_stage/`, `spot/verifier/levels.py` |
| members | a window's statements, one a member | `spot/verifier/one_stage/members.py` |
| SPOT's Lean verifier | the audit's verdict of record (`spot-one-stage-zk`, `spot-verifier`) | `spot/verifier/lean/` |
| the challenger | the session's live coins | `spot/verifier/challenger/` |
| the parties | challenger, inference server, proof server, verifier (`SPOT_PROTOCOL`) | `spot/verifier/parties.py` |
| the integrity profile | ε at δ over the proof blocks, and what applications read off it | `spot/verifier/profile.py`, `spot/verifier/consumers.py` |
| SPOT[Flock] | the Lean check compose's two lists run | `spot_flock/verifier/lean/` |
| the agreement | the Lean verifier against upstream's Rust verifier | `flock/prover/agreement/` |
| the recursion | V\*, `FlockRec`, the `rec_*.py` provers: kept code that goes | `recursion/`, `spot_flock_recursion/` |

## What varies

| implementation | one-stage | two-stage | recursive | CPU build | sm_90 build | sm_120 build |
| --- | --- | --- | --- | --- | --- | --- |
| Flock | served | exists | exists | served | exists | exists |

two-stage: built, and refused by the verifier; recursive: built, and outside the first version.

A new implementation supplies: an argument system's prover for one statement over committed hm96 rows, its Lean
verifier, and its build for each GPU architecture (`flock/prover/cuda_chunk_patch.py` takes the architecture).
Fixed for every implementation: the docket and its digest, the sampling designs and the sample read from the challenger's
coins, hm96-sha512 rows in frame-v3-sha512 trees, the ZK server as the only exit, and
Lean's verdict of record.

## Interfaces

- From the workload: one circuit in proof blocks with its public layout, and its values committed as hm96 rows
  (`verity/commitments/periods/interface.py`), registered in the docket before any coin.
- From `run`: the sampling design and SPOT's parameters (`levels`), the challenger's live coins, and the parties' machines.
- It gives: each audit's verdict of record and its integrity profile, ε at δ, which the accounting and policy
  protocols read (`spot/verifier/consumers.py`).

## Try it

~~~sh
uv run pytest -q tests/verity/protocols/proof/spot/verifier/test_one_stage.py tests/verity/protocols/proof/spot/verifier/test_levels.py
~~~

It needs no GPU and no Lean build: `90 passed`, over the partition query, the docket, the sample, the proved-block
checks, the profile, and the levels read off the parameters.
