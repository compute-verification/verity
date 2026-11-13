**`census:DsaIndexerQ_v1{H=32,D=128,ROPE=64}` verifies bound with all four verdicts, and so does the head block's Boolean version at 64 keys.** These are the two rows of run `r20261112-183158-1477`, which ended with rc 0. `dsa-q`'s replacement in the retirement's pull request stays `dsa-q-row` until circuits' lead says otherwise. [read by PARTSRETIRE from the run's `out/job.log`, `out/head64b/` and `out/dsa-q-word/` with `scratch/cir259/table.py`]

| `PARTS` type | staged with `--bind` as | N | ANDs, `k_log` | staging | Lean verifier, with its partition object and Program |
|---|---|---|---|---|---|
| `sparse-mla-head` | `census:SparseMlaHeadBlock_v2{DR=64,DV=512,NVIS=64,FIRST=false,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v2"},SCALE={"f64":"0x1.0000000000000p-4"}}` | 1 | 30,893,497, 27 | 1:45:08 at 202.6 GB; program.json 2,334,922,323 bytes | all four; Lean at most 100.5 GB |
| `dsa-q`, as the census names it | `census:DsaIndexerQ_v1{H=32,D=128,ROPE=64}` | 1 | 21,088,736, 26 | 31:59 at 43.0 GB; program.json 121,076 bytes | all four; Lean at most 140.4 GB |

- **"All four"** is as in CENSUSBIND's table:
  - `statement` derives the units;
  - `ks-form` passes;
  - `verify --zk` accepts one CPU session;
  - another Definition's Program (`SiluMul_v4{I=256}`) is refused at both `statement` and `verify --zk` with "setup: partition: the verifier's program is not the one the partition object names".

  `verify --zk` with no Program is refused: "refused: a session without the verifier's program (--program): …".
- **Both are at N = 1**, staged with `stage.py --bind` from `31b1b56aec2a` on vy-nebius-cpu-11. Neither has a session at a full statement's count.
- **Memory.** The Lean verifier's 140.4 GB on `DsaIndexerQ_v1` is above the 65.6 GB that was the largest in CENSUSBIND's table. Its `circuit.txt` is 35,924,397,430 bytes.
