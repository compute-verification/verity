# CIR-274 item 2: the census at the work head, 4 then 78 layers

Read Thu 12 Nov 5:47 PM PST. Run `r20261113-013401-0b2d` (vy-nebius-cpu-11, `--mem-gb 32`, one run, `census_head.sh`), rc 0,
class SUCCESS. Source SHA `35f065487448224709c1ccfa45c90d409bf8d91b` (`$OUT/source_sha.txt`, `git rev-parse HEAD` in the shipped
tree), the work branch's head. `$OUT` keeps `p3_l4_r0.json` and `p3_l78_r0.json`. [run r20261113-013401-0b2d]

The 48 GB watch (VmRSS each second, `kill -9` by PID past it) never fired: peaks 447236 kB at 4 layers and 1473660 kB at 78
(`watch_l4.txt`, `watch_l78.txt`). [run r20261113-013401-0b2d]

| | 4 layers | 78 layers |
|---|---|---|
| program digest | `7f36f17ae005fc05…` (319 nodes) | `3c99109308321b3b…` (6979 nodes) |
| window digest | `d5301f52fe1a2e2a20afabcab027a811…` | `8e581226193ac78933b22693d53fe5e6…` |
| units, total | 3433430 | 60701290 |
| templates, total | 23 | 23 |
| wire patterns | 117 | 1747 |
| window wall s | 9.9 | 182.8 |
| peak GiB (in process) | 0.525 | 1.406 |

By family, at 4 layers / 78 layers [run r20261113-013401-0b2d]:

| family | units | templates | wires |
|---|---|---|---|
| DeepGemmFp8Linear | 3336284 / 58948098 | 8 / 8 | 52 / 1014 |
| FlashInferFp8Linear | 77588 / 1370886 | 8 / 8 | 45 / 859 |
| RMSNormFusedCudaVector8 | 9934 / 219946 | 3 / 3 | 20 / 464 |
| gather | 6588 / 130020 | 4 / 4 | 40 / 780 |
| DsaRMSNorm | 1584 / 30888 | 2 / 2 | 24 / 468 |
| RMSNormCudaVector8 | 1452 / 1452 | 2 / 2 | 2 / 2 |

The 4-layer window digest equals `d5301f52fe1a…` (full: `d5301f52fe1a2e2a…99e4463f05`): yes, the work head's window at 4 layers
is the one already reported. Everything but RMSNormCudaVector8 grows with the layers; it is the head's final norm, once a
request. [run r20261113-013401-0b2d; the last sentence inferred]
