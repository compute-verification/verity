# A5/6 pod run — component timings (CPU, SECURE pod `ctr-a56-cpu-1`, cpu3m 16 vCPU)

Hosts built on the pod from the shipped trees with the pinned toolchain (sp1 6.4.0): APPROVED `zk/sp1` → ELF `9fd00e60fb6a7476…` (matches approved entry 9fd00e60: **True**), vk `0x0018b8e4e955ad4c…`; fa2 `zk/sp1-attn` → ELF `33cba05834171e25…` (matches TABLE-3A's 33cba058: **True**; UNAPPROVED), vk `0x00413469f3964a62…`. Machine: x86_64, 192 vCPU, SP1_PROVER=cpu.

Every row is a COMPONENT TIMING (one batch, one prover process, CPU). The fa2 column is labelled `fa2 guest 33cba058, UNAPPROVED`; the head VU is a PROPOSED partition change. The approved column's operands are REAL W11 rows over a SYNTHETIC one-word-per-leaf interior tree (P2: no record commits these words). `verify-many` = the approved host's long-lived verifier: approved-column proofs accepted; fa2 proofs REFUSED by construction (other ELF/vk) — the `GUEST_NOT_APPROVED` outcome a certificate would give.

| column | batch | kind | obligations | gates | statement B | witness B | mode | cycles | setup s | prove s | proof bytes | verify s (same host) | verify-many (approved host) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| approved 9fd00e60 | `dot-bf16-K128_n1` | `DotBf16_v1{K=128}|call` | 1 | 8 | 11,053 | 1,849 | core | 2,085,068 | 2.72 | **73.56** | 2,831,128 | 0.124 | ok 0.108 s |
| approved 9fd00e60 | `tc-dot16-pv_n1` | `AmpereBF16TcDot16_v1|batch (x4)` | 1 | 4 | 5,760 | 1,025 | core | 1,119,903 | 2.80 | **55.02** | 2,814,016 | 0.119 | ok 0.107 s |
| approved 9fd00e60 | `f2fp-bf16_n1` | `F2fpBf16_v1|primitive` | 1 | 1 | 447 | 63 | core | 184,849 | 2.74 | **33.71** | 2,797,496 | 0.121 | ok 0.105 s |
| fa2 (UNAPPROVED 33cba058) | `head_T9_h5` | `AttentionHead_v1{T=9,D=128}|call` | 2 | 502 | 69,530 | 57,431 | core | 3,589,483 | 14.71 | **623.60** | 18,046,614 | 0.722 | **REFUSED** (ok=false) |
| fa2 (UNAPPROVED 33cba058) | `head_T65_h5` | `AttentionHead_v1{T=65,D=128}|call` | 3 | 1,881 | 383,661 | 365,783 | core | 18,949,447 | 14.49 | **860.04** | 22,730,124 | 0.838 | **REFUSED** (ok=false) |
| approved 9fd00e60 | `dot-bf16-K128_n1` | `DotBf16_v1{K=128}|call` | 1 | 8 | 11,053 | 1,849 | compressed | 2,085,068 | 2.36 | **125.41** | 1,272,602 | 0.044 | ok 0.045 s ‡ |
| approved 9fd00e60 | `tc-dot16-pv_n1` | `AmpereBF16TcDot16_v1|batch (x4)` | 1 | 4 | 5,760 | 1,025 | compressed | 1,119,903 | 2.35 | **102.97** | 1,272,602 | 0.044 | ok 0.044 s ‡ |
| approved 9fd00e60 | `f2fp-bf16_n1` | `F2fpBf16_v1|primitive` | 1 | 1 | 447 | 63 | compressed | 184,849 | 2.33 | **87.13** | 1,272,602 | 0.045 | ok 0.044 s ‡ |
| fa2 (UNAPPROVED 33cba058) | `head_T9_h5` | `AttentionHead_v1{T=9,D=128}|call` | 2 | 502 | 69,530 | 57,431 | compressed | 3,589,483 | 12.52 | **809.86** | 1,272,602 | 0.044 | **REFUSED** (ok=false) ‡ |
| fa2 (UNAPPROVED 33cba058) | `head_T65_h5` | `AttentionHead_v1{T=65,D=128}|call` | 3 | 1,881 | 383,661 | 365,783 | compressed | 18,949,447 | 12.58 | **1,122.24** | 1,272,602 | 0.044 | **REFUSED** (ok=false) ‡ |

‡ rows reconstructed from the pod's `prove.log` (the host's printed prove/setup/bytes/verify figures; the pass-2 `records.jsonl` lines were lost when the pod was terminated before the final pull): cycles copied from the core row of the same statement, shards not recovered; the same-host verify s is the prove command's own post-prove verify.

verify-many ready: setup 2.37 s (vk derivation once), client 27.36 s.
