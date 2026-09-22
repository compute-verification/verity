# A5/6 extension — batch proving (n = 64) on one RTX 4090, 2026-09-11 (budget tag `2026-09-11-ext`)

Directive: `docs/handoffs-in/cursor-directive-2026-09-10-2059.md` priority 3 ("measure batch proving ... keep any unmeasured
improvement explicitly assumed"), integrator 04:12Z. Every number below is **MEASURED** on the pod named in §0 unless it says
NOT MEASURED or ASSUMED. Raw records: `gpu-r1.jsonl` / `gpu-r2.jsonl` (schema `veritor/a56/proof-record/v1`, one line per
(batch, mode); `prove_seconds` is the prover's own timer, `setup_seconds` — the SP1 GPU server start, 42–57 s per fresh host
process — is recorded beside it and excluded from every per-VU figure, as in L3's `prices.json`). Prices-contract rows:
`out/prices/prices_n64_2026-09-11.json` (§4). L3's `prices.json` and D7's `prices_reps.json` were not edited.

## 0. Pod, spend, identities

| | |
|---|---|
| pod | `ctr-a56-gpu-3-veritor-campaign`, RunPod **COMMUNITY** RTX 4090 24 GB, $0.34/h (cap ≤ $0.44/h) |
| first host `hgstmgxbvw93vd` 04:08–04:55Z | GPU **broken** on that machine (torch and `sp1-gpu-server` both fail with "CUDA unknown error", driver 580.126); both CUDA hosts were built there (371 s, parallel) and pulled; terminated, 404 confirmed; ≈ $0.27, no proof measured |
| second host `hcmnrxqgnxeedg` 04:57Z– | driver 570.133, CUDA ok; prebuilt hosts shipped; every measurement below; watchdog 1.6 h |
| prover | SP1 6.4.0, `SP1_PROVER=cuda` (SDK's stock `sp1-gpu-server` 6.6.0, spawned per host process) |
| approved guest | `veritor-zk-host` `--features cuda`, ELF `9fd00e60fb6a74762de987cde211aca9de4576de844569da779acac6bf3863db`, vk `0x0018b8e4…bd5797` — APPROVED |
| fa2 guest | `veritor-attn-host` `--features cuda`, ELF `33cba05834171e2518b6cf122ac40e36797ba111b8299a7f7bb546f4d7a219c9`, vk `0x00413469…ccadfb` — **UNAPPROVED** (enforcement evidence only) |
| spend | see §5 (ledger `~/.runpod/spend-2026-09-11.md`, budget file `~/.runpod/budget-2026-09-11-ext.json`) |

The batches are the ones built on 2026-09-10 and left "not measured" at n = 64 (`zk/bench/a56_approved_batches.py`: REAL W11
operands under a SYNTHETIC one-word-per-leaf tree standing in for the uncommitted attention interior — P2; `a56_head_batches.py`:
the PROPOSED head VU over record-style 16-word leaves), plus one new batch built tonight on the laptop under `lane_run`:
`zk/bench/a56_head_n64.py` — **64 head VUs in one format-3 statement** (128 obligations; 36 × `AttentionHead_v1{T=9,D=128}` +
28 × `{T=8,D=128}`, because the W11 natural fixture has 3 layers × 12 heads and 64 units of one kind do not exist in it;
transparent-checked on the laptop, 1.7 s).

## 1. n = 64 approved batches vs their n = 1 rows (item 1)

| batch | family (guest / ELF) | n | witness: execute s | core prove s (batch) | core s / VU | compressed prove s (batch) | compress (comp − core) s / VU | GPU-server setup s (excluded) | proof bytes core / compressed | cycles (batch) | peak RSS host proc. | amortisation core s/VU, n=1 ÷ n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `dot-bf16-K128_n1` | qk-f32-K128 (approved `9fd00e60`) | 1 | 2.61 | 5.10 | 5.097 | 7.02 | 1.924 | 43 | 2,831,128 / 1,272,602 | 2,085,068 | 82 MB | — |
| `dot-bf16-K128_n64` | qk-f32-K128 (approved `9fd00e60`) | 64 | 20.392 | 63.17 | 0.987 | 84.37 | 0.331 | 42 | 54,535,172 / 1,272,602 | 123,127,124 | 1258 MB | 5.2× |
| `tc-dot16-pv_n1` | attn-pv-dot16-K64 (approved `9fd00e60`) | 1 | 2.455 | 4.06 | 4.059 | 6.18 | 2.121 | 46 | 2,814,016 / 1,272,602 | 1,119,903 | 70 MB | — |
| `tc-dot16-pv_n64` | attn-pv-dot16-K64 (approved `9fd00e60`) | 64 | 9.829 | 30.36 | 0.474 | 37.85 | 0.117 | 42 | 27,291,936 / 1,272,602 | 62,390,237 | 670 MB | 8.6× |
| `f2fp-bf16_n1` | f2fp-bf16 (approved `9fd00e60`) | 1 | 2.195 | 2.21 | 2.208 | 4.46 | 2.250 | 47 | 2,797,496 / 1,272,602 | 184,849 | 64 MB | — |
| `f2fp-bf16_n64` | f2fp-bf16 (approved `9fd00e60`) | 64 | 2.539 | 4.89 | 0.076 | 7.31 | 0.038 | 45 | 2,820,672 / 1,272,602 | 1,581,697 | 76 MB | 28.9× |

`witness` = the host's `execute` wall (statement + witness parse, Merkle, gates in the executor; the Python build of every batch was
≤ 0.1 s per VU on the laptop yesterday). Peak RSS = GNU `time` `ru_maxrss` of the host process (the spawned `sp1-gpu-server`'s own RSS is not
necessarily included; GPU device memory not recorded). Single shots (reps = 1 per mode) — a second core rep did not fit the window.

Reading: the per-VU core price falls **5.2× (qk K=128), 8.6× (p·v 64-key), 28.9× (f2fp)** from n = 1 to n = 64 on the
approved guest — the amortisation is largest where the per-VU work is smallest (f2fp: 2 openings, 1 gate) and smallest where
the Merkle openings dominate (qk K=128: 257 one-word leaves per VU, 16,448 in the batch). These are **corroborating rows with
REAL operands**; L3's `prices.json` rows for the same families (synthetic contiguous leaves, "optimistic proxy", 4090 SECURE)
are lower per VU at both n — qk-f32-K128 2.45 → 0.46 s, attn-pv-dot16 (16-key) 1.31 → 0.10 s, f2fp-bf16 1.16 → 0.040 s — and
stand; nothing in this file replaces them (§4).

## 2. The head VU (O1) at n = 1 and n = 64 — fa2 guest, UNAPPROVED

| batch | family (guest / ELF) | n | witness: execute s | core prove s (batch) | core s / VU | compressed prove s (batch) | compress (comp − core) s / VU | GPU-server setup s (excluded) | proof bytes core / compressed | cycles (batch) | peak RSS host proc. | amortisation core s/VU, n=1 ÷ n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `head_T9_h5` | attn-head-T9-D128 (fa2 `33cba058` UNAPPROVED) | 1 | 7.012 | 50.87 | 50.866 | 59.56 | 8.693 | 57 | 24,698,512 / 1,272,602 | 3,589,483 | 619 MB | — |
30,835,368 / — | 18,949,447 | 749 MB | — |
| `head_n64` | attn-head-mixed-T8T9-D128 (fa2 `33cba058` UNAPPROVED) | 64 | 17.546 | 96.10 | 1.502 | 108.96 | 0.201 ‡ | 57 | 51,924,812 / 1,272,602 | 90,300,966 | 1103 MB | 33.9× |

‡ the `head_n64` compressed figure is reconstructed from the driver's log line (pod clock 06:06:00Z, `final-pull-0606Z.txt`): the pod was
terminated before the driver wrote the record (`reconstructed_from` in `gpu-r1.jsonl`). The approved host's `verify-many` refused every fa2
proof (ok = false, statement_match = true), as it must — the guest is not on the approved list.

The fa2 guest's fixed floor (F1: the 64 MB `include_bytes!` MUFU tables in the program image) is **≈ 50 s of core proving on
the 4090 at n = 1** (T=9: 50.9 s; T=65: 61.6 s, so 56 more keys cost 10.8 s) against 2.2–5.1 s for an approved n = 1
statement. At n = 64 the floor is paid once: 96.1 s for 64 heads = **1.50 s per head VU, 33.9× below the n = 1 row**
(marginal ≈ 0.72 s per additional head). This is the first measured amortisation of the head VU; the earlier CPU-pod row
(16 vCPU, 600 s+) is superseded for pricing purposes by these GPU rows, which are the hardware `prices.json` prices on.

## 3. Table-free head guest vs the tabled guest (item 2) — **NOT MEASURED**

Not built tonight. The variant is a guest **and gate-set change** (each `mufu_ex2_ftz` / `mufu_rcp_ftz` gate takes the table
row as an opened input authenticated against a committed table root, as `reduce-mufu@1`'s `*_tab` gates do on the approved
guest; the builder must open one table row per lookup — T + 1 per block — and the family digest changes), not a build flag;
with the first pod's broken GPU consuming 45 min of the window it could not be designed, built (≈ 6 min on the pod) and
measured before 06:15Z. What the measured rows say about it, labelled:

- MEASURED: the tabled guest's floor on the 4090 is ≈ 50 s core at n = 1 and ≈ 0.72 s marginal per head at n = 64 (§2).
- ASSUMED (not measured): a table-free guest removes the image floor at n = 1 and adds per-lookup opening work (23 hashes per
  row for a 2²³-entry table, ≈ 10 rows per one-block head); its n = 1 price would sit near the tabled guest's *marginal* cost
  plus that opening cost, and its n = 64 price would be close to the tabled guest's 1.50 s/VU. These are assumptions, not
  prices; conformance vectors + 12 negatives are needed before any promotion, as the directive says.

## 4. Prices-contract rows for `frontier_v3` — `out/prices/prices_n64_2026-09-11.json`

Schema `veritor/prices/v1`, family rows in the shape `frontier_v3.pricing.price_chain` reads (`name`, `auth_mode`, `guest`,
`elf_sha256`, `batch`, `prove_s`, `compress_s`, `total_s`, `witness_s`, `verify_s_per_batch`, `cycles_per_vu`, `reps`,
`hardware`, `provenance.label = MEASURED`, `commit`). Per-VU: `prove_s` = core / n; `compress_s` = (compressed − core) / n
when both modes were proved on this pod, else `null` with the note "compressed NOT MEASURED"; `total_s` accordingly. `reps`
states 1 or 2 per mode — **single shots, not medians** (three reps did not fit the window; D7's ≥ 3-rep rule means
`apply_prices_reps` would not consume them anyway — these rows are for a `--prices` variant run, not for `prices_reps`).

**For A7 — which families / rows changed:**

- **New families** (nothing in `prices.json` prices the head VU): `attn-head-T9-D128`, `attn-head-T65-D128` (n = 1) and
  `attn-head-mixed-T8T9-D128` (n = 64), guest `fa2-attn (UNAPPROVED)`, ELF `33cba058…`. Consumable only in an **O1 variant
  chain**; never in the approved chain.
- **Re-measured with REAL operands** on this pod at n = 1 and n = 64: `qk-f32-K128`, `f2fp-bf16` (same family names as L3,
  `auth_mode = multiproof`, approved ELF) and `attn-pv-dot16-K64` (a **different shape** from L3's 16-key `attn-pv-dot16`:
  64 keys, 130 openings/VU — named apart so the two are not conflated). L3's rows are not replaced; `frontier_v3` reads
  `prices.json` unless this file is passed as `--prices`, and the key `(name, auth_mode, guest, batch, tree_depth)` differs
  (`tree_depth` null here) so `apply_prices_reps` cannot transfer a price across.
- Unchanged: every other family; `prices.json`; `prices_reps.json`.

## 5. Spend and shutdown

| pod | window | rate | spend |
|---|---|---|---|
| `hgstmgxbvw93vd` (broken GPU; hosts built) | 04:08–04:55Z (0.78 h) | $0.34/h | ≈ $0.27 |
| `hcmnrxqgnxeedg` (all measurements) | 04:57–06:06Z (1.16 h) | $0.34/h | ≈ $0.39 |
| **total** | | | **≈ $0.66 of the $2.50 cap** (budget file `~/.runpod/budget-2026-09-11-ext.json` polls the exact figure) |

Both pods terminated by me; `GET /pods/<id>` → 404 confirmed for both at 04:55Z and 06:06Z; `runpod.py list` shows no a56 pod; both watchdogs killed.

## 6. Files

- `zk/bench/a56_gpu_run.sh` (pod bootstrap: both hosts `--features cuda`, parallel), `zk/bench/a56_pod_prove.py` (+ peak RSS
  of the host process tree via GNU `time`, `max_rss_kb`), `zk/bench/a56_head_n64.py` (the 64-head batch),
  `zk/bench/a56_prices_n64.py` (records → prices rows).
- `out/contract/a56-coverage/gpu/`: `gpu-r1.jsonl`, `gpu-r2.jsonl`, `stages.log`, `gpu-prove.log`, `toolchain.txt`, this report.
- `out/prices/prices_n64_2026-09-11.json`.
- Slim archive: `~/veritor-archive/A56-gpu-n64-2026-09-11/` (records, logs, host `info` JSON, batch manifests; no proofs, no binaries).
