# PoUW's receipt folds

The one reference for the folds PoUW's receipts are made with: each fold's step, its key rule, the start states and keys
of the schedules that run it, and what is proved about it. Every implementation (pfill's filler, EGEMM's kernels, E2E's
PouwSegCheck statements, P2-FF's twin, the filler scorer's fold table) is held to it by its published vectors.

- `receipt_folds.py`: the reference, stdlib only (its steps also take numpy uint32 arrays).
- `receipt_folds.cuh`: the same in C++, `__host__ __device__` under nvcc, plain inline under g++. The steps are
  force-inlined and the key rules aren't, so a kernel that moves onto the header keeps its SASS (pfill's did).
- `vectors.json`: the published test vectors (inputs, keys, the expected chain), made by
  `python receipt_conform.py --write`. A vector that moves is a receipt that moves: say why in the PR.
- `receipt_conform.py`: the checks, and `python receipt_conform.py` to run them on the reference and the header.
- `conform_host.cpp`: the header's host driver the checks build.

## The folds

All words are uint32 and arithmetic wraps. An MMA gives a lane four words (d0, d1, d2, d3): rows (g, g + 8) by columns
(2t, 2t + 1).

| fold | step | key rule | soundness |
|---|---|---|---|
| duo | per word: u = y ^ d ^ k, y ← u·a + z, z ← rotl(u, r) + y | pfill: key_ok; served: fixed odd keys | needs (D) past four words; one, two and three proved, four at both schedules' keys |
| xin13 | y ← (rotl(y, 13) ^ d0)·a0 + d1, then (rotl(y, 19) ^ d2)·a1 + d3 | key_ok_r9; key_ok_r11 for r11's Theorem 1 | needs (RT) and (G₃) |
| li | y ← (y ^ d0 ^ k0)·a0 + d1, then (y ^ d2 ^ k1)·a1 + d3 | key_ok_r9 | needs (RT), (G₃) and (HO); with pinned openings and a checkpoint a segment, nothing beyond odd multipliers (proved) |
| ram | y ← (rotl(y, 7) + d0)·a0 + d1, then (rotl(y, 19) + d2)·a1 + d3 | key_ok | binds pairs only, through Q = a0·d0 + d1 |
| xin7 | xin13 with rotations (7, 19) | key_ok | retired: ±128 is an int8 ratio at its step boundary |

`FOLDS` in `receipt_folds.py` carries each status with its source:

- **(D), duo's premise.** Two different runs of a chain's words leave different final states on some start state
  (pfill commits the whole state; a served chain's receipt y ^ rotl(z, 16) needs the same up to one constant). It is
  proved at every width for one, two and three words, for odd multipliers and rotations 0 < r < n (Lean:
  `Pouw.SecurityProofs.Receipt.DuoWordInjective`, `Pouw.SecurityProofs.Receipt.DuoPairNoKernel`,
  `Pouw.SecurityProofs.Receipt.DuoTripleNoKernel`). At four words it is false for odd keys with r1 + r3 ≡ 0 (mod 32),
  a2 ≡ −1 (mod 2^r1) and a3 ≡ ±1 (mod 2^(33 − r1)): flipping the top bit of all four words changes no final state.
  key_ok rejects such keys, and neither schedule has r_t + r_(t+2) ≡ 0 (mod 32). At four words, any kernel needs
  a2 ≡ −1 (mod 2^r1), for odd a1, a2, a3, rotations 0 < r2, r3 < n and 0 < r1 ≤ n − 3 (Lean:
  `Pouw.SecurityProofs.Receipt.DuoQuadNoKernel`). key_ok rejects a ≡ −1 (mod 32) and pfill's rotations are 5 to 29,
  and no served position t has SERVED_KA[(t + 2) mod 8] ≡ −1 (mod 2^SERVED_ROT[t]), whose rotations are 3 to 27. So
  both schedules bind every four-word run of the whole state. `benchmarks/pouw/theory/duo_kernel.c` finds every
  four-word kernel at width 4 to be the top-bit family, and every kernel it has found at widths 6 and 8 is too.
  In a run of any length, four words with those keys have no kernel whatever the words before and after them (Lean
  `Pouw.SecurityProofs.Receipt.DuoWindowNoKernel`), so two runs of either schedule with the same final states differ,
  if at all, at two words at least four positions apart; and a run whose state after every four words is committed
  has no kernel at any length (`Pouw.SecurityProofs.Receipt.DuoCheckpointNoKernel`). No schedule binds every length:
  every run of more than 2·4^n words has a kernel, by counting (`Pouw.SecurityProofs.Receipt.DuoLongRunKernel`), so
  the deployed schedules' first kernel lies between five words and 2·4^32 + 1. Still open: five or more words for the
  deployed keys, and the served receipt's form (equal up to one constant) at any length.
- **(RT) and (G₃).** RS-E-REF Part IX (note:20261015T1931Z-draft-theorem-e-referee) says (RT) is open and that no lemma
  removes it, and that (G₃) suffices for (G₄).
- **r11's results.** P2-FF r11 (note:20261016T1549Z-draft-fullfloor-s-r11) finds that pricing can't replace (RT). It
  proves (G₃) at one MMA under key_ok_r11 (its Theorem 1), and for chains apart from runs of two or more non-XOR δ.
- **(HO), li's extra premise.** No hand-off joins two words whose rows or columns are equal or complementary
  (note:20261016T1117Z-draft-fullfloor-s-r9 §2).
- **li with pinned openings and a checkpoint a segment.** When every absorbed word is recomputed from committed operands
  and a pinned start, and each segment commits the XOR of the chains' states, li needs none of (RT), (G₃), (HO) or a key
  rule beyond odd multipliers. A half step is a bijection of the state, a checkpoint is injective in each word any chain
  absorbed, and changing one word by XOR with x changes the checkpoint at x's lowest set bit and at no bit below it, at
  every width (Lean: `Pouw.SecurityProofs.Receipt.LiHalfBijective`, `Pouw.SecurityProofs.Receipt.LiCheckpointWordInjective`,
  `Pouw.SecurityProofs.Receipt.LiCheckpointLowBit`). So a skipped value that lands on one word of a segment is always
  caught. One receipt word a window is not enough: li is a T-function, so a persistent skip's copies all differ first at
  the same bit, and an even number of them can cancel there (note:20261017T1848Z-draft-li-alt).

The key rules:

- **key_ok:** a is odd, a mod 2^24 is at least 2^8 + 1 away from 0, and a is not ±1 mod 32.
- **lambda8:** a is odd, and a·b mod 2^30 lies outside [−128, 128] for every b in 1..128.
- **key_ok_r9:** key_ok and lambda8.
- **KJ19:** for each j < 19, the signed residue of a mod 2^(31−j) is at least 2^(19−j) in size.
- **key_ok_r11:** key_ok_r9 and KJ19.

An exhaustive count over the 2^31 odd keys (`key_rates` in the vectors) gives KJ19 rejecting 10,481,280 of them
(4.88e-3, as r11 states), and 0.447% of key_ok_r9 keys (r11 says 0.42%).

## The schedules

The steps are shared and the schedules are not. Each schedule's start state and keys are in `receipt_folds.py`, and its
vectors are under `schedules`.

| schedule | runs | chains a thread | start | keys |
|---|---|---|---|---|
| pfill-lane | pfill.cuh's `pf_absorb`, pfill_ref.py | 4; fragment f goes to chain f % 4 | y = pos, z = pos·0x9E3779B9 | the key block's first two rows that pass the rule, else 0x9E3779B9 and 0x7F4A7C15; k0 and k1 are rows 14 and 15 |
| served-duo | fgemm.cu, EGEMM's duo, E2E's duo | 8; entry 4t + e goes to chain 4(t % 2) + e | y = mix32(seed ^ mix32(idx)), z = mix32(y + 0x9E3779B9) | position t's key_mul(0, t), key_xor(0, t), rot_of(DUO, 0, t) |
| egemm-xin13 | EGEMM's x13_l2o | 4; warp tile t goes to chain t % 4 | y = (tile·512 + ctid)·4 + c | the first two mix32(seed ^ (0x5A5A0000 + i)) that pass key_ok_r9 |
| e2e-xin | E2E's PouwXinAbsorb | 8; warp tile t goes to chain t | duo's y | per chain, from mix32(0xC0000000 + i) \| 1 under lambda8 |

E2E's xin keys are sampled under lambda8 alone, not key_ok_r9. Of its 16 keys, XA0[1] fails key_ok and XA1[0] fails KJ19
(`tests/test_receipt_folds.py`).

## Conforming

A Python implementation calls the checks with its own functions:

```python
import receipt_conform as C
C.check_steps({"xin13": my_step, "duo": my_duo_step})   # receipt_folds' signatures
C.check_key_rules({"key_ok_r9": my_rule})
C.check_header("my_folds.cuh")                          # a header defining receipt_folds.cuh's rf_ functions
```

`p2ff_r9`, `pfill_ref`, `egemm_ref`, `e2e_segment` and `scorer_folds` check the existing implementations whole, each
against the schedule it runs. `pfill_cuh` builds pfill.cuh for the host with g++, as pfill_host.cpp does, once a fold build
(default, `-DPFILL_LI`, `-DPFILL_XIN13`) and runs `pf_absorb`, the build's key rule, `pf_keys` and the lane starts over
the vectors. `e2e_definitions` evaluates E2E's Boolean Definitions (PouwDuoAbsorb_v2, PouwXinAbsorb_v2
and their finals) on bits over the same vectors. `benchmarks/pouw/tests/test_receipts_adopters.py` runs them on the ones under
`benchmarks/pouw` and on E2E's, and skips each until it lands. For any module, run
`python receipt_conform.py --adopter NAME PATH`. A kernel is held to its host twin by its own gate, and the twin is held
to these vectors.
