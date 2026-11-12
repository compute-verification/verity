module

public import Definitions.Pouw.PearlC.Defs
public import Definitions.Pouw.PearlC.Fp4
public import Definitions.Pouw.PearlC.Sha512

/-!
# Pearl-C4's noise as SHA-512 gives it (trusted; definitions only)

`Fp4Sem.lines` is abstract. This module is the noise Pearl-C4's registered hashing (`pearl-c-nvfp4-v0`, h0) samples, as
a function of the salt, the commitment roots and the unit's index, through the spec's own SHA-512 (`Sha512.digest`), not
a random oracle. D-SK (`Fp4Sem.SaltKeyedB`) reads `seedB` and `drawB`, and `sha512Fp4Sem` (beside `Fp4Sem`) is the
semantics whose lines these rules give; the oracle `H` is then unused, and what is uniform is the salt.

The rules, op for op, are `verity/protocols/pouw/prover/schemes/pearl_c4.py` at h0:
- `label name` (`pearl_c4.LABEL_BLOCKS`): the label's ASCII bytes and a NUL, zero-padded to 128 bytes, for the labels
  `"pearl-c4/v1/seed-A"`, `"pearl-c4/v1/seed-B"` and `"pearl-c4/v2/line"` (`verity.commitments.domains`).
- `labelled name msg` (`pearl_c4.labelled`): the first 32 bytes of SHA-512 over the label block and `msg`.
- `seedB` (`pearl_c4.seed_b`): `labelled "pearl-c4/v1/seed-B" (salt ‖ root_B)`, one per weight and salt.
- `seedA` (`pearl_c4.unit_seed_a`, `PearlC4.noise`): `labelled "pearl-c4/v1/seed-A" (salt ‖ root_A ‖ root_B ‖ u64le u)`,
  one per unit `u`.
- `lineBlock seed role c` (`pearl_c4.line_block`): the whole 64 bytes of SHA-512 over the line label's block,
  `seed ‖ [role] ‖ u32le c`.
- `lineList seed role i` (`pearl_c4.line`): bytes `[16·(i mod 4), 16·(i mod 4) + 16)` of `lineBlock seed role ⌊i/4⌋`,
  each byte two E2M1 codes, its low nibble first (`pearl_c4._codes`).
- The lines (`PearlC4.encode`, `_side`, `basis`), with the roles `ROLE_EA = 0`, `ROLE_FA = 1`, `ROLE_EB = 2` and
  `ROLE_FB = 3`: `E_A` row `i` is line `i` of `seedA` at role 0, `F_A` row `t` line `t` of `seedB` at role 1, `E_B` row `j`
  line `j` of `seedB` at role 2, and `F_B` row `t` line `t` of `seedB` at role 3 (`drawB`).

**What is a parameter.**
- The salt's bytes, `salt : S → List ℕ`: the 32-byte epoch salt is `NoiseSha512.Salt32`, uniform when the game samples
  it uniformly.
- `rootB w`, the 64-byte root of weight `w`'s hiding commitment, fixed with the weights before the salt; and `rootA u act`,
  the 64-byte root of unit `u`'s committed activations `act`. Each commitment's opening (its leaves' salts) is the
  committer's randomness, so the noise does not determine either root from the codes.
- `alpha` and `beta` are the base `Fp4Sem`'s: only the lines are pinned here.

`pearl_c4.py`'s `-h3` hashing, which seeds each `E_A` line from its own row (`row_seed`), is not registered, and is not
stated here. The Python's refusals (a line block index of `2^32` or more) are total here: the indices are taken mod
`2^64` and `2^32` as `u64le` and `u32le` bytes.
-/

@[expose] public section

namespace Pouw.PearlC

open Pouw.PearlC.Sha512 Pouw.PearlC.Fp4

namespace NoiseSha512Fp4

/-- A label's block: its ASCII bytes and a NUL, zero-padded to 128 bytes (`pearl_c4.LABEL_BLOCKS`). -/
def label (name : String) : List ℕ := padTo 128 (ascii name ++ [0])

/-- `H_name(msg)`: the first 32 bytes of SHA-512 of the label's block and `msg` (`pearl_c4.labelled`). -/
def labelled (name : String) (msg : List ℕ) : List ℕ := (digest (label name ++ msg)).take 32

/-- `seed_B` of weight root `rootB` at the salt's bytes `salt` (`pearl_c4.seed_b`). -/
def seedB (salt rootB : List ℕ) : List ℕ := labelled "pearl-c4/v1/seed-B" (salt ++ rootB)

/-- Unit `u`'s `seed_A`, at activation root `rootA` and weight root `rootB` (`pearl_c4.unit_seed_a`). -/
def seedA (salt rootA rootB : List ℕ) (u : ℕ) : List ℕ :=
  labelled "pearl-c4/v1/seed-A" (salt ++ rootA ++ rootB ++ leBytes 8 u)

/-- Lines `4c` to `4c + 3` of `(seed, role)`, 16 bytes each: SHA-512 under the line label of `seed ‖ role ‖ u32le c`
(`pearl_c4.line_block`). -/
def lineBlock (seed : List ℕ) (role c : ℕ) : List ℕ := digest (label "pearl-c4/v2/line" ++ seed ++ [role] ++ leBytes 4 c)

/-- Two E2M1 codes per byte, its low nibble first (`pearl_c4._codes`). -/
def codes (raw : List ℕ) : List ℕ := raw.flatMap fun b => [b % 16, b / 16 % 16]

/-- One noise line's 32 codes as a list: quarter `i mod 4` of `lineBlock seed role ⌊i/4⌋` (`pearl_c4.line`). -/
def lineList (seed : List ℕ) (role i : ℕ) : List ℕ := codes (((lineBlock seed role (i / 4)).drop (16 * (i % 4))).take 16)

/-- **One noise line**, code `t` of line `i` of `(seed, role)`. -/
def line (seed : List ℕ) (role i : ℕ) (t : Fin 32) : Nib :=
  ⟨(lineList seed role i).getD t.val 0 % 16, Nat.mod_lt _ (by decide)⟩

/-- `E_A` (role 0, from `seed_A`) and `F_A` (role 1, from `seed_B`) of a unit of shape `sh`. -/
def drawA (sh : Shape) (seedA seedB : List ℕ) : Matrix (Fin sh.m) (Fin 32) Nib × Matrix (Fin sh.k) (Fin 32) Nib :=
  (fun i t => line seedA 0 i t, fun l t => line seedB 1 l t)

/-- **`B̃`'s lines from `seed_B`**: `E_B`'s `n` lines (role 2) and `F_B`'s `k` lines (role 3) of a unit of shape `sh`. -/
def drawB (sh : Shape) (seedB : List ℕ) : Matrix (Fin sh.n) (Fin 32) Nib × Matrix (Fin sh.k) (Fin 32) Nib :=
  (fun j t => line seedB 2 j t, fun l t => line seedB 3 l t)

end NoiseSha512Fp4

end Pouw.PearlC
