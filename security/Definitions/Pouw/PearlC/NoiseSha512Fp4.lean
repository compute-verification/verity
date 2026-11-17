module

public import Definitions.Pouw.PearlC.Defs
public import Definitions.Pouw.PearlC.Fp4
public import Definitions.Pouw.PearlC.Sha512

/-! Pearl-C4's noise as its registered h0 hashing samples it (`verity/protocols/experimental/pearl_c4/prover/pearl_c4.py`), with the salt's bytes and the commitment roots as parameters. -/

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

def drawA (sh : Shape) (seedA seedB : List ℕ) : Matrix (Fin sh.m) (Fin 32) Nib × Matrix (Fin sh.k) (Fin 32) Nib :=
  (fun i t => line seedA 0 i t, fun l t => line seedB 1 l t)

/-- **`B̃`'s lines from `seed_B`**: `E_B`'s `n` lines (role 2) and `F_B`'s `k` lines (role 3) of a unit of shape `sh`. -/
def sampleB (sh : Shape) (seedB : List ℕ) : Matrix (Fin sh.n) (Fin 32) Nib × Matrix (Fin sh.k) (Fin 32) Nib :=
  (fun j t => line seedB 2 j t, fun l t => line seedB 3 l t)

end NoiseSha512Fp4

end Pouw.PearlC
