import Definitions.Pouw.PearlC.PeelExact

/-!
# U's pinned order over a `Pipeline` (definitions only)

`PeelExact`'s U with the E4M3 chains' step and window a parameter, so that a device whose FP8 step is another
`Pipeline` has its own U beside Π's: sm_120's `mma.sync` m16n8k32 (`sm120E4m3`, `⟨[32], 26, −133⟩`,
`BLACKWELL_SM120_E4M3_M16N8K32`) for `pearl-c-sm120-v1` at `G = 4`. The four E4M3 chains (`C̃`, `A′·F_B`, `B̃·F_A`,
`G = F_Bᵀ·F_A`) run on `p` in windows of `G`; the roundings, the factor rows and the K16 step
(`PeelExact.hopperBf16K16`, `⟨[16], 26, −133⟩`) are `PeelExact`'s own.

* **The chain** (`windowWordP? p`, `chainWordP? p G`): each window's `p.step`s from `+0`, each word finite, and each
  window's word promoted by one FP32 add (`PeelExact.promoteWord?`) into a total from `+0`.
* **U** (`peelExactP p G`, `peelExactValP p G`): `PeelExact.peelExact`'s order with `chainWordP? p G` for its chains.
* **No promotion** (`pureWordP? p`, `peelExactPureP p`, `peelExactPureValP p`): the chains as one fold over all the
  atoms. `G = 0` is this: `chainWordP? p 0 = pureWordP? p` and `peelExactP p 0 = peelExactPureP p`, since the one
  promotion from `+0` returns the pure chain's word (`AccumProofs`).
* **Π's instance**: at `hopperE4m3K32` and `G = 4` every definition is `PeelExact`'s by `rfl` (`chainWordP?_hopper`,
  `peelExactP_hopper`, `peelExactValP_hopper`).

This module imports no proofs, so the named protocol may read it.
-/

namespace Pouw.PearlC.PeelExactP

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.PeelExact Matrix

variable {m n k : Nat}

/-- One window's word on the step `p`: its atoms' `p.step`s from `+0`, each word finite. -/
def windowWordP? (p : Pipeline) (g : List AtomIn) : Option Nat :=
  g.foldlM (fun w x => finite? (p.step w (List.ofFn x.1) (List.ofFn x.2))) 0

/-- **The chain on the step `p`**: the atoms in windows of `G`, each window's word promoted in order into a total from
`+0`. -/
def chainWordP? (p : Pipeline) (G : Nat) (atoms : List AtomIn) : Option Nat :=
  (atoms.toChunks G).foldlM (fun t g => (windowWordP? p g).bind (promoteWord? t)) 0

/-- **U under the pinned order, on the step `p` in windows of `G`**: `PeelExact.peelExact` with `chainWordP? p G` for
`C̃`, `A′·F_B`, `B̃·F_A` and `G = F_Bᵀ·F_A`; the factor rows and the four `k16Step`s unchanged. -/
def peelExactP (p : Pipeline) (G : Nat) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code)
    (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat) :
    Matrix (Fin m) (Fin n) (Option Nat) := fun i j => do
  let c ← chainWordP? p G (rowAtoms (Aq i) (Bq j))
  let af ← (List.ofFn fun q => chainWordP? p G (rowAtoms (Aq i) (FBᵀ q))).mapM id
  let bf ← (List.ofFn fun q => chainWordP? p G (rowAtoms (Bq j) (FAᵀ q))).mapM id
  let g ← (List.ofFn fun q => (List.ofFn fun q' => chainWordP? p G (rowAtoms (FBᵀ q) (FAᵀ q'))).mapM id).mapM id
  let pa ← paRow? (List.ofFn (beA i)) af
  let pb ← pbRow? (List.ofFn (beB j)) bf g
  uWord? c pa pb

/-- `peelExactP`'s entries as values: `none` where a step leaves the finite range. -/
def peelExactValP (p : Pipeline) (G : Nat) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code)
    (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat) :
    Matrix (Fin m) (Fin n) (Option ℚ) := fun i j => (peelExactP p G Aq Bq FA FB beA beB i j).bind wordQ

/-- **The pure chain on words**: `p.step` over all the atoms from `+0`, each word finite, with no promotion. -/
def pureWordP? (p : Pipeline) (atoms : List AtomIn) : Option Nat :=
  atoms.foldlM (fun w x => finite? (p.step w (List.ofFn x.1) (List.ofFn x.2))) 0

/-- **U with pure chains on the step `p`**: `peelExactP`'s order with `pureWordP? p` for its four E4M3 chains. It is
`peelExactP p 0` (`AccumProofs.peelExactP_zero_eq_pure`). -/
def peelExactPureP (p : Pipeline) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code)
    (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat) :
    Matrix (Fin m) (Fin n) (Option Nat) := fun i j => do
  let c ← pureWordP? p (rowAtoms (Aq i) (Bq j))
  let af ← (List.ofFn fun q => pureWordP? p (rowAtoms (Aq i) (FBᵀ q))).mapM id
  let bf ← (List.ofFn fun q => pureWordP? p (rowAtoms (Bq j) (FAᵀ q))).mapM id
  let g ← (List.ofFn fun q => (List.ofFn fun q' => pureWordP? p (rowAtoms (FBᵀ q) (FAᵀ q'))).mapM id).mapM id
  let pa ← paRow? (List.ofFn (beA i)) af
  let pb ← pbRow? (List.ofFn (beB j)) bf g
  uWord? c pa pb

/-- `peelExactPureP`'s entries as values. -/
def peelExactPureValP (p : Pipeline) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code)
    (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat) :
    Matrix (Fin m) (Fin n) (Option ℚ) := fun i j => (peelExactPureP p Aq Bq FA FB beA beB i j).bind wordQ

end Pouw.PearlC.PeelExactP
