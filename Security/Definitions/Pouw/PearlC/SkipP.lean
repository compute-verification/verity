import Definitions.Pouw.PearlC.Accum
import Definitions.Pouw.PearlC.Skip

/-!
# Pearl-C's skip debit over any accumulator model (definitions only)

Twins of `Skip`'s debit (A), with the atom a parameter: `p : Pipeline` takes the place of `hopperE4m3K32`, and the
words' promotion depth `G` that of `Params.pi.G = 4`. The chain is `Accum`'s (`Accum.groupWordP`, `Accum.totalP`,
`Accum.chainP`). At `hopperE4m3K32` and `G = 4` every twin is today's definition (`Pouw.PearlC.FreezeP`'s `rfl`
lemmas: `stepP_hopper`, `jointFlagsP_hopper`, `wordFlagsP_hopper`, …); at `sm120E4m3K32` they are the sm_120 chain's.
Masks, the masked fold and the unit's debit (`Skip.foldMask`, `Skip.allMasks`, `Skip.chainDebit`, `Skip.needed`) do
not read the atom and are reused.

Without promotion (`G = 0`, `pearl-c-sm120-v2`) the debit is the pure chain's lone skips (`pureFlagsP`, one replay of
`Accum.pureChain` per atom), and a device record's flags are `flagsAt p G` (`unitFlagsAt p G`): `pureFlagsP` at
`G = 0`, `wordFlagsP p G` otherwise; `flagsAt hopperE4m3K32 4` is `Skip.wordFlags` by `rfl`
(`FreezePPure.flagsAt_hopper`).
-/

namespace Pouw.PearlC.SkipP

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.Skip Pouw.PearlC.Accum

variable (p : Pipeline)

/-- One atom of `p`: the step `Accum.groupWordP` folds. -/
def stepP (w : Nat) (x : AtomIn) : Nat := p.step w (List.ofFn x.1) (List.ofFn x.2)

/-- A group's word promoted into the total. -/
def promoteGroupP (t : Option ℚ) (g : List AtomIn) : Option ℚ := promote t (groupWordP p g)

/-- A group's word from `+0` with the atoms `m` flags skipped in place. -/
def wordInP (g : List AtomIn) (m : List Bool) : Nat := foldMask (stepP p) 0 g m

/-- The promoted total before group `i`. -/
def inTotalP (gs : List (List AtomIn)) (i : Nat) : Option ℚ := (gs.take i).foldl (promoteGroupP p) (some 0)

/-- The words of the groups after group `i`. -/
def laterWordsP (gs : List (List AtomIn)) (i : Nat) : List Nat := (gs.drop (i + 1)).map (groupWordP p)

/-- Whether skipping the atoms `m` flags in group `g` together leaves the output unchanged. -/
def keepsP (g : List AtomIn) (t : Option ℚ) (ws : List Nat) (m : List Bool) : Bool :=
  decide (ws.foldl promote (promote t (wordInP p g m)) = ws.foldl promote (promote t (groupWordP p g)))

/-- **The joint flags** of a group under `p`: atom `j` is flagged when some set of the group's atoms containing it,
skipped together in place, leaves the output unchanged. -/
def jointAtP (g : List AtomIn) (t : Option ℚ) (ws : List Nat) : List Bool :=
  (List.range g.length).map fun j => (allMasks g.length).any fun m => m.getD j false && keepsP p g t ws m

/-- The joint flags of group `i`. -/
def jointFlagsP (gs : List (List AtomIn)) (i : Nat) : List Bool :=
  jointAtP p (gs.getD i []) (inTotalP p gs i) (laterWordsP p gs i)

/-- **The word's flags**, debit (A), under `p` and promotion every `G` atoms: atom `t` of the word over rows `a` and `b`
is in `jointFlagsP` of its group `t / G`, at position `t % G`. -/
def wordFlagsP (G : Nat) (k : ℕ) (a b : Fin k → Code) (t : ℕ) : Bool :=
  (jointFlagsP p ((rowAtoms a b).toChunks G) (t / G)).getD (t % G) false

variable {m n k : Nat} in
/-- **The unit's flags** under `p` and `G`. -/
def unitFlagsP (G : Nat) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) :
    Fin m → Fin n → ℕ → Bool :=
  fun i j t => wordFlagsP p G k (Aq i) (Bq j) t

/-! ## No promotion (`G = 0`): the lone skips of the pure chain -/

/-- The pure chain (`Accum.pureChain p`) over `atoms` with atom `t` skipped alone: one replay of the other atoms. -/
def pureSkipP (atoms : List AtomIn) (t : Nat) : Option ℚ := pureChain p (atoms.eraseIdx t)

/-- **The word's flags without promotion**: atom `t` of the word over rows `a` and `b` is flagged when skipping it
alone leaves the pure chain's output unchanged (`FreezePPure.pureFlags_iff`). Joint skips of several atoms are not
flagged. -/
def pureFlagsP (k : ℕ) (a b : Fin k → Code) (t : ℕ) : Bool :=
  decide (t < k / 32) && decide (pureSkipP p (rowAtoms a b) t = pureChain p (rowAtoms a b))

/-- **The word's flags at a device's promotion period** `G` (`PearlCDevice.G`): `pureFlagsP` when `G = 0` (no
promotion), else `wordFlagsP p G`. -/
def flagsAt (G : Nat) : (k : ℕ) → (Fin k → Code) → (Fin k → Code) → ℕ → Bool :=
  if G = 0 then pureFlagsP p else wordFlagsP p G

variable {m n k : Nat} in
/-- **The unit's flags at a device record** (`PearlCDevice.flags`), under `p` and `G`. -/
def unitFlagsAt (G : Nat) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) :
    Fin m → Fin n → ℕ → Bool :=
  fun i j t => flagsAt p G k (Aq i) (Bq j) t

end Pouw.PearlC.SkipP
