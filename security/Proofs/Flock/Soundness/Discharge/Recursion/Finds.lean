module

public import Proofs.Flock.Soundness.Discharge.Composed.Sound
public import Proofs.Flock.Soundness.Discharge.Recursion.Defs
public import Proofs.Flock.Soundness.Discharge.Recursion.RegLink
public import Proofs.Flock.Soundness.Discharge.ZkReg.BindHJ
public import Proofs.Flock.Soundness.Discharge.ZkReg.Checked
import Proofs.Flock.Soundness.Game
public import Proofs.Flock.Soundness.Registered.ReadsX
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar

/-! # What V*'s bridge obligation reads of `Recursive.Finds`, without its recursion imports. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

open Game Model Accounting Extract LinkFinder Audit Audit.Partition ZkSession ZkLink StrictCR ZkReg

open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

variable {D Sl : Type} [Inhabited D] [Inhabited Sl]

section Session

variable (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts)
  (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (cap₀ : Cap D)

/-- **A fork of a `--zk` table's finders in a session** (`SessCRZ`'s): at some rep `r`, two continuations of the prover
from one of the table's committed objects (a level cap, the pads root, the `τ` commitment) that open it to different
rows, pairs or values; two runs after `Commit` that open a level-0 position to different rows; or one run that opens the
table two ways. Each outputs a collision of `H` at an outcome it reaches. -/
def SessForkZ (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : Prop :=
  (∃ r ∈ Finset.range 2, ∃ ℓ ∈ Finset.Icc 1 T.sch.levels.length,
      (advRSFinderZ A Hs T pre post cap₀ r ℓ σ).Finds Hs.H) ∨
    (∃ r ∈ Finset.range 2, (advPadsSFinderZ A Hs T pre post cap₀ r σ).Finds Hs.H) ∨
    (∃ r ∈ Finset.range 2, (advTauSFinderZ A Hs T pre post cap₀ r σ).Finds Hs.H) ∨
    (adv₀SFinderZ A Hs T pre post cap₀ σ).Finds Hs.H ∨
    (selfSFinderZ A Hs T pre post cap₀ σ).Finds Hs.H

end Session

section Batch

variable (A : Arith F K) (Hs : Hashes F D Sl)

/-- **A fork of table `j`'s finders in a compiled `--zk` session** with the prover's session strategy `τ`
(`TableCRZ`'s finders). -/
abbrev TableForkZ (ts : List (TabZK F mPts)) (j : Fin ts.length) (τ : Strategy (sessionZC A Hs ts)) : Prop :=
  SessForkZ A Hs (ts.get j) ((gamesZC A Hs ts τ.1).take j) ((gamesZC A Hs ts τ.1).drop (j + 1)) (τ.1 j)
    (stratZ A Hs ts j τ)

end Batch

section Link

variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
  {L : SamplingDesign n} {decode : DecoderZK (C := C) planZ tab} {Dh : Type}
  {Hc : List UInt8 → Dh} (vb : ValueBindingZ A P planZ tab decode Hc) (wo : vb.Pos → vb.Opening)
  (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R)

/-- **A fork of the registered-value finders**: for some commit string `c` a sample can read, a trial from the prover's
continuation reads at `c` a value other than the registrant's opening of `c`'s registered commitment, and the two
openings are a collision of `Hc` (`regFinderZC`). -/
def RegForkZC (k Rw : ℕ) : Prop :=
  ∃ (c : vb.Pos) (_ : Nonempty (ΩcZ A P planZ tab vb L c)), (regFinderZC A Hs P planZ tab vb wo R τ c k Rw).Finds Hc

/-- **A fork of the link finders**: for some commit string `c` a sample can read, the link finder's trials from the
prover's continuation (`finderG`, `M` further trials) open `c` to two different values, a collision of `Hc`
(`outCZC`). -/
def LinkForkZC (k Rw M : ℕ) : Prop :=
  ∃ (c : vb.Pos) (_ : Nonempty (ΩcZ A P planZ tab vb L c)),
    FindsColl Hc (finderG (trialGZC A Hs P planZ tab vb R τ c k Rw) M)
      (finderStrat _ (trialStratZC A Hs P planZ tab vb R τ c k Rw) M) (outCZC A Hs P planZ tab vb R τ c)

end Link

end FlockSoundness.Discharge.Recursion
