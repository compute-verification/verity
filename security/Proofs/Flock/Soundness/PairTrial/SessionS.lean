module

public import Proofs.Flock.Soundness.PairTrial.Session
public import Proofs.Flock.Soundness.PairTrial.PickS
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Merkle

/-!
# Trial: one session finder and one self-clash, with the witness searched for

A scratch, not to merge. `PairTrial.Session`'s finder takes the position at which two runs conflict by choice. Here
it searches the positions (`advRSFinderZS`, through `conflictPickS`), and `advRSZ_le_strictS` is `advRSZ_le_strict`
at that finder, with the same statement. `clashLPick` is the other kind of search: a self-clash is two members of
the run's own list of openings, so the list searched is that list's pairs.
-/

@[expose] public section

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting

variable {F : Type} [Field F] {D Sl : Type}

/-- The members of a list taken two at a time, in the list's order. -/
def pairsOf {X : Type} (xs : List X) : List (X × X) := xs.flatMap fun x => xs.map fun y => (x, y)

theorem mem_pairsOf {X : Type} {xs : List X} {x y : X} (hx : x ∈ xs) (hy : y ∈ xs) : (x, y) ∈ pairsOf xs :=
  List.mem_flatMap.2 ⟨x, hx, List.mem_map.2 ⟨y, hy, rfl⟩⟩

/-- `ClashL`'s test at two openings: one position, both verify, different rows. -/
abbrev ClashAtL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cap : Cap D)
    (lanes depth : ℕ) (w : (ℕ × List F × Sl × List D) × (ℕ × List F × Sl × List D)) : Prop :=
  w.1.1 = w.2.1 ∧ VerifiedL H E Lf cap lanes depth w.1 ∧ VerifiedL H E Lf cap lanes depth w.2 ∧ w.1.2.1 ≠ w.2.2.1

/-- **A self-clash's pair, searched for**: the pair of the first two openings of the run's list that clash. -/
noncomputable def clashLPick (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (pairLf : List F → Sl → List F → Sl → List UInt8 × List UInt8) (cap : Cap D) (lanes depth : ℕ)
    (xs : List (ℕ × List F × Sl × List D)) : StrictCR.Pair :=
  StrictCR.pickOf (pairsOf xs) (ClashAtL H E Lf cap lanes depth) fun w =>
    Merkle.openPairL H E Lf.leaf pairLf w.1.1 w.1.2.1 w.2.2.1 w.1.2.2.1 w.2.2.2.1 w.1.2.2.2 w.2.2.2.2

/-- **A clash's searched pair is a collision**: `ClashL.collision` with its pair found in the run's own list. -/
theorem clashLPick_isColl {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {Lf : Merkle.Leaf (List F) Sl D H}
    {pairLf : List F → Sl → List F → Sl → List UInt8 × List UInt8}
    (hpair : ∀ c c' s s', c ≠ c' → Lf.leaf c s = Lf.leaf c' s' →
      Hash.Collision H (pairLf c s c' s').1 (pairLf c s c' s').2)
    {cap : Cap D} {lanes depth : ℕ} {xs : List (ℕ × List F × Sl × List D)} (h : ClashL H E Lf cap lanes depth xs) :
    StrictCR.IsColl H (clashLPick H E Lf pairLf cap lanes depth xs) := by
  obtain ⟨x, hx, y, hy, hxy, hvx, hvy, hne⟩ := h
  unfold clashLPick
  refine StrictCR.pickOf_isColl H (fun w hw => ?_) ⟨(x, y), mem_pairsOf hx hy, hxy, hvx, hvy, hne⟩
  obtain ⟨h1, h2, h3, h4⟩ := hw
  refine Merkle.openPairL_spec H E Lf pairLf hpair cap depth w.1.1 w.1.2.1 w.2.2.1 w.1.2.2.1 w.2.2.2.1 w.1.2.2.2
    w.2.2.2.2 h4 h2.2 ?_
  rw [h1]
  exact h3.2

end FlockSoundness.Discharge.Exec

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Audit ZkSession StrictCR
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]

section Session

variable (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts)
  (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (cap₀ : Cap D)
  (pairLf : List F → Sl → List F → Sl → List UInt8 × List UInt8)

/-- **`Adv_{r,ℓ}`'s finder in the session**, with the position searched for: the pair of the two continuations'
openings at the first position of the level where they conflict. -/
noncomputable def advRSFinderZS (r ℓ : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : Finder Pair :=
  forkFinder (plurSZ Hs T pre.length) (slotOf r ℓ + 4)
    (fun d => conflictPickS
      (fun outs : List (OutZC F D Sl) => opRZ Hs.H Hs.E Hs.Lf T.sch r ℓ d (outs.getD pre.length dfltOutZC))
      fun x p => opRZPair Hs.H Hs.E Hs.Lf pairLf T.sch r ℓ d
        (x.1.getD pre.length dfltOutZC, x.2.getD pre.length dfltOutZC) p)
    none (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)

/-- `advRSZ_le_strict` at the finder that searches: the same statement. -/
theorem advRSZ_le_strictS
    (hpair : ∀ c c' s s', c ≠ c' → Hs.Lf.leaf c s = Hs.Lf.leaf c' s' →
      Hash.Collision Hs.H (pairLf c s c' s').1 (pairLf c s c' s').2)
    (r ℓ : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) {q : ℝ}
    (h : (advRSFinderZS A Hs T pre post cap₀ pairLf r ℓ σ).CR Hs.H q) :
    advRSZ A Hs T pre post cap₀ r ℓ σ ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  ((forkE_capAcc_le Hs T (plurSZ Hs T pre.length) (slotOf r ℓ + 4) pre.length r ℓ
    (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)).trans
  (forkE_conflict_leS Hs.H (plurSZ Hs T pre.length) (slotOf r ℓ + 4)
    (fun d (outs : List (OutZC F D Sl)) => opRZ Hs.H Hs.E Hs.Lf T.sch r ℓ d (outs.getD pre.length dfltOutZC))
    (fun d x p => opRZPair Hs.H Hs.E Hs.Lf pairLf T.sch r ℓ d
      (x.1.getD pre.length dfltOutZC, x.2.getD pre.length dfltOutZC) p)
    (fun _ x _ h => opRZ_conflict_pair hpair
      (x := (x.1.getD pre.length dfltOutZC, x.2.getD pre.length dfltOutZC)) h)
    (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ))).trans h.le

end Session

end FlockSoundness.Discharge.ZkLink
