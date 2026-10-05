import FlockSoundness.Protocol.Discharge.ZkLink.LockRep
import FlockSoundness.Protocol.Discharge.Exec.CompiledL

/-!
# Where a bad compiled `--zk` run goes wrong

`CompiledL`'s bad split for the compiled `--zk` table (`TopRelZ`): an accepted run whose openings disagree with the
fixed tables (`AgreeAllZ`) either clashes within the run (`SelfClashZ`), or opens a position of one fixed table to
something else. Each committed kind has its openings by position:

* the level-0 oracle at the witness cap, its rows `width₀` lanes wide (`op₀Z`, `opOfW`), and each level's table at its
  cap (`opRZ`, `Exec.opOfL` on `slotOpensZ`);
* rep `r`'s pads oracle at its root (`opPadsZ`): at each pads position, the pair of the first query whose opening
  verifies there;
* rep `r`'s `τ` oracle at its commitment (`opTauZ`): `τ`, if it opens the commitment.

Every clash, and every pair of runs opening one position of one cap, root or commitment two ways, is a collision of `H`
(`SelfClashZ.collision`, `opOfW_conflict_collision`, `opPadsZ_conflict_collision`, `opTauZ_conflict_collision`).
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type}

/-! ## Level 0's wide rows -/

/-- Two verifying `w`-lane openings of one position to different rows, under any salts. -/
def ClashW (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cap : Cap D)
    (w depth : ℕ) (xs : List (ℕ × List F × Sl × List D)) : Prop :=
  ∃ x ∈ xs, ∃ y ∈ xs, x.1 = y.1 ∧ VerifiedZ H E Lf cap w depth x ∧ VerifiedZ H E Lf cap w depth y ∧ x.2.1 ≠ y.2.1

/-! ## The levels' caps -/

/-- Rep `r`'s level-`ℓ` cap's openings, by position. -/
noncomputable def opRZ (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (r ℓ : ℕ) (cap : Cap D) (out : OutZC F D Sl) :
    Fin (2 ^ (lvOf sch ℓ).logLen) → Option (Fin (2 ^ (lvOf sch ℓ).lanes) → F) :=
  Exec.opOfL H E Lf (lvOf sch ℓ).lanes (pathLen (lvOf sch ℓ)) (lvOf sch ℓ).logLen
    (slotOpensZ out r ℓ (lvOf sch ℓ).queries) cap

/-! ## The pads roots -/

open Classical in
/-- **Rep `r`'s pads openings, by position**: at `p`, the pair of the first query whose opening verifies at `p` from its
leaf under its salt to the root `c 0`. -/
noncomputable def opPadsZ (Hs : Hashes F D Sl) (N q r : ℕ) (c : Cap D) (out : OutZC F D Sl) (p : Fin N) :
    Option (F × F) :=
  (((List.range q).filter fun jj => decide (Merkle.VerifiesL Hs.H Hs.Ep Hs.Lp (fun _ => c 0) (Nat.log 2 N) p.val
      ((out.opens.inner r).pads jj).1 ((out.opens.inner r).pads jj).2.1 ((out.opens.inner r).pads jj).2.2)).head?).map
    fun jj => ((out.opens.inner r).pads jj).1

/-- The pads oracle with the given pairs on `N` positions, zero elsewhere. -/
def padsTab (N : ℕ) (v : Fin N → F × F) : Oracle F :=
  fun lane p => if h : p < N then (if lane = 0 then (v ⟨p, h⟩).1 else (v ⟨p, h⟩).2) else 0

/-- **A pads clash**: within one run, two of a rep's pads openings verify at one position to one root with different
pairs, under any salts. -/
def PadsClashZ (Hs : Hashes F D Sl) (N q : ℕ) (out : OutZC F D Sl) : Prop :=
  ∃ r < 2, ∃ root : D, ∃ jj < q, ∃ jj' < q, ∃ p : Fin N,
    Merkle.VerifiesL Hs.H Hs.Ep Hs.Lp (fun _ => root) (Nat.log 2 N) p.val ((out.opens.inner r).pads jj).1
        ((out.opens.inner r).pads jj).2.1 ((out.opens.inner r).pads jj).2.2 ∧
      Merkle.VerifiesL Hs.H Hs.Ep Hs.Lp (fun _ => root) (Nat.log 2 N) p.val ((out.opens.inner r).pads jj').1
        ((out.opens.inner r).pads jj').2.1 ((out.opens.inner r).pads jj').2.2 ∧
      ((out.opens.inner r).pads jj).1 ≠ ((out.opens.inner r).pads jj').1

/-! ## `τ`'s commitments -/

open Classical in
/-- **Rep `r`'s `τ` opening**: `τ`, if it opens the commitment `c 0` under its salt. -/
noncomputable def opTauZ (Hs : Hashes F D Sl) (r : ℕ) (c : Cap D) (out : OutZC F D Sl) (_ : Unit) : Option F :=
  if Hs.Lτ.leaf (out.opens.inner r).tau.1 (out.opens.inner r).tau.2 = c 0 then some (out.opens.inner r).tau.1
  else none

/-- The `τ` oracle with the given value everywhere. -/
def tauTab (v : Unit → F) : Oracle F := fun _ _ => v ()

/-! ## Self-clashes -/

/-- **A self-clash**: within one run, two verifying openings of a position to different rows (at the witness cap or at
a rep's recorded cap), or a pads clash. -/
def SelfClashZ (Hs : Hashes F D Sl) (T : TabZK F mPts) (out : OutZC F D Sl) : Prop :=
  ClashW Hs.H Hs.E Hs.Lf out.cap₀ (width₀ T) (pathLen (lvOf T.sch 0)) (opens₀Z T.sch out) ∨
    (∃ r < 2, ∃ ℓ, 1 ≤ ℓ ∧ ∃ cap, (if r = 0 then out.pend.1 else out.pend.2).lig.caps[ℓ - 1]? = some cap ∧
      Exec.ClashL Hs.H Hs.E Hs.Lf cap (lvOf T.sch ℓ).lanes (pathLen (lvOf T.sch ℓ))
        (slotOpensZ out r ℓ (lvOf T.sch ℓ).queries)) ∨
    PadsClashZ Hs T.N T.q out

/-! ## Where a bad run goes wrong -/

end FlockSoundness.Discharge.ZkLink
