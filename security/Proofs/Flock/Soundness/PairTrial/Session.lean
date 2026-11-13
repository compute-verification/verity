module

public import Proofs.Flock.Soundness.Discharge.ZkLink.StrictZ
public import Proofs.Flock.Soundness.PairTrial.Pick
public import Proofs.Flock.Soundness.MerklePair
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Merkle

/-!
# Trial: one session finder, end to end, over the pair functions

A scratch, not to merge. `advRSFinderZ` (`StrictZ`) outputs, at a rep's level cap, a pair that `conflictPick` chooses
from the proposition `Merkle.Collision H`. Here the same finder outputs the pair of the two runs' first verifying
openings at the position (`opRZPair`, through `Exec.opOfLPair` and `Merkle.openPairL`), and `advRSZ_le_strictP` is
`advRSZ_le_strict` at that finder, with the same statement.

The leaf's pair and its spec are the parameters `pairLf` and `hpair`: they stand for the fields `Hs.Lf.pair` and
`Hs.Lf.pair_spec` that `Merkle.Leaf` gets when it is restated (`PairTrial.Leaf`).
-/

@[expose] public section

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting

variable {F : Type} [Field F] {D Sl : Type}

open Classical in
/-- A run's first verifying opening at `p`: the opening whose row `opOfL` returns. -/
noncomputable def firstL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (lanes depth d : ℕ) (xs : List (ℕ × List F × Sl × List D)) (cap : Cap D) (p : Fin (2 ^ d)) :
    Option (ℕ × List F × Sl × List D) :=
  (xs.filter fun x => decide (x.1 = p.val ∧ VerifiedL H E Lf cap lanes depth x)).head?

/-- **The pair of two runs at a position**: the pair of their first verifying openings there (`Merkle.openPairL`),
and the empty pair when either run has none. -/
noncomputable def opOfLPair (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (pairLf : List F → Sl → List F → Sl → List UInt8 × List UInt8) (lanes depth d : ℕ)
    (xs xs' : List (ℕ × List F × Sl × List D)) (cap : Cap D) (p : Fin (2 ^ d)) : List UInt8 × List UInt8 :=
  match firstL H E Lf lanes depth d xs cap p, firstL H E Lf lanes depth d xs' cap p with
  | some x, some y => Merkle.openPairL H E Lf.leaf pairLf p.val x.2.1 y.2.1 x.2.2.1 y.2.2.1 x.2.2.2 y.2.2.2
  | _, _ => ([], [])

theorem firstL_of_isSome {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {Lf : Merkle.Leaf (List F) Sl D H}
    {lanes depth d : ℕ} {xs : List (ℕ × List F × Sl × List D)} {cap : Cap D} {p : Fin (2 ^ d)}
    (h : (opOfL H E Lf lanes depth d xs cap p).isSome) : ∃ x, firstL H E Lf lanes depth d xs cap p = some x := by
  unfold opOfL at h
  rw [Option.isSome_map] at h
  exact Option.isSome_iff_exists.1 h

theorem firstL_some {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {Lf : Merkle.Leaf (List F) Sl D H}
    {lanes depth d : ℕ} {xs : List (ℕ × List F × Sl × List D)} {cap : Cap D} {p : Fin (2 ^ d)}
    {x : ℕ × List F × Sl × List D} (h : firstL H E Lf lanes depth d xs cap p = some x) :
    x.1 = p.val ∧ VerifiedL H E Lf cap lanes depth x := by
  unfold firstL at h
  have hmem := List.mem_of_mem_head? h
  rw [List.mem_filter, decide_eq_true_eq] at hmem
  exact hmem.2

/-- **The pair of two runs that open a position to different rows under one cap is a collision**:
`opOfL_conflict_collision` with its pair named. -/
theorem opOfL_conflict_pair {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {Lf : Merkle.Leaf (List F) Sl D H}
    {pairLf : List F → Sl → List F → Sl → List UInt8 × List UInt8}
    (hpair : ∀ c c' s s', c ≠ c' → Lf.leaf c s = Lf.leaf c' s' →
      Hash.Collision H (pairLf c s c' s').1 (pairLf c s c' s').2)
    {lanes depth d : ℕ} {cap : Cap D} {xs xs' : List (ℕ × List F × Sl × List D)} {p : Fin (2 ^ d)}
    (h1 : (opOfL H E Lf lanes depth d xs cap p).isSome) (h2 : (opOfL H E Lf lanes depth d xs' cap p).isSome)
    (hne : opOfL H E Lf lanes depth d xs cap p ≠ opOfL H E Lf lanes depth d xs' cap p) :
    Hash.Collision H (opOfLPair H E Lf pairLf lanes depth d xs xs' cap p).1
      (opOfLPair H E Lf pairLf lanes depth d xs xs' cap p).2 := by
  obtain ⟨x, hx⟩ := firstL_of_isSome h1
  obtain ⟨y, hy⟩ := firstL_of_isSome h2
  obtain ⟨hxp, hxv⟩ := firstL_some hx
  obtain ⟨hyp, hyv⟩ := firstL_some hy
  have hox : opOfL H E Lf lanes depth d xs cap p = some (rowFin lanes x.2.1) :=
    congrArg (Option.map fun x : ℕ × List F × Sl × List D => rowFin lanes x.2.1) hx
  have hoy : opOfL H E Lf lanes depth d xs' cap p = some (rowFin lanes y.2.1) :=
    congrArg (Option.map fun x : ℕ × List F × Sl × List D => rowFin lanes x.2.1) hy
  have e : opOfLPair H E Lf pairLf lanes depth d xs xs' cap p =
      Merkle.openPairL H E Lf.leaf pairLf p.val x.2.1 y.2.1 x.2.2.1 y.2.2.1 x.2.2.2 y.2.2.2 := by
    unfold opOfLPair
    rw [hx, hy] <;> rfl
  rw [e]
  refine Merkle.openPairL_spec H E Lf pairLf hpair cap depth p.val x.2.1 y.2.1 x.2.2.1 y.2.2.1 x.2.2.2 y.2.2.2
    (fun hxy => hne ?_) (hxp ▸ hxv.2) (hyp ▸ hyv.2)
  rw [hox, hoy, hxy]

end FlockSoundness.Discharge.Exec

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Audit ZkSession StrictCR
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]

/-- **The pair of two runs at a position of a rep's level-`ℓ` cap.** -/
noncomputable def opRZPair (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (pairLf : List F → Sl → List F → Sl → List UInt8 × List UInt8) (sch : Schedule) (r ℓ : ℕ) (cap : Cap D)
    (x : OutZC F D Sl × OutZC F D Sl) (p : Fin (2 ^ (lvOf sch ℓ).logLen)) : List UInt8 × List UInt8 :=
  Exec.opOfLPair H E Lf pairLf (lvOf sch ℓ).lanes (pathLen (lvOf sch ℓ)) (lvOf sch ℓ).logLen
    (slotOpensZ x.1 r ℓ (lvOf sch ℓ).queries) (slotOpensZ x.2 r ℓ (lvOf sch ℓ).queries) cap p

/-- **Every pair a rep's collision finder outputs is a collision**: `opRZ_conflict_collision` with its pair named. -/
theorem opRZ_conflict_pair {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {Lf : Merkle.Leaf (List F) Sl D H}
    {pairLf : List F → Sl → List F → Sl → List UInt8 × List UInt8}
    (hpair : ∀ c c' s s', c ≠ c' → Lf.leaf c s = Lf.leaf c' s' →
      Hash.Collision H (pairLf c s c' s').1 (pairLf c s c' s').2)
    {sch : Schedule} {r ℓ : ℕ} {cap : Cap D} {x : OutZC F D Sl × OutZC F D Sl}
    {p : Fin (2 ^ (lvOf sch ℓ).logLen)} (h : Rewinding.ConflictAt (opRZ H E Lf sch r ℓ cap) x p) :
    Hash.Collision H (opRZPair H E Lf pairLf sch r ℓ cap x p).1 (opRZPair H E Lf pairLf sch r ℓ cap x p).2 :=
  Exec.opOfL_conflict_pair hpair h.1 h.2.1 h.2.2

section Session

variable (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts)
  (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (cap₀ : Cap D)
  (pairLf : List F → Sl → List F → Sl → List UInt8 × List UInt8)

/-- **`Adv_{r,ℓ}`'s finder in the session**, over the pair function: `advRSFinderZ` with the pair of the two
continuations' openings as its output. -/
noncomputable def advRSFinderZP (r ℓ : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : Finder Pair :=
  forkFinder (plurSZ Hs T pre.length) (slotOf r ℓ + 4)
    (fun d => conflictPickP
      (fun outs : List (OutZC F D Sl) => opRZ Hs.H Hs.E Hs.Lf T.sch r ℓ d (outs.getD pre.length dfltOutZC))
      fun x p => opRZPair Hs.H Hs.E Hs.Lf pairLf T.sch r ℓ d
        (x.1.getD pre.length dfltOutZC, x.2.getD pre.length dfltOutZC) p)
    none (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)

/-- `advRSZ_le_strict` at the finder over the pair function: the same statement. -/
theorem advRSZ_le_strictP
    (hpair : ∀ c c' s s', c ≠ c' → Hs.Lf.leaf c s = Hs.Lf.leaf c' s' →
      Hash.Collision Hs.H (pairLf c s c' s').1 (pairLf c s c' s').2)
    (r ℓ : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) {q : ℝ}
    (h : (advRSFinderZP A Hs T pre post cap₀ pairLf r ℓ σ).CR Hs.H q) :
    advRSZ A Hs T pre post cap₀ r ℓ σ ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  ((forkE_capAcc_le Hs T (plurSZ Hs T pre.length) (slotOf r ℓ + 4) pre.length r ℓ
    (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)).trans
  (forkE_conflict_leP Hs.H (plurSZ Hs T pre.length) (slotOf r ℓ + 4)
    (fun d (outs : List (OutZC F D Sl)) => opRZ Hs.H Hs.E Hs.Lf T.sch r ℓ d (outs.getD pre.length dfltOutZC))
    (fun d x p => opRZPair Hs.H Hs.E Hs.Lf pairLf T.sch r ℓ d
      (x.1.getD pre.length dfltOutZC, x.2.getD pre.length dfltOutZC) p)
    (fun _ x _ h => opRZ_conflict_pair hpair
      (x := (x.1.getD pre.length dfltOutZC, x.2.getD pre.length dfltOutZC)) h)
    (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ))).trans h.le

end Session

end FlockSoundness.Discharge.ZkLink
