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

/-- A row's first `w` lanes. -/
def rowW (w : ℕ) (row : List F) : Fin w → F := fun lane => row.getD lane.val 0

open Classical in
/-- The `w`-lane row a run opens at `p`: its first verifying opening there. -/
noncomputable def opOfW (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (w depth d : ℕ) (xs : List (ℕ × List F × Sl × List D)) (cap : Cap D) (p : Fin (2 ^ d)) : Option (Fin w → F) :=
  ((xs.filter fun x => decide (x.1 = p.val ∧ VerifiedZ H E Lf cap w depth x)).head?).map fun x => rowW w x.2.1

/-- Two verifying `w`-lane openings of one position to different rows, under any salts. -/
def ClashW (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cap : Cap D)
    (w depth : ℕ) (xs : List (ℕ × List F × Sl × List D)) : Prop :=
  ∃ x ∈ xs, ∃ y ∈ xs, x.1 = y.1 ∧ VerifiedZ H E Lf cap w depth x ∧ VerifiedZ H E Lf cap w depth y ∧ x.2.1 ≠ y.2.1

omit [Fintype F] [DecidableEq F] in
open Classical in
theorem opOfW_some {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {Lf : Merkle.Leaf (List F) Sl D H}
    {w depth d : ℕ} {xs : List (ℕ × List F × Sl × List D)} {cap : Cap D} {p : Fin (2 ^ d)}
    (h : (opOfW H E Lf w depth d xs cap p).isSome) :
    ∃ x ∈ xs, x.1 = p.val ∧ VerifiedZ H E Lf cap w depth x ∧
      opOfW H E Lf w depth d xs cap p = some (rowW w x.2.1) := by
  unfold opOfW at h ⊢
  rw [Option.isSome_map] at h
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 h
  have hmem := List.mem_of_mem_head? hx
  rw [List.mem_filter, decide_eq_true_eq] at hmem
  exact ⟨x, hmem.1, hmem.2.1, hmem.2.2, by rw [hx]; rfl⟩

omit [Field F] [Fintype F] [DecidableEq F] in
/-- **A clash is a collision**, under any salts (`Merkle.opening_bindingL`). -/
theorem ClashW.collision {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {Lf : Merkle.Leaf (List F) Sl D H}
    {cap : Cap D} {w depth : ℕ} {xs : List (ℕ × List F × Sl × List D)} (h : ClashW H E Lf cap w depth xs) :
    Merkle.Collision H := by
  obtain ⟨x, -, y, -, hxy, hx, hy, hne⟩ := h
  exact Merkle.opening_bindingL H E Lf cap depth x.1 x.2.1 y.2.1 x.2.2.1 y.2.2.1 x.2.2.2 y.2.2.2 hne hx.2
    (hxy ▸ hy.2)

omit [Fintype F] [DecidableEq F] in
/-- Two runs that open a position to different `w`-lane rows under one cap give a collision. -/
theorem opOfW_conflict_collision {H : List UInt8 → D} {E : Merkle.Enc (List F) D}
    {Lf : Merkle.Leaf (List F) Sl D H} {w depth d : ℕ} {cap : Cap D} {xs xs' : List (ℕ × List F × Sl × List D)}
    {p : Fin (2 ^ d)} (h1 : (opOfW H E Lf w depth d xs cap p).isSome)
    (h2 : (opOfW H E Lf w depth d xs' cap p).isSome)
    (hne : opOfW H E Lf w depth d xs cap p ≠ opOfW H E Lf w depth d xs' cap p) :
    Merkle.Collision H := by
  obtain ⟨x, -, hxp, hx, hox⟩ := opOfW_some h1
  obtain ⟨y, -, hyp, hy, hoy⟩ := opOfW_some h2
  refine Merkle.opening_bindingL H E Lf cap depth p.val x.2.1 y.2.1 x.2.2.1 y.2.2.1 x.2.2.2 y.2.2.2
    (fun hxy => hne ?_) (hxp ▸ hx.2) (hyp ▸ hy.2)
  rw [hox, hoy, hxy]

/-- **Every pair the witness cap's collision finder counts is a collision.** -/
theorem op₀Z_conflict_collision {H : List UInt8 → D} {E : Merkle.Enc (List F) D}
    {Lf : Merkle.Leaf (List F) Sl D H} {T : TabZK F mPts} {cap : Cap D} {x : OutZC F D Sl × OutZC F D Sl}
    {p : Fin (2 ^ (lvOf T.sch 0).logLen)} (h : Rewinding.ConflictAt (op₀Z H E Lf T cap) x p) :
    Merkle.Collision H :=
  opOfW_conflict_collision h.1 h.2.1 h.2.2

/-! ## The levels' caps -/

/-- Rep `r`'s level-`ℓ` cap's openings, by position. -/
noncomputable def opRZ (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (r ℓ : ℕ) (cap : Cap D) (out : OutZC F D Sl) :
    Fin (2 ^ (lvOf sch ℓ).logLen) → Option (Fin (2 ^ (lvOf sch ℓ).lanes) → F) :=
  Exec.opOfL H E Lf (lvOf sch ℓ).lanes (pathLen (lvOf sch ℓ)) (lvOf sch ℓ).logLen
    (slotOpensZ out r ℓ (lvOf sch ℓ).queries) cap

omit [Fintype F] [DecidableEq F] in
/-- **Every pair a rep's collision finder counts is a collision.** -/
theorem opRZ_conflict_collision {H : List UInt8 → D} {E : Merkle.Enc (List F) D}
    {Lf : Merkle.Leaf (List F) Sl D H} {sch : Schedule} {r ℓ : ℕ} {cap : Cap D} {x : OutZC F D Sl × OutZC F D Sl}
    {p : Fin (2 ^ (lvOf sch ℓ).logLen)} (h : Rewinding.ConflictAt (opRZ H E Lf sch r ℓ cap) x p) :
    Merkle.Collision H :=
  Exec.opOfL_conflict_collision h.1 h.2.1 h.2.2

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

omit [Field F] [Fintype F] [DecidableEq F] in
theorem PadsClashZ.collision {Hs : Hashes F D Sl} {N q : ℕ} {out : OutZC F D Sl} (h : PadsClashZ Hs N q out) :
    Merkle.Collision Hs.H := by
  obtain ⟨_, _, root, jj, _, jj', _, p, h1, h2, hne⟩ := h
  exact Merkle.opening_bindingL Hs.H Hs.Ep Hs.Lp (fun _ => root) (Nat.log 2 N) p.val _ _ _ _ _ _ hne h1 h2

omit [Field F] [Fintype F] [DecidableEq F] in
open Classical in
theorem opPadsZ_some {Hs : Hashes F D Sl} {N q r : ℕ} {c : Cap D} {out : OutZC F D Sl} {p : Fin N}
    (h : (opPadsZ Hs N q r c out p).isSome) :
    ∃ jj < q, Merkle.VerifiesL Hs.H Hs.Ep Hs.Lp (fun _ => c 0) (Nat.log 2 N) p.val ((out.opens.inner r).pads jj).1
        ((out.opens.inner r).pads jj).2.1 ((out.opens.inner r).pads jj).2.2 ∧
      opPadsZ Hs N q r c out p = some ((out.opens.inner r).pads jj).1 := by
  unfold opPadsZ at h ⊢
  rw [Option.isSome_map] at h
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 h
  have hmem := List.mem_of_mem_head? hx
  rw [List.mem_filter, decide_eq_true_eq, List.mem_range] at hmem
  exact ⟨x, hmem.1, hmem.2, by rw [hx]; rfl⟩

omit [Field F] [Fintype F] [DecidableEq F] in
/-- **Every pair a pads root's collision finder counts is a collision.** -/
theorem opPadsZ_conflict_collision {Hs : Hashes F D Sl} {N q r : ℕ} {c : Cap D}
    {x : OutZC F D Sl × OutZC F D Sl} {p : Fin N} (h : Rewinding.ConflictAt (opPadsZ Hs N q r c) x p) :
    Merkle.Collision Hs.H := by
  obtain ⟨jj, -, h1, e1⟩ := opPadsZ_some h.1
  obtain ⟨jj', -, h2, e2⟩ := opPadsZ_some h.2.1
  refine Merkle.opening_bindingL Hs.H Hs.Ep Hs.Lp (fun _ => c 0) (Nat.log 2 N) p.val _ _ _ _ _ _
    (fun e => h.2.2 ?_) h1 h2
  rw [e1, e2, e]

/-! ## `τ`'s commitments -/

open Classical in
/-- **Rep `r`'s `τ` opening**: `τ`, if it opens the commitment `c 0` under its salt. -/
noncomputable def opTauZ (Hs : Hashes F D Sl) (r : ℕ) (c : Cap D) (out : OutZC F D Sl) (_ : Unit) : Option F :=
  if Hs.Lτ.leaf (out.opens.inner r).tau.1 (out.opens.inner r).tau.2 = c 0 then some (out.opens.inner r).tau.1
  else none

/-- The `τ` oracle with the given value everywhere. -/
def tauTab (v : Unit → F) : Oracle F := fun _ _ => v ()

omit [Field F] [Fintype F] [DecidableEq F] in
/-- **Every pair a `τ` commitment's collision finder counts is a collision** (`Leaf.bind`). -/
theorem opTauZ_conflict_collision {Hs : Hashes F D Sl} {r : ℕ} {c : Cap D} {x : OutZC F D Sl × OutZC F D Sl}
    {p : Unit} (h : Rewinding.ConflictAt (opTauZ Hs r c) x p) : Merkle.Collision Hs.H := by
  classical
  obtain ⟨h1, h2, hne⟩ := h
  by_cases ha : Hs.Lτ.leaf (x.1.opens.inner r).tau.1 (x.1.opens.inner r).tau.2 = c 0
  · by_cases hb : Hs.Lτ.leaf (x.2.opens.inner r).tau.1 (x.2.opens.inner r).tau.2 = c 0
    · simp only [opTauZ, ha, hb, ↓reduceIte, ne_eq, Option.some.injEq] at hne
      exact Hs.Lτ.bind _ _ _ _ hne (ha.trans hb.symm)
    · simp [opTauZ, hb] at h2
  · simp [opTauZ, ha] at h1

/-! ## Self-clashes -/

/-- **A self-clash**: within one run, two verifying openings of a position to different rows (at the witness cap or at
a rep's recorded cap), or a pads clash. -/
def SelfClashZ (Hs : Hashes F D Sl) (T : TabZK F mPts) (out : OutZC F D Sl) : Prop :=
  ClashW Hs.H Hs.E Hs.Lf out.cap₀ (width₀ T) (pathLen (lvOf T.sch 0)) (opens₀Z T.sch out) ∨
    (∃ r < 2, ∃ ℓ, 1 ≤ ℓ ∧ ∃ cap, (if r = 0 then out.pend.1 else out.pend.2).lig.caps[ℓ - 1]? = some cap ∧
      Exec.ClashL Hs.H Hs.E Hs.Lf cap (lvOf T.sch ℓ).lanes (pathLen (lvOf T.sch ℓ))
        (slotOpensZ out r ℓ (lvOf T.sch ℓ).queries)) ∨
    PadsClashZ Hs T.N T.q out

/-- **A self-clash is a collision.** -/
theorem SelfClashZ.collision {Hs : Hashes F D Sl} {T : TabZK F mPts} {out : OutZC F D Sl}
    (h : SelfClashZ Hs T out) : Merkle.Collision Hs.H := by
  rcases h with h | ⟨_, _, _, _, _, _, h⟩ | h
  · exact h.collision
  · exact h.collision
  · exact h.collision

/-! ## Where a bad run goes wrong -/

end FlockSoundness.Discharge.ZkLink
