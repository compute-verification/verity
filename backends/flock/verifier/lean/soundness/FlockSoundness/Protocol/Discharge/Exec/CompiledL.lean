import FlockSoundness.Protocol.CompiledSound

/-!
# Compiled soundness of one table under a leaf scheme (README.md §1.2, PROTOCOL.md §14)

`table_sound_compiled` (`CompiledSound.lean`) bounds `tableC`, whose openings verify from the unsalted leaf
`H(encode row)`. This file proves the same bound for `Model.tableCL A H E Lf`, whose openings carry salts and verify
from the row's leaf under its salt (`OpensOKL`), for any leaf scheme `Lf` (`Merkle.Leaf`). At `Refine.leaf512` that
is the `hm96-sha512/v1` table flock-verify checks (`accepts_tables_hm96`).

The argument is `CompiledSound`'s, and only the openings change. A salted opening is `(position, row, salt, path)`. A
clash is two verifying openings of one position to different rows, under any salts, and it is a collision of `H` by
`Merkle.opening_bindingL`: the leaf scheme's binding (`Leaf.bind`) does what the encoding's injectivity does for the
unsalted leaf. The lockstep with the oracle table (`lockTableL`) is `lockTable` with the salts in the openings'
message; the reps' locksteps (`lockRep`) are unchanged.

* `table_sound_compiledL`: `Pr[accepted ∧ T*(P*) uncommitted] ≤ tableError + √(N₀·Q₀·Adv₀) + Σ √(N_ℓ·Q_ℓ·Adv_{r,ℓ})
  + Pr[self-clash]`, with each `Adv` and the self-clash a collision of `H` (`op₀L_conflict_collision`,
  `opRL_conflict_collision`, `SelfClashL.collision`).
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D Sl : Type}

/-! ## The lockstep -/

/-- **The salted table's leaf relation** below the level-0 cap. -/
def TopRelL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (sch : Schedule)
    (C₀ : Oracle F) (cap₀ : Cap D) (i : IRep F D × IRep F D) (a : Bool × Oracle F) (out : OutCL F D Sl) : Prop :=
  a.2 = C₀ ∧ out.cap₀ = cap₀ ∧
    (out.accepted = true → AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens →
      AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens → a.1 = true) ∧
    (out.accepted = true →
      OpensOKL H E Lf cap₀ 0 out.pend.1 out.opens out.salts ∧ OpensOKL H E Lf cap₀ 1 out.pend.2 out.opens out.salts) ∧
    RecStruct sch i.1 out.pend.1 ∧ RecStruct sch i.2 out.pend.2

/-! ## Salted openings -/

/-- The salted openings of rep `r`'s first level-`ℓ` request, `(position, row, salt, path)` for its first `Q`
queries. -/
def slotOpensL (out : OutCL F D Sl) (r ℓ Q : ℕ) : List (ℕ × List F × Sl × List D) :=
  match ((if r = 0 then out.pend.1 else out.pend.2).reqs.find? fun req => req.level == ℓ) with
  | none => []
  | some req => ((req.positions.take Q).zipIdx).map fun pj =>
      (pj.1, (out.opens r ℓ pj.2).1, out.salts r ℓ pj.2, (out.opens r ℓ pj.2).2)

/-- A salted opening verifies against `cap`: a row of `2^lanes` elements whose path verifies from its leaf under its
salt. -/
def VerifiedL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cap : Cap D)
    (lanes depth : ℕ) (x : ℕ × List F × Sl × List D) : Prop :=
  x.2.1.length = 2 ^ lanes ∧ Merkle.VerifiesL H E Lf cap depth x.1 x.2.1 x.2.2.1 x.2.2.2

open Classical in
/-- The row a run opens at `p`: its first verifying opening there. -/
noncomputable def opOfL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (lanes depth d : ℕ) (xs : List (ℕ × List F × Sl × List D)) (cap : Cap D) (p : Fin (2 ^ d)) :
    Option (Fin (2 ^ lanes) → F) :=
  ((xs.filter fun x => decide (x.1 = p.val ∧ VerifiedL H E Lf cap lanes depth x)).head?).map
    fun x => rowFin lanes x.2.1

/-- Two verifying openings of one position to different rows, under any salts. -/
def ClashL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cap : Cap D)
    (lanes depth : ℕ) (xs : List (ℕ × List F × Sl × List D)) : Prop :=
  ∃ x ∈ xs, ∃ y ∈ xs, x.1 = y.1 ∧ VerifiedL H E Lf cap lanes depth x ∧
    VerifiedL H E Lf cap lanes depth y ∧ x.2.1 ≠ y.2.1

/-! ## Per-cap openings -/

/-- Both reps' level-0 salted openings. -/
def opens₀L (sch : Schedule) (out : OutCL F D Sl) : List (ℕ × List F × Sl × List D) :=
  slotOpensL out 0 0 (lvOf sch 0).queries ++ slotOpensL out 1 0 (lvOf sch 0).queries

/-- The level-0 cap's openings, by position. -/
noncomputable def op₀L (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (cap : Cap D) (out : OutCL F D Sl) :
    Fin (2 ^ (lvOf sch 0).logLen) → Option (Fin (2 ^ (lvOf sch 0).lanes) → F) :=
  opOfL H E Lf (lvOf sch 0).lanes (pathLen (lvOf sch 0)) (lvOf sch 0).logLen (opens₀L sch out) cap

/-- Rep `r`'s level-`ℓ` cap's openings, by position. -/
noncomputable def opRL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (r ℓ : ℕ) (cap : Cap D) (out : OutCL F D Sl) :
    Fin (2 ^ (lvOf sch ℓ).logLen) → Option (Fin (2 ^ (lvOf sch ℓ).lanes) → F) :=
  opOfL H E Lf (lvOf sch ℓ).lanes (pathLen (lvOf sch ℓ)) (lvOf sch ℓ).logLen
    (slotOpensL out r ℓ (lvOf sch ℓ).queries) cap

/-- **A self-clash**: within one run, two verifying openings of a position to different rows, at the level-0 cap or
at a rep's recorded cap. -/
def SelfClashL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (out : OutCL F D Sl) : Prop :=
  ClashL H E Lf out.cap₀ (lvOf sch 0).lanes (pathLen (lvOf sch 0)) (opens₀L sch out) ∨
    ∃ r < 2, ∃ ℓ, 1 ≤ ℓ ∧ ∃ cap,
      (if r = 0 then out.pend.1 else out.pend.2).caps[ℓ - 1]? = some cap ∧
        ClashL H E Lf cap (lvOf sch ℓ).lanes (pathLen (lvOf sch ℓ))
          (slotOpensL out r ℓ (lvOf sch ℓ).queries)

/-! ## Where a bad run goes wrong -/

/-! ## Clashes and conflicts are collisions -/

/-! ## The plurality prover and the collision finders -/

/-! ## The theorem -/

end FlockSoundness.Discharge.Exec
