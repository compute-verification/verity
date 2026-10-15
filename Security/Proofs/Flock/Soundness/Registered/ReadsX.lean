import Proofs.Flock.Soundness.Registered.Binding
import Proofs.Flock.Soundness.Binding.RowsX

/-!
# Registered reads of rows each hashed under its own encoding

`ReadsX` is `Reads` over `HmRowsX`: the registrant's leaf at a registered read is `hm96X`'s over that commit string's
encoding (`ReadsX.leaf`), the same SHA-512 of the same commit string as hm96's. `Checked`, `Off`, `pick` and their
bounds (`regLeaf_eq`, `prob_off_le`) are `Reads`' at that leaf: they read only the leaf, never how its bytes were made.
-/

namespace FlockSoundness.Registered

open Game Refine Binding Audit
open scoped ENNReal

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}

/-- **A layout's registered reads, its rows each under its own encoding**: `Reads` with the registrant's opening under
`hm96X` at the read's encoding (`committed`). -/
structure ReadsX (lay : Layout P plan tab decode) (rs : HmRowsX lay) where
  isRead : lay.Pos → Prop
  read_row : ∀ p, isRead p → rs.isRow p
  port : lay.Pos → Flock.Registered.Port
  readAt : lay.Pos → ℕ
  wv : lay.Pos → lay.Val
  ws : lay.Pos → List UInt8
  wpath : lay.Pos → ByteArray
  committed : ∀ p, isRead p →
    Flock.Registered.opensLeaf (port p)
      ((ExplicitLeaf.hm96X (rs.enc p) (rs.ext p) (rs.ext_spec p)).leaf (wv p) (ws p)).1 (readAt p) (wpath p) = true

namespace ReadsX

variable {lay : Layout P plan tab decode} {rs : HmRowsX lay} (rd : ReadsX lay rs)

/-- The registrant's leaf at `p`: `hm96X`'s leaf of the row and salt it committed there. -/
def leaf (p : lay.Pos) : Dig := (ExplicitLeaf.hm96X (rs.enc p) (rs.ext p) (rs.ext_spec p)).leaf (rd.wv p) (rd.ws p)

/-- **What the verifier's registered check establishes** at the registration `R`: `Reads.Checked`. -/
def Checked (R : Reg) (path : lay.Pos → ByteArray) : Prop :=
  ∀ p, rd.isRead p → Flock.Registered.opensLeaf (rd.port p) (lay.regLeaf R p).1 (rd.readAt p) (path p) = true

/-- The session's opening at `p` under the registration `R`, through `path p`, and the registrant's. -/
def openings (R : Reg) (path : lay.Pos → ByteArray) (p : lay.Pos) : TwoOpenings :=
  ⟨rd.port p, rd.readAt p, lay.regLeaf R p, path p, rd.leaf p, rd.wpath p⟩

/-- Some registered read's leaf under the registration `R` is not the registrant's. -/
def Off (R : Reg) : Prop := ∃ p, rd.isRead p ∧ lay.regLeaf R p ≠ rd.leaf p

/-- A registered read whose leaf under `R` is not the registrant's, if there is one (else any commit string). -/
noncomputable def offAt (R : Reg) : lay.Pos := Classical.epsilon fun p => rd.isRead p ∧ lay.regLeaf R p ≠ rd.leaf p

/-- The registered-path finder's output: the extractor's pair from the session's opening at `offAt R` and the
registrant's. -/
noncomputable def pick (R : Reg) (path : lay.Pos → ByteArray) : Option (List UInt8 × List UInt8) :=
  some (rd.openings R path (rd.offAt R)).pair

/-- **A registered read's leaf is the registrant's, or SHA-512 collides**: `Reads.regLeaf_eq`. -/
theorem regLeaf_eq {R : Reg} {path : lay.Pos → ByteArray} (hc : rd.Checked R path) {p : lay.Pos} (hp : rd.isRead p) :
    lay.regLeaf R p = rd.leaf p ∨ Collides (rd.openings R path p).pair := by
  by_cases h : lay.regLeaf R p = rd.leaf p
  · exact Or.inl h
  · exact Or.inr ((rd.openings R path p).collides ⟨h, hc p hp, rd.committed p hp⟩)

/-- On `Off R` where the check holds, `offAt R`'s openings conflict. -/
theorem offAt_conflict {R : Reg} {path : lay.Pos → ByteArray} (hc : rd.Checked R path) (h : rd.Off R) :
    (rd.openings R path (rd.offAt R)).Conflict := by
  have hs : rd.isRead (rd.offAt R) ∧ lay.regLeaf R (rd.offAt R) ≠ rd.leaf (rd.offAt R) := Classical.epsilon_spec h
  exact ⟨hs.2, hc _ hs.1, rd.committed _ hs.1⟩

/-- **The registered reads' term under `SHA512CRStrict` (`cr/sha-512`)**: `Reads.prob_off_le`. -/
theorem prob_off_le {β : Type} (G : Game β) (σ : Strategy G) (R : Reg) (accept : β → Prop)
    (path : β → lay.Pos → ByteArray) (hchk : ∀ o, accept o → rd.Checked R (path o)) {qR : ℝ}
    (hR : (StrictCR.oneRun G σ fun o => rd.pick R (path o)).CR H512 qR) :
    prob (fun o => accept o ∧ rd.Off R) G σ ≤ ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) := by
  have hle : prob (fun o => accept o ∧ rd.Off R) G σ ≤ prob (fun o => StrictCR.IsColl H512 (rd.pick R (path o))) G σ :=
    prob_mono (fun o ⟨ha, hoff⟩ => (rd.openings R (path o) (rd.offAt R)).isColl (rd.offAt_conflict (hchk o ha) hoff)) G σ
  have hev : (prob (fun o => StrictCR.IsColl H512 (rd.pick R (path o))) G σ).toReal ≤ qR ^ 2 / 2 ^ (513 : ℝ) := by
    rw [prob_toReal]
    have := StrictCR.Finder.CR.le hR
    rwa [StrictCR.oneRun_ev] at this
  have hfin : prob (fun o => StrictCR.IsColl H512 (rd.pick R (path o))) G σ ≠ ⊤ :=
    ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)
  exact hle.trans ((ENNReal.le_ofReal_iff_toReal_le hfin (by positivity)).2 hev)

end ReadsX

end FlockSoundness.Registered
