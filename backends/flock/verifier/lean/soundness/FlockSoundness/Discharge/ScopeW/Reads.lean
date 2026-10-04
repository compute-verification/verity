import FlockSoundness.Registered.Binding
import FlockSoundness.Discharge.ScopeW.Rows

set_option Elab.async false

/-!
# Registered reads at rows under their own schemas

`Registered.Reads` reads a session's registered rows through `HmRows`, every row hashed under one injective encoding
(`ExplicitLeaf.hm96`). Under the widened scope a row's leaf is its schema's (`HmRowsP.scheme`, `hm96X`), and
`HmRowsP.toHmRows` has no row, so a `Registered.Reads` over it reads nothing. `ReadsP` is `Registered.Reads` at
`HmRowsP`: the same reads, the registrant's leaf at `p` hm96's of `p`'s own encoding (`hm96X (rs.enc p)`), so a read can
stand at any row `rowShapeW` admits.

* `ReadsP`, `leaf`, `Checked`, `openings`, `Off`, `offAt`, `pick`: `Registered.Reads`' at `HmRowsP`. `Checked`, `Off`
  and `openings` read the layout's leaves (`Layout.regLeaf`) and the registrant's, as `Reads`' do;
* `regLeaf_eq`, `offAt_conflict`, `prob_off_le`: `Reads`' at `HmRowsP`, through `TwoOpenings.collides`;
* `registered_weights`: `Reads.registered_weights` at `HmRowsP`: the row a drawn unit's witness holds at a registered
  read is the registrant's, or the row's leaf collides (`hm96X`'s pair: the row's extractor `HmRowsP.pair` at equal
  encodings, `hm96Pair` otherwise), or the session's opening and the registrant's do.
-/

namespace FlockSoundness.Discharge.ScopeW

open Game Refine Binding Audit
open FlockSoundness.Registered (TwoOpenings)
open scoped ENNReal

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}

/-- **A layout's registered reads, at rows under their own schemas** (`Registered.Reads` at `HmRowsP`). The commit
strings whose rows the session reads from a value registered before (`isRead`), each a row (`read_row`); at each, the
registered root the verifier holds (`port`) and the position the verifier's program reads there (`readAt`); and the
registrant's opening there: the row and salt it committed (`wv`, `ws`) and their path (`wpath`), which open that root at
that position under the row's own leaf scheme (`committed`). -/
structure ReadsP (lay : Layout P plan tab decode) (rs : HmRowsP lay) where
  isRead : lay.Pos → Prop
  read_row : ∀ p, isRead p → rs.isRow p
  port : lay.Pos → Flock.Registered.Port
  readAt : lay.Pos → ℕ
  wv : lay.Pos → lay.Val
  ws : lay.Pos → List UInt8
  wpath : lay.Pos → ByteArray
  committed : ∀ p, isRead p →
    Flock.Registered.opensLeaf (port p) ((hm96X (rs.enc p) (rs.pair p) (rs.pair_spec p)).leaf (wv p) (ws p)).1
      (readAt p) (wpath p) = true

namespace ReadsP

variable {lay : Layout P plan tab decode} {rs : HmRowsP lay} (rd : ReadsP lay rs)

/-- The registrant's leaf at `p`: hm96's leaf of `p`'s own encoding of the row and salt it committed there. -/
def leaf (p : lay.Pos) : Dig := (hm96X (rs.enc p) (rs.pair p) (rs.pair_spec p)).leaf (rd.wv p) (rd.ws p)

/-- **What the verifier's registered check establishes** at the registration `R` (`Registered.Reads.Checked`): at each
registered read `p`, the session's leaf there (`Layout.regLeaf`) opens the root the verifier holds, at the position the
verifier's program reads, through `path p`. -/
def Checked (R : Reg) (path : lay.Pos → ByteArray) : Prop :=
  ∀ p, rd.isRead p → Flock.Registered.opensLeaf (rd.port p) (lay.regLeaf R p).1 (rd.readAt p) (path p) = true

/-- The session's opening at `p` under the registration `R`, through `path p`, and the registrant's. -/
def openings (R : Reg) (path : lay.Pos → ByteArray) (p : lay.Pos) : TwoOpenings :=
  ⟨rd.port p, rd.readAt p, lay.regLeaf R p, path p, rd.leaf p, rd.wpath p⟩

/-- Some registered read's leaf under the registration `R` is not the registrant's. -/
def Off (R : Reg) : Prop := ∃ p, rd.isRead p ∧ lay.regLeaf R p ≠ rd.leaf p

/-- A registered read whose leaf under `R` is not the registrant's, if there is one (else any commit string). -/
noncomputable def offAt (R : Reg) : lay.Pos := Classical.epsilon fun p => rd.isRead p ∧ lay.regLeaf R p ≠ rd.leaf p

/-- The registered-path finder's output: the extractor's pair from the session's opening at `offAt R` through `path` and
the registrant's. -/
noncomputable def pick (R : Reg) (path : lay.Pos → ByteArray) : Option (List UInt8 × List UInt8) :=
  some (rd.openings R path (rd.offAt R)).pair

/-- **A registered read's leaf is the registrant's, or SHA-512 collides** (`Registered.Reads.regLeaf_eq`). -/
theorem regLeaf_eq {R : Reg} {path : lay.Pos → ByteArray} (hc : rd.Checked R path) {p : lay.Pos} (hp : rd.isRead p) :
    lay.regLeaf R p = rd.leaf p ∨ Collides (rd.openings R path p).pair := by
  by_cases h : lay.regLeaf R p = rd.leaf p
  · exact Or.inl h
  · exact Or.inr ((rd.openings R path p).collides ⟨h, hc p hp, rd.committed p hp⟩)

/-- **The registered row, at rows under their own schemas** (`Registered.Reads.registered_weights`). Where the verifier's
registered check holds at the registration `R`, in a satisfying witness of a drawn unit's table, the row the witness
holds at a registered read is the row the registrant committed there (`wv`), or SHA-512 collides: in the row's leaf
(`hm96X`'s pair), or between the session's opening of the registered root and the registrant's. -/
theorem registered_weights (hHm : rs.Computes) {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S)
    (M : Msg plan tab S R u) (hM : Sat plan tab S R u M) {g : Fin C.N} (hg : g ∈ P.io u)
    (hrd : rd.isRead (lay.pos g)) {path : lay.Pos → ByteArray} (hc : rd.Checked R path) :
    lay.row S R u M (lay.pos g) = rd.wv (lay.pos g) ∨
      Collides ((hm96X (rs.enc (lay.pos g)) (rs.pair (lay.pos g)) (rs.pair_spec (lay.pos g))).pair
        (lay.row S R u M (lay.pos g)) (lay.salt S R u M (lay.pos g)) (rd.wv (lay.pos g)) (rd.ws (lay.pos g))) ∨
      Collides (rd.openings R path (lay.pos g)).pair := by
  rcases rd.regLeaf_eq hc hrd with h | h
  · by_cases hv : lay.row S R u M (lay.pos g) = rd.wv (lay.pos g)
    · exact Or.inl hv
    · refine Or.inr (Or.inl ((hm96X _ _ _).pair_spec _ _ _ _ hv ?_))
      have hs : rs.scheme (lay.pos g) =
          hm96X (rs.enc (lay.pos g)) (rs.pair (lay.pos g)) (rs.pair_spec (lay.pos g)) := by
        simp only [HmRowsP.scheme, rd.read_row _ hrd, ↓reduceIte]
      have hr := rs.registered hHm S R u hu M hM g hg
      rw [hs] at hr
      exact hr.trans h
  · exact Or.inr (Or.inr h)

/-- On `Off R` where the check holds, `offAt R`'s openings conflict. -/
theorem offAt_conflict {R : Reg} {path : lay.Pos → ByteArray} (hc : rd.Checked R path) (h : rd.Off R) :
    (rd.openings R path (rd.offAt R)).Conflict := by
  have hs : rd.isRead (rd.offAt R) ∧ lay.regLeaf R (rd.offAt R) ≠ rd.leaf (rd.offAt R) := Classical.epsilon_spec h
  exact ⟨hs.2, hc _ hs.1, rd.committed _ hs.1⟩

/-- **The registered reads' term under `SHA512CRStrict` (`cr/sha-512`)** (`Registered.Reads.prob_off_le`). In a game
`G` played by `σ` whose accepted outcomes (`accept`) the verifier's registered check holds on at the registration `R`,
through the outcome's paths: the probability of accepting while some registered read's leaf is not the registrant's is
at most `qR²/2^513`, for the finder that plays `G` once and outputs the extractor's pair (`pick`). -/
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

end ReadsP

end FlockSoundness.Discharge.ScopeW
