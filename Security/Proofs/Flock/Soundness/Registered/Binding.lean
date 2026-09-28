import Proofs.Flock.Soundness.Registered.Defs
import Proofs.Flock.Soundness.Binding.Weights
import Proofs.Flock.Soundness.StrictCR

/-!
# Registered reads bind the registrant's rows, under `cr/sha-512`

* `TwoOpenings.collides`: two conflicting openings of one position under one registered root give an explicit SHA-512
  collision, the executable's extractor's (`Flock.Registered.opensLeaf_binding`) read on byte lists.
* `TwoOpenings.conflict_strict`: so a finder that holds two conflicting openings with probability `ε` and makes at most
  `q` SHA-512 evaluations has `ε ≤ q²/2^513`, under `SHA512CRStrict` (`cr/sha-512`) for the finder that outputs the
  extractor's pair.
* `Reads.regLeaf_eq`: where the verifier's registered check holds (`Reads.Checked`), a registered read's leaf in the
  layout is the registrant's, or SHA-512 collides; `Reads.registered_weights` is `HmRows.registered_weights_hm96` at a
  registered read, against the row the registrant committed.
* `Reads.prob_off_le`: in any game whose accepted outcomes the registered check holds on, the probability of accepting
  while some registered read's leaf is not the registrant's is at most `qR²/2^513`, under `SHA512CRStrict` for the finder
  that plays the game once and outputs the extractor's pair (`Reads.pick`).
-/

namespace FlockSoundness.Registered

open Game Refine Binding Audit
open scoped ENNReal

/-- A SHA-512 collision of two byte arrays is one of the model's SHA-512 on their bytes. -/
theorem collides_bytes (x : ByteArray × ByteArray) (h : Flock.Registered.Collides x) :
    Collides (x.1.data.toList, x.2.data.toList) :=
  ⟨fun e => h.1 (ByteArray.ext (Array.ext' e)), Subtype.ext (by simp only [H512_val, bytes_of_list]; exact h.2)⟩

/-- **Two conflicting openings give an explicit SHA-512 collision**: the extractor's pair (`TwoOpenings.pair`) is two
different byte strings with one digest under the model's SHA-512 (`Refine.H512`). -/
theorem TwoOpenings.collides (o : TwoOpenings) (h : o.Conflict) : Collides o.pair := by
  obtain ⟨hne, h1, h2⟩ := h
  simp only [TwoOpenings.pair]
  exact collides_bytes _ (Flock.Registered.opensLeaf_binding o.port o.pos o.leaf₁.1 o.path₁ o.leaf₂.1 o.path₂ o.leaf₁.2
    o.leaf₂.2 (fun e => hne (Subtype.ext e)) h1 h2)

/-- As a finder's output, two conflicting openings' pair is a collision. -/
theorem TwoOpenings.isColl (o : TwoOpenings) (h : o.Conflict) : StrictCR.IsColl H512 (some o.pair) :=
  ⟨o.pair.1, o.pair.2, congrArg some Prod.mk.eta.symm, o.collides h⟩

/-- **Registered-path binding under `SHA512CRStrict` (`cr/sha-512`).** A finder (a game `g` played by `s`) that, on each
outcome `b`, holds two openings `o b` of one position under one registered root, and outputs the extractor's pair, with
at most `q` SHA-512 evaluations on every outcome (`cost`): the probability that the two openings conflict is at most
`q²/2^513`. -/
theorem TwoOpenings.conflict_strict {β : Type} (g : Game β) (s : Strategy g) (o : β → TwoOpenings) {cost : β → ℝ}
    {q : ℝ} (hcost : ∀ b, cost b ≤ q)
    (hCR : Assumptions.SHA512CRStrict H512 g s (fun b => some (o b).pair) cost q) :
    (prob (fun b => (o b).Conflict) g s).toReal ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  (ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (prob_mono (fun b hb => (o b).isColl hb) g s)).trans (hCR hcost)

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}

namespace Reads

variable {lay : Layout P plan tab decode} {rs : HmRows lay} (rd : Reads lay rs)

/-- **A registered read's leaf is the registrant's, or SHA-512 collides.** Where the verifier's registered check holds at
the registration `R` through `path`, the layout's leaf at a registered read `p` (`Layout.regLeaf R p`, the leaf the value
binding commits to there) is the registrant's, or the session's opening and the registrant's give an explicit SHA-512
collision. -/
theorem regLeaf_eq {R : Reg} {path : lay.Pos → ByteArray} (hc : rd.Checked R path) {p : lay.Pos} (hp : rd.isRead p) :
    lay.regLeaf R p = rd.leaf p ∨ Collides (rd.openings R path p).pair := by
  by_cases h : lay.regLeaf R p = rd.leaf p
  · exact Or.inl h
  · exact Or.inr ((rd.openings R path p).collides ⟨h, hc p hp, rd.committed p hp⟩)

/-- **`registered_weights_hm96` at a registered read.** Where the verifier's registered check holds at the registration
`R`, in a satisfying witness of a drawn unit's table, the row the witness holds at a registered read is the row the
registrant committed there (`wv`), or SHA-512 collides: in hm96's leaf (`hm96Pair`), or between the session's opening of
the registered root and the registrant's. -/
theorem registered_weights (hHm : rs.Computes) {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S)
    (M : Msg plan tab S R u) (hM : Sat plan tab S R u M) {g : Fin C.N} (hg : g ∈ P.io u)
    (hrd : rd.isRead (lay.pos g)) {path : lay.Pos → ByteArray} (hc : rd.Checked R path) :
    lay.row S R u M (lay.pos g) = rd.wv (lay.pos g) ∨
      Collides (hm96Pair (rs.enc (lay.row S R u M (lay.pos g))) (lay.salt S R u M (lay.pos g))
        (rs.enc (rd.wv (lay.pos g))) (rd.ws (lay.pos g))) ∨
      Collides (rd.openings R path (lay.pos g)).pair := by
  rcases rd.regLeaf_eq hc hrd with h | h
  · exact (rs.registered_weights_hm96 hHm hu M hM hg (rd.read_row _ hrd) (rd.wv _) (rd.ws _) h.symm).imp_right Or.inl
  · exact Or.inr (Or.inr h)

/-- On `Off R` where the check holds, `offAt R`'s openings conflict. -/
theorem offAt_conflict {R : Reg} {path : lay.Pos → ByteArray} (hc : rd.Checked R path) (h : rd.Off R) :
    (rd.openings R path (rd.offAt R)).Conflict := by
  have hs : rd.isRead (rd.offAt R) ∧ lay.regLeaf R (rd.offAt R) ≠ rd.leaf (rd.offAt R) := Classical.epsilon_spec h
  exact ⟨hs.2, hc _ hs.1, rd.committed _ hs.1⟩

/-- **The registered reads' term under `SHA512CRStrict` (`cr/sha-512`).** In a game `G` played by `σ` whose accepted
outcomes (`accept`) the verifier's registered check holds on at the registration `R`, through the outcome's paths: the
probability of accepting while some registered read's leaf is not the registrant's is at most `qR²/2^513`, for the
finder that plays `G` once and outputs the extractor's pair (`Reads.pick`), at a budget `qR` of SHA-512 evaluations
(the game's and the extractor's). -/
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

end Reads

end FlockSoundness.Registered
