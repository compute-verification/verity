import Proofs.Properties.Verity.Sampled.Moves
import Proofs.Flock.Soundness.Discharge.ZkExec.Setup

/-!
# Sampled proofs: the session's record draws the draw v2 drew

v2 refuses a session whose record's `unit_draw` is not the draw it drew (`SampledProofs.OneStageZk.drawHeld`): under
the law `L` and population `n` v1's check draws under (`SampledProofs.OneStageZk.drawLaw`, `LawOf`), the record must print,
in canonical UTF-8, as the draw of `L` over `n` units drawing the evidence's units (`recordDrawn`). So wherever v2 drew
`units` in what the prover sent (`Drew`), the record it reads there (`sentRecord`) prints as that draw (`DrawHeld`,
`draw_held`).

`DrawHeld` takes the composition's acceptance (`Verity.SameAccepts`), as `sent_verified` does: `Drew`'s case is one of the
components' cases, zipped with the parameters and the evidence in order, while `sentRecord` reads the case `caseOf` finds by
sampled proofs' id. The two are one case only where every entry carries its component's id (`caseOf_of`); otherwise the
record `sentRecord` reads can be another entry's, which no check of the case `Drew` names holds.
-/

namespace Verity.Sampled

open FlockSoundness.Refine (ebind_ok)

/-- **v2's check at the coins `ξ` accepts only where its sessions' records draw the draw it drew** (`drawHeld`). -/
theorem drawHeld_of_checkBlobsAt {p : _root_.SampledProofs.OneStageZk.Params}
    {c : _root_.SampledProofs.OneStage.Commitment} {ξ : ByteArray} {e : _root_.SampledProofs.OneStageZk.Evidence}
    {bs : _root_.Core.Blobs} {pub : _root_.SampledProofs.OneStage.Public} {prof : _root_.Core.Integrity}
    (h : _root_.SampledProofs.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    _root_.SampledProofs.OneStageZk.drawHeld p.base.verifier c e bs = .ok () := by
  unfold _root_.SampledProofs.OneStageZk.checkBlobsAt at h
  dsimp only at h
  split at h
  · cases h
  · rename_i r hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨⟨⟩, hd, -⟩ := ebind_ok hr
    exact hd

/-- **The parameters and the commitment's law and population are `drawLaw`'s** (`LawOf`). -/
theorem drawLaw_of_lawOf {p : _root_.SampledProofs.OneStageZk.Params} {c : _root_.SampledProofs.OneStage.Commitment}
    {L : Flock.Draw.Law} {n : ℕ} (h : LawOf p c L n) :
    _root_.SampledProofs.OneStageZk.drawLaw p.base.verifier c.record = some (L, n) := by
  obtain ⟨law, derived, fl, h1, h2, h3, h4, rfl⟩ := h
  unfold _root_.SampledProofs.OneStageZk.drawLaw
  simp only [h1, h2, h3, h4]

/-- **Where `drawHeld` passes, the record v2 reads draws the evidence's units** (`sessionRecord`, `drawnUnits`), under
the law and population `drawLaw` gives. -/
theorem recordDrawn_of_drawHeld {v : _root_.SampledProofs.Registration.Verifier}
    {c : _root_.SampledProofs.OneStage.Commitment} {e : _root_.SampledProofs.OneStageZk.Evidence}
    {bs : _root_.Core.Blobs} {L : Flock.Draw.Law} {n : ℕ} {record : String}
    (h : _root_.SampledProofs.OneStageZk.drawHeld v c e bs = .ok ())
    (hl : _root_.SampledProofs.OneStageZk.drawLaw v c.record = some (L, n)) (hr : sessionRecord e bs = some record) :
    _root_.SampledProofs.OneStageZk.recordDrawn L n (drawnUnits e.base.draw) record = true := by
  unfold _root_.SampledProofs.OneStageZk.drawHeld at h
  rw [hl] at h
  dsimp only at h
  unfold sessionRecord at hr
  cases h0 : e.sessions[0]? with
  | none => rw [h0] at hr; cases hr
  | some s =>
    simp only [h0, Option.bind_some] at hr
    obtain ⟨hj, hs⟩ := Array.getElem?_eq_some_iff.1 h0
    have hd := FlockSoundness.Discharge.ZkExec.array_forM_ok h 0 hj
    rw [hs] at hd
    unfold _root_.SampledProofs.OneStageZk.sessionDrawn at hd
    obtain ⟨recB, hb, hd⟩ := ebind_ok hd
    rw [hb] at hr
    have hr' : String.fromUTF8? recB = some record := hr
    rw [hr'] at hd
    dsimp only at hd
    split at hd
    · rename_i hrd
      unfold drawnUnits
      generalize _root_.SampledProofs.getD e.base.draw "units" = j at hrd ⊢
      cases j <;> exact hrd
    · cases hd

/-- **v2 holds the record it reads to the draw it drew**: wherever composite `k` accepted what the prover sent as a
`same` composition and v2 drew `units` there under the law `L` over `n` units (`Drew`), the session's record v2 reads
(`sentRecord`) prints, in canonical UTF-8, as that draw (`recordDrawn`). -/
def DrawHeld (k : Verity.Registration.Composite) : Prop :=
  ∀ s dr L n ξ units record, Verity.SameAccepts k s → Drew k s dr L n ξ units → sentRecord s = some record →
    _root_.SampledProofs.OneStageZk.recordDrawn L n units record = true

/-- **v2's refusal holds the record to its draw** (`drawHeld`), at every composite. -/
theorem draw_held (k : Verity.Registration.Composite) : DrawHeld k := by
  intro s dr L n ξ units record hS hd hrec
  obtain ⟨a, b, e, p, c, ev, hcase, hacc, hlaw, rfl, -⟩ := hd
  obtain ⟨hp, hc, hev, -, -, pub, prof, hck⟩ := hacc
  have hcs := caseOf_of hS hcase hp hc hev
  have hr : sessionRecord ev (_root_.Core.Blobs.ofArray s.blobs) = some record := by
    simpa only [sentRecord, hcs, ok_toOption, Option.bind_some] using hrec
  exact recordDrawn_of_drawHeld (drawHeld_of_checkBlobsAt hck) (drawLaw_of_lawOf hlaw) hr

end Verity.Sampled
