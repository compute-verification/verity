import Proofs.Properties.Verity.SampledSizes

/-!
# Session 0's record carries the verifier's own draw, from v2's refusals

The record step of `Canon.DrawAt` asks that, at every case v2 accepts, session 0's record carry the draw file `dj` of
the units the evidence's draw names. v2 refuses evidence of other than one session (`OneStageZk.oneSession_of_accept`),
a record whose draw is not a subset draw of the verifier's `k` over the registered population, of the evidence's draw's
units, ascending and below it (`OneStageZk.drawLaw_of_accept`), or whose draw file is not the verifier's own
(`OneStageZk.drawFile_of_accept`), proved units other than the record's draw's (`OneStageZk.provedUnits_of_accept`), and a
public file whose `units.indices` is not the identity (`OneStageZk.unitsIdentity_of_accept`). With `drawJ kd` as `dj`,
that is `recordDrawn_of_accept`, at the population `n` and subset size `kd` the play fixes of each accepting case
(`hcase`).
-/

open Lean (Json)

namespace Verity.Canon

/-- The draw file of a set `S` of units of `[0, n)`: the verifier's own subset draw of `kd` over `[0, n)` of `S`'s units
in order (`OneStageZk.drawFile`), rebuilt (`Flock.norm`) as a record's draw is. -/
def drawJ (kd : ℕ) {n : ℕ} (S : Finset (Fin n)) : Json :=
  Flock.norm (_root_.SampledProofs.OneStageZk.drawFile kd n ((S.image Fin.val).sort (· ≤ ·)).toArray)

/-- The units of `[0, n)` in an ascending array `us` of units below `n`, in order, are `us`. -/
theorem sort_filter_mem {n : ℕ} {us : Array ℕ} (hs : us.toList.Pairwise (· < ·)) (hlt : ∀ u ∈ us, u < n) :
    (((Finset.univ.filter (fun u : Fin n => u.val ∈ us)).image Fin.val).sort (· ≤ ·)).toArray = us := by
  have h : ((Finset.univ.filter (fun u : Fin n => u.val ∈ us)).image Fin.val).sort (· ≤ ·) = us.toList := by
    refine (Finset.sortedLT_sort _).eq_of_mem_iff hs.sortedLT fun a => ?_
    simp only [Finset.mem_sort, Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and,
      Array.mem_toList_iff]
    constructor
    · rintro ⟨u, hu, rfl⟩
      exact hu
    · intro ha
      exact ⟨⟨a, hlt a ha⟩, ha, rfl⟩
  rw [h]

/-- **Session 0's record carries the verifier's own draw**: at every case v2 accepts in what was sent `s` whose
verifier's law over the registration's population `n` is the subset law of `kd` (`hcase`, what the play fixes of an
accepting case's draw), the transcript of the evidence's first session carries the draw file `drawJ kd` of the units the
evidence's draw names. This is `RecordDrawn (drawJ kd) s`'s body. -/
theorem recordDrawn_of_accept {s : Sent} {n kd : ℕ}
    (hcase : ∀ p c e pub prof, SampledCase s p c e →
      _root_.SampledProofs.OneStageZk.checkBlobs p c e (Core.Blobs.ofArray s.blobs) = .accept pub prof →
      _root_.SampledProofs.Draw.flockLaw p.base.verifier.law n = .ok (.subset kd) ∧
        _root_.SampledProofs.nat? (_root_.SampledProofs.getD c.record "population") = some n) :
    ∀ p c e pub prof s0 rest t, SampledCase s p c e →
      _root_.SampledProofs.OneStageZk.checkBlobs p c e (Core.Blobs.ofArray s.blobs) = .accept pub prof →
      e.sessions.toList = s0 :: rest → Properties.SampledProofs.OneStageZk.SessionTranscript p e s.blobs s0 t →
      FlockSoundness.Discharge.Exec.recordDraw t.record =
        some (drawJ kd (Finset.univ.filter
          (fun u : Fin n => u.val ∈ _root_.SampledProofs.OneStage.drawnOf e.base.draw))) := by
  intro p c e pub prof s0 rest t hc hacc hl ht
  obtain ⟨_, _, _, _, b, _, _, _, _, hb, hu, _⟩ := ht
  obtain ⟨hlaw, hpop⟩ := hcase p c e pub prof hc hacc
  obtain ⟨n', k', hpop', hlaw', hpw, hlt, -, -⟩ :=
    _root_.SampledProofs.OneStageZk.drawLaw_of_accept hacc hl hb hu
  obtain ⟨n'', k'', hpop'', hlaw'', hfile⟩ := _root_.SampledProofs.OneStageZk.drawFile_of_accept hacc hl hb hu
  rw [hpop] at hpop' hpop''
  cases hpop'
  cases hpop''
  rw [hlaw] at hlaw' hlaw''
  cases hlaw'
  cases hlaw''
  have hd : _root_.SampledProofs.OneStage.drawnOf e.base.draw =
      _root_.SampledProofs.OneStageZk.drawnUnits e.base.draw := rfl
  rw [Properties.SampledProofs.OneStageZk.Run.exec_recordDraw, hfile, hd]
  unfold drawJ
  rw [sort_filter_mem hpw hlt]

end Verity.Canon
