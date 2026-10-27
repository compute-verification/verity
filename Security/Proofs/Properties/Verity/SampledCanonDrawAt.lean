import Proofs.Properties.Verity.SampledProj
import Proofs.Properties.Verity.SampledCanonDraw
import Proofs.Flock.Soundness.Discharge.ZkLink.DrawOk

/-!
# The record's draw at `proj`

`Canon.proj o` is session 0's transcript of what the prover sent (`Canon.transcriptOf`, `Canon.Sess0`), at an accepting
case `proj` picks by choice. At every accepting case, `Vocab.draw_at_rounds` says the draw `drawOS` makes from the
play's release of the draw's family is the units the evidence's draw names, or the finder outputs a break from one of
the uses `roundsIn` reads. The record step (`Canon.RecordDrawn`) is that session 0's record carries `dj` of those
units. Together they give `Canon.DrawAt` at `proj o` itself, at whichever case `proj` picked (`Canon.drawAt_proj`).

`RecordDrawn` follows from two of v2's refusals (`Canon.recordDrawn_of_accept`): `draw-file`, session 0's record
carries, byte for byte, the canonical draw file of the verifier's draw (`Canon.DrawFileOfAccept`), and `proved-units`,
the units the evidence's draw names are ascending units of the population (`Canon.ProvedUnitsOfAccept`), at `dj` the
canonical draw file (`UnitDraw.toJson` of `drawOf`). `draw-law` makes the verifier's own law and population the subset
law of `kd` over `n` (`Canon.DrawLawOfAccept`), so `drawAt_proj_of_accept` needs no registration premise beyond it.
-/

open Lean (Json)

namespace Verity.Canon

open _root_.SampledProofs.Challenger

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## The draw at `proj` -/

/-- **Session 0's record carries the draw of the evidence's units**: at every case v2 accepts in what was sent `s`, the
transcript of the evidence's first session carries the draw file `dj` of the units the evidence's draw names. -/
def RecordDrawn {n : ℕ} (dj : Finset (Fin n) → Json) (s : Sent) : Prop :=
  ∀ p c e pub prof s0 rest t, SampledCase s p c e →
    _root_.SampledProofs.OneStageZk.checkBlobs p c e (Core.Blobs.ofArray s.blobs) = .accept pub prof →
    e.sessions.toList = s0 :: rest → Properties.SampledProofs.OneStageZk.SessionTranscript p e s.blobs s0 t →
    Exec.recordDraw t.record =
      some (dj (Finset.univ.filter (fun u : Fin n => u.val ∈ _root_.SampledProofs.OneStage.drawnOf e.base.draw)))

/-- **What the play fixes of an accepting case's draw**: the challenger's key is the play's registered one, the
verifier's law is a subset law of `kd` over the record's population `n`, and the bytes `x` stream to what the play's
session released of the draw's family after the registration's digest. -/
def CaseDraw (V : Vocab d α) (θ : ByteArray) (o : Outcome) {n : ℕ} (kd : ℕ) {Lb : Fin 1 → ℕ}
    (x : (i : Fin 1) → Fin (Lb i) → Fin 256) (p : _root_.SampledProofs.OneStageZk.Params)
    (c : _root_.SampledProofs.OneStage.Commitment) : Prop :=
  ∃ (key : ChallengerKey) (label : String), p.base.challenger = some (key, label) ∧ key.pub = (registeredKey θ).2 ∧
    _root_.SampledProofs.Draw.flockLaw p.base.verifier.law n = .ok (.subset kd) ∧
    _root_.SampledProofs.nat? (_root_.SampledProofs.getD c.record "population") = some n ∧
    FlockSoundness.Audit.Law.streamOf (x 0) = ((V.device θ).session o.log).released label c.digest

/-- **A break from one of the uses `roundsIn` reads**, at an accepting case of what the play's outcome `o` sent. -/
def RoundsBreak (V : Vocab d α) (θ : ByteArray) (o : Outcome) : Prop :=
  ∃ p c e pub prof (key : ChallengerKey) (label : String) (uses : List Use)
    (wants : List (Option (String × Array String))) (order : List (String × String)) (groups : List (List String))
    (coins : Json) (i : ℕ) (u : Use),
    SampledCase o.sent p c e ∧
    _root_.SampledProofs.OneStageZk.checkBlobs p c e (Core.Blobs.ofArray o.sent.blobs) = .accept pub prof ∧
    p.base.challenger = some (key, label) ∧
    _root_.SampledProofs.OneStageZk.roundsIn label c e (Core.Blobs.ofArray o.sent.blobs) =
      .ok (uses.toArray, wants.toArray, order.toArray, groups.toArray, coins) ∧
    uses[i]? = some u ∧ Properties.Challenger.Break (V.deviceOf θ o) key u.label (Properties.Challenger.useCoins coins i)

/-- **`Canon.DrawAt` at `proj o`**: where session 0 has a transcript, every accepting case's draw is fixed by the play
(`hcase`: the registered key; the law and population, `hlaw` and `hpop` of `draw_at_rounds`, by registration; the
stream of `x`, its `hx`, by the device's release, `rounds_at_play`, which also fixes `Lb 0` as the release's size) and
session 0's record carries the draw of the evidence's units (`RecordDrawn`), the record at `proj o` carries the draw
file of the draw `drawOS` makes from `x`; or, at the case `proj` picked, the finder outputs a break from one of the uses
`roundsIn` reads (`RoundsBreak`). -/
theorem drawAt_proj (V : Vocab d α) (hR : Properties.Challenger.RoundsByDevice) {θ : ByteArray} (o : Outcome)
    {n kd : ℕ} {Lb : Fin 1 → ℕ} (x : (i : Fin 1) → Fin (Lb i) → Fin 256) {dj : Finset (Fin n) → Json}
    (h0 : ∃ t, Sess0 o.sent t)
    (hcase : ∀ p c e pub prof, SampledCase o.sent p c e →
      _root_.SampledProofs.OneStageZk.checkBlobs p c e (Core.Blobs.ofArray o.sent.blobs) = .accept pub prof →
      CaseDraw V θ o kd x p c)
    (hrec : RecordDrawn dj o.sent) :
    DrawAt dj (Game.LiveDrive.drawOS (n := n) (.subset kd) n x) (proj o) ∨ RoundsBreak V θ o := by
  obtain ⟨p, c, e, pub, prof, s0, rest, hc, hacc, hl, ht⟩ := transcriptOf_spec h0
  obtain ⟨key, label, hch, hk, hlaw, hpop, hx⟩ := hcase p c e pub prof hc hacc
  obtain ⟨uses, wants, order, groups, coins, hin, hd | ⟨i, u, hu, hb⟩⟩ :=
    V.draw_at_rounds hR o hacc hch hk hl x hlaw hpop hx
  · left
    show Exec.recordDraw (proj o).record = some (dj _)
    rw [hd]
    exact hrec p c e pub prof s0 rest _ hc hacc hl ht
  · exact .inr ⟨p, c, e, pub, prof, key, label, uses, wants, order, groups, coins, i, u, hc, hacc, hch, hin, hu, hb⟩

/-! ## The record step from v2's refusals -/

/-- **`draw-law`, at acceptance**: at every case v2 accepts in `s`, the verifier's own law is the subset law of `kd`
over `n`, the record's registered population. -/
def DrawLawOfAccept (s : Sent) (n kd : ℕ) : Prop :=
  ∀ p c e pub prof, SampledCase s p c e →
    _root_.SampledProofs.OneStageZk.checkBlobs p c e (Core.Blobs.ofArray s.blobs) = .accept pub prof →
    _root_.SampledProofs.Draw.flockLaw p.base.verifier.law n = .ok (.subset kd) ∧
      _root_.SampledProofs.nat? (_root_.SampledProofs.getD c.record "population") = some n

/-- **`draw-file`, at acceptance**: at every case v2 accepts in `s`, session 0's record carries, byte for byte, the
canonical draw file (`UnitDraw.toJson`) of the verifier's draw: the subset law of `kd` over `n` at the units the
evidence's draw names. -/
def DrawFileOfAccept (s : Sent) (n kd : ℕ) : Prop :=
  ∀ p c e pub prof s0 rest t, SampledCase s p c e →
    _root_.SampledProofs.OneStageZk.checkBlobs p c e (Core.Blobs.ofArray s.blobs) = .accept pub prof →
    e.sessions.toList = s0 :: rest → Properties.SampledProofs.OneStageZk.SessionTranscript p e s.blobs s0 t →
    Exec.recordDraw t.record = some (_root_.Flock.Draw.UnitDraw.toJson
      { law := .subset kd, population := n, units := _root_.SampledProofs.OneStage.drawnOf e.base.draw })

/-- **`proved-units`, at acceptance**: at every case v2 accepts in `s`, the units the evidence's draw names, which are
its proved units, are ascending units of the population `n`. -/
def ProvedUnitsOfAccept (s : Sent) (n : ℕ) : Prop :=
  ∀ p c e pub prof, SampledCase s p c e →
    _root_.SampledProofs.OneStageZk.checkBlobs p c e (Core.Blobs.ofArray s.blobs) = .accept pub prof →
    (_root_.SampledProofs.OneStage.drawnOf e.base.draw).toList.Pairwise (· < ·) ∧
      ∀ u ∈ _root_.SampledProofs.OneStage.drawnOf e.base.draw, u < n

/-- **The units of a set of units an ascending array names are that array**, sorted. -/
theorem sort_filter_eq {n : ℕ} (U : Array ℕ) (hU : U.toList.Pairwise (· < ·)) (hn : ∀ x ∈ U, x < n) :
    (((Finset.univ.filter fun u : Fin n => u.val ∈ U).image Fin.val).sort (· ≤ ·)).toArray = U := by
  apply Array.toList_inj.1
  rw [List.toList_toArray]
  refine List.Perm.eq_of_pairwise' (r := (· ≤ ·)) (Finset.pairwise_sort _ _) (hU.imp le_of_lt) ?_
  refine List.perm_of_nodup_nodup_toFinset_eq (Finset.sort_nodup _ _) (hU.imp ne_of_lt) ?_
  rw [Finset.sort_toFinset]
  ext x
  simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and, List.mem_toFinset, Array.mem_toList_iff]
  constructor
  · rintro ⟨u, hu, rfl⟩
    exact hu
  · intro hx
    exact ⟨⟨x, hn x hx⟩, hx, rfl⟩

/-- **The verifier's draw is `drawOf` of the units its draw names**, where those are ascending units of the population. -/
theorem unitDraw_eq_drawOf {n : ℕ} (kd : ℕ) (U : Array ℕ) (hU : U.toList.Pairwise (· < ·)) (hn : ∀ x ∈ U, x < n) :
    ({ law := .subset kd, population := n, units := U } : _root_.Flock.Draw.UnitDraw) =
      FlockSoundness.Discharge.ZkLink.drawOf (.subset kd) n (Finset.univ.filter fun u : Fin n => u.val ∈ U) := by
  unfold FlockSoundness.Discharge.ZkLink.drawOf
  rw [sort_filter_eq U hU hn]

/-- **`RecordDrawn` from v2's refusals**: at `dj` the canonical draw file of the verifier's draw of each set of units,
session 0's record carries `dj` of the evidence's units wherever v2 accepts only records carrying the verifier's draw
file (`draw-file`) and evidence whose draw names ascending units of the population (`proved-units`). -/
theorem recordDrawn_of_accept {n kd : ℕ} {dj : Finset (Fin n) → Json} {s : Sent}
    (hdj : ∀ S, dj S = (FlockSoundness.Discharge.ZkLink.drawOf (.subset kd) n S).toJson)
    (hfile : DrawFileOfAccept s n kd) (hunits : ProvedUnitsOfAccept s n) : RecordDrawn dj s := by
  intro p c e pub prof s0 rest t hc hacc hl ht
  obtain ⟨hasc, hlt⟩ := hunits p c e pub prof hc hacc
  rw [hfile p c e pub prof s0 rest t hc hacc hl ht, hdj, ← unitDraw_eq_drawOf kd _ hasc hlt]

/-- **`Canon.DrawAt` at `proj o` from v2's refusals**: `drawAt_proj` with the law and population from `draw-law` and
the record step from `draw-file` and `proved-units` (`recordDrawn_of_accept`), at `dj` the canonical draw file. What
the play fixes is the registered key and the stream of `x` (`rounds_at_play`). -/
theorem drawAt_proj_of_accept (V : Vocab d α) (hR : Properties.Challenger.RoundsByDevice) {θ : ByteArray}
    (o : Outcome) {n kd : ℕ} {Lb : Fin 1 → ℕ} (x : (i : Fin 1) → Fin (Lb i) → Fin 256) {dj : Finset (Fin n) → Json}
    (hdj : ∀ S, dj S = (FlockSoundness.Discharge.ZkLink.drawOf (.subset kd) n S).toJson)
    (h0 : ∃ t, Sess0 o.sent t)
    (hplay : ∀ p c e pub prof, SampledCase o.sent p c e →
      _root_.SampledProofs.OneStageZk.checkBlobs p c e (Core.Blobs.ofArray o.sent.blobs) = .accept pub prof →
      ∃ (key : ChallengerKey) (label : String), p.base.challenger = some (key, label) ∧
        key.pub = (registeredKey θ).2 ∧
        FlockSoundness.Audit.Law.streamOf (x 0) = ((V.device θ).session o.log).released label c.digest)
    (drawLaw_of_accept : DrawLawOfAccept o.sent n kd) (drawFile_of_accept : DrawFileOfAccept o.sent n kd)
    (provedUnits_of_accept : ProvedUnitsOfAccept o.sent n) :
    DrawAt dj (Game.LiveDrive.drawOS (n := n) (.subset kd) n x) (proj o) ∨ RoundsBreak V θ o :=
  drawAt_proj V hR o x h0
    (fun p c e pub prof hc hacc => by
      obtain ⟨key, label, hch, hk, hx⟩ := hplay p c e pub prof hc hacc
      obtain ⟨hlaw, hpop⟩ := drawLaw_of_accept p c e pub prof hc hacc
      exact ⟨key, label, hch, hk, hlaw, hpop, hx⟩)
    (recordDrawn_of_accept hdj drawFile_of_accept provedUnits_of_accept)

end Verity.Canon
