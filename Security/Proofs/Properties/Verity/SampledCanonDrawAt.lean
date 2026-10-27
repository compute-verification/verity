import Proofs.Properties.Verity.SampledProj
import Proofs.Properties.Verity.SampledCanonDraw

/-!
# The record's draw at `proj`

`Canon.proj o` is session 0's transcript of what the prover sent (`Canon.transcriptOf`, `Canon.Sess0`), at an accepting
case `proj` picks by choice. At every accepting case, `Vocab.draw_at_accept` says the draw `drawOS` makes from the
play's release of the draw's family is the units the evidence's draw names, or the finder outputs a break. The record
step (`Canon.RecordDrawn`) is that session 0's record carries `dj` of those units. Together they give `Canon.DrawAt` at
`proj o` itself, at whichever case `proj` picked (`Canon.drawAt_proj`).

What `RecordDrawn` takes from the verifier's code, at session 0 of an accepting case:

* flock-verify's setup at the record's draw decodes it (`Draw.UnitDraw.ofJson`) to a subset draw over the registered
  population with no closure (`HmRow.drawnWith`, `Stmt.setupTables`);
* the record's `unit_draw.units` are, through the public file's `units.indices`, the units `run` proves for the session
  (`OneStageZk.provedUnits`, `OneStageZk.push_ok`), and those are the evidence's `proved_units` when session 0 is the
  evidence's only session;
* v1's record checks make the evidence's `proved_units` the units its draw names (`Record.checks`, `Draw.checkProvedFull`);
* `dj S` is the one draw file, up to `Flock.norm`, that decodes to `drawOf (.subset kd) n S` (the canonical encoding
  `DrawFileZK` states one direction of).
-/

open Lean (Json)

namespace Verity.Canon

open _root_.SampledProofs.Challenger

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

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

/-- **`Canon.DrawAt` at `proj o`**: where session 0 has a transcript, every accepting case's draw is fixed by the play
(`hcase`: the law and population, `hlaw` and `hpop` of `draw_at_accept`, by registration; the stream of `x`, its `hx`,
by the device's release, `rounds_at_play`, which also fixes `Lb 0` as the release's size) and session 0's record
carries the draw of the evidence's units (`RecordDrawn`), the record at `proj o` carries the draw file of the draw
`drawOS` makes from `x`; or, at the case `proj` picked, the finder outputs a break from one of `roundsWhy`'s uses. -/
theorem drawAt_proj (V : Vocab d α) (hR : Properties.Challenger.RoundsByDevice) {θ : ByteArray} (o : Outcome)
    {n kd : ℕ} {Lb : Fin 1 → ℕ} (x : (i : Fin 1) → Fin (Lb i) → Fin 256) {dj : Finset (Fin n) → Json}
    (h0 : ∃ t, Sess0 o.sent t)
    (hcase : ∀ p c e pub prof, SampledCase o.sent p c e →
      _root_.SampledProofs.OneStageZk.checkBlobs p c e (Core.Blobs.ofArray o.sent.blobs) = .accept pub prof →
      CaseDraw V θ o kd x p c)
    (hrec : RecordDrawn dj o.sent) :
    DrawAt dj (Game.LiveDrive.drawOS (n := n) (.subset kd) n x) (proj o) ∨
      ∃ p c e pub prof (key : ChallengerKey) (label : String) (us : List Use) (segs : Array Json) (i : ℕ) (u : Use),
        SampledCase o.sent p c e ∧
        _root_.SampledProofs.OneStageZk.checkBlobs p c e (Core.Blobs.ofArray o.sent.blobs) = .accept pub prof ∧
        p.base.challenger = some (key, label) ∧ us[i]? = some u ∧
        Properties.Challenger.Break (V.deviceOf θ o) key u.label (Properties.Challenger.useCoins
          (Json.mkObj [("ledger", _root_.SampledProofs.getD e.base.coins "ledger"),
            ("main", _root_.SampledProofs.getD e.base.coins "main"),
            ("pins", _root_.SampledProofs.getD e.base.coins "pins"), ("segments", Json.arr segs),
            ("session", _root_.SampledProofs.getD e.base.coins "session")]) i) := by
  obtain ⟨p, c, e, pub, prof, s0, rest, hc, hacc, hl, ht⟩ := transcriptOf_spec h0
  obtain ⟨key, label, hch, hk, hlaw, hpop, hx⟩ := hcase p c e pub prof hc hacc
  obtain ⟨us, segs, hd | ⟨i, u, hu, hb⟩⟩ := V.draw_at_accept hR o hacc hch hk x hlaw hpop hx
  · left
    show Exec.recordDraw (proj o).record = some (dj _)
    rw [hd]
    exact hrec p c e pub prof s0 rest _ hc hacc hl ht
  · exact .inr ⟨p, c, e, pub, prof, key, label, us, segs, i, u, hc, hacc, hch, hu, hb⟩

end Verity.Canon
