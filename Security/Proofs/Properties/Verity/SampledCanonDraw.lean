import Proofs.Properties.Verity.SampledCanon
import Proofs.Flock.Claims.Complete.Json.Record
import Proofs.Flock.Claims.Sampling.Select.OS

/-!
# The draw's coins, read in the play

v2's acceptance runs v1's check at the draw's coins (`OneStage.checkLive`): it reads them through the challenger from
the evidence's coins (`OneStage.coinsOf`, `Challenger.drawCoins`) and refuses unless they draw the evidence's draw
(`OneStage.coinsReason`, `OneStage.checkLive_draw`). It also runs `roundsWhy`, whose first use is the draw's: its
family, released after the registration's digest, its coins read from a record that carries the evidence's coins with
the draw's segment first (`OneStageZk.roundsWhy_draw`, `OneStageZk.DrawRecord`). `drawCoins` reads a record only
through its keys and five fields (`Challenger.drawCoins_congr`), so the two reads are one (`OneStageZk.draw_src`), and
`RoundsByDevice` at the play's device names it: what the play's session released of the draw's family after the
digest (`Vocab.rounds_at_play`).

Under a subset law the coins draw the units `Draw.drawUnits` draws from them, which are `Audit.Law.runOS`'s on the
bytes they are (`drawUnits_subset_runOS`), so the draw `Game.LiveDrive.drawOS` makes from bytes whose stream is the
session's release is the evidence's draw (`Vocab.draw_at_play`, `Vocab.draw_at_accept`), or the finder outputs a break
from one use's record.
-/

open Lean (Json)

namespace SampledProofs.Challenger

/-- `drawCoins` reads only a record of exactly its five keys. -/
theorem keys_of_drawCoins {key : ChallengerKey} {w l : String} {a : ByteArray} {X : Json} {ξ : ByteArray}
    (h : drawCoins key w l a X = .ok ξ) : keys? X = some #["ledger", "main", "pins", "segment", "session"] := by
  unfold drawCoins at h
  cases hb : (keys? X == some #["ledger", "main", "pins", "segment", "session"])
  · simp [hb, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  · exact beq_iff_eq.1 hb

/-- **`drawCoins` reads a record through its keys and five fields**: records that agree there read alike. -/
theorem drawCoins_congr {key : ChallengerKey} {w l : String} {a : ByteArray} {X Y : Json}
    (hk : keys? X = keys? Y) (h1 : getD X "ledger" = getD Y "ledger") (h2 : getD X "main" = getD Y "main")
    (h3 : getD X "pins" = getD Y "pins") (h4 : getD X "segment" = getD Y "segment")
    (h5 : getD X "session" = getD Y "session") :
    drawCoins key w l a X = drawCoins key w l a Y := by
  unfold drawCoins
  rw [hk, h1, h2, h3, h4, h5]

theorem mapM_first {α β ε : Type} {f : α → Except ε β} {x : α} {l : List α} {ys : List β}
    (h : l.mapM f = .ok ys) (hx : l[0]? = some x) : ∃ y, ys[0]? = some y ∧ f x = .ok y := by
  cases l with
  | nil => cases hx
  | cons a l =>
    simp only [List.getElem?_cons_zero, Option.some.injEq] at hx
    subst hx
    rw [List.mapM_cons] at h
    cases ha : f a with
    | error e => rw [ha] at h; cases h
    | ok y =>
      rw [ha] at h
      cases hl : l.mapM f with
      | error e => rw [hl] at h; cases h
      | ok zs => rw [hl] at h; cases h; exact ⟨y, rfl, rfl⟩

theorem first_of_mapM {key : ChallengerKey} {w : String} {us : List Use} {coins : Json} {u : Use} {ls : Array Json}
    {v : List ByteArray} {F : Use × Json → Except String ByteArray}
    (hseg : getD coins "segments" = .arr ls) (hsz : (ls.size == us.length) = true)
    (hm : List.mapM F (us.zip ls.toList) = .ok v) (hu : us[0]? = some u)
    (hF : ∀ x r, F x = .ok r → drawCoins key w x.1.label x.1.after (Json.mkObj [("ledger", getD coins "ledger"),
      ("main", getD coins "main"), ("pins", getD coins "pins"), ("segment", x.2), ("session", getD coins "session")]) =
        .ok r) :
    ∃ segs s ξ, getD coins "segments" = .arr segs ∧ segs[0]? = some s ∧ v[0]? = some ξ ∧
      drawCoins key w u.label u.after (Json.mkObj [("ledger", getD coins "ledger"), ("main", getD coins "main"),
        ("pins", getD coins "pins"), ("segment", s), ("session", getD coins "session")]) = .ok ξ := by
  have hsz' : ls.size = us.length := beq_iff_eq.1 hsz
  cases hl : ls.toList with
  | nil =>
    have : us.length = 0 := by rw [← hsz', ← Array.length_toList, hl]; rfl
    rw [List.length_eq_zero_iff.1 this] at hu
    cases hu
  | cons s rest =>
    cases us with
    | nil => cases hu
    | cons u' us' =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hu
      subst hu
      rw [hl, List.zip_cons_cons] at hm
      obtain ⟨y, hy, hfy⟩ := mapM_first (x := (u', s)) hm (List.getElem?_cons_zero)
      refine ⟨ls, s, y, hseg, ?_, hy, hF (u', s) y hfy⟩
      rw [← Array.getElem?_toList, hl]
      rfl

/-- **The first use's coins, as `roundCoins` reads them**: `drawCoins` on the record of `coins` with its first segment
as the segment. -/
theorem roundCoins_first {key : ChallengerKey} {w : String} {us : List Use} {order : List (String × String)}
    {groups : List (List String)} {coins : Json} {ξs : List ByteArray} {u : Use}
    (h : roundCoins key w us order groups coins = .ok ξs) (hu : us[0]? = some u) :
    ∃ segs s ξ, getD coins "segments" = .arr segs ∧ segs[0]? = some s ∧ ξs[0]? = some ξ ∧
      drawCoins key w u.label u.after (Json.mkObj [("ledger", getD coins "ledger"), ("main", getD coins "main"),
        ("pins", getD coins "pins"), ("segment", s), ("session", getD coins "session")]) = .ok ξ := by
  unfold roundCoins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases (‹(throw _ : Except String _) = Except.ok _›); done) | skip
  all_goals
    cases h
    refine first_of_mapM ‹_› ‹_› ‹_› hu fun x r hr => ?_
    try dsimp only at hr
    split at hr
    · cases hr
    · split at hr
      · cases hr; assumption
      · split at hr <;> first | (cases hr; done) | (cases (‹(throw _ : Except String _) = Except.ok _›))

end SampledProofs.Challenger

namespace SampledProofs.OneStage

/-- The units a draw object names, as `coinsReason` reads them. -/
def drawnOf (draw : Json) : Array Nat := match getD draw "units" with | .arr us => us.filterMap nat? | _ => #[]

/-- **Where `coinsReason` names no reason, the coins draw the draw's units**: under the verifier's law as derived. -/
theorem coinsReason_units {v : Registration.Verifier} {law rec_ draw : Json} {src : ByteArray} {n : Nat}
    (h : coinsReason v law rec_ draw src = none) (hpop : nat? (getD rec_ "population") = some n) :
    ∃ derived fl read, Draw.deriveLaw v.program v.query v.law v.workTable = .ok derived ∧
      Draw.flockLaw law n = .ok fl ∧ Draw.drawUnits (derived.getD fl) n src = .ok (drawnOf draw, read) := by
  unfold coinsReason at h
  rw [hpop] at h
  dsimp only at h
  cases hd : drawnFrom v law n src with
  | error why => rw [hd] at h; cases h
  | ok ur =>
    obtain ⟨units, read⟩ := ur
    rw [hd] at h
    dsimp only at h
    have hu : units = drawnOf draw := by
      unfold drawnOf
      cases hdu : getD draw "units" <;> rw [hdu] at h <;> dsimp only at h <;>
        (split at h
         · cases h
         · rename_i hb; simpa using hb)
    subst hu
    unfold drawnFrom at hd
    simp only [bind, Except.bind] at hd
    cases hdl : Draw.deriveLaw v.program v.query v.law v.workTable with
    | error e => rw [hdl] at hd; cases hd
    | ok derived =>
      rw [hdl] at hd
      cases hfl : Draw.flockLaw law n with
      | error e => rw [hfl] at hd; cases hd
      | ok fl => rw [hfl] at hd; exact ⟨derived, fl, read, rfl, rfl, hd⟩

/-- `check` accepts only where the verifier's law derives. -/
theorem check_derive {p : Params} {c : Commitment} {e : Evidence} {pub : Public} {prof : Core.Integrity}
    (h : check p c e = .accept pub prof) :
    ∃ law, Draw.derive p.verifier.program p.verifier.query p.verifier.law p.verifier.workTable = .ok law := by
  unfold check at h
  cases hd : Draw.derive p.verifier.program p.verifier.query p.verifier.law p.verifier.workTable with
  | ok law => exact ⟨law, rfl⟩
  | error err =>
    exfalso
    simp only [hd] at h
    split at h
    · cases h
    · cases h

/-- **v1 accepts at the challenger's coins for the draw**: with a challenger and a window, `checkLive` accepts only
where `drawCoins` reads `src` from the evidence's coins, the law derives, and `src` draws the evidence's draw. -/
theorem checkLive_draw {p : Params} {c : Commitment} {e : Evidence} {pub : Public} {prof : Core.Integrity}
    {key : Challenger.ChallengerKey} {label w : String} (h : checkLive p c e = .accept pub prof)
    (hc : p.challenger = some (key, label)) (hw : p.verifier.windowRoot = some w) :
    ∃ src law, Challenger.drawCoins key w label c.digest e.coins = .ok src ∧
      Draw.derive p.verifier.program p.verifier.query p.verifier.law p.verifier.workTable = .ok law ∧
      coinsReason p.verifier law c.record e.draw src = none := by
  unfold checkLive coinsOf at h
  rw [hc, hw] at h
  dsimp only at h
  cases hd : Challenger.drawCoins key w label c.digest e.coins with
  | error why => rw [hd] at h; cases h
  | ok src =>
    rw [hd] at h
    change checkAt p c (some src) e = .accept pub prof at h
    unfold checkAt at h
    cases hcw : coinsWhy p c e src with
    | some why => simp only [Option.bind_some, hcw] at h; cases h
    | none =>
      simp only [Option.bind_some, hcw] at h
      obtain ⟨law, hl⟩ := check_derive h
      refine ⟨src, law, rfl, hl, ?_⟩
      unfold coinsWhy at hcw
      dsimp only at hcw
      rw [hl] at hcw
      exact hcw

end SampledProofs.OneStage

namespace SampledProofs.OneStageZk

/-- v2 accepts only where v1's check accepts on its evidence with the sessions' verdicts. -/
theorem checkBlobs_live {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {pub : OneStage.Public} {prof : Core.Integrity} (h : checkBlobs p c e bs = .accept pub prof) :
    ∃ vs, OneStage.checkLive p.base c { e.base with verdicts := vs } = .accept pub prof := by
  unfold checkBlobs at h
  split at h
  · cases h
  · rename_i r _
    split at h
    · rename_i pub' prof' hl
      split at h
      · split at h
        · cases h
        · cases h; exact ⟨r.out, hl⟩
      · cases h
    · rename_i hn
      exact absurd h (hn _ _)

/-- The loop's uses and segments only grow by appending. -/
def Heads (s s' : Array Challenger.Use × Array (Option (String × Array String)) × Array Json ×
    Array (String × String) × Array (List String)) : Prop :=
  (∃ t, s'.1 = s.1 ++ t) ∧ ∃ t, s'.2.2.1 = s.2.2.1 ++ t

theorem heads_refl (s) : Heads s s := ⟨⟨#[], by simp⟩, ⟨#[], by simp⟩⟩

theorem heads_trans (a b c) (h1 : Heads a b) (h2 : Heads b c) : Heads a c := by
  obtain ⟨⟨t1, e1⟩, ⟨t2, e2⟩⟩ := h1
  obtain ⟨⟨t3, e3⟩, ⟨t4, e4⟩⟩ := h2
  exact ⟨⟨t1 ++ t3, by rw [e3, e1, Array.append_assoc]⟩, ⟨t2 ++ t4, by rw [e4, e2, Array.append_assoc]⟩⟩

/-- **The draw's use, as `roundsWhy` runs it**: where it names no reason, the uses `roundCoins` reads begin with the
draw's (its family, after the registration's digest, a coin a segment opening), and their record carries the evidence's
ledger, main, pins and session, with the evidence's segment first. -/
theorem roundsWhy_draw {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    (h : roundsWhy p c e bs = none) :
    ∃ key label w opened uses order groups segs ξs, p.base.challenger = some (key, label) ∧
      p.base.verifier.windowRoot = some w ∧ getD (getD e.base.coins "segment") "openings" = .arr opened ∧
      uses[0]? = some ⟨label, c.digest, opened.size⟩ ∧ segs[0]? = some (getD e.base.coins "segment") ∧
      Challenger.roundCoins key w uses order groups (Json.mkObj [("ledger", getD e.base.coins "ledger"),
        ("main", getD e.base.coins "main"), ("pins", getD e.base.coins "pins"), ("segments", Json.arr segs),
        ("session", getD e.base.coins "session")]) = .ok ξs := by
  unfold roundsWhy at h
  split at h
  · rename_i key label w hc hw
    dsimp only at h
    split at h
    · rename_i hr
      simp only [bind, Except.bind, pure, Except.pure] at hr
      split at hr
      · rename_i opened hop
        split at hr
        · cases hr
        · rename_i v hfor
          split at hr
          · cases hr
          · rename_i ξs hrc
            obtain ⟨⟨t1, e1⟩, ⟨t2, e2⟩⟩ := loop_mono Heads heads_refl heads_trans _ (fun x s r hx => by
                repeat' split at hx
                all_goals first
                  | (cases hx; done)
                  | (rename_i heq; cases heq; done)
                  | (cases hx; exact heads_refl _)
                  | (cases hx; exact ⟨⟨_, rfl⟩, ⟨_, rfl⟩⟩)) _ _ _ hfor
            refine ⟨key, label, w, opened, v.1.toList, v.2.2.2.1.toList, v.2.2.2.2.toList, v.2.2.1, ξs, hc, hw, hop,
              ?_, ?_, hrc⟩
            · rw [e1]; simp
            · rw [e2, Array.getElem?_append_left (by simp)]; rfl
      · cases hr
    · cases h
  · cases h

theorem getD_mkObj {l : List (String × Json)} (hd : FlockComplete.JsonRT.Distinct l) {k : String} {v : Json}
    (hm : (k, v) ∈ l) : getD (Json.mkObj l) k = v := by
  simp only [getD, field?, FlockComplete.JsonRT.getObjVal?_mkObj hd hm, Except.toOption, Option.getD_some]

section
variable (a b c d f : Json)

theorem getD_use_ledger : getD (Json.mkObj [("ledger", a), ("main", b), ("pins", c), ("segment", d), ("session", f)])
    "ledger" = a := getD_mkObj (by simp) (by simp)
theorem getD_use_main : getD (Json.mkObj [("ledger", a), ("main", b), ("pins", c), ("segment", d), ("session", f)])
    "main" = b := getD_mkObj (by simp) (by simp)
theorem getD_use_pins : getD (Json.mkObj [("ledger", a), ("main", b), ("pins", c), ("segment", d), ("session", f)])
    "pins" = c := getD_mkObj (by simp) (by simp)
theorem getD_use_segment : getD (Json.mkObj [("ledger", a), ("main", b), ("pins", c), ("segment", d), ("session", f)])
    "segment" = d := getD_mkObj (by simp) (by simp)
theorem getD_use_session : getD (Json.mkObj [("ledger", a), ("main", b), ("pins", c), ("segment", d), ("session", f)])
    "session" = f := getD_mkObj (by simp) (by simp)

theorem getD_rounds_ledger : getD (Json.mkObj [("ledger", a), ("main", b), ("pins", c), ("segments", d),
    ("session", f)]) "ledger" = a := getD_mkObj (by simp) (by simp)
theorem getD_rounds_main : getD (Json.mkObj [("ledger", a), ("main", b), ("pins", c), ("segments", d),
    ("session", f)]) "main" = b := getD_mkObj (by simp) (by simp)
theorem getD_rounds_pins : getD (Json.mkObj [("ledger", a), ("main", b), ("pins", c), ("segments", d),
    ("session", f)]) "pins" = c := getD_mkObj (by simp) (by simp)
theorem getD_rounds_segments : getD (Json.mkObj [("ledger", a), ("main", b), ("pins", c), ("segments", d),
    ("session", f)]) "segments" = d := getD_mkObj (by simp) (by simp)
theorem getD_rounds_session : getD (Json.mkObj [("ledger", a), ("main", b), ("pins", c), ("segments", d),
    ("session", f)]) "session" = f := getD_mkObj (by simp) (by simp)
end

/-- The record `roundCoins` reads the draw's use from carries the evidence's coins `E`: its ledger, main, pins and
session, and its first segment `E`'s. -/
def DrawRecord (coins E : Json) : Prop :=
  getD coins "ledger" = getD E "ledger" ∧ getD coins "main" = getD E "main" ∧ getD coins "pins" = getD E "pins" ∧
    getD coins "session" = getD E "session" ∧
    ∀ segs, getD coins "segments" = .arr segs → segs[0]? = some (getD E "segment")

/-- The record `roundsWhy` hands `roundCoins` is a `DrawRecord` of the evidence's coins. -/
theorem drawRecord_rounds (E : Json) (segs : Array Json) (hs : segs[0]? = some (getD E "segment")) :
    DrawRecord (Json.mkObj [("ledger", getD E "ledger"), ("main", getD E "main"), ("pins", getD E "pins"),
      ("segments", Json.arr segs), ("session", getD E "session")]) E := by
  refine ⟨getD_rounds_ledger .., getD_rounds_main .., getD_rounds_pins .., getD_rounds_session .., fun s h => ?_⟩
  rw [getD_rounds_segments] at h
  cases h
  exact hs

/-- **The draw's coins are its use's**: where v2 accepts, v1's check read the draw's coins `src` through the challenger
(`coinsOf`), the coins draw the evidence's draw (`coinsReason`), and `src` is the coins `roundCoins` reads for its first
use, the draw's, from a record that carries the evidence's coins (`DrawRecord`). -/
theorem draw_src {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs} {pub : OneStage.Public}
    {prof : Core.Integrity} {key : Challenger.ChallengerKey} {label w : String} {us : List Challenger.Use}
    {order : List (String × String)} {groups : List (List String)} {coins : Json} {ξs : List ByteArray} {k : Nat}
    (hacc : checkBlobs p c e bs = .accept pub prof) (hc : p.base.challenger = some (key, label))
    (hw : p.base.verifier.windowRoot = some w) (hrc : Challenger.roundCoins key w us order groups coins = .ok ξs)
    (hu : us[0]? = some ⟨label, c.digest, k⟩) (hrec : DrawRecord coins e.base.coins) :
    ∃ src law, ξs[0]? = some src ∧
      Draw.derive p.base.verifier.program p.base.verifier.query p.base.verifier.law p.base.verifier.workTable =
        .ok law ∧ OneStage.coinsReason p.base.verifier law c.record e.base.draw src = none := by
  obtain ⟨vs, hl⟩ := checkBlobs_live hacc
  obtain ⟨src, law, hd, hder, hr⟩ := OneStage.checkLive_draw hl hc hw
  obtain ⟨segs, s, ξ, hsegs, hs, hξ, hdξ⟩ := Challenger.roundCoins_first hrc hu
  obtain ⟨hL, hM, hP, hS, hG⟩ := hrec
  have hs' := hG segs hsegs
  rw [hs] at hs'
  cases hs'
  have heq := Challenger.drawCoins_congr (key := key) (w := w) (l := label) (a := c.digest)
    (X := Json.mkObj [("ledger", getD coins "ledger"), ("main", getD coins "main"), ("pins", getD coins "pins"),
      ("segment", getD e.base.coins "segment"), ("session", getD coins "session")]) (Y := e.base.coins)
    (by rw [Challenger.keys_of_drawCoins hdξ, Challenger.keys_of_drawCoins hd])
    (by rw [getD_use_ledger, hL]) (by rw [getD_use_main, hM]) (by rw [getD_use_pins, hP])
    (by rw [getD_use_segment]) (by rw [getD_use_session, hS])
  have : ξ = src := by
    have h2 : Challenger.drawCoins key w label c.digest e.base.coins = .ok ξ := heq ▸ hdξ
    rw [hd] at h2
    cases h2
    rfl
  subst this
  exact ⟨ξ, law, hξ, hder, hr⟩

end SampledProofs.OneStageZk

namespace SampledProofs.OneStage

/-- **Under a subset law the coins draw the draw's units**: where `coinsReason` names no reason at the law the
verifier's derives to, a subset law of `kd` over the record's population `n`, `drawUnits` draws exactly the draw's
units from `src`. -/
theorem units_of_reason {v : Registration.Verifier} {law rec_ draw : Json} {src : ByteArray} {n kd : Nat}
    (h : coinsReason v law rec_ draw src = none)
    (hder : Draw.derive v.program v.query v.law v.workTable = .ok law)
    (hlaw : Draw.flockLaw v.law n = .ok (.subset kd)) (hpop : nat? (getD rec_ "population") = some n) :
    ∃ read, Draw.drawUnits (.subset kd) n src = .ok (drawnOf draw, read) := by
  obtain ⟨derived, fl, read, hdl, hfl, hdu⟩ := coinsReason_units h hpop
  have hnone : derived = none :=
    Properties.SampledProofs.SecurityProofs.deriveLaw_subset _ _ _ _ _
      (Properties.SampledProofs.SecurityProofs.flockLaw_subset _ _ _ hlaw).1 hdl
  subst hnone
  have hv : Draw.derive v.program v.query v.law v.workTable = .ok v.law := by
    unfold Draw.derive
    rw [hdl]
    try rfl
  rw [hv] at hder
  cases hder
  rw [hlaw] at hfl
  cases hfl
  exact ⟨read, hdu⟩

end SampledProofs.OneStage

namespace Verity.Game.LiveDrive

/-- **The draw the bytes `x` give is the units `drawUnits` draws from their stream**, under a subset law. -/
theorem drawOS_of_units {n kd : ℕ} {Lb : Fin 1 → ℕ} (x : (i : Fin 1) → Fin (Lb i) → Fin 256)
    {units : Array ℕ} {read : ℕ}
    (h : _root_.SampledProofs.Draw.drawUnits (.subset kd) n (FlockSoundness.Audit.Law.streamOf (x 0)) =
      .ok (units, read)) :
    drawOS (n := n) (.subset kd) n x = Finset.univ.filter fun u : Fin n => u.val ∈ units := by
  have hr := Properties.SampledProofs.SecurityProofs.drawUnits_subset_runOS kd n x
  rw [h] at hr
  cases hd : FlockSoundness.Audit.Law.runOS x (.subset kd) n with
  | none => rw [hd] at hr; first | cases hr | simp [Except.toOption] at hr
  | some d =>
    rw [hd] at hr
    have hu : d.units = units := by
      first | exact (Option.some.inj hr).symm | simpa [Except.toOption] using hr.symm
    subst hu
    show (match FlockSoundness.Audit.Law.runOS x (.subset kd) n with
      | some d => Finset.univ.filter fun u : Fin n => u.val ∈ d.units
      | none => Finset.univ) = _
    rw [hd] <;> rfl

end Verity.Game.LiveDrive

namespace Verity

open _root_.SampledProofs.Challenger

namespace Vocab

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- **The draw the play's releases give is the evidence's draw**: where v2 accepts under a challenger key that is the
play's registered one, at uses whose first is the draw's read from a record of the evidence's coins (`DrawRecord`),
and the verifier's law is a subset law of `kd` over the record's population `n`, the draw `drawOS` makes from bytes
`x` whose stream is what the play's session released of the draw's family after the registration's digest is the
units the evidence's draw names; or the finder outputs a break from one use's record (`RoundsByDevice`). -/
theorem draw_at_play (hR : Properties.Challenger.RoundsByDevice) {θ : ByteArray} (o : Outcome)
    {p : _root_.SampledProofs.OneStageZk.Params} {c : _root_.SampledProofs.OneStage.Commitment}
    {e : _root_.SampledProofs.OneStageZk.Evidence} {bs pub prof} {key : ChallengerKey} {label w : String}
    {us : List Use} {order : List (String × String)}
    {groups : List (List String)} {coins : Json} {ξs : List ByteArray} {k : ℕ}
    (hacc : _root_.SampledProofs.OneStageZk.checkBlobs p c e bs = .accept pub prof)
    (hc : p.base.challenger = some (key, label)) (hw : p.base.verifier.windowRoot = some w)
    (hk : key.pub = (registeredKey θ).2) (hrc : roundCoins key w us order groups coins = .ok ξs)
    (hu : us[0]? = some ⟨label, c.digest, k⟩) (hrec : _root_.SampledProofs.OneStageZk.DrawRecord coins e.base.coins)
    {n kd : ℕ} {Lb : Fin 1 → ℕ} (x : (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hlaw : _root_.SampledProofs.Draw.flockLaw p.base.verifier.law n = .ok (.subset kd))
    (hpop : _root_.SampledProofs.nat? (_root_.SampledProofs.getD c.record "population") = some n)
    (hx : FlockSoundness.Audit.Law.streamOf (x 0) = ((V.device θ).session o.log).released label c.digest) :
    Game.LiveDrive.drawOS (n := n) (.subset kd) n x =
        Finset.univ.filter (fun u : Fin n => u.val ∈ _root_.SampledProofs.OneStage.drawnOf e.base.draw) ∨
      ∃ (i : ℕ) (u : Use), us[i]? = some u ∧
        Properties.Challenger.Break (V.deviceOf θ o) key u.label (Properties.Challenger.useCoins coins i) := by
  rcases V.rounds_at_play hR o hk hrc with ⟨-, hrel, -⟩ | hb
  · left
    obtain ⟨src, law, hξ, hder, hr⟩ := _root_.SampledProofs.OneStageZk.draw_src hacc hc hw hrc hu hrec
    have h0 := hrel 0 _ hu
    rw [hξ] at h0
    obtain ⟨read, hdu⟩ := _root_.SampledProofs.OneStage.units_of_reason hr hder hlaw hpop
    have hs : FlockSoundness.Audit.Law.streamOf (x 0) = src := by rw [hx]; exact (Option.some.inj h0).symm
    rw [← hs] at hdu
    exact Game.LiveDrive.drawOS_of_units x hdu
  · exact .inr hb

/-- **`draw_at_play` from v2's acceptance alone**: at the uses and record `roundsWhy` runs (`roundsWhy_draw`), the draw
`drawOS` makes from bytes whose stream is the play's release of the draw's family after the registration's digest is
the evidence's draw, or the finder outputs a break from one of those uses' records. -/
theorem draw_at_accept (hR : Properties.Challenger.RoundsByDevice) {θ : ByteArray} (o : Outcome)
    {p : _root_.SampledProofs.OneStageZk.Params} {c : _root_.SampledProofs.OneStage.Commitment}
    {e : _root_.SampledProofs.OneStageZk.Evidence} {bs pub prof} {key : ChallengerKey} {label : String}
    (hacc : _root_.SampledProofs.OneStageZk.checkBlobs p c e bs = .accept pub prof)
    (hc : p.base.challenger = some (key, label)) (hk : key.pub = (registeredKey θ).2)
    {n kd : ℕ} {Lb : Fin 1 → ℕ} (x : (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hlaw : _root_.SampledProofs.Draw.flockLaw p.base.verifier.law n = .ok (.subset kd))
    (hpop : _root_.SampledProofs.nat? (_root_.SampledProofs.getD c.record "population") = some n)
    (hx : FlockSoundness.Audit.Law.streamOf (x 0) = ((V.device θ).session o.log).released label c.digest) :
    ∃ (us : List Use) (segs : Array Json),
      (Game.LiveDrive.drawOS (n := n) (.subset kd) n x =
          Finset.univ.filter (fun u : Fin n => u.val ∈ _root_.SampledProofs.OneStage.drawnOf e.base.draw) ∨
        ∃ (i : ℕ) (u : Use), us[i]? = some u ∧
          Properties.Challenger.Break (V.deviceOf θ o) key u.label (Properties.Challenger.useCoins
            (Json.mkObj [("ledger", _root_.SampledProofs.getD e.base.coins "ledger"),
              ("main", _root_.SampledProofs.getD e.base.coins "main"),
              ("pins", _root_.SampledProofs.getD e.base.coins "pins"), ("segments", Json.arr segs),
              ("session", _root_.SampledProofs.getD e.base.coins "session")]) i)) := by
  obtain ⟨key', label', w, opened, us, order, groups, segs, ξs, hc', hw, -, hu, hs, hrc⟩ :=
    _root_.SampledProofs.OneStageZk.roundsWhy_draw (_root_.SampledProofs.OneStageZk.roundsWhy_of_checkBlobs hacc)
  rw [hc] at hc'
  cases hc'
  exact ⟨us, segs, V.draw_at_play hR o hacc hc hw hk hrc hu
    (_root_.SampledProofs.OneStageZk.drawRecord_rounds _ segs hs) x hlaw hpop hx⟩

end Vocab

end Verity
