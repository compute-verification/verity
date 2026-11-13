module

public import Proofs.Properties.Verity.SampledTwo.Steps

/-!
# The two-stage check accepts where each of its steps passes

`Steps` reads an accepted check as its steps. This is the other direction: the check
(`VStar.Recursive.SampledTwo.checkBlobsAtWith`, at the switch on) accepts wherever each step passes on the values the
step before it returned, so no step holds the evidence to anything the listed ones do not.

* `inputAt_of_steps`: `inputAt` returns where the interior registration's commitment reads, its sample's bytes are
  read, every session's verifier accepts, the verdicts are stamped and the interior's receipt names its place.
* `accepts_of_steps`: the check accepts where `inputAt` returned, neither stage's bytes run past one coin's
  (`stagesFit`), the run of record ran and its symbol is `accept`, the first stage's bytes are the ones it read
  (`sourceRead`), the first session shows none of the second stage's coins (`interiorUnshown`), the stages' sessions
  are in order (`stagesOrdered`), the second stage's law is computed (`secondLawOf`) and the interior audit's check at
  that law accepts (`secondCheck`).
* `accepts_iff_steps`: the check accepts exactly there.

That an honest prover's evidence passes each step is not shown here.
-/

@[expose] public section

namespace VStar.Recursive.SampledTwo

open Lean (Json)
open Spot

private theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error => cases h
  | ok a => exact ⟨a, rfl, h⟩

private theorem ok_bind {ε α β : Type} {x : Except ε α} {a : α} (f : α → Except ε β) (h : x = .ok a) :
    x >>= f = f a := by
  subst h
  rfl

/-- **`inputAt` returns where each of its steps does** (the converse of `inputAt_ok`). -/
theorem inputAt_of_steps {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence} {bs : Core.Blobs}
    {s : String} {c₂ : OneStage.Commitment} {ξ₂ : ByteArray} {r : OneStageZk.Run} {vs : Array (String × Json)}
    {received₂ : String} (h1 : Flock.canon e.record = .ok s) (h2 : OneStage.commitmentOf s.toUTF8 = .ok c₂)
    (h3 : OneStageZk.coinsOf p.audit c₂ e.interior = .ok ξ₂)
    (h4 : OneStageZk.membersRun p.audit p.members c₂ e.interior e.statements bs = .ok r) (h5 : r.all = true)
    (h6 : stamped p e bs r.out = .ok vs)
    (h7 : Spot.getD e.interior.base.receipt "received" = Json.str received₂) :
    inputAt p c ξ e bs = .ok (c₂, ξ₂, r, inputOf p c ξ ξ₂ e received₂ vs) := by
  unfold inputAt
  first
    | (simp only [h1, h2, h3, h4, h5, h6, h7, bind, Except.bind, pure, Except.pure, Bool.not_true,
        Bool.false_eq_true, ↓reduceIte]
       first | done | rfl)
    | (rw [ok_bind _ h1, ok_bind _ h2, ok_bind _ h3, ok_bind _ h4]
       simp only [h5, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
       rw [ok_bind _ h6]
       simp only [h7]
       rfl)
    | (simp [h1, h2, h3, h4, h5, h6, h7, bind, Except.bind, pure, Except.pure]
       done)

/-- **Steps 1 to 5 return where each does**: `prepare`, at the switch on, returns the interior audit's verdict at the
second stage's law and the run of record. -/
theorem prepare_of_steps {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence} {bs : Core.Blobs}
    {c₂ : OneStage.Commitment} {ξ₂ : ByteArray} {r : OneStageZk.Run} {i : Spot.Run.Input} {run law : Json}
    (hi : inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i)) (hfit : stagesFit ξ ξ₂ = .ok ())
    (hr : Spot.Run.run i true = .ok run) (hs : sourceRead ξ run = .ok ()) (hu : interiorUnshown p e = .ok ())
    (ho : stagesOrdered p e = .ok ()) (hl : secondLawOf p run = .ok law) :
    prepare true p c ξ e bs =
      .ok (secondCheck p.audit.base.verifier p.audit.base.delta c₂ law ξ₂
        { e.interior.base with verdicts := r.out }, run) := by
  unfold prepare
  first
    | (simp only [hi, hfit, hr, hs, hu, ho, hl, bind, Except.bind, pure, Except.pure]
       first | done | rfl)
    | (rw [ok_bind _ hi]
       try dsimp only
       rw [ok_bind _ hfit, ok_bind _ hr, ok_bind _ hs, ok_bind _ hu, ok_bind _ ho, ok_bind _ hl]
       rfl)
    | (simp [hi, hfit, hr, hs, hu, ho, hl, bind, Except.bind, pure, Except.pure]
       done)

/-- **The two-stage check accepts where each of its steps passes** (module docstring), at the switch on. -/
theorem accepts_of_steps {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence} {bs : Core.Blobs}
    {c₂ : OneStage.Commitment} {ξ₂ : ByteArray} {r : OneStageZk.Run} {i : Spot.Run.Input} {run law : Json}
    {pub : OneStage.Public} {prof : Core.Integrity}
    (hi : inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i)) (hfit : stagesFit ξ ξ₂ = .ok ())
    (hr : Spot.Run.run i true = .ok run) (hs : sourceRead ξ run = .ok ()) (hu : interiorUnshown p e = .ok ())
    (ho : stagesOrdered p e = .ok ()) (hl : secondLawOf p run = .ok law)
    (hj : Spot.jeq (Spot.getD run "symbol") (Json.str "accept") = true)
    (htwo : secondCheck p.audit.base.verifier p.audit.base.delta c₂ law ξ₂
      { e.interior.base with verdicts := r.out } = .accept pub prof) :
    checkBlobsAtWith true p c ξ e bs = .accept pub prof := by
  have hp := prepare_of_steps hi hfit hr hs hu ho hl
  unfold checkBlobsAtWith
  first
    | (rw [hp]
       simp [finishWith, htwo, hj]
       done)
    | (rw [hp]
       dsimp only
       unfold finishWith
       rw [htwo]
       simp [hj]
       done)
    | (simp [hp, finishWith, htwo, hj]
       done)

/-- **The two-stage check accepts exactly where each of its steps passes**, at the switch on: `accepts_of_steps` and
its converse, the steps on the values the step before returned. -/
theorem accepts_iff_steps {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence} {bs : Core.Blobs}
    {pub : OneStage.Public} {prof : Core.Integrity} :
    checkBlobsAtWith true p c ξ e bs = .accept pub prof ↔
      ∃ c₂ ξ₂ r i run law, inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) ∧ stagesFit ξ ξ₂ = .ok () ∧
        Spot.Run.run i true = .ok run ∧ sourceRead ξ run = .ok () ∧ interiorUnshown p e = .ok () ∧
        stagesOrdered p e = .ok () ∧ secondLawOf p run = .ok law ∧
        Spot.jeq (Spot.getD run "symbol") (Json.str "accept") = true ∧
        secondCheck p.audit.base.verifier p.audit.base.delta c₂ law ξ₂
          { e.interior.base with verdicts := r.out } = .accept pub prof := by
  constructor
  · intro h
    unfold checkBlobsAtWith at h
    cases hp : prepare true p c ξ e bs with
    | error err => simp [hp] at h
    | ok v =>
      obtain ⟨two, run⟩ := v
      simp only [hp] at h
      obtain ⟨-, hj, htwo⟩ := finishWith_accept h
      unfold prepare at hp
      obtain ⟨⟨c₂, ξ₂, r, i⟩, hi, hp⟩ := bind_ok hp
      obtain ⟨u₀, hfit, hp⟩ := bind_ok hp
      obtain ⟨run', hr, hp⟩ := bind_ok hp
      obtain ⟨u, hs, hp⟩ := bind_ok hp
      obtain ⟨u', hu, hp⟩ := bind_ok hp
      obtain ⟨u'', ho, hp⟩ := bind_ok hp
      obtain ⟨law, hl, hp⟩ := bind_ok hp
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at hp
      obtain ⟨rfl, rfl⟩ := hp
      exact ⟨c₂, ξ₂, r, i, run', law, hi, hfit, hr, hs, hu, ho, hl, hj, htwo⟩
  · rintro ⟨c₂, ξ₂, r, i, run, law, hi, hfit, hr, hs, hu, ho, hl, hj, htwo⟩
    exact accepts_of_steps hi hfit hr hs hu ho hl hj htwo

end VStar.Recursive.SampledTwo
