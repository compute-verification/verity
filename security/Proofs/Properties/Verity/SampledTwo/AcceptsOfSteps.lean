module

public import Proofs.Properties.Verity.SampledTwo.Steps

/-!
# The two-stage check accepts where each of its steps passes

`Steps` reads an accepted check as its steps. This is the other direction: the check
(`VStar.Recursive.SampledTwo.checkBlobsAtWith`, at the switch on) accepts wherever each step passes on the values the
step before it returned, so no step holds the evidence to anything the listed ones do not.

* `readBack_of`: `readBack` returns a record where the verifier's parser reads it from the record's canonical text
  and it prints back to that text.
* `firstTreeless_of_lines`: `firstTreeless` passes where the first session's log reads as lines none of which is a
  coin tree's.
* `digestIs_of`: `digestIs` passes where the record's digest is the one named.
* `repStreamsOrdered_of`: `repStreamsOrdered` passes where the interior audit reads its session's coin trees and
  the log opens each one's rounds stream by stream.
* `inputAt_of_steps`: `inputAt` returns where the interior registration's commitment reads, its sample's bytes are
  read, every session's verifier accepts, the verdicts are stamped, the interior's receipt names its place and each
  registration's record reads back (`readBack`), the boundary's with the commitment's digest.
* `accepts_of_steps`: the check accepts where every second-stage session's statement is under `2^64` bytes
  (`stmtsBytesFit`), `inputAt` returned, neither stage's bytes run past one coin's
  (`stagesFit`), the run of record ran and its symbol is `accept`, the first stage's bytes are the ones it read
  (`sourceRead`), the first session shows none of the second stage's coins (`interiorUnshown`), the stages' sessions
  are in order (`stagesOrdered`), the first session's log holds no coin tree's line (`firstTreeless`), the second
  session's log opens each coin tree's rounds stream by stream (`repStreamsOrdered`), the second stage's law is
  computed (`secondLawOf`) and the interior audit's check at that sampling design accepts (`secondCheck`).
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

/-- **`readBack` returns the record the verifier's parser reads from its canonical text**, where that record prints
back to the text (the converse of `readBack_ok`). -/
theorem readBack_of {record j : Json} {s : String} (h1 : Flock.canon record = .ok s)
    (h2 : Flock.JsonParse.parse s = .ok j) (h3 : Flock.canon j = .ok s) : readBack record = .ok j := by
  unfold readBack
  rw [ok_bind _ h1, ok_bind _ h2, ok_bind _ h3]
  simp only [beq_self_eq_true, ↓reduceIte]
  rfl

/-- **`firstTreeless` passes where no line of the first session's log is a coin tree's** (the converse of
`firstTreeless_lines`, on one window or two). -/
theorem firstTreeless_of_lines {p : Params} {e : Evidence} {ls : List String} (hl : firstLines e = .ok ls)
    (hno : ∀ l ∈ ls, Spot.Challenger.treeLine l = false) : firstTreeless p e = .ok () := by
  have hany : treeShown ls = false := by
    unfold treeShown
    exact List.any_eq_false.2 fun l hl' => by simp [hno l hl']
  unfold firstTreeless
  split
  · split
    · rfl
    · simp only [hl, hany, bind, Except.bind, Bool.false_eq_true, ↓reduceIte]
  · rfl

/-- **`digestIs` passes where the record's digest is the one named** (the converse of `digestIs_ok`). -/
theorem digestIs_of {j : Json} {d : ByteArray} (h : Registration.digest j = .ok d) : digestIs j d = .ok () := by
  unfold digestIs
  rw [ok_bind _ h]
  first
    | (rw [if_pos rfl]
       rfl)
    | (simp only [↓reduceIte]
       rfl)
    | (simp; done)

/-- **`interiorDigestIs` passes where both records' digests are the one named** (the converse of
`interiorDigestIs_ok`). -/
theorem interiorDigestIs_of {record j : Json} {d : ByteArray} (h : Registration.digest record = .ok d)
    (h' : Registration.digest j = .ok d) : interiorDigestIs record j d = .ok () := by
  unfold interiorDigestIs
  rw [ok_bind _ h, ok_bind _ h']
  first
    | (rw [if_pos ⟨rfl, rfl⟩]
       rfl)
    | (simp only [and_self, ↓reduceIte]
       rfl)
    | (simp; done)

/-- **`repStreamsOrdered` passes where the log opens each coin tree's rounds stream by stream** (the converse of
`repStreamsOrdered_trees`). -/
theorem repStreamsOrdered_of {p : Params} {e : Evidence} {ts : Array Challenger.LoggedTree}
    (ht : OneStageZk.treesOf p.audit.base e.interior.base = .ok ts)
    (h : ∀ t ∈ ts, Challenger.repStreamOrder t = true) : repStreamsOrdered p e = .ok () := by
  unfold repStreamsOrdered
  rw [ok_bind _ ht]
  unfold treesOrdered
  simp only [Array.all_eq_true'.2 h, ↓reduceIte]

/-- **`inputAt` returns where each of its steps does** (the converse of `inputAt_ok` and `inputAt_interior`). -/
theorem inputAt_of_steps {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence} {bs : Core.Blobs}
    {s : String} {c₂ : OneStage.Commitment} {ξ₂ : ByteArray} {r : OneStageZk.Run} {vs : Array (String × Json)}
    {received₂ : String} {j₁ j₂ : Json} (h1 : Flock.canon e.record = .ok s)
    (h2 : OneStage.commitmentOf s.toUTF8 = .ok c₂) (h3 : OneStageZk.coinsOf p.audit c₂ e.interior = .ok ξ₂)
    (h4 : OneStageZk.membersRun p.audit p.members c₂ e.interior e.statements bs = .ok r) (h5 : r.all = true)
    (h6 : stamped p e bs r.out = .ok vs)
    (h7 : Spot.getD e.interior.base.receipt "received" = Json.str received₂)
    (h8 : readBack c.record = .ok j₁) (h9 : readBack e.record = .ok j₂)
    (h10 : Registration.digest j₁ = .ok c.digest) (h11 : Registration.digest e.record = .ok c₂.digest)
    (h12 : Registration.digest j₂ = .ok c₂.digest) :
    inputAt p c ξ e bs =
      .ok (c₂, ξ₂, r,
        inputOf p { c with record := j₁ } ξ ξ₂ { e with record := j₂, stmts := stmtsOwn p e bs } received₂ vs) := by
  have hd := digestIs_of h10
  have hd₂ := interiorDigestIs_of h11 h12
  unfold inputAt
  first
    | (simp only [h1, h2, h3, h4, h5, h6, h7, h8, h9, hd, hd₂, bind, Except.bind, pure, Except.pure, Bool.not_true,
        Bool.false_eq_true, ↓reduceIte]
       first | done | rfl)
    | (rw [ok_bind _ h1, ok_bind _ h2, ok_bind _ h3, ok_bind _ h4]
       simp only [h5, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
       rw [ok_bind _ h6]
       simp only [h7]
       rw [ok_bind _ h8, ok_bind _ hd, ok_bind _ h9, ok_bind _ hd₂]
       rfl)
    | (simp [h1, h2, h3, h4, h5, h6, h7, h8, h9, hd, hd₂, bind, Except.bind, pure, Except.pure]
       done)

/-- **Steps 0 to 5 return where each does**: `prepare`, at the switch on, returns the interior audit's verdict at the
second stage's law and the run of record. -/
theorem prepare_of_steps {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence} {bs : Core.Blobs}
    {c₂ : OneStage.Commitment} {ξ₂ : ByteArray} {r : OneStageZk.Run} {i : Spot.Run.Input} {run law : Json}
    (hb : stmtsBytesFit p e bs = .ok ()) (hss : stmtsSessions p e bs = .ok ())
    (hi : inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i)) (hfit : stagesFit ξ ξ₂ = .ok ())
    (hr : Spot.Run.run i true = .ok run) (hs : sourceRead ξ run = .ok ()) (hu : interiorUnshown p e = .ok ())
    (ho : stagesOrdered p e = .ok ()) (ht : firstTreeless p e = .ok ()) (hq : repStreamsOrdered p e = .ok ())
    (hl : secondLawOf p run = .ok law) :
    prepare true p c ξ e bs =
      .ok (secondCheck p.audit.base.verifier p.audit.base.delta c₂ law ξ₂
        { e.interior.base with verdicts := r.out }, run) := by
  unfold prepare
  simp only [hb, hss]
  unfold prepareRun
  first
    | (simp only [hi, hfit, hr, hs, hu, ho, ht, hq, hl, bind, Except.bind, pure, Except.pure]
       first | done | rfl)
    | (rw [ok_bind _ hi]
       try dsimp only
       rw [ok_bind _ hfit, ok_bind _ hr, ok_bind _ hs, ok_bind _ hu, ok_bind _ ho, ok_bind _ ht, ok_bind _ hq,
         ok_bind _ hl]
       rfl)
    | (simp [hi, hfit, hr, hs, hu, ho, ht, hq, hl, bind, Except.bind, pure, Except.pure]
       done)

/-- **The two-stage check accepts where each of its steps passes** (module docstring), at the switch on. -/
theorem accepts_of_steps {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence} {bs : Core.Blobs}
    {c₂ : OneStage.Commitment} {ξ₂ : ByteArray} {r : OneStageZk.Run} {i : Spot.Run.Input} {run law : Json}
    {pub : OneStage.Public} {prof : Core.Integrity}
    (hb : stmtsBytesFit p e bs = .ok ()) (hss : stmtsSessions p e bs = .ok ())
    (hi : inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i)) (hfit : stagesFit ξ ξ₂ = .ok ())
    (hr : Spot.Run.run i true = .ok run) (hs : sourceRead ξ run = .ok ()) (hu : interiorUnshown p e = .ok ())
    (ho : stagesOrdered p e = .ok ()) (ht : firstTreeless p e = .ok ()) (hq : repStreamsOrdered p e = .ok ())
    (hl : secondLawOf p run = .ok law)
    (hj : Spot.jeq (Spot.getD run "symbol") (Json.str "accept") = true)
    (htwo : secondCheck p.audit.base.verifier p.audit.base.delta c₂ law ξ₂
      { e.interior.base with verdicts := r.out } = .accept pub prof) :
    checkBlobsAtWith true p c ξ e bs = .accept pub prof := by
  have hp := prepare_of_steps hb hss hi hfit hr hs hu ho ht hq hl
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
      stmtsBytesFit p e bs = .ok () ∧ stmtsSessions p e bs = .ok () ∧
      ∃ c₂ ξ₂ r i run law, inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) ∧ stagesFit ξ ξ₂ = .ok () ∧
        Spot.Run.run i true = .ok run ∧ sourceRead ξ run = .ok () ∧ interiorUnshown p e = .ok () ∧
        stagesOrdered p e = .ok () ∧ firstTreeless p e = .ok () ∧ repStreamsOrdered p e = .ok () ∧
        secondLawOf p run = .ok law ∧
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
      have hss := prepare_sessions hp
      obtain ⟨hp, hb⟩ := prepare_ok hp
      unfold prepareRun at hp
      obtain ⟨⟨c₂, ξ₂, r, i⟩, hi, hp⟩ := bind_ok hp
      obtain ⟨u₀, hfit, hp⟩ := bind_ok hp
      obtain ⟨run', hr, hp⟩ := bind_ok hp
      obtain ⟨u, hs, hp⟩ := bind_ok hp
      obtain ⟨u', hu, hp⟩ := bind_ok hp
      obtain ⟨u'', ho, hp⟩ := bind_ok hp
      obtain ⟨u₃, ht, hp⟩ := bind_ok hp
      obtain ⟨u₄, hq, hp⟩ := bind_ok hp
      obtain ⟨law, hl, hp⟩ := bind_ok hp
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at hp
      obtain ⟨rfl, rfl⟩ := hp
      exact ⟨hb, hss, c₂, ξ₂, r, i, run', law, hi, hfit, hr, hs, hu, ho, ht, hq, hl, hj, htwo⟩
  · rintro ⟨hb, hss, c₂, ξ₂, r, i, run, law, hi, hfit, hr, hs, hu, ho, ht, hq, hl, hj, htwo⟩
    exact accepts_of_steps hb hss hi hfit hr hs hu ho ht hq hl hj htwo

end VStar.Recursive.SampledTwo
