module

public import VStarRecursive.SampledTwo

/-!
# What an accepted two-stage check ran

The two-stage form's check (`VStar.Recursive.SampledTwo.checkBlobsAtWith`) accepts only through its steps. Where it
accepts at the switch `form` (`accepted_steps`):

* the switch is on;
* every second-stage session's statement is under `2^64` bytes (`stmtsBytesFit`, by `accepted_stmtsBytesFit`);
* the run of record's input is `inputAt`'s (`inputAt_ok`): the interior registration's commitment is its record's, its
  sample read the bytes `ξ₂` (`OneStageZk.coinsOf`), every session `OneStageZk.membersRun` ran accepted (`r.all`), and
  the input is `inputOf` at those sessions' verdicts stamped (`stamped`), each registration's record the one the
  verifier's parser reads from its canonical text (`readBack`, read by `readBack_ok`), the boundary's with the
  commitment's digest (`digestIs`, read by `digestIs_ok`), and the interior's, as given and as read back, with the
  interior commitment's (`interiorDigestIs`, read by `inputAt_interior`);
* neither stage's bytes run past one coin's (`stagesFit`, read as `ξ.size ≤ STAGE_BYTES` and `ξ₂.size ≤ STAGE_BYTES`
  by `accepted_fits`);
* `Spot.Run.run`, at the switch on, returned a run of record whose symbol is `accept`, and the first stage's bytes `ξ`
  run no further than what it read (`sourceRead`, read as `ξ.size ≤ read` by `sourceRead_le`), and on two windows the
  first session's log shows no second-stage coin after the interior's digest (`interiorUnshown`, read by
  `interiorUnshown_roots`), and the stages' sessions are in order on one ledger (`stagesOrdered`, by
  `accepted_ordered`, read by `orderReason_none` and `Ordered.stagesOrdered_order`), and on two windows no line of the
  first session's log is a coin tree's (`firstTreeless`, by `accepted_treeless`, read by `firstTreeless_lines`), and
  the second session's log opens each coin tree's rounds stream by stream (`repStreamsOrdered`, by
  `accepted_repOrdered`, read by `repStreamsOrdered_trees` and `repStreamOrder_before`: of two rounds the log shows
  opened of one tree, neither its last, the one opened first is of a stream the tree's spec lists earlier, or an
  earlier round of the same stream);
* the interior audit's verdict at its coins, over the sessions' verdicts, is the accept, with the check's public
  outputs and profile.

The pieces read an accepted play (`Verity.SampledTwo.AcceptsOn`) through these.
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

/-- **The last step accepts** only with the switch on, the run of record's symbol `accept` and the interior audit
accepting with the same public outputs and profile. -/
theorem finishWith_accept {form : Bool} {two : Core.Outcome OneStage.Public Core.Integrity} {run : Json}
    {pub : OneStage.Public} {prof : Core.Integrity} (h : finishWith form two run = .accept pub prof) :
    form = true ∧ Spot.jeq (Spot.getD run "symbol") (Json.str "accept") = true ∧ two = .accept pub prof := by
  unfold finishWith at h
  cases two with
  | reject why => cases h
  | undecided why => cases h
  | accept pub' prof' =>
    cases hj : Spot.jeq (Spot.getD run "symbol") (Json.str "accept") <;> cases form <;> simp_all

/-- **What `readBack` returned** is the value the verifier's parser reads from the record's canonical text, and it
prints back to that text. -/
theorem readBack_ok {record j : Json} (h : readBack record = .ok j) :
    ∃ s, Flock.canon record = .ok s ∧ Flock.JsonParse.parse s = .ok j ∧ Flock.canon j = .ok s := by
  unfold readBack at h
  obtain ⟨s, hs, h⟩ := bind_ok h
  obtain ⟨j', hj, h⟩ := bind_ok h
  obtain ⟨s', hs', h⟩ := bind_ok h
  cases hc : (s' == s)
  · rw [hc] at h
    cases h
  · rw [hc] at h
    simp only [↓reduceIte, pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hss : s' = s := by simpa using hc
    rw [hss] at hs'
    exact ⟨s, hs, hj, hs'⟩

/-- **What `digestIs` passed**: the record's digest is the one named. -/
theorem digestIs_ok {j : Json} {d : ByteArray} (h : digestIs j d = .ok ()) : Registration.digest j = .ok d := by
  unfold digestIs at h
  obtain ⟨d', hd, h⟩ := bind_ok h
  split at h
  · rename_i heq
    rw [hd, heq]
  · first
      | (cases h; done)
      | (simp only [throw, throwThe, MonadExceptOf.throw] at h
         cases h)
      | (simp at h; done)

/-- **What `interiorDigestIs` passed**: both records' digests are the one named. -/
theorem interiorDigestIs_ok {record j : Json} {d : ByteArray} (h : interiorDigestIs record j d = .ok ()) :
    Registration.digest record = .ok d ∧ Registration.digest j = .ok d := by
  unfold interiorDigestIs at h
  obtain ⟨d₁, hd₁, h⟩ := bind_ok h
  obtain ⟨d₂, hd₂, h⟩ := bind_ok h
  split at h
  · rename_i heq
    rw [hd₁, hd₂, heq.1, heq.2]
    exact ⟨rfl, rfl⟩
  · first
      | (cases h; done)
      | (simp only [throw, throwThe, MonadExceptOf.throw] at h
         cases h)
      | (simp at h; done)

/-- **What `inputAt` computed** where it returned: the run of record's input is `inputOf` at each registration's
record as `readBack` returned it, and the boundary's so returned has the commitment's digest (`digestIs`). -/
theorem inputAt_ok {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence} {bs : Core.Blobs}
    {c₂ : OneStage.Commitment} {ξ₂ : ByteArray} {r : OneStageZk.Run} {i : Spot.Run.Input}
    (h : inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i)) :
    (∃ s, Flock.canon e.record = .ok s ∧ OneStage.commitmentOf s.toUTF8 = .ok c₂) ∧
      OneStageZk.coinsOf p.audit c₂ e.interior = .ok ξ₂ ∧
      OneStageZk.membersRun p.audit p.members c₂ e.interior e.statements bs = .ok r ∧ r.all = true ∧
      ∃ vs received₂ j₁ j₂, stamped p e bs r.out = .ok vs ∧
        i = inputOf p { c with record := j₁ } ξ ξ₂ { e with record := j₂ } received₂ vs ∧
        readBack c.record = .ok j₁ ∧ readBack e.record = .ok j₂ ∧ Registration.digest j₁ = .ok c.digest := by
  unfold inputAt at h
  obtain ⟨s, h1, h⟩ := bind_ok h
  obtain ⟨c₂', h2, h⟩ := bind_ok h
  obtain ⟨ξ₂', h3, h⟩ := bind_ok h
  obtain ⟨r', h4, h⟩ := bind_ok h
  cases h5 : r'.all
  · simp [h5] at h
    cases h
  · simp only [h5, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at h
    obtain ⟨vs, h6, h⟩ := bind_ok h
    split at h
    · obtain ⟨j₁, hj₁, h⟩ := bind_ok h
      obtain ⟨u, hd, h⟩ := bind_ok h
      cases u
      obtain ⟨j₂, hj₂, h⟩ := bind_ok h
      obtain ⟨u₂, -, h⟩ := bind_ok h
      cases u₂
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl, rfl⟩ := h
      exact ⟨⟨s, h1, h2⟩, h3, h4, h5, vs, _, j₁, j₂, h6, rfl, hj₁, hj₂, digestIs_ok hd⟩
    · cases h

/-- **The interior's records have the interior commitment's digest** where `inputAt` returned (`interiorDigestIs`):
the evidence's interior record, which `interiorUnshown` reads the digest of, and its read-back, which the run of
record is given. -/
theorem inputAt_interior {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence} {bs : Core.Blobs}
    {c₂ : OneStage.Commitment} {ξ₂ : ByteArray} {r : OneStageZk.Run} {i : Spot.Run.Input}
    (h : inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i)) :
    Registration.digest e.record = .ok c₂.digest ∧
      ∃ j₂, readBack e.record = .ok j₂ ∧ Registration.digest j₂ = .ok c₂.digest := by
  unfold inputAt at h
  obtain ⟨s, -, h⟩ := bind_ok h
  obtain ⟨c₂', -, h⟩ := bind_ok h
  obtain ⟨ξ₂', -, h⟩ := bind_ok h
  obtain ⟨r', -, h⟩ := bind_ok h
  cases h5 : r'.all
  · simp [h5] at h
    cases h
  · simp only [h5, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at h
    obtain ⟨vs, -, h⟩ := bind_ok h
    split at h
    · obtain ⟨j₁, -, h⟩ := bind_ok h
      obtain ⟨u, -, h⟩ := bind_ok h
      cases u
      obtain ⟨j₂, hj₂, h⟩ := bind_ok h
      obtain ⟨u₂, hd₂, h⟩ := bind_ok h
      cases u₂
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl, rfl⟩ := h
      exact ⟨(interiorDigestIs_ok hd₂).1, j₂, hj₂, (interiorDigestIs_ok hd₂).2⟩
    · cases h

/-- **An accepted two-stage check ran every step** (module docstring). -/
theorem accepted_steps {form : Bool} {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    form = true ∧ ∃ c₂ ξ₂ r i run, inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) ∧ Spot.Run.run i true = .ok run ∧
      sourceRead ξ run = .ok () ∧ interiorUnshown p e = .ok () ∧
      Spot.jeq (Spot.getD run "symbol") (Json.str "accept") = true ∧
      ∃ law, secondLawOf p run = .ok law ∧
        secondCheck p.audit.base.verifier p.audit.base.delta c₂ law ξ₂ { e.interior.base with verdicts := r.out } =
          .accept pub prof := by
  unfold checkBlobsAtWith at h
  cases hp : prepare form p c ξ e bs with
  | error err => simp [hp] at h
  | ok v =>
    obtain ⟨two, run⟩ := v
    simp only [hp] at h
    obtain ⟨hf, hj, htwo⟩ := finishWith_accept h
    subst hf
    replace hp := (prepare_ok hp).1
    unfold prepareRun at hp
    obtain ⟨⟨c₂, ξ₂, r, i⟩, hi, hp⟩ := bind_ok hp
    obtain ⟨u₀, -, hp⟩ := bind_ok hp
    obtain ⟨run', hr, hp⟩ := bind_ok hp
    obtain ⟨u, hs, hp⟩ := bind_ok hp
    obtain ⟨u', hu, hp⟩ := bind_ok hp
    obtain ⟨u'', -, hp⟩ := bind_ok hp
    obtain ⟨u₃, -, hp⟩ := bind_ok hp
    obtain ⟨u₄, -, hp⟩ := bind_ok hp
    obtain ⟨law, hl, hp⟩ := bind_ok hp
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at hp
    obtain ⟨rfl, rfl⟩ := hp
    exact ⟨rfl, c₂, ξ₂, r, i, run', hi, hr, hs, hu, hj, law, hl, htwo⟩

/-- **An accepted two-stage check found the stages' sessions in order** (`stagesOrdered`). -/
theorem accepted_ordered {form : Bool} {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : checkBlobsAtWith form p c ξ e bs = .accept pub prof) : stagesOrdered p e = .ok () := by
  unfold checkBlobsAtWith at h
  cases hp : prepare form p c ξ e bs with
  | error err => simp [hp] at h
  | ok v =>
    replace hp := (prepare_ok hp).1
    unfold prepareRun at hp
    obtain ⟨⟨_, _, _, _⟩, -, hp⟩ := bind_ok hp
    obtain ⟨_, -, hp⟩ := bind_ok hp
    obtain ⟨_, -, hp⟩ := bind_ok hp
    obtain ⟨_, -, hp⟩ := bind_ok hp
    obtain ⟨_, -, hp⟩ := bind_ok hp
    obtain ⟨u, ho, -⟩ := bind_ok hp
    cases u
    exact ho

/-- **An accepted two-stage check found no coin tree's line in the first session's log** (`firstTreeless`). -/
theorem accepted_treeless {form : Bool} {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : checkBlobsAtWith form p c ξ e bs = .accept pub prof) : firstTreeless p e = .ok () := by
  unfold checkBlobsAtWith at h
  cases hp : prepare form p c ξ e bs with
  | error err => simp [hp] at h
  | ok v =>
    replace hp := (prepare_ok hp).1
    unfold prepareRun at hp
    obtain ⟨⟨_, _, _, _⟩, -, hp⟩ := bind_ok hp
    obtain ⟨_, -, hp⟩ := bind_ok hp
    obtain ⟨_, -, hp⟩ := bind_ok hp
    obtain ⟨_, -, hp⟩ := bind_ok hp
    obtain ⟨_, -, hp⟩ := bind_ok hp
    obtain ⟨_, -, hp⟩ := bind_ok hp
    obtain ⟨u, ht, -⟩ := bind_ok hp
    cases u
    exact ht

/-- **An accepted two-stage check found each coin tree's rounds opened stream by stream** (`repStreamsOrdered`). -/
theorem accepted_repOrdered {form : Bool} {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : checkBlobsAtWith form p c ξ e bs = .accept pub prof) : repStreamsOrdered p e = .ok () := by
  unfold checkBlobsAtWith at h
  cases hp : prepare form p c ξ e bs with
  | error err => simp [hp] at h
  | ok v =>
    replace hp := (prepare_ok hp).1
    unfold prepareRun at hp
    obtain ⟨⟨_, _, _, _⟩, -, hp⟩ := bind_ok hp
    obtain ⟨_, -, hp⟩ := bind_ok hp
    obtain ⟨_, -, hp⟩ := bind_ok hp
    obtain ⟨_, -, hp⟩ := bind_ok hp
    obtain ⟨_, -, hp⟩ := bind_ok hp
    obtain ⟨_, -, hp⟩ := bind_ok hp
    obtain ⟨_, -, hp⟩ := bind_ok hp
    obtain ⟨u, ho, -⟩ := bind_ok hp
    cases u
    exact ho

/-- **An accepted two-stage check found every second-stage session's statement under `2^64` bytes**
(`stmtsBytesFit`). -/
theorem accepted_stmtsBytesFit {form : Bool} {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : checkBlobsAtWith form p c ξ e bs = .accept pub prof) : stmtsBytesFit p e bs = .ok () := by
  unfold checkBlobsAtWith at h
  cases hp : prepare form p c ξ e bs with
  | error err => simp [hp] at h
  | ok v => exact (prepare_ok hp).2

/-- **A round that comes before one that comes before a third comes before the third** (`Challenger.roundBefore`). -/
theorem roundBefore_trans {a b c : Nat × Nat} (hab : Challenger.roundBefore a b = true)
    (hbc : Challenger.roundBefore b c = true) : Challenger.roundBefore a c = true := by
  unfold Challenger.roundBefore at hab hbc ⊢
  simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hab hbc ⊢
  omega

/-- **Where each round comes before the next, each comes before every later one** (`Challenger.roundsRise`). -/
theorem roundsRise_pairwise : ∀ {ps : List (Nat × Nat)} {a : Nat × Nat}, Challenger.roundsRise a ps = true →
    (a :: ps).Pairwise fun x y => Challenger.roundBefore x y = true
  | [], a, _ => List.pairwise_singleton _ a
  | b :: ps, a, h => by
    unfold Challenger.roundsRise at h
    rw [Bool.and_eq_true] at h
    have ih := roundsRise_pairwise h.2
    refine List.pairwise_cons.2 ⟨fun y hy => ?_, ih⟩
    rcases List.mem_cons.1 hy with rfl | hy'
    · exact h.1
    · exact roundBefore_trans h.1 ((List.pairwise_cons.1 ih).1 y hy')

/-- **Where each round comes before the next, each comes before every later one** (`Challenger.roundsOrdered`). -/
theorem roundsOrdered_pairwise {ps : List (Nat × Nat)} (h : Challenger.roundsOrdered ps = true) :
    ps.Pairwise fun x y => Challenger.roundBefore x y = true := by
  cases ps with
  | nil => exact List.Pairwise.nil
  | cons a ps => exact roundsRise_pairwise h

/-- **Where a coin tree's rounds are opened stream by stream, a round opened earlier is an earlier round**: where
`Challenger.repStreamOrder` passes the tree `t`, of two rounds the release log shows opened of it, the `i`-th before
the `j`-th and neither its last, the `i`-th is of a stream the tree's spec lists before the `j`-th's
(`Challenger.streamPlace`), or an earlier round of the same place's stream (`Challenger.roundBefore`). So before the
log opens a round's challenges, the only rounds of the tree it has opened are that stream's earlier rounds and rounds
of streams the spec lists earlier. -/
theorem repStreamOrder_before {t : Challenger.LoggedTree} (h : Challenger.repStreamOrder t = true) {i j : Nat}
    (hij : i < j) (hj : j + 1 < t.opened.size) :
    Challenger.roundBefore
      (Challenger.streamPlace t.spec (t.opened[i]'(by omega)).stream, (t.opened[i]'(by omega)).round)
      (Challenger.streamPlace t.spec (t.opened[j]'(by omega)).stream, (t.opened[j]'(by omega)).round) = true := by
  have hp : (t.opened.pop.toList.map fun o => (Challenger.streamPlace t.spec o.stream, o.round)).Pairwise
      fun x y => Challenger.roundBefore x y = true := roundsOrdered_pairwise h
  rw [List.pairwise_iff_getElem] at hp
  have hlen : (t.opened.pop.toList.map fun o => (Challenger.streamPlace t.spec o.stream, o.round)).length =
      t.opened.size - 1 := by simp
  have := hp i j (by omega) (by omega) hij
  simpa using this

/-- **What `treesOrdered` passed**: `Challenger.repStreamOrder` passes each of the trees. -/
theorem treesOrdered_ok {ts : Array Challenger.LoggedTree} (h : treesOrdered ts = .ok ()) :
    ∀ t ∈ ts, Challenger.repStreamOrder t = true := by
  unfold treesOrdered at h
  by_cases ha : ts.all Challenger.repStreamOrder = true
  · exact fun t ht => (Array.all_eq_true'.1 ha) t ht
  · simp [ha] at h

/-- **What `repStreamsOrdered` passed**: the interior audit read its session's coin trees (`OneStageZk.treesOf`, the
trees `OneStageZk.membersRun` holds the second stage's sessions to), and the log opens each one's rounds stream by
stream (`Challenger.repStreamOrder`, read by `repStreamOrder_before`). -/
theorem repStreamsOrdered_trees {p : Params} {e : Evidence} (h : repStreamsOrdered p e = .ok ()) :
    ∃ ts, OneStageZk.treesOf p.audit.base e.interior.base = .ok ts ∧
      ∀ t ∈ ts, Challenger.repStreamOrder t = true := by
  unfold repStreamsOrdered at h
  obtain ⟨ts, hts, h⟩ := bind_ok h
  exact ⟨ts, hts, treesOrdered_ok h⟩

/-- **In an accepted two-stage check, a round of a second-stage coin tree opened earlier is an earlier round**: the
interior audit read its session's coin trees, and of two rounds the log shows opened of one of them, the `i`-th
before the `j`-th and neither the tree's last, the `i`-th is of a stream the tree's spec lists before the `j`-th's,
or an earlier round of the same place's stream. The check's refusal `Challenger.REP_STREAM_ORDER` is what holds the
log to it; no premise on the play enters. -/
theorem accepted_roundsBefore {form : Bool} {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    ∃ ts, OneStageZk.treesOf p.audit.base e.interior.base = .ok ts ∧
      ∀ t ∈ ts, ∀ (i j : Nat) (_hij : i < j) (hj : j + 1 < t.opened.size),
        Challenger.roundBefore
          (Challenger.streamPlace t.spec (t.opened[i]'(by omega)).stream, (t.opened[i]'(by omega)).round)
          (Challenger.streamPlace t.spec (t.opened[j]'(by omega)).stream, (t.opened[j]'(by omega)).round) = true := by
  obtain ⟨ts, hts, hall⟩ := repStreamsOrdered_trees (accepted_repOrdered h)
  exact ⟨ts, hts, fun t ht i j hij hj => repStreamOrder_before (hall t ht) hij hj⟩

/-- **On two windows no line of the first session's log is a coin tree's**: where the check's `firstTreeless` passed,
the boundary's window is `w₁`, the interior audit's is `w₂` and another, and the first session's log reads as the
lines `ls` (`firstLines`), no line of `ls` is a coin tree's (`Spot.Challenger.treeLine`). -/
theorem firstTreeless_lines {p : Params} {e : Evidence} (h : firstTreeless p e = .ok ()) {w₁ w₂ : String}
    {ls : List String} (h1 : p.boundary.verifier.windowRoot = some w₁)
    (h2 : p.audit.base.verifier.windowRoot = some w₂) (hne : w₁ ≠ w₂) (hl : firstLines e = .ok ls) :
    ∀ l ∈ ls, Spot.Challenger.treeLine l = false := by
  have hb : (some w₁ == p.audit.base.verifier.windowRoot) = false := by
    rw [h2]
    simpa using hne
  unfold firstTreeless at h
  rw [h1] at h
  simp only [hb, Bool.false_eq_true, ↓reduceIte, hl, bind, Except.bind] at h
  intro l hl'
  cases ht : Spot.Challenger.treeLine l
  · rfl
  · have hany : treeShown ls = true := List.any_eq_true.2 ⟨l, hl', ht⟩
    rw [hany] at h
    simp at h

/-- **The stages' order holds** where `orderReason` finds no fault: the first stage's session has an end line, the
second stage's a reserve line, and the end's `seq` is below the reserve's. -/
theorem orderReason_none {ended reserve : Option Spot.Challenger.LedgerEntry} (h : orderReason ended reserve = none) :
    ∃ en r, ended = some en ∧ reserve = some r ∧ en.seq < r.seq := by
  cases ended with
  | none => simp [orderReason] at h
  | some en =>
    cases reserve with
    | none => simp [orderReason] at h
    | some r =>
      by_cases hlt : en.seq < r.seq
      · exact ⟨en, r, rfl, rfl, hlt⟩
      · simp [orderReason, hlt] at h

/-- **Neither stage's bytes run past one coin's**, where `stagesFit` passed. -/
theorem stagesFit_le {ξ ξ₂ : ByteArray} {u : Unit} (h : stagesFit ξ ξ₂ = .ok u) :
    ξ.size ≤ STAGE_BYTES ∧ ξ₂.size ≤ STAGE_BYTES := by
  unfold stagesFit at h
  by_cases h1 : STAGE_BYTES < ξ.size
  · simp [h1] at h
  · by_cases h2 : STAGE_BYTES < ξ₂.size
    · simp [h1, h2] at h
    · omega

/-- **An accepted two-stage check's stages each read at most one coin's bytes**: the first stage's bytes `ξ` and the
bytes `ξ₂` the interior audit's sample read (`inputAt`'s) are each at most `STAGE_BYTES` long. -/
theorem accepted_fits {form : Bool} {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : checkBlobsAtWith form p c ξ e bs = .accept pub prof) {c₂ : OneStage.Commitment} {ξ₂ : ByteArray}
    {r : OneStageZk.Run} {i : Spot.Run.Input} (hi : inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i)) :
    ξ.size ≤ STAGE_BYTES ∧ ξ₂.size ≤ STAGE_BYTES := by
  unfold checkBlobsAtWith at h
  cases hp : prepare form p c ξ e bs with
  | error err => simp [hp] at h
  | ok v =>
    replace hp := (prepare_ok hp).1
    unfold prepareRun at hp
    obtain ⟨⟨c₂', ξ₂', r', i'⟩, hi', hp⟩ := bind_ok hp
    obtain ⟨u, hfit, -⟩ := bind_ok hp
    rw [hi] at hi'
    simp only [Except.ok.injEq, Prod.mk.injEq] at hi'
    obtain ⟨-, rfl, -, -⟩ := hi'
    exact stagesFit_le hfit

/-- **The evidence's first-stage bytes are no more than the run read**, where the run reports what it read and
`sourceRead` passed. -/
theorem sourceRead_le {ξ : ByteArray} {run : Json} (h : sourceRead ξ run = .ok ()) {read : Nat}
    (hr : Spot.nat? (Spot.getD run "read") = some read) : ξ.size ≤ read := by
  unfold sourceRead at h
  rw [hr] at h
  by_cases hlt : read < ξ.size
  · simp [hlt] at h
  · omega

/-- **On two windows the first session's log shows no second-stage coin after the interior's digest**: where the
check's `interiorUnshown` passed, the interior audit's parameters name the challenger key `k` and the family `label`,
the boundary's window is `w` and not the interior audit's, the interior registration's digest is `after` and the
first session `e.coins` names on `w` under `k` has the roots `roots`, those roots show no coin of `label` released
after `after`. -/
theorem interiorUnshown_roots {p : Params} {e : Evidence} (h : interiorUnshown p e = .ok ()) {w label : String}
    {after : ByteArray} {k : Spot.Challenger.ChallengerKey} {c : Spot.Challenger.LiveCommitment}
    {roots : Spot.Challenger.Roots} (hk : p.audit.base.challenger = some (k, label))
    (hw : p.boundary.verifier.windowRoot = some w)
    (hne : some w ≠ p.audit.base.verifier.windowRoot)
    (ha : Spot.Registration.digest e.record = .ok after)
    (ht : Spot.Challenger.trustedChallengePhaseOf (Spot.getD e.coins "ledger") (Json.mkObj [("root", Json.str w)]) k
      (Spot.getD e.coins "pins") (Spot.getD e.coins "main") (Spot.getD e.coins "session") = .ok (c, roots)) :
    interiorShown roots label after = false := by
  have hb : (some w == p.audit.base.verifier.windowRoot) = false := by simpa using hne
  unfold interiorUnshown at h
  rw [hk, hw] at h
  simp only [hb, Bool.false_eq_true, ↓reduceIte, ha, ht, bind, Except.bind] at h
  cases hc : interiorShown roots label after
  · rfl
  · rw [hc] at h
    simp at h

end VStar.Recursive.SampledTwo
