import SampledProofs.OneStageZk

/-!
# v2 accepts only the public file and the sizes its registration names

`OneStageZk.checkBlobs` refuses evidence whose public file is not the one the registration names (`publicWhy`), then a
window whose sizes are not the ones v2 reads (`sizesWhy`), before it runs `checkProofs`. So where it accepts:

* the evidence's public digest is the registration's `window.public_file` (`public_of_accept`): every file a session
  reads as its public file is fetched under the digest the statement registers, never one the evidence alone chose;
* the draw opens exactly the positive count of coins the registration's `window.draw_coins` names (`draw_of_accept`);
* session 0 holds at most `MAX_ROUNDS` rounds after its link points, each use of fewer than `2 ^ 64` F128s
  (`rounds_of_accept`);
* `checkProofs` accepts the same outcome (`checkProofs_of_accept`).
-/

namespace SampledProofs.OneStageZk

/-- `publicWhy` names no reason only where the registration names the public file `pub`. -/
theorem registeredPublic_of_publicWhy {c : OneStage.Commitment} {pub : String} (h : publicWhy c pub = none) :
    registeredPublic c = some pub := by
  unfold publicWhy at h
  cases hd : registeredPublic c with
  | none =>
    rw [hd] at h
    cases h
  | some d =>
    rw [hd] at h
    by_cases hp : pub = d
    · rw [hp]
    · simp [hp] at h

/-- **v2 accepts only the public file the registration names**: where `checkBlobs` accepts, the evidence's public
digest is the registration's `window.public_file`. -/
theorem public_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) :
    registeredPublic c = some e.pub := by
  unfold checkBlobs at h
  cases hw : publicWhy c e.pub with
  | some w =>
    rw [hw] at h
    cases h
  | none => exact registeredPublic_of_publicWhy hw

/-- Where `checkBlobs` accepts, `sizesWhy` names no reason. -/
theorem sizesWhy_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) :
    sizesWhy c e bs = none := by
  unfold checkBlobs at h
  cases hw : publicWhy c e.pub with
  | some w =>
    rw [hw] at h
    cases h
  | none =>
    rw [hw] at h
    cases hs : sizesWhy c e bs with
    | none => rfl
    | some w =>
      rw [hs] at h
      cases h

/-- Where `checkBlobs` accepts, `recordWhy` names no reason. -/
theorem recordWhy_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) :
    recordWhy p c e bs = none := by
  have hs := sizesWhy_of_accept h
  unfold checkBlobs at h
  cases hw : publicWhy c e.pub with
  | some w =>
    rw [hw] at h
    cases h
  | none =>
    rw [hw] at h
    rw [hs] at h
    cases hr : recordWhy p c e bs with
    | none => rfl
    | some w =>
      rw [hr] at h
      cases h

/-- Where `checkBlobs` accepts, `checkProofs` accepts the same outcome. -/
theorem checkProofs_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) :
    checkProofs p c e bs = .accept x y := by
  have hs := sizesWhy_of_accept h
  have hr := recordWhy_of_accept h
  unfold checkBlobs at h
  cases hw : publicWhy c e.pub with
  | some w =>
    rw [hw] at h
    cases h
  | none =>
    rw [hw] at h
    rw [hs] at h
    rw [hr] at h
    exact h

/-- A count the registration names is positive. -/
theorem registeredDraw_pos {c : OneStage.Commitment} {m : Nat} (h : registeredDraw c = some m) : 0 < m := by
  unfold registeredDraw at h
  split at h
  next k _ =>
    split at h
    next hk =>
      cases h
      exact hk
    next => cases h
  next => cases h

/-- **v2's draw opens the coins the registration names**: where `checkBlobs` accepts, the registration's
`window.draw_coins` is a positive count `m`, and the draw's segment opens exactly `m` coins. -/
theorem draw_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) :
    ∃ m, registeredDraw c = some m ∧ 0 < m ∧ ∀ opened : Array Json,
      getD (getD e.base.coins "segment") "openings" = .arr opened → opened.size = m := by
  have hs := sizesWhy_of_accept h
  have hd : drawWhy c e = none := by
    unfold sizesWhy at hs
    cases hd : drawWhy c e with
    | none => rfl
    | some w =>
      rw [hd] at hs
      cases hs
  unfold drawWhy at hd
  cases hr : registeredDraw c with
  | none =>
    exfalso
    split at hd
    · unfold drawCountWhy at hd
      rw [hr] at hd
      cases hd
    · cases hd
  | some m =>
    refine ⟨m, rfl, registeredDraw_pos hr, fun opened ho => ?_⟩
    rw [ho] at hd
    unfold drawCountWhy at hd
    rw [hr] at hd
    by_cases hn : opened.size = m
    · exact hn
    · simp [hn] at hd

/-- `sessionsSizesWhy` names no reason for the sessions `s0 :: rest` only where it names none for `s0`. -/
theorem sessionSizesWhy_head {bs : Core.Blobs} {s0 : Session} {rest : List Session} {i : Nat}
    (h : sessionsSizesWhy bs (s0 :: rest) i = none) : sessionSizesWhy bs s0 i = none := by
  cases h0 : sessionSizesWhy bs s0 i with
  | none => rfl
  | some w =>
    exfalso
    revert h
    simp [sessionsSizesWhy, h0]

/-- `roundsCapWhy` names no reason only for uses within its caps. -/
theorem roundsCap_of_none {us : List (Challenger.Use × Array String)} (h : roundsCapWhy us = none) :
    us.length ≤ MAX_ROUNDS + 1 ∧ ∀ x ∈ us, x.2.size < 2 ^ 64 := by
  unfold roundsCapWhy at h
  by_cases hlen : MAX_ROUNDS + 1 < us.length
  · rw [if_pos hlen] at h
    cases h
  · rw [if_neg hlen] at h
    refine ⟨Nat.le_of_not_lt hlen, fun x hx => ?_⟩
    cases hf : us.find? tooWide with
    | some z =>
      rw [hf] at h
      cases h
    | none =>
      have := List.find?_eq_none.1 hf x hx
      simpa [tooWide] using this

/-- **v2's sessions are within its caps**: where `checkBlobs` accepts, session 0's uses of the challenger's coins, as
its record lists them (`sessionUses`), are at most `MAX_ROUNDS` rounds after its link points, each of fewer than
`2 ^ 64` F128s. -/
theorem rounds_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) :
    ∀ (s0 : Session) rest, e.sessions.toList = s0 :: rest →
      ∀ b record us0, bs.get s0.record s!"session {s0.name}'s record" = .ok b →
        String.fromUTF8? b = some record → sessionUses 0 record = .ok us0 →
        us0.length ≤ MAX_ROUNDS + 1 ∧ ∀ x ∈ us0, x.2.size < 2 ^ 64 := by
  intro s0 rest hl b record us0 hb hrec hus
  have hs := sizesWhy_of_accept h
  have hz : sessionsSizesWhy bs e.sessions.toList 0 = none := by
    unfold sizesWhy at hs
    cases hd : drawWhy c e with
    | some w =>
      rw [hd] at hs
      cases hs
    | none =>
      rw [hd] at hs
      exact hs
  rw [hl] at hz
  have h0 := sessionSizesWhy_head hz
  unfold sessionSizesWhy at h0
  simp only [hb, hrec, hus] at h0
  cases hc : roundsCapWhy us0 with
  | some w =>
    rw [hc] at h0
    cases h0
  | none => exact roundsCap_of_none hc

/-- **one-session**: where `checkBlobs` accepts, the evidence names exactly one session. -/
theorem oneSession_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) :
    ∃ s, e.sessions.toList = [s] := by
  have hr := recordWhy_of_accept h
  unfold recordWhy at hr
  cases hl : e.sessions.toList with
  | nil =>
    rw [hl] at hr
    simp [sessionsWhy] at hr
  | cons s rest =>
    cases rest with
    | nil => exact ⟨s, rfl⟩
    | cons t rest =>
      rw [hl] at hr
      simp [sessionsWhy] at hr <;> omega

/-- Where `checkBlobs` accepts, the evidence's session `s0` is its only one, and its record, as the blobs hold it,
carries the verifier's own draw (`recordDrawWhy`), over a public file whose `units.indices` is the identity. -/
theorem record_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) {s0 : Session}
    {rest : List Session} (hl : e.sessions.toList = s0 :: rest) {b : ByteArray} {record : String}
    (hb : bs.get s0.record s!"session {s0.name}'s record" = .ok b) (hrec : String.fromUTF8? b = some record) :
    rest = [] ∧ recordDrawWhy p c e record = none ∧
      ∀ pubB, bs.get e.pub "public file" = .ok pubB → unitsIdentityWhy pubB = none := by
  obtain ⟨s, hs⟩ := oneSession_of_accept h
  rw [hs] at hl
  obtain ⟨rfl, rfl⟩ := List.cons.inj hl
  have hr := recordWhy_of_accept h
  unfold recordWhy at hr
  rw [hs] at hr
  simp only [hb, hrec, Option.or_eq_none_iff, Option.map_eq_none_iff] at hr
  refine ⟨rfl, hr.1, fun pubB hp => ?_⟩
  have h2 := hr.2
  rw [hp] at h2
  exact h2

/-- `subsetAt` names the population `n` and subset size `k` only where the record's `population` is `n` and the law
over it is a subset law of `k`. -/
theorem subsetAt_some {law record : Json} {n k : Nat} (h : subsetAt law record = some (n, k)) :
    nat? (getD record "population") = some n ∧ Draw.flockLaw law n = .ok (.subset k) := by
  unfold subsetAt at h
  cases hn : nat? (getD record "population") with
  | none =>
    rw [hn] at h
    cases h
  | some m =>
    rw [hn] at h
    dsimp only at h
    cases hf : Draw.flockLaw law m with
    | error w =>
      rw [hf] at h
      cases h
    | ok l =>
      rw [hf] at h
      cases l <;> simp at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨rfl, hf⟩

/-- Where `checkBlobs` accepts, at the evidence's one session's record: the verifier's own population and subset size
(`subsetOf`), at which the draw-law, draw-file and proved-units refusals name no reason. -/
theorem recordAt_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) {s0 : Session}
    {rest : List Session} (hl : e.sessions.toList = s0 :: rest) {b : ByteArray} {record : String}
    (hb : bs.get s0.record s!"session {s0.name}'s record" = .ok b) (hrec : String.fromUTF8? b = some record) :
    ∃ n k, subsetOf p c = some (n, k) ∧ drawLawAt n k (drawnUnits e.base.draw) record = none ∧
      drawFileAt n k (drawnUnits e.base.draw) record = none ∧ provedAt e.base.provedUnits record = none := by
  have hr := (record_of_accept h hl hb hrec).2.1
  unfold recordDrawWhy at hr
  cases hsub : subsetOf p c with
  | none =>
    rw [hsub] at hr
    cases hr
  | some nk =>
    obtain ⟨n, k⟩ := nk
    rw [hsub] at hr
    simp only [Option.or_eq_none_iff] at hr
    exact ⟨n, k, rfl, hr.1, hr.2.1, hr.2.2⟩

theorem atomEq_eq {a b : Json} (h : atomEq a b = true) : a = b := by
  cases a <;> cases b <;> first | (cases h; done) | simp_all [atomEq]

theorem atomsEq_eq {a b : List Json} (h : atomsEq a b = true) : a = b := by
  induction a generalizing b with
  | nil =>
    cases b with
    | nil => rfl
    | cons y ys => cases h
  | cons x xs ih =>
    cases b with
    | nil => cases h
    | cons y ys =>
      simp only [atomsEq, Bool.and_eq_true] at h
      rw [atomEq_eq h.1, ih h.2]

theorem valueEq_eq {a b : Json} (h : valueEq a b = true) : a = b := by
  cases a <;> cases b <;>
    first
    | exact congrArg Json.arr (Array.toList_inj.1 (atomsEq_eq h))
    | exact atomEq_eq h

theorem fieldsEq_eq {a b : List (String × Json)} (h : fieldsEq a b = true) : a = b := by
  induction a generalizing b with
  | nil =>
    cases b with
    | nil => rfl
    | cons q ys =>
      obtain ⟨l, y⟩ := q
      cases h
  | cons q xs ih =>
    obtain ⟨k, x⟩ := q
    cases b with
    | nil => cases h
    | cons q' ys =>
      obtain ⟨l, y⟩ := q'
      simp only [fieldsEq, Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨hk, hv⟩, hr⟩ := h
      rw [hk, valueEq_eq hv, ih hr]

/-- An object whose fields `fieldsOf` reads rebuilds (`Flock.norm`) to the object of those fields. -/
theorem norm_of_fieldsOf {x : Json} {l : List (String × Json)} (h : fieldsOf x = some l) :
    Flock.norm x = .obj (Flock.normObj l) := by
  rcases x with _ | _ | _ | _ | _ | ⟨⟨t⟩⟩
  all_goals first
    | (simp only [fieldsOf, Option.some.injEq] at h
       subst h
       first | rfl | simp only [Flock.norm] | simp [Flock.norm])
    | (simp [fieldsOf] at h)
    | (cases h)

/-- Two values with the same fields (`sameFields`) rebuild (`Flock.norm`) to the same value. -/
theorem norm_of_sameFields {a b : Json} (h : sameFields a b = true) : Flock.norm a = Flock.norm b := by
  unfold sameFields at h
  cases ha : fieldsOf a with
  | none =>
    rw [ha] at h
    cases h
  | some u =>
    cases hb : fieldsOf b with
    | none =>
      rw [ha, hb] at h
      cases h
    | some v =>
      rw [ha, hb] at h
      have h' : fieldsEq u v = true := h
      rw [norm_of_fieldsOf ha, norm_of_fieldsOf hb, fieldsEq_eq h']

/-- `Flock.Witness.recordDraw` is `rawDraw` rebuilt. -/
theorem recordDraw_of_rawDraw {record : String} {d : Json} (h : rawDraw record = some d) :
    Flock.Witness.recordDraw record = some (Flock.norm d) := by
  unfold rawDraw at h
  unfold Flock.Witness.recordDraw
  cases hp : Flock.JsonParse.parse record with
  | error w =>
    rw [hp] at h
    cases h
  | ok j =>
    rw [hp] at h ⊢
    dsimp only at h ⊢
    cases hg : j.getObjVal? "unit_draw" with
    | error w =>
      rw [hg] at h
      cases h
    | ok d' =>
      rw [hg] at h ⊢
      cases d' <;> first | (cases h; rfl) | cases h

/-- `drawLawAt` names no reason only where the units `us` are `k` ascending units below `n`, and the record's draw decodes
to a subset draw of `k` over `n` with no closure. -/
theorem drawLaw_of_none {n k : Nat} {us : Array Nat} {record : String} (h : drawLawAt n k us record = none) :
    us.toList.Pairwise (· < ·) ∧ (∀ u ∈ us, u < n) ∧ us.size = k ∧
      ∃ j D, Flock.Witness.recordDraw record = some j ∧ Flock.Draw.UnitDraw.ofJson j = .ok D ∧
        D.law = .subset k ∧ D.population = n ∧ D.closure = #[] := by
  unfold drawLawAt at h
  split at h
  · cases h
  · next hc =>
    have hc' : (decide (us.toList.Pairwise (· < ·)) && us.toList.all (fun x => decide (x < n)) &&
        us.size == k) = true := by
      cases hq : (decide (us.toList.Pairwise (· < ·)) && us.toList.all (fun x => decide (x < n)) && us.size == k)
      · rw [hq] at hc
        exact absurd rfl hc
      · rfl
    rw [Bool.and_eq_true, Bool.and_eq_true] at hc'
    obtain ⟨⟨h1, h2⟩, h3⟩ := hc'
    refine ⟨of_decide_eq_true h1, fun u hu => of_decide_eq_true (List.all_eq_true.1 h2 u (Array.mem_toList_iff.2 hu)),
      beq_iff_eq.1 h3, ?_⟩
    cases hj : Flock.Witness.recordDraw record with
    | none =>
      rw [hj] at h
      cases h
    | some j =>
      rw [hj] at h
      dsimp only at h
      cases hD : Flock.Draw.UnitDraw.ofJson j with
      | error w =>
        rw [hD] at h
        cases h
      | ok D =>
        rw [hD] at h
        dsimp only at h
        refine ⟨j, D, rfl, rfl, ?_⟩
        by_cases hq : (subsetLaw D.law k && D.population == n && D.closure.isEmpty) = true
        · rw [Bool.and_eq_true, Bool.and_eq_true] at hq
          obtain ⟨⟨hl, hp⟩, he⟩ := hq
          refine ⟨?_, beq_iff_eq.1 hp, Array.isEmpty_iff.1 he⟩
          have hlaw : ∀ l : Flock.Draw.Law, subsetLaw l k = true → l = .subset k := by
            intro l hl'
            cases l <;> simp [subsetLaw] at hl' ⊢
            all_goals exact hl'
          exact hlaw D.law hl
        · rw [if_neg hq] at h
          cases h

/-- `drawFileAt` names no reason only where the record's draw is the verifier's own draw file, rebuilt. -/
theorem drawFile_of_none {n k : Nat} {us : Array Nat} {record : String} (h : drawFileAt n k us record = none) :
    Flock.Witness.recordDraw record = some (Flock.norm (drawFile k n us)) := by
  unfold drawFileAt at h
  cases hd : rawDraw record with
  | none =>
    rw [hd] at h
    cases h
  | some d =>
    rw [hd] at h
    dsimp only at h
    split at h
    · cases h
    · split at h
      · cases h
      · next hs =>
        have hs' : sameFields d (drawFile k n us) = true := by
          cases hq : sameFields d (drawFile k n us)
          · rw [hq] at hs
            exact absurd rfl hs
          · rfl
        rw [recordDraw_of_rawDraw hd, norm_of_sameFields hs']

/-- `provedAt` names no reason only where the record's draw names the units `proved`. -/
theorem proved_of_none {proved : Array Nat} {record : String} (h : provedAt proved record = none) :
    ∀ j, Flock.Witness.recordDraw record = some j → drawnUnits j = proved := by
  intro j hj
  unfold provedAt at h
  rw [hj] at h
  dsimp only at h
  by_cases hq : (drawnUnits j == proved) = true
  · exact beq_iff_eq.1 hq
  · rw [if_neg hq] at h
    cases h

/-- `unitsIdentityWhy` names no reason only where the public file's header names a list as its `units.indices` whose
`i`-th entry is `i`. -/
theorem unitsIdentity_of_none {pubB : ByteArray} (h : unitsIdentityWhy pubB = none) :
    ∃ line hd is, String.fromUTF8? (headerLine pubB) = some line ∧ Json.parse line = .ok hd ∧
      getD (getD hd "units") "indices" = .arr is ∧ ∀ i < is.size, nat? (is.getD i Json.null) = some i := by
  unfold unitsIdentityWhy at h
  cases hu : String.fromUTF8? (headerLine pubB) with
  | none =>
    rw [hu] at h
    cases h
  | some line =>
    rw [hu] at h
    dsimp only at h
    cases hj : Json.parse line with
    | error w =>
      rw [hj] at h
      cases h
    | ok hd =>
      rw [hj] at h
      dsimp only at h
      rcases hi : getD (getD hd "units") "indices" with _ | _ | _ | _ | is | _
      all_goals rw [hi] at h
      all_goals first | (cases h; done) | skip
      refine ⟨line, hd, is, rfl, rfl, rfl, fun i hlt => ?_⟩
      have hid : isIdentity is = true := by
        cases hb : isIdentity is
        · simp [hb] at h
        · rfl
      unfold isIdentity at hid
      have := List.all_eq_true.1 hid i (List.mem_range.2 hlt)
      simpa using this

/-- **draw-law**: where `checkBlobs` accepts, at the evidence's session `s0` and its record as the blobs hold it, the
verifier's law over the registration's population `n` is a subset law of `k`, the evidence's draw's units are `k`
ascending units below `n`, and the record's draw decodes to a subset draw of `k` over `n` with no closure. -/
theorem drawLaw_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) {s0 : Session}
    {rest : List Session} (hl : e.sessions.toList = s0 :: rest) {b : ByteArray} {record : String}
    (hb : bs.get s0.record s!"session {s0.name}'s record" = .ok b) (hrec : String.fromUTF8? b = some record) :
    ∃ n k, nat? (getD c.record "population") = some n ∧ Draw.flockLaw p.base.verifier.law n = .ok (.subset k) ∧
      (drawnUnits e.base.draw).toList.Pairwise (· < ·) ∧ (∀ u ∈ drawnUnits e.base.draw, u < n) ∧
      (drawnUnits e.base.draw).size = k ∧
      ∃ j D, Flock.Witness.recordDraw record = some j ∧ Flock.Draw.UnitDraw.ofJson j = .ok D ∧
        D.law = .subset k ∧ D.population = n ∧ D.closure = #[] := by
  obtain ⟨n, k, hs, hlaw, -, -⟩ := recordAt_of_accept h hl hb hrec
  obtain ⟨hpop, hfl⟩ := subsetAt_some hs
  exact ⟨n, k, hpop, hfl, drawLaw_of_none hlaw⟩

/-- **draw-file**: where `checkBlobs` accepts, at the evidence's session `s0` and its record as the blobs hold it, the
record's draw is the verifier's own draw file, rebuilt: the subset draw of the verifier's `k` over the registration's
population `n` of the units the evidence's draw names (`drawFile`). -/
theorem drawFile_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) {s0 : Session}
    {rest : List Session} (hl : e.sessions.toList = s0 :: rest) {b : ByteArray} {record : String}
    (hb : bs.get s0.record s!"session {s0.name}'s record" = .ok b) (hrec : String.fromUTF8? b = some record) :
    ∃ n k, nat? (getD c.record "population") = some n ∧ Draw.flockLaw p.base.verifier.law n = .ok (.subset k) ∧
      Flock.Witness.recordDraw record = some (Flock.norm (drawFile k n (drawnUnits e.base.draw))) := by
  obtain ⟨n, k, hs, -, hfile, -⟩ := recordAt_of_accept h hl hb hrec
  obtain ⟨hpop, hfl⟩ := subsetAt_some hs
  exact ⟨n, k, hpop, hfl, drawFile_of_none hfile⟩

/-- **proved-units**: where `checkBlobs` accepts, at the evidence's session `s0` and its record as the blobs hold it, the
evidence's `proved_units` are the units the record's draw names. -/
theorem provedUnits_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) {s0 : Session}
    {rest : List Session} (hl : e.sessions.toList = s0 :: rest) {b : ByteArray} {record : String}
    (hb : bs.get s0.record s!"session {s0.name}'s record" = .ok b) (hrec : String.fromUTF8? b = some record) :
    ∀ j, Flock.Witness.recordDraw record = some j → drawnUnits j = e.base.provedUnits := by
  obtain ⟨n, k, -, -, -, hp⟩ := recordAt_of_accept h hl hb hrec
  exact proved_of_none hp

/-- **units-identity**: where `checkBlobs` accepts, the public file the blobs hold under the evidence's digest has a
header whose `units.indices` is the identity. -/
theorem unitsIdentity_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) {pubB : ByteArray}
    (hp : bs.get e.pub "public file" = .ok pubB) :
    ∃ line hd is, String.fromUTF8? (headerLine pubB) = some line ∧ Json.parse line = .ok hd ∧
      getD (getD hd "units") "indices" = .arr is ∧ ∀ i < is.size, nat? (is.getD i Json.null) = some i := by
  obtain ⟨s, hs⟩ := oneSession_of_accept h
  have hr := recordWhy_of_accept h
  unfold recordWhy at hr
  rw [hs] at hr
  dsimp only at hr
  cases hb : bs.get s.record s!"session {s.name}'s record" with
  | error w =>
    rw [hb] at hr
    cases hr
  | ok b =>
    cases hu : String.fromUTF8? b with
    | none =>
      rw [hb] at hr
      simp [hu] at hr
    | some record =>
      exact unitsIdentity_of_none ((record_of_accept h hs hb hu).2.2 pubB hp)

end SampledProofs.OneStageZk
