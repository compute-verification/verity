import Properties.Challenger

/-!
# Proofs of `Properties.Challenger`

`liveAnswer`: `Live.factors` turns `answer`'s check into `checkAt` at the coins the evidence reads, and `answerAt` runs
the same decoders. `coinsFromLog`: the coins' target from `liveAnswer` and the reader's property `DrawnByDevice`.
`drawnByDevice_nontrivial`: a reader that accepts every record fails `DrawnByDevice`'s statement, and
`roundsByDevice_nontrivial` likewise `RoundsByDevice`'s.
`oneStage_check_of_accept`: one-stage's live check accepts only what its coin-free `check` accepts, so the properties
proved about `check` (`acceptMeans_iff`) hold of what it accepts.
`verifies_iff`: the reader's LMS check is `LmsVerifies`; `keyOfParams_eq_some`: the key v2's compiled check reads from
the registered parameters; `verifies_registered`: under that key, the check accepts exactly what `LmsVerifies` accepts
at `lmsKey (keyOfParams ps)`. `runsWhy_none`: a log whose labels `runsWhy` passes decomposes, group by group, into
its uses' runs, and `Session.runs_of_runsWhy` states that as `Session.Runs`.
-/

namespace Properties.Challenger

/-- `answer` accepts exactly when its three byte strings decode and the check accepts what they decode to. -/
theorem answer_accept_iff (Q : Core.Protocol) {a b e : ByteArray} {p : Q.Params} {c : Q.Commitment} {ev : Q.Evidence}
    (hp : (Core.canonicalJson a >>= Q.params) = .ok p) (hc : Q.commitment b = .ok c) (hev : Q.evidence e = .ok ev) :
    (Q.answer a b e).verdict = .accept ↔ ∃ pub prof, Q.check p c ev = .accept pub prof := by
  unfold Core.Protocol.answer
  rw [hp]
  simp only []
  rw [hc]
  simp only []
  rw [hev]
  simp only []
  cases Q.check p c ev <;> simp

theorem liveAnswer : LiveAnswer := by
  intro P L a b e h
  unfold Core.Protocol.answer at h
  cases hp : (Core.canonicalJson a >>= P.params) with
  | error r => simp [hp] at h
  | ok p =>
    cases hc : P.commitment b with
    | error r => simp [hp, hc] at h
    | ok c =>
      cases hev : P.evidence e with
      | error r => simp [hp, hc, hev] at h
      | ok ev =>
        have hacc : ∃ pub prof, P.check p c ev = .accept pub prof := by
          cases ho : P.check p c ev <;> simp [hp, hc, hev, ho] at h ⊢
        cases hξ : L.coins p c ev with
        | error r =>
          obtain ⟨pub, prof, ho⟩ := hacc
          rw [L.factors, hξ] at ho
          simp [Core.atCoins] at ho
        | ok ξ =>
          refine ⟨p, c, ev, ξ, rfl, rfl, rfl, hξ, ?_⟩
          obtain ⟨pub, prof, ho⟩ := hacc
          rw [L.factors, hξ] at ho
          simp only [Core.atCoins] at ho
          have hp' : (Core.canonicalJson a >>= (L.at ξ).params) = .ok p := hp
          have hc' : (L.at ξ).commitment b = .ok c := hc
          have hev' : (L.at ξ).evidence e = .ok ev := hev
          exact (answer_accept_iff (L.at ξ) hp' hc' hev').2 ⟨pub, prof, ho⟩

theorem coinsFromLog (hd : DrawnByDevice) : CoinsFromLog := by
  intro P L R dev a b e h
  obtain ⟨p, c, ev, ξ, hp, hc, hev, hξ, hacc⟩ := liveAnswer P L a b e h
  refine ⟨p, c, ev, ξ, hp, hc, hev, hacc, fun d hd' => ?_⟩
  obtain ⟨raw, hraw, hdec⟩ := R.reads p c d hd' ev ξ hξ
  rcases hd dev d.key d.window d.label d.after (R.record ev) raw hraw with ⟨s, hs, hopen, hrel⟩ | hb
  · exact .inl ⟨s, hs, hopen, by rw [← hrel]; exact hdec⟩
  · exact .inr hb

/-- `DrawnByDevice`'s statement is not trivially true: a reader that accepts every record fails it, since the empty
device released nothing and the finder finds no break in an empty run. -/
theorem drawnByDevice_nontrivial : ¬ DrawnByDeviceAt (fun _ _ _ _ _ => .ok ByteArray.empty) := by
  intro h
  have hnone : findBreak ⟨fun _ => [], fun _ => []⟩ ⟨Lms.deployed, ByteArray.empty⟩ "" Lean.Json.null = none := rfl
  rcases h ⟨fun _ => [], fun _ => []⟩ ⟨Lms.deployed, ByteArray.empty⟩ "" "" ByteArray.empty Lean.Json.null
      ByteArray.empty rfl with ⟨s, hs, -⟩ | ⟨w, hw, -⟩
  · simp at hs
  · rw [hnone] at hw
    cases hw

/-- `RoundsByDevice`'s statement is not trivially true: a reader that accepts every record fails it, since the empty
device's ledger opens no session and an empty list of uses has no use whose record could carry a break. -/
theorem roundsByDevice_nontrivial : ¬ RoundsByDeviceAt (fun _ _ _ _ _ _ => .ok []) := by
  intro h
  rcases h ⟨fun _ => [], fun _ => []⟩ ⟨Lms.deployed, ByteArray.empty⟩ "" [] [] [] Lean.Json.null [] rfl with
    ⟨s, hs, -⟩ | ⟨i, u, hu, -⟩
  · simp at hs
  · simp at hu

/-- What one-stage's protocol accepts, its coin-free `check` accepts: `checkAt` only adds the coins' refusals. -/
theorem oneStage_check_of_accept {p : SampledProofs.OneStage.Params} {c : SampledProofs.OneStage.Commitment}
    {e : SampledProofs.OneStage.Evidence} {pub : SampledProofs.OneStage.Public} {prof : Core.Integrity}
    (h : SampledProofs.OneStage.protocol.check p c e = .accept pub prof) :
    SampledProofs.OneStage.check p c e = .accept pub prof := by
  change Core.atCoins (SampledProofs.OneStage.coinsOf p c e)
    (fun ξ => SampledProofs.OneStage.checkAt p c ξ e) = _ at h
  cases hξ : SampledProofs.OneStage.coinsOf p c e with
  | error r => rw [hξ] at h; simp [Core.atCoins] at h
  | ok ξ =>
    rw [hξ] at h
    simp only [Core.atCoins, SampledProofs.OneStage.checkAt] at h
    cases hw : ξ.bind (SampledProofs.OneStage.coinsWhy p c e) with
    | some why => rw [hw] at h; simp at h
    | none => rw [hw] at h; exact h

/-- **The reader's LMS check is the one definition's**: `Challenger.verifies`, the check of the record, the close and
each ledger line, accepts exactly the signatures `LmsVerifies` accepts. -/
theorem verifies_iff (key : SampledProofs.Challenger.ChallengerKey) (m sig : ByteArray) :
    SampledProofs.Challenger.verifies key m sig = true ↔ LmsVerifies key m sig := by
  unfold SampledProofs.Challenger.verifies LmsVerifies MemoryChallenger.Properties.LmsVerifies
  cases hd : Lms.Signature.decode key.params sig with
  | error e => simp
  | ok σ =>
    simp only [Except.ok.injEq, exists_eq_left']
    cases hv : Lms.verify key.params key.pub m σ with
    | error e => simp [Except.toOption]
    | ok b => cases b <;> simp [Except.toOption]

/-- **The key v2's compiled check reads**: `keyOfParams ps` is `k` exactly when the entry of `ps` named v2's id has
parameters v2 reads (`OneStageZk.paramsOfJson` on their canonical JSON) whose challenger is `k`, the key `roundsWhy`
hands `roundCoins` and `OneStage.coinsOf` hands `drawCoins`. -/
theorem keyOfParams_eq_some {ps : List Core.Composition.Named} {k : SampledProofs.Challenger.ChallengerKey} :
    keyOfParams ps = some k ↔ ∃ n p label, ps.find? (·.id == SampledProofs.OneStageZk.PROTOCOL) = some n ∧
      paramsOfNamed n = .ok p ∧ p.base.challenger = some (k, label) := by
  unfold keyOfParams
  cases hn : ps.find? (·.id == SampledProofs.OneStageZk.PROTOCOL) with
  | none => simp
  | some n =>
    cases hp : paramsOfNamed n with
    | error e => simp [hp, Except.toOption]
    | ok p =>
      cases hc : p.base.challenger with
      | none => simp [hp, hc, Except.toOption]
      | some kl => obtain ⟨k', l⟩ := kl; simp [hp, hc, Except.toOption]

/-- **The compiled check's LMS verify is `LmsVerifies` at the registered key**: where v2's compiled check reads the
key `k` from the registered components' parameters `ps` (`keyOfParams`, `keyOfParams_eq_some`), every LMS check its
challenger reader makes (`Challenger.verifies`) accepts exactly the signatures that
`MemoryChallenger.Properties.LmsVerifies` accepts at `lmsKey (keyOfParams ps)`. -/
theorem verifies_registered {ps : List Core.Composition.Named} {k : SampledProofs.Challenger.ChallengerKey}
    (h : keyOfParams ps = some k) (m sig : ByteArray) :
    SampledProofs.Challenger.verifies k m sig = true ↔
      MemoryChallenger.Properties.LmsVerifies (lmsKey (keyOfParams ps)).1 (lmsKey (keyOfParams ps)).2 m sig := by
  rw [h]
  exact verifies_iff k m sig

theorem findIdx?_split {α : Type} [Inhabited α] {p : α → Bool} {xs : List α} {i : Nat} (hi : xs.findIdx? p = some i) :
    ∃ pre post u, xs = pre ++ u :: post ∧ p u = true ∧ (∀ x ∈ pre, p x = false) ∧ xs.getD i default = u ∧
      xs.eraseIdx i = pre ++ post := by
  obtain ⟨hlt, hp, hpre⟩ := List.findIdx?_eq_some_iff_getElem.1 hi
  refine ⟨xs.take i, xs.drop (i + 1), xs[i], ?_, hp, ?_, ?_, List.eraseIdx_eq_take_drop_succ xs i⟩
  · rw [List.getElem_cons_drop, List.take_append_drop]
  · intro x hx
    obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hx
    have hji : j < i := by simp at hj; omega
    simpa [List.getElem_take] using hpre j hji
  · simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hlt]

open SampledProofs.Challenger in
/-- **`runsReason` passes exactly a run decomposition**: when the walk finds no reason, the labels `ls` are some order
`vs` of the uses `pending`, each family's in `pending`'s order, each use its family once a coin. -/
theorem runsReason_none : ∀ (f : Nat) (pending : List Use) (ls : List String), runsReason f pending ls = none →
    ∃ vs : List Use, (∀ ℓ, vs.filter (·.label == ℓ) = pending.filter (·.label == ℓ)) ∧
      ls = vs.flatMap fun u => List.replicate u.coins u.label
  | _, [], [], _ => ⟨[], fun _ => rfl, rfl⟩
  | _, _ :: _, [], h => by simp [runsReason] at h
  | 0, _, _ :: _, h => by simp [runsReason] at h
  | f + 1, pending, l :: ls, h => by
    simp only [runsReason] at h
    cases hi : pending.findIdx? (·.label == l) with
    | none => simp [hi] at h
    | some i =>
      obtain ⟨pre, post, u, hsplit, hu, hpre, hget, herase⟩ := findIdx?_split hi
      simp only [hi, hget] at h
      split at h
      · rename_i hc
        obtain ⟨vs, hvs, hls⟩ := runsReason_none f _ _ h
        have hul : u.label = l := by simpa using hu
        have hn : u.coins ≤ ls.length + 1 := by simp at hc; exact hc.1
        have htake : (l :: ls).take u.coins = List.replicate u.coins l := by
          rw [List.eq_replicate_iff]
          refine ⟨by simp; omega, fun b hb => ?_⟩
          simp at hc
          exact hc.2 b hb
        refine ⟨u :: vs, fun ℓ => ?_, ?_⟩
        · rw [hsplit, List.filter_append, List.filter_cons, List.filter_cons, hvs ℓ, herase, List.filter_append]
          by_cases hl : l = ℓ
          · subst hl
            have : pre.filter (·.label == l) = [] := List.filter_eq_nil_iff.2 fun x hx => by simp [hpre x hx]
            simp [hul, this]
          · simp [hul, hl]
        · rw [← List.take_append_drop u.coins (l :: ls), htake, hls]
          simp [hul]
      · simp at h

open SampledProofs.Challenger in
/-- **A log `runsWhy` passes releases each group's uses one run each**: for every group `g`, the labels restricted to
`g`'s families decompose as `Session.Runs` states. -/
theorem runsWhy_none {us : List Use} {groups : List (List String)} {labels : List String}
    (h : runsWhy us groups labels = none) : ∀ g ∈ groups, ∃ vs : List Use,
      (∀ ℓ, vs.filter (·.label == ℓ) = us.filter fun u => u.label == ℓ && g.contains ℓ && u.coins != 0) ∧
        labels.filter g.contains = vs.flatMap fun u => List.replicate u.coins u.label := by
  intro g hg
  obtain ⟨vs, hvs, hls⟩ := runsReason_none _ _ _ (List.findSome?_eq_none_iff.1 h g hg)
  refine ⟨vs, fun ℓ => ?_, hls⟩
  rw [hvs ℓ, List.filter_filter]
  refine List.filter_congr fun u _ => ?_
  by_cases hl : u.label = ℓ
  · simp [hl]
  · have : (u.label == ℓ) = false := by simpa using hl
    simp [this]

/-- **The refusal's guarantee on a session's log**: a session whose log's labels `runsWhy` passes releases each group's
uses one run each (`Session.Runs`), the conjunct `RoundsByDevice` states of the session behind an accepted read. -/
theorem Session.runs_of_runsWhy {s : Session} {us : List SampledProofs.Challenger.Use} {groups : List (List String)}
    (h : SampledProofs.Challenger.runsWhy us groups (SampledProofs.Challenger.receivedOrder s.log) = none) :
    ∀ g ∈ groups, s.Runs g us :=
  runsWhy_none h

end Properties.Challenger
