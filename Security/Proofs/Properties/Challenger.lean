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
at `lmsKey (keyOfParams ps)`.
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
theorem roundsByDevice_nontrivial : ¬ RoundsByDeviceAt (fun _ _ _ _ _ => .ok []) := by
  intro h
  rcases h ⟨fun _ => [], fun _ => []⟩ ⟨Lms.deployed, ByteArray.empty⟩ "" [] [] Lean.Json.null [] rfl with
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
      (Core.canonicalJson n.params.toUTF8 >>= SampledProofs.OneStageZk.paramsOfJson) = .ok p ∧
        p.base.challenger = some (k, label) := by
  unfold keyOfParams
  cases hn : ps.find? (·.id == SampledProofs.OneStageZk.PROTOCOL) with
  | none => simp
  | some n =>
    simp only [Option.bind, Option.some.injEq, exists_eq_left']
    cases hp : (Core.canonicalJson n.params.toUTF8 >>= SampledProofs.OneStageZk.paramsOfJson) with
    | error e => simp [Except.toOption]
    | ok p =>
      simp only [Except.toOption, Except.ok.injEq, exists_eq_left']
      cases hc : p.base.challenger with
      | none => simp
      | some kl => obtain ⟨k', l⟩ := kl; simp

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

end Properties.Challenger
