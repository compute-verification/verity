import Proofs.Properties.Verity.SampledProj

/-!
# The records the finder runs on, in what was sent

`roundsWhy` runs the round coins at uses, an order, groups and coins it reads from the evidence and the sessions'
records before it runs them; `OneStageZk.roundsIn` is that read, so where `roundsWhy` passes, `roundCoins` returns on
`roundsIn`'s uses and coins (`roundsIn_of_roundsWhy`), the first of them the draw's (`roundsIn_draw`).

`Canon.recsSent` is every record the finder runs on in what was sent: for each of the components' cases that decodes as
v2's parameters, commitment and evidence, the records of `roundsIn`'s uses (`Vocab.recsOf`). It reads only the bytes
sent, so the finder is a fixed algorithm, and it holds every accepting case's records (`mem_recsSent`), whichever case a
proof reads. Every accepting case names the device's registered key (`key_of_case`), so a break on one of its uses is
the collision `colAt` reads or the forger's success (`brk_of_break`), and off those, there is none (`no_break`).
-/

open Lean (Json)

namespace SampledProofs.OneStageZk

open Flock (V)

/-- **The uses `roundsWhy` runs the round coins at**, and their wants, order, groups and coins, before it runs them:
the draw's use after the registration's digest, then each session's (`sessionUses`), with the evidence's coins and
every session's segments after the draw's. -/
def roundsIn (label : String) (c : OneStage.Commitment) (e : Evidence) (bs : Core.Blobs) :
    V (Array Challenger.Use × Array (Option (String × Array String)) × Array (String × String) ×
      Array (List String) × Json) := do
  let draw := getD e.base.coins "segment"
  let .arr opened := getD draw "openings" | throw Challenger.SEGMENT_FORM
  let mut uses : Array Challenger.Use := #[{ label, after := c.digest, coins := opened.size }]
  let mut wants : Array (Option (String × Array String)) := #[none]
  let mut segments : Array Json := #[draw]
  let mut order : Array (String × String) := #[]
  let mut groups : Array (List String) := #[]
  for (s, i) in e.sessions.toList.zipIdx do
    let some record := String.fromUTF8? (← bs.get s.record s!"session {s.name}'s record")
      | throw s!"session {s.name}'s record is not UTF-8"
    let us ← (sessionUses i record).mapError (s!"session {s.name}: {·}")
    unless s.coins.size == us.length do
      throw s!"session {s.name}: {s.coins.size} segments for its {us.length} uses of the challenger's coins"
    let points := Hello.slotStream i POINTS
    let fams := (us.map (·.1.label)).eraseDups
    order := order ++ (fams.map (label, ·)).toArray ++ ((fams.filter (· != points)).map (points, ·)).toArray
    groups := groups.push fams
    uses := uses ++ us.toArray.map (·.1)
    wants := wants ++ us.toArray.map fun (u, hs) => some (s!"{s.name}'s {u.label}", hs)
    segments := segments ++ s.coins
  let coins := Json.mkObj [("ledger", getD e.base.coins "ledger"), ("main", getD e.base.coins "main"),
    ("pins", getD e.base.coins "pins"), ("segments", Json.arr segments), ("session", getD e.base.coins "session")]
  return (uses, wants, order, groups, coins)

/-- **`roundsWhy` passes only at the uses `roundsIn` reads**: where it names no reason, v2's parameters name a
challenger key and label and a window root, `roundsIn` reads the uses, wants, order, groups and coins, and
`roundCoins` returns on them under that key on that window. -/
theorem roundsIn_of_roundsWhy {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    (h : roundsWhy p c e bs = none) :
    ∃ key label w uses wants order groups coins ξs, p.base.challenger = some (key, label) ∧
      p.base.verifier.windowRoot = some w ∧ roundsIn label c e bs = .ok (uses, wants, order, groups, coins) ∧
      Challenger.roundCoins key w uses.toList order.toList groups.toList coins = .ok ξs := by
  unfold roundsWhy at h
  split at h
  · rename_i key label w hc hw
    dsimp only at h
    split at h
    · rename_i hr
      unfold roundsIn
      simp only [bind, Except.bind, pure, Except.pure] at hr ⊢
      repeat' split at hr
      all_goals try (cases hr; done)
      rename_i hop _ st hf _ ξs hq _ _ _
      refine ⟨key, label, w, st.1, st.2.1, st.2.2.2.1, st.2.2.2.2, _, ξs, hc, hw, ?_, hq⟩
      simp only [hop]
      split
      · rename_i hx; cases hx.symm.trans hf
      · rename_i hx; cases hx.symm.trans hf; rfl
    · cases h
  · cases h

/-- An invariant of a loop's state that each step keeps holds of the state the loop returns. -/
theorem forIn_except_inv {α σ ε : Type} (I : σ → Prop) {f : α → σ → Except ε (ForInStep σ)}
    (hf : ∀ x s r, I s → f x s = .ok r → I r.value) : ∀ (l : List α) (s s' : σ), I s → forIn l s f = .ok s' → I s'
  | [], s, s', hs, h => by
    simp only [List.forIn_nil, pure, Except.pure] at h
    cases h
    exact hs
  | x :: l, s, s', hs, h => by
    rw [List.forIn_cons] at h
    simp only [bind, Except.bind] at h
    split at h
    · cases h
    · rename_i r hr
      have hI := hf x s r hs hr
      cases r with
      | done b => cases h; exact hI
      | yield b => exact forIn_except_inv I hf l b s' hI h

/-- **`roundsIn`'s first use is the draw's**, released after the registration's digest, and its coins carry the
evidence's, with the draw's segment first. -/
theorem roundsIn_draw {label : String} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {uses : Array Challenger.Use} {wants : Array (Option (String × Array String))} {order : Array (String × String)}
    {groups : Array (List String)} {coins : Json}
    (h : roundsIn label c e bs = .ok (uses, wants, order, groups, coins)) :
    ∃ k segs, uses.toList[0]? = some ⟨label, c.digest, k⟩ ∧ segs[0]? = some (getD e.base.coins "segment") ∧
      coins = Json.mkObj [("ledger", getD e.base.coins "ledger"), ("main", getD e.base.coins "main"),
        ("pins", getD e.base.coins "pins"), ("segments", Json.arr segs), ("session", getD e.base.coins "session")] := by
  unfold roundsIn at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  split at h
  · rename_i opened _
    split at h
    · cases h
    · rename_i st hf
      cases h
      have hI := forIn_except_inv (fun st : Array Challenger.Use × Array (Option (String × Array String)) × Array Json ×
          Array (String × String) × Array (List String) =>
          st.1[0]? = some ⟨label, c.digest, opened.size⟩ ∧ st.2.2.1[0]? = some (getD e.base.coins "segment"))
        (fun x s r hs hr => by
          try simp only at hr
          repeat' split at hr
          all_goals first
            | (cases hr; done)
            | (cases hr; exact hs)
            | (cases hr; simp only [ForInStep.value]; exact ⟨by rw [Array.getElem?_append_left (by
                  have := Array.getElem?_eq_some_iff.1 hs.1; exact this.1)]; exact hs.1,
                by rw [Array.getElem?_append_left (by
                  have := Array.getElem?_eq_some_iff.1 hs.2; exact this.1)]; exact hs.2⟩))
        _ _ _ ⟨rfl, rfl⟩ hf
      refine ⟨opened.size, st.2.2.1, ?_, hI.2, rfl⟩
      rw [Array.getElem?_toList]
      exact hI.1
  · cases h

end SampledProofs.OneStageZk

namespace Verity.Canon

open Accounting Core
open _root_.SampledProofs.OneStageZk (roundsIn)

variable {ι α : Type} [Fintype α] {d : Device ι}

/-- The records the finder runs on in one of the components' cases: where it decodes as v2's parameters, commitment and
evidence and the parameters name a challenger, the records of the uses `roundsIn` reads (`Vocab.recsOf`); else none. -/
def recsCase (nm : Core.Composition.Named) (cs : Core.Composition.Case) (bs : Blobs) : List (String × Json) :=
  match canonicalJson nm.params.toUTF8 >>= _root_.SampledProofs.OneStageZk.paramsOfJson,
      _root_.SampledProofs.OneStage.commitmentOf cs.commitment,
      _root_.SampledProofs.OneStageZk.evidenceOf cs.evidence with
  | .ok p, .ok c, .ok e =>
    match p.base.challenger with
    | some (_, label) =>
      match roundsIn label c e bs with
      | .ok (uses, _, _, _, coins) => Vocab.recsOf uses.toList coins
      | .error _ => []
    | none => []
  | _, _, _ => []

/-- **The records the finder runs on in what was sent**: each of the components' cases' (`recsCase`), where the
registration's parameters and evidence decode. -/
def recsSent (s : Sent) : List (String × Json) :=
  match canonicalJson s.params >>= Core.Composition.decodeRegisteredParams,
      Core.Composition.decodeRegisteredEvidence s.evidence with
  | .ok (_, ps), .ok (_, ev) =>
    (Properties.Composition.triples s.components ps ev).flatMap fun x => recsCase x.2.1 x.2.2 (Blobs.ofArray s.blobs)
  | _, _ => []

/-- `recsSent` at the play's outcome, as `Vocab.colAt` and `Vocab.forgerAt` read records. -/
def recsAt (_θ : ByteArray) (o : Outcome) : List (String × Json) := recsSent o.sent

/-- **Every accepting case's records are among `recsSent`'s**: where sampled proofs' case decodes to `(p, c, e)`, its
parameters name a challenger of label `label`, and `roundsIn` reads the uses `uses` and coins `coins`, each of their
records is one `recsSent` runs on. -/
theorem mem_recsSent {s : Sent} {p : _root_.SampledProofs.OneStageZk.Params}
    {c : _root_.SampledProofs.OneStage.Commitment} {e : _root_.SampledProofs.OneStageZk.Evidence}
    (hcase : SampledCase s p c e) {key : _root_.SampledProofs.Challenger.ChallengerKey} {label : String}
    (hc : p.base.challenger = some (key, label)) {uses : Array _root_.SampledProofs.Challenger.Use}
    {wants : Array (Option (String × Array String))} {order : Array (String × String)} {groups : Array (List String)}
    {coins : Json} (hin : roundsIn label c e (Blobs.ofArray s.blobs) = .ok (uses, wants, order, groups, coins))
    {r : String × Json} (hr : r ∈ Vocab.recsOf uses.toList coins) : r ∈ recsSent s := by
  obtain ⟨n, ps, b, dg, k, ev, hD, nm, cs, hx, hp, hcm, he⟩ := hcase
  have hrc : r ∈ recsCase nm cs (Blobs.ofArray s.blobs) := by
    simpa only [recsCase, hp, hcm, he, hc, hin] using hr
  simp only [recsSent, hD.1, hD.2.2]
  exact List.mem_flatMap.2 ⟨_, hx, hrc⟩

/-- **Every accepting case names the device's registered key**: where sampled proofs' verdict accepts the play that
ended with `o` after `θ`, any of its sampled cases whose parameters name a challenger names `θ`'s registered key. -/
theorem key_of_case (V : Vocab d α) {θ : ByteArray} {o : Outcome} (hθ : o.params = θ) (hv : V.Verdict .sampled o)
    {p : _root_.SampledProofs.OneStageZk.Params} {c : _root_.SampledProofs.OneStage.Commitment}
    {e : _root_.SampledProofs.OneStageZk.Evidence} (hcase : SampledCase o.sent p c e)
    {key : _root_.SampledProofs.Challenger.ChallengerKey} {label : String}
    (hc : p.base.challenger = some (key, label)) : key = Vocab.keyAt θ := by
  obtain ⟨-, hra, -⟩ := hv
  obtain ⟨n, ps, b, dg, k, ev, hD, nm, cs, hx, hp, -, -⟩ := hcase
  obtain ⟨n', ps', b', d', k', ev', hD', _, _, hreg⟩ := hra
  have hps : ps' = ps := congrArg Prod.snd (Except.ok.inj (hD'.1.symm.trans hD.1))
  obtain ⟨hfit, -⟩ := hreg.2.2.2.2.2.2
  rw [hps] at hfit
  have hn : nm ∈ ps := (List.of_mem_zip (List.of_mem_zip hx).2).1
  have hid : nm.id = _root_.SampledProofs.OneStageZk.PROTOCOL := OnsiteIds.id_of_triples hfit.1 _ hx
  have hfind := OnsiteIds.find_of_unique hn hid (by rw [hfit.1]; exact OnsiteIds.onsite_count _)
  have hkey : Properties.Challenger.keyOfParams ps = some key :=
    Properties.Challenger.keyOfParams_eq_some.2 ⟨nm, p, label, hfind, hp, hc⟩
  have hd : (canonicalJson θ >>= Core.Composition.decodeRegisteredParams) = .ok (n, ps) := by
    rw [← hθ]
    exact hD.1
  have hk : registeredKey θ = Properties.Challenger.lmsKey (some key) := by
    unfold registeredKey
    rw [hd, ← hkey]
    rfl
  show key = ⟨(registeredKey θ).1, (registeredKey θ).2⟩
  rw [hk]
  rfl

/-- **A break on an accepting case's uses is counted**: where the sampled-proofs forger is `forgerAt recsAt` (`hf`), a
break at the play's device under the registered key on the record of one of the uses `roundsIn` reads in a sampled
case of what was sent is the collision `colAt recsAt pairs` reads off the outcome or the forger's success. -/
theorem brk_of_break (V : Vocab d α) {pairs : ByteArray → Outcome → List (ByteArray × ByteArray)}
    (hf : ∀ θ o, V.forger .sampled θ o = V.forgerAt recsAt θ o) {θ : ByteArray} {o : Outcome}
    {p : _root_.SampledProofs.OneStageZk.Params} {c : _root_.SampledProofs.OneStage.Commitment}
    {e : _root_.SampledProofs.OneStageZk.Evidence} (hcase : SampledCase o.sent p c e)
    {key : _root_.SampledProofs.Challenger.ChallengerKey} {label : String} (hc : p.base.challenger = some (key, label))
    {uses : Array _root_.SampledProofs.Challenger.Use} {wants : Array (Option (String × Array String))}
    {order : Array (String × String)} {groups : Array (List String)} {coins : Json}
    (hin : roundsIn label c e (Blobs.ofArray o.sent.blobs) = .ok (uses, wants, order, groups, coins))
    (hk : key = Vocab.keyAt θ)
    (hb : ∃ (i : ℕ) (u : _root_.SampledProofs.Challenger.Use), uses.toList[i]? = some u ∧
      Properties.Challenger.Break (V.deviceOf θ o) key u.label (Properties.Challenger.useCoins coins i)) :
    ColEvent (V.colAt recsAt pairs θ) o ∨ V.Forges .sampled θ o := by
  subst hk
  obtain ⟨i, u, hu, hb⟩ := hb
  obtain ⟨hi, -⟩ := List.getElem?_eq_some_iff.1 hu
  have hm : (u.label, Properties.Challenger.useCoins coins i) ∈ Vocab.recsOf uses.toList coins := by
    unfold Vocab.recsOf
    exact List.mem_filterMap.2 ⟨i, List.mem_range.2 hi, by simp [hu]⟩
  exact V.break_counted (hf θ o) (mem_recsSent hcase hc hin hm) hb

/-- **Off the counted break, an accepting case's uses have no break**: where sampled proofs' verdict accepts the play
that ended with `o` after `θ`, the forger is `forgerAt recsAt` and neither `colAt recsAt pairs`'s collision nor the
forger's success occurs, no record of the uses `roundsIn` reads in a sampled case of what was sent has a break at the
play's device under the key that case names. -/
theorem no_break (V : Vocab d α) {pairs : ByteArray → Outcome → List (ByteArray × ByteArray)}
    (hf : ∀ θ o, V.forger .sampled θ o = V.forgerAt recsAt θ o) {θ : ByteArray} {o : Outcome} (hθ : o.params = θ)
    (hv : V.Verdict .sampled o) (hb : ¬ (ColEvent (V.colAt recsAt pairs θ) o ∨ V.Forges .sampled θ o))
    {p : _root_.SampledProofs.OneStageZk.Params} {c : _root_.SampledProofs.OneStage.Commitment}
    {e : _root_.SampledProofs.OneStageZk.Evidence} (hcase : SampledCase o.sent p c e)
    {key : _root_.SampledProofs.Challenger.ChallengerKey} {label : String} (hc : p.base.challenger = some (key, label))
    {uses : Array _root_.SampledProofs.Challenger.Use} {wants : Array (Option (String × Array String))}
    {order : Array (String × String)} {groups : Array (List String)} {coins : Json}
    (hin : roundsIn label c e (Blobs.ofArray o.sent.blobs) = .ok (uses, wants, order, groups, coins)) :
    ¬ ∃ (i : ℕ) (u : _root_.SampledProofs.Challenger.Use), uses.toList[i]? = some u ∧
      Properties.Challenger.Break (V.deviceOf θ o) key u.label (Properties.Challenger.useCoins coins i) :=
  fun h => hb (brk_of_break V hf hcase hc hin (key_of_case V hθ hv hcase hc) h)

end Verity.Canon
