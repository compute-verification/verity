module

public import Proofs.Flock.ZkAny.RelayStop
public import Proofs.Flock.ZkAny.RelayMask

/-!
# A round's coins keep the session's stop

Between a round of a session's rep and its coins, the loop's frames hold one frame at the rep's key: the round's put
sets every frame at the key to the round's, a start pushes a frame only at a key with none (`startAt`), and the other
updates that keep the rep's frame (`KeptStep`) touch no frame at its key. The frame's take of the coins neither reads
nor writes `stopped`, so their put leaves the session's stop as the round's put left it:

* `OnlyAt`: every frame at `p`'s key is `p`. `onlyAt_put`: a put sets it at the put's key; `onlyAt_keptStep`,
  `onlyAt_kept`: the updates that keep the rep's frame keep it.
* `stopped_put_onlyAt`: there, a put at that key of a frame with `p`'s `stopped` leaves every session's stop as it was.
* `coins_stopped`: the frame's take of a round's coins keeps `stopped` (`answer_stopped`).
* **`stopped_coins`**: so the coins' put after a round (`Frames.coinsStep`'s `Loop.coinsOf`) leaves the session's stop
  as it was.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (F128)
open _root_.Flock.Firewall.Frames (Sessions Rep)

/-- **Every frame at `p`'s key** in the loop's frames `R` is `p`. -/
def OnlyAt (R : Sessions) (p : Rep) : Prop := ∀ q ∈ R.reps, keyOf q = keyOf p → q = p

/-- A frame `Sessions.put` replaces has the put's key. -/
theorem keyOf_of_put {p q : Rep} (h : (q.sid == p.sid && q.table == p.table && q.rep == p.rep) = true) :
    keyOf q = keyOf p := by
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨h1, h2⟩, h3⟩ := h
  simp [keyOf, h1, h2, h3]

/-- **A put sets every frame at its key.** -/
theorem onlyAt_put (R : Sessions) (p : Rep) : OnlyAt (R.put p) p := by
  intro q hq hk
  unfold _root_.Flock.Firewall.Frames.Sessions.put at hq
  simp only [Array.mem_map] at hq
  obtain ⟨q₀, -, e⟩ := hq
  split at e
  · exact e.symm
  · rename_i hn
    rw [← e] at hk
    exfalso
    apply hn
    simp only [keyOf, Prod.mk.injEq] at hk
    obtain ⟨h1, h2, h3⟩ := hk
    simp [h1, h2, h3]

/-- **An update that keeps the rep's frame keeps it the only one at its key**, where it has one. -/
theorem onlyAt_keptStep {sid i r : Nat} {R R' : Sessions} {p : Rep} (h : KeptStep sid i r R R')
    (hk : keyOf p = (sid, i, r)) (hs : (frameAt R sid i r).isSome) (ha : OnlyAt R p) : OnlyAt R' p := by
  cases h with
  | begin => exact ha
  | link => exact ha
  | start q =>
    unfold startAt
    split
    · exact ha
    · rename_i hn
      intro q' hq' hkq
      rcases Array.mem_push.mp hq' with hq' | hq'
      · exact ha q' hq' hkq
      · rw [hq'] at hkq
        exfalso
        apply hn
        have e : keyOf q = (sid, i, r) := hkq.trans hk
        simp only [keyOf, Prod.mk.injEq] at e
        obtain ⟨e1, e2, e3⟩ := e
        rw [e1, e2, e3]
        rw [← find_frameAt] at hs
        simpa using hs
  | put p' hk' =>
    intro q hq hkq
    unfold _root_.Flock.Firewall.Frames.Sessions.put at hq
    simp only [Array.mem_map] at hq
    obtain ⟨q₀, hq₀, e⟩ := hq
    split at e
    · rw [← e] at hkq
      exact absurd (hkq.trans hk) hk'
    · rw [← e] at hkq ⊢
      exact ha q₀ hq₀ hkq
  | keep =>
    unfold keepOf
    split <;> exact ha
  | file root r' b hf =>
    unfold _root_.Flock.Firewall.Frames.Sessions.file at hf
    split at hf
    · cases hf
    · cases hf
      exact ha
  | inner root r' j b hf =>
    unfold _root_.Flock.Firewall.Frames.Sessions.innerRound at hf
    simp only at hf
    split at hf
    · cases hf
    · split at hf <;> (cases hf; exact ha)
  | close sid' roots hne =>
    intro q hq hkq
    unfold closeOf at hq
    exact ha q (Array.mem_filter.mp hq).1 hkq

/-- **Updates that keep the rep's frame keep it the only one at its key**, where it has one. -/
theorem onlyAt_kept {sid i r : Nat} {R R' : Sessions} {p : Rep} (h : Kept sid i r R R') (hk : keyOf p = (sid, i, r))
    (hs : (frameAt R sid i r).isSome) (ha : OnlyAt R p) : OnlyAt R' p := by
  induction h with
  | refl => exact ha
  | step h₁ hst ih =>
    exact onlyAt_keptStep hst hk (by rw [frameAt_kept h₁ hs]; exact hs) (by first | exact ih | exact ih hs ha)

/-- **A put at a key held by one frame, of a frame with its `stopped`, leaves every session's stop as it was.** -/
theorem stopped_put_onlyAt {R : Sessions} {p₀ p : Rep} (ha : OnlyAt R p₀) (hk : keyOf p = keyOf p₀)
    (hs : p.st.m.stopped = p₀.st.m.stopped) (sid : Nat) : (R.put p).stopped sid = R.stopped sid := by
  have hsid : p.sid = p₀.sid := by
    simp only [keyOf, Prod.mk.injEq] at hk
    exact hk.1
  apply Bool.eq_iff_iff.mpr
  rw [stopped_iff, stopped_iff]
  constructor
  · rintro ⟨q, hq, hqs, hqst⟩
    unfold _root_.Flock.Firewall.Frames.Sessions.put at hq
    simp only [Array.mem_map] at hq
    obtain ⟨q₀, hq₀, e⟩ := hq
    split at e
    · rename_i hm
      rw [← e] at hqs hqst
      have e₀ := ha q₀ hq₀ ((keyOf_of_put hm).trans hk)
      refine ⟨q₀, hq₀, ?_, ?_⟩
      · rw [e₀, ← hsid]
        exact hqs
      · rw [e₀, ← hs]
        exact hqst
    · rw [← e] at hqs hqst
      exact ⟨q₀, hq₀, hqs, hqst⟩
  · rintro ⟨q, hq, hqs, hqst⟩
    unfold _root_.Flock.Firewall.Frames.Sessions.put
    by_cases hm : (q.sid == p.sid && q.table == p.table && q.rep == p.rep) = true
    · have e₀ := ha q hq ((keyOf_of_put hm).trans hk)
      refine ⟨p, Array.mem_map.mpr ⟨q, hq, by simp [hm]⟩, ?_, ?_⟩
      · rw [hsid, ← e₀]
        exact hqs
      · rw [hs, ← e₀]
        exact hqst
    · exact ⟨q, Array.mem_map.mpr ⟨q, hq, by simp [hm]⟩, hqs, hqst⟩

/-- **The frame's take of a round's coins keeps `stopped`.** -/
theorem coins_stopped {c : _root_.Flock.Firewall.Frame.Cfg} {st st' : _root_.Flock.Firewall.Frame.State}
    {cs : Array F128} (h : _root_.Flock.Firewall.Frame.step c st (.coins (.ok cs)) = (.ok, st')) :
    st'.m.stopped = st.m.stopped := by
  have e := answer_stopped c st.m cs st.m.stopped
  have e2 : ({ st.m with stopped := st.m.stopped } : _root_.Flock.Firewall.Frame.Masker).answer c cs =
      st.m.answer c cs := rfl
  rw [e2, _root_.Flock.Firewall.Frame.answer_of_step h] at e
  simp only [Except.map, Except.ok.injEq] at e
  have e' := congrArg _root_.Flock.Firewall.Frame.Masker.stopped e
  exact e'

/-- **A round's coins keep the session's stop**: where the rep has a frame when its round reads the loop's frames
(`R`), only updates that keep it come before the round puts back `p₁` (on `R₁`) and between that put and the coins'
read (`R₂`), the put of the frame's take of the coins `cs` (`Loop.coinsOf`) leaves every session's stop as it was. -/
theorem stopped_coins {R R₁ R₂ : Sessions} {sid i r : Nat} {p₁ : Rep} (hfr : (frameAt R sid i r).isSome)
    (hk : keyOf p₁ = (sid, i, r)) (h₀₁ : Kept sid i r R R₁) (h₁₂ : Kept sid i r (R₁.put p₁) R₂) {cs : Array F128}
    {st₂ : _root_.Flock.Firewall.Frame.State}
    (hcs : _root_.Flock.Firewall.Frame.step p₁.cfg p₁.st (.coins (.ok cs)) = (.ok, st₂)) (sid' : Nat) :
    (R₂.put (Loop.coinsOf p₁ st₂ cs)).stopped sid' = R₂.stopped sid' := by
  have hp : frameAt (R₁.put p₁) sid i r = some p₁ :=
    frameAt_put_self R₁ p₁ hk (by rw [frameAt_kept h₀₁ hfr]; exact hfr)
  have ha : OnlyAt R₂ p₁ := onlyAt_kept h₁₂ hk (by rw [hp]; rfl) (onlyAt_put R₁ p₁)
  exact stopped_put_onlyAt (p := Loop.coinsOf p₁ st₂ cs) ha rfl (coins_stopped hcs) sid'

end Claims.ZKServer.Relay
