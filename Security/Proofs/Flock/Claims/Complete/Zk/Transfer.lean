import Proofs.Flock.Claims.Complete.Replay.Steps
import Flock.With.ZkEq
import Flock.With.SimWalk

/-!
# A replay that returns on the prover's rounds returns on rounds that carry them

The masked prover replays its own zerocheck and lincheck (`Zk.replay`, inside `repPZk`) on its rounds; the verifier
replays them (`Zk.replayWith`) on the record's. A replay reads a round's count, coins and retained bytes only, and
compares retained bytes without hashing (`squeezeWith`), so it returns the same on any rounds that carry the
prover's, whatever the hashes.

* **Carrying** (`Carry t u`): `u` is at `t`'s round with `t`'s pending bytes, and every round of `t` is retained and
  carried by `u`'s (`Replay.Carries`).
* **The simulation** (`xferS`): a run from `t` that returns is matched by one from any `u` carrying `t`, with the same
  result, ending where it carries the first's end. The squeezes (`squeezeWith_xfer`) and the transcript actions that
  read only the pending bytes (`tm_xfer`) are related, so the replay is (`replayWith_xfer`).
-/

namespace FlockComplete

open Flock Transcript FlockComplete.Replay

/-- **`u` carries `t`**: the same round and pending bytes, and every round of `t` retained and carried by `u`'s. -/
def Carry (t u : Transcript) : Prop :=
  u.k = t.k ∧ u.pending = t.pending ∧
    ∀ j (h : j < t.rounds.size), t.rounds[j].bytes.isSome ∧ Carries u.rounds j t.rounds[j]

theorem stateT_bind_ok {σ α β : Type} {x : StateT σ V α} {f : α → StateT σ V β} {s s' : σ} {b : β}
    (h : (x >>= f) s = .ok (b, s')) : ∃ a s₁, x s = .ok (a, s₁) ∧ f a s₁ = .ok (b, s') := by
  change Except.bind (x s) _ = _ at h
  rcases e : x s with _ | ⟨a, s₁⟩
  · rw [e] at h; cases h
  · rw [e] at h; exact ⟨a, s₁, rfl, h⟩

theorem stateT_bind_of {σ α β : Type} {x : StateT σ V α} {f : α → StateT σ V β} {s s₁ : σ} {a : α}
    (h : x s = .ok (a, s₁)) : (x >>= f) s = f a s₁ := by
  change Except.bind (x s) _ = _
  rw [h]
  rfl

/-- **The transfer simulation**: a run from `t` that returns is matched, from any `u` carrying `t`, by a run with the
same result that ends carrying the first's end. -/
def xferS : With.Sim (TMW V) (TMW V) where
  R x y := ∀ t u a t', Carry t u → x t = .ok (a, t') → ∃ u', y u = .ok (a, u') ∧ Carry t' u'
  pure a t u a' t' hc h := by
    change Except.ok (a, t) = _ at h
    cases h
    exact ⟨u, rfl, hc⟩
  bind {α β x y f g} hxy hfg t u b t'' hc h := by
    obtain ⟨a, t', h1, h2⟩ := stateT_bind_ok h
    obtain ⟨u', e1, c1⟩ := hxy t u a t' hc h1
    obtain ⟨u'', e2, c2⟩ := hfg a t' u' b t'' c1 h2
    exact ⟨u'', by rw [stateT_bind_of e1]; exact e2, c2⟩
  throw e t u a t' hc h := by cases h
  lift v t u a t' hc h := by
    change Except.bind v _ = _ at h
    cases v with
    | error _ => cases h
    | ok c =>
      change Except.ok (c, t) = _ at h
      cases h
      exact ⟨u, rfl, hc⟩

/-- **A transcript action that reads only the pending bytes** and keeps the rounds is related to itself. -/
theorem tm_xfer {α : Type} (x : TM α)
    (hx : ∀ t a t', x t = .ok (a, t') → t'.rounds = t.rounds ∧
      ∀ u : Transcript, u.k = t.k → u.pending = t.pending → x u = .ok (a, ⟨u.rounds, t'.k, t'.pending⟩)) :
    xferS.R (liftTM x : TMW V α) (liftTM x) := by
  intro t u a t' hc h
  obtain ⟨hr, hu⟩ := hx t a t' h
  refine ⟨_, hu u hc.1 hc.2.1, rfl, rfl, ?_⟩
  rw [hr]
  exact hc.2.2

/-- An action that only rewrites the pending bytes. -/
theorem pending_xfer (x : TM Unit) (g : ByteArray → ByteArray)
    (hx : ∀ t : Transcript, x t = .ok ((), { t with pending := g t.pending })) :
    xferS.R (liftTM x : TMW V Unit) (liftTM x) :=
  tm_xfer _ fun t a t' h => by
    rw [hx] at h
    cases h
    refine ⟨rfl, fun u hk hp => ?_⟩
    rw [hx, ← hk, ← hp]

theorem label_xfer (s : String) : xferS.R (liftTM (label s) : TMW V Unit) (liftTM (label s)) :=
  pending_xfer _ (fun P => P ++ header 2 0 s.toUTF8.size ++ padded s.toUTF8) (Flock.RecAlgebra.label_eq s)

theorem absorbF_xfer (v : F128) : xferS.R (liftTM (absorbF v) : TMW V Unit) (liftTM (absorbF v)) :=
  pending_xfer _ (fun P => F128.pushBytes (P ++ header 3 1 1) v) (Flock.RecAlgebra.absorbF_eq v)

theorem absorbSlice_xfer (vs : Array F128) : xferS.R (liftTM (absorbSlice vs) : TMW V Unit) (liftTM (absorbSlice vs)) :=
  pending_xfer _ (fun P => vs.foldl F128.pushBytes (P ++ header 3 2 vs.size))
    (Flock.RecAlgebra.absorbSlice_eq vs)

theorem absorbBytes_xfer (b : ByteArray) : xferS.R (liftTM (absorbBytes b) : TMW V Unit) (liftTM (absorbBytes b)) :=
  pending_xfer _ (fun P => P ++ header 5 0 b.size ++ padded b) (Flock.RecAlgebra.absorbBytes_eq b)

theorem check_xfer (c : Bool) (why : String) :
    xferS.R (liftTM (Flock.check c why) : TMW V Unit) (liftTM (check c why)) :=
  tm_xfer _ fun t a t' h => by
    cases c
    · cases h
    · rw [Flock.RecAlgebra.check_eq rfl] at h
      cases h
      exact ⟨rfl, fun u hk hp => by rw [Flock.RecAlgebra.check_eq rfl, ← hk, ← hp]⟩

/-- **A squeeze is related to itself under any two oracles**: a returning squeeze read a retained round, whose bytes
are compared without hashing. -/
theorem squeezeWith_xfer (hs₁ hs₂ : Hashes V) (kind : UInt8) (n : ℕ) :
    xferS.R (squeezeWith hs₁ kind n) (squeezeWith hs₂ kind n) := by
  intro t u a t' hc h
  obtain ⟨hk, hp, hR⟩ := hc
  unfold squeezeWith at h ⊢
  by_cases hn : n = 0
  · subst hn
    simp only [ite_true] at h ⊢
    change Except.ok _ = _ at h
    cases h
    refine ⟨_, rfl, hk, by simp [framed, hp], fun j hj => hR j hj⟩
  · simp only [hn, ite_false] at h ⊢
    rcases e : t.rounds[t.k]? with _ | r
    · rw [e] at h; cases h
    rw [e] at h
    obtain ⟨hlt, rfl⟩ := Array.getElem?_eq_some_iff.1 e
    obtain ⟨hb, r', hr', hN, hC, hB⟩ := hR t.k hlt
    obtain ⟨b, hbs⟩ := Option.isSome_iff_exists.1 hb
    rw [hk, hr']
    dsimp only at h ⊢
    by_cases hN' : t.rounds[t.k].n ≠ n
    · rw [ite_eq_left hN'] at h; cases h
    rw [ite_eq_right hN'] at h
    rw [ite_eq_right (by rw [hN]; exact hN')]
    unfold Round.answersWith at h ⊢
    rw [hbs] at h
    rw [hB, hbs]
    simp only [pure_bind] at h ⊢
    rw [show framed u kind n = framed t kind n by simp [framed, hp]]
    by_cases hf : framed t kind n = b
    · simp only [hf, decide_true, Bool.not_true, Bool.false_eq_true, ite_false] at h ⊢
      change Except.ok _ = _ at h
      cases h
      refine ⟨⟨u.rounds, t.k + 1, ByteArray.empty⟩, by rw [hC]; rfl, rfl, rfl, fun j hj => hR j hj⟩
    · simp only [hf, decide_false, Bool.not_false, ite_true] at h
      cases h

theorem sampleWith_xfer (hs₁ hs₂ : Hashes V) : xferS.R (sampleWith hs₁) (sampleWith hs₂) := by
  unfold sampleWith
  sim_walk [squeezeWith_xfer hs₁ hs₂]

theorem sampleVecWith_xfer (hs₁ hs₂ : Hashes V) (n : ℕ) : xferS.R (sampleVecWith hs₁ n) (sampleVecWith hs₂ n) :=
  squeezeWith_xfer hs₁ hs₂ 2 n

/-- **The masked replay is related to itself under any two oracles.** -/
theorem replayWith_xfer (hs₁ hs₂ : Hashes V) (st : Setup) (lay : Zk.PadLayout) (p : Proof) (root : ByteArray) :
    xferS.R (Zk.replayWith hs₁ st lay p root) (Zk.replayWith hs₂ st lay p root) := by
  unfold Zk.replayWith
  sim_walk [sampleWith_xfer hs₁ hs₂, sampleVecWith_xfer hs₁ hs₂, label_xfer, absorbF_xfer, absorbSlice_xfer,
    absorbBytes_xfer, check_xfer]

end FlockComplete
