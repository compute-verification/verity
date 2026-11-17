module

public import Proofs.Flock.SessionLoop.RepSrc
public import Proofs.Flock.Claims.ZK.FinishSwRep
public import Proofs.Flock.Claims.ZK.FinExe

/-! Where `src` samples the tape's level 0, the loop's tables hold the tape's packed table, lanes and salts
(`Loop.Level0Src.realOf`, `Loop.RepSrc.salt`). -/

@[expose] public section

namespace Flock.ZkServer.Finish

theorem tabOfLevel0_fields {st : _root_.Flock.Setup} {l : _root_.Flock.ZkServer.Level0.Level0}
    {t : _root_.Flock.ZkServer.Finish.Tab} (h : _root_.Flock.ZkServer.Finish.tabOfLevel0 st l = .ok t) :
    t.g = l.toIn.g ∧ t.extra = l.toIn.extra ∧ t.lanePad = l.toIn.lanePad := by
  unfold _root_.Flock.ZkServer.Finish.tabOfLevel0 at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  rcases hs : _root_.Flock.Zk.schedule st.wideLeaf st.m with e | ⟨sch, tPad⟩
  · rw [hs] at h; cases h
  · rw [hs] at h
    simp only at h
    split at h
    · cases h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨rfl, rfl, rfl⟩

end Flock.ZkServer.Finish

namespace Claims.ZKMal.Masked

open Claims FlockSoundness.Discharge

theorem linkIn_g (x : Msg) (w : Claims.Flock.Wit) (t : UE.Tape) (j : Fin (jOf x)) :
    (linkIn x w t j).g = ligTab x w t j := by
  have hj : j.val < (FlockDeploy.tabsOf (stsOf x)).size := by
    rw [tabs_size (live_of_lt j)]
    exact j.isLt
  show (FlockComplete.tablesP (FlockDeploy.tabsOf (stsOf x)) (trsE x w (UE.tapeZk t)))[j.val]! = _
  unfold FlockComplete.tablesP
  rw [getElem!_pos _ j.val (by simpa using hj), Array.getElem_map, Array.getElem_range]

end Claims.ZKMal.Masked

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

theorem Loop.Level0Src.realOf {src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray)} {x : Msg} {w : Flock.Wit}
    (hS : Loop.Level0Src src x w) (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce)
    (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape) (j : Fin (Masked.jOf x)) (r : Fin 2)
    {tb : _root_.Flock.ZkServer.Finish.Tab} {hi : Array ByteArray}
    (h : Loop.realOf src x w (ν, t) (2 * j.val + r.val) = .ok (tb, hi)) :
    tb.g = (Masked.linkIn x w t j).g ∧ tb.extra = (Masked.linkIn x w t j).extra ∧
      tb.lanePad = (Masked.linkIn x w t j).lanePad ∧ tb.g = Masked.ligTab x w t j := by
  obtain ⟨l, hl, hin⟩ := hS ν t j r
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) (Masked.lt_of_jOf j))
  unfold Loop.realOf at h
  rw [hsx] at h
  dsimp only at h
  unfold Loop.level0At at hl
  rcases hs : src x w (ν, t) (2 * j.val + r.val) with e | ⟨g, d, hi'⟩
  · rw [hs] at hl
    cases hl
  · rw [hs] at h hl
    simp only [bind, Except.bind] at h hl
    rw [hl] at h
    simp only at h
    split at h
    · cases h
    · rename_i tb' htb
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, -⟩ := h
      obtain ⟨hg, he, hp⟩ := _root_.Flock.ZkServer.Finish.tabOfLevel0_fields htb
      rw [← hin]
      exact ⟨hg, he, hp, hg.trans ((congrArg _root_.Flock.ZkServer.Link.In.g hin).trans (Masked.linkIn_g x w t j))⟩

theorem Loop.Level0Src.realOf_lanes {src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray)} {x : Msg} {w : Flock.Wit}
    (hS : Loop.Level0Src src x w) (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce)
    (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape) (j : Fin (Masked.jOf x)) (r : Fin 2)
    {tb : _root_.Flock.ZkServer.Finish.Tab} {hi : Array ByteArray}
    (h : Loop.realOf src x w (ν, t) (2 * j.val + r.val) = .ok (tb, hi)) :
    (∀ e c, c < 2 ^ (Masked.lv0 x j).cols → tb.extra e c = (Masked.UE.tapeZk t).extra j.val e c) ∧
      ∀ l i, l < 2 ^ (Masked.lv0 x j).lanes + _root_.Flock.Zk.EXTRA_LANES → i < (Masked.zkTab x j).tPad →
        tb.lanePad l i = (Masked.UE.tapeZk t).lanePad j.val l i := by
  obtain ⟨-, he, hp, -⟩ := hS.realOf ν t j r h
  exact ⟨fun e c hc => by rw [he]; exact if_pos hc, fun l i hl hi => by rw [hp]; exact if_pos ⟨hl, hi⟩⟩

theorem Loop.realOf_src {R : Type} {src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray)}
    {x : Msg} {w : Flock.Wit} {ρ : R} {k : ℕ} {tb : _root_.Flock.ZkServer.Finish.Tab} {hi : Array ByteArray}
    (h : Loop.realOf src x w ρ k = .ok (tb, hi)) : ∃ g d, src x w ρ k = .ok (g, d, hi) := by
  unfold Loop.realOf at h
  split at h
  · rcases hs : src x w ρ k with e | ⟨g, d, hi'⟩
    · rw [hs] at h
      cases h
    · rw [hs] at h
      simp only [bind, Except.bind] at h
      split at h
      · cases h
      · split at h
        · cases h
        · simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
          exact ⟨g, d, by rw [h.2]⟩
  · cases h

theorem Loop.RepSrc.salt {L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)}
    {src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray)} {x : Msg} {w : Flock.Wit}
    (hS : Loop.RepSrc L₀ src x w) (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce)
    (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape) (j : Fin (Masked.jOf x)) (r : Fin 2)
    {tb : _root_.Flock.ZkServer.Finish.Tab} {hi : Array ByteArray}
    (h : Loop.realOf src x w (ν, t) (2 * j.val + r.val) = .ok (tb, hi)) :
    ∀ l, 1 ≤ l → l < (Masked.zkTab x j).sch.levels.size →
      ∀ p < 2 ^ ((Masked.zkTab x j).sch.levels.getD l default).d,
        _root_.Flock.ZkServer.Finish.saltAt (hi.getD (l - 1) ByteArray.empty) p =
          (Masked.UE.tapeZk t).salt j.val r.val l p := by
  obtain ⟨g, d, hs⟩ := Loop.realOf_src h
  exact (hS ν t j r).2.2.2 g d hi hs

end Claims.ZKServer
