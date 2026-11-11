module

public import Proofs.Flock.SessionLoop.RepSrc
public import Proofs.Flock.Claims.ZK.FinishSwRep
public import Proofs.Flock.Claims.ZK.FinExe

/-!
# The loop's tables at its level-0 source

Where `src` samples the tape's level 0 (`Loop.Level0Src`, `Loop.Src`'s level-0 half), each table the loop's finish proves
on (`Loop.realOf`, `tabOfLevel0` at the level 0 it drew) holds the tape's packed table, and its extra lanes and lane
paddings within level 0's sizes:

* `Finish.tabOfLevel0_fields`: a table from a level 0 holds that level 0's own inputs (`toIn`).
* `Masked.linkIn_g`: the packed table `Masked.linkIn` hands the link is `Masked.ligTab`, the one `swSrcE_repCut`'s `hg`
  names.
* **`Loop.Level0Src.realOf`**: at rep stream `2j + r`, the loop's table holds `linkIn`'s packed table, extra lanes and
  lane paddings, so its packed table is `ligTab` (`hg`), and its extra lanes and lane paddings are the tape's within
  level 0's sizes (`Loop.Level0Src.realOf_lanes`).
* `Loop.realOf_src`, **`Loop.RepSrc.salt`**: the loop's levels ≥ 1 salts at rep stream `2j + r` are the source's, so,
  where `src` samples them as the tape's (`Loop.RepSrc`), they are the tape's on the schedule (`swSrcE_repCut`'s `hsalt`).
-/

@[expose] public section

namespace Flock.Firewall.Finish

/-- **A table from a level 0 holds its inputs**: its packed table, extra lanes and lane paddings are the level 0's
(`Level0.toIn`). -/
theorem tabOfLevel0_fields {st : _root_.Flock.Setup} {l : _root_.Flock.Firewall.Level0.Level0}
    {t : _root_.Flock.Firewall.Finish.Tab} (h : _root_.Flock.Firewall.Finish.tabOfLevel0 st l = .ok t) :
    t.g = l.toIn.g ∧ t.extra = l.toIn.extra ∧ t.lanePad = l.toIn.lanePad := by
  unfold _root_.Flock.Firewall.Finish.tabOfLevel0 at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  split at h
  · cases h
  rcases hs : _root_.Flock.Zk.schedule st.m with e | ⟨sch, tPad⟩
  · rw [hs] at h; cases h
  · rw [hs] at h
    simp only at h
    split at h
    · cases h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨rfl, rfl, rfl⟩

end Flock.Firewall.Finish

namespace Claims.ZKMal.Masked

open Claims FlockSoundness.Discharge

/-- **The link's packed table is `ligTab`**: `linkIn`'s, at a table of the session. -/
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

/-- **The loop's table is the tape's** where `src` samples the tape's level 0: at rep stream `2j + r`, where the loop's
table (`Loop.realOf`) is `tb`, its packed table, extra lanes and lane paddings are `Masked.linkIn`'s, so its packed
table is `Masked.ligTab` (`swSrcE_repCut`'s `hg`). -/
theorem Loop.Level0Src.realOf {src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray)} {x : Msg} {w : Flock.Wit}
    (hS : Loop.Level0Src src x w) (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce)
    (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape) (j : Fin (Masked.jOf x)) (r : Fin 2)
    {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray}
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
      obtain ⟨hg, he, hp⟩ := _root_.Flock.Firewall.Finish.tabOfLevel0_fields htb
      rw [← hin]
      exact ⟨hg, he, hp, hg.trans ((congrArg _root_.Flock.Firewall.Link.In.g hin).trans (Masked.linkIn_g x w t j))⟩

/-- **The loop's table's extra lanes and lane paddings are the tape's within level 0's sizes** (`swSrcE_repCut_v`'s
`hex` and `hlp`), where `src` samples the tape's level 0: each extra lane below `2^cols`, and each of the `2^lanes + 4`
lanes' paddings below `t_pad`. -/
theorem Loop.Level0Src.realOf_lanes {src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray)} {x : Msg} {w : Flock.Wit}
    (hS : Loop.Level0Src src x w) (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce)
    (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape) (j : Fin (Masked.jOf x)) (r : Fin 2)
    {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray}
    (h : Loop.realOf src x w (ν, t) (2 * j.val + r.val) = .ok (tb, hi)) :
    (∀ e c, c < 2 ^ (Masked.lv0 x j).cols → tb.extra e c = (Masked.UE.tapeZk t).extra j.val e c) ∧
      ∀ l i, l < 2 ^ (Masked.lv0 x j).lanes + _root_.Flock.Zk.EXTRA_LANES → i < (Masked.zkTab x j).tPad →
        tb.lanePad l i = (Masked.UE.tapeZk t).lanePad j.val l i := by
  obtain ⟨-, he, hp, -⟩ := hS.realOf ν t j r h
  exact ⟨fun e c hc => by rw [he]; exact if_pos hc, fun l i hl hi => by rw [hp]; exact if_pos ⟨hl, hi⟩⟩

/-- **The loop's rep salts are the source's**: where the loop's table and salts at rep stream `k` are `(tb, hi)`, the source
drew `hi` there. -/
theorem Loop.realOf_src {R : Type} {src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray)}
    {x : Msg} {w : Flock.Wit} {ρ : R} {k : ℕ} {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray}
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

/-- **The loop's levels ≥ 1 salts are the tape's on the schedule** where `src` samples them as the tape's (`Loop.RepSrc`'s
last part): at rep stream `2j + r`, the salts the loop's finish reads (`Finish.proofFile`'s `salt`) are, at every level
`1 ≤ l < |levels|` and position of its tree, the tape's `salt j r l p` (`swSrcE_repCut`'s `hsalt`). -/
theorem Loop.RepSrc.salt {L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)}
    {src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray)} {x : Msg} {w : Flock.Wit}
    (hS : Loop.RepSrc L₀ src x w) (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce)
    (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape) (j : Fin (Masked.jOf x)) (r : Fin 2)
    {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray}
    (h : Loop.realOf src x w (ν, t) (2 * j.val + r.val) = .ok (tb, hi)) :
    ∀ l, 1 ≤ l → l < (Masked.zkTab x j).sch.levels.size →
      ∀ p < 2 ^ ((Masked.zkTab x j).sch.levels.getD l default).d,
        _root_.Flock.Firewall.Finish.saltAt (hi.getD (l - 1) ByteArray.empty) p =
          (Masked.UE.tapeZk t).salt j.val r.val l p := by
  obtain ⟨g, d, hs⟩ := Loop.realOf_src h
  exact (hS ν t j r).2.2.2 g d hi hs

end Claims.ZKServer
