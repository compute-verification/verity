module

public import Proofs.Flock.ZkAny.RelayLoopKept
public import Proofs.Flock.SessionLoop.Statement

/-!
# The level-0 half of the model's tables being the loop's

`RealAt`'s level-0 half, at the loop's own tables (`Loop.realOf src`): each table the level 0 the loop holds for the
session gives is the one `Loop.realOf src` gives the table's reps.

* `tab_of_toIn`: two level 0s `Level0.ofDraws` gave at one `m` with one input (`toIn`) give one table.
* **`real_of_builtAt`**: where every level 0 bound to the session is the one the loop drew for its table from
  `src`'s rep 0 (`BuiltAt`), the rep streams' names distinct, and rep 1's level 0 has rep 0's inputs, at each rep of
  each table the level 0 gives table `t`, `Loop.realOf src` gives `t`.
* `level0Src_rep1`: where the loop's host and operating system hand it the tape's level-0 inputs (`Loop.Level0Src`)
  at a statement the masked session proves (`Masked.Live`), rep 1's level 0 has rep 0's inputs. **`realAt_h0`**: so
  `real_of_builtAt` holds there.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (V F128 SessionSpec Stmt)
open _root_.Flock.Firewall.Frames (Release)

/-- **Two level 0s `Level0.ofDraws` gave at one `m` with one input give one table**: `Finish.tabOfLevel0` reads a level
0's `m`, `t_pad`, input and commitment, which its `m` and input fix (`Level0.ofDraws_ok`). -/
theorem tab_of_toIn {m : Nat} {g g' : Array F128} {d d' : ByteArray} {l l' : _root_.Flock.Firewall.Level0.Level0}
    (hl : _root_.Flock.Firewall.Level0.ofDraws m g d = .ok l)
    (hl' : _root_.Flock.Firewall.Level0.ofDraws m g' d' = .ok l') (he : l.toIn = l'.toIn) (st : _root_.Flock.Setup) :
    _root_.Flock.Firewall.Finish.tabOfLevel0 st l = _root_.Flock.Firewall.Finish.tabOfLevel0 st l' := by
  obtain ⟨hm, -, ⟨s, hz, hlv⟩, hrw, hly, -⟩ := _root_.Flock.Firewall.Level0.ofDraws_ok hl
  obtain ⟨hm', -, ⟨s', hz', hlv'⟩, hrw', hly', -⟩ := _root_.Flock.Firewall.Level0.ofDraws_ok hl'
  obtain ⟨hs, htp⟩ := Prod.mk.inj (Except.ok.inj (hz.symm.trans hz'))
  have hlv2 : l.lv = l'.lv := by rw [hlv, hlv', hs]
  have hrows : l.rows = l'.rows := by rw [hrw, hrw', hlv2, htp, he]
  have hlay : l.layers = l'.layers := by rw [hly, hly', hlv2, htp, he]
  have hc : _root_.Flock.Firewall.Finish.commitOf l = _root_.Flock.Firewall.Finish.commitOf l' := by
    unfold _root_.Flock.Firewall.Finish.commitOf
    rw [hrows, hlay, hlv2, show l.salt = l'.salt from congrArg _root_.Flock.Firewall.Link.In.salt he]
  unfold _root_.Flock.Firewall.Finish.tabOfLevel0
  rw [hm.trans hm'.symm, htp, show l.g = l'.g from congrArg _root_.Flock.Firewall.Link.In.g he, he, hc]

/-- **The level-0 half of `RealAt` at the loop's own tables**, at the loop `σ` for session `sid`, where the model's
samples are `src`'s at its statement, witness and tapes (`hsrc`), the session's entry is the setup's (`SessOn`), the
rep streams' names are distinct (`hnd`), every level 0 bound to the session is the one the loop drew for its table from
rep stream `2j`'s samples (`BuiltAt`), and rep stream `2j + 1`'s samples give a level 0 with its inputs (`h1`): at each
rep `r` of each table `i` of the session's entry, where the level 0 the loop holds for the table gives table `t`,
`Loop.realOf src` gives `t` at rep stream `2i + r` (`tables_inj`, `tab_of_toIn`). -/
theorem real_of_builtAt {Rr : Type} {M : LoopModel Claims.Flock.Wit Rr}
    {src : Msg → Claims.Flock.Wit → Rr → Nat → V (Array F128 × ByteArray × Array ByteArray)} {sid : Nat}
    {σ : LoopAt} {s0 : _} {spec : SessionSpec} {sts : Array Stmt}
    (hsrc : M.src = src M.x M.w M.ρ) (hx : FlockDeploy.setupOf true M.x = some (s0, spec, sts))
    (hK : SessOn spec sts σ.R sid)
    (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup) (hB : BuiltAt M spec sts σ.L0 sid)
    (h1 : ∀ j g d hi l, j < sts.size → M.src (2 * j) = .ok (g, d, hi) →
      _root_.Flock.Firewall.Level0.ofDraws sts[j]!.m g d = .ok l →
      ∃ g' d' hi' l', M.src (2 * j + 1) = .ok (g', d', hi') ∧
        _root_.Flock.Firewall.Level0.ofDraws sts[j]!.m g' d' = .ok l' ∧ l'.toIn = l.toIn) :
    ∀ (i r : Nat) (rel : Release) (name : String) (l0 : _root_.Flock.Firewall.Level0.Level0) (stm : Stmt)
      (t : _root_.Flock.Firewall.Finish.Tab), r < 2 →
      sessAt σ.R sid = some rel → rel.spec.tables[i]? = some name →
      _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0 → rel.sts[i]? = some stm →
      _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t →
      ∃ hi, Loop.realOf src M.x M.w M.ρ (2 * i + r) = .ok (t, hi) := by
  intro i r rel name l0 stm t hr hss hn hl0 hst ht
  obtain ⟨hsp, hsts⟩ := hK.eq hss
  obtain ⟨j, g, d, hi, hj, hsj, hdj⟩ := hB name l0 hl0
  have hji : j = i := tables_inj hnd hj (by rw [← hsp]; exact hn)
  rw [hji] at hsj hdj
  obtain ⟨hlt, hse⟩ := Array.getElem?_eq_some_iff.mp hst
  have hstm : sts[i]! = stm := by rw [← hsts, getElem!_pos rel.sts i hlt, hse]
  have key : ∀ k g d hi l, k / 2 = i → M.src k = .ok (g, d, hi) →
      _root_.Flock.Firewall.Level0.ofDraws sts[i]!.m g d = .ok l →
      _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l = .ok t →
      Loop.realOf src M.x M.w M.ρ k = .ok (t, hi) := by
    intro k g d hi l hk hs hd htl
    rw [hsrc] at hs
    rw [hstm] at hd
    unfold Loop.realOf
    rw [hx]
    dsimp only
    rw [hs]
    simp only [bind, Except.bind]
    rw [hk, hstm, hd]
    simp only [htl, pure, Except.pure]
  rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
  · rw [Nat.add_zero]
    exact ⟨hi, key (2 * i) g d hi l0 (by omega) hsj hdj ht⟩
  · obtain ⟨g', d', hi', l', hs', hd', he⟩ := h1 i g d hi l0 (by rw [← hsts]; exact hlt) hsj hdj
    exact ⟨hi', key (2 * i + 1) g' d' hi' l' (by omega) hs' hd' ((tab_of_toIn hd' hdj he _).trans ht)⟩

/-- **Rep 1's level 0 has rep 0's inputs** where the loop's host and operating system hand it the tape's level-0
inputs (`Loop.Level0Src`), at a statement the masked session proves (`Masked.Live`): at table `j` of the setup, where
rep stream `2j`'s samples give level 0 `l`, rep stream `2j + 1`'s give one with `l`'s inputs, both being
`Masked.linkIn`'s. -/
theorem level0Src_rep1 {src : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      Nat → V (Array F128 × ByteArray × Array ByteArray)} {x : Msg} {w : Claims.Flock.Wit}
    (hS : Loop.Level0Src src x w) (ν : (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).Nonce)
    (t : (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).Tape) (hlive : ZKMal.Masked.Live x) {s0 : _}
    {spec : SessionSpec} {sts : Array Stmt} (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) {j : Nat}
    {g : Array F128} {d : ByteArray} {hi : Array ByteArray} {l : _root_.Flock.Firewall.Level0.Level0}
    (hj : j < sts.size) (hs : src x w (ν, t) (2 * j) = .ok (g, d, hi))
    (hd : _root_.Flock.Firewall.Level0.ofDraws sts[j]!.m g d = .ok l) :
    ∃ g' d' hi' l', src x w (ν, t) (2 * j + 1) = .ok (g', d', hi') ∧
      _root_.Flock.Firewall.Level0.ofDraws sts[j]!.m g' d' = .ok l' ∧ l'.toIn = l.toIn := by
  have hsts : ZKMal.Masked.stsOf x = sts := by
    unfold ZKMal.Masked.stsOf
    rw [hx]
  have hjj : j < ZKMal.Masked.jOf x := by
    unfold ZKMal.Masked.jOf
    rw [ite_eq_left hlive, hsts]
    exact hj
  obtain ⟨l₀, h₀, hi₀⟩ := hS ν t ⟨j, hjj⟩ ⟨0, by decide⟩
  obtain ⟨l₁, h₁, hi₁⟩ := hS ν t ⟨j, hjj⟩ ⟨1, by decide⟩
  have h₀' : Loop.level0At src x w (ν, t) sts (2 * j) = .ok l₀ := by rw [← hsts]; exact h₀
  have h₁' : Loop.level0At src x w (ν, t) sts (2 * j + 1) = .ok l₁ := by rw [← hsts]; exact h₁
  unfold Loop.level0At at h₀' h₁'
  rw [hs, show 2 * j / 2 = j by omega] at h₀'
  rw [show (2 * j + 1) / 2 = j by omega] at h₁'
  simp only [bind, Except.bind] at h₀'
  have e0 : l₀ = l := Except.ok.inj (h₀'.symm.trans hd)
  rcases hs1 : src x w (ν, t) (2 * j + 1) with e | ⟨g', d', hi'⟩
  · rw [hs1] at h₁'
    cases h₁'
  · rw [hs1] at h₁'
    simp only [bind, Except.bind] at h₁'
    exact ⟨g', d', hi', l₁, rfl, h₁', by rw [hi₁, ← hi₀, e0]⟩

/-- **The level-0 half of `RealAt` where the loop samples the tape's level 0**: `real_of_builtAt`, rep 1's level 0
having rep 0's inputs by `level0Src_rep1`, at a statement the masked session proves. -/
theorem realAt_h0 {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {src : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      Nat → V (Array F128 × ByteArray × Array ByteArray)} {sid : Nat} {σ : LoopAt} {s0 : _}
    {spec : SessionSpec} {sts : Array Stmt}
    (hsrc : M.src = src M.x M.w M.ρ) (hx : FlockDeploy.setupOf true M.x = some (s0, spec, sts))
    (hK : SessOn spec sts σ.R sid)
    (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup) (hB : BuiltAt M spec sts σ.L0 sid)
    (hS : Loop.Level0Src src M.x M.w) (hlive : ZKMal.Masked.Live M.x) :
    ∀ (i r : Nat) (rel : Release) (name : String) (l0 : _root_.Flock.Firewall.Level0.Level0) (stm : Stmt)
      (t : _root_.Flock.Firewall.Finish.Tab), r < 2 →
      sessAt σ.R sid = some rel → rel.spec.tables[i]? = some name →
      _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0 → rel.sts[i]? = some stm →
      _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t →
      ∃ hi, Loop.realOf src M.x M.w M.ρ (2 * i + r) = .ok (t, hi) :=
  real_of_builtAt hsrc hx hK hnd hB fun j g d hi l hj hs hd => by
    rw [hsrc] at hs ⊢
    exact level0Src_rep1 hS M.ρ.1 M.ρ.2 hlive hx hj hs hd

end Claims.ZKServer.Relay
