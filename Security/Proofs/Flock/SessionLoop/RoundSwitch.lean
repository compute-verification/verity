module

public import Proofs.Flock.SessionLoop.SchedAt
public import Proofs.Flock.SessionLoop.SchedUnrefused
public import Proofs.Flock.SessionLoop.RoundSwitchCut
public import Proofs.Flock.SessionLoop.RoundSwitchSv
public import Proofs.Flock.SessionLoop.RoundSwitchRel

/-!
# A rep's ring-switch rounds

`LiveSched`'s equation at rep stream `2j + r`'s rounds from its rank check's round (`Masked.rankSlot`) to the round
before the opening's binding (`Masked.repSlot` `0`): the opening's first round (the PCS label, where the frame runs the
rank check, and ring-switch claim `0`), claims `1` to `N − 1`, and `γ`'s squeeze, `N = 2 + points · regions` the rep's
claims. Past the algebraic phase passing is byte equality, so the release is the frame's scheduled round, the ring
switch's body at the switch vectors of the frame's table over the claims' points; the switched session sends the same
body at its table's vectors (`Masked.swSrcE_sv`).

* **`LeanRelease.release_sched_switch`**: the release of the worker's round there is the switched session's message
  without its leading domain frame.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open _root_.Flock.Firewall.Finish (Tab)

namespace LeanRelease

/-- **The ring switch's rounds**: at rep stream `2j + r`'s round `k` from its rank check's round, before the opening's
binding (`Masked.repSlot` `0`), the release of the worker's round is the switched session's message without its leading
domain frame. The frame releases the switch's body `k` at the loop's table's switched vectors (`switch_frame`); no
refusal shows by the round (`unrefused_of_rank`, the rep's rank check holding at the round itself by
`rank_unrefused_pcs` and past it by `rank_unrefused_sched`), so the session sends the body at the switched source's
vectors (`Loop.swMsgAt_switch`), which are the loop's (`Loop.switch_sv`), and the body has no domain frame
(`Loop.dropDom_rs`). -/
theorem release_sched_switch (vs : Loop.finS.SwVals) (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (hreal : real = Loop.realOf src)
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) (hS : Loop.Src L₀ src x w)
    (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (hTV : Loop.TakesValsAt vs (ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops) {v : Msg}
    (hpos : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk (Loop.finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true)
    (href : (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) (j : Fin (Masked.jOf x)) (r : Fin 2) {k : ℕ}
    (hn : ops.length = Masked.rankSlot x j.val r.val + k)
    (hk : ops.length < Masked.repSlot x (2 * j.val + r.val) fun _ => 0) :
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom (Loop.finS.swMsgAt x (Loop.finS.wt x w) (vs x w ν) t (ops.map fun p => Loop.finS.decOf p.2)) := by
  have hx : Masked.Live x := Masked.live_of hSc hR
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨tr, hir, zc, lc, hre, hst, hpts, hkN, hrel⟩ :=
    switch_frame (withFiles (withLink L₀ src) src) real wk hx hSv ν t κ m₀ ρ₀ ops
      (by rw [hreal]; exact Loop.realTabs_of src x w (ν, t) hsx) hlt hl hv j r hn hk
  have hrel' : (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v = _ := hrel
  rw [hrel']
  -- no refusal shows by the round
  have hjh : (2 * j.val + r.val) / 2 = j.val := by omega
  have hrm : (2 * j.val + r.val) % 2 = r.val := by omega
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [Masked.stsOf_half x j r]
    exact (Masked.zkTab x j).hz
  have hRk : Masked.rankSlot x j.val r.val = Masked.repBase x (2 * j.val + r.val) +
      ((Masked.stsOf x)[j.val]!.m + (Masked.stsOf x)[j.val]!.c.kLog - 6) := rfl
  have hβ : Masked.repSlot x (2 * j.val + r.val) (fun _ => 0) ≤ Masked.betaSlot x j.val r.val := by
    unfold Masked.betaSlot
    rw [Masked.repSlot_eq hsx hz, Masked.repSlot_eq hsx hz]
    try dsimp only
    omega
  have hN := unrefused_of_rank Masked.dataE (withFiles (withLink L₀ src) src) real wk x w hx hSv (ν, t) κ m₀ ρ₀ ops
    hlt.le hl href (q := 2 * j.val + r.val) (by omega) (by omega) (by rw [hjh, hrm]; omega) fun hN' => by
      rw [hjh, hrm] at hN' ⊢
      by_cases h0 : ops.length = Masked.rankSlot x j.val r.val
      · exact rank_unrefused_pcs Masked.dataE (withFiles (withLink L₀ src) src) real x hx hSv (ν, t) m₀ ρ₀ ops hlt.le
          href hj hr2 h0
      · exact rank_unrefused_sched Masked.dataE (withFiles (withLink L₀ src) src) real wk x w hx hSv (ν, t) κ m₀ ρ₀
          ops hlt.le hl href hj hr2 (by omega)
  -- the session's round: the switch's body at the switched source's vectors, which are the loop's
  generalize hcs : (ops.map fun p => Loop.finS.decOf p.2) = cs
  rw [hcs] at hN
  obtain ⟨extras, hext⟩ := Masked.extra_ok hx hsx (Loop.finS.pad (K := refK x) cs) hj
  obtain ⟨hsz, hsv⟩ := Loop.switch_sv vs hR hx hsx
    (show Loop.Level0Src src x w ∧ Loop.RepSrc L₀ src x w from hS).1 ν t ops hpos hlt.le hcs j r
    (by rw [hreal] at hre; exact hre) hst hpts (k := k) (by omega) hN hext
  rw [hsz] at hkN
  rw [Loop.swMsgAt_switch vs w hx hsx ν t cs (by rw [← hcs, List.length_map]; exact hlt) j r
      (by rw [← hcs, List.length_map]; exact hn) (by rw [← hn]; exact hN) hext hkN,
    Loop.dropDom_rs _ hkN, hsz]
  congr 1
  exact Loop.rsBody_congr [] hsv

end LeanRelease

end Claims.ZKServer
