import Proofs.Flock.Claims.Recursion.BindsDeployed
import Proofs.Flock.Claims.Recursion.FlockKSDeploy
import Proofs.Flock.Claims.Recursion.DeployMargin

/-!
# The route modulo what `E*` reads: what it needs of the last round

The route's last-round comparison modulo `E*`'s reading makes the plurality prover `P_I` send, at the last inner round,
some message with the plurality reading, not the extraction's own message. `P_I`'s run there makes `IVStar`'s calls on
its own message, and `PrBAbort_le_PrBI` needs every call of a kept run among the calls `E_O`'s witnesses record at that
draw (`innerF_vLog_sub`). So the route reads (P4) as `Bound.LastBindsEC`, where the columns or the sets of calls differ, and
`Bound.LastCoinFree`, that `IVStar`'s last step reads no last coin (both in `Definitions/Claims/Recursion.lean`). Here
`LastCoinFree` follows from `DE`'s last step reading none (`Bound.lastCoinFree_of_step`), which the deployed plain
verifier meets: its decision passes the coins with the last dropped (`RecDeploy.lastStepFree_of_V`,
`RecDeploy.lastCoinFree_deployed`).
-/

namespace Claims.Recursion

open Claims

variable {Coin Col T : Type} [Inhabited Coin]

theorem eval_bindOC_congr (H : Msg → Msg) {α β : Type} {p p' : OC α} {f f' : α → OC β}
    (hp : p.eval H = p'.eval H) (hf : ∀ a, (f a).eval H = (f' a).eval H) :
    (bindOC p f).eval H = (bindOC p' f').eval H := by
  rw [LastGap.eval_bind, LastGap.eval_bind, hp, hf]

theorem eval_allOC_map_congr (H : Msg → Msg) {ι : Type} (f g : ι → OC Bool) :
    ∀ l : List ι, (∀ j ∈ l, (f j).eval H = (g j).eval H) →
      (allOC (l.map f)).eval H = (allOC (l.map g)).eval H
  | [], _ => rfl
  | j :: l, h => by
    simp only [List.map_cons, allOC]
    refine eval_bindOC_congr H (h j List.mem_cons_self) fun b => ?_
    cases b
    · rfl
    · exact eval_allOC_map_congr H f g l fun i hi => h i (List.mem_cons_of_mem _ hi)

theorem take_eq_of_dropLast {α : Type} {l l' : List α} (hl : l.length = l'.length) (hd : l.dropLast = l'.dropLast)
    {n : ℕ} (hn : n < l.length) : l.take n = l'.take n := by
  have h1 : ∀ m : List α, n < m.length → m.take n = m.dropLast.take n := fun m hm => by
    rw [List.dropLast_eq_take, List.take_take]
    congr 1
    omega
  rw [h1 l hn, h1 l' (by omega), hd]

/-- **`DE`'s last step, as `agrees` checks it, reads no last coin.** -/
def Bound.LastStepFree (H : Msg → Msg) (Bd : Bound Coin Col T) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (cs cs' : List Coin), ms.length = Bd.DE.V.k x → cs.length = ms.length →
    cs'.length = ms.length → cs.dropLast = cs'.dropLast →
    (bindOC (Bd.DE.V.next x ms cs) fun b => OC.ret (decide (b = .acc))).eval H =
      (bindOC (Bd.DE.V.next x ms cs') fun b => OC.ret (decide (b = .acc))).eval H

/-- `agrees` at the last round reads no last coin where `DE`'s last step reads none: each earlier step reads the coins
before the last only. -/
theorem Bound.agrees_congr_last {H : Msg → Msg} {Bd : Bound Coin Col T} (h : Bd.LastStepFree H) {x : Msg}
    {ms : List Msg} {cs cs' : List Coin} (hk : ms.length = Bd.DE.V.k x) (hc : cs.length = ms.length)
    (hc' : cs'.length = ms.length) (hd : cs.dropLast = cs'.dropLast) :
    (Bd.agrees x ms cs).eval H = (Bd.agrees x ms cs').eval H := by
  unfold Bound.agrees
  refine eval_allOC_map_congr H _ _ _ fun j hj => ?_
  rw [List.mem_range] at hj
  by_cases hj1 : j + 1 < ms.length
  · have ht : cs.take (j + 1) = cs'.take (j + 1) := take_eq_of_dropLast (hc.trans hc'.symm) hd (by omega)
    have hg : cs.getD j default = cs'.getD j default := by
      rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
        ← List.getElem?_take_of_lt (show j < j + 1 by omega), ht,
        List.getElem?_take_of_lt (show j < j + 1 by omega)]
    rw [ht, hg]
  · rw [ite_eq_right hj1, ite_eq_right hj1, List.take_of_length_le (show ms.length ≤ j + 1 by omega),
      List.take_of_length_le (show cs.length ≤ j + 1 by omega),
      List.take_of_length_le (show cs'.length ≤ j + 1 by omega)]
    exact h x ms cs cs' hk hc hc' hd

/-- **`IVStar` reads no last coin where `DE`'s last step reads none**: its last step reads the coins only in
`agrees`. -/
theorem Bound.lastCoinFree_of_step {H : Msg → Msg} {Bd : Bound Coin Col T} (h : Bd.LastStepFree H) :
    Bd.LastCoinFree H := by
  intro x as cs cs' hk hc hc' hd
  have hlt : ¬ as.length < Bd.DE.V.k x := by omega
  simp only [Bound.IVStar, hlt, ↓reduceIte]
  rcases Bd.final (as.getLastD []) with _ | ⟨os, last⟩
  · rfl
  · dsimp only
    by_cases hcnd : os.length + 1 = as.length ∧ ∀ j < os.length, (os.getD j []).length = Bd.nAt x j
    · rw [ite_eq_left hcnd, ite_eq_left hcnd]
      refine eval_bindOC_congr H rfl fun ok => ?_
      cases ok
      · rfl
      · have hlen : (Bd.opened os last).length = as.length := by
          simp only [Bound.opened, List.length_append, List.length_map, List.length_singleton]
          exact hcnd.1
        exact eval_bindOC_congr H (Bd.agrees_congr_last h (hlen.trans hk) (hc.trans hlen.symm)
          (hc'.trans hlen.symm) hd) fun _ => rfl
    · rw [ite_eq_right hcnd, ite_eq_right hcnd]

end Claims.Recursion

namespace RecDeploy

open Claims (Msg Out OC)
open Claims.Recursion (Bound bindOC)
open FlockDeploy (Coin)

/-- **The deployed plain verifier's last step, as `agrees` checks it, reads no last coin**: a coin request answers its
own coin and accepts nothing; past a first coin a decision passes the coins with the last dropped
(`FlockDeploy.act_cons_snoc`); and with a single coin the statement sets up no session, so the step refuses. -/
theorem next_lastStep (H : Msg → Msg) (x : Msg) (ms : List Msg) (L : List Coin) (c c' : Coin)
    (hk : ms.length = (FlockDeploy.V false).k x) (hc : L.length + 1 = ms.length) :
    (bindOC (FlockDeploy.next false x ms (L ++ [c])) fun b => OC.ret (decide (b = Out.acc))).eval H =
      (bindOC (FlockDeploy.next false x ms (L ++ [c'])) fun b => OC.ret (decide (b = Out.acc))).eval H := by
  by_cases hz : ∃ t, ms.getLast? = some (0 :: t)
  · obtain ⟨t, ht⟩ := hz
    have hfalse : ∀ c₀ : Coin,
        (bindOC (FlockDeploy.next false x ms (L ++ [c₀])) fun b => OC.ret (decide (b = Out.acc))).eval H =
          (false, []) := by
      intro c₀
      rw [Claims.Recursion.LastGap.eval_bind]
      rcases FlockDeploy.act_snoc_zero (zk := false) (x := x) (cs := L) (c := c₀) ht with h | h
      · rw [FlockDeploy.next_rej H h]
        rfl
      · rw [FlockDeploy.next_coin H h]
        rfl
    rw [hfalse, hfalse]
  · have ha : ∀ t, ms.getLast? ≠ some (0 :: t) := fun t h => hz ⟨t, h⟩
    rcases L with _ | ⟨c₀, l⟩
    · have hk1 : (FlockDeploy.V false).k x = 1 := by
        simp only [List.length_nil, Nat.zero_add] at hc
        omega
      have hsu : FlockDeploy.setupOf false x = none := by
        rcases h : FlockDeploy.setupOf false x with _ | ⟨s, spec, sts⟩
        · rfl
        · simp only [FlockDeploy.V, h] at hk1
          omega
      have hr : ∀ c₀ : Coin, FlockDeploy.act false x ms ([] ++ [c₀]) = .rej := fun c₀ => by
        unfold FlockDeploy.act
        rw [hsu]
      rw [Claims.Recursion.LastGap.eval_bind, Claims.Recursion.LastGap.eval_bind, FlockDeploy.next_rej H (hr c),
        FlockDeploy.next_rej H (hr c')]
    · have heq : FlockDeploy.next false x ms ((c₀ :: l) ++ [c']) = FlockDeploy.next false x ms ((c₀ :: l) ++ [c]) := by
        simp only [FlockDeploy.next, FlockDeploy.act_cons_snoc false x ms c₀ l c c' ha]
      rw [heq]

/-- **`LastStepFree` at every bound record whose `DE` steps as the deployed plain verifier.** -/
theorem lastStepFree_of_V {Col T : Type} (H : Msg → Msg) (Bd : Bound Coin Col T) (hV : Bd.DE.V = FlockDeploy.V false) :
    Bd.LastStepFree H := by
  intro x ms cs cs' hk hc hc' hd
  rw [hV] at hk
  have hne : ∀ l : List Coin, l.length = ms.length → l ≠ [] := fun l hl h => by
    have h0 := V_k_pos_flock false x
    subst h
    simp only [List.length_nil] at hl
    omega
  obtain ⟨L, c, rfl⟩ : ∃ L c, cs = L ++ [c] :=
    ⟨cs.dropLast, cs.getLast (hne cs hc), (List.dropLast_concat_getLast (hne cs hc)).symm⟩
  obtain ⟨L', c', rfl⟩ : ∃ L c, cs' = L ++ [c] :=
    ⟨cs'.dropLast, cs'.getLast (hne cs' hc'), (List.dropLast_concat_getLast (hne cs' hc')).symm⟩
  rw [List.dropLast_concat, List.dropLast_concat] at hd
  subst hd
  rw [hV]
  exact next_lastStep H x ms L c c' hk (by simp only [List.length_append, List.length_singleton] at hc; exact hc)

/-- **`LastCoinFree` at the deployed bound record**, at the deployed plain verifier's own steps, whatever the code
side's other choices. -/
theorem lastCoinFree_deployed (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (H : Msg → Msg) :
    (bdDeployed rel (FlockDeploy.V false).next colsE wE join rowIn final w).LastCoinFree H :=
  Bound.lastCoinFree_of_step (lastStepFree_of_V H _ rfl)

end RecDeploy
