import Proofs.Flock.Claims.KS.PlainDrawRColsE
import Proofs.Flock.Claims.KS.PlainLock
import Proofs.Flock.Claims.Complete.Zk.Run
import Proofs.Flock.Claims.Recursion.VStarCross

/-!
# (P4) modulo what `E*` reads, and `LastCoinFree`, at the deployed reader (`nextR`)

`StepsBase` reads, beside `KSFlockLin`, two facts of the bound record's last step: that it reads no
last coin (`Bound.LastCoinFree`) and that it binds its last message modulo what `E*` reads (`Bound.LastBindsEC`). Here
they are at `bdDeployed` with the deployed reader's steps `nextR rd`, its columns `colsRV rd` for `LastBindsEC`
(`FlockDeploy.KS.bdR`), and any columns for `LastCoinFree`.

* **`LastCoinFree`** (`RecDeploy.lastCoinFree_R`), with no premise. `nextR` answers a coin request with the plain
  verifier's coin, and a coin request at one nonempty coin list is one at every other (`act_coin_transfer`), so at the
  last round the step either answers a coin at both last coins, which accepts nothing, or decides at neither and reads
  `decR`, which reads the coins only at the first and between the first and the last (`decR_congr_coins`). With a single
  coin the statement sets up no session, and the step refuses (`nextR_lastStep`, `RecDeploy.lastStepFree_of_R`).
* **`LastBindsEC`** (`RecDeploy.lastBindsEC_R`), from `Binds`, `LastStepFree` and `RecDeploy.RLastBinds`, the reader's
  binding of its own last message: `Bound.lastBindsEC_of_lastStep`, as `lastBindsEC_deployed` is at the deployed plain
  verifier. The columns `bdR` reads off an accepted last round's opened view are `colsR` there (`colsRV_eq`), so
  `RLastBinds` is stated at `colsR`. It splits into its columns half (`RecDeploy.RColsBind`) and its calls half
  (`RecDeploy.RCallsBind`) (`RecDeploy.rLastBinds_of`), as `PlainLastBindsOrSlack` does at the plain verifier, with no
  decoder slack: `nextR` decides on proofs, not proof files.
-/

namespace RecDeploy

open Claims (Msg Out OC)
open Claims.Recursion (Bound bindOC)
open FlockDeploy (Coin FinalP nextR actR decR)

/-! ## `LastCoinFree` -/

/-- A coin request at some coins is one at any nonempty coins, of their own last coin. -/
theorem act_coin_transfer {x : Msg} {as : List Msg} {cs cs' : List Coin} {c : Coin} (hc' : cs' ≠ [])
    (h : FlockDeploy.act false x as cs = .coin c) : FlockDeploy.act false x as cs' = .coin (cs'.getLast hc') := by
  rcases hs : FlockDeploy.setupOf false x with _ | ⟨s, spec, sts⟩
  · rw [FlockDeploy.KS.act_none hs] at h
    cases h
  · obtain ⟨t, ht⟩ := act_coin_tag h
    exact FlockComplete.act_coin_any hs ht (List.getLast?_eq_some_getLast hc')

/-- **Off a coin request, `nextR`'s step reads no last coin**, past a first coin: `act` asks for no coin at either last
coin (`act_coin_transfer`), and `decR` reads the coins only at the first and between the first and the last. -/
theorem actR_congr_last (rd : Msg → Option FinalP) (x : Msg) (ms : List Msg) (c₀ : Coin) (l : List Coin)
    (c c' : Coin) (h : ∀ c₂, FlockDeploy.act false x ms (c₀ :: l ++ [c]) ≠ .coin c₂) :
    actR rd x ms (c₀ :: l ++ [c']) = actR rd x ms (c₀ :: l ++ [c]) := by
  have h' : ∀ c₂, FlockDeploy.act false x ms (c₀ :: l ++ [c']) ≠ .coin c₂ := fun c₂ hc₂ =>
    h _ (act_coin_transfer (List.cons_ne_nil _ _) hc₂)
  have hd : decR rd x ms (c₀ :: l ++ [c']) = decR rd x ms (c₀ :: l ++ [c]) :=
    FlockDeploy.decR_congr_coins rd x ms (by simp) (by simp)
  unfold actR
  rw [hd]
  cases ha : FlockDeploy.act false x ms (c₀ :: l ++ [c']) with
  | coin c₂ => exact absurd ha (h' c₂)
  | rej =>
    cases ha' : FlockDeploy.act false x ms (c₀ :: l ++ [c]) with
    | coin c₃ => exact absurd ha' (h c₃)
    | rej => rfl
    | decide => rfl
  | decide =>
    cases ha' : FlockDeploy.act false x ms (c₀ :: l ++ [c]) with
    | coin c₃ => exact absurd ha' (h c₃)
    | rej => rfl
    | decide => rfl

/-- **`nextR`'s last step, as `agrees` checks it, reads no last coin**: at a coin request it answers a coin at either
last coin, which accepts nothing; off one it steps alike (`actR_congr_last`); and with a single coin the statement sets
up no session, so the step refuses. -/
theorem nextR_lastStep (rd : Msg → Option FinalP) (H : Msg → Msg) (x : Msg) (ms : List Msg) (L : List Coin)
    (c c' : Coin) (hk : ms.length = (FlockDeploy.V false).k x) (hc : L.length + 1 = ms.length) :
    (bindOC (nextR rd x ms (L ++ [c])) fun b => OC.ret (decide (b = Out.acc))).eval H =
      (bindOC (nextR rd x ms (L ++ [c'])) fun b => OC.ret (decide (b = Out.acc))).eval H := by
  rcases L with _ | ⟨c₀, l⟩
  · have hk1 : (FlockDeploy.V false).k x = 1 := by
      simp only [List.length_nil, Nat.zero_add] at hc
      omega
    have hsu : FlockDeploy.setupOf false x = none := by
      rcases h : FlockDeploy.setupOf false x with _ | ⟨s, spec, sts⟩
      · rfl
      · simp only [FlockDeploy.V, h] at hk1
        omega
    have hr : ∀ c₀ : Coin, nextR rd x ms ([] ++ [c₀]) = .ret .rej := fun c₀ => by
      have ha : FlockDeploy.act false x ms ([] ++ [c₀]) = .rej := by
        unfold FlockDeploy.act
        rw [hsu]
      have hd : decR rd x ms ([] ++ [c₀]) = none := by
        unfold decR
        rw [hsu]
        rfl
      simp only [nextR, actR, ha, hd]
    rw [hr c, hr c']
  · by_cases hz : ∃ c₂, FlockDeploy.act false x ms (c₀ :: l ++ [c]) = .coin c₂
    · obtain ⟨c₂, hc₂⟩ := hz
      have hc₃ := act_coin_transfer (c := c₂) (cs' := c₀ :: l ++ [c']) (List.cons_ne_nil _ _) hc₂
      have hm : ∀ {cs : List Coin} {c₄ : Coin}, FlockDeploy.act false x ms cs = .coin c₄ →
          (bindOC (nextR rd x ms cs) fun b => OC.ret (decide (b = Out.acc))).eval H = (false, []) := by
        intro cs c₄ h
        rw [Claims.Recursion.LastGap.eval_bind]
        simp only [nextR, actR, h, Claims.OC.eval]
        rfl
      rw [hm hc₂, hm hc₃]
    · push Not at hz
      have he : nextR rd x ms (c₀ :: l ++ [c']) = nextR rd x ms (c₀ :: l ++ [c]) := by
        simp only [nextR, actR_congr_last rd x ms c₀ l c c' hz]
      rw [he]

/-- **`LastStepFree` at every bound record whose `DE` steps as the deployed reader.** -/
theorem lastStepFree_of_R {Col T : Type} (H : Msg → Msg) (rd : Msg → Option FinalP) (Bd : Bound Coin Col T)
    (hV : Bd.DE.V = FlockDeploy.VR rd) : Bd.LastStepFree H := by
  intro x ms cs cs' hk hc hc' hd
  rw [hV] at hk
  have hne : ∀ l : List Coin, l.length = ms.length → l ≠ [] := fun l hl h => by
    have h0 := V_k_pos_flock false x
    subst h
    simp only [List.length_nil] at hl
    change ms.length = (FlockDeploy.V false).k x at hk
    omega
  obtain ⟨L, c, rfl⟩ : ∃ L c, cs = L ++ [c] :=
    ⟨cs.dropLast, cs.getLast (hne cs hc), (List.dropLast_concat_getLast (hne cs hc)).symm⟩
  obtain ⟨L', c', rfl⟩ : ∃ L c, cs' = L ++ [c] :=
    ⟨cs'.dropLast, cs'.getLast (hne cs' hc'), (List.dropLast_concat_getLast (hne cs' hc')).symm⟩
  rw [List.dropLast_concat, List.dropLast_concat] at hd
  subst hd
  rw [hV]
  exact nextR_lastStep rd H x ms L c c' hk
    (by simp only [List.length_append, List.length_singleton] at hc; exact hc)

/-- **`LastCoinFree` at the deployed bound record at the deployed reader's steps**, whatever its columns and the code
side's other choices. -/
theorem lastCoinFree_R (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Msg → Option FinalP) (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ)
    (join : List Msg → Msg) (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (H : Msg → Msg) :
    (bdDeployed rel (nextR rd) colsE wE join rowIn final w).LastCoinFree H :=
  Bound.lastCoinFree_of_step (lastStepFree_of_R H rd _ rfl)

/-! ## `LastBindsEC` -/

/-- **The deployed reader binds its last message**: `Bound.LastStepBinds` at `nextR rd`, with the columns its decision
opens (`colsR`). Two last messages it accepts after the same rounds and coins, where it answered each earlier round with
that round's coin, that differ in the opened columns or in the set of the decision's calls, show a SHA-512 collision
among those calls. -/
def RLastBinds (H : Msg → Msg) (rd : Msg → Option FinalP) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (l l' : Msg) (cs : List Coin), ms.length + 1 = (FlockDeploy.V false).k x →
    cs.length = ms.length + 1 →
    (∀ j < ms.length, ((nextR rd x (ms.take (j + 1)) (cs.take (j + 1))).eval H).1 =
      .msg (cs.getD j default).bytes) →
    ((nextR rd x (ms ++ [l]) cs).eval H).1 = .acc → ((nextR rd x (ms ++ [l']) cs).eval H).1 = .acc →
    (FlockDeploy.colsR rd x (ms ++ [l]) cs ≠ FlockDeploy.colsR rd x (ms ++ [l']) cs ∨
      ((nextR rd x (ms ++ [l]) cs).eval H).2.toFinset ≠ ((nextR rd x (ms ++ [l']) cs).eval H).2.toFinset) →
    ∃ q ∈ (((nextR rd x (ms ++ [l]) cs).eval H).2 ++ ((nextR rd x (ms ++ [l']) cs).eval H).2).map Prod.fst,
      ∃ q' ∈ (((nextR rd x (ms ++ [l]) cs).eval H).2 ++ ((nextR rd x (ms ++ [l']) cs).eval H).2).map Prod.fst,
        Claims.Collision H q q'

/-- **The deployed reader binds its last message's columns**: two last messages it accepts after the same rounds and
coins, whose decisions open different columns (`colsR`), show a SHA-512 collision among the two decisions' calls. -/
def RColsBind (H : Msg → Msg) (rd : Msg → Option FinalP) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (l l' : Msg) (cs : List Coin), ms.length + 1 = (FlockDeploy.V false).k x →
    cs.length = ms.length + 1 →
    ((nextR rd x (ms ++ [l]) cs).eval H).1 = .acc → ((nextR rd x (ms ++ [l']) cs).eval H).1 = .acc →
    FlockDeploy.colsR rd x (ms ++ [l]) cs ≠ FlockDeploy.colsR rd x (ms ++ [l']) cs →
    ∃ q ∈ (((nextR rd x (ms ++ [l]) cs).eval H).2 ++ ((nextR rd x (ms ++ [l']) cs).eval H).2).map Prod.fst,
      ∃ q' ∈ (((nextR rd x (ms ++ [l]) cs).eval H).2 ++ ((nextR rd x (ms ++ [l']) cs).eval H).2).map Prod.fst,
        Claims.Collision H q q'

/-- **The deployed reader binds its last message's calls**: two last messages it accepts after the same rounds and
coins, where it answered each earlier round with that round's coin, whose decisions call SHA-512 on different sets,
show a collision among those calls. The calls are compared as sets, as at the plain verifier. -/
def RCallsBind (H : Msg → Msg) (rd : Msg → Option FinalP) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (l l' : Msg) (cs : List Coin), ms.length + 1 = (FlockDeploy.V false).k x →
    cs.length = ms.length + 1 →
    (∀ j < ms.length, ((nextR rd x (ms.take (j + 1)) (cs.take (j + 1))).eval H).1 =
      .msg (cs.getD j default).bytes) →
    ((nextR rd x (ms ++ [l]) cs).eval H).1 = .acc → ((nextR rd x (ms ++ [l']) cs).eval H).1 = .acc →
    ((nextR rd x (ms ++ [l]) cs).eval H).2.toFinset ≠ ((nextR rd x (ms ++ [l']) cs).eval H).2.toFinset →
    ∃ q ∈ (((nextR rd x (ms ++ [l]) cs).eval H).2 ++ ((nextR rd x (ms ++ [l']) cs).eval H).2).map Prod.fst,
      ∃ q' ∈ (((nextR rd x (ms ++ [l]) cs).eval H).2 ++ ((nextR rd x (ms ++ [l']) cs).eval H).2).map Prod.fst,
        Claims.Collision H q q'

/-- `RLastBinds` is its columns half and its calls half. -/
theorem rLastBinds_of {H : Msg → Msg} {rd : Msg → Option FinalP} (hC : RColsBind H rd) (hK : RCallsBind H rd) :
    RLastBinds H rd := by
  intro x ms l l' cs hk hc hearly hacc hacc' hne
  rcases hne with hne | hne
  · exact hC x ms l l' cs hk hc hacc hacc' hne
  · exact hK x ms l l' cs hk hc hearly hacc hacc' hne

/-- **The columns `bdR` reads off an accepted last round are the decision's**: on the opened view of messages
`ms ++ [l]` that `nextR rd` accepts at the coins `cs` at SHA-512, each earlier round answered by its coin's bytes,
`colsRV rd` is `colsR rd` (`colsRV_eq`). -/
theorem colsRV_plainView {rd : Msg → Option FinalP} {x : Msg} {ms : List Msg} {cs : List Coin}
    (hk : cs.length = (FlockDeploy.V false).k x) (hc : cs.length = ms.length)
    (hacc : ((nextR rd x ms cs).eval FlockDeploy.sha512).1 = .acc) :
    FlockDeploy.colsRV rd (plainView ms cs) = FlockDeploy.colsR rd x ms cs := by
  have hne : cs ≠ [] := fun h => by
    have h0 := V_k_pos_flock false x
    subst h
    simp only [List.length_nil] at hk
    omega
  have hlen : (cs.dropLast.map (fun c => Out.msg c.bytes) ++ [Out.acc]).length = ms.length := by
    simp only [List.length_append, List.length_map, List.length_dropLast, List.length_singleton]
    have := List.length_pos_iff.mpr hne
    omega
  refine FlockDeploy.KS.colsRV_eq ?_ ?_ hk hacc
  · show (ms.zip _).map Prod.fst = ms
    exact List.map_fst_zip (by omega)
  · show ((ms.zip _).map Prod.snd).map outCoin = cs.dropLast ++ [default]
    rw [List.map_snd_zip (by omega), List.map_append, List.map_map]
    congr 1
    conv_rhs => rw [← List.map_id cs.dropLast]
    refine List.map_congr_left fun c _ => ?_
    rw [Function.comp_apply, id_eq, outCoin]
    exact coinOf_bytes c

/-- **`LastStepBinds` at the deployed bound record at the deployed reader's steps and columns**, from `RLastBinds`,
at SHA-512, whatever the code side's other choices. -/
theorem lastStepBinds_R (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Msg → Option FinalP) (wE : Msg → ℝ) (join : List Msg → Msg) (rowIn : Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (h : RLastBinds FlockDeploy.sha512 rd) :
    (FlockDeploy.KS.bdR rel rd wE join rowIn final w).LastStepBinds FlockDeploy.sha512 := by
  intro x ms l l' cs hk hc hearly hacc hacc' hne
  change ms.length + 1 = (FlockDeploy.V false).k x at hk
  change ((nextR rd x (ms ++ [l]) cs).eval FlockDeploy.sha512).1 = .acc at hacc
  change ((nextR rd x (ms ++ [l']) cs).eval FlockDeploy.sha512).1 = .acc at hacc'
  have hans := bdDeployed_ans rel (nextR rd) (FlockDeploy.colsRV rd) wE join rowIn final w
  simp only [hans] at hearly
  have hview : ∀ ms' : List Msg,
      (FlockDeploy.KS.bdR rel rd wE join rowIn final w).lastView ms' cs = plainView ms' cs := fun ms' => by
    unfold Bound.lastView plainView
    simp only [hans]
  rw [hview, hview] at hne
  change FlockDeploy.colsRV rd (plainView (ms ++ [l]) cs) ≠ FlockDeploy.colsRV rd (plainView (ms ++ [l']) cs) ∨
    ((nextR rd x (ms ++ [l]) cs).eval FlockDeploy.sha512).2.toFinset ≠
      ((nextR rd x (ms ++ [l']) cs).eval FlockDeploy.sha512).2.toFinset at hne
  have hlen : ∀ a : Msg, cs.length = (ms ++ [a]).length := fun a => by
    simp only [List.length_append, List.length_singleton]
    exact hc
  rw [colsRV_plainView (hc.trans hk) (hlen l) hacc, colsRV_plainView (hc.trans hk) (hlen l') hacc'] at hne
  exact h x ms l l' cs hk hc hearly hacc hacc' hne

/-- **(P4) modulo what `E*` reads, at the deployed reader** (`Bound.LastBindsEC` at `bdR` with the deployed rows'
digest input), at SHA-512, from the reader's binding of its last message (`RLastBinds`), whatever the code side's join,
`final` and weights. -/
theorem lastBindsEC_R (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Msg → Option FinalP) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (h : RLastBinds FlockDeploy.sha512 rd) :
    (FlockDeploy.KS.bdR rel rd wE join rowInDeployed final w).LastBindsEC FlockDeploy.sha512 :=
  Bound.lastBindsEC_of_lastStep (binds_deployed rel _ _ wE join final w) (lastStepFree_of_R _ rd _ rfl)
    (lastStepBinds_R rel rd wE join rowInDeployed final w h)

/-- **What `StepsBase` reads of the last step at the deployed reader**: `LastBindsEC` and `LastCoinFree` at `bdR`, at
SHA-512, from the reader's columns half and calls half. -/
theorem lastR_of (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Msg → Option FinalP) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hC : RColsBind FlockDeploy.sha512 rd) (hK : RCallsBind FlockDeploy.sha512 rd) :
    (FlockDeploy.KS.bdR rel rd wE join rowInDeployed final w).LastBindsEC FlockDeploy.sha512 ∧
      (FlockDeploy.KS.bdR rel rd wE join rowInDeployed final w).LastCoinFree FlockDeploy.sha512 :=
  ⟨lastBindsEC_R rel rd wE join final w (rLastBinds_of hC hK),
    lastCoinFree_R rel rd _ wE join rowInDeployed final w FlockDeploy.sha512⟩

end RecDeploy
