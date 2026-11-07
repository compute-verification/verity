module

public import Proofs.Flock.ZkAny.UnswitchedZk

/-!
# The deferred release per outcome at its `Finish` verdict, at any values

The deferred release's verdict at its `Finish` (`goesOnAt` at its last round) is public. Where it stops, the release is
the unswitched prover's run cut there (`LeanRelease.refDBelow_unP`, `ZkServer.release_stopped`), and that prover is zero
knowledge at any values (`RefSession.withStops_unP_zk`). Where it goes on, it is the run of the go prover
`RefSession.goP`: the unswitched prover before the `Finish`, and at the `Finish` the switched prover's last message where
the workers' rows hold at the sampled pads, else `abort`. The go prover is simulated where its run goes on (`gkE_go_pow`)
through protocols whose messages are the switched prover's at one coin vector, so nothing asks the switched round
encoding to read the coins before its round only.

* `RefSession.goMsgAt`, `RefSession.goP`, `goP_next_lt`, `goP_next_rows`: the go prover and its messages.
* `RefSession.goP_prefixHides`: (HA) at `goP`, which before the last coin is the unswitched prover
  (`TPreModel.runs_close_un`).
* `RefSession.fixP`, `RefSession.fixOffP`: the switched prover's messages at one coin vector, after the last coin `fin`
  where the tables admit the coins and `off` otherwise, or `off` at every coin vector. `fixP_shvzk` holds the first to
  the honest prover's simulator (`TCoinModel.shvzk_sw`), `fixOffModel` the second to the refusing dummy
  (`TOffModel.close_sw`).
* **`RefSession.goP_shvzkLe`**: (HB) one-sided at `goP`, within `6·|L|·δ₁`. Where the rows hold, `goP` sends `fixP`'s
  messages and the unswitched prover's ending in `abort` are `fixOffP`'s; elsewhere `goP` sends the unswitched prover's
  ending in `abort`, and `fixOffP`'s ending in `abort` are `fixP`'s. On a test that doesn't grow when the last message
  becomes `abort`, the two bounds at the unswitched prover and `fixOffP` against the refusing dummy cancel.
* **`RefSession.goP_go`**: `gkE_go_pow` at `goP`: where its run goes on at its last round, the honest prover's
  Goldreich–Kahan simulator simulates it within `8·|L|·δ₁ + 24t·|L|·δ₁ + 2·3^{-t}` plus `4t·3^t` times its collision
  finder's success. `withStops_goP_go` is it at the session with the frame's stops.
* `LeanRelease.FinishRows`: the named premise that the ZK server's own `Finish` takes a session only where the workers'
  rows hold at the sampled pads.
* **`LeanRelease.release_go`**: where the deferred release goes on at its `Finish`, it is `goP`'s run at the session
  with the frame's stops, from the transport premises at the unswitched prover, `FinalRoundEq` and `FinishRows`.
* **`LeanRelease.zk_finish_out`**: the deferred release at the model server meets `Claims.ZKOut` at its `Finish`
  verdict, for every worker and any values whose unswitched round encoding is causal.
-/

@[expose] public section

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- Two-sided: if every event's probabilities differ by at most `δ`, so do the expectations of every test with values in
`[0, 1]`. -/
theorem abs_Ex_sub_le_of_events {Ω₁ Ω₂ V : Type} [Fintype Ω₁] [Fintype Ω₂] [Nonempty Ω₁] [Nonempty Ω₂]
    (f : Ω₁ → V) (g : Ω₂ → V) {δ : ℝ} (h : ∀ E : V → Prop, |Pr (fun ω => E (f ω)) - Pr (fun ω => E (g ω))| ≤ δ)
    (Ψ : V → ℝ) (h0 : ∀ v, 0 ≤ Ψ v) (h1 : ∀ v, Ψ v ≤ 1) :
    |Ex (fun ω => Ψ (f ω)) - Ex (fun ω => Ψ (g ω))| ≤ δ :=
  FlockSoundness.ZK.GK.abs_avgR_sub_le_of_events f g
    (fun U => by rw [avgR_eq_Ex, avgR_eq_Ex, Ex_ind_comp, Ex_ind_comp]; exact h U) Ψ h0 h1

/-- Setting the last of `K + 1` messages. -/
theorem map_range_set {α : Type} (K : ℕ) (f : ℕ → α) (a : α) :
    ((List.range (K + 1)).map f).set K a = (List.range (K + 1)).map fun j => if j = K then a else f j := by
  refine List.ext_getElem (by simp) fun j _ _ => ?_
  by_cases h : j = K
  · subst h
    simp
  · simp [h, Ne.symm h]

namespace RefSession

theorem pad_ofFn_self (S : RefSession) {K : ℕ} (c : Fin K → S.Coin) : S.pad (List.ofFn c) = c := by
  have h := S.pad_ofFn c
  rwa [List.take_of_length_le (List.length_ofFn).le] at h

/-! ## The go prover -/

open Classical in
/-- **The go prover's message after the coins `cs`** at the values `v`: before the last coin the unswitched prover's
(`unMsgAt`); after it the switched prover's last message where the rows hold at the tape at the padded coins, else
`abort`. -/
noncomputable def goMsgAt (S : RefSession) (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (v : (c : Fin (refK x) → S.Coin) → (∀ j, ZkView.Seen (S.C x j).z) → ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals)
    (t : S.Tape) (cs : List S.Coin) : Msg :=
  if cs.length < refK x then S.unMsgAt x wt v t cs
  else if S.RowsAtTape x (S.pad cs) wt (v (S.pad cs)) t then S.swMsgAt x wt v t cs
  else S.abort

/-- **The go prover** at the workers' values `vs`: `refP` with the models' simulator (`withSim` at `coinModel`), its
messages on each witness `goMsgAt`; its dummy, coins, nonce, tape and opening check `refP`'s. -/
noncomputable def goP (S : RefSession) (vs : S.SwVals) : Proto Flock.Wit :=
  { S.refP.withSim (fun x _ c => S.coinModel x c) with
    msg := fun x w ν t cs => .ret (S.goMsgAt x (S.wt x w) (vs x w ν) t cs) }

noncomputable instance instFintypeGoPCoin (S : RefSession) (vs : S.SwVals) : Fintype (S.goP vs).Coin := S.finCoin

theorem goP_simulatorE (S : RefSession) (vs : S.SwVals) (t M : ℕ) :
    (S.goP vs).simulatorE t M = (S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t M :=
  Proto.simulatorE_msg _ _ t M

/-- The go prover's first message is the session's `Hello`. -/
theorem goP_next_nil (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit) (ν : S.Nonce)
    (t : S.Tape) : ((((S.goP vs).prover w).next x [] (ν, t)).eval H).1 = S.hello x ν := by
  show ((strat ((S.goP vs).hello x ν) (S.goP vs).abort (S.goP vs).dec ((S.goP vs).chk x ν)
    ((S.goP vs).msg x w ν t) []).eval H).1 = _
  rw [strat_eval_nil]
  rfl

/-- After `Hello`, the go prover sends its message at the coins the openings open while they pass the coin-tree check,
else its `abort` (`strat_eval_cons`). -/
theorem goP_next_cons (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit) (ν : S.Nonce)
    (t : S.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) :
    ((((S.goP vs).prover w).next x ((m₀, ρ₀) :: ops) (ν, t)).eval H).1 =
      if ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 then
        S.goMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2)
      else S.abort := by
  show ((strat ((S.goP vs).hello x ν) (S.goP vs).abort (S.goP vs).dec ((S.goP vs).chk x ν)
    ((S.goP vs).msg x w ν t) ((m₀, ρ₀) :: ops)).eval H).1 = _
  rw [strat_eval_cons]
  rfl

/-- **Before the `Finish`, the go prover is the unswitched prover**: after at most `refK x` rounds past `Hello`. -/
theorem goP_next_lt (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit) (ν : S.Nonce)
    (t : S.Tape) (h : List (Msg × Msg)) (hl : h.length ≤ refK x) :
    ((((S.goP vs).prover w).next x h (ν, t)).eval H).1 = ((((S.unP vs).prover w).next x h (ν, t)).eval H).1 := by
  rcases h with _ | ⟨⟨m₀, ρ₀⟩, ops⟩
  · rw [S.goP_next_nil, S.unP_next_nil]
  · rw [S.goP_next_cons, S.unP_next_cons]
    have hl' : (ops.map fun p => S.decOf p.2).length < refK x := by
      simp only [List.length_cons] at hl
      simp only [List.length_map]
      omega
    unfold goMsgAt
    rw [ite_eq_left hl']

/-- **At the `Finish`, where the rows hold at the coins the openings open, the go prover sends the switched prover's
last message** while the openings pass, else its `abort`. -/
theorem goP_next_rows (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit) (ν : S.Nonce)
    (t : S.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hl : ops.length = refK x)
    (hr : S.RowsAtTape x (S.pad (ops.map fun p => S.decOf p.2)) (S.wt x w)
      (vs x w ν (S.pad (ops.map fun p => S.decOf p.2))) t) :
    ((((S.goP vs).prover w).next x ((m₀, ρ₀) :: ops) (ν, t)).eval H).1 =
      if ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 then
        S.swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2)
      else S.abort := by
  have hl' : ¬(ops.map fun p => S.decOf p.2).length < refK x := by
    simp only [List.length_map]
    omega
  rw [S.goP_next_cons]
  unfold goMsgAt
  rw [ite_eq_right hl', ite_eq_left hr]

/-- `goP_next_lt` at the session with the frame's stops. -/
theorem withStops_goP_next_lt (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit) (ν : S.Nonce) (t : S.Tape) (h : List (Msg × Msg))
    (hl : h.length ≤ refK x) :
    (((((S.withStops Rf stop).goP vs).prover w).next x h (ν, t)).eval H).1 =
      (((((S.withStops Rf stop).unP vs).prover w).next x h (ν, t)).eval H).1 :=
  (S.withStops Rf stop).goP_next_lt vs H x w ν t h hl

/-- `goP_next_rows` at the session with the frame's stops, in `S`'s own terms: its `abort` is the stop. -/
theorem withStops_goP_next_rows (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit) (ν : S.Nonce) (t : S.Tape) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (hl : ops.length = refK x)
    (hr : S.RowsAtTape x (S.pad (ops.map fun p => S.decOf p.2)) (S.wt x w)
      (vs x w ν (S.pad (ops.map fun p => S.decOf p.2))) t) :
    (((((S.withStops Rf stop).goP vs).prover w).next x ((m₀, ρ₀) :: ops) (ν, t)).eval H).1 =
      if ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 then
        (S.withStops Rf stop).swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2)
      else stop :=
  (S.withStops Rf stop).goP_next_rows vs H x w ν t m₀ ρ₀ ops hl hr

/-! ## (HA) at the go prover -/

/-- **Before the last coin, the go prover sends the unswitched round encoding at the coins the answers open**
(`UnCausal`). -/
theorem goPre_msg (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit) (ν : S.Nonce)
    (hc : S.UnCausal vs) (t : S.Tape) (a : Fin (refK x + 1) → Out) (i : Fin (refK x + 1)) :
    S.refP.pmG H x ν (fun cs => .ret (S.goMsgAt x (S.wt x w) (vs x w ν) t cs)) (ansHist a i) =
      S.refP.pmG H x ν (fun _ => .ret (S.enc x (S.coinsOf a) (i.val - 1)
        (S.unCutOf x (S.coinsOf a) (S.wt x w) (vs x w ν (S.coinsOf a)) t))) (ansHist a i) := by
  rw [← S.unPre_msg vs H x w ν hc t a i]
  unfold Proto.pmG
  congr 2
  refine strat_congr_gen _ _ _ _ _ _ _ fun hne => ?_
  have hi : 0 < i.val := by
    rcases Nat.eq_zero_or_pos i.val with h0 | h0
    · exact absurd (by simp [ansHist, h0]) hne
    · exact h0
  have hl : ((ansHist a i).tail.map fun p => S.decOf p.2).length = i.val - 1 := by
    simp only [List.length_map, List.length_tail, ansHist, List.length_ofFn]
  show OC.ret (S.goMsgAt x (S.wt x w) (vs x w ν) t ((ansHist a i).tail.map fun p => S.decOf p.2)) =
    OC.ret (S.unMsgAt x (S.wt x w) (vs x w ν) t ((ansHist a i).tail.map fun p => S.decOf p.2))
  unfold goMsgAt
  rw [hl, ite_eq_left (by omega)]

/-- `preModel` at `goP`. -/
noncomputable def goPreModel (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (ν : S.Nonce) :
    (S.goP vs).TPreModel x where
  J := S.J x
  C := S.C x
  cz a := S.cz x (S.coinsOf a)
  out a σ i := S.refP.pmG H x ν (fun _ => .ret (S.enc x (S.coinsOf a) (i.val - 1) σ)) (ansHist a i)
  wt := S.wt x
  wt₀ := S.wt₀ x
  eR := S.eP x
  kR := S.kP x

theorem goPreModel_link (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop)
    (hV : S.Complete R) (hD : S.DummyReg) (hc : S.Causal) (huc : S.UnCausal vs) (hT : S.TapeOKOn A) (x : Msg)
    (hx : A x) (w : Flock.Wit) (hR : R x w) (ν : S.Nonce) : (S.goPreModel vs H x ν).UnLink H ν w := by
  refine ⟨hV x w hR, fun a j => hD x (S.coinsOf a) j, ?_, fun p => (hT x hx).1 p _,
    ⟨fun a => vs x w ν (S.coinsOf a), ?_, fun t a => funext fun i => S.goPre_msg vs H x w ν huc t a i⟩,
    fun t a => funext fun i => S.pre_msg H x ν hc (S.wt₀ x) t a i⟩
  · intro wt ω p a a' i h
    show S.refP.pmG H x ν _ (ansHist a i) = S.refP.pmG H x ν _ (ansHist a' i)
    rw [ansHist_congr a a' i h]
    refine congrArg (fun g => S.refP.pmG H x ν g (ansHist a' i)) (funext fun _ => congrArg OC.ret ?_)
    exact hc x wt ω p (S.coinsOf a) (S.coinsOf a') (i.val - 1) fun k hk =>
      congrArg (fun o => S.decOf (msgOf o)) (h k.succ (Fin.lt_def.mpr (by simp only [Fin.val_succ]; omega)))
  · intro ω s p a a' i h
    show S.refP.pmG H x ν _ (ansHist a i) = S.refP.pmG H x ν _ (ansHist a' i)
    rw [ansHist_congr a a' i h]
    refine congrArg (fun g => S.refP.pmG H x ν g (ansHist a' i)) (funext fun _ => congrArg OC.ret ?_)
    exact huc x w ν ω s p (S.coinsOf a) (S.coinsOf a') (i.val - 1) fun k hk =>
      congrArg (fun o => S.decOf (msgOf o)) (h k.succ (Fin.lt_def.mpr (by simp only [Fin.val_succ]; omega)))

/-- **(HA) at `goP`**: before the last coin, the go prover's runs and the dummy's are within `2·|L|·δ₁`
(`TPreModel.runs_close_un`), on a relation whose statements are in `A`. -/
theorem goP_prefixHides (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hV : S.Complete R)
    (hD : S.DummyReg) (hc : S.Causal) (huc : S.UnCausal vs) (hT : S.TapeOKOn A) (hRA : ∀ x w, R x w → A x)
    (hS : S.Tables) : (S.goP vs).PrefixHides H R fun x => 2 * S.nLeaves x * δ₁.toReal :=
  (S.goP vs).prefixHides_of_runs H R _ fun x w hR ν V E =>
    (S.goPreModel vs H x ν).runs_close_un H hT1 hδ₁ ν w
      (S.goPreModel_link vs H R A hV hD hc huc hT x (hRA x w hR) w hR ν) (hS x) V E

/-! ## The switched prover at one coin vector -/

open Classical in
/-- **The switched prover's message after `j` coins at the coin vector `c`**, on table witnesses `wt`, values `v` and
tape `t`: before the last coin round `j`'s encoding of the switched cut at `c`; after it `fin` of the switched views
where the tables admit `c`, else `off` of the switched cut. -/
noncomputable def fixMsgAt (S : RefSession) (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (c : Fin (refK x) → S.Coin)
    (v : (∀ j, ZkView.Seen (S.C x j).z) → ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals) (t : S.Tape) (j : ℕ) : Msg :=
  if j < refK x then S.enc x c j (S.swCutOf x c wt v t)
  else if S.Admits x c then S.fin x c (S.swViewsAt x c wt v t)
  else S.off x c (S.swCutOf x c wt v t)

/-- `fixMsgAt` with `off` after the last coin at every coin vector. -/
noncomputable def fixOffMsgAt (S : RefSession) (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (c : Fin (refK x) → S.Coin)
    (v : (∀ j, ZkView.Seen (S.C x j).z) → ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals) (t : S.Tape) (j : ℕ) : Msg :=
  if j < refK x then S.enc x c j (S.swCutOf x c wt v t) else S.off x c (S.swCutOf x c wt v t)

theorem fixMsgAt_lt (S : RefSession) (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (c : Fin (refK x) → S.Coin)
    (v : (∀ j, ZkView.Seen (S.C x j).z) → ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals) (t : S.Tape) {j : ℕ}
    (h : j < refK x) : S.fixMsgAt x wt c v t j = S.enc x c j (S.swCutOf x c wt v t) := by
  unfold fixMsgAt
  exact ite_eq_left h

theorem fixOffMsgAt_lt (S : RefSession) (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (c : Fin (refK x) → S.Coin)
    (v : (∀ j, ZkView.Seen (S.C x j).z) → ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals) (t : S.Tape) {j : ℕ}
    (h : j < refK x) : S.fixOffMsgAt x wt c v t j = S.enc x c j (S.swCutOf x c wt v t) := by
  unfold fixOffMsgAt
  exact ite_eq_left h

/-- **The switched prover at the coins `l`**: `refP` whose message on a witness after `j` coins is `fixMsgAt` at the
padded coins `l`, whatever coins it was sent. -/
noncomputable def fixP (S : RefSession) (vs : S.SwVals) (l : List S.Coin) : Proto Flock.Wit :=
  { S.refP with
    msg := fun x w ν t cs => .ret (S.fixMsgAt x (S.wt x w) (S.pad l) (vs x w ν (S.pad l)) t cs.length) }

/-- **The refusing switched prover at the coins `l`**: `offP` whose message on a witness after `j` coins is
`fixOffMsgAt` at the padded coins `l`. -/
noncomputable def fixOffP (S : RefSession) (vs : S.SwVals) (l : List S.Coin) : Proto Flock.Wit :=
  { S.offP with
    msg := fun x w ν t cs => .ret (S.fixOffMsgAt x (S.wt x w) (S.pad l) (vs x w ν (S.pad l)) t cs.length) }

theorem fixP_realVec (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit) (ν : S.Nonce)
    (t : S.Tape) (c : Fin (refK x) → S.Coin) :
    (S.fixP vs (List.ofFn c)).realVec H x w ν t c =
      (List.range (refK x + 1)).map (S.fixMsgAt x (S.wt x w) c (vs x w ν c) t) := by
  show (List.range (refK x + 1)).map (fun j => S.fixMsgAt x (S.wt x w) (S.pad (List.ofFn c))
    (vs x w ν (S.pad (List.ofFn c))) t ((List.ofFn c).take j).length) = _
  rw [S.pad_ofFn_self c]
  refine List.map_congr_left fun j hj => ?_
  have hj' := List.mem_range.1 hj
  rw [List.length_take, List.length_ofFn, Nat.min_eq_left (by omega)]

theorem fixOffP_realVec (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit) (ν : S.Nonce)
    (t : S.Tape) (c : Fin (refK x) → S.Coin) :
    (S.fixOffP vs (List.ofFn c)).realVec H x w ν t c =
      (List.range (refK x + 1)).map (S.fixOffMsgAt x (S.wt x w) c (vs x w ν c) t) := by
  show (List.range (refK x + 1)).map (fun j => S.fixOffMsgAt x (S.wt x w) (S.pad (List.ofFn c))
    (vs x w ν (S.pad (List.ofFn c))) t ((List.ofFn c).take j).length) = _
  rw [S.pad_ofFn_self c]
  refine List.map_congr_left fun j hj => ?_
  have hj' := List.mem_range.1 hj
  rw [List.length_take, List.length_ofFn, Nat.min_eq_left (by omega)]

/-- `refP`'s model at a coin vector, typed at `fixP`. -/
noncomputable def toFix (S : RefSession) (vs : S.SwVals) (x : Msg) (l : List S.Coin) :
    S.refP.TCoinModel x → (S.fixP vs l).TCoinModel x
  | .inl M => .inl { J := M.J, C := M.C, cz := M.cz, wt := M.wt, wt₀ := M.wt₀, lift := M.lift, eR := M.eR,
                     kR := M.kR }
  | .inr M => .inr { J := M.J, C := M.C, cz := M.cz, out := M.out, wt := M.wt, wt₀ := M.wt₀, eR := M.eR,
                     kR := M.kR }

open Classical in
/-- `coinModel` at `fixP` at the coin vector itself. -/
noncomputable def fixCoinModel (S : RefSession) (vs : S.SwVals) (x : Msg) (c : Fin (refK x) → S.Coin) :
    (S.fixP vs (List.ofFn c)).TCoinModel x :=
  if S.Admits x c then
    .inl { J := S.J x, C := S.C x, cz := S.cz x c, wt := S.wt x, wt₀ := S.wt₀ x
           lift := fun V => (List.range (refK x + 1)).map fun j =>
             if j < refK x then S.enc x c j (S.cutV x c V) else S.fin x c V
           eR := S.randAt x c, kR := S.kT x c }
  else
    .inr { J := S.J x, C := S.C x, cz := S.cz x c, wt := S.wt x, wt₀ := S.wt₀ x
           out := fun σ => (List.range (refK x + 1)).map fun j => if j < refK x then S.enc x c j σ else S.off x c σ
           eR := S.eP x, kR := S.kP x }

theorem fixCoinModel_eq (S : RefSession) (vs : S.SwVals) (x : Msg) (c : Fin (refK x) → S.Coin) :
    S.fixCoinModel vs x c = S.toFix vs x (List.ofFn c) (S.coinModel x c) := by
  unfold fixCoinModel coinModel
  split_ifs <;> rfl

theorem fixCoinModel_link (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop)
    (hV : S.Complete R) (hD : S.DummyReg) (hc : S.Causal) (hT : S.TapeOKOn A) (x : Msg) (hx : A x) (w : Flock.Wit)
    (hR : R x w) (ν : S.Nonce) (c : Fin (refK x) → S.Coin) : (S.fixCoinModel vs x c).SwLink H ν c w := by
  unfold fixCoinModel
  split_ifs with hA
  · refine ⟨vs x w ν c, hV x w hR, fun p => (hT x hx).2 c p _, fun (t : S.Tape) => ?_⟩
    refine (S.fixP_realVec vs H x w ν t c).trans ?_
    refine Eq.trans ?_ (congrArg (fun V => (List.range (refK x + 1)).map fun j =>
      if j < refK x then S.enc x c j (S.cutV x c V) else S.fin x c V) (S.swViewsAt_eq x c (S.wt x w) (vs x w ν c) t))
    show (List.range (refK x + 1)).map (S.fixMsgAt x (S.wt x w) c (vs x w ν c) t) =
      (List.range (refK x + 1)).map fun j =>
        if j < refK x then S.enc x c j (S.cutV x c (S.swViewsAt x c (S.wt x w) (vs x w ν c) t))
        else S.fin x c (S.swViewsAt x c (S.wt x w) (vs x w ν c) t)
    refine List.map_congr_left fun j _ => ?_
    unfold fixMsgAt
    rw [S.cutV_swViewsAt, ite_eq_left hA]
  · refine Proto.TCoinModel.swLink_inr (G := S.fixP vs (List.ofFn c)) (x := x) _ H ν c w ?_
    refine ⟨hV x w hR, fun j => hD x c j, fun p => (hT x hx).1 p _, ⟨vs x w ν c, fun (t : S.Tape) => ?_⟩,
      fun (t : S.Tape) => ?_⟩
    · refine (S.fixP_realVec vs H x w ν t c).trans ?_
      show (List.range (refK x + 1)).map (S.fixMsgAt x (S.wt x w) c (vs x w ν c) t) =
        (List.range (refK x + 1)).map fun j =>
          if j < refK x then S.enc x c j (S.swCutOf x c (S.wt x w) (vs x w ν c) t)
          else S.off x c (S.swCutOf x c (S.wt x w) (vs x w ν c) t)
      refine List.map_congr_left fun j _ => ?_
      unfold fixMsgAt
      rw [ite_eq_right hA]
    · show (List.range (refK x + 1)).map (fun j => S.msgAt x (S.wt₀ x) t ((List.ofFn c).take j)) =
        (List.range (refK x + 1)).map fun j =>
          if j < refK x then S.enc x c j (S.cutOf x c (S.wt₀ x) t) else S.off x c (S.cutOf x c (S.wt₀ x) t)
      refine List.map_congr_left fun j hj => ?_
      rw [S.msgAt_take x hc _ t c j (List.mem_range.1 hj), ite_eq_right hA]

theorem fixCoinModel_ok (S : RefSession) (vs : S.SwVals) (x : Msg) (c : Fin (refK x) → S.Coin) (hT : S.Tables) :
    (S.fixCoinModel vs x c).OK := by
  unfold fixCoinModel
  split_ifs with h
  · exact ⟨hT x, h⟩
  · exact hT x

theorem fixCoinModel_size (S : RefSession) (vs : S.SwVals) (x : Msg) (c : Fin (refK x) → S.Coin) :
    (S.fixCoinModel vs x c).size ≤ S.nLeaves x := by
  unfold fixCoinModel
  split_ifs
  · show ∑ j, Fintype.card ((S.C x j).T (S.cz x c j)).tab.Hid ≤ Fintype.card (ZkView.Leaves fun j => (S.C x j).z)
    rw [Fintype.card_sigma]
    exact Finset.sum_le_sum fun j _ => Fintype.card_le_of_injective _ (hidPos_injective ((S.C x j).T (S.cz x c j)))
  · exact le_rfl

/-- **`SHVZK` of `fixP` at a coin vector against `refP`'s model's simulator** (`TCoinModel.shvzk_sw` at `toFix`). -/
theorem toFix_shvzk (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (x : Msg) (ν : S.Nonce)
    (c : Fin (refK x) → S.Coin) (w : Flock.Wit) {n : ℕ} (E : List Msg → Prop) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (sim : Ω → List Msg) :
    ∀ (m : S.refP.TCoinModel x) (ev : Ω → m.SimAt) (kS : ℕ), (∀ p, Nat.card {s // ev s = p} = kS) →
      (∀ s, sim s = ((m.simAt (ev s) ν c).eval H).1) → (S.toFix vs x (List.ofFn c) m).SwLink H ν c w →
      (S.toFix vs x (List.ofFn c) m).OK → (S.toFix vs x (List.ofFn c) m).size ≤ n →
      |Pr (fun r : S.Tape => E ((S.fixP vs (List.ofFn c)).realVec H x w ν r c)) - Pr (fun s : Ω => E (sim s))|
        ≤ 2 * n * δ₁.toReal
  | .inl M, ev, kS, hev, hsim, hL, hOK, hn =>
      Proto.TCoinModel.shvzk_sw (G := S.fixP vs (List.ofFn c)) H hT1 hδ₁ ν c w E sim
        (S.toFix vs x (List.ofFn c) (.inl M)) ev kS hev hsim hL hOK hn
  | .inr M, ev, kS, hev, hsim, hL, hOK, hn =>
      Proto.TCoinModel.shvzk_sw (G := S.fixP vs (List.ofFn c)) H hT1 hδ₁ ν c w E sim
        (S.toFix vs x (List.ofFn c) (.inr M)) ev kS hev hsim hL hOK hn

/-- **`fixP`'s messages at its coin vector against the go prover's simulator** (`refP`'s, `withSim` at `coinModel`):
within `2·|L|·δ₁` on every event, whatever values the workers supply, on a relation whose statements are in `A`. -/
theorem fixP_shvzk (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hV : S.Complete R)
    (hD : S.DummyReg) (hc : S.Causal) (hT : S.TapeOKOn A) (hRA : ∀ x w, R x w → A x) (hS : S.Tables) (x : Msg)
    (w : Flock.Wit) (hR : R x w) (ν : S.Nonce) (c : Fin (refK x) → S.Coin) (E : List Msg → Prop) :
    |Pr (fun r : S.Tape => E ((S.fixP vs (List.ofFn c)).realVec H x w ν r c)) -
      Pr (fun s : (S.goP vs).SimRand x => E (((S.goP vs).sim x ν c s).eval H).1)| ≤ 2 * S.nLeaves x * δ₁.toReal := by
  obtain ⟨p₀⟩ := Proto.TCoinModel.instNonemptySimAt (S.coinModel x c)
  have e := S.fixCoinModel_eq vs x c
  have hL := S.fixCoinModel_link vs H R A hV hD hc hT x (hRA x w hR) w hR ν c
  have hOK := S.fixCoinModel_ok vs x c hS
  have hn := S.fixCoinModel_size vs x c
  rw [e] at hL hOK hn
  exact S.toFix_shvzk vs H hT1 hδ₁ x ν c w E (Ω := (S.goP vs).SimRand x)
    (fun s => (((S.goP vs).sim x ν c s).eval H).1) (S.coinModel x c) (fun s => s (ν, c)) _
    (fun p => Proto.eval_fiber_card
      (β := fun q : S.Nonce × (Fin (refK x) → S.Coin) => (S.coinModel x q.2).SimAt) (ν, c) p p₀)
    (fun _ => rfl) hL hOK hn

/-- The model of `fixOffP` at its coin vector: the switched cut view with `enc` and `off`. -/
noncomputable def fixOffModel (S : RefSession) (vs : S.SwVals) (x : Msg) (c : Fin (refK x) → S.Coin) :
    (S.fixOffP vs (List.ofFn c)).TOffModel x where
  J := S.J x
  C := S.C x
  cz := S.cz x c
  out σ := (List.range (refK x + 1)).map fun j => if j < refK x then S.enc x c j σ else S.off x c σ
  wt := S.wt x
  wt₀ := S.wt₀ x
  eR := S.eP x
  kR := S.kP x

theorem fixOffModel_link (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop)
    (hV : S.Complete R) (hD : S.DummyReg) (hc : S.Causal) (hT : S.TapeOKOn A) (x : Msg) (hx : A x) (w : Flock.Wit)
    (hR : R x w) (ν : S.Nonce) (c : Fin (refK x) → S.Coin) : (S.fixOffModel vs x c).SwLink H ν c w := by
  refine ⟨hV x w hR, fun j => hD x c j, fun p => (hT x hx).1 p _, ⟨vs x w ν c, fun (t : S.Tape) => ?_⟩,
    fun (t : S.Tape) => ?_⟩
  · exact S.fixOffP_realVec vs H x w ν t c
  · show (List.range (refK x + 1)).map (fun j => S.offMsgAt x (S.wt₀ x) t ((List.ofFn c).take j)) =
      (List.range (refK x + 1)).map fun j =>
        if j < refK x then S.enc x c j (S.cutOf x c (S.wt₀ x) t) else S.off x c (S.cutOf x c (S.wt₀ x) t)
    exact List.map_congr_left fun j hj => S.offMsgAt_take x hc _ t c j (List.mem_range.1 hj)

/-! ## (HB) one-sided at the go prover -/

theorem unPS_realVec (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit) (ν : S.Nonce)
    (huc : S.UnCausal vs) (t : S.Tape) (c : Fin (refK x) → S.Coin) :
    (S.unPS vs).realVec H x w ν t c = (List.range (refK x + 1)).map fun j =>
      if j < refK x then S.enc x c j (S.unCutOf x c (S.wt x w) (vs x w ν c) t)
      else S.off x c (S.unCutOf x c (S.wt x w) (vs x w ν c) t) := by
  show (List.range (refK x + 1)).map (fun j => S.unMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take j)) = _
  exact List.map_congr_left fun j hj => S.unMsgAt_take vs x w ν huc t c j (List.mem_range.1 hj)

/-- The switched prover's last message at a coin vector is `fixMsgAt`'s. -/
theorem swMsgAt_ofFn (S : RefSession) (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (v : (c : Fin (refK x) → S.Coin) → (∀ j, ZkView.Seen (S.C x j).z) → ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals)
    (t : S.Tape) (c : Fin (refK x) → S.Coin) :
    S.swMsgAt x wt v t ((List.ofFn c).take (refK x)) = S.fixMsgAt x wt c (v c) t (refK x) := by
  have hl : ((List.ofFn c).take (refK x)).length = refK x := by
    simp only [List.length_take, List.length_ofFn, Nat.min_self]
  unfold swMsgAt fixMsgAt
  rw [hl, S.pad_ofFn c]

open Classical in
/-- **The go prover's messages at a coin vector** (`UnCausal`): the unswitched round encoding before the last coin;
then `fixMsgAt`'s where the rows hold at the tape, else `abort`. -/
theorem goP_realVec (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit) (ν : S.Nonce)
    (huc : S.UnCausal vs) (t : S.Tape) (c : Fin (refK x) → S.Coin) :
    (S.goP vs).realVec H x w ν t c = (List.range (refK x + 1)).map fun j =>
      if j < refK x then S.enc x c j (S.unCutOf x c (S.wt x w) (vs x w ν c) t)
      else if S.RowsAtTape x c (S.wt x w) (vs x w ν c) t then S.fixMsgAt x (S.wt x w) c (vs x w ν c) t j
      else S.abort := by
  show (List.range (refK x + 1)).map (fun j => S.goMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take j)) = _
  refine List.map_congr_left fun j hj => ?_
  have hj' := List.mem_range.1 hj
  have hl : ((List.ofFn c).take j).length = j := by simp only [List.length_take, List.length_ofFn]; omega
  by_cases h : j < refK x
  · have e : S.goMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take j) =
        S.unMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take j) := by
      unfold goMsgAt
      rw [hl, ite_eq_left h]
    rw [e, S.unMsgAt_take vs x w ν huc t c j hj']
    split_ifs
    rfl
  · obtain rfl : j = refK x := by omega
    have e : S.goMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take (refK x)) =
        if S.RowsAtTape x c (S.wt x w) (vs x w ν c) t then
          S.swMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take (refK x))
        else S.abort := by
      unfold goMsgAt
      rw [hl, ite_eq_right h, S.pad_ofFn c]
    rw [e, S.swMsgAt_ofFn, ite_eq_right h]

theorem goP_realVec_rows (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit) (ν : S.Nonce)
    (huc : S.UnCausal vs) (t : S.Tape) (c : Fin (refK x) → S.Coin)
    (hr : S.RowsAtTape x c (S.wt x w) (vs x w ν c) t) :
    (S.goP vs).realVec H x w ν t c = (S.fixP vs (List.ofFn c)).realVec H x w ν t c := by
  rw [S.goP_realVec vs H x w ν huc t c, S.fixP_realVec vs H x w ν t c]
  refine List.map_congr_left fun j _ => ?_
  by_cases h : j < refK x
  · rw [ite_eq_left h, S.fixMsgAt_lt x _ c _ t h, S.swCutOf_of_rows x c _ _ t hr]
  · rw [ite_eq_right h, ite_eq_left hr]

theorem goP_realVec_not_rows (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit)
    (ν : S.Nonce) (huc : S.UnCausal vs) (t : S.Tape) (c : Fin (refK x) → S.Coin)
    (hr : ¬S.RowsAtTape x c (S.wt x w) (vs x w ν c) t) :
    (S.goP vs).realVec H x w ν t c = ((S.unPS vs).realVec H x w ν t c).set (refK x) S.abort := by
  rw [S.goP_realVec vs H x w ν huc t c, S.unPS_realVec vs H x w ν huc t c, map_range_set]
  refine List.map_congr_left fun j hj => ?_
  have hj' := List.mem_range.1 hj
  split_ifs <;> first | rfl | omega

theorem unPS_realVec_set_rows (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit)
    (ν : S.Nonce) (huc : S.UnCausal vs) (t : S.Tape) (c : Fin (refK x) → S.Coin)
    (hr : S.RowsAtTape x c (S.wt x w) (vs x w ν c) t) :
    ((S.unPS vs).realVec H x w ν t c).set (refK x) S.abort =
      ((S.fixOffP vs (List.ofFn c)).realVec H x w ν t c).set (refK x) S.abort := by
  rw [S.unPS_realVec vs H x w ν huc t c, S.fixOffP_realVec vs H x w ν t c, map_range_set, map_range_set]
  refine List.map_congr_left fun j _ => ?_
  unfold fixOffMsgAt
  rw [S.swCutOf_of_rows x c _ _ t hr]

theorem fixOffP_realVec_set (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit)
    (ν : S.Nonce) (t : S.Tape) (c : Fin (refK x) → S.Coin) :
    ((S.fixOffP vs (List.ofFn c)).realVec H x w ν t c).set (refK x) S.abort =
      ((S.fixP vs (List.ofFn c)).realVec H x w ν t c).set (refK x) S.abort := by
  rw [S.fixOffP_realVec vs H x w ν t c, S.fixP_realVec vs H x w ν t c, map_range_set, map_range_set]
  refine List.map_congr_left fun j hj => ?_
  have hj' := List.mem_range.1 hj
  unfold fixOffMsgAt fixMsgAt
  split_ifs <;> first | rfl | omega

/-- **(HB) one-sided at `goP`, at one coin vector**: on every test with values in `[0, 1]` that doesn't grow when the
last message becomes `abort`, the go prover's messages are at most the honest simulator's plus `6·|L|·δ₁`. Pointwise on
the tape, the go prover's test plus `fixOffP`'s ending in `abort` is at most `fixP`'s plus the unswitched prover's
ending in `abort`; `fixP` is within `2·|L|·δ₁` of the simulator (`fixP_shvzk`), and the unswitched prover and `fixOffP`
are each within `2·|L|·δ₁` of the refusing dummy (`close_un`, `close_sw`). -/
theorem goP_shvzkLe_at (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hV : S.Complete R)
    (hD : S.DummyReg) (hc : S.Causal) (huc : S.UnCausal vs) (hT : S.TapeOKOn A) (hRA : ∀ x w, R x w → A x)
    (hS : S.Tables) (x : Msg) (w : Flock.Wit) (hR : R x w) (ν : S.Nonce) (c : Fin (refK x) → S.Coin)
    (Ψ : List Msg → ℝ) (h0 : ∀ v, 0 ≤ Ψ v) (h1 : ∀ v, Ψ v ≤ 1) (hΨ : ∀ v, Ψ (v.set (refK x) S.abort) ≤ Ψ v) :
    Ex (fun r : S.Tape => Ψ ((S.goP vs).realVec H x w ν r c)) ≤
      Ex (fun s : (S.goP vs).SimRand x => Ψ (((S.goP vs).sim x ν c s).eval H).1) + 6 * S.nLeaves x * δ₁.toReal := by
  have hpt : ∀ r : S.Tape,
      Ψ ((S.goP vs).realVec H x w ν r c) +
          Ψ (((S.fixOffP vs (List.ofFn c)).realVec H x w ν r c).set (refK x) S.abort) ≤
        Ψ ((S.fixP vs (List.ofFn c)).realVec H x w ν r c) +
          Ψ (((S.unPS vs).realVec H x w ν r c).set (refK x) S.abort) := by
    intro r
    by_cases hr : S.RowsAtTape x c (S.wt x w) (vs x w ν c) r
    · exact le_of_eq (by
        rw [S.goP_realVec_rows vs H x w ν huc r c hr, S.unPS_realVec_set_rows vs H x w ν huc r c hr])
    · rw [S.goP_realVec_not_rows vs H x w ν huc r c hr, S.fixOffP_realVec_set vs H x w ν r c]
      linarith [hΨ ((S.fixP vs (List.ofFn c)).realVec H x w ν r c)]
  have hint := Ex_mono hpt
  rw [Ex_add, Ex_add] at hint
  have b1 : |Ex (fun r : S.Tape => Ψ ((S.fixP vs (List.ofFn c)).realVec H x w ν r c)) -
      Ex (fun s : (S.goP vs).SimRand x => Ψ (((S.goP vs).sim x ν c s).eval H).1)| ≤ 2 * S.nLeaves x * δ₁.toReal :=
    abs_Ex_sub_le_of_events (fun r : S.Tape => (S.fixP vs (List.ofFn c)).realVec H x w ν r c)
      (fun s : (S.goP vs).SimRand x => (((S.goP vs).sim x ν c s).eval H).1)
      (fun E => S.fixP_shvzk vs H R A hT1 hδ₁ hV hD hc hT hRA hS x w hR ν c E) Ψ h0 h1
  have b2 : |Ex (fun r : S.Tape => Ψ (((S.unPS vs).realVec H x w ν r c).set (refK x) S.abort)) -
      Ex (fun r : S.Tape => Ψ (((S.unPS vs).dVec H x ν r c).set (refK x) S.abort))| ≤
        2 * S.nLeaves x * δ₁.toReal :=
    abs_Ex_sub_le_of_events (fun r : S.Tape => (S.unPS vs).realVec H x w ν r c)
      (fun r : S.Tape => (S.unPS vs).dVec H x ν r c)
      (fun E => (S.unOffModel vs x c).close_un H hT1 hδ₁ ν c w
        (S.unOffModel_link vs H R A hV hD hc huc hT x (hRA x w hR) w hR ν c) (hS x) E)
      (fun v => Ψ (v.set (refK x) S.abort)) (fun _ => h0 _) (fun _ => h1 _)
  have b3 : |Ex (fun r : S.Tape => Ψ (((S.fixOffP vs (List.ofFn c)).realVec H x w ν r c).set (refK x) S.abort)) -
      Ex (fun r : S.Tape => Ψ (((S.unPS vs).dVec H x ν r c).set (refK x) S.abort))| ≤
        2 * S.nLeaves x * δ₁.toReal :=
    abs_Ex_sub_le_of_events (fun r : S.Tape => (S.fixOffP vs (List.ofFn c)).realVec H x w ν r c)
      (fun r : S.Tape => (S.unPS vs).dVec H x ν r c)
      (fun E => (S.fixOffModel vs x c).close_sw H hT1 hδ₁ ν c w
        (S.fixOffModel_link vs H R A hV hD hc hT x (hRA x w hR) w hR ν c) (hS x) E)
      (fun v => Ψ (v.set (refK x) S.abort)) (fun _ => h0 _) (fun _ => h1 _)
  rw [abs_le] at b1 b2 b3
  linarith [b1.1, b1.2, b2.1, b2.2, b3.1, b3.2]

/-- **(HB) one-sided at `goP`** (`goP_shvzkLe_at` at every coin vector), on a relation whose statements are in `A`. -/
theorem goP_shvzkLe (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hV : S.Complete R)
    (hD : S.DummyReg) (hc : S.Causal) (huc : S.UnCausal vs) (hT : S.TapeOKOn A) (hRA : ∀ x w, R x w → A x)
    (hS : S.Tables) : (S.goP vs).SHVZKLe H R fun x => 6 * S.nLeaves x * δ₁.toReal :=
  fun x w hR ν c Ψ h0 h1 hΨ =>
    S.goP_shvzkLe_at vs H R A hT1 hδ₁ hV hD hc huc hT hRA hS x w hR ν c Ψ h0 h1 hΨ

/-- **Where the go prover's run goes on at its last round, it is simulated**: `gkE_go_pow` at `goP` with (HA) within
`2·|L|·δ₁` and (HB) one-sided within `6·|L|·δ₁`, at the honest prover's Goldreich–Kahan simulator (`refP`'s with the
models' simulator, which reads no worker's values), for any values whose unswitched round encoding is causal, under
`Claims.Hm96Hiding` and the session's Props, on a relation whose statements are in `A`. -/
theorem goP_go (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop) (t : ℕ)
    (ht : 0 < t) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (hS : S.Tables) (hV : S.Complete R) (hD : S.DummyReg) (hc : S.Causal) (huc : S.UnCausal vs)
    (hT : S.TapeOKOn A) (hRA : ∀ x w, R x w → A x) :
    ∀ x w, R x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = (S.goP vs).k x → ∀ Ev : List (Msg × Out) → Prop,
      Pr (fun ω : (S.goP vs).Nonce × (S.goP vs).Rand =>
          goesOnAt (S.goP vs).abort ((S.goP vs).k x - 1) (runStar H ((S.goP vs).prover w) Vs x ω θ).msgs = true ∧
            Ev (runStar H ((S.goP vs).prover w) Vs x ω θ).msgs) ≤
        Pr (fun c : ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t)).C x =>
            Ev (((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t)).out H Vs θ x c)) +
          (8 * S.nLeaves x * δ₁.toReal + 24 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t +
            4 * t * 3 ^ t * (Fork H ((S.goP vs).prover w) Vs θ x).Pr H) := by
  have h := (S.goP vs).gkE_go_pow H R t ht (fun x => 2 * S.nLeaves x * δ₁.toReal)
    (fun x => 6 * S.nLeaves x * δ₁.toReal) (S.goP_prefixHides vs H R A hT1 hδ₁ hV hD hc huc hT hRA hS)
    (S.goP_shvzkLe vs H R A hT1 hδ₁ hV hD hc huc hT hRA hS) (fun x => by positivity) (S.binding H)
  intro x w hR Θ Vs θ hk Ev
  have h1 := h x w hR Θ Vs θ hk Ev
  rw [S.goP_simulatorE vs] at h1
  refine h1.trans (le_of_eq ?_)
  ring

/-- **`goP_go` at the session with the frame's stops**, where the stops read the coins before their round only
(`StopsCausal`). -/
theorem withStops_goP_go (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables) (hV : S.Complete R)
    (hD : S.DummyReg) (hc : S.Causal) (hRf : S.StopsCausal Rf) (huc : S.UnCausal vs) (hT : S.TapeOKOn A)
    (hRA : ∀ x w, R x w → A x) :
    ∀ x w, R x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = ((S.withStops Rf stop).goP vs).k x →
      ∀ Ev : List (Msg × Out) → Prop,
      Pr (fun ω : ((S.withStops Rf stop).goP vs).Nonce × ((S.withStops Rf stop).goP vs).Rand =>
          goesOnAt ((S.withStops Rf stop).goP vs).abort (((S.withStops Rf stop).goP vs).k x - 1)
              (runStar H (((S.withStops Rf stop).goP vs).prover w) Vs x ω θ).msgs = true ∧
            Ev (runStar H (((S.withStops Rf stop).goP vs).prover w) Vs x ω θ).msgs) ≤
        Pr (fun c : (((S.withStops Rf stop).refP.withSim fun x _ c => (S.withStops Rf stop).coinModel x c).simulatorE
              t (t * 3 ^ t)).C x =>
            Ev ((((S.withStops Rf stop).refP.withSim fun x _ c => (S.withStops Rf stop).coinModel x c).simulatorE
              t (t * 3 ^ t)).out H Vs θ x c)) +
          (8 * S.nLeaves x * δ₁.toReal + 24 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t +
            4 * t * 3 ^ t * (Fork H (((S.withStops Rf stop).goP vs).prover w) Vs θ x).Pr H) :=
  (S.withStops Rf stop).goP_go vs H R A t ht hT1 hδ₁ (S.withStops_tables Rf stop hS)
    (S.withStops_complete Rf stop hV) (S.withStops_dummyReg Rf stop hD) (S.withStops_causal Rf stop hRf hc)
    (S.withStops_unCausal Rf stop vs hRf huc) hT hRA

end RefSession

end Claims.ZKMal

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge

namespace LeanRelease

/-- **The `Finish` goes on only where the rows hold**: after every history of `refK x` openings whose rounds all passed
and whose `Finish` the ZK server's own `Finish` takes (`live`), whose openings pass, whose last coins the frame takes and
whose messages are the deferred release's own (`ZkServer.OwnHist`), the workers' values `vs` satisfy every table's rows
at the pads the tape samples at the coins the openings open (`RefSession.RowsAtTape`). A named hypothesis, one of the
transport's trusts, beside `FinalRoundEq`: the ZK server's own `Finish` stops where the workers' values fail a constraint
at the sampled pads (`refusesFinish`), since `Flock.Firewall.Finish.repFw` releases a rep's inner proof only where none
fails on its pads, and where that check passes at both reps the rows hold (`Composed.gated_of_cons`). -/
def FinishRows (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)),
    ops.length = refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops) →
    S.RowsAtTape x (S.pad (ops.map fun p => S.decOf p.2)) (S.wt x w)
      (vs x w ν (S.pad (ops.map fun p => S.decOf p.2))) t

/-- **Where the deferred release goes on at its `Finish`, it is the go prover's run.** At the session with the frame's
stops, a last round `n x` one past the `Finish` and the `Finish` one past the last coin, for every worker and any values
`vs`: a release whose message at the `Finish` is not the stop took the `Finish` (`live`), so the rows hold at its coins
(`FinishRows`); below the `Finish` it is the unswitched prover's run (`refDBelow_unP`), which is the go prover's there
(`goP_next_lt`), and at the `Finish` it is `Final.encode` of the ZK server's files (`FinalRoundEq`), the switched
prover's last message, which the go prover sends where the rows hold (`goP_next_rows`). -/
theorem release_go (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (n : Msg → ℕ) (wk : L.toSrv.Worker)
    (x : Msg) (w : Flock.Wit) (hn : n x = refK x + 2) (hlast : L.last x = refK x + 1) (hR : StopsAt S Rf L H x)
    (hC : CheckIs S L H x) (hH : HelloEq S L x) (hLk : LinkRoundEqStop S vs L x w)
    (hE : LiveRoundEqStop S vs L H wk x w) (hEf : FillRoundEqStop S vs L H wk x w)
    (hF : FinalRoundEq S vs L H wk x w) (hRw : FinishRows S vs L H wk x w) (ask : List Msg → Out)
    (ρ : S.refP.Nonce × S.refP.Rand) (κ : L.toSrv.Chk)
    (hg : goesOnAt L.stop (n x - 1) (interact ask (L.toSrv.nextD L.fillOf n wk x w ρ κ) (n x) []) = true) :
    interact ask (L.toSrv.nextD L.fillOf n wk x w ρ κ) (n x) [] =
      interact ask (fun h => (((((S.withStops Rf L.stop).goP vs).prover w).next x h ρ).eval H).1) (n x) [] := by
  obtain ⟨ν, t⟩ := ρ
  have hO := openingsRefused_of_checkIs S L H x hC
  have hRef := L.refDBelow_unP S Rf vs H n wk x w hn hlast hR hO hH hLk hE hEf
  set τ := interact ask (L.toSrv.nextD L.fillOf n wk x w (ν, t) κ) (n x) [] with hτ
  have hτn : τ.length ≤ n x := interact_length_le ask _ (n x) []
  have hN : n x - 1 < τ.length := by
    by_contra hc
    unfold goesOnAt at hg
    rw [List.getElem?_eq_none (show τ.length ≤ n x - 1 by omega)] at hg
    simp at hg
  have hgo : (τ[n x - 1]'hN).1 ≠ L.stop := by
    unfold goesOnAt at hg
    rw [List.getElem?_eq_getElem hN] at hg
    simpa using hg
  obtain ⟨⟨m₀, ρ₀⟩, ops, hfe⟩ : ∃ p ops, histOf (τ.take (n x - 1)) = p :: ops :=
    List.exists_cons_of_length_pos (by rw [length_histOf_take]; omega)
  have hops : ops.length = refK x := by
    have e := congrArg List.length hfe
    rw [length_histOf_take, List.length_cons] at e
    omega
  have hg1 := interact_getElem_fst ask (L.toSrv.nextD L.fillOf n wk x w (ν, t) κ) (n x) [] (n x - 1) hN
  rw [← hτ, List.nil_append, List.getElem?_eq_getElem hN, hfe] at hg1
  have hstop : L.toSrv.nextD L.fillOf n wk x w (ν, t) κ ((m₀, ρ₀) :: ops) ≠ L.stop := by
    have e : (τ[n x - 1]'hN).1 = L.toSrv.nextD L.fillOf n wk x w (ν, t) κ ((m₀, ρ₀) :: ops) := Option.some.inj hg1
    rw [← e]
    exact hgo
  have hlive : L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true := by
    cases hl : L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) with
    | true => rfl
    | false =>
      refine absurd ?_ hstop
      have hno : ¬(((m₀, ρ₀) :: ops).length + 1 < n x ∨ L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true) := by
        rintro (h1 | h1)
        · simp only [List.length_cons] at h1
          omega
        · rw [hl] at h1
          cases h1
      unfold ZkServer.nextD
      exact ite_eq_right hno
  have hfl : ((m₀, ρ₀) :: ops).length = L.last x := by simp only [List.length_cons, hops, hlast]
  have hk : L.kindAt x ((m₀, ρ₀) :: ops).length = .recomputed := by
    unfold kindAt
    rw [ite_eq_right (show ¬((m₀, ρ₀) :: ops).length = 0 by simp), ite_eq_left (Or.inr hfl)]
  have hnext : L.toSrv.nextD L.fillOf n wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = L.ownMsg x w (ν, t) ((m₀, ρ₀) :: ops) := by
    unfold ZkServer.nextD
    rw [ite_eq_left (Or.inr hlive)]
    exact L.frameD_recomp L.fillOf wk x w (ν, t) κ _ hk
  have hrc : L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false := by
    cases hr : L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) with
    | false => rfl
    | true =>
      refine absurd ?_ hstop
      rw [hnext]
      unfold ownMsg
      exact ite_eq_left hr
  have hok : ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true := by
    cases hb : ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 with
    | true => rfl
    | false =>
      rw [hO ν t m₀ ρ₀ ops hops.le hb] at hrc
      cases hrc
  have hown := L.toSrv.ownHist_release L.fillOf n wk x w ask (ν, t) κ (n x - 1) (by omega)
  rw [← hτ, hfe] at hown
  have hrows := hRw ν t κ m₀ ρ₀ ops hops hlive hok hrc hown
  refine (interact_congr ask (L.toSrv.nextD L.fillOf n wk x w (ν, t) κ)
    (fun h => (((((S.withStops Rf L.stop).goP vs).prover w).next x h (ν, t)).eval H).1) (n x) [] fun i hi => ?_).symm
  simp only [List.nil_append]
  rw [← hτ] at hi ⊢
  by_cases hlt : i + 1 < n x
  · have hgl : (histOf (τ.take i)).length = i := by rw [length_histOf_take]; omega
    have hown_i := L.toSrv.ownHist_release L.fillOf n wk x w ask (ν, t) κ i (by omega)
    rw [← hτ] at hown_i
    unfold ZkServer.nextD
    rw [ite_eq_left (Or.inl (by rw [hgl]; exact hlt)), ← hRef (ν, t) κ _ hown_i (by rw [hgl]; exact hlt)]
    exact S.withStops_goP_next_lt Rf L.stop vs H x w ν t _ (by rw [hgl]; omega)
  · have hh : histOf (τ.take i) = (m₀, ρ₀) :: ops := by
      rw [show i = n x - 1 by omega]
      exact hfe
    rw [hh, hnext, S.withStops_goP_next_rows Rf L.stop vs H x w ν t m₀ ρ₀ ops hops hrows, ite_eq_left hok]
    have hRf0 := (hR ν t m₀ ρ₀ ops hops.le hok).symm.trans hrc
    refine (S.swMsgAt_withStops_go Rf L.stop x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2) hRf0
      (Or.inr (by simp only [List.length_map]; omega))).trans ?_
    unfold ownMsg
    rw [hrc]
    simp only [Bool.false_eq_true, ↓reduceIte]
    rw [ite_eq_left hfl]
    exact (hF ν t κ m₀ ρ₀ ops hops hlive hok hrc hown).symm

/-- **The deferred release at the model server is zero knowledge per outcome at its `Finish` verdict, at any
values.** At the session with the frame's stops, a last round `refK x + 2` one past the `Finish` and the `Finish` one
past the last coin, for every worker `wk` and any values `vs` whose unswitched round encoding is causal (`UnCausal`),
the release meets `Claims.ZKOut` at the verdict `goesOnAt` read off it: where it goes on, at the honest prover's
Goldreich–Kahan simulator with `ε = 8·|L|·δ₁ + 24t·|L|·δ₁ + 2·3^{-t}` plus `4t·3^t` times the go prover's collision
finder's success (`release_go`, `withStops_goP_go`); else at the refusing prover's simulator cut at the `Finish` with
`ε = 8·|L|·δ₁ + 8t·|L|·δ₁ + 2·3^{-t}` plus `4t·3^t` times the unswitched prover's (`refDBelow_unP`,
`withStops_unP_zk`). Under `Claims.Hm96Hiding`, the session's Props, the stops reading the coins before their round only,
and the transport's trusts: the `Hello`, the stops, the check, the link and the frame rounds at the unswitched prover,
the `Finish` (`FinalRoundEq`) and the rows where it goes on (`FinishRows`). No premise asks the workers' rows to hold
elsewhere or their switched round encoding to be causal. -/
theorem zk_finish_out (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (wk : L.toSrv.Worker)
    (Rl : Rel Flock.Wit) (A : Msg → Prop) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables) (hV : S.Complete Rl)
    (hD : S.DummyReg) (hc : S.Causal) (hRf : S.StopsCausal Rf) (huc : S.UnCausal vs) (hT : S.TapeOKOn A)
    (hRA : ∀ x w, Rl x w → A x) (hlast : ∀ x, L.last x = refK x + 1) (hR : ∀ x, StopsAt S Rf L H x)
    (hC : ∀ x, CheckIs S L H x) (hH : ∀ x, HelloEq S L x) (hLk : ∀ x w, Rl x w → LinkRoundEqStop S vs L x w)
    (hE : ∀ x w, Rl x w → LiveRoundEqStop S vs L H wk x w) (hEf : ∀ x w, Rl x w → FillRoundEqStop S vs L H wk x w)
    (hF : ∀ x w, Rl x w → FinalRoundEq S vs L H wk x w) (hRw : ∀ x w, Rl x w → FinishRows S vs L H wk x w) :
    ZKOut (fun _ => S.refP.Nonce × S.refP.Rand) (fun _ => L.toSrv.Chk) H (fun x => refK x + 2)
      (fun w _ => L.toSrv.proverD L.fillOf (fun x => refK x + 2) wk w) Rl
      (fun x τ => goesOnAt L.stop (refK x + 1) τ)
      (verdictSim
        (((S.withStops Rf L.stop).refP.withSim fun x _ c => (S.withStops Rf L.stop).coinModel x c).simulatorE t
          (t * 3 ^ t))
        ((S.withStops Rf L.stop).offP.simulatorE t (t * 3 ^ t)) (fun x => refK x + 2) L.stop)
      (fun x w _Θ Vs θ o =>
        if o then 8 * S.nLeaves x * δ₁.toReal + 24 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t +
            4 * t * 3 ^ t * (Fork H (((S.withStops Rf L.stop).goP vs).prover w) Vs θ x).Pr H
          else 8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t +
            4 * t * 3 ^ t * (Fork H (((S.withStops Rf L.stop).unP vs).prover w) Vs θ x).Pr H) :=
  L.toSrv.zk_deferred_out L.fillOf H (fun x => refK x + 2) wk ((S.withStops Rf L.stop).goP vs).prover
    ((S.withStops Rf L.stop).unP vs).prover Rl
    (fun x w hRl ask ρ κ hg => L.release_go S Rf vs H (fun x => refK x + 2) wk x w rfl (hlast x) (hR x) (hC x)
      (hH x) (hLk x w hRl) (hE x w hRl) (hEf x w hRl) (hF x w hRl) (hRw x w hRl) ask ρ κ hg)
    (fun x w hRl => L.refDBelow_unP S Rf vs H (fun x => refK x + 2) wk x w rfl (hlast x) (hR x)
      (openingsRefused_of_checkIs S L H x (hC x)) (hH x) (hLk x w hRl) (hE x w hRl) (hEf x w hRl))
    _ _
    (fun x w _Θ Vs θ => 8 * S.nLeaves x * δ₁.toReal + 24 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t +
      4 * t * 3 ^ t * (Fork H (((S.withStops Rf L.stop).goP vs).prover w) Vs θ x).Pr H)
    (fun x => 8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t)
    (fun _ => 4 * t * 3 ^ t)
    (fun x w hRl Θ Vs θ hk Ev => S.withStops_goP_go Rf L.stop vs H Rl A t ht hT1 hδ₁ hS hV hD hc hRf huc hT hRA
      x w hRl Θ Vs θ hk Ev)
    (S.withStops_unP_zk Rf L.stop vs H Rl A t ht hT1 hδ₁ hS hV hD hc hRf huc hT hRA)

end LeanRelease

end Claims.ZKServer
