module

public import Proofs.Flock.ZkAny.Deferred

/-!
# The unswitched prover is zero knowledge, at any values

The unswitched prover (`RefSession.unP`) sends, before the last coin, the round encoding of the unswitched cut (the
workers' values on the drawn pads, rows or not), and after it the refusal `off`. Its zero knowledge needs no row of any
table to hold: it never opens a hidden leaf, so Lemma A (`card_unCutAt_ideal`) and the hm96 leaves' hiding bound every
message it sends against a dummy that refuses at every coin vector.

* `TPreModel.UnLink`, `TPreModel.runs_close_un`; `TOffModel.UnLink`, `TOffModel.close_un`: the switched prover's
  `SwLink`, `runs_close_sw`, `close_sw` at the unswitched one, within `2·|L|·δ₁`, whatever values the workers supply.
* `RefSession.offMsgAt`, `RefSession.offP`: the refusing reference protocol, `refP` whose dummy refuses at every coin
  vector and whose SHVZK simulator at a coin vector is that dummy's messages there. It reads no worker's values.
* `RefSession.unPS`: `offP` with `unP vs`'s messages; its prover is `unP vs`'s and its simulator `offP`'s
  (`unPS_prover`, `unPS_simulatorE`). `unPS_prefixHides` and `unPS_shvzk` are (HA) and (HB) at it.
* **`RefSession.unP_zk`**: `ZKmalR` of `unP vs` at `offP`'s Goldreich–Kahan simulator, at
  `ε_zk = 8·|L|·δ₁ + 8t·|L|·δ₁ + 2·3^{-t}` and `γ = 4t·3^t`, for any values `vs` whose round encoding is causal, under
  `Claims.Hm96Hiding` and the session's Props; one simulator serves every `vs`. `RefSession.withStops_unP_zk` is it at
  the session with the frame's stops.
-/

@[expose] public section

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Proto

/-- The unswitched prover's messages before the last coin at the answers `a`, on table witnesses `wt` and the workers'
values `vs a`, bases `ω` and pre-final leaves `lv`. -/
noncomputable def TPreModel.unMsgs {W : Type} {G : Proto W} {x : Msg} (M : G.TPreModel x)
    (wt : ∀ j, FlockSoundness.Model.Witness (M.C j).S.m) (vs : ∀ a j, ((M.C j).T (M.cz a j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (M.C j).z (M.C j).Hi) (a : Fin (G.K x + 1) → Out)
    (lv : ZkView.Leaves (fun j => (M.C j).z) → ByteArray) : Fin (G.K x + 1) → Msg :=
  M.out a (unCutAt M.C (M.cz a) wt (vs a) ω lv)

/-- **The unswitched prover's round encoding reads the answers before each round only**, whatever each leaf's pair
`(M·y, c(y))`, as `TPreModel.SwCausal` has the switched prover's. -/
def TPreModel.UnCausal {W : Type} {G : Proto W} {x : Msg} (M : G.TPreModel x)
    (wt : ∀ j, FlockSoundness.Model.Witness (M.C j).S.m) (vs : ∀ a j, ((M.C j).T (M.cz a j)).tab.Vals) : Prop :=
  ∀ (ω : ∀ j, ZkView.Base (M.C j).z (M.C j).Hi) (p : ZkView.Leaves (fun j => (M.C j).z) → ZkView.Bits × ByteArray)
    (a a' : Fin (G.K x + 1) → Out) (i : Fin (G.K x + 1)), (∀ j, j < i → a j = a' j) →
    M.unMsgs wt vs ω a (fun l => ZkView.leafH (unDigOf M.C (M.cz a) wt (vs a) ω l + (p l).1, (p l).2)) i =
      M.unMsgs wt vs ω a' (fun l => ZkView.leafH (unDigOf M.C (M.cz a') wt (vs a') ω l + (p l).1, (p l).2)) i

/-- **Before the last coin, the prover's messages on `w` are the unswitched prover's at values the workers supply**
(`vs`), its round encoding causal, at the executable's leaves; on the dummy they are the honest prover's
(`TPreModel.SwLink` at the unswitched prover). -/
def TPreModel.UnLink {W : Type} {G : Proto W} {x : Msg} (M : G.TPreModel x) (H : Msg → Msg) (ν : G.Nonce) (w : W) :
    Prop :=
  (∀ j, (M.C j).Valid (M.wt w j)) ∧
    (∀ a j, ((M.C j).T (M.cz a j)).tab.sReg (M.wt₀ j) = ((M.C j).T (M.cz a j)).tab.pubReg) ∧ M.Causal ∧
    (∀ p, Nat.card {r // M.eR r = p} = M.kR) ∧
    (∃ vs : ∀ a j, ((M.C j).T (M.cz a j)).tab.Vals, M.UnCausal (M.wt w) vs ∧
      ∀ r a, (G.preStrat H x ν (G.msg x w ν)).msg r a =
        M.unMsgs (M.wt w) vs (M.eR r).1 a (unExeOf M.C (M.cz a) (M.wt w) (vs a) (M.eR r).1 (M.eR r).2)) ∧
    ∀ r a, (G.preStrat H x ν (G.dmsg x ν)).msg r a =
      M.msgs M.wt₀ (M.eR r).1 a (exeOf M.C (M.cz a) M.wt₀ (M.eR r).1 (M.eR r).2)

/-- **Lemma C's bound on the runs before the last coin at the unswitched prover** (`runs_close_of_ideal`, its two table
witnesses the unswitched prover on `w` and the honest one on the dummy), within `2·|L|·δ₁`, whatever values the workers
supply. -/
theorem TPreModel.runs_close_un {W : Type} {G : Proto W} {x : Msg} (M : G.TPreModel x) (H : Msg → Msg)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce) (w : W)
    (hL : M.UnLink H ν w) (hOK : M.OK) (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1))
    (E : (Fin (G.K x + 1) → Msg) → Prop) :
    |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
      Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))|
      ≤ 2 * M.size * δ₁.toReal := by
  obtain ⟨hv, hreg₀, hc, heR, ⟨vs, hsc, hlink⟩, hlink₀⟩ := hL
  let msgs : Bool → (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) → (Fin (G.K x + 1) → Out) →
      (ZkView.Leaves (fun j => (M.C j).z) → ByteArray) → Fin (G.K x + 1) → Msg := fun b ω a lv =>
    cond b (M.unMsgs (M.wt w) vs ω a lv) (M.msgs M.wt₀ ω a lv)
  let dig : Bool → (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) → (Fin (G.K x + 1) → Out) →
      ZkView.Leaves (fun j => (M.C j).z) → ZkView.Bits := fun b ω a =>
    cond b (unDigOf M.C (M.cz a) (M.wt w) (vs a) ω) (digOf M.C (M.cz a) M.wt₀ ω)
  let P : Bool → FlockSoundness.ZK.Strategy
      ((∀ j, ZkView.Base (M.C j).z (M.C j).Hi) × (ZkView.Leaves (fun j => (M.C j).z) → ZkView.Bits × ByteArray))
      Msg Out (G.K x + 1) := fun b =>
    { msg := fun q a => msgs b q.1 a fun l => ZkView.leafH (dig b q.1 a l + (q.2 l).1, (q.2 l).2)
      causal := fun q a a' i h => by
        cases b
        · exact hc M.wt₀ q.1 q.2 a a' i h
        · exact hsc q.1 q.2 a a' i h }
  have hP1 : ∀ ω p a, (P true).msg (ω, p) a = M.unMsgs (M.wt w) vs ω a
      fun l => ZkView.leafH (unDigOf M.C (M.cz a) (M.wt w) (vs a) ω l + (p l).1, (p l).2) := fun _ _ _ => rfl
  have hP0 : ∀ ω p a, (P false).msg (ω, p) a = M.msgs M.wt₀ ω a
      fun l => ZkView.leafH (digOf M.C (M.cz a) M.wt₀ ω l + (p l).1, (p l).2) := fun _ _ _ => rfl
  have hl : ∀ r, (G.preStrat H x ν (G.msg x w ν)).msg r =
      (P true).msg ((M.eR r).1, fun l => (ZkView.My ((M.eR r).2 l), ZkView.saltHash ((M.eR r).2 l))) :=
    fun r => funext fun a => by rw [hP1, hlink r a, unExeOf_eq]
  have hl₀ : ∀ r, (G.preStrat H x ν (G.dmsg x ν)).msg r =
      (P false).msg ((M.eR r).1, fun l => (ZkView.My ((M.eR r).2 l), ZkView.saltHash ((M.eR r).2 l))) :=
    fun r => funext fun a => by rw [hP0, hlink₀ r a, exeOf_eq]
  have : Nonempty (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) := ⟨fun _ => 0⟩
  exact G.runs_close_of_ideal H x ν w ZkView.My ZkView.saltHash hT1 hδ₁ ZkView.leafH msgs dig P
    (fun _ _ _ _ => rfl) true false
    (fun a σ => card_unCutAt_ideal M.C (M.cz a) (M.wt w) M.wt₀ (vs a) (fun j => (hOK j).ok.2 _ _ (hv j)) (hreg₀ a)
      (M.out a) σ)
    M.eR M.kR heR hl hl₀ V E

/-- **At a coin vector, the prover's messages on `w` are `out` of the unswitched prover's cut view at values the
workers supply**, at the executable's leaves; on the dummy they are `out` of the honest prover's (`TOffModel.SwLink`
at the unswitched prover). -/
def TOffModel.UnLink {W : Type} {G : Proto W} {x : Msg} (M : G.TOffModel x) (H : Msg → Msg) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (w : W) : Prop :=
  (∀ j, (M.C j).Valid (M.wt w j)) ∧
    (∀ j, ((M.C j).T (M.cz j)).tab.sReg (M.wt₀ j) = ((M.C j).T (M.cz j)).tab.pubReg) ∧
    (∀ p, Nat.card {r // M.eR r = p} = M.kR) ∧
    (∃ vs : ∀ j, ((M.C j).T (M.cz j)).tab.Vals, ∀ r, G.realVec H x w ν r c =
      M.out (unCutAt M.C M.cz (M.wt w) vs (M.eR r).1 (unExeOf M.C M.cz (M.wt w) vs (M.eR r).1 (M.eR r).2))) ∧
    ∀ r, G.dVec H x ν r c = M.out (cutAt M.C M.cz M.wt₀ (M.eR r).1 (exeOf M.C M.cz M.wt₀ (M.eR r).1 (M.eR r).2))

/-- **Lemma C at a coin vector, at the unswitched prover** (`refused_close_of_ideal`, its two table witnesses the
unswitched prover on `w` and the honest one on the dummy): within `2·|L|·δ₁`, whatever values the workers supply. -/
theorem TOffModel.close_un {W : Type} {G : Proto W} {x : Msg} (M : G.TOffModel x) (H : Msg → Msg) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    (w : W) (hL : M.UnLink H ν c w) (hOK : M.OK) (E : List Msg → Prop) :
    |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun r : G.Rand => E (G.dVec H x ν r c))|
      ≤ 2 * M.size * δ₁.toReal := by
  obtain ⟨hv, hreg₀, heR, ⟨vs, hR⟩, hD⟩ := hL
  have : Nonempty (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) := ⟨fun _ => 0⟩
  exact G.refused_close_of_ideal H x w ν c ZkView.My ZkView.saltHash hT1 hδ₁ ZkView.leafH
    (fun b ω lv => cond b (M.out (unCutAt M.C M.cz (M.wt w) vs ω lv)) (M.out (cutAt M.C M.cz M.wt₀ ω lv)))
    (fun b ω => cond b (unDigOf M.C M.cz (M.wt w) vs ω) (digOf M.C M.cz M.wt₀ ω)) true false
    (fun σ => card_unCutAt_ideal M.C M.cz (M.wt w) M.wt₀ vs (fun j => (hOK j).ok.2 _ _ (hv j)) hreg₀ M.out σ)
    M.eR M.kR heR
    (fun r => (hR r).trans (congrArg (fun lv => M.out (unCutAt M.C M.cz (M.wt w) vs (M.eR r).1 lv))
      (unExeOf_eq M.C M.cz (M.wt w) vs (M.eR r).1 (M.eR r).2)))
    (fun r => (hD r).trans (congrArg (fun lv => M.out (cutAt M.C M.cz M.wt₀ (M.eR r).1 lv))
      (exeOf_eq M.C M.cz M.wt₀ (M.eR r).1 (M.eR r).2))) E

end Proto

namespace RefSession

/-- **The refusing prover's message after the coins `cs`**, on table witnesses `wt` and tape `t`: before the last coin
`msgAt`'s round encoding of the cut view, after it `off` of the cut view, at every coin vector. -/
noncomputable def offMsgAt (S : RefSession) (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (t : S.Tape) (cs : List S.Coin) : Msg :=
  if cs.length < refK x then S.enc x (S.pad cs) cs.length (S.cutOf x (S.pad cs) wt t)
  else S.off x (S.pad cs) (S.cutOf x (S.pad cs) wt t)

/-- **The refusing reference protocol**: `refP` whose dummy refuses at every coin vector (`offMsgAt` on the dummy's
table witnesses) and whose SHVZK simulator at a coin vector is that dummy's messages there, on a tape of its own. -/
noncomputable def offP (S : RefSession) : Proto Flock.Wit :=
  { S.refP with
    dmsg := fun x _ t cs => .ret (S.offMsgAt x (S.wt₀ x) t cs)
    SimRand := fun _ => S.Tape
    finSim := fun _ => S.finTape
    neSim := fun _ => S.neTape
    sim := fun x _ c t => .ret ((List.range (refK x + 1)).map fun j => S.offMsgAt x (S.wt₀ x) t ((List.ofFn c).take j)) }

noncomputable instance instFintypeOffPCoin (S : RefSession) : Fintype S.offP.Coin := S.finCoin

/-- **The unswitched prover against the refusing protocol's simulator**: `offP` with its messages on each witness
`unP vs`'s. -/
noncomputable def unPS (S : RefSession) (vs : S.SwVals) : Proto Flock.Wit :=
  { S.offP with msg := (S.unP vs).msg }

noncomputable instance instFintypeUnPSCoin (S : RefSession) (vs : S.SwVals) : Fintype (S.unPS vs).Coin := S.finCoin

theorem unPS_prover (S : RefSession) (vs : S.SwVals) : (S.unPS vs).prover = (S.unP vs).prover := rfl

theorem unPS_simulatorE (S : RefSession) (vs : S.SwVals) (t M : ℕ) :
    (S.unPS vs).simulatorE t M = S.offP.simulatorE t M :=
  Proto.simulatorE_msg _ _ t M

/-- **Before the last coin, the unswitched prover sends the unswitched round encoding at the coins the answers open**
(`UnCausal`). -/
theorem unPre_msg (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit) (ν : S.Nonce)
    (hc : S.UnCausal vs) (t : S.Tape) (a : Fin (refK x + 1) → Out) (i : Fin (refK x + 1)) :
    S.refP.pmG H x ν (fun cs => .ret (S.unMsgAt x (S.wt x w) (vs x w ν) t cs)) (ansHist a i) =
      S.refP.pmG H x ν (fun _ => .ret (S.enc x (S.coinsOf a) (i.val - 1)
        (S.unCutOf x (S.coinsOf a) (S.wt x w) (vs x w ν (S.coinsOf a)) t))) (ansHist a i) := by
  unfold Proto.pmG
  congr 2
  refine strat_congr_gen _ _ _ _ _ _ _ fun hne => ?_
  have hi : 0 < i.val := by
    rcases Nat.eq_zero_or_pos i.val with h0 | h0
    · exact absurd (by simp [ansHist, h0]) hne
    · exact h0
  have hl : ((ansHist a i).tail.map fun p => S.decOf p.2).length = i.val - 1 := by
    simp only [List.length_map, List.length_tail, ansHist, List.length_ofFn]
  show OC.ret (S.unMsgAt x (S.wt x w) (vs x w ν) t ((ansHist a i).tail.map fun p => S.decOf p.2)) = OC.ret _
  unfold unMsgAt
  rw [hl, ite_eq_left (by omega),
    S.enc_unCutOf vs x w ν hc t _ (S.coinsOf a) (i.val - 1) fun k hk => S.pad_ansHist a i k hk]

/-- **Before the last coin, the refusing prover sends the round encoding at the coins the answers open** (`Causal`). -/
theorem offPre_msg (S : RefSession) (H : Msg → Msg) (x : Msg) (ν : S.Nonce) (hc : S.Causal)
    (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m) (t : S.Tape) (a : Fin (refK x + 1) → Out)
    (i : Fin (refK x + 1)) :
    S.refP.pmG H x ν (fun cs => .ret (S.offMsgAt x wt t cs)) (ansHist a i) =
      S.refP.pmG H x ν (fun _ => .ret (S.enc x (S.coinsOf a) (i.val - 1) (S.cutOf x (S.coinsOf a) wt t)))
        (ansHist a i) := by
  unfold Proto.pmG
  congr 2
  refine strat_congr_gen _ _ _ _ _ _ _ fun hne => ?_
  have hi : 0 < i.val := by
    rcases Nat.eq_zero_or_pos i.val with h0 | h0
    · exact absurd (by simp [ansHist, h0]) hne
    · exact h0
  have hl : ((ansHist a i).tail.map fun p => S.decOf p.2).length = i.val - 1 := by
    simp only [List.length_map, List.length_tail, ansHist, List.length_ofFn]
  show OC.ret (S.offMsgAt x wt t ((ansHist a i).tail.map fun p => S.decOf p.2)) = OC.ret _
  unfold offMsgAt
  rw [hl, ite_eq_left (by omega), S.enc_cutOf x hc wt t _ (S.coinsOf a) (i.val - 1) fun k hk => S.pad_ansHist a i k hk]

/-- **The unswitched prover's messages at the prefixes of a coin vector** (`UnCausal`): round `j`'s unswitched encoding
at `c` before the last coin, then `off` of the unswitched cut at `c`. -/
theorem unMsgAt_take (S : RefSession) (vs : S.SwVals) (x : Msg) (w : Flock.Wit) (ν : S.Nonce) (hc : S.UnCausal vs)
    (t : S.Tape) (c : Fin (refK x) → S.Coin) (j : ℕ) (hj : j < refK x + 1) :
    S.unMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take j) =
      if j < refK x then S.enc x c j (S.unCutOf x c (S.wt x w) (vs x w ν c) t)
      else S.off x c (S.unCutOf x c (S.wt x w) (vs x w ν c) t) := by
  have hl : ((List.ofFn c).take j).length = j := by simp only [List.length_take, List.length_ofFn]; omega
  unfold unMsgAt
  rw [hl]
  by_cases h : j < refK x
  · rw [ite_eq_left h, ite_eq_left h]
    exact S.enc_unCutOf vs x w ν hc t _ c j fun k hk => S.pad_take c j k hk
  · rw [ite_eq_right h, ite_eq_right h]
    obtain rfl : j = refK x := by omega
    rw [S.pad_ofFn c]

/-- **The refusing prover's messages at the prefixes of a coin vector** (`Causal`). -/
theorem offMsgAt_take (S : RefSession) (x : Msg) (hc : S.Causal) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (t : S.Tape) (c : Fin (refK x) → S.Coin) (j : ℕ) (hj : j < refK x + 1) :
    S.offMsgAt x wt t ((List.ofFn c).take j) =
      if j < refK x then S.enc x c j (S.cutOf x c wt t) else S.off x c (S.cutOf x c wt t) := by
  have hl : ((List.ofFn c).take j).length = j := by simp only [List.length_take, List.length_ofFn]; omega
  unfold offMsgAt
  rw [hl]
  by_cases h : j < refK x
  · rw [ite_eq_left h, ite_eq_left h]
    exact S.enc_cutOf x hc wt t _ c j fun k hk => S.pad_take c j k hk
  · rw [ite_eq_right h, ite_eq_right h]
    obtain rfl : j = refK x := by omega
    rw [S.pad_ofFn c]

/-- `preModel` at `unPS`. -/
noncomputable def unPreModel (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (ν : S.Nonce) :
    (S.unPS vs).TPreModel x where
  J := S.J x
  C := S.C x
  cz a := S.cz x (S.coinsOf a)
  out a σ i := S.refP.pmG H x ν (fun _ => .ret (S.enc x (S.coinsOf a) (i.val - 1) σ)) (ansHist a i)
  wt := S.wt x
  wt₀ := S.wt₀ x
  eR := S.eP x
  kR := S.kP x

theorem unPreModel_link (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop)
    (hV : S.Complete R) (hD : S.DummyReg) (hc : S.Causal) (huc : S.UnCausal vs) (hT : S.TapeOKOn A) (x : Msg)
    (hx : A x) (w : Flock.Wit) (hR : R x w) (ν : S.Nonce) : (S.unPreModel vs H x ν).UnLink H ν w := by
  refine ⟨hV x w hR, fun a j => hD x (S.coinsOf a) j, ?_, fun p => (hT x hx).1 p _,
    ⟨fun a => vs x w ν (S.coinsOf a), ?_, fun t a => funext fun i => S.unPre_msg vs H x w ν huc t a i⟩,
    fun t a => funext fun i => S.offPre_msg H x ν hc (S.wt₀ x) t a i⟩
  · intro wt ω p a a' i h
    show S.refP.pmG H x ν _ (ansHist a i) = S.refP.pmG H x ν _ (ansHist a' i)
    rw [ansHist_congr a a' i h]
    refine congrArg (fun g => S.refP.pmG H x ν g (ansHist a' i)) (funext fun _ => congrArg OC.ret ?_)
    exact hc x wt ω p (S.coinsOf a) (S.coinsOf a') (i.val - 1) fun k hk =>
      congrArg (fun o => S.decOf (msgOf o)) (h k.succ (Fin.lt_def.mpr (by simp only [Fin.val_succ]; omega)))
  · intro ω p a a' i h
    show S.refP.pmG H x ν _ (ansHist a i) = S.refP.pmG H x ν _ (ansHist a' i)
    rw [ansHist_congr a a' i h]
    refine congrArg (fun g => S.refP.pmG H x ν g (ansHist a' i)) (funext fun _ => congrArg OC.ret ?_)
    exact huc x w ν ω p (S.coinsOf a) (S.coinsOf a') (i.val - 1) fun k hk =>
      congrArg (fun o => S.decOf (msgOf o)) (h k.succ (Fin.lt_def.mpr (by simp only [Fin.val_succ]; omega)))

/-- The model at a coin vector at `unPS`: the cut view with `enc` and `off`, at every coin vector. -/
noncomputable def unOffModel (S : RefSession) (vs : S.SwVals) (x : Msg) (c : Fin (refK x) → S.Coin) :
    (S.unPS vs).TOffModel x where
  J := S.J x
  C := S.C x
  cz := S.cz x c
  out σ := (List.range (refK x + 1)).map fun j => if j < refK x then S.enc x c j σ else S.off x c σ
  wt := S.wt x
  wt₀ := S.wt₀ x
  eR := S.eP x
  kR := S.kP x

theorem unOffModel_link (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop)
    (hV : S.Complete R) (hD : S.DummyReg) (hc : S.Causal) (huc : S.UnCausal vs) (hT : S.TapeOKOn A) (x : Msg)
    (hx : A x) (w : Flock.Wit) (hR : R x w) (ν : S.Nonce) (c : Fin (refK x) → S.Coin) :
    (S.unOffModel vs x c).UnLink H ν c w := by
  refine ⟨hV x w hR, fun j => hD x c j, fun p => (hT x hx).1 p _, ⟨vs x w ν c, fun (t : S.Tape) => ?_⟩,
    fun (t : S.Tape) => ?_⟩
  · show (List.range (refK x + 1)).map (fun j => S.unMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take j)) =
      (List.range (refK x + 1)).map fun j =>
        if j < refK x then S.enc x c j (S.unCutOf x c (S.wt x w) (vs x w ν c) t)
        else S.off x c (S.unCutOf x c (S.wt x w) (vs x w ν c) t)
    exact List.map_congr_left fun j hj => S.unMsgAt_take vs x w ν huc t c j (List.mem_range.1 hj)
  · show (List.range (refK x + 1)).map (fun j => S.offMsgAt x (S.wt₀ x) t ((List.ofFn c).take j)) =
      (List.range (refK x + 1)).map fun j =>
        if j < refK x then S.enc x c j (S.cutOf x c (S.wt₀ x) t) else S.off x c (S.cutOf x c (S.wt₀ x) t)
    exact List.map_congr_left fun j hj => S.offMsgAt_take x hc _ t c j (List.mem_range.1 hj)

/-- **(HA) at `unPS`**: before the last coin, the unswitched prover's runs and the dummy's are within `2·|L|·δ₁`
(`TPreModel.runs_close_un`), on a relation whose statements are in `A`. -/
theorem unPS_prefixHides (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hV : S.Complete R)
    (hD : S.DummyReg) (hc : S.Causal) (huc : S.UnCausal vs) (hT : S.TapeOKOn A) (hRA : ∀ x w, R x w → A x)
    (hS : S.Tables) : (S.unPS vs).PrefixHides H R fun x => 2 * S.nLeaves x * δ₁.toReal :=
  (S.unPS vs).prefixHides_of_runs H R _ fun x w hR ν V E =>
    (S.unPreModel vs H x ν).runs_close_un H hT1 hδ₁ ν w
      (S.unPreModel_link vs H R A hV hD hc huc hT x (hRA x w hR) w hR ν) (hS x) V E

/-- **(HB) at `unPS`**: at every coin vector, the unswitched prover's messages and the refusing dummy's are within
`2·|L|·δ₁` (`TOffModel.close_un`), on a relation whose statements are in `A`. -/
theorem unPS_shvzk (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hV : S.Complete R)
    (hD : S.DummyReg) (hc : S.Causal) (huc : S.UnCausal vs) (hT : S.TapeOKOn A) (hRA : ∀ x w, R x w → A x)
    (hS : S.Tables) : (S.unPS vs).SHVZK H R fun x => 2 * S.nLeaves x * δ₁.toReal := fun x w hR ν c E =>
  (S.unOffModel vs x c).close_un H hT1 hδ₁ ν c w
    (S.unOffModel_link vs H R A hV hD hc huc hT x (hRA x w hR) w hR ν c) (hS x) E

/-- **The unswitched prover is zero knowledge, at any values**: `ZKmalR` of `unP vs` at the refusing protocol's
Goldreich–Kahan simulator (`offP`'s, which reads no worker's values), at `ε_zk = 8·|L|·δ₁ + 8t·|L|·δ₁ + 2·3^{-t}`
and `γ = 4t·3^t`, for any values `vs` whose round encoding is causal (`UnCausal`), under `Claims.Hm96Hiding` and the
session's Props, on a relation whose statements are in `A` (`gkE_zk_pow` at `unPS`). -/
theorem unP_zk (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop) (t : ℕ)
    (ht : 0 < t) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (hS : S.Tables) (hV : S.Complete R) (hD : S.DummyReg) (hc : S.Causal) (huc : S.UnCausal vs)
    (hT : S.TapeOKOn A) (hRA : ∀ x w, R x w → A x) :
    ZKmalR H (fun x => refK x + 2) (S.unP vs).prover R (S.offP.simulatorE t (t * 3 ^ t))
      (fun x => 8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t := by
  have h : ZKmalR H (fun x => refK x + 2) (S.unP vs).prover R (S.offP.simulatorE t (t * 3 ^ t))
      (fun x => 4 * (2 * S.nLeaves x * δ₁.toReal) + 4 * t * (2 * S.nLeaves x * δ₁.toReal) + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t := by
    have h0 := (S.unPS vs).gkE_zk_pow H R t ht _ _
      (S.unPS_prefixHides vs H R A hT1 hδ₁ hV hD hc huc hT hRA hS)
      (S.unPS_shvzk vs H R A hT1 hδ₁ hV hD hc huc hT hRA hS) (fun x => by positivity) (S.binding H)
    rw [S.unPS_simulatorE vs] at h0
    exact h0
  intro x w hR Θ Vs θ hk Ev
  refine (h x w hR Θ Vs θ hk Ev).trans (le_of_eq ?_)
  beta_reduce
  ring

/-- **`unP_zk` at the session with the frame's stops**: the stop branch's prover (`RefDBelow` holds at it,
`LeanRelease.refDBelow_unP`) is zero knowledge at `withStops`' refusing simulator, for any values `vs` whose round
encoding is causal, where the stops read the coins before their round only (`StopsCausal`). -/
theorem withStops_unP_zk (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables) (hV : S.Complete R)
    (hD : S.DummyReg) (hc : S.Causal) (hRf : S.StopsCausal Rf) (huc : S.UnCausal vs) (hT : S.TapeOKOn A)
    (hRA : ∀ x w, R x w → A x) :
    ZKmalR H (fun x => refK x + 2) ((S.withStops Rf stop).unP vs).prover R
      ((S.withStops Rf stop).offP.simulatorE t (t * 3 ^ t))
      (fun x => 8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t :=
  (S.withStops Rf stop).unP_zk vs H R A t ht hT1 hδ₁ (S.withStops_tables Rf stop hS) (S.withStops_complete Rf stop hV)
    (S.withStops_dummyReg Rf stop hD) (S.withStops_causal Rf stop hRf hc) (S.withStops_unCausal Rf stop vs hRf huc) hT
    hRA

end RefSession

end Claims.ZKMal
