import Proofs.Flock.Soundness.Discharge.ZkHidden.View
import Proofs.Flock.Soundness.Discharge.ZkEveryCoin.Headline
import Proofs.Flock.Soundness.Game.Prob

/-!
# V*'s outer sessions: the honest prover's self-check, and the coins it refuses

V*'s outer sessions are `--zk` sessions of `J` tables, and their honest prover stops when its own batched constraint
fails (`zk_veil.rs`'s `prove_inner` under `SelfCheck::Stop`: `⟨c, h⟩ + t ≠ 0` after the batching coins, before `τ`'s
commitment). The stop reads the witness, so the simulator could not reproduce it unless it never fires. `ZeroKnowledgeHidden`'s
real view never stops; this module shows the honest prover's transmission with the stop in it is that view.

* `Table.Stops`: rep `r`'s self-check fails, `⟨λ, h⟩ ≠ θ` (in characteristic 2 the same as `⟨c, h⟩ + t ≠ 0`), and
  `Table.not_stops`: never, under `InnerHolds`. `Table.StopsAt`: the same at one rep. `Table.viewWith_pre`,
  `Coined.viewExe_pre`: the check reads only what the executable prover sent before `τ`.
* `sendsJ`: the session's honest prover at `J` tables: what a stopped run sends (`stopped`, any function of the prover's
  randomness) if some table's self-check stops, else the joint view.
* `stopsJ_never`, `prCoin_stopsJ`, `sendsJ_honest`: at `Admits` coins, valid witnesses and public registered values,
  `CoinedOK`'s inner relation (`InnerHolds`) makes the stop bit constantly false, so `sendsJ` is the joint view.
* `ZkViewAtJS`, `zkViewAtJS_of`, `zk_session_viewHJS`: `ZkViewAtJ` with the real side the honest prover's transmission,
  at the same bound, and at `zk_session_viewHJ`'s plan under its hypotheses. The abort costs nothing.

The prover's other stops (`β = 0`, dependent `ρ₀, ρ₁`, the mask rank, a zerocheck outer coin `1`) read only opened coins:
they are `CoinedSetup.Admits`, which `ZkViewAtJ` assumes. The rest of the module drops that assumption:

* `ZkEveryCoin.coin_close_at`: `ZkEveryCoin.coin_close` with the registered values public at its one coin vector: at a
  table whose coins the prover refuses, the prefix it sent against the honest prefix on the dummy witness
  (`table_abort_hm96`), and Lemma B elsewhere.
* `sendsAllJ`, `SelfCheckAtReps`, `stopsAllJ_never`, `sendsAllJ_honest`: the session's honest prover at any coin
  vector: `stopped` if the self-check stops at a rep whose outer coins aren't `1` (never, by `SelfCheckAtReps`:
  `Composed.innerOf_holds_rep_of`), else each table's output (`ZkEveryCoin.outExe`: its view, or its cut where it
  refuses the table's coins).
* `ZkViewAllJ`, `zkViewAllJ_of_ok`, `zk_session_view_allHJ`: at every coin vector, the honest prover's transmission
  against each table's simulator (`ZkEveryCoin.simAll`, which refuses where the prover does) within
  `Σ_j 2·|LeafPos_j|·δ₁`, at `zk_session_viewHJ`'s plan under its hypotheses. There is no hypothesis on the coins and no
  probability charge.
-/

namespace FlockSoundness.ZK.Table

section Stops

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL) [CharP F 2]

/-- **The honest prover's self-check stops** (`zk_veil.rs`'s `SelfCheck::Stop`): at some rep `r`, its pads miss the
inner proof's batched constraint, `⟨λ, h⟩ ≠ θ`, `λ` and `θ` the verifier's replay on the messages before it. -/
def Stops (w : W) (ω : T.Rand) : Prop :=
  ∃ r, ∑ i, T.lam r (T.view w ω).pre i * T.hFull (ω.2.2.1 r) i ≠ T.theta r (T.view w ω).pre

/-- Under `InnerHolds` the self-check never stops. -/
theorem not_stops {w : W} (h : T.InnerHolds w) (ω : T.Rand) : ¬ T.Stops w ω := fun ⟨r, hr⟩ => hr (h ω r)

/-- **The self-check stops at rep `r`**: `Stops` at one rep. -/
def StopsAt (w : W) (ω : T.Rand) (r : Fin 2) : Prop :=
  ∑ i, T.lam r (T.view w ω).pre i * T.hFull (ω.2.2.1 r) i ≠ T.theta r (T.view w ω).pre

end Stops

section Leaves

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt (B × Salt) D Hi VL) [CharP F 2]

/-- `viewWith` changes only the hidden leaves: its messages before `τ` are `view`'s. -/
theorem viewWith_pre (w : W) (ω : T.Rand) (L : T.Hid → D) : (T.viewWith w ω L).pre = (T.view w ω).pre := rfl

end Leaves

end FlockSoundness.ZK.Table

namespace FlockSoundness.Discharge.ZkView

variable {K W Hi VL : Type} [Field K] {z : Sizes}

/-- The executable prover's messages before `τ` are `Table.view`'s, which `Table.Stops` reads. -/
theorem Coined.viewExe_pre (T : Coined K W Hi VL z) (w : W) (ω : T.tab.Rand) (ys : T.tab.Hid → Salt) :
    (T.viewExe w ω ys).pre = (T.tab.view w ω).pre := by
  unfold Coined.viewExe
  exact FlockSoundness.ZK.Table.viewWith_pre _ _ _ _

end FlockSoundness.Discharge.ZkView

namespace FlockSoundness.Discharge.ZkEveryCoin

open Game Audit Exec Refine Flock FlockLevel3 Model ZkSession ZkLink ZkExec DecodeZL Composed ZkView
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

open Classical in
/-- **One table at one coin vector** (`coin_close`, the registered values public at `c` alone): the prover's output and
the simulator's are within `2·|LeafPos|·2^-193`, both ways. -/
theorem coin_close_at (C : CoinedSetup) (hC : CoinedOK execArith (Refine.fOf Flock.Zk.PAD_C) C)
    (hKey : Assumptions.Zk.HashDerivedKeyHm96) (c : CoinsZK C.Rest) (w w0 : Witness C.S.m) (hv : C.Valid w)
    (hreg : (C.T c).tab.sReg w = (C.T c).tab.pubReg) (E : (C.T c).tab.View ⊕ (C.T c).tab.Cut → Prop) :
    prCoin (fun p : (C.T c).tab.Rand × ((C.T c).tab.Hid → ZkView.Salt) => E (outExe C c w p.1 p.2))
        ≤ prCoin (fun p : (C.T c).tab.SimRand × ((C.T c).tab.Hid → ZkView.Salt) => E (simAll C c w0 p.1 p.2))
          + 2 * Fintype.card (LeafPos C.z) * (2 : ℝ≥0∞) ^ (-193 : ℤ)
      ∧ prCoin (fun p : (C.T c).tab.SimRand × ((C.T c).tab.Hid → ZkView.Salt) => E (simAll C c w0 p.1 p.2))
        ≤ prCoin (fun p : (C.T c).tab.Rand × ((C.T c).tab.Hid → ZkView.Salt) => E (outExe C c w p.1 p.2))
          + 2 * Fintype.card (LeafPos C.z) * (2 : ℝ≥0∞) ^ (-193 : ℤ) := by
  by_cases hA : C.Admits c
  · simp only [outExe, simAll, hA, ite_true]
    obtain ⟨h1, h2⟩ := zkViewAtReg_of_ok C hC hKey c hA
      (fun v => (Sum.inl (v 0) : (C.T c).tab.View ⊕ (C.T c).tab.Cut)) (fun _ => w) (fun _ => w0) (fun _ => hv)
      (fun _ => hreg) E
    have eR := ZK.prCoin_congr_equiv (Equiv.funUnique (Fin 1) ((C.T c).tab.Rand × ((C.T c).tab.Hid → ZkView.Salt)))
      (fun p => E (.inl ((C.T c).viewExe w p.1 p.2)))
    have eS := ZK.prCoin_congr_equiv (Equiv.funUnique (Fin 1) ((C.T c).tab.SimRand × ((C.T c).tab.Hid → ZkView.Salt)))
      (fun p => E (.inl ((C.T c).simExe w0 p.1 p.2)))
    have hs : ∑ _j : Fin 1, 2 * (Fintype.card (C.T c).tab.Hid : ℝ≥0∞) * (2 : ℝ≥0∞) ^ (-193 : ℤ)
        ≤ 2 * Fintype.card (LeafPos C.z) * (2 : ℝ≥0∞) ^ (-193 : ℤ) := by
      rw [Fin.sum_univ_one]
      gcongr
      exact_mod_cast card_hid_le C c
    exact ⟨(eR.symm.trans_le h1).trans (add_le_add (le_of_eq eS) hs),
      (eS.symm.trans_le h2).trans (add_le_add (le_of_eq eR) hs)⟩
  · simp only [outExe, simAll, hA, ite_false, ZkView.Coined.viewExe_eq]
    have hT := leafT_hm96 C c
    have hS : (C.T c).tab.Admits (sentAt C c) :=
      ZK.Session.sent_admits (fun _ : Fin 1 => (C.T c).tab) (fun _ => (C.T c).rep) 0
    have h1 : Assumptions.Hm96Hiding (C.T c).hm96.My (C.T c).hm96.cs ((2 : ℝ≥0∞) ^ (-193 : ℤ)) := by
      rw [ZkView.Coined.hm96At_My (C.T c) ZkView.My rfl, ZkView.Coined.hm96At_cs (C.T c) ZkView.My rfl]
      exact hKey
    have h := ZK.Table.table_abort_hm96 (C.T c).tab (C.T c).hm96 _ hT h1 (sentAt C c) hS w hreg w0
      (fun v => E (.inr v))
    rw [card_pre] at h
    exact h

end FlockSoundness.Discharge.ZkEveryCoin

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec Composed
  Placed StrictCR DecodeZL
open scoped ENNReal

set_option synthInstance.maxHeartbeats 1000000
set_option maxHeartbeats 4000000

open Classical in
/-- **What the session's honest prover sends** at `J` tables, coin vector `c`, witnesses `w` and prover randomness `p`:
`stopped p` if some table's self-check stops, else the joint view `enc` encodes. -/
noncomputable def sendsJ {J : ℕ} (C : Fin J → CoinedSetup) (c : ∀ j, CoinsZK (C j).Rest) {Out : Type}
    (enc : (∀ j, ((C j).T (c j)).tab.View) → Out) (w : ∀ j, Witness (C j).S.m)
    (stopped : (∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) → Out)
    (p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) : Out :=
  if ∃ j, ((C j).T (c j)).tab.Stops (w j) (p j).1 then stopped p
  else enc fun j => ((C j).T (c j)).viewExe (w j) (p j).1 (p j).2

/-- **The self-check never stops** at coins the prover doesn't refuse, valid witnesses and public registered values:
`CoinedOK`'s inner relation. -/
theorem stopsJ_never {J : ℕ} {C : Fin J → CoinedSetup} (hC : ∀ j, CoinedOK execArith (fOf Zk.PAD_C) (C j))
    {c : ∀ j, CoinsZK (C j).Rest} (hc : ∀ j, (C j).Admits (c j)) {w : ∀ j, Witness (C j).S.m}
    (hv : ∀ j, (C j).Valid (w j)) (hreg : ∀ j, ((C j).T (c j)).tab.sReg (w j) = ((C j).T (c j)).tab.pubReg)
    (p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) :
    ¬ ∃ j, ((C j).T (c j)).tab.Stops (w j) (p j).1 := by
  rintro ⟨j, hj⟩
  obtain ⟨-, -, -, hT⟩ := hC j
  obtain ⟨-, -, -, -, -, -, -, hinner⟩ := hT (c j) (hc j)
  exact ((C j).T (c j)).tab.not_stops (hinner (w j) (hv j) (hreg j)) (p j).1 hj

/-- The stop bit is false with probability one: the simulator outputs `false` at no cost. -/
theorem prCoin_stopsJ {J : ℕ} {C : Fin J → CoinedSetup} (hC : ∀ j, CoinedOK execArith (fOf Zk.PAD_C) (C j))
    {c : ∀ j, CoinsZK (C j).Rest} (hc : ∀ j, (C j).Admits (c j)) {w : ∀ j, Witness (C j).S.m}
    (hv : ∀ j, (C j).Valid (w j)) (hreg : ∀ j, ((C j).T (c j)).tab.sReg (w j) = ((C j).T (c j)).tab.pubReg) :
    Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
      ∃ j, ((C j).T (c j)).tab.Stops (w j) (p j).1) = 0 :=
  le_zero_iff.1 ((Game.prCoin_mono (q := fun _ => False) fun p hp => stopsJ_never hC hc hv hreg p hp).trans
    Game.prCoin_false.le)

/-- **The honest prover's transmission is the joint view** at coins it doesn't refuse, valid witnesses and public
registered values. -/
theorem sendsJ_honest {J : ℕ} {C : Fin J → CoinedSetup} (hC : ∀ j, CoinedOK execArith (fOf Zk.PAD_C) (C j))
    {c : ∀ j, CoinsZK (C j).Rest} (hc : ∀ j, (C j).Admits (c j)) {w : ∀ j, Witness (C j).S.m}
    (hv : ∀ j, (C j).Valid (w j)) (hreg : ∀ j, ((C j).T (c j)).tab.sReg (w j) = ((C j).T (c j)).tab.pubReg)
    {Out : Type} (enc : (∀ j, ((C j).T (c j)).tab.View) → Out)
    (stopped : (∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) → Out)
    (p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) :
    sendsJ C c enc w stopped p = enc fun j => ((C j).T (c j)).viewExe (w j) (p j).1 (p j).2 := by
  have hn := stopsJ_never hC hc hv hreg p
  unfold sendsJ
  split <;> first | contradiction | rfl

/-- **Zero knowledge at `J` `CoinedSetup`s, the self-check included** (`ZkViewAtJ` with the real side `sendsJ`): at any
coin vector of each table that the honest prover doesn't refuse, for any encoding `enc` of the joint view, anything
`stopped` a stopped run sends, and any event `E`, the honest prover's transmission at valid witnesses `w j` whose
registered values are the public ones and the simulators' joint view at any `w0 j` are within `Σ_j 2·|Hid_j|·δ₁`, both
ways. -/
def ZkViewAtJS {J : ℕ} (C : Fin J → CoinedSetup) (δ₁ : ℝ≥0∞) : Prop :=
  ∀ (c : ∀ j, CoinsZK (C j).Rest), (∀ j, (C j).Admits (c j)) → ∀ {Out : Type}
    (enc : (∀ j, ((C j).T (c j)).tab.View) → Out) (w w0 : ∀ j, Witness (C j).S.m), (∀ j, (C j).Valid (w j)) →
    ∀ (_hreg : ∀ j, ((C j).T (c j)).tab.sReg (w j) = ((C j).T (c j)).tab.pubReg)
      (stopped : (∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) → Out) (E : Out → Prop),
    Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
        E (sendsJ C c enc w stopped p))
        ≤ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E (enc fun j => ((C j).T (c j)).simExe (w0 j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁
      ∧ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
          E (enc fun j => ((C j).T (c j)).simExe (w0 j) (p j).1 (p j).2))
        ≤ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E (sendsJ C c enc w stopped p))
          + ∑ j, 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁

/-- `ZkViewAtJ` gives `ZkViewAtJS` at the same bound when every table is `CoinedOK`. -/
theorem zkViewAtJS_of {J : ℕ} {C : Fin J → CoinedSetup} {δ₁ : ℝ≥0∞}
    (hC : ∀ j, CoinedOK execArith (fOf Zk.PAD_C) (C j)) (h : ZkViewAtJ C δ₁) : ZkViewAtJS C δ₁ := by
  intro c hc Out enc w w0 hv hreg stopped E
  have hs : ∀ p, sendsJ C c enc w stopped p = enc fun j => ((C j).T (c j)).viewExe (w j) (p j).1 (p j).2 :=
    sendsJ_honest hC hc hv hreg enc stopped
  simp only [hs]
  exact h c hc enc w w0 hv E

/-! ## At every coin vector -/

open Classical in
/-- **What the session's honest prover sends at any coin vector** `c`: `stopped p` if the self-check stops at a rep of
some table whose outer coins aren't `1`, else `enc` of each table's output (`ZkEveryCoin.outExe`: its view where the
prover doesn't refuse the table's coins, and otherwise its cut `Table.abort` to what it sent before it stopped).

The session stops at its first refused coin and sends no proof after (the barrier), so what it sent is a function, at
fixed coins, of the tables' cuts up to there, each a function of the table's output; tables after it, cut or whole, only
add to what `enc` may read. A rep's self-check runs after its zerocheck opened its outer coins, where a `1` stops the
session first (`refuse_outer`), so every rep whose self-check runs is one the guard ranges over. -/
noncomputable def sendsAllJ {J : ℕ} (C : Fin J → CoinedSetup) (c : ∀ j, CoinsZK (C j).Rest) {Out : Type}
    (enc : (∀ j, ((C j).T (c j)).tab.View ⊕ ((C j).T (c j)).tab.Cut) → Out) (w : ∀ j, Witness (C j).S.m)
    (stopped : (∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) → Out)
    (p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) : Out :=
  if ∃ j r, (∀ x ∈ (C j).rOuter (c j) r, x ≠ 1) ∧ ((C j).T (c j)).tab.StopsAt (w j) (p j).1 r then stopped p
  else enc fun j => ZkEveryCoin.outExe (C j) (c j) (w j) (p j).1 (p j).2

/-- **The self-check passes at every rep whose outer coins aren't `1`**, at every coin vector, valid witness whose
registered values are the public ones and draw (`Composed.innerOf_holds_rep_of` at a setup's table). -/
def SelfCheckAtReps (C : CoinedSetup) : Prop :=
  ∀ (c : CoinsZK C.Rest) (r : Fin 2), (∀ x ∈ C.rOuter c r, x ≠ 1) → ∀ w, C.Valid w →
    (C.T c).tab.sReg w = (C.T c).tab.pubReg → ∀ ω, ¬ (C.T c).tab.StopsAt w ω r

/-- **The self-check never stops at a rep whose outer coins aren't `1`**, at any coin vector, valid witnesses and public
registered values. -/
theorem stopsAllJ_never {J : ℕ} {C : Fin J → CoinedSetup} (hR : ∀ j, SelfCheckAtReps (C j))
    {c : ∀ j, CoinsZK (C j).Rest} {w : ∀ j, Witness (C j).S.m} (hv : ∀ j, (C j).Valid (w j))
    (hreg : ∀ j, ((C j).T (c j)).tab.sReg (w j) = ((C j).T (c j)).tab.pubReg)
    (p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) :
    ¬ ∃ j r, (∀ x ∈ (C j).rOuter (c j) r, x ≠ 1) ∧ ((C j).T (c j)).tab.StopsAt (w j) (p j).1 r := by
  rintro ⟨j, r, hr, hj⟩
  exact hR j (c j) r hr (w j) (hv j) (hreg j) (p j).1 hj

/-- **The honest prover's transmission is `enc` of the tables' outputs** at any coin vector, valid witnesses and public
registered values. -/
theorem sendsAllJ_honest {J : ℕ} {C : Fin J → CoinedSetup} (hR : ∀ j, SelfCheckAtReps (C j))
    {c : ∀ j, CoinsZK (C j).Rest} {w : ∀ j, Witness (C j).S.m} (hv : ∀ j, (C j).Valid (w j))
    (hreg : ∀ j, ((C j).T (c j)).tab.sReg (w j) = ((C j).T (c j)).tab.pubReg) {Out : Type}
    (enc : (∀ j, ((C j).T (c j)).tab.View ⊕ ((C j).T (c j)).tab.Cut) → Out)
    (stopped : (∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) → Out)
    (p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) :
    sendsAllJ C c enc w stopped p = enc fun j => ZkEveryCoin.outExe (C j) (c j) (w j) (p j).1 (p j).2 := by
  have hn := stopsAllJ_never hR hv hreg p
  unfold sendsAllJ
  split <;> first | contradiction | rfl

/-- **Zero knowledge at `J` `CoinedSetup`s at every coin vector, the self-check and the refused coins included**: at any
coin vector `c`, for any encoding `enc` of the tables' outputs, anything `stopped` a stopped run sends and any event `E`,
the honest prover's transmission (`sendsAllJ`) at valid witnesses `w j` whose registered values are the public ones and
the simulators' output (`ZkEveryCoin.simAll`: `S_shvzk` at a table whose coins the prover doesn't refuse, the honest
prefix on the dummy witness `w0 j` with the public region values at one it refuses), each table's randomness
independent, are within `Σ_j 2·|LeafPos_j|·δ₁`, both ways. The refusal reads only the coins, so the simulator refuses
exactly where the prover does. -/
def ZkViewAllJ {J : ℕ} (C : Fin J → CoinedSetup) (δ₁ : ℝ≥0∞) : Prop :=
  ∀ (c : ∀ j, CoinsZK (C j).Rest) {Out : Type}
    (enc : (∀ j, ((C j).T (c j)).tab.View ⊕ ((C j).T (c j)).tab.Cut) → Out) (w w0 : ∀ j, Witness (C j).S.m),
    (∀ j, (C j).Valid (w j)) → ∀ (_hreg : ∀ j, ((C j).T (c j)).tab.sReg (w j) = ((C j).T (c j)).tab.pubReg)
      (stopped : (∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) → Out) (E : Out → Prop),
    Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
        E (sendsAllJ C c enc w stopped p))
        ≤ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E (enc fun j => ZkEveryCoin.simAll (C j) (c j) (w0 j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card (ZkView.LeafPos (C j).z) * δ₁
      ∧ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
          E (enc fun j => ZkEveryCoin.simAll (C j) (c j) (w0 j) (p j).1 (p j).2))
        ≤ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E (sendsAllJ C c enc w stopped p))
          + ∑ j, 2 * Fintype.card (ZkView.LeafPos (C j).z) * δ₁

set_option maxRecDepth 8000 in
/-- **Zero knowledge at `J` `CoinedSetup`s at every coin vector, from their facts**: each table's output against its
simulator's at its coin vector (`ZkEveryCoin.coin_close_at`: Lemma B where the prover doesn't refuse the table's coins,
`table_abort_hm96` where it does), composed over the tables' independent randomness (`prCoin_pi_close`); the self-check
passes at each table's reps (`SelfCheckAtReps`). -/
theorem zkViewAllJ_of_ok {J : ℕ} (C : Fin J → CoinedSetup) (hC : ∀ j, CoinedOK execArith (fOf Zk.PAD_C) (C j))
    (hR : ∀ j, SelfCheckAtReps (C j)) (hKey : Assumptions.Zk.HashDerivedKeyHm96) :
    ZkViewAllJ C ((2 : ℝ≥0∞) ^ (-193 : ℤ)) := by
  intro c Out enc w w0 hv hreg stopped E
  have hs : ∀ p, sendsAllJ C c enc w stopped p = enc fun j => ZkEveryCoin.outExe (C j) (c j) (w j) (p j).1 (p j).2 :=
    sendsAllJ_honest hR hv hreg enc stopped
  simp only [hs]
  have : Nonempty ZkView.Salt := ⟨0⟩
  have : ∀ j, Nonempty (((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) :=
    fun j => ⟨((rand_nonempty ((C j).T (c j)).tab).some, fun _ => 0)⟩
  have : ∀ j, Nonempty (((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) :=
    fun j => ⟨((simRand_nonempty ((C j).T (c j)).tab).some, fun _ => 0)⟩
  have one := fun j => ZkEveryCoin.coin_close_at (C j) (hC j) hKey (c j) (w j) (w0 j) (hv j) (hreg j)
  exact ⟨FlockSoundness.ZK.prCoin_pi_close
      (Ω := fun j => ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt))
      (Ω' := fun j => ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt))
      (V := fun j => ((C j).T (c j)).tab.View ⊕ ((C j).T (c j)).tab.Cut)
      (fun j p => ZkEveryCoin.outExe (C j) (c j) (w j) p.1 p.2)
      (fun j p => ZkEveryCoin.simAll (C j) (c j) (w0 j) p.1 p.2)
      (fun j => 2 * Fintype.card (ZkView.LeafPos (C j).z) * (2 : ℝ≥0∞) ^ (-193 : ℤ)) (fun j E' => (one j E').1)
      fun v => E (enc v),
    FlockSoundness.ZK.prCoin_pi_close
      (Ω := fun j => ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt))
      (Ω' := fun j => ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt))
      (V := fun j => ((C j).T (c j)).tab.View ⊕ ((C j).T (c j)).tab.Cut)
      (fun j p => ZkEveryCoin.simAll (C j) (c j) (w0 j) p.1 p.2)
      (fun j p => ZkEveryCoin.outExe (C j) (c j) (w j) p.1 p.2)
      (fun j => 2 * Fintype.card (ZkView.LeafPos (C j).z) * (2 : ℝ≥0∞) ^ (-193 : ℤ)) (fun j E' => (one j E').2)
      fun v => E (enc v)⟩

variable {mPts n : ℕ} {Reg : Type} (P : LigProver)

/-- **One hidden-output `--zk` sampled session of `J` tables is zero knowledge, its self-check included**: under
`zk_session_viewHJ`'s hypotheses (`ZeroKnowledgeHidden`'s), the honest prover's transmission (`sendsJ`) at `planZHJ`'s
tables, at the same `Σ_j 2·|Hid_j|·2^-193`. -/
theorem zk_session_viewHJS (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false)
    (hdm : ∀ (S : Finset (Fin n)) (j : Fin (nTab I))
      (y : DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)), avoidsMask y.x.st = true)
    (hKey : Assumptions.Zk.HashDerivedKeyHm96) :
    ∀ (S : Finset (Fin n)) (R : Reg)
      (ys : ∀ j : Fin (nTab I), DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)),
      ZkViewAtJS (fun j => coinedOf P ((planZHJ (Reg := Reg) dj y₀ S R).get
        (Fin.cast (planZHJ_length dj y₀ S R).symm j)) (ys j).x.st) ((2 : ℝ≥0∞) ^ (-193 : ℤ)) :=
  fun S R ys => zkViewAtJS_of (fun j => (coinedOf_setup_of P _ (ys j).x.st (ys j).x.wf.1
    (by rw [planZHJ_get_eq dj y₀ S R j (ys j)]; rfl)
    (maskSetup_of_setupHidden ((tagsAt_typed _ _ _).trans ((ZkExec.zkTags_typed I).trans ht)) (ys j).x.h)
    (hdm S j (ys j))).2.2.2.2.2) (zk_session_viewHJ P I dj y₀ ht hdm hKey S R ys)

/-- **One hidden-output `--zk` sampled session of `J` tables is zero knowledge at every coin vector**, its self-check
and its refused coins included: under `zk_session_viewHJ`'s hypotheses (`ZeroKnowledgeHidden`'s), the honest prover's
transmission (`sendsAllJ`) at `planZHJ`'s tables and the per-table simulators' output (`ZkEveryCoin.simAll`), at any coin
vector, within `Σ_j 2·|LeafPos_j|·2^-193`. No hypothesis on the coins: where the prover refuses a table's coins, every
pre-final leaf is unopened and T1 is paid on all of them (`|LeafPos_j| = |Pre_j|`, against `|Hid_j|` where it doesn't).

As `zk_session_viewHJ`, this is honest-verifier zero knowledge at each fixed coin vector; a verifier that chooses its
coins is covered by the coin server's commitment at `Hello` (`coin_tree.rs`) under `keyed-cr`, which no statement here
models yet. -/
theorem zk_session_view_allHJ (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false)
    (hdm : ∀ (S : Finset (Fin n)) (j : Fin (nTab I))
      (y : DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)), avoidsMask y.x.st = true)
    (hKey : Assumptions.Zk.HashDerivedKeyHm96) :
    ∀ (S : Finset (Fin n)) (R : Reg)
      (ys : ∀ j : Fin (nTab I), DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)),
      ZkViewAllJ (fun j => coinedOf P ((planZHJ (Reg := Reg) dj y₀ S R).get
        (Fin.cast (planZHJ_length dj y₀ S R).symm j)) (ys j).x.st) ((2 : ℝ≥0∞) ^ (-193 : ℤ)) :=
  fun S R ys => zkViewAllJ_of_ok _ (fun j => (coinedOf_setup_of P _ (ys j).x.st (ys j).x.wf.1
    (by rw [planZHJ_get_eq dj y₀ S R j (ys j)]; rfl)
    (maskSetup_of_setupHidden ((tagsAt_typed _ _ _).trans ((ZkExec.zkTags_typed I).trans ht)) (ys j).x.h)
    (hdm S j (ys j))).2.2.2.2.2)
    (fun j c r hr w hv hreg ω hs => hs (innerOf_holds_rep_of P _ (ys j).x.st (ys j).x.wf.1
      (by rw [planZHJ_get_eq dj y₀ S R j (ys j)]; rfl)
      (maskSetup_of_setupHidden ((tagsAt_typed _ _ _).trans ((ZkExec.zkTags_typed I).trans ht)) (ys j).x.h)
      (hdm S j (ys j)) c r hr w hv hreg ω)) hKey

end FlockSoundness.Discharge.ZkHidden
