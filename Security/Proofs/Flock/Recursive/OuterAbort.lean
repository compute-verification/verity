import Proofs.Flock.Soundness.Discharge.ZkHidden.View
import Proofs.Flock.Soundness.Game.Prob

/-!
# V*'s outer sessions: the honest prover's self-check never stops

V*'s outer sessions are `--zk` sessions of `J` tables, and their honest prover stops when its own batched constraint
fails (`zk_veil.rs`'s `prove_inner` under `SelfCheck::Stop`: `⟨c, h⟩ + t ≠ 0` after the batching coins, before `τ`'s
commitment). The stop reads the witness, so the simulator could not reproduce it unless it never fires. `ZeroKnowledgeHidden`'s
real view never stops; this module shows the honest prover's transmission with the stop in it is that view.

* `Table.Stops`: rep `r`'s self-check fails, `⟨λ, h⟩ ≠ θ` (in characteristic 2 the same as `⟨c, h⟩ + t ≠ 0`), and
  `Table.not_stops`: never, under `InnerHolds`. `Table.viewWith_pre`, `Coined.viewExe_pre`: the check reads only what
  the executable prover sent before `τ`.
* `sendsJ`: the session's honest prover at `J` tables: what a stopped run sends (`stopped`, any function of the prover's
  randomness) if some table's self-check stops, else the joint view.
* `stopsJ_never`, `prCoin_stopsJ`, `sendsJ_honest`: at `Admits` coins, valid witnesses and public registered values,
  `CoinedOK`'s inner relation (`InnerHolds`) makes the stop bit constantly false, so `sendsJ` is the joint view.
* `ZkViewAtJS`, `zkViewAtJS_of`, `zk_session_viewHJS`: `ZkViewAtJ` with the real side the honest prover's transmission,
  at the same bound, and at `zk_session_viewHJ`'s plan under its hypotheses. The abort costs nothing.

The prover's other stops (`β = 0`, dependent `ρ₀, ρ₁`, the mask rank, a zerocheck outer coin `1`) read only opened coins:
they are `CoinedSetup.Admits`, which `ZkViewAtJ` assumes.
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

end Stops

section Leaves

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [AddCommGroup B]
  (T : Table F K W Salt (B × Salt) D Hi VL) [CharP F 2]

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
  exact h c hc enc w w0 hv hreg E

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

end FlockSoundness.Discharge.ZkHidden
