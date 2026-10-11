import Proofs.Flock.Soundness.Discharge.ZkEveryCoin.Defs
import Proofs.Flock.Soundness.Discharge.Composed.Headline

/-!
# One `--zk` session is zero knowledge at every coin vector

`Composed.zk_session_view` holds at the coin vectors the honest prover doesn't refuse (`Admits`). At every coin vector,
with the verifier's coins uniform:

* where the prover refuses (`¬ Admits`, `refused_coins`, a zerocheck outer coin `1` among them), it stops, and what it
  sent is a function of a cut of its view (`Table.abort`). The refusal reads only the coins, which are public, so the
  simulator aborts at exactly those coins and outputs the honest prefix on the dummy witness (`simAll`). Lemma A at every
  coin vector (`table_prefinal_indep_pub`) and T1 on every pre-final leaf bound the difference by `2·|LeafPos|·δ₁`
  (`Table.table_abort_hm96`), with no hypothesis on the coins: every leaf is unopened at a refusal, so T1 is paid on all
  of them, and Lemma B's re-randomization of the opened rows (`PadOnto`, which `CoinedOK` gives only where the prover
  doesn't refuse) isn't needed;
* where it doesn't refuse, Lemma B (`zkViewAt_of_ok`), within `2·|Hid|·δ₁ ≤ 2·|LeafPos|·δ₁`.

No coin vector is left: the outer coin `1`, where the view would not be simulated, is a refusal.

* `zkViewAll_of_ok`: `ZkViewAll` at a `CoinedSetup` from `CoinedOK` and `hash-derived-key`.
* `outer_one_le`: at a draw's tables, the prover refuses for an outer coin `1` with probability at most
  `2·(m − 13)·2^-128` (completeness).
* `zk_session_view_all`: at every draw `verify --zk` sets up, `ZkViewAll` at `2^-193`.
-/

namespace FlockSoundness.Discharge.ZkEveryCoin

open Game Audit Exec Refine Flock FlockLevel3 Model ZkSession ZkLink ZkExec DecodeZL Composed ZkView
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

local notation "𝔽" => FlockLevel3.GF128

theorem avg_const_le {C : Type} [Fintype C] (x : ℝ≥0∞) : avg (fun _ : C => x) ≤ x := by
  rcases isEmpty_or_nonempty C with h | h
  · simp [avg]
  · exact (avg_const x).le

theorem prCoin_le_one {C : Type} [Fintype C] (p : C → Prop) : prCoin p ≤ 1 := by
  unfold prCoin
  refine le_trans (avg_mono fun c => ?_) (avg_const_le (C := C) 1)
  split_ifs <;> simp

theorem card_hid_le (C : CoinedSetup) (c : CoinsZK C.Rest) :
    Fintype.card (C.T c).tab.Hid ≤ Fintype.card (LeafPos C.z) := by
  simp only [ZK.Table.Hid, LeafPos, Fintype.card_sum, Fintype.card_prod, Fintype.card_fin]
  change C.z.nU + 2 * C.z.nPU ≤ C.z.nS + C.z.nU + (2 * (C.z.nQ + C.z.nPU) + 2)
  omega

theorem card_pre (C : CoinedSetup) (c : CoinsZK C.Rest) :
    Fintype.card (C.T c).tab.Pre = Fintype.card (LeafPos C.z) := by
  simp only [ZK.Table.Pre, ZK.Table.Hid, LeafPos, Fintype.card_sum, Fintype.card_prod, Fintype.card_fin]
  change C.z.nS + (2 * C.z.nQ + 2) + (C.z.nU + 2 * C.z.nPU) = C.z.nS + C.z.nU + (2 * (C.z.nQ + C.z.nPU) + 2)
  ring

/-- `τ`'s commitment is the hm96 leaf of `SHA-512(TAU_TAG ‖ τ)` (`leafSplit`). -/
theorem leafT_hm96 (C : CoinedSetup) (c : CoinsZK C.Rest) (t : 𝔽) (y : ZkView.Salt) :
    (C.T c).tab.leafT t y = (C.T c).hm96.H
      (ZkView.bitsOf (Sha512.hash (Zk.TAU_TAG ++ F128.toBytes t)) + (C.T c).hm96.My y, (C.T c).hm96.cs y) := by
  rw [congrFun (ZkView.Coined.hm96At_My (C.T c) ZkView.My rfl) y]
  exact ZkView.leafSplit (Sha512.hash (Zk.TAU_TAG ++ F128.toBytes t)) y (Sha512.hash_size _)

open Classical in
/-- **One coin vector, at any hm96 hiding bound `δ`**: the prover's and the simulator's probabilities at `c` are within
`2·|LeafPos|·δ`. -/
theorem coin_close_le (C : CoinedSetup) (hC : CoinedOK execArith (Refine.fOf Flock.Zk.PAD_C) C)
    {δ : ℝ≥0∞} (hT1 : Assumptions.Hm96Hiding ZkView.My ZkView.saltHash δ)
    (w w0 : Witness C.S.m) (hv : C.Valid w) (hreg : ∀ c, (C.T c).tab.sReg w = (C.T c).tab.pubReg)
    (E : ∀ c : CoinsZK C.Rest, (C.T c).tab.View ⊕ (C.T c).tab.Cut → Prop) (c : CoinsZK C.Rest) :
    prCoin (fun p : (C.T c).tab.Rand × ((C.T c).tab.Hid → ZkView.Salt) => E c (outExe C c w p.1 p.2))
        ≤ prCoin (fun p : (C.T c).tab.SimRand × ((C.T c).tab.Hid → ZkView.Salt) => E c (simAll C c w0 p.1 p.2))
          + 2 * Fintype.card (LeafPos C.z) * δ
      ∧ prCoin (fun p : (C.T c).tab.SimRand × ((C.T c).tab.Hid → ZkView.Salt) => E c (simAll C c w0 p.1 p.2))
        ≤ prCoin (fun p : (C.T c).tab.Rand × ((C.T c).tab.Hid → ZkView.Salt) => E c (outExe C c w p.1 p.2))
          + 2 * Fintype.card (LeafPos C.z) * δ := by
  by_cases hA : C.Admits c
  · simp only [outExe, simAll, hA, ite_true]
    obtain ⟨h1, h2⟩ := zkViewAt_of_ok C hC hT1 c hA
      (fun v => (Sum.inl (v 0) : (C.T c).tab.View ⊕ (C.T c).tab.Cut)) (fun _ => w) (fun _ => w0) (fun _ => hv)
      (fun _ => hreg c) (E c)
    have eR := ZK.prCoin_congr_equiv (Equiv.funUnique (Fin 1) ((C.T c).tab.Rand × ((C.T c).tab.Hid → ZkView.Salt)))
      (fun p => E c (.inl ((C.T c).viewExe w p.1 p.2)))
    have eS := ZK.prCoin_congr_equiv (Equiv.funUnique (Fin 1) ((C.T c).tab.SimRand × ((C.T c).tab.Hid → ZkView.Salt)))
      (fun p => E c (.inl ((C.T c).simExe w0 p.1 p.2)))
    have hs : ∑ _j : Fin 1, 2 * (Fintype.card (C.T c).tab.Hid : ℝ≥0∞) * δ
        ≤ 2 * Fintype.card (LeafPos C.z) * δ := by
      rw [Fin.sum_univ_one]
      gcongr
      exact_mod_cast card_hid_le C c
    exact ⟨(eR.symm.trans_le h1).trans (add_le_add (le_of_eq eS) hs),
      (eS.symm.trans_le h2).trans (add_le_add (le_of_eq eR) hs)⟩
  · simp only [outExe, simAll, hA, ite_false, ZkView.Coined.viewExe_eq]
    have hT := leafT_hm96 C c
    have hS : (C.T c).tab.Admits (sentAt C c) :=
      ZK.Session.sent_admits (fun _ : Fin 1 => (C.T c).tab) (fun _ => (C.T c).rep) 0
    have h1 : Assumptions.Hm96Hiding (C.T c).hm96.My (C.T c).hm96.cs δ := by
      rw [ZkView.Coined.hm96At_My (C.T c) ZkView.My rfl, ZkView.Coined.hm96At_cs (C.T c) ZkView.My rfl]
      exact hT1
    have h := ZK.Table.table_abort_hm96 (C.T c).tab (C.T c).hm96 _ hT h1 (sentAt C c) hS w (hreg c) w0
      (fun v => E c (.inr v))
    rw [card_pre] at h
    exact h

open Classical in
/-- **One coin vector**: the prover's and the simulator's probabilities at `c` are within `2·|LeafPos|·2^-193`
(`coin_close_le` under `hash-derived-key`). -/
theorem coin_close (C : CoinedSetup) (hC : CoinedOK execArith (Refine.fOf Flock.Zk.PAD_C) C)
    (hKey : Assumptions.Zk.HashDerivedKeyHm96)
    (w w0 : Witness C.S.m) (hv : C.Valid w) (hreg : ∀ c, (C.T c).tab.sReg w = (C.T c).tab.pubReg)
    (E : ∀ c : CoinsZK C.Rest, (C.T c).tab.View ⊕ (C.T c).tab.Cut → Prop) (c : CoinsZK C.Rest) :
    prCoin (fun p : (C.T c).tab.Rand × ((C.T c).tab.Hid → ZkView.Salt) => E c (outExe C c w p.1 p.2))
        ≤ prCoin (fun p : (C.T c).tab.SimRand × ((C.T c).tab.Hid → ZkView.Salt) => E c (simAll C c w0 p.1 p.2))
          + 2 * Fintype.card (LeafPos C.z) * (2 : ℝ≥0∞) ^ (-193 : ℤ)
      ∧ prCoin (fun p : (C.T c).tab.SimRand × ((C.T c).tab.Hid → ZkView.Salt) => E c (simAll C c w0 p.1 p.2))
        ≤ prCoin (fun p : (C.T c).tab.Rand × ((C.T c).tab.Hid → ZkView.Salt) => E c (outExe C c w p.1 p.2))
          + 2 * Fintype.card (LeafPos C.z) * (2 : ℝ≥0∞) ^ (-193 : ℤ) :=
  coin_close_le C hC (δ := (2 : ℝ≥0∞) ^ (-193 : ℤ)) hKey w w0 hv hreg E c

/-- **Zero knowledge at every coin vector at a `CoinedSetup` from its facts**: Lemma B where the prover doesn't refuse
(`zkViewAt_of_ok`), and the refusal where it does (`Table.table_abort_hm96`, which reads nothing of `CoinedOK`). -/
theorem zkViewAll_of_ok (C : CoinedSetup) [Fintype (CoinsZK C.Rest)]
    (hC : CoinedOK execArith (Refine.fOf Flock.Zk.PAD_C) C)
    (hKey : Assumptions.Zk.HashDerivedKeyHm96) :
    ZkViewAll C ((2 : ℝ≥0∞) ^ (-193 : ℤ)) := by
  intro w w0 hv hreg E
  have hc := coin_close C hC hKey w w0 hv hreg E
  constructor
  · refine (avg_mono fun c => (hc c).1).trans ?_
    rw [avg_add]
    gcongr
    exact avg_const_le _
  · refine (avg_mono fun c => (hc c).2).trans ?_
    rw [avg_add]
    gcongr
    exact avg_const_le _

/-- **A coordinate that can be overwritten in place is uniform**: if `upd c x` puts `x` at `f`'s coordinate and nowhere
else, `f` of a uniform `c` hits any value with probability at most `1/|Val|`. -/
theorem prCoin_coord_le {Ω Val : Type} [Fintype Ω] [Fintype Val] [DecidableEq Val] (f : Ω → Val) (upd : Ω → Val → Ω)
    (h1 : ∀ c x, f (upd c x) = x) (h2 : ∀ c x y, upd (upd c x) y = upd c y) (h3 : ∀ c, upd c (f c) = c) (x : Val) :
    prCoin (fun c => f c = x) ≤ (Fintype.card Val : ℝ≥0∞)⁻¹ := by
  classical
  rcases isEmpty_or_nonempty Ω with hC | ⟨⟨c₀⟩⟩
  · simp [prCoin, avg]
  let e : Ω ≃ Val × {c // f c = x} :=
    { toFun := fun c => (f c, ⟨upd c x, h1 c x⟩)
      invFun := fun p => upd p.2.1 p.1
      left_inv := fun c => by simp only [h2, h3]
      right_inv := fun p => by
        obtain ⟨y, d, hd⟩ := p
        refine Prod.ext (h1 d y) (Subtype.ext ?_)
        change upd (upd d y) x = d
        rw [h2, ← hd, h3] }
  have : Nonempty {c // f c = x} := ⟨⟨upd c₀ x, h1 c₀ x⟩⟩
  have h := ZK.prCoin_congr_equiv e (fun p => p.1 = x)
  rw [show (fun c => f c = x) = fun c => (e c).1 = x from rfl, h,
    ZK.prCoin_fst (E := {c // f c = x}) (fun y : Val => y = x), prCoin_eq_card]
  simp [Finset.filter_eq']

theorem card_f128 : Fintype.card F128 = 2 ^ 128 := by
  rw [Fintype.card_congr f128Equiv, Fintype.card_prod, Fintype.card_fin, ← pow_add]

/-- Rep `r`'s `j`-th zerocheck outer coin of a draw's coins, overwritten with `x`. -/
noncomputable def setOuter (P : LigProver) {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (r : Fin 2) (j : Fin (st.m - 13))
    (c : CoinsZK (coinedOf P Tz st).Rest) (x : F128) : CoinsZK (coinedOf P Tz st).Rest :=
  { c with rest := { c.rest with rOuter := Function.update c.rest.rOuter r (Function.update (c.rest.rOuter r) j x) } }

/-- **Each zerocheck outer coin is uniform over F128**: it is `1` with probability at most `2^-128`. -/
theorem prCoin_outer_one (P : LigProver) {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (r : Fin 2) (j : Fin (st.m - 13)) :
    prCoin (fun c : CoinsZK (coinedOf P Tz st).Rest => c.rest.rOuter r j = 1) ≤ (2 : ℝ≥0∞) ^ (-128 : ℤ) := by
  classical
  refine (prCoin_coord_le (fun c : CoinsZK (coinedOf P Tz st).Rest => c.rest.rOuter r j) (setOuter P Tz st r j)
    (fun c x => ?_) (fun c x y => ?_) (fun c => ?_) 1).trans (le_of_eq ?_)
  · simp [setOuter]
  · simp [setOuter]
  · simp only [setOuter, Function.update_eq_self]
    rfl
  · rw [card_f128, ENNReal.zpow_neg, Nat.cast_pow, Nat.cast_ofNat]
    norm_cast

/-- **The prover refuses for an outer coin `1` rarely** (completeness), at a draw's tables: each of the `m − 13` outer
coins of each of the two reps is uniform over F128. -/
theorem outer_one_le (P : LigProver) {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) :
    prCoin (fun c : CoinsZK (coinedOf P Tz st).Rest => ¬ ∀ r, ∀ x ∈ (coinedOf P Tz st).rOuter c r, x ≠ 1)
      ≤ 2 * (st.m - 13 : ℕ) * (2 : ℝ≥0∞) ^ (-128 : ℤ) := by
  classical
  have hsub : ∀ c : CoinsZK (coinedOf P Tz st).Rest, (¬ ∀ r, ∀ x ∈ (coinedOf P Tz st).rOuter c r, x ≠ 1) →
      ∃ i ∈ (Finset.univ : Finset (Fin 2 × Fin (st.m - 13))), c.rest.rOuter i.1 i.2 = 1 := by
    intro c h
    have : ¬ ∀ r, ∀ x ∈ List.ofFn (c.rest.rOuter r), x ≠ 1 := h
    push Not at this
    obtain ⟨r, x, hx, h1⟩ := this
    obtain ⟨j, hj⟩ := List.mem_ofFn.1 hx
    exact ⟨(r, j), Finset.mem_univ _, hj.trans h1⟩
  have hm := prCoin_mono hsub
  have he := prCoin_exists_le (C := CoinsZK (coinedOf P Tz st).Rest) (Finset.univ : Finset (Fin 2 × Fin (st.m - 13)))
    (fun i c => c.rest.rOuter i.1 i.2 = 1)
  have hs : ∑ i : Fin 2 × Fin (st.m - 13), prCoin (fun c : CoinsZK (coinedOf P Tz st).Rest => c.rest.rOuter i.1 i.2 = 1)
      ≤ ∑ _i : Fin 2 × Fin (st.m - 13), (2 : ℝ≥0∞) ^ (-128 : ℤ) :=
    Finset.sum_le_sum fun i _ => prCoin_outer_one P Tz st i.1 i.2
  refine hm.trans (he.trans (hs.trans (le_of_eq ?_)))
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin, nsmul_eq_mul,
    Nat.cast_mul, Nat.cast_ofNat]

variable {mPts n : ℕ} {Reg : Type} (P : LigProver)

/-- **One `--zk` sampled session is zero knowledge at every coin vector.** At every draw `verify --zk` sets up, with the
verifier's coins uniform: for every event on the coins and what the honest prover sends (its view, or the prefix of it
before it stops at a refused coin), at any valid witness whose registered values are the public ones, its probability
and its probability under the simulator (`S_shvzk`, or the honest prefix on the dummy witness where the prover refuses,
at any `w0`) are within `2·|LeafPos|·2^-193`, both ways, T1's on every leaf. A zerocheck outer coin `1` is a refusal
(`CoinedSetup.Admits`), so no coin vector is left unsimulated.

As `Composed.zk_session_view`, this is for coins fixed independently of the prover; a verifier that chooses them is
covered by the challenger's commitment at `Hello`, before the prover's first message (`coin_tree.rs`), whose binding is
`keyed-cr`, which no statement here models yet.

For an untyped statement (`ht`) whose Δ and regions avoid the mask slot (`avoidsMask`, a named hypothesis until the
verifier's D11 refusal in `Net.ofRows` discharges it), and for any prover of Ligerito's levels ≥ 1 (`LigProver`), as
`Composed.zk_session_view`. -/
theorem zk_session_view_all (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) (ht : I.tags.typed = false)
    (hdm : ∀ (S : Finset (Fin n)) (y : DrawSetupZK I mPts S (dj S)), avoidsMask y.x.st = true)
    (hKey : Assumptions.Zk.HashDerivedKeyHm96) :
    ∀ (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZK I mPts S (dj S)),
      ZkViewAll (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st) ((2 : ℝ≥0∞) ^ (-193 : ℤ)) :=
  fun S R y => zkViewAll_of_ok _ (coinedOf_setup (Reg := Reg) P I dj y₀ ht hdm S R y).2.2.2.2.2 hKey

end FlockSoundness.Discharge.ZkEveryCoin
