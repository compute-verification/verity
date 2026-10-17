import Proofs.Flock.Claims.KS.Refit

/-!
# Flock's KS at a raised `N₀`

The sampled extractor runs a session's `E^rep` at the inner `k′ = ⌈4·N₀′/e⌉` (`Sampling.Deploy.innerK`). Where `N₀′`
only bounds the session's `N₀`, that is Flock's `E^rep` at the record with `N₀` raised (`Flock.Deploy.raise`). KS's
proof reads `N₀` monotonically but in one place: one wait's bound needs the word no longer than `N₀`, `k ≥ 1` and
`N₀/(e·k) ≤ 1/4` (`oneWait_of_tableBound`, `rate_le`), KS needs `k ≥ 8` (`KS_of_oneWait`), and `κ` at `k′` is the
sampled `κ_F`. The exception is the weight: `TableBoundθ` asks the escapes to fit the weight `C` with the conflict
fork's event of weight `k` (`Escapes.FitsFork C k`), at a multiplicity it leaves open, so at `k′ > k` the same weight
need not hold it. Scaled with `k`, it does (`Escapes.FitsFork.raise`, `tableBoundθAt_raise`):

* `FlockDeploy.KS.zk_KS_raise`: KS at `deployZk` raised to `N₀′ ≥ N₀` and a weight `C′` with `C·k′ ≤ C′·k` and
  `k′ ≤ 2C′`, from `zk_KS_of_acc`'s premises.

The old masked escapes (`oldEscZ`, at multiplicity 2) need no scaling: they leave level 0's `2·2^{d₀}·Q₀` of the
weight unused, so they fit the same weight with the fork event of weight `k′` up to `4·N₀/e + 2·2^{d₀}·Q₀`:

* `FlockDeploy.KS.oldEscZ_fitsFork_zk_le`: at any such `k′`;
* `FlockDeploy.KS.oldEscZ_fitsFork_zk_raise`: at the raised `k′`, for `N₀′ ≤ N₀ + 2^{d₀}·Q₀/2`;
* `FlockDeploy.KS.oldEscZ_fitsFork_zk_raise_len`: so for `N₀′ ≤ N₀ + 109·2^{d₀}`, as `Q₀ ≥ 218`; at one table,
  `N₀′ ≤ 110·N₀`.
-/

namespace Claims.Flock

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **The record raised**: `N₀` raised to `N₀′`, and the weight `C′`. -/
def Deploy.raise (D : Deploy Coin Col T) (N₀' C' : Msg → ℝ) : Deploy Coin Col T :=
  { D with N₀ := N₀', weight := C' }

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem Deploy.k_le_raise (D : Deploy Coin Col T) (N₀' C' : Msg → ℝ) (x : Msg) (h : D.N₀ x ≤ N₀' x) :
    D.k x ≤ (D.raise N₀' C').k x := by
  unfold Deploy.k Deploy.raise
  exact Nat.ceil_mono (div_le_div_of_nonneg_right (by linarith) (Real.exp_pos 1).le)

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **Escapes that fit `C` with a fork event of weight `c > 0` fit `C′` with one of weight `c′ ≥ c`**, when
`C·c′ ≤ C′·c`. -/
theorem _root_.Claims.Escapes.FitsFork.raise {Ω : Type} {H : Msg → Msg} {V : Verifier Coin} {P : Prover Ω} {x : Msg}
    {e : Escapes H V P x} {C C' c c' : ℝ} (h : e.FitsFork C c) (hc : 0 < c) (hcc : c ≤ c') (hC : C * c' ≤ C' * c) :
    e.FitsFork C' c' := by
  obtain ⟨mult, hm, hmc, hβ⟩ := h
  refine ⟨mult, hm, ?_, hβ⟩
  have hs : 0 ≤ (mult : ℝ) * ∑ i, e.c i := mul_nonneg (Nat.cast_nonneg _) (Finset.sum_nonneg fun i _ => e.c_nonneg i)
  have h1 : c * (mult * ∑ i, e.c i) ≤ c' * (mult * ∑ i, e.c i) := mul_le_mul_of_nonneg_right hcc hs
  have h2 : c' * (mult * (c + ∑ i, e.c i)) ≤ c' * (2 * C) := mul_le_mul_of_nonneg_left hmc (hc.le.trans hcc)
  refine le_of_mul_le_mul_left ?_ hc
  nlinarith

/-- **`TableBoundθ` at a statement, raised**: the same escapes, refit to the raised `k′` at a weight `C′` with
`C·k′ ≤ C′·k`, where `k > 0`. -/
theorem tableBoundθAt_raise (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop)
    (N₀' C' : Msg → ℝ) (x : Msg) (h : TableBoundθAt H D N Ok x) (hk : 0 < D.k x) (hN : D.N₀ x ≤ N₀' x)
    (hC : D.weight x * (D.raise N₀' C').k x ≤ C' x * D.k x) : TableBoundθAt H (D.raise N₀' C') N Ok x := by
  intro Ω _ _ P
  obtain ⟨e, hfit, hb⟩ := h Ω P
  exact ⟨e, hfit.raise (by exact_mod_cast hk) (by exact_mod_cast D.k_le_raise N₀' C' x hN) hC, hb⟩

end Claims.Flock

namespace FlockDeploy.KS

open Flock

/-- **The deployed masked (`--zk`) verifier: KS at a raised `N₀`**, at SHA-512: at `N₀′ ≥ N₀` and a weight `C′` that
scales the deployed one with `k` (`C·k′ ≤ C′·k`) and covers `k′` (`k′ ≤ 2C′`), from the premises of KS at the record
itself (`zk_KS_of_acc`). -/
theorem zk_KS_raise (rel : Readable.Parsed → Readable.Relation)
    (hone : ∀ x, (∃ s, (deployZk rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployZk rel).V x)
    (h : ∀ x s ins spec sts, ¬ Claims.Flock.NoAcc sha512 (V true) x → parse true rel x = some s →
      Claims.Flock.InScope true s → setupOf true x = some (ins, spec, sts) →
      Claims.Flock.TableBoundθAt sha512 (deployZk rel) (wordLen true) (Claims.Flock.orZero (wide true)) x)
    (N₀' C' : Claims.Msg → ℝ) (hN : ∀ x, N₀ true x ≤ N₀' x)
    (hC : ∀ x, weight true x * ((deployZk rel).raise N₀' C').k x ≤ C' x * (deployZk rel).k x ∧
      (((deployZk rel).raise N₀' C').k x : ℝ) ≤ 2 * C' x) :
    Claims.Flock.KS sha512 ((deployZk rel).raise N₀' C') := by
  have hkle : ∀ x, (deployZk rel).k x ≤ ((deployZk rel).raise N₀' C').k x :=
    fun x => (deployZk rel).k_le_raise N₀' C' x (hN x)
  have htab : Claims.Flock.TableBoundθ sha512 ((deployZk rel).raise N₀' C') (wordLen true)
      (Claims.Flock.orZero (wide true)) := by
    refine (Claims.Flock.tableBoundθ_iff _ _ _ _).2 fun x => ?_
    rcases k_range_zk rel x with hk | hno
    · exact Claims.Flock.tableBoundθAt_raise _ _ _ _ N₀' C' x (zk_tableBoundθ_of_acc rel _ h x) (by omega) (hN x)
        (hC x).1
    · exact Claims.Flock.tableBoundθAt_of_noAcc _ _ _ _ x hno (hC x).2 (εtable_nonneg true x)
  have hsp : ∀ x, Claims.Flock.Splits ((deployZk rel).raise N₀' C') x ∨ Claims.Flock.NoAcc sha512 (V true) x :=
    fun x => (hone x).imp_left fun ⟨_, hs, h1⟩ => splits_of_one ((deployZk rel).raise N₀' C') rfl rfl hs h1
  exact Claims.Flock.KS_of_oneWait sha512 _ (fun _ => cap_pos)
    (fun x => (k_range_zk rel x).elim (fun hk => (hsp x).imp (fun h => ⟨hk.trans (hkle x), h⟩) id) .inr)
    (fun x => εtable_nonneg true x)
    (Claims.Flock.oneWait_of_tableBound sha512 _ (wordLen true) (fun x => (wordLen_le_N₀ true x).trans (hN x))
      (fun x => εtable_nonneg true x) (V_k_pos true)
      (fun x => (k_range_zk rel x).imp (fun hk => by have := hkle x; omega) id) hone _ (fun _ => .inl rfl)
      (Claims.Flock.colsOk_orZero _ _ _ _ (zk_colsOk sha512 rel)) (zk_conflictTied rel) htab)

/-- **The old masked escapes fit the masked weight with a fork event of weight `k′ ≤ 4·N₀/e + 2·2^{d₀}·Q₀`**, under
`oldEscZ_fitsFork_zk`'s hypotheses: at multiplicity 2 the fork events leave level 0's `2·2^{d₀}·Q₀` of the weight
unused. -/
theorem oldEscZ_fitsFork_zk_le (rel : Readable.Parsed → Readable.Relation) {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) {spec : SessionSpec} {st : Stmt} (h0 : (tables true x).head? = some (spec, st))
    {sch : FlockSoundness.Accounting.Schedule}
    (hsch : sch = FlockSoundness.Accounting.m1Schedule st.m) (Np : ℕ)
    (hNp : Np ≤ Claims.Flock.Numbers.padDomain (Zk.PadLayout.new st.m st.c.kLog (Claims.Flock.zkSched st.m).1.k0))
    (j : Claims.Flock.IZ sch.levels.length → Fin ((deployZk rel).V.k x)) (hj : Function.Injective j)
    (ev : Claims.Flock.IZ sch.levels.length → Ω → (Fin ((deployZk rel).V.k x) → Coin) →
      (Fin ((deployZk rel).V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin ((deployZk rel).V.k x) → Coin) → Prop)
    (tied : ∀ i ω s s', ev i ω s s' → Claims.TiedColl sha512
      ((Claims.run sha512 P (deployZk rel).V x ω s).vLog ++
        (Claims.run sha512 P (deployZk rel).V x ω (Claims.forkCoins (j i) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (deployZk rel).V x ω s).vLog)
    (k' : ℝ) (hk' : k' ≤ 4 * N₀ true x / Real.exp 1 +
      2 * (2 ^ (sch.levels.headD default).logLen * (sch.levels.headD default).queries)) :
    (Claims.Flock.oldEscZ sha512 (deployZk rel).V P x sch Np j ev ev₁ tied tied₁).FitsFork
      ((deployZk rel).weight x) k' := by
  refine ⟨2, fun r => ?_, ?_, ?_⟩
  · have hc : (Finset.univ.filter fun i : Claims.Flock.IZ sch.levels.length => j i = r).card ≤ 1 :=
      Finset.card_le_one.2 fun a ha b hb => by
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
        exact hj (ha.trans hb.symm)
    have hr : (if r.val = 0 then 1 else 0) ≤ 1 := by split <;> omega
    exact Nat.add_le_add hc hr
  · set f : FlockSoundness.Accounting.Level → ℝ := fun l => 2 ^ l.logLen * l.queries with hf
    have hsum : ∑ i : Claims.Flock.IZ sch.levels.length, Claims.Flock.cZ sch Np i =
        2 * ∑ ℓ : Fin sch.levels.length, f (sch.levels.getD (ℓ.val + 1) default) + 2 * ((Np : ℝ) * Np) + 2 := by
      show ∑ i : (Fin 2 × Fin sch.levels.length) ⊕ (Fin 2 ⊕ Fin 2), Claims.Flock.cZ sch Np i = _
      rw [Fintype.sum_sum_type, Fintype.sum_sum_type, Fintype.sum_prod_type]
      simp only [Claims.Flock.cZ, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
        Nat.cast_ofNat, mul_one]
      rw [add_assoc]
      rfl
    have hd : (default : FlockSoundness.Accounting.Level).queries = 0 := rfl
    have hhead := Claims.Flock.head_add_sum_shift f (by simp [hf, hd]) sch.levels
    set pd := Claims.Flock.Numbers.padDomain (Zk.PadLayout.new st.m st.c.kLog (Claims.Flock.zkSched st.m).1.k0)
    have htw : tableWeight true st = Claims.Flock.Numbers.weightOf sch pd := by
      simp only [tableWeight, ↓reduceIte, hsch, pd]
    have hNp2 : (Np : ℝ) * Np ≤ (pd : ℝ) ^ 2 := by
      have := Nat.mul_le_mul hNp hNp
      rw [sq]
      exact_mod_cast this
    have hlen : tableLen true st = 2 ^ (Claims.Flock.zkLevel0 st.m).d := rfl
    have hfits := kFits_all true st.m
    simp only [level0, wSched, ↓reduceIte, ← hsch] at hfits
    have hpow : (2 : ℝ) ^ (Claims.Flock.zkLevel0 st.m).d ≤ 2 ^ (sch.levels.headD default).logLen :=
      pow_le_pow_right₀ (by norm_num) hfits
    have hdiv : 4 * (2 : ℝ) ^ (Claims.Flock.zkLevel0 st.m).d / Real.exp 1 ≤
        4 * 2 ^ (sch.levels.headD default).logLen / Real.exp 1 := by
      gcongr
    have hsplit := weight_split h0
    rw [hlen] at hsplit
    have e : 4 * (N₀ true x - 2 ^ (Claims.Flock.zkLevel0 st.m).d) / Real.exp 1 =
        4 * N₀ true x / Real.exp 1 - 4 * 2 ^ (Claims.Flock.zkLevel0 st.m).d / Real.exp 1 := by ring
    rw [e] at hsplit
    show ((2 : ℕ) : ℝ) * (k' + ∑ i : Claims.Flock.IZ sch.levels.length, Claims.Flock.cZ sch Np i) ≤
      2 * weight true x
    rw [hsum]
    rw [htw, Claims.Flock.Numbers.weightOf, weightNat_cast] at hsplit
    simp only [hf] at hhead hk'
    push_cast at hsplit ⊢
    linarith
  · show ∑ _l : Unit, (1 : ℝ) ≤ (V true).k x
    rw [Fintype.sum_unique]
    exact_mod_cast V_k_pos true x

/-- **The old masked escapes fit the masked weight at a raised `N₀`**: at `N₀′ ≤ N₀ + 2^{d₀}·Q₀/2` the raised `k′`
is below `4·N₀/e + 2·2^{d₀}·Q₀`, as `2^{d₀}·Q₀ ≥ 1` (`oldEscZ_fitsFork_zk_le`), so the same escapes fit the same
weight with the fork event of weight `k′`, whatever the raised record's weight `C′`. -/
theorem oldEscZ_fitsFork_zk_raise (rel : Readable.Parsed → Readable.Relation) {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) {spec : SessionSpec} {st : Stmt} (h0 : (tables true x).head? = some (spec, st))
    (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) {sch : FlockSoundness.Accounting.Schedule}
    (hsch : sch = FlockSoundness.Accounting.m1Schedule st.m) (Np : ℕ)
    (hNp : Np ≤ Claims.Flock.Numbers.padDomain (Zk.PadLayout.new st.m st.c.kLog (Claims.Flock.zkSched st.m).1.k0))
    (j : Claims.Flock.IZ sch.levels.length → Fin ((deployZk rel).V.k x)) (hj : Function.Injective j)
    (ev : Claims.Flock.IZ sch.levels.length → Ω → (Fin ((deployZk rel).V.k x) → Coin) →
      (Fin ((deployZk rel).V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin ((deployZk rel).V.k x) → Coin) → Prop)
    (tied : ∀ i ω s s', ev i ω s s' → Claims.TiedColl sha512
      ((Claims.run sha512 P (deployZk rel).V x ω s).vLog ++
        (Claims.run sha512 P (deployZk rel).V x ω (Claims.forkCoins (j i) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (deployZk rel).V x ω s).vLog)
    (N₀' C' : Claims.Msg → ℝ)
    (hN : N₀' x ≤ N₀ true x + 2 ^ (sch.levels.headD default).logLen * (sch.levels.headD default).queries / 2) :
    (Claims.Flock.oldEscZ sha512 (deployZk rel).V P x sch Np j ev ev₁ tied tied₁).FitsFork
      ((deployZk rel).weight x) (((deployZk rel).raise N₀' C').k x) := by
  refine oldEscZ_fitsFork_zk_le rel P x h0 hsch Np hNp j hj ev ev₁ tied tied₁ _ ?_
  set q : ℝ := 2 ^ (sch.levels.headD default).logLen * (sch.levels.headD default).queries with hq
  have hq1 : 1 ≤ q := by
    have hq0 : 1 ≤ (sch.levels.headD default).queries := by
      have := m1_queries₀_pos ⟨st.m - 25, by omega⟩
      simp only [Nat.sub_add_cancel h25] at this
      rw [hsch]
      exact this
    have h1 : (1 : ℝ) ≤ 2 ^ (sch.levels.headD default).logLen := one_le_pow₀ (by norm_num)
    have h2 : (1 : ℝ) ≤ (sch.levels.headD default).queries := by exact_mod_cast hq0
    nlinarith
  have he := Real.exp_pos 1
  have he2 : 2 ≤ Real.exp 1 := by linarith [Real.add_one_le_exp (1 : ℝ)]
  have hb : 0 ≤ 4 * (N₀ true x + q / 2) / Real.exp 1 :=
    div_nonneg (mul_nonneg (by norm_num) (by linarith [N₀_nonneg true x])) he.le
  have hceil : (⌈4 * N₀' x / Real.exp 1⌉₊ : ℝ) < 4 * (N₀ true x + q / 2) / Real.exp 1 + 1 :=
    (Nat.cast_le.2 (Nat.ceil_mono (div_le_div_of_nonneg_right (by linarith) he.le))).trans_lt
      (Nat.ceil_lt_add_one hb)
  have hk : (((deployZk rel).raise N₀' C').k x : ℝ) = (⌈4 * N₀' x / Real.exp 1⌉₊ : ℝ) := rfl
  have hsplit : 4 * (N₀ true x + q / 2) / Real.exp 1 = 4 * N₀ true x / Real.exp 1 + 2 * q / Real.exp 1 := by ring
  have hqe : 2 * q / Real.exp 1 ≤ q := by
    rw [div_le_iff₀ he]
    nlinarith
  rw [hk]
  linarith

/-- C-Flock ZK's level 0 reads at least 218 positions at every `m` it serves. -/
theorem m1_queries₀_ge : ∀ m : Fin 11,
    218 ≤ ((FlockSoundness.Accounting.m1Schedule (m.val + 25)).levels.headD default).queries := by
  decide +kernel

/-- **The old masked escapes fit the masked weight at `N₀′ ≤ N₀ + 109·2^{d₀}`**, table 0's length `2^{d₀}`: level 0
reads at least 218 positions (`m1_queries₀_ge`) of a word at least table 0's length (`kFits_all`), so this is within
`oldEscZ_fitsFork_zk_raise`'s `N₀ + 2^{d₀}·Q₀/2`. At one table, `N₀′ ≤ 110·N₀`. -/
theorem oldEscZ_fitsFork_zk_raise_len (rel : Readable.Parsed → Readable.Relation) {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) {spec : SessionSpec} {st : Stmt} (h0 : (tables true x).head? = some (spec, st))
    (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) {sch : FlockSoundness.Accounting.Schedule}
    (hsch : sch = FlockSoundness.Accounting.m1Schedule st.m) (Np : ℕ)
    (hNp : Np ≤ Claims.Flock.Numbers.padDomain (Zk.PadLayout.new st.m st.c.kLog (Claims.Flock.zkSched st.m).1.k0))
    (j : Claims.Flock.IZ sch.levels.length → Fin ((deployZk rel).V.k x)) (hj : Function.Injective j)
    (ev : Claims.Flock.IZ sch.levels.length → Ω → (Fin ((deployZk rel).V.k x) → Coin) →
      (Fin ((deployZk rel).V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin ((deployZk rel).V.k x) → Coin) → Prop)
    (tied : ∀ i ω s s', ev i ω s s' → Claims.TiedColl sha512
      ((Claims.run sha512 P (deployZk rel).V x ω s).vLog ++
        (Claims.run sha512 P (deployZk rel).V x ω (Claims.forkCoins (j i) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (deployZk rel).V x ω s).vLog)
    (N₀' C' : Claims.Msg → ℝ) (hN : N₀' x ≤ N₀ true x + 109 * tableLen true st) :
    (Claims.Flock.oldEscZ sha512 (deployZk rel).V P x sch Np j ev ev₁ tied tied₁).FitsFork
      ((deployZk rel).weight x) (((deployZk rel).raise N₀' C').k x) := by
  refine oldEscZ_fitsFork_zk_raise rel P x h0 h25 h35 hsch Np hNp j hj ev ev₁ tied tied₁ N₀' C' (hN.trans ?_)
  have hq : (218 : ℝ) ≤ (sch.levels.headD default).queries := by
    have := m1_queries₀_ge ⟨st.m - 25, by omega⟩
    simp only [Nat.sub_add_cancel h25] at this
    rw [hsch]
    exact_mod_cast this
  have hfits := kFits_all true st.m
  simp only [level0, wSched, ↓reduceIte, ← hsch] at hfits
  have hpow : (2 : ℝ) ^ (Claims.Flock.zkLevel0 st.m).d ≤ 2 ^ (sch.levels.headD default).logLen :=
    pow_le_pow_right₀ (by norm_num) hfits
  have hlen : tableLen true st = 2 ^ (Claims.Flock.zkLevel0 st.m).d := rfl
  have h2 : (0 : ℝ) ≤ 2 ^ (Claims.Flock.zkLevel0 st.m).d := by positivity
  rw [hlen]
  nlinarith

end FlockDeploy.KS
