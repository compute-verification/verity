import Proofs.Accounting.Compose.ServiceBudget
import Proofs.Accounting.Compose.ServiceCovers
import Proofs.Accounting.Headline.Luck

/-!
# `CircuitWork`: the circuit's work under live coins

* **The luck per stage** (`luckAt_of_perTile`): each credited tile's `tileGame … .folded` is `Pinned` (its check is
  `same`), so (H-μ) through Lemma B at fibre 1 (`operandLuck_of_pointMass`) and B′ through `LuckOfReceipt` give
  `Pr[folded ∧ same] ≤ (μ + (1 − μ)·r_B)·Pr[folded]` (`r_B ≤ 1` by `Scheme.LuckProved`), and `GuessLuckAt` gives the
  whole tile's at its guess bound; a tile of no credited kind is held to `1`.
* **Two events**: `ServiceBudget` at `shareCone` and `budget_tail` at `t = ln(1/δ_s)` bound the accepted runs with
  `a·S_F + b·S_0 ≥ Λ·ln(1/δ_s)` by `δ_s` (`S_F`, `S_0` the folded and whole shares at `shareCone`), and
  `DrawnRightOff` gives one event of probability `δ_bind + δ_ks` off which every accepted drawn tile is right, so `Of`
  gives the play's ideal game's pass, which is the share game's.
* **The words** (`work_of_bounds`): off both, `BlockFixed` and `ServiceCovers` give the good tiles' words at least
  `(1 − (c_F + c₀))·N_w`, `c_F`, `c₀` the folded and whole shares at `tileCone`, and the words not skipped whole at
  least `(1 − S_0)·N_w`, since a tile's words are at most `N_w` times its `shareCone`; `Σ_t tileWords ≥ N_w`
  (`wordsOf_le_sum`: each credited word is in its unit's tile, under `RegisteredCircuit.Wf` and `units ≤ N_p`). `c_F + c₀` is
  `S_F + S_0` plus the cones `Z_F + Z_0 ≤ D₀` of tiles of no words, which `WorkAt` takes in the cheaper share
  (`workAt_of`).
-/

namespace Accounting.SecurityProofs.Compose.WorkProof

open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-! ## The luck per stage -/

theorem pos_ite_iff (P : Prop) [Decidable P] : (0 : ℝ) < (if P then 1 else 0) ↔ P := by
  by_cases h : P
  · rw [ite_eq_left h]
    exact ⟨fun _ => h, fun _ => one_pos⟩
  · rw [ite_eq_right h]
    exact ⟨fun h' => absurd h' (lt_irrefl 0), fun h' => absurd h' h⟩

/-- A tile's stage game in mode `s` lucky at `q` keeps the tile's checked values in mode `s` at `q`. -/
theorem tileLuck_of {C V : Type} [Fintype C] (mode : C → V → Mode) (right same : C → V → Prop) (s : Mode)
    (t : V) (q : ℝ) (h : RoundGame.LuckAt q (tileGame mode right same s t).game) :
    prob (fun c => mode c t = s ∧ same c t) ≤ q * prob (fun c => mode c t = s) := by
  obtain ⟨-, h2, -⟩ := h
  have hf : (fun c => 0 < (if mode c t = s then (1 : ℝ) else 0)) = fun c => mode c t = s :=
    funext fun c => propext (pos_ite_iff _)
  calc prob (fun c => mode c t = s ∧ same c t)
      ≤ prob (fun c => same c t ∧ 0 < (if mode c t = s then (1 : ℝ) else 0)) :=
        prob_mono fun c hc => ⟨hc.2, (pos_ite_iff _).mpr hc.1⟩
    _ ≤ q * prob (fun c => 0 < (if mode c t = s then (1 : ℝ) else 0)) := h2
    _ = q * prob (fun c => mode c t = s) := congrArg (fun P : C → Prop => q * prob P) hf

/-- **The four claims give the stages' luck** at `tileLuck` folded and `tileLuck0` whole. -/
theorem luckAt_of_perTile {ι K C : Type} [Fintype C] (Sc : Scheme K) (Cd : RegisteredCircuit ι) {Np : ℕ}
    (hLP : Sc.LuckProved) : ∀ {M : ℕ} (W : SplitStages C (Fin Np) M),
    PerTile Sc Cd W (fun k G _ => OperandHMu (Sc.mu k) G) →
    PerTile Sc Cd W (fun k G _ => ReceiptLuckAt (Sc.rB k) G) →
    PerTile Sc Cd W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0) →
    W.LuckAt (tileLuck Sc Cd Np) (tileLuck0 Sc Cd Np) (tileStage Cd Np)
  | _, .done, _, _, _ => trivial
  | _, .stage mode right same next, h1, h2, h3 => by
    obtain ⟨h1, h1n⟩ := h1
    obtain ⟨h2, h2n⟩ := h2
    obtain ⟨h3, h3n⟩ := h3
    refine ⟨fun t ht => ?_, fun c => luckAt_of_perTile Sc Cd hLP (next c) (h1n c) (h2n c) (h3n c)⟩
    have hone : ∀ s : Mode, prob (fun c => mode c t = s ∧ same c t) ≤ 1 * prob (fun c => mode c t = s) :=
      fun s => by
        rw [one_mul]
        exact prob_mono fun c hc => hc.1
    rcases hk : tileKind Cd Np t with _ | k
    · have e1 : tileLuck Sc Cd Np t = 1 := by simp [tileLuck, hk, Option.elim]
      have e0 : tileLuck0 Sc Cd Np t = 1 := by simp [tileLuck0, hk, Option.elim]
      rw [e1, e0]
      exact ⟨hone _, hone _⟩
    · cases hc : Sc.credited k with
      | false =>
        have e1 : tileLuck Sc Cd Np t = 1 := by simp [tileLuck, hk, hc, Option.elim]
        have e0 : tileLuck0 Sc Cd Np t = 1 := by simp [tileLuck0, hk, hc, Option.elim]
        rw [e1, e0]
        exact ⟨hone _, hone _⟩
      | true =>
        have e1 : tileLuck Sc Cd Np t = Sc.foldLuck k := by simp [tileLuck, hk, hc, Option.elim]
        have e0 : tileLuck0 Sc Cd Np t = Sc.guess k := by simp [tileLuck0, hk, hc, Option.elim]
        rw [e1, e0]
        have hPin : (tileGame mode right same .folded t).Pinned := ⟨fun _ h => h, fun _ => trivial⟩
        have hL := Accounting.SecurityProofs.TS.LuckOfReceipt (tileGame mode right same .folded t)
          (Sc.mu k) (Sc.rB k) hPin
          (Accounting.SecurityProofs.TS.Headline.operandLuck_of_pointMass (Sc.mu k) _ (h1 t ht k hk hc))
          (h2 t ht k hk hc) (hLP k).2.2.1
        exact ⟨tileLuck_of mode right same .folded t (Sc.foldLuck k) hL,
          tileLuck_of mode right same .whole t (Sc.guess k) (h3 t ht k hk hc)⟩

/-! ## The tiles' words and cones -/

/-- **The tiles' words count every credited word**: a list's elements each in one of `N` tiles (`f i < N`) are at
most the tiles' counts. -/
theorem length_le_sum {Np : ℕ} (f : ℕ → ℕ) (q : ℕ → Bool) : ∀ M : List ℕ, (∀ i ∈ M, f i < Np) →
    (M.filter q).length ≤ ∑ t : Fin Np, ((M.filter fun j => f j == t.val).filter q).length
  | [], _ => by simp
  | i :: M, h => by
    have ih := length_le_sum f q M fun j hj => h j (List.mem_cons_of_mem i hj)
    have hi : f i < Np := h i List.mem_cons_self
    have hsub : ∀ t : Fin Np, ((M.filter fun j => f j == t.val).filter q).length ≤
        (((i :: M).filter fun j => f j == t.val).filter q).length := fun t =>
      (((List.sublist_cons_self i M).filter (fun j => f j == t.val)).filter q).length_le
    cases hq : q i with
    | false =>
      rw [List.filter_cons_of_neg (p := q) (a := i) (l := M) (by simp [hq])]
      exact ih.trans (Finset.sum_le_sum fun t _ => hsub t)
    | true =>
      rw [List.filter_cons_of_pos (p := q) (a := i) (l := M) hq, List.length_cons]
      have hlt : ∑ t : Fin Np, ((M.filter fun j => f j == t.val).filter q).length <
          ∑ t : Fin Np, (((i :: M).filter fun j => f j == t.val).filter q).length := by
        refine Finset.sum_lt_sum (fun t _ => hsub t) ⟨⟨f i, hi⟩, Finset.mem_univ _, ?_⟩
        have hft : (f i == (⟨f i, hi⟩ : Fin Np).val) = true := beq_self_eq_true (f i)
        rw [List.filter_cons_of_pos (p := fun j => f j == (⟨f i, hi⟩ : Fin Np).val) (a := i) (l := M) hft,
          List.filter_cons_of_pos (p := q) (a := i) hq, List.length_cons]
        exact Nat.lt_add_one _
      omega

theorem wordsOf_le_sum {ι K : Type} (Sc : Scheme K) (Cd : RegisteredCircuit ι) {Np : ℕ} (hw : Cd.Wf)
    (hN : Cd.units ≤ Np) : (wordsOf Sc Cd : ℝ) ≤ ∑ t : Fin Np, (tileWords Sc Cd Np t : ℝ) := by
  rw [← Nat.cast_sum]
  exact Nat.cast_le.mpr (length_le_sum Cd.unit (fun i => Sc.credited (kindAt Cd i)) (wordGates Cd)
    fun i hi => Nat.lt_of_lt_of_le (hw.1 i (List.mem_range.mp (List.mem_filter.mp hi).1)).2 hN)

section Cones

variable {ι K : Type} (Sc : Scheme K) (Cd : RegisteredCircuit ι) (Np : ℕ)

theorem tileWeight_nonneg (t : Fin Np) : 0 ≤ tileWeight Sc Cd Np t :=
  div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)

theorem tileCone_nonneg (t : Fin Np) : 0 ≤ tileCone Sc Cd Np t :=
  Finset.sum_nonneg fun t' _ => by
    split_ifs
    · exact tileWeight_nonneg Sc Cd Np t'
    · exact le_rfl

theorem shareCone_nonneg (t : Fin Np) : 0 ≤ shareCone Sc Cd Np t := by
  unfold shareCone
  split_ifs
  · exact le_rfl
  · exact tileCone_nonneg Sc Cd Np t

/-- A tile's cone holds the tile itself. -/
theorem tileWeight_le_tileCone (t : Fin Np) : tileWeight Sc Cd Np t ≤ tileCone Sc Cd Np t := by
  classical
  have hnn : ∀ t' ∈ (Finset.univ : Finset (Fin Np)),
      0 ≤ (if Cd.Reaches t.val t'.val then tileWeight Sc Cd Np t' else 0) := fun t' _ => by
    split_ifs
    · exact tileWeight_nonneg Sc Cd Np t'
    · exact le_rfl
  have hself : (if Cd.Reaches t.val t.val then tileWeight Sc Cd Np t else 0) = tileWeight Sc Cd Np t :=
    ite_eq_left Relation.ReflTransGen.refl
  calc tileWeight Sc Cd Np t = (if Cd.Reaches t.val t.val then tileWeight Sc Cd Np t else 0) := hself.symm
    _ ≤ tileCone Sc Cd Np t := Finset.single_le_sum hnn (Finset.mem_univ t)

/-- **A tile's words are at most `N_w` times its detected weight.** -/
theorem tileWords_le_share (t : Fin Np) :
    (tileWords Sc Cd Np t : ℝ) ≤ shareCone Sc Cd Np t * (wordsOf Sc Cd : ℝ) := by
  unfold shareCone
  split_ifs with h0
  · simp [h0]
  · have hpos : 0 < wordsOf Sc Cd :=
      Nat.lt_of_lt_of_le (Nat.pos_of_ne_zero h0) (CoversProof.tileWords_le_wordsOf Sc Cd Np t)
    have hw : tileWeight Sc Cd Np t * (wordsOf Sc Cd : ℝ) = tileWords Sc Cd Np t :=
      div_mul_cancel₀ _ (Nat.cast_pos.mpr hpos).ne'
    calc (tileWords Sc Cd Np t : ℝ) = tileWeight Sc Cd Np t * (wordsOf Sc Cd : ℝ) := hw.symm
      _ ≤ tileCone Sc Cd Np t * (wordsOf Sc Cd : ℝ) :=
        mul_le_mul_of_nonneg_right (tileWeight_le_tileCone Sc Cd Np t) (Nat.cast_nonneg _)

/-- `tileCone` is `shareCone` plus the cones of tiles of no words. -/
theorem cone_share_split (md : Fin Np → Mode) (m : Mode) :
    (∑ u, if md u = m then tileCone Sc Cd Np u else 0) =
      (∑ u, if md u = m then shareCone Sc Cd Np u else 0) +
        ∑ u, if md u = m then (if tileWords Sc Cd Np u = 0 then tileCone Sc Cd Np u else 0) else 0 := by
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun u _ => ?_
  unfold shareCone
  split_ifs <;> simp

theorem share_nonneg (md : Fin Np → Mode) (m : Mode) :
    0 ≤ ∑ u, if md u = m then shareCone Sc Cd Np u else 0 :=
  Finset.sum_nonneg fun u _ => by
    split_ifs
    · exact shareCone_nonneg Sc Cd Np u
    · exact le_rfl

theorem zero_share_nonneg (md : Fin Np → Mode) (m : Mode) :
    0 ≤ ∑ u, if md u = m then (if tileWords Sc Cd Np u = 0 then tileCone Sc Cd Np u else 0) else 0 :=
  Finset.sum_nonneg fun u _ => by
    split_ifs <;> first | exact tileCone_nonneg Sc Cd Np u | exact le_rfl

theorem zero_part_le (m : Mode) {z : ℝ} (h0 : 0 ≤ z) :
    (if m = .folded then z else 0) + (if m = .whole then z else 0) ≤ z := by
  cases m <;> simp [h0]

/-- The folded and whole cones of tiles of no words are at most `D₀`. -/
theorem zero_shares_le (md : Fin Np → Mode) :
    (∑ u, if md u = .folded then (if tileWords Sc Cd Np u = 0 then tileCone Sc Cd Np u else 0) else 0) +
      (∑ u, if md u = .whole then (if tileWords Sc Cd Np u = 0 then tileCone Sc Cd Np u else 0) else 0) ≤
        zeroCone Sc Cd Np := by
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun u _ => zero_part_le (md u) ?_
  split_ifs
  · exact tileCone_nonneg Sc Cd Np u
  · exact le_rfl

end Cones

/-! ## The work -/

theorem sum_ite_split {N : ℕ} (c : Fin N → Prop) [DecidablePred c] (f : Fin N → ℝ) :
    (∑ j, if c j then f j else 0) + (∑ j, if c j then 0 else f j) = ∑ j, f j := by
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  split_ifs <;> simp

/-- The good tiles' words are at least `(1 − D)·N_w` when the rest are at most `D·N_w`. -/
theorem fixed_bound {N : ℕ} (c : Fin N → Prop) [DecidablePred c] (f : Fin N → ℝ) (Nw D fixed : ℝ)
    (hN : Nw ≤ ∑ j, f j) (hcov : (∑ j, if c j then 0 else f j) ≤ D * Nw)
    (hfix : (∑ j, if c j then f j else 0) ≤ fixed) : (1 - D) * Nw ≤ fixed := by
  have h := sum_ite_split c f
  calc (1 - D) * Nw = Nw - D * Nw := by ring
    _ ≤ fixed := by linarith

/-- The words not skipped whole are at least `(1 − S_0)·N_w`, `S_0` the whole share at weights `g ≥ f/N_w`. -/
theorem folded_bound {N : ℕ} (md : Fin N → Mode) (f g : Fin N → ℝ) (Nw fd : ℝ) (hN : Nw ≤ ∑ j, f j)
    (hfg : ∀ j, f j ≤ g j * Nw) (hfold : (∑ j, if md j = .whole then 0 else f j) ≤ fd) :
    (1 - ∑ j, if md j = .whole then g j else 0) * Nw ≤ fd := by
  have hsplit : (∑ j, if md j = .whole then f j else 0) + (∑ j, if md j = .whole then 0 else f j) =
      ∑ j, f j := sum_ite_split (fun j => md j = .whole) f
  have hw : (∑ j, if md j = .whole then f j else 0) ≤ (∑ j, if md j = .whole then g j else 0) * Nw := by
    rw [Finset.sum_mul]
    refine Finset.sum_le_sum fun j _ => ?_
    split_ifs
    · exact hfg j
    · simp
  calc (1 - ∑ j, if md j = .whole then g j else 0) * Nw
      = Nw - (∑ j, if md j = .whole then g j else 0) * Nw := by ring
    _ ≤ fd := by linarith

/-- **The zero-word cones in the cheaper share**: `d_F = S_F + Z`, `d₀ = S_0` for `a ≤ b`, else `d_F = S_F`,
`d₀ = S_0 + Z`, `Z = Z_F + Z_0 ≤ D₀`. -/
theorem workAt_of {Nw a b Λl D0 fixed fd SF S0 ZF Z0 D : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hNw : 0 ≤ Nw)
    (hSF : 0 ≤ SF) (hS0 : 0 ≤ S0) (hZF : 0 ≤ ZF) (hZ0 : 0 ≤ Z0) (hZD : ZF + Z0 ≤ D0)
    (hD : D = SF + ZF + (S0 + Z0)) (hfix : (1 - D) * Nw ≤ fixed) (hfold : (1 - S0) * Nw ≤ fd)
    (hlt : a * SF + b * S0 < Λl) : WorkAt Nw a b (Λl + min a b * D0) fixed fd := by
  subst hD
  have hZa := mul_le_mul_of_nonneg_left hZD ha
  have hZb := mul_le_mul_of_nonneg_left hZD hb
  rcases le_total a b with hab | hab
  · refine ⟨SF + (ZF + Z0), S0, by linarith, hS0, by linarith, hfold, ?_⟩
    rw [min_eq_left hab]
    linarith
  · refine ⟨SF, S0 + (ZF + Z0), hSF, by linarith, by linarith,
      by linarith [mul_nonneg (add_nonneg hZF hZ0) hNw], ?_⟩
    rw [min_eq_right hab]
    linarith

/-- **One accepted run's work**, at its modes `md` and its good tiles `good`, from `Covers`, `BlockFixed` at the
tiles' words and the two games' deficiencies. -/
theorem work_of_bounds {ι K : Type} (Sc : Scheme K) (Cd : RegisteredCircuit ι) {Np : ℕ} (md : Fin Np → Mode)
    (good : Fin Np → Prop) [DecidablePred good] {a b Λl D fixed fd : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hN : (wordsOf Sc Cd : ℝ) ≤ ∑ j, (tileWords Sc Cd Np j : ℝ))
    (hD : D = 1 * (∑ u, if md u = .folded then tileCone Sc Cd Np u else 0) +
      1 * (∑ u, if md u = .whole then tileCone Sc Cd Np u else 0) + 0)
    (hcov : (∑ j, if good j then (0 : ℝ) else (tileWords Sc Cd Np j : ℝ)) ≤ D * (wordsOf Sc Cd : ℝ))
    (hfix : (∑ j, if good j then (tileWords Sc Cd Np j : ℝ) else 0) ≤ fixed)
    (hfold : (∑ j, if md j = .whole then (0 : ℝ) else (tileWords Sc Cd Np j : ℝ)) ≤ fd)
    (hlt : a * (∑ u, if md u = .folded then shareCone Sc Cd Np u else 0) +
      b * (∑ u, if md u = .whole then shareCone Sc Cd Np u else 0) + 0 < Λl) :
    WorkAt (wordsOf Sc Cd : ℝ) a b (Λl + min a b * zeroCone Sc Cd Np) fixed fd := by
  have hfixB := fixed_bound good (fun j => (tileWords Sc Cd Np j : ℝ)) _ D fixed hN hcov hfix
  have hfoldB := folded_bound md (fun j => (tileWords Sc Cd Np j : ℝ)) (shareCone Sc Cd Np) _ fd hN
    (tileWords_le_share Sc Cd Np) hfold
  have hFs := cone_share_split Sc Cd Np md .folded
  have hWs := cone_share_split Sc Cd Np md .whole
  exact workAt_of ha hb (Nat.cast_nonneg _) (share_nonneg Sc Cd Np md .folded) (share_nonneg Sc Cd Np md .whole)
    (zero_share_nonneg Sc Cd Np md .folded) (zero_share_nonneg Sc Cd Np md .whole) (zero_shares_le Sc Cd Np md)
    (by linear_combination hD + hFs + hWs) hfixB hfoldB (by linarith)

/-! ## The window's proofs' layer -/

section Layer

variable {ι C Ξ V : Type} (Cd : RegisteredCircuit ι) {Np Kd : ℕ} (hr : 0 < Cd.rounds) (hK : Kd ≤ Np)
  (pr : ProofRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np V)

theorem pos_of_mem (w : Fin 1) (t : Fin Np) (ℓ : ℕ) (h : ℓ ∈ Cd.slots t.val) :
    ∃ k, (Cd.layer hr hK pr).pos w t k = ℓ := by
  obtain ⟨k, hk⟩ := List.mem_iff_getElem?.mp h
  refine ⟨k, ?_⟩
  show (Cd.slots t.val)[k]?.getD ((Cd.slots t.val).headD Cd.dag.length) = ℓ
  rw [hk, Option.getD_some]

theorem writerReads : (Cd.layer hr hK pr).WriterReads := by
  intro ℓ u hu
  rw [CoversProof.src_eq] at hu
  split at hu
  · rename_i h
    obtain rfl := Option.some.inj hu
    have hm : ℓ ∈ Cd.slots (Cd.unit ℓ) :=
      List.mem_append_left _ ((CoversProof.mem_unitGates_iff Cd _ _).mpr ⟨h.1, rfl⟩)
    exact pos_of_mem Cd hr hK pr 0 ⟨Cd.unit ℓ, h.2.2⟩ ℓ hm
  · cases hu

end Layer

/-- `CircuitWork`: off `DrawnRightOff`'s event and the budget's tail, `work_of_bounds`. -/
theorem circuitWork : Accounting.Guarantees.Compose.CircuitWork := by
  intro ι K T L U C Ξ V _ _ _ _ _ _ _ Sc Cd Np Kd hN hr hK W Pl pr folded ρ p p0 a b wmax Λ δs δbind δks hS hρ
    hρN hwmax hlk ha hb hΛ hSR hG hOf hMoved hJS hPins hMu hRB hGs hUF
  obtain ⟨hwf, -, hδs, hLP, hlc⟩ := hS
  obtain ⟨E, hE, hEoff⟩ := Accounting.SecurityProofs.TS.DrawnRightOff C (Equiv.Perm (Fin Np) × Ξ) 1 Cd.rounds Kd
    (Cd.layer hr hK pr) δbind δks hMoved hJS hPins
  have hBud : RoundGame.BudgetAt Λ (serviceGame (Ξ := Ξ) (tileStage Cd Np) W (shareCone Sc Cd Np) a b Kd) :=
    Budget.serviceBudget (tileStage Cd Np) W (shareCone Sc Cd Np) (tileLuck Sc Cd Np) (tileLuck0 Sc Cd Np) Kd ρ p
      p0 a b wmax Λ (shareCone_nonneg Sc Cd Np) hρ hρN hwmax hlk (luckAt_of_perTile Sc Cd hLP W hMu hRB hGs) ha
      hb hSR
  have hTail := Accounting.SecurityProofs.TS.budget_tail _ Λ hΛ hBud (Real.log (1 / δs))
  have hexp : Real.exp (-Real.log (1 / δs)) = δs := by
    rw [Real.exp_neg, Real.exp_log (one_div_pos.mpr hδs), one_div, inv_inv]
  rw [hexp] at hTail
  have hCov : (Cd.layer hr hK pr).Covers Pl (tileMode Cd W) (fun _ t => (tileWords Sc Cd Np t : ℝ))
      (wordsOf Sc Cd : ℝ) :=
    CoversProof.serviceCovers Sc Cd hr hK (W.joint (tileStage Cd Np)) Pl pr hwf hlc hG
  have hFix := hUF (writerReads Cd hr hK pr)
  have hCount := wordsOf_le_sum Sc Cd hwf hN
  have hsub : ∀ x, (Pl.accepts x ∧
      ¬ WorkAt (wordsOf Sc Cd : ℝ) a b (rOf Sc Cd Np a b Λ δs) (Pl.run x).fixed (folded x)) →
      E x ∨ ((serviceGame (Ξ := Ξ) (tileStage Cd Np) W (shareCone Sc Cd Np) a b Kd).accepts x ∧
        Λ * Real.log (1 / δs) ≤
          (serviceGame (Ξ := Ξ) (tileStage Cd Np) W (shareCone Sc Cd Np) a b Kd).deficiency x) := by
    rintro x ⟨hacc, hnw⟩
    by_cases hEx : E x
    · exact Or.inl hEx
    refine Or.inr ⟨?_, not_lt.mp fun hlt => hnw ?_⟩
    · have h1 := hOf.2 x fun w i => hEoff x hEx w i (hOf.1 x hacc w i)
      rw [hG] at h1
      exact h1
    · classical
      have hD : Pl.game.deficiency x =
          1 * (∑ u, if W.modeAt (tileStage Cd Np) x.1.1 u = .folded then tileCone Sc Cd Np u else 0) +
            1 * (∑ u, if W.modeAt (tileStage Cd Np) x.1.1 u = .whole then tileCone Sc Cd Np u else 0) + 0 := by
        rw [hG]
        rfl
      have hlt' : a * (∑ u, if W.modeAt (tileStage Cd Np) x.1.1 u = .folded then shareCone Sc Cd Np u else 0) +
          b * (∑ u, if W.modeAt (tileStage Cd Np) x.1.1 u = .whole then shareCone Sc Cd Np u else 0) + 0 <
            Λ * Real.log (1 / δs) := hlt
      have hcv : (∑ j : Fin Np, if (Cd.layer hr hK pr).Good (tileMode Cd W) x 0 j then (0 : ℝ)
          else (tileWords Sc Cd Np j : ℝ)) ≤ Pl.game.deficiency x * (wordsOf Sc Cd : ℝ) := by
        have h := hCov x hacc
        simp only [Fin.sum_univ_one] at h
        exact h
      have hfx := hFix x hacc
      have hfix1 : (∑ j : Fin Np, if (Cd.layer hr hK pr).Good (tileMode Cd W) x 0 j
          then (tileWords Sc Cd Np j : ℝ) else 0) ≤ (Pl.run x).fixed := by
        have h := hfx.1
        simp only [Fin.sum_univ_one, Nat.cast_one, div_one] at h
        exact h
      have hfold1 : (∑ j : Fin Np, if W.modeAt (tileStage Cd Np) x.1.1 j = .whole then (0 : ℝ)
          else (tileWords Sc Cd Np j : ℝ)) ≤ folded x := by
        have h := hfx.2
        simp only [Fin.sum_univ_one, Nat.cast_one, div_one] at h
        exact h
      exact work_of_bounds Sc Cd (W.modeAt (tileStage Cd Np) x.1.1)
        (fun j => (Cd.layer hr hK pr).Good (tileMode Cd W) x 0 j) ha hb hCount hD hcv hfix1 hfold1 hlt'
  have hE' : prob E ≤ δbind + δks := by simpa using hE
  refine (prob_mono hsub).trans ((prob_or_le _ _).trans ?_)
  linarith

end Accounting.SecurityProofs.Compose.WorkProof
