import Definitions.Pouw.Guarantees.Dither
import Proofs.Pouw.RelativeSecurity

/-!
# The no-fold Dither protocol's work-soundness: the freed-compute bound and the honest witness

* `FreedOfFloor`: the guarantee `Definitions.Pouw.Guarantees.Dither.FreedOfFloor`, weak duality of the period's linear
  program.
-/

namespace Pouw.SecurityProofs.Dither

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Finset

/-! ## Freed compute -/

/-- `Definitions.Pouw.Guarantees.Dither.FreedOfFloor`. -/
theorem FreedOfFloor : Pouw.Guarantees.Dither.FreedOfFloor := by
  intro G Pd W ε X μ z z' hμ hneeds hval hW hε hfit' hslack hfit hfloor
  refine ⟨W, z', hW, hfit', ?_⟩
  have h1 : ∑ j, Pd.val j * z j ≤ ∑ r, μ r * (Pd.cap r - X r) := by
    calc ∑ j, Pd.val j * z j ≤ ∑ j, (∑ r, μ r * Pd.needs j r) * z j :=
          Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_right (hval j) (hfit.1 j)
      _ = ∑ r, μ r * ∑ j, z j * Pd.needs j r := by
          simp_rw [Finset.sum_mul, Finset.mul_sum]
          rw [Finset.sum_comm]
          exact Finset.sum_congr rfl fun r _ => Finset.sum_congr rfl fun j _ => by ring
      _ ≤ ∑ r, μ r * (Pd.cap r - X r) :=
          Finset.sum_le_sum fun r _ => mul_le_mul_of_nonneg_left (by linarith [hfit.2 r]) (hμ r)
  have h2 : ∑ r, μ r * (W * Pd.hon r) ≤ ∑ r, μ r * Pd.cap r := by
    refine Finset.sum_le_sum fun r _ => mul_le_mul_of_nonneg_left ?_ (hμ r)
    have hs : 0 ≤ ∑ j, z' j * Pd.needs j r :=
      Finset.sum_nonneg fun j _ => mul_nonneg (hfit'.1 j) (hneeds j r)
    linarith [hfit'.2 r]
  have h3 : ∑ r, μ r * (Pd.cap r - X r) =
      ∑ r, μ r * (Pd.cap r - W * Pd.hon r) + (∑ r, μ r * (W * Pd.hon r) - ∑ r, μ r * X r) := by
    rw [← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun r _ => by ring
  have h4 : ε * ∑ r, μ r * (W * Pd.hon r) ≤ ε * ∑ r, μ r * Pd.cap r := mul_le_mul_of_nonneg_left h2 hε
  unfold Period.value
  nlinarith [h1, h3, h4, hfloor, hslack]

/-! ## The honest witness (rule I) -/

/-- Programs as functions carrying a tag at least their output type's size; the machine's own costs are `0`. The
witness model (`wmodel`) charges a program its tag's bits on every resource, so every program uses at least its
output's bits. -/
noncomputable def sized : Machine where
  Alg I O := {g : (I → O) × ℕ // Nat.card O ≤ g.2}
  eval f := f.1.1
  time _ := 0
  cost _ _ := 0
  size _ := 0
  seq f g := ⟨(g.1.1 ∘ f.1.1, f.1.2 + g.1.2), by have := g.2; omega⟩
  pair f g := ⟨(fun x => (f.1.1 x, g.1.1 x), f.1.2 * g.1.2), by
    rw [Nat.card_prod]; exact Nat.mul_le_mul f.2 g.2⟩
  fst := fun {I _} => ⟨(Prod.fst, Nat.card I), le_rfl⟩
  snd := fun {_ J} => ⟨(Prod.snd, Nat.card J), le_rfl⟩
  eval_seq _ _ _ := rfl
  eval_pair _ _ _ := rfl
  eval_fst _ := rfl
  eval_snd _ := rfl
  time_nonneg _ := le_rfl
  cost_le_time _ _ := le_rfl
  size_le_time _ := le_rfl
  size_nonneg _ := le_rfl
  time_seq_le _ _ := by norm_num
  le_time_seq _ _ := le_rfl
  time_pair_le _ _ := by norm_num
  le_time_pair_left _ _ := le_rfl
  le_time_pair_right _ _ := le_rfl
  time_fst := rfl
  time_snd := rfl

/-- The witness's three resources: resource `r` takes `wt r` units a bit, and supplies `wt r` units a second. -/
def wt : Fin 3 → ℝ := ![1, 2, 3]

/-- **The witness model**: three resources; a program tagged `t` takes `log₂ t` seconds and `wt r · log₂ t` units of
each resource `r`, holds nothing and imports nothing. -/
noncomputable def wmodel : Accel where
  Mc := sized
  Res := Fin 3
  rate := wt
  mem := 0
  use f _ r := wt r * (Nat.log 2 f.1.2 : ℝ)
  held _ _ := 0
  wall f _ := (Nat.log 2 f.1.2 : ℝ)
  ext _ _ := 0
  P1 := True
  P2 := True

/-- The witness's prices: resource 1 is free, so the price is not uniform. -/
def μw : Fin 3 → ℝ := ![1, 0, 2]

/-- The no-fold knobs on a `1 × 1 × 1` instance with one segment of length 1, bit noise on both operands, one word
checked a round, and the commitment `h` into 512 bits. -/
abbrev Kw (h : (Fin 1 → Fin 1 → Fin 1 → Word) → (Fin 512 → Bool)) : NoFold where
  n := 1
  n' := 1
  nb := 1
  L := 1
  Draw := (Fin 1 → Fin (1 * 1) → Fin 2) × (Fin (1 * 1) → Fin 1 → Fin 2)
  noise ab d := (fun i t => ab.1 i t + ((d.1 i t).val : Word), fun t j => ab.2 t j + ((d.2 t j).val : Word))
  s := 1
  Chk := Unit
  cells _ _ := (0, 0, 0)
  Root := Fin 512 → Bool
  merkle := h
  Path := Unit
  opens _ _ _ _ := True

/-- The no-fold round at `Kw h`. -/
abbrev Pw (h : (Fin 1 → Fin 1 → Fin 1 → Word) → (Fin 512 → Bool)) : RoundProto := noFoldRound (Kw h)

variable (h : (Fin 1 → Fin 1 → Fin 1 → Word) → (Fin 512 → Bool))

/-- The committed operands: zero. -/
def a0 : (Pw h).Inst := (fun _ _ => 0, fun _ _ => 0)

/-- The honest words program, tagged its output's size. -/
noncomputable def hw : wmodel.Mc.Alg (Pw h).Draw (Pw h).Tr :=
  (⟨(noFoldHonest (Kw h) (a0 h), Nat.card (Pw h).Tr), le_rfl⟩ :
    {g : ((Pw h).Draw → (Pw h).Tr) × ℕ // Nat.card (Pw h).Tr ≤ g.2})

/-- The honest commitment program, tagged its output's size. -/
noncomputable def hh : wmodel.Mc.Alg (Pw h).Draw (Pw h).Root :=
  (⟨(fun d => h (noFoldHonest (Kw h) (a0 h) d), Nat.card (Pw h).Root), le_rfl⟩ :
    {g : ((Pw h).Draw → (Pw h).Root) × ℕ // Nat.card (Pw h).Root ≤ g.2})

/-- The witness service: a two-round period's live coins; nothing committed before the coins; the answer each round's
commitment and words; the verifier checks each round's drawn word. -/
noncomputable def Sw : Service where
  Seed := Paths ((Pw h).Draw × (Pw h).Chk) 2
  lam := 4
  card_seed := by
    refine (Fintype.card_congr (Equiv.prodCongr (Equiv.refl _) (Equiv.prodPUnit _))).trans ?_
    simp [noFoldRound, Fintype.card_prod, Fintype.card_fin]
  Com := Unit
  Ans := ((Pw h).Root × (Pw h).Tr) × ((Pw h).Root × (Pw h).Tr)
  Coins := Unit
  verify _ σ T := @decide ((Pw h).ok (a0 h) σ.1.1 σ.1.2 T.1.2 ∧ (Pw h).ok (a0 h) σ.2.1.1 σ.2.1.2 T.2.2)
    (Classical.propDecidable _)

/-- The witness analysis map: every adversary's period is cut into two honest rounds. -/
noncomputable def run2 : PeriodMap wmodel (Sw h) (Pw h) 2 :=
  fun _ _ _ _ => { play := .round (a0 h) (hw h) (hh h) fun _ => .round (a0 h) (hw h) (hh h) fun _ => .done,
                   Bind := fun _ _ => False, ZK := fun _ _ => False }

/-- The honest prover: it commits nothing before the coins and answers each round's commitment and words, tagged its
output's size. -/
noncomputable def honest : wmodel.Mc.Alg ((Sw h).Seed × Unit) ((Sw h).Com × (Sw h).Ans) :=
  (⟨(fun ω => ((), ((h (noFoldHonest (Kw h) (a0 h) ω.1.1.1), noFoldHonest (Kw h) (a0 h) ω.1.1.1),
      (h (noFoldHonest (Kw h) (a0 h) ω.1.2.1.1), noFoldHonest (Kw h) (a0 h) ω.1.2.1.1))),
      Nat.card ((Sw h).Com × (Sw h).Ans)), le_rfl⟩ :
    {g : ((Sw h).Seed × Unit → (Sw h).Com × (Sw h).Ans) × ℕ // Nat.card ((Sw h).Com × (Sw h).Ans) ≤ g.2})

/-- The witness period: one other job, worth `7` a unit and using `wt r` of each resource a unit; the honest round
uses `544 wt r` (its 32 bits of words and 512 bits of commitment); the period is `T = 2000` seconds. -/
noncomputable def Pdw : Period wmodel where
  T := 2000
  J := Unit
  val _ := 7
  needs _ r := wt r
  hon r := wt r * 544

theorem card_tr : Nat.card (Pw h).Tr = 2 ^ 32 := by
  simp [noFoldRound, Nat.card_eq_fintype_card, ZMod.card]

theorem card_root : Nat.card (Pw h).Root = 2 ^ 512 := by
  simp [noFoldRound, Nat.card_eq_fintype_card]

theorem card_out : Nat.card ((Sw h).Com × (Sw h).Ans) = 2 ^ 1088 := by
  change Nat.card (Unit × (((Pw h).Root × (Pw h).Tr) × ((Pw h).Root × (Pw h).Tr))) = _
  rw [Nat.card_prod, Nat.card_prod, Nat.card_prod, Nat.card_unique, card_tr, card_root]
  simp only [one_mul, ← pow_add]

/-- A program of the witness model uses at least `n` bits' worth where its output has `2^n` values. -/
theorem log_tag_ge {I O : Type} (f : wmodel.Mc.Alg I O) {n : ℕ} (hn : Nat.card O = 2 ^ n) :
    n ≤ Nat.log 2 f.1.2 := by
  have h0 : Nat.card O ≤ f.1.2 := f.2
  have h1 : 2 ^ n ≤ f.1.2 := hn.symm.trans_le h0
  calc n = Nat.log 2 (2 ^ n) := (Nat.log_pow (by norm_num) n).symm
    _ ≤ Nat.log 2 f.1.2 := Nat.log_mono_right h1

theorem log_card_tr : Nat.log 2 (Nat.card (Pw h).Tr) = 32 := by
  rw [card_tr]; exact Nat.log_pow (by norm_num) 32

theorem log_card_root : Nat.log 2 (Nat.card (Pw h).Root) = 512 := by
  rw [card_root]; exact Nat.log_pow (by norm_num) 512

theorem sum_μw_wt : ∑ r, μw r * wt r = 7 := by
  simp [Fin.sum_univ_three, μw, wt]; norm_num

theorem cost_wmodel {I O : Type} (f : wmodel.Mc.Alg I O) (x : I) :
    wmodel.cost μw f x = 7 * (Nat.log 2 f.1.2 : ℝ) := by
  change ∑ r, μw r * (wt r * (Nat.log 2 f.1.2 : ℝ)) = _
  simp_rw [← mul_assoc]
  rw [← Finset.sum_mul, sum_μw_wt]

theorem avg_le_one {Ω : Type} [Fintype Ω] (f : Ω → ℝ) (hf : ∀ x, f x ≤ 1) : avg f ≤ 1 := by
  unfold avg
  rcases Nat.eq_zero_or_pos (Fintype.card Ω) with h0 | hpos
  · simp [h0]
  · rw [div_le_one (by exact_mod_cast hpos)]
    calc ∑ ω, f ω ≤ ∑ _ω : Ω, (1 : ℝ) := Finset.sum_le_sum fun ω _ => hf ω
      _ = Fintype.card Ω := by simp

/-- A shortfall is `0` where the cost reaches `c H` with nothing imported. -/
theorem shortfall_zero {c H cost : ℝ} (hH : 0 < H) (h : c * H ≤ cost) : roundShortfall c 0 H cost 0 = 0 := by
  unfold roundShortfall
  rw [zero_mul, add_zero]
  refine max_eq_left ?_
  rw [sub_nonpos, le_div_iff₀ hH]
  exact h

theorem g1_wmodel : wmodel.G1 := by
  intro I O f x
  show (0 : ℝ) + 0 ≤ 0
  norm_num

theorem g2_wmodel : wmodel.G2 := by
  refine ⟨fun r => ?_, fun f x r => ⟨?_, le_rfl⟩⟩
  · fin_cases r <;> simp [wmodel, wt]
  · exact mul_nonneg (by fin_cases r <;> simp [wt]) (Nat.cast_nonneg _)

/-- **SAK(matmul) holds in the witness model at `c_mm = 1`**: every words program uses at least the words' 32 bits. -/
theorem sak_wmodel (η : ℝ) : SAKMatmul wmodel (Pw h) μw 1 η 0 (7 * 32) := by
  intro _ _ _ _ a Mw
  refine avg_le_one _ fun x => ?_
  have hs : roundShortfall 1 0 (7 * 32) (wmodel.cost μw Mw x.1) (wmodel.ext Mw x.1) = 0 := by
    refine shortfall_zero (by norm_num) ?_
    rw [cost_wmodel]
    have := log_tag_ge Mw (card_tr h)
    have : (32 : ℝ) ≤ (Nat.log 2 Mw.1.2 : ℝ) := by exact_mod_cast this
    linarith
  split_ifs <;> simp [hs]

/-- **SHA-512's speed holds in the witness model at `c_sha = 1`**: every commitment program uses at least the
commitment's 512 bits. -/
theorem sha_wmodel (η : ℝ) : ShaSpeed wmodel (Pw h) μw 1 η 0 (7 * 512) := by
  intro _ _ _ _ a Mw Mh _
  refine avg_le_one _ fun x => ?_
  have hs : roundShortfall 1 0 (7 * 512) (wmodel.cost μw Mh x.1) (wmodel.ext Mh x.1) = 0 := by
    refine shortfall_zero (by norm_num) ?_
    rw [cost_wmodel]
    have := log_tag_ge Mh (card_root h)
    have : (512 : ℝ) ≤ (Nat.log 2 Mh.1.2 : ℝ) := by exact_mod_cast this
    linarith
  split_ifs <;> simp [hs]

theorem prob_false {Ω : Type} [Fintype Ω] : prob (fun _ : Ω => False) = 0 := by
  unfold prob
  simp

/-- **The service is sound in the witness**, at no break and for any finders: the cut's rounds are honest, so its play
accepts on every coin. -/
theorem service_wmodel (sha : Claims.Msg → Claims.Msg) (fb fz : Finders wmodel.Mc (Sw h) 2) :
    ServiceSound wmodel (Sw h) (Pw h) (run2 h) (Equiv.refl _) sha fb fz 0 0 0 0 := by
  intro Ω' _ M _ r
  refine ⟨fun w => ?_, fun w => ?_, fun x _ => Or.inl ⟨fun _ => rfl, fun _ => rfl, trivial⟩⟩
  · show prob (fun _ => False) ≤ 0 + 0 * _
    rw [prob_false]
    simp
  · show prob (fun _ => False) ≤ 0 + 0 * _
    rw [prob_false]
    simp

theorem import_wmodel : ImportWithinCertifier wmodel (Sw h) 0 := fun _ _ _ _ _ => le_rfl

theorem use_hw (d : (Pw h).Draw) (r : Fin 3) : wmodel.use (hw h) d r = wt r * 32 := by
  change wt r * (Nat.log 2 (Nat.card (Pw h).Tr) : ℝ) = _
  rw [log_card_tr]
  norm_num

theorem use_hh (d : (Pw h).Draw) (r : Fin 3) : wmodel.use (hh h) d r = wt r * 512 := by
  change wt r * (Nat.log 2 (Nat.card (Pw h).Root) : ℝ) = _
  rw [log_card_root]
  norm_num

/-- The cut's two rounds use `2 · 544 wt r` of each resource, on every path. -/
theorem play_use (Ω' : Type) [Fintype Ω'] (M : wmodel.Mc.Alg ((Sw h).Seed × Ω') ((Sw h).Com × (Sw h).Ans)) (r : Ω')
    (x : Paths ((Pw h).Draw × (Pw h).Chk) 2) (q : Fin 3) :
    (run2 h Ω' M r).play.use x q = 2 * (wt q * 544) := by
  change wmodel.use (hw h) x.1.1 q + wmodel.use (hh h) x.1.1 q +
    (wmodel.use (hw h) x.2.1.1 q + wmodel.use (hh h) x.2.1.1 q + 0) = _
  rw [use_hw, use_hh, use_hw, use_hh]
  ring

/-- **Commit-and-open's extraction holds in the witness**: every adversary's output has at least `1088` bits (two
rounds' words and commitments), so the two honest rounds are within its run. -/
theorem extract_wmodel : Extract wmodel (Sw h) (Pw h) (run2 h) (Equiv.refl _) := by
  intro _ _ _ _ Ω' _ M _ r
  refine ⟨⟨fun _ => rfl, fun _ => ⟨fun _ => rfl, fun _ => trivial⟩⟩, fun x _ => ⟨fun q => ?_, ?_⟩⟩
  · refine le_trans (le_of_eq (play_use h Ω' M r x q)) ?_
    change 2 * (wt q * 544) ≤ wt q * (Nat.log 2 M.1.2 : ℝ)
    have h1 := log_tag_ge M (card_out h)
    have h2 : (1088 : ℝ) ≤ (Nat.log 2 M.1.2 : ℝ) := by exact_mod_cast h1
    have h3 : 0 ≤ wt q := by fin_cases q <;> simp [wt]
    nlinarith
  · show (0 : ℝ) + 0 + (0 + 0 + 0) ≤ 0
    norm_num

/-- **The witness period's dual price**: `μw` prices the job at its value, and the honest strategy credited two rounds
does `2000 − 1088` units of the job beside them, worth the capacity they leave. -/
theorem dualOpt_wmodel : Pdw.DualOpt μw 2 := by
  refine ⟨fun r => by fin_cases r <;> simp [μw], fun _ r => by fin_cases r <;> simp [Pdw, wt], fun _ => ?_,
    fun _ => 2000 - 1088, ⟨fun _ => by norm_num, fun r => ?_⟩, ?_⟩
  · show (7 : ℝ) ≤ ∑ r, μw r * wt r
    rw [sum_μw_wt]
  · show (2 : ℕ) * (wt r * 544) + ∑ _j : Unit, (2000 - 1088 : ℝ) * wt r ≤ wt r * 2000
    simp only [Finset.univ_unique, Finset.sum_singleton]
    norm_num
    linarith
  · show ∑ _j : Unit, (7 : ℝ) * (2000 - 1088) = ∑ r, μw r * (wt r * 2000 - (2 : ℕ) * (wt r * 544))
    simp [Fin.sum_univ_three, μw, wt]
    norm_num

/-- **The honest witness** (rule I): in a model of three resources at a non-uniform price, over a two-round period of
the no-fold round, with one other job at the period's dual price, the honest prover meets every hypothesis of
`WorkSound` at `c_mm = c_sha = 1`, `β = K = 0`, no break, and every `η > 0` and `0 < δ ≤ 1`; it commits first, is
accepted on every coin, and its credited rounds use exactly two honest rounds' resources. -/
theorem honest_witness (η δ : ℝ) (hη : 0 < η) (hδ : 0 < δ) (hδ1 : δ ≤ 1) (sha : Claims.Msg → Claims.Msg)
    (fb fz : Finders wmodel.Mc (Sw h) 2) :
    Pdw.DualOpt μw 2 ∧ (0 : ℝ) < 7 * 32 ∧ (0 : ℝ) < 7 * 512 ∧ (7 * 32 + 7 * 512 : ℝ) = ∑ r, μw r * Pdw.hon r ∧
      (1 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧ 0 < η ∧ 0 < δ ∧ δ ≤ 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0 ∧
      (0 : ℝ) ≤ 0 ∧ wmodel.G1 ∧ wmodel.G2 ∧ wmodel.P1 ∧ wmodel.P2 ∧
      SAKMatmul wmodel (Pw h) μw 1 η 0 (7 * 32) ∧ ShaSpeed wmodel (Pw h) μw 1 η 0 (7 * 512) ∧
      ServiceSound wmodel (Sw h) (Pw h) (run2 h) (Equiv.refl _) sha fb fz 0 0 0 0 ∧
      ImportWithinCertifier wmodel (Sw h) 0 ∧ Extract wmodel (Sw h) (Pw h) (run2 h) (Equiv.refl _) ∧
      (∀ σ σ' r, (wmodel.Mc.eval (honest h) (σ, r)).1 = (wmodel.Mc.eval (honest h) (σ', r)).1) ∧
      (∀ σ, (Sw h).verify (wmodel.Mc.eval (honest h) (σ, ())).1 σ (wmodel.Mc.eval (honest h) (σ, ())).2 = true) ∧
      ∀ x q, (run2 h Unit (honest h) ()).play.use x q = (2 : ℕ) * Pdw.hon q := by
  refine ⟨dualOpt_wmodel, by norm_num, by norm_num, ?_, le_rfl, le_rfl, hη, hδ, hδ1, le_rfl, le_rfl, le_rfl,
    le_rfl, g1_wmodel, g2_wmodel, trivial, trivial, sak_wmodel h η, sha_wmodel h η, service_wmodel h sha fb fz,
    import_wmodel h, extract_wmodel h, fun _ _ _ => rfl, fun σ => ?_, fun x q => ?_⟩
  · show (7 * 32 + 7 * 512 : ℝ) = ∑ r, μw r * (wt r * 544)
    simp [Fin.sum_univ_three, μw, wt]
    norm_num
  · exact @decide_eq_true _ (Classical.propDecidable _) ⟨fun _ => rfl, fun _ => rfl⟩
  · refine (play_use h Unit (honest h) () x q).trans ?_
    show 2 * (wt q * 544) = ((2 : ℕ) : ℝ) * (wt q * 544)
    norm_num

end Pouw.SecurityProofs.Dither
