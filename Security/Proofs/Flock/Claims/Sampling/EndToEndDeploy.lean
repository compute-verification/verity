import Proofs.Flock.Claims.Sampling.EndToEndDraw
import Proofs.Flock.Claims.Sampling.Escape
import Proofs.Flock.Claims.Sampling.Select.Steps
import Proofs.Flock.Claims.Sampling.Select.Take
import Proofs.Flock.Claims.Sampling.Reference.DeployedCode

/-!
# `EndToEndDraw`'s facts as equalities, and the deployed design's shape

At a `SubsetDesign` over one class of `n` replay units (`EndToEndDraw`'s hypotheses):

* `Pa_subset_of_range`, `miss_subset`, `missAt_subset`: `P_a(B)`, `miss(t)` and `missAt D t x` are the hypergeometric
  `C(n − |B|, kd)/C(n, kd)` and `C(n − t, kd)/C(n, kd)`, for `B ⊆ [0, n)` and `t ≤ n`.
* `notMem_M_iff`: at every `δ`, `M_δ`'s complement in `[0, n)` is a count threshold `K₀ ≤ |B|`, `K₀ ≤ n + 1`.
* `notMem_M_iff_count`: at `δ = C(n − K₀, kd)/C(n, kd)` (`1 ≤ kd ≤ n`), that threshold is `min K₀ (n − kd + 1)`
  (`ratio_le_iff`).

The deployed sampled verifier's statements are of that shape. At `deployOf F (SamplingDeploy.code c reg …)` under
`RegMatches` (`deployOf_shape`), the design is `slotDesign`'s one rule, the population's first stage, and the partition
is `Slots.layout`'s one class, the population, each input row a proof unit of its owner's replay unit. So a statement
whose first stage is `subset kd` is a `SubsetDesign` over one class (`subsetDesign_iff`), and the facts above are its.
-/

namespace Claims.Sampling.EndToEndDraw

open scoped ENNReal

/-! ## The count -/

theorem choose_div_anti {n kd a b : ℕ} (h : a ≤ b) :
    ((n - b).choose kd : ℝ) / (n.choose kd : ℝ) ≤ ((n - a).choose kd : ℝ) / (n.choose kd : ℝ) :=
  div_le_div_of_nonneg_right (by exact_mod_cast Nat.choose_le_choose kd (by omega)) (Nat.cast_nonneg _)

/-- `C(a, k) < C(b, k)` for `a < b` and `1 ≤ k ≤ b`. -/
theorem choose_lt_choose {a b k : ℕ} (hab : a < b) (hk : 1 ≤ k) (hkb : k ≤ b) : a.choose k < b.choose k := by
  obtain ⟨b, rfl⟩ : ∃ b', b = b' + 1 := ⟨b - 1, by omega⟩
  obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
  rw [Nat.choose_succ_succ']
  have h1 := Nat.choose_pos (n := b) (k := k) (by omega)
  have h2 := Nat.choose_le_choose (k + 1) (show a ≤ b by omega)
  omega

/-- `C(n − a, kd) ≤ C(n − K₀, kd)` exactly when `a` reaches `min K₀ (n − kd + 1)`. -/
theorem choose_le_iff {n kd K₀ a : ℕ} (hkd : 1 ≤ kd) (hkn : kd ≤ n) :
    (n - a).choose kd ≤ (n - K₀).choose kd ↔ min K₀ (n - kd + 1) ≤ a := by
  constructor
  · intro h
    by_contra hlt
    rw [not_le, lt_min_iff] at hlt
    exact absurd h (not_le.2 (choose_lt_choose (by omega) hkd (by omega)))
  · intro h
    rcases min_le_iff.1 h with h | h
    · exact Nat.choose_le_choose kd (by omega)
    · rw [Nat.choose_eq_zero_of_lt (by omega)]
      exact Nat.zero_le _

theorem ratio_le_iff {n kd K₀ a : ℕ} (hkd : 1 ≤ kd) (hkn : kd ≤ n) :
    ((n - a).choose kd : ℝ) / (n.choose kd : ℝ) ≤ ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) ↔
      min K₀ (n - kd + 1) ≤ a := by
  rw [div_le_div_iff_of_pos_right (by exact_mod_cast Nat.choose_pos hkn), Nat.cast_le]
  exact choose_le_iff hkd hkn

/-! ## One class drawn `subset kd` -/

section OneClass

variable {d : Design} {kd : ℕ} {P : Partition} {c : List (List RU)} {n : ℕ}

theorem Pa_subset_of_range (hd : SubsetDesign d kd) (hP : P.classes = [c]) (hn : P.rus.length = n)
    {B : Finset ℕ} (hB : B ⊆ Finset.range n) :
    d.Pa P B = ((n - B.card).choose kd : ℝ) / (n.choose kd : ℝ) := by
  rw [Pa_subset hd hP, hn, Claims.Sampling.EndToEndDraw.window_zero fun v hv => Finset.mem_range.1 (hB hv)]

/-- **The miss is the hypergeometric tail**: `miss(t) = C(n − t, kd)/C(n, kd)` for `t ≤ n`. -/
theorem miss_subset (hd : SubsetDesign d kd) (hP : P.classes = [c]) (hn : P.rus.length = n) {t : ℕ}
    (ht : t ≤ n) : d.miss P t = ((n - t).choose kd : ℝ) / (n.choose kd : ℝ) := by
  have hle : ∀ B : {B : Finset ℕ // B ⊆ Finset.range P.rus.length ∧ t ≤ B.card},
      d.Pa P B.1 ≤ ((n - t).choose kd : ℝ) / (n.choose kd : ℝ) := by
    rintro ⟨B, hB, hc⟩
    rw [hn] at hB
    rw [Pa_subset_of_range hd hP hn hB]
    exact choose_div_anti hc
  have hB₀ : Finset.range t ⊆ Finset.range P.rus.length := fun v hv =>
    Finset.mem_range.2 (by rw [hn]; exact lt_of_lt_of_le (Finset.mem_range.1 hv) ht)
  have : Nonempty {B : Finset ℕ // B ⊆ Finset.range P.rus.length ∧ t ≤ B.card} := ⟨⟨_, hB₀, by simp⟩⟩
  unfold Design.miss
  refine le_antisymm (ciSup_le hle) (le_ciSup_of_le ⟨_, Set.forall_mem_range.2 hle⟩ ⟨_, hB₀, by simp⟩ ?_)
  rw [Pa_subset_of_range hd hP hn (hn ▸ hB₀), Finset.card_range]

/-- **`missAt` is the hypergeometric tail** at a statement of one class drawn `subset kd`. -/
theorem missAt_subset {Coin Col T : Type} (D : Deploy Coin Col T) (x : Msg) (s : Stmt) (hs : D.parse x = some s)
    (hd : SubsetDesign s.design kd) (hP : (D.part s).classes = [c]) (hn : (D.part s).rus.length = n) {t : ℕ}
    (ht : t ≤ n) : missAt D t x = ((n - t).choose kd : ℝ) / (n.choose kd : ℝ) := by
  unfold missAt
  rw [hs]
  dsimp only
  rw [miss_subset hd hP hn ht]
  exact max_eq_right (by positivity)

/-- **`M_δ` is a count threshold**: at every `δ`, some `K₀ ≤ n + 1` has `C(n − K₀, kd)/C(n, kd) ≤ δ` (when `K₀ ≤ n`)
and puts a set of slots outside `M_δ` exactly when it has `K₀` or more. -/
theorem notMem_M_iff (hd : SubsetDesign d kd) (hP : P.classes = [c]) (hn : P.rus.length = n) (δ : ℝ) :
    ∃ K₀ ≤ n + 1, (K₀ ≤ n → ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) ≤ δ) ∧
      ∀ B ⊆ Finset.range n, (B ∉ d.M P δ ↔ K₀ ≤ B.card) := by
  classical
  have hex : ∃ K, K = n + 1 ∨ ((n - K).choose kd : ℝ) / (n.choose kd : ℝ) ≤ δ := ⟨n + 1, Or.inl rfl⟩
  refine ⟨Nat.find hex, Nat.find_min' hex (Or.inl rfl), fun hK => ?_, fun B hB => ?_⟩
  · rcases Nat.find_spec hex with h1 | h1
    · omega
    · exact h1
  · have hcard : B.card ≤ n := by simpa using Finset.card_le_card hB
    simp only [Design.M, Set.mem_ofPred_eq, not_lt]
    rw [Pa_subset_of_range hd hP hn hB]
    constructor
    · intro hle
      exact Nat.find_min' hex (Or.inr hle)
    · intro hK
      rcases Nat.find_spec hex with h1 | h1
      · omega
      · exact (choose_div_anti hK).trans h1

/-- **At the claims' `δ`**, `C(n − K₀, kd)/C(n, kd)` with `1 ≤ kd ≤ n`: a set of slots is outside `M_δ` exactly when
it has at least `min K₀ (n − kd + 1)`. -/
theorem notMem_M_iff_count (hd : SubsetDesign d kd) (hP : P.classes = [c]) (hn : P.rus.length = n)
    (hkd : 1 ≤ kd) (hkn : kd ≤ n) (K₀ : ℕ) {B : Finset ℕ} (hB : B ⊆ Finset.range n) :
    B ∉ d.M P (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)) ↔ min K₀ (n - kd + 1) ≤ B.card := by
  simp only [Design.M, Set.mem_ofPred_eq, not_lt]
  rw [Pa_subset_of_range hd hP hn hB]
  exact ratio_le_iff hkd hkn

end OneClass

/-! ## The deployed shape -/

/-- `slotDesign`'s design: the population's first stage, `all` under the `All` rule, else the law of record's. -/
theorem slotDesign_rules {rule : _root_.SampledProofs.Select.Rule} {p : _root_.SampledProofs.Program}
    {q : Lean.Json} {t : Option Lean.Json} {d : Design}
    (h : _root_.Properties.SampledProofs.slotDesign rule p q t = .ok d) :
    ∃ first, d.rules = [⟨first, none⟩] ∧ (rule = .all → first = .all) ∧
      (rule ≠ .all → ∃ L, first = _root_.Properties.SampledProofs.oneStageOf L) := by
  unfold _root_.Properties.SampledProofs.slotDesign at h
  obtain ⟨_, -, h⟩ := _root_.Properties.SampledProofs.SecurityProofs.bind_eq_ok.1 h
  obtain ⟨_, -, h⟩ := _root_.Properties.SampledProofs.SecurityProofs.bind_eq_ok.1 h
  obtain ⟨derived, -, h⟩ := _root_.Properties.SampledProofs.SecurityProofs.bind_eq_ok.1 h
  cases rule with
  | all =>
    obtain rfl := (Except.ok.inj h).symm
    exact ⟨_, rfl, fun _ => rfl, fun h => absurd rfl h⟩
  | law _ =>
    obtain ⟨parsed, -, h⟩ := _root_.Properties.SampledProofs.SecurityProofs.bind_eq_ok.1 h
    obtain rfl := (Except.ok.inj h).symm
    exact ⟨_, rfl, (fun h => by cases h), fun _ => ⟨_, rfl⟩⟩

/-- `Slots.layout`'s one class, the population. -/
theorem layout_classes {p : _root_.SampledProofs.Program} {q : Lean.Json} {l : _root_.SampledProofs.Slots.Layout}
    (h : _root_.SampledProofs.Slots.layout p q = .ok l) : ∃ pop, l.classes = #[pop] := by
  obtain ⟨-, -, pop, rfl, -⟩ := _root_.Properties.SampledProofs.SecurityProofs.layout_ok p q l h
  exact ⟨pop, rfl⟩

theorem slotPartition_classes {l : _root_.SampledProofs.Slots.Layout} {pop : Array (Array _root_.SampledProofs.Slots.RU)}
    (h : l.classes = #[pop]) : ∃ c, (_root_.Properties.SampledProofs.slotPartition l).classes = [c] := by
  unfold _root_.Properties.SampledProofs.slotPartition
  rw [h]
  exact ⟨_, rfl⟩

/-- A design of one rule `⟨first, none⟩` is `SubsetDesign kd` exactly when its first stage is `subset kd`. -/
theorem subsetDesign_iff {d : Design} {first : OneStage} {kd : ℕ} (h : d.rules = [⟨first, none⟩]) :
    SubsetDesign d kd ↔ first = .subset kd := by
  rw [SubsetDesign, h]
  constructor
  · intro e; cases e; rfl
  · rintro rfl; rfl

section Deployed

variable {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (c : SamplingDeploy.Code)
  (reg : Msg → Option SamplingDeploy.Reg) (L : Msg → ℕ)
  (bytes : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → Fin (L x) → Fin 256)
  (sr : (x : Msg) → Finset (Fin (c.V.k x))) (sess : Msg → Claims.Seen → Option (Msg × Claims.Seen))

/-- **The deployed statement's shape**: at the deployed code under `RegMatches`, a statement that parses and
partitions has one rule, the population's first stage, over one class, the population. -/
theorem deployOf_shape (hreg : SamplingDeploy.RegMatches c reg) {x : Msg} {s : Stmt} {Pt : Partition}
    (hx : (SamplingDeploy.deployOf F (SamplingDeploy.code c reg L bytes sr sess)).parse x = some s)
    (hp : (SamplingDeploy.deployOf F (SamplingDeploy.code c reg L bytes sr sess)).partition s = some Pt) :
    ∃ first pop, s.design.rules = [⟨first, none⟩] ∧ Pt.classes = [pop] := by
  obtain ⟨g, l, -, -, hl, rfl, hd⟩ := hreg x s Pt hx hp
  obtain ⟨first, hr, -⟩ := slotDesign_rules hd
  obtain ⟨pop, hc⟩ := layout_classes hl
  obtain ⟨c', hc'⟩ := slotPartition_classes hc
  exact ⟨first, c', hr, hc'⟩

end Deployed

end Claims.Sampling.EndToEndDraw
