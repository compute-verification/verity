import Proofs.Flock.Claims.Sampling.EndToEndDraw
import Proofs.Flock.Claims.Sampling.Escape
import Proofs.Flock.Claims.Sampling.Select.Steps
import Proofs.Flock.Claims.Sampling.Select.Take
import Proofs.Flock.Claims.Sampling.Reference.DeployedCode

/-!
# `EndToEndDraw`'s facts as equalities, and the deployed design's shape

At a `SubsetDesign` over one class of `n` proof blocks (`EndToEndDraw`'s hypotheses):

* `Pa_subset_of_range`, `miss_subset`, `missAt_subset`: `P_a(B)`, `miss(t)` and `missAt D t x` are the hypergeometric
  `C(n − |B|, kd)/C(n, kd)` and `C(n − t, kd)/C(n, kd)`, for `B ⊆ [0, n)` and `t ≤ n`.
* `notMem_M_iff`: at every `δ`, `M_δ`'s complement in `[0, n)` is a count threshold `K₀ ≤ |B|`, `K₀ ≤ n + 1`.
* `notMem_M_iff_count`: at `δ = C(n − K₀, kd)/C(n, kd)` (`1 ≤ kd ≤ n`), that threshold is `min K₀ (n − kd + 1)`
  (`ratio_le_iff`).

The deployed sampled verifier's statements are not of that shape. At `deployOf F (SamplingDeploy.code c reg …)` under
`RegMatches` (`deployOf_shape`), the design is `slotDesign`'s two rules, the population's first stage and the input
class drawn `all`, and the partition is `Slots.layout`'s two classes, the population and the input class (empty, or one
stratum). So `SubsetDesign` and a one-class partition fail there (`deployOf_not_subsetDesign`,
`deployOf_not_oneClass`). At a population drawn `subset kd` (`SubsetInputsDesign`), with `N` population slots:

* `Pa_subsetInputs`: `P_a(B)` is `C(N − |B ∩ [0, N)|, kd)/C(N, kd)` when `B` misses the input class, else `0`.
* `miss_subsetInputs`, `missAt_subsetInputs`: `miss(t) = missAt D t x = C(N − t, kd)/C(N, kd)` for `t ≤ N`.
* `notMem_M_iff_subsetInputs`: outside `M_δ` at `δ = C(N − K₀, kd)/C(N, kd)` is: `B` meets the input class, or at
  least `min K₀ (N − kd + 1)` of its slots.
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

variable {d : Design} {kd : ℕ} {P : Partition} {c : List (List (List ℕ))} {n : ℕ}

theorem Pa_subset_of_range (hd : SubsetDesign d kd) (hP : P.classes = [c]) (hn : P.blocks.length = n)
    {B : Finset ℕ} (hB : B ⊆ Finset.range n) :
    d.Pa P B = ((n - B.card).choose kd : ℝ) / (n.choose kd : ℝ) := by
  rw [Pa_subset hd hP, hn, Claims.Sampling.EndToEndDraw.window_zero fun v hv => Finset.mem_range.1 (hB hv)]

/-- **The miss is the hypergeometric tail**: `miss(t) = C(n − t, kd)/C(n, kd)` for `t ≤ n`. -/
theorem miss_subset (hd : SubsetDesign d kd) (hP : P.classes = [c]) (hn : P.blocks.length = n) {t : ℕ}
    (ht : t ≤ n) : d.miss P t = ((n - t).choose kd : ℝ) / (n.choose kd : ℝ) := by
  have hle : ∀ B : {B : Finset ℕ // B ⊆ Finset.range P.blocks.length ∧ t ≤ B.card},
      d.Pa P B.1 ≤ ((n - t).choose kd : ℝ) / (n.choose kd : ℝ) := by
    rintro ⟨B, hB, hc⟩
    rw [hn] at hB
    rw [Pa_subset_of_range hd hP hn hB]
    exact choose_div_anti hc
  have hB₀ : Finset.range t ⊆ Finset.range P.blocks.length := fun v hv =>
    Finset.mem_range.2 (by rw [hn]; exact lt_of_lt_of_le (Finset.mem_range.1 hv) ht)
  have : Nonempty {B : Finset ℕ // B ⊆ Finset.range P.blocks.length ∧ t ≤ B.card} := ⟨⟨_, hB₀, by simp⟩⟩
  unfold Design.miss
  refine le_antisymm (ciSup_le hle) (le_ciSup_of_le ⟨_, Set.forall_mem_range.2 hle⟩ ⟨_, hB₀, by simp⟩ ?_)
  rw [Pa_subset_of_range hd hP hn (hn ▸ hB₀), Finset.card_range]

/-- **`missAt` is the hypergeometric tail** at a statement of one class drawn `subset kd`. -/
theorem missAt_subset {Coin Col T : Type} (D : Deploy Coin Col T) (x : Msg) (s : AuditStatement) (hs : D.parse x = some s)
    (hd : SubsetDesign s.design kd) (hP : (D.part s).classes = [c]) (hn : (D.part s).blocks.length = n) {t : ℕ}
    (ht : t ≤ n) : missAt D t x = ((n - t).choose kd : ℝ) / (n.choose kd : ℝ) := by
  unfold missAt
  rw [hs]
  dsimp only
  rw [miss_subset hd hP hn ht]
  exact max_eq_right (by positivity)

/-- **`M_δ` is a count threshold**: at every `δ`, some `K₀ ≤ n + 1` has `C(n − K₀, kd)/C(n, kd) ≤ δ` (when `K₀ ≤ n`)
and puts a set of slots outside `M_δ` exactly when it has `K₀` or more. -/
theorem notMem_M_iff (hd : SubsetDesign d kd) (hP : P.classes = [c]) (hn : P.blocks.length = n) (δ : ℝ) :
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
theorem notMem_M_iff_count (hd : SubsetDesign d kd) (hP : P.classes = [c]) (hn : P.blocks.length = n)
    (hkd : 1 ≤ kd) (hkn : kd ≤ n) (K₀ : ℕ) {B : Finset ℕ} (hB : B ⊆ Finset.range n) :
    B ∉ d.M P (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)) ↔ min K₀ (n - kd + 1) ≤ B.card := by
  simp only [Design.M, Set.mem_ofPred_eq, not_lt]
  rw [Pa_subset_of_range hd hP hn hB]
  exact ratio_le_iff hkd hkn

end OneClass

/-! ## The deployed shape -/

/-- `slotDesign`'s design: the population's first stage (`all` under the `All` rule, else the law of record's), then
the input class drawn `all`. -/
theorem slotDesign_rules {rule : _root_.SampledProofs.Select.Rule} {p : _root_.SampledProofs.Program}
    {q : Lean.Json} {t : Option Lean.Json} {d : Design}
    (h : _root_.Properties.SampledProofs.slotDesign rule p q t = .ok d) :
    ∃ first, d.rules = [first, .all] ∧ (rule = .all → first = .all) ∧
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

/-- `Slots.layout`'s classes: the population, then the input class, empty or one stratum. -/
theorem layout_classes {p : _root_.SampledProofs.Program} {q : Lean.Json} {l : _root_.SampledProofs.Slots.Layout}
    (h : _root_.SampledProofs.Slots.layout p q = .ok l) :
    ∃ pop ins, l.classes = #[pop, ins] ∧ (ins = #[] ∨ ∃ st, ins = #[st]) := by
  unfold _root_.SampledProofs.Slots.layout at h
  obtain ⟨_, -, h⟩ := _root_.Properties.SampledProofs.SecurityProofs.bind_eq_ok.1 h
  obtain ⟨_, -, h⟩ := _root_.Properties.SampledProofs.SecurityProofs.bind_eq_ok.1 h
  dsimp only at h
  obtain ⟨pop, -, h⟩ := _root_.Properties.SampledProofs.SecurityProofs.bind_eq_ok.1 h
  obtain rfl := (Except.ok.inj h).symm
  refine ⟨pop, _, rfl, ?_⟩
  split
  · exact Or.inl rfl
  · exact Or.inr ⟨_, rfl⟩

theorem slotPartition_classes {l : _root_.SampledProofs.Slots.Layout}
    {pop ins : Array (Array _root_.SampledProofs.Slots.ProofBlock)}
    (h : l.classes = #[pop, ins]) (hins : ins = #[] ∨ ∃ st, ins = #[st]) :
    ∃ c ic, (_root_.Properties.SampledProofs.slotPartition l).classes = [c, ic] ∧ (ic = [] ∨ ∃ st, ic = [st]) := by
  unfold _root_.Properties.SampledProofs.slotPartition
  rw [h]
  rcases hins with rfl | ⟨st, rfl⟩
  · exact ⟨_, _, rfl, Or.inl rfl⟩
  · exact ⟨_, _, rfl, Or.inr ⟨_, rfl⟩⟩

section Deployed

variable {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (c : SamplingDeploy.Code)
  (reg : Msg → Option SamplingDeploy.Reg) (L : Msg → ℕ)
  (bytes : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → Fin (L x) → Fin 256)
  (sr : (x : Msg) → Finset (Fin (c.V.k x))) (sess : Msg → Claims.Seen → Option (Msg × Claims.Seen))

/-- **The deployed statement's shape**: at the deployed code under `RegMatches`, a statement that parses and
partitions has two rules, the population's first stage and the input class drawn `all`, over two classes, the
population and the input class (empty, or one stratum). -/
theorem deployOf_shape (hreg : SamplingDeploy.RegMatches c reg) {x : Msg} {s : AuditStatement} {Pt : Partition}
    (hx : (SamplingDeploy.deployOf F (SamplingDeploy.code c reg L bytes sr sess)).parse x = some s)
    (hp : (SamplingDeploy.deployOf F (SamplingDeploy.code c reg L bytes sr sess)).partition s = some Pt) :
    ∃ first pop ins, s.design.rules = [first, .all] ∧ Pt.classes = [pop, ins] ∧
      (ins = [] ∨ ∃ st, ins = [st]) := by
  obtain ⟨g, l, -, -, hl, rfl, hd⟩ := hreg x s Pt hx hp
  obtain ⟨first, hr, -⟩ := slotDesign_rules hd
  obtain ⟨pop, ins, hc, hins⟩ := layout_classes hl
  obtain ⟨c', ic, hc', hic⟩ := slotPartition_classes hc hins
  exact ⟨first, c', ic, hr, hc', hic⟩

theorem deployOf_not_subsetDesign (hreg : SamplingDeploy.RegMatches c reg) {x : Msg} {s : AuditStatement} {Pt : Partition}
    (hx : (SamplingDeploy.deployOf F (SamplingDeploy.code c reg L bytes sr sess)).parse x = some s)
    (hp : (SamplingDeploy.deployOf F (SamplingDeploy.code c reg L bytes sr sess)).partition s = some Pt) (kd : ℕ) :
    ¬ SubsetDesign s.design kd := by
  obtain ⟨_, _, _, hr, -⟩ := deployOf_shape F c reg L bytes sr sess hreg hx hp
  intro h
  rw [SubsetDesign, hr] at h
  simp at h

theorem deployOf_not_oneClass (hreg : SamplingDeploy.RegMatches c reg) {x : Msg} {s : AuditStatement} {Pt : Partition}
    (hx : (SamplingDeploy.deployOf F (SamplingDeploy.code c reg L bytes sr sess)).parse x = some s)
    (hp : (SamplingDeploy.deployOf F (SamplingDeploy.code c reg L bytes sr sess)).partition s = some Pt)
    (c₁ : List (List (List ℕ))) : Pt.classes ≠ [c₁] := by
  obtain ⟨_, _, _, -, hc, -⟩ := deployOf_shape F c reg L bytes sr sess hreg hx hp
  rw [hc]
  simp

end Deployed

/-! ## The population drawn `subset kd`, then the input class -/

/-- The deployed design at a population drawn `subset kd`: that, then the input class drawn `all`. -/
def SubsetInputsDesign (d : Design) (kd : ℕ) : Prop := d.rules = [.subset kd, .all]

theorem all_Pa_of_empty {ns : List ℕ} {W : Finset ℕ} (h : W = ∅) : OneStage.Pa ns .all W = 1 := by
  simp [OneStage.Pa, h]

theorem all_Pa_of_ne {ns : List ℕ} {W : Finset ℕ} (h : W ≠ ∅) : OneStage.Pa ns .all W = 0 := by
  simp [OneStage.Pa, h]

theorem blocks_length_two {P : Partition} {c ic : List (List (List ℕ))} (hP : P.classes = [c, ic]) :
    P.blocks.length = c.flatten.length + ic.flatten.length := by
  simp [Partition.blocks, hP]

/-- A set of slots below `N + M` that misses `[N, N + M)` is below `N`. -/
theorem subset_range_of_window {N M : ℕ} {B : Finset ℕ} (hB : B ⊆ Finset.range (N + M)) (h : window N M B = ∅) :
    B ⊆ Finset.range N := by
  intro v hv
  have hv' := Finset.mem_range.1 (hB hv)
  rw [Finset.mem_range]
  by_contra hlt
  have : v - N ∈ window N M B :=
    (Claims.Sampling.mem_window (by omega)).2 (by rwa [Nat.add_sub_of_le (by omega)])
  rw [h] at this
  simp at this

theorem window_range_of_le {N M t : ℕ} (ht : t ≤ N) : window N M (Finset.range t) = ∅ := by
  ext v
  simp only [window, Finset.mem_image, Finset.mem_filter, Finset.mem_range, Finset.notMem_empty, iff_false,
    not_exists, not_and]
  intro w hw _
  omega

theorem Pa_subsetInputs {d : Design} {kd : ℕ} (hd : SubsetInputsDesign d kd) {P : Partition}
    {c ic : List (List (List ℕ))} (hP : P.classes = [c, ic]) (B : Finset ℕ) :
    d.Pa P B = ((c.flatten.length - (window 0 c.flatten.length B).card).choose kd : ℝ) /
      (c.flatten.length.choose kd : ℝ) * OneStage.Pa (strataSizes ic) .all (window c.flatten.length ic.flatten.length B) := by
  have h1 : d.Pa P B = OneStage.Pa (strataSizes c) (.subset kd) (window 0 c.flatten.length B) *
      (OneStage.Pa (strataSizes ic) .all (window (0 + c.flatten.length) ic.flatten.length B) * 1) := by
    unfold Design.Pa
    rw [show d.rules = [.subset kd, .all] from hd, hP]
    rfl
  rw [h1, mul_one, zero_add]
  congr 1
  change ((((strataSizes c).sum - _).choose kd : ℕ) : ℝ) / (((strataSizes c).sum.choose kd : ℕ) : ℝ) = _
  rw [strataSizes_sum]

section SubsetInputs

variable {d : Design} {kd : ℕ} {P : Partition} {c ic : List (List (List ℕ))}

/-- On a set that misses the input class, `P_a` is the population's hypergeometric miss. -/
theorem Pa_subsetInputs_of_window (hd : SubsetInputsDesign d kd) (hP : P.classes = [c, ic]) {B : Finset ℕ}
    (hB : B ⊆ Finset.range (c.flatten.length + ic.flatten.length))
    (hw : window c.flatten.length ic.flatten.length B = ∅) :
    d.Pa P B = ((c.flatten.length - B.card).choose kd : ℝ) / (c.flatten.length.choose kd : ℝ) := by
  have hB' := subset_range_of_window hB hw
  rw [Pa_subsetInputs hd hP, all_Pa_of_empty hw, mul_one,
    Claims.Sampling.EndToEndDraw.window_zero fun v hv => Finset.mem_range.1 (hB' hv)]

/-- **The deployed miss is the population's hypergeometric tail**: `miss(t) = C(N − t, kd)/C(N, kd)` for `t ≤ N`, the
population's `N` slots. -/
theorem miss_subsetInputs (hd : SubsetInputsDesign d kd) (hP : P.classes = [c, ic]) {t : ℕ}
    (ht : t ≤ c.flatten.length) :
    d.miss P t = ((c.flatten.length - t).choose kd : ℝ) / (c.flatten.length.choose kd : ℝ) := by
  have hlen := blocks_length_two hP
  have hle : ∀ B : {B : Finset ℕ // B ⊆ Finset.range P.blocks.length ∧ t ≤ B.card},
      d.Pa P B.1 ≤ ((c.flatten.length - t).choose kd : ℝ) / (c.flatten.length.choose kd : ℝ) := by
    rintro ⟨B, hB, hc⟩
    rw [hlen] at hB
    by_cases hw : window c.flatten.length ic.flatten.length B = ∅
    · rw [Pa_subsetInputs_of_window hd hP hB hw]
      exact choose_div_anti hc
    · rw [Pa_subsetInputs hd hP, all_Pa_of_ne hw, mul_zero]
      positivity
  have hB₀ : Finset.range t ⊆ Finset.range P.blocks.length := fun v hv =>
    Finset.mem_range.2 (by rw [hlen]; have := Finset.mem_range.1 hv; omega)
  have : Nonempty {B : Finset ℕ // B ⊆ Finset.range P.blocks.length ∧ t ≤ B.card} := ⟨⟨_, hB₀, by simp⟩⟩
  unfold Design.miss
  refine le_antisymm (ciSup_le hle) (le_ciSup_of_le ⟨_, Set.forall_mem_range.2 hle⟩ ⟨_, hB₀, by simp⟩ ?_)
  rw [Pa_subsetInputs_of_window hd hP (hlen ▸ hB₀) (window_range_of_le ht), Finset.card_range]

theorem missAt_subsetInputs {Coin Col T : Type} (D : Deploy Coin Col T) (x : Msg) (s : AuditStatement)
    (hs : D.parse x = some s) (hd : SubsetInputsDesign s.design kd) (hP : (D.part s).classes = [c, ic]) {t : ℕ}
    (ht : t ≤ c.flatten.length) :
    missAt D t x = ((c.flatten.length - t).choose kd : ℝ) / (c.flatten.length.choose kd : ℝ) := by
  unfold missAt
  rw [hs]
  dsimp only
  rw [miss_subsetInputs hd hP ht]
  exact max_eq_right (by positivity)

/-- **The deployed threshold**: at `δ = C(N − K₀, kd)/C(N, kd)` (`1 ≤ kd ≤ N`), a set of slots is outside `M_δ` exactly
when it meets the input class or has at least `min K₀ (N − kd + 1)` slots. -/
theorem notMem_M_iff_subsetInputs (hd : SubsetInputsDesign d kd) (hP : P.classes = [c, ic]) (hkd : 1 ≤ kd)
    (hkn : kd ≤ c.flatten.length) (K₀ : ℕ) {B : Finset ℕ} (hB : B ⊆ Finset.range P.blocks.length) :
    B ∉ d.M P (((c.flatten.length - K₀).choose kd : ℝ) / (c.flatten.length.choose kd : ℝ)) ↔
      (window c.flatten.length ic.flatten.length B).Nonempty ∨ min K₀ (c.flatten.length - kd + 1) ≤ B.card := by
  rw [blocks_length_two hP] at hB
  simp only [Design.M, Set.mem_ofPred_eq, not_lt]
  by_cases hw : window c.flatten.length ic.flatten.length B = ∅
  · rw [Pa_subsetInputs_of_window hd hP hB hw, ratio_le_iff hkd hkn, hw]
    simp
  · rw [Pa_subsetInputs hd hP, all_Pa_of_ne hw, mul_zero]
    exact iff_of_true (by positivity) (Or.inl (Finset.nonempty_iff_ne_empty.2 hw))

end SubsetInputs

end Claims.Sampling.EndToEndDraw
