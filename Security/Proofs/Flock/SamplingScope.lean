import Definitions.Claims.Sampling

/-!
# The sampling scope: what it excludes, and that `KS` keeps content in it

`KS`, `KSCount` and `Bits` hold on the statements in scope (`Claims.Sampling.Scope`), which excludes every design whose
`P_a` is not its draw's miss probability. On a program of two input gates, partitioned into two units:

* `subset 3` is the hole: `P_a = 0` at every set (`Pa_hole`), so at every `δ ≥ 0` no witness is in the relation
  (`not_relS_hole`), and `KS` there would be vacuous. It is out of scope (`not_inScope_hole`).
* `subset 1` is in scope (`inScope_toy`). Its `P_a` is `1` at the empty set and `0` at both units (`Pa_toy`), so
  `M_{1/4}` holds `∅` and not `{0, 1}` (`M_toy`).
* At every deployment that reads that statement with its numbers (`ToyNumbers`), `κ` at `δ = 1/4` is below `1`
  (`κ_lt_one`). At the collision-free hash `id` neither finder finds a collision, so `KS` there bounds every prover's
  `Pr[acc ∧ faulty ∉ M_{1/4}]` below `1` (`ks_lt_one`). `toyD` is such a deployment, with a decoder its Flock spec
  admits (`listWithin_toyD`).
-/

namespace Claims.Sampling.ScopeWitness

open Claims.Sampling

/-! ## The program, its partition and the two designs -/

/-- Two input gates, both returned. -/
def toyProg : Prog := ⟨[.input, .input], [0, 1]⟩

/-- A query; the deployments here evaluate it to `toyPart`. -/
def toyQuery : Query := ⟨"", 0, []⟩

/-- One class of one stratum of two proof blocks, `{0}` and `{1}`. -/
def toyPart : Partition := ⟨[[[[0], [1]]]]⟩

/-- The program, drawn `subset 1`: in scope. -/
def toyStmt : AuditStatement := ⟨[], toyProg, toyQuery, ⟨[.subset 1]⟩, []⟩

/-- The program, drawn `subset 3` from two units: out of scope. -/
def holeStmt : AuditStatement := ⟨[], toyProg, toyQuery, ⟨[.subset 3]⟩, []⟩

/-- A `K`-subset of fewer than `K` members misses every set with probability `0`. -/
theorem Pa_subset_gt {ns : List ℕ} {K : ℕ} (h : ns.sum < K) (B : Finset ℕ) : OneStage.Pa ns (.subset K) B = 0 := by
  simp [OneStage.Pa, Nat.choose_eq_zero_of_lt h]

/-! ## The hole: `subset 3` of two units -/

/-- **`subset 3` of two units has `P_a = 0` at every set.** -/
theorem Pa_hole (B : Finset ℕ) : holeStmt.design.Pa toyPart B = 0 := by
  simp only [Design.Pa, holeStmt, toyPart, classesPa]
  rw [Pa_subset_gt (by decide), zero_mul]

/-- **No witness is in the relation** at the hole, at any `δ ≥ 0`: `M_δ` is empty. -/
theorem not_relS_hole (H : Msg → Msg) (Cm : CommitmentScheme Bool) {δ : ℝ} (hδ : 0 ≤ δ) (w : Wit) :
    ¬ RelS H Cm δ holeStmt toyPart w := by
  intro h
  have h' : δ < holeStmt.design.Pa toyPart (faulty H Cm holeStmt toyPart w) := h
  rw [Pa_hole] at h'
  linarith

/-- **The hole is out of scope**: its rule draws more units than its class has. -/
theorem not_inScope_hole : ¬ InScope toyPart holeStmt := by
  rintro ⟨_, _, _, hw⟩
  have := hw (.subset 3, [[[0], [1]]]) (by simp [holeStmt, toyPart])
  simp [OneStage.WellFormed, strataSizes] at this

/-! ## The in-scope design: `subset 1` of two units -/

/-- **The toy is in the sampling scope.** -/
theorem inScope_toy : InScope toyPart toyStmt := by
  refine ⟨⟨fun g x hx i hi => ?_, by decide⟩, ?_, ?_⟩
  · rcases g with _ | _ | g
    · simp [toyStmt, toyProg] at hx
      subst hx
      simp [BGate.reads] at hi
    · simp [toyStmt, toyProg] at hx
      subst hx
      simp [BGate.reads] at hi
    · simp [toyStmt, toyProg] at hx
  · refine ⟨by decide, by decide, fun g hg _ => ?_, by decide, by decide⟩
    have hg2 : g < 2 := hg
    interval_cases g <;> decide
  · refine ⟨rfl, ?_⟩
    simp [toyStmt, toyPart, OneStage.WellFormed, strataSizes]

/-- The toy design's `P_a`: a uniform unit of two misses the set's members among them. -/
theorem Pa_toy (B : Finset ℕ) :
    toyStmt.design.Pa toyPart B = ((2 - (window 0 2 B).card).choose 1 : ℝ) / 2 := by
  simp [Design.Pa, toyStmt, toyPart, classesPa, OneStage.Pa, strataSizes]

/-- **`M_{1/4}` has content**: it holds the empty set, and not the set of both units. -/
theorem M_toy : ∅ ∈ toyStmt.design.M toyPart (1 / 4) ∧ ({0, 1} : Finset ℕ) ∉ toyStmt.design.M toyPart (1 / 4) := by
  have h0 : window 0 2 (∅ : Finset ℕ) = ∅ := by decide
  have h2 : (window 0 2 ({0, 1} : Finset ℕ)).card = 2 := by decide
  refine ⟨?_, ?_⟩
  · show (1 / 4 : ℝ) < toyStmt.design.Pa toyPart ∅
    rw [Pa_toy, h0]
    norm_num
  · show ¬ (1 / 4 : ℝ) < toyStmt.design.Pa toyPart {0, 1}
    rw [Pa_toy, h2]
    norm_num

/-! ## `KS`'s bound at the in-scope design is below `1` -/

section Bound

variable {Coin Col T : Type}

/-- A deployment that reads `toyStmt` at `x`, evaluates its query to `toyPart`, and runs its sessions' `E^rep` with
`m′ = 20` waits, `N₀ = 3000`, `K′ = 10^9` and `ε_table = 0`. -/
structure ToyNumbers (D : Deploy Coin Col T) (x : Msg) : Prop where
  parse : D.parse x = some toyStmt
  partition : D.partition toyStmt = some toyPart
  εtable : D.εtable x = 0
  innerWaits : D.innerWaits x = 20
  innerN₀ : D.innerN₀ x = 3000
  innerCap : D.innerCap x = 10 ^ 9

theorem scope_of_toyNumbers {D : Deploy Coin Col T} {x : Msg} (h : ToyNumbers D x) : Scope D x :=
  ⟨toyStmt, toyPart, h.parse, h.partition, inScope_toy⟩

theorem k_of_toyNumbers {D : Deploy Coin Col T} {x : Msg} (h : ToyNumbers D x) : D.k x = 2 ^ 40 * 5 := by
  have hp : D.part toyStmt = toyPart := by
    simp only [Deploy.part]
    rw [h.partition]
    rfl
  simp only [Deploy.k, nPU, h.parse, hp]
  rfl

/-- `e^{−a/8} ≤ 8/(a + 8)`, from `1 + y ≤ e^y`. -/
theorem exp_neg_div_eight_le {a : ℝ} (ha : 0 ≤ a) : Real.exp (-a / 8) ≤ 8 / (a + 8) := by
  rw [neg_div, Real.exp_neg]
  have h := Real.add_one_le_exp (a / 8)
  have hpos : 0 < a / 8 + 1 := by positivity
  calc (Real.exp (a / 8))⁻¹ ≤ (a / 8 + 1)⁻¹ := inv_anti₀ hpos h
    _ = 8 / (a + 8) := by field_simp

/-- `k′ = ⌈12000/e⌉` is between `4411` and `4446`. -/
theorem innerK_bounds {D : Deploy Coin Col T} {x : Msg} (h : ToyNumbers D x) :
    (4411 : ℝ) ≤ D.innerK x ∧ (D.innerK x : ℝ) ≤ 4446 := by
  have he1 := Real.exp_one_gt_d9
  have he2 := Real.exp_one_lt_d9
  have hpos : 0 < Real.exp 1 := Real.exp_pos 1
  have h1 : 4 * D.innerN₀ x / Real.exp 1 ≤ D.innerK x := Nat.le_ceil _
  have h2 : (D.innerK x : ℝ) < 4 * D.innerN₀ x / Real.exp 1 + 1 :=
    Nat.ceil_lt_add_one (by rw [h.innerN₀]; positivity)
  rw [h.innerN₀] at h1 h2
  have hlo : (4411 : ℝ) ≤ 4 * 3000 / Real.exp 1 := by
    rw [le_div_iff₀ hpos]
    linarith
  have hhi : 4 * 3000 / Real.exp 1 ≤ (4445 : ℝ) := by
    rw [div_le_iff₀ hpos]
    linarith
  exact ⟨hlo.trans h1, by linarith⟩

/-- The sessions' knowledge error at these numbers is below `1/100`. -/
theorem κF_le {D : Deploy Coin Col T} {x : Msg} (h : ToyNumbers D x) : κF D x ≤ 1 / 100 := by
  obtain ⟨hlo, hhi⟩ := innerK_bounds h
  have hexp := exp_neg_div_eight_le (a := (D.innerK x : ℝ)) (by positivity)
  have hexp' : 8 / ((D.innerK x : ℝ) + 8) ≤ 8 / 4419 := by
    apply div_le_div_of_nonneg_left (by norm_num) (by norm_num)
    linarith
  have hw : (2 : ℝ) ^ (-(D.innerWaits x : ℝ)) = 1 / 2 ^ 20 := by
    rw [h.innerWaits, Real.rpow_neg (by norm_num), Real.rpow_natCast]
    norm_num
  have hcap : 2 * (D.innerK x : ℝ) / D.innerCap x ≤ 2 * 4446 / 10 ^ 9 := by
    rw [h.innerCap]
    push_cast
    gcongr
    norm_num
  simp only [κF, h.εtable, hw]
  linarith

/-- The waits' slack factor is at most `3/2`. -/
theorem slack_le : slack ≤ 3 / 2 := by
  have hp1 : (2 : ℝ) ^ (-40 : ℝ) ≤ 1 / 2 := by
    calc (2 : ℝ) ^ (-40 : ℝ) ≤ (2 : ℝ) ^ (-1 : ℝ) := Real.rpow_le_rpow_of_exponent_le (by norm_num) (by norm_num)
      _ = 1 / 2 := by rw [Real.rpow_neg_one]; norm_num
  have h : ∀ t : ℝ, 0 ≤ t → t ≤ 1 / 2 → (2 : ℝ) ^ t ≤ 3 / 2 := by
    intro t h0 h1
    have h := rpow_one_add_le_one_add_mul_self (s := 1) (by norm_num) h0 (by linarith)
    rw [one_add_one_eq_two, mul_one] at h
    linarith
  exact h _ (by positivity) hp1

/-- **`κ` at `δ = 1/4` is below `1`** at every deployment with the toy's numbers, at every hash. -/
theorem κ_lt_one {D : Deploy Coin Col T} {x : Msg} (h : ToyNumbers D x) : κ D (1 / 4) x < 1 := by
  have hs0 : 0 ≤ slack := Real.rpow_nonneg (by norm_num) _
  have hs := slack_le
  have hF := κF_le h
  have hk := k_of_toyNumbers h
  have hkr : (D.k x : ℝ) = 2 ^ 40 * 5 := by rw [hk]; push_cast; ring
  have hcapr : (D.cap x : ℝ) = 2 ^ 128 * (2 ^ 40 * 5) := by rw [Deploy.cap, hk]; push_cast; ring
  have hout : 2 * (D.k x : ℝ) / D.cap x ≤ 1 / 1000 := by rw [hkr, hcapr]; norm_num
  have hexp : Real.exp (-(D.k x : ℝ) / 8) ≤ 1 / 1000 := by
    refine (exp_neg_div_eight_le (by positivity)).trans ?_
    rw [hkr]
    norm_num
  have h1 : slack ^ 2 * (1 / 4 + κF D x) ≤ slack ^ 2 * (1 / 4 + 1 / 100) :=
    mul_le_mul_of_nonneg_left (by linarith) (by positivity)
  have h1' : slack ^ 2 * (1 / 4 + 1 / 100) ≤ (3 / 2) ^ 2 * (1 / 4 + 1 / 100) :=
    mul_le_mul_of_nonneg_right (pow_le_pow_left₀ hs0 hs 2) (by norm_num)
  have h2 : (1 + slack) * (2 * D.k x / D.cap x + Real.exp (-(D.k x : ℝ) / 8)) ≤ (1 + slack) * (1 / 1000 + 1 / 1000) :=
    mul_le_mul_of_nonneg_left (by linarith) (by linarith)
  have h2' : (1 + slack) * (1 / 1000 + 1 / 1000) ≤ (1 + 3 / 2) * (1 / 1000 + 1 / 1000) :=
    mul_le_mul_of_nonneg_right (by linarith) (by norm_num)
  unfold κ
  linarith

/-- No finder finds a collision of the injective hash `id`. -/
theorem finder_pr_id (F : Finder) : F.Pr id = 0 := by
  simp [Finder.Pr, Claims.Pr, Collision, Hash.Collision]

theorem prB_id [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {Ω : Type} [Fintype Ω] [Nonempty Ω] (V : Verifier Coin)
    (P : Prover Ω) (x : Msg) : PrB id V P x = 0 := by
  unfold PrB
  split_ifs
  · exact finder_pr_id _
  · rfl

/-- **`KS` has content in scope**: at the collision-free hash `id`, `KS` at a deployment with the toy's numbers bounds
every prover's `Pr[acc ∧ faulty ∉ M_{1/4}]`, and so its chance of being accepted on a trace with both units faulty, below
`1`, at every decoder Flock's spec admits. -/
theorem ks_lt_one [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] {D : Deploy Coin Col T}
    {x : Msg} (h : ToyNumbers D x) (hks : KS id D) (decode : List (ℕ × Col) → List T)
    (hdec : Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) : PrBad id D.V (R id D.commit D.parse D.partition (1 / 4)) (E D decode) x P < 1 := by
  refine lt_of_le_of_lt (hks (1 / 4) decode (by norm_num) hdec x (scope_of_toyNumbers h) Ω P) ?_
  rw [prB_id, finder_pr_id, mul_zero, Real.sqrt_zero, mul_zero, add_zero, add_zero]
  exact κ_lt_one h

end Bound

/-! ## A deployment with the toy's numbers -/

/-- A verifier of one round that accepts. -/
def toyV : Verifier Bool := ⟨fun _ => 1, fun _ _ _ => .ret .acc⟩

/-- A Flock deployment for `toyD`'s sessions; one trace, of codeword length `0`. -/
def toyFlock : Flock.Deploy Bool ℕ Unit where
  zk := false
  V := toyV
  parse _ := none
  scopeCheck _ := false
  table _ _ := none
  cols _ _ := []
  enc _ _ := 0
  len _ := 0
  θ _ := 0
  dec _ := ([], 0)
  read _ := ⟨[]⟩
  εtable _ := 0
  N₀ _ := 0
  weight _ := 0
  waits _ := 0
  cap _ := 0
  Tape := Unit
  P _ := ⟨fun _ _ _ => .ret []⟩
  δc _ _ := 0

/-- **A sampled deployment with the toy's numbers** at every statement. -/
def toyD : Deploy Bool ℕ Unit where
  V := toyV
  parse _ := some toyStmt
  scopeCheck _ := true
  partition _ := some toyPart
  commit := ⟨fun _ _ => .ret true, fun _ _ _ => []⟩
  universal _ := toyProg
  flock := toyFlock
  stage2 _ := ∅
  checked _ _ := []
  rounds _ _ := ∅
  session _ _ _ := none
  reveal _ _ _ _ := []
  innerWaits _ := 20
  innerN₀ _ := 3000
  innerCap _ := 10 ^ 9
  εtable _ := 0
  weight _ := 0
  openCalls _ := 0
  Tape := Unit
  P _ := ⟨fun _ _ _ => .ret []⟩
  δc _ _ := 0

theorem toyNumbers_toyD (x : Msg) : ToyNumbers toyD x := ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- The decoder that lists the one trace meets Flock's spec at `toyD`. -/
theorem listWithin_toyD : Flock.ListWithin toyD.flock.len toyD.flock.enc toyD.flock.θ fun _ => [()] :=
  fun _ _ _ => List.mem_singleton.2 rfl

/-- **At `toyD`, every statement is in scope and `κ` at `δ = 1/4` is below `1`.** -/
theorem toyD_scope_κ (x : Msg) : Scope toyD x ∧ κ toyD (1 / 4) x < 1 :=
  ⟨scope_of_toyNumbers (toyNumbers_toyD x), κ_lt_one (toyNumbers_toyD x)⟩

/-- **`KS` at `toyD` and the hash `id`** bounds every prover's `Pr[acc ∧ faulty ∉ M_{1/4}]` below `1`. -/
theorem toyD_ks_lt_one (hks : KS id toyD) (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    PrBad id toyD.V (R id toyD.commit toyD.parse toyD.partition (1 / 4)) (E toyD fun _ => [()]) x P < 1 :=
  ks_lt_one (toyNumbers_toyD x) hks _ listWithin_toyD Ω P

end Claims.Sampling.ScopeWitness
