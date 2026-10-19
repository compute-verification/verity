import Proofs.Flock.Claims.Witnesses.Shared

/-!
# Refutation witnesses: Flock's claims

Each of Flock's claims (`Definitions.Claims`, `Claims.Flock`) fails at a toy record:

* `ksD`: the verifier accepts after one round, nothing parses (so no statement has a witness), the decoder lists the
  one trace, and the numbers give `k = 32` accepted reruns of `K = 1024`, `m′ = 4` waits and `ε_table = 0`, so
  `κ ≤ 1/16 + 1/16 + e^{−4} < 1/2` and `γ = 16`. `KS_fails`: the silent prover is accepted surely, every extraction
  fails and `B` finds nothing, so `Pr[acc ∧ ¬R] = 1 > κ`. `Bits_fails`: there `λ > 0`, and the prover makes no call.
  `ScopeMatches_fails`: its scope is empty and its check takes everything.
* `compD`: the verifier rejects after one round, and every statement parses as `inStmt`, in scope (`m = 22`,
  `k_log = 6`, the forms the proof covers) with an empty circuit, so every witness is in the relation.
  `Complete_fails`: every prover with a tape fails at `δ_c < 1`. `Nontrivial_fails`: a recorded run is rejected.
* `ZK_fails` (`ZKMal.not_flockZK_collSim_toy`): the colliding simulator, which met the form before `Fork`, fails at
  `ZKMal.toyD`.
* `effD`: a decoder that lists the one trace and counts one step. `Efficient_fails`: at `p = 0` its count exceeds
  `p`. Before the count was the record's (`Flock.Deploy.dec`), a count of `0` passed as an argument met the claim.
* `ListWithin_fails`: the decoder that lists nothing, at a code with a trace (`not_listWithin_nil`).

Two of the claims are still met by a trivial instance, both because they read the honest prover as a parameter:

* `Complete_vacuous`: a prover with no tape meets `Complete` at `δ_c = 0`, at every record.
* `ZK_vacuous`: at every masked record, a prover that ignores its witness and calls nothing meets `ZK` at `ε_zk = 0`
  and `γ = 0` (`ZKmalR_vacuous`).
-/

namespace Claims.Witness.Flock

open Claims

/-- The toy record of `KS` and `Bits`. -/
noncomputable def ksD (Coin : Type) : Claims.Flock.Deploy Coin Unit Unit where
  zk := false
  V := accV Coin
  parse _ := none
  scopeCheck _ := true
  table _ _ := none
  cols _ _ := []
  enc _ _ := ()
  len _ := 0
  θ _ := 0
  dec _ := ([()], 0)
  read _ := ⟨[]⟩
  εtable _ := 0
  N₀ _ := 8 * Real.exp 1
  weight _ := 0
  waits _ := 4
  cap _ := 1024

/-- A statement in Flock's scope, with an empty circuit. -/
def inStmt : Claims.Flock.Stmt where
  m := 22
  kLog := 6
  C := ⟨0, [], []⟩
  inp _ := []
  out _ := []
  pub := []
  form :=
    { hm96Rows := true, typed := false, retainedRounds := true, leaf := "hm96-sha512/v1", tables := 1,
      publicInputs := false, sharedRows := false, registeredReads := false, hiddenOutputs := false,
      unitInputGroups := 1, tail := false, unitLog := 0, readsOnlyPortBits := true, rowsV1 := true,
      rowsV2OrSeg := false, wiresAcyclic := false }

/-- The toy record of `Complete` and `Nontrivial`: every statement parses as `inStmt`, and the verifier rejects. -/
def compD (Coin : Type) : Claims.Flock.Deploy Coin Unit Unit where
  zk := false
  V := rejV Coin
  parse _ := some inStmt
  scopeCheck _ := true
  table _ _ := some inStmt
  cols _ _ := []
  enc _ _ := ()
  len _ := 0
  θ _ := 0
  dec _ := ([()], 0)
  read _ := ⟨[]⟩
  εtable _ := 0
  N₀ _ := 0
  weight _ := 0
  waits _ := 0
  cap _ := 0

/-- The empty column, the zero of a word of field-element lists. -/
local instance zeroCol : Zero (List Bool) := ⟨[]⟩

/-- The toy record of `Efficient`: its decoder lists the one trace and counts one step. -/
def effD (Coin : Type) : Claims.Flock.Deploy Coin (List Bool) Unit where
  zk := false
  V := accV Coin
  parse _ := none
  scopeCheck _ := true
  table _ _ := none
  cols _ _ := []
  enc _ _ := []
  len _ := 0
  θ _ := 0
  dec _ := ([()], 1)
  read _ := ⟨[]⟩
  εtable _ := 0
  N₀ _ := 0
  weight _ := 0
  waits _ := 0
  cap _ := 0

theorem listWithin_unit {Col : Type} [DecidableEq Col] [Zero Col] (len : Unit → ℕ) (enc : Unit → ℕ → Col)
    (θ : Unit → ℝ) : Claims.Flock.ListWithin len enc θ fun _ => [()] :=
  fun _ _ _ => List.mem_singleton.2 rfl

theorem inStmt_inScope : Claims.Flock.InScope false inStmt := by
  refine ⟨by decide, by decide, by decide, by decide, rfl, by decide, rfl, Or.inl rfl, by decide, by decide,
    fun h => absurd h (by decide), Or.inl ⟨rfl, by decide⟩, by decide, rfl, Or.inl rfl⟩

theorem inStmt_relS (w : Claims.Flock.Wit) : Claims.Flock.RelS inStmt w := fun _ _ => rfl

section Numbers

variable {Coin : Type}

theorem ksD_k (x : Msg) : (ksD Coin).k x = 32 := by
  have h : 4 * (8 * Real.exp 1) / Real.exp 1 = ((32 : ℕ) : ℝ) := by
    rw [div_eq_iff (Real.exp_pos 1).ne']; push_cast; ring
  simp only [Claims.Flock.Deploy.k, ksD, h, Nat.ceil_natCast]

theorem exp_neg_four_le : Real.exp (-((32 : ℕ) : ℝ) / 8) ≤ 1 / 5 := by
  have h4 : (4 : ℝ) + 1 ≤ Real.exp 4 := Real.add_one_le_exp 4
  rw [show -((32 : ℕ) : ℝ) / 8 = -4 by norm_num, Real.exp_neg, one_div]
  exact inv_anti₀ (by norm_num) (by linarith)

theorem ksD_κ (x : Msg) : Claims.Flock.κ (ksD Coin) x < 1 / 2 := by
  have he := exp_neg_four_le
  have hw : (2 : ℝ) ^ (-((4 : ℕ) : ℝ)) = 1 / 16 := by
    rw [Real.rpow_neg (by norm_num), Real.rpow_natCast]; norm_num
  unfold Claims.Flock.κ
  rw [ksD_k]
  simp only [ksD] at hw ⊢
  rw [hw]
  push_cast at he ⊢
  linarith

theorem ksD_κ_pos (x : Msg) : 0 < Claims.Flock.κ (ksD Coin) x := by
  unfold Claims.Flock.κ
  simp only [ksD]
  positivity

theorem ksD_γ (x : Msg) : Claims.Flock.γ (ksD Coin) x = 16 := by
  simp [Claims.Flock.γ, ksD, accV]

theorem ksD_lam_pos (x : Msg) : 0 < lamOf (Claims.Flock.κ (ksD Coin) x) (Claims.Flock.γ (ksD Coin) x) := by
  have hκ := ksD_κ (Coin := Coin) x
  have hκ0 := ksD_κ_pos (Coin := Coin) x
  have h1 : 1 < Real.logb 2 (1 / Claims.Flock.κ (ksD Coin) x) := by
    rw [Real.lt_logb_iff_rpow_lt (by norm_num) (by positivity), Real.rpow_one, lt_div_iff₀ hκ0]
    linarith
  have h2 : Real.logb 2 16 = 4 := by
    rw [show (16 : ℝ) = 2 ^ ((4 : ℕ) : ℝ) by rw [Real.rpow_natCast]; norm_num,
      Real.logb_rpow (by norm_num) (by norm_num)]
    norm_num
  unfold lamOf
  rw [ksD_γ, h2]
  have := lt_min h1 (show (1 : ℝ) < 255.5 - 4 / 2 by norm_num)
  linarith

theorem ksD_noR (x : Msg) (w : Claims.Flock.Wit) : ¬ Claims.Flock.R (ksD Coin).parse x w := by
  rintro ⟨s, hs, -⟩
  simp [ksD] at hs

end Numbers

section Claims

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **Flock's `KS` fails** at `ksD`. -/
theorem KS_fails (H : Msg → Msg) : ¬ Claims.Flock.KS H (ksD Coin) := fun h =>
  Witness.KS_fails H _ _ _ _ [] (ksD_noR []) (by linarith [ksD_κ (Coin := Coin) []])
    (h _ (listWithin_unit _ _ _))

/-- **Flock's `Bits` fails** at `ksD`. -/
theorem Bits_fails (H : Msg → Msg) : ¬ Claims.Flock.Bits H (ksD Coin) := fun h =>
  Witness.Bits_fails H _ _ _ [] (ksD_noR []) (ksD_lam_pos []) (h _ (listWithin_unit _ _ _))

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **Flock's `ScopeMatches` fails** at `ksD`, whose scope is empty and whose check takes everything. -/
theorem ScopeMatches_fails : ¬ Claims.Flock.ScopeMatches (ksD Coin) := fun h => by
  obtain ⟨s, hs, -⟩ := (h []).2 rfl
  simp [ksD] at hs

omit [Nonempty Coin] in
/-- **Flock's `Complete` fails** at `compD`, at every prover with a tape and every `δ_c < 1`. -/
theorem Complete_fails {Ω : Type} [Nonempty Ω] (P : Claims.Flock.Wit → Prover Ω) (δc : Msg → ℝ) (x : Msg)
    (hδ : δc x < 1) : ¬ Claims.Flock.Complete constH (compD Coin) P δc :=
  Witness.Complete_fails P _ _ δc (w := ⟨[]⟩) ⟨inStmt, rfl, inStmt_inScope⟩ ⟨inStmt, rfl, inStmt_relS _⟩ hδ

omit [Nonempty Coin] in
/-- **Flock's `Complete` is met at a prover with no tape**, at every hash and record and `δ_c = 0`. -/
theorem Complete_vacuous {Col T : Type} [DecidableEq Col] (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T) :
    Claims.Flock.Complete H D noTape 0 :=
  Witness.Complete_vacuous H D.V _ _

omit [Fintype Coin] [Nonempty Coin] in
/-- **Flock's `Nontrivial` fails** at `compD`, which rejects the recorded run. -/
theorem Nontrivial_fails (H : Msg → Msg) : ¬ Claims.Flock.Nontrivial H (compD Coin) [⟨[], [], fun _ => default⟩] :=
  fun h => Witness.Nontrivial_fails H h.1

/-- **Flock's `ZK` fails** at the colliding simulator, at `ZKMal.toyD`, the quiet prover, every `ε_zk < 1` and every
`γ`. -/
theorem ZK_fails {N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (u v : Msg)
    (εzk γ : Msg → ℝ) (x : Msg) (hε : εzk x < 1) :
    ¬ Claims.Flock.ZK H ZKMal.toyD (fun _ => ZKMal.quietP N Ω []) (ZKMal.collSim u v) εzk γ :=
  ZKMal.not_flockZK_collSim_toy H u v εzk γ x hε

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **Flock's `ZK` is met by a prover that ignores its witness**, at every masked record, `ε_zk = 0` and `γ = 0`. -/
theorem ZK_vacuous {Col T N Ω : Type} [DecidableEq Col] [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω]
    (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T) (hzk : D.zk = true) :
    Claims.Flock.ZK H D (fun _ => ZKMal.quietP N Ω []) (quietSim fun x => D.V.k x + 1) 0 0 :=
  ⟨hzk, ZKmalR_vacuous H _ _⟩

/-- **Flock's `Efficient` fails** at `effD` and `p = 0`: its decoder counts one step. -/
theorem Efficient_fails (H : Msg → Msg) : ¬ Claims.Flock.Efficient H (effD Coin) 0 := fun h => by
  have := h.2.2.1 []
  simp [effD] at this

end Claims

theorem fill_enc {Col T : Type} [Zero Col] (enc : T → ℕ → Col) (M : T) (n : ℕ) (i : Fin n) :
    Claims.Flock.fill n ((List.range n).map fun j => (j, enc M j)) i = enc M i.val := by
  have hsome : ((List.range n).find? fun j => j == i.val).isSome :=
    List.find?_isSome.2 ⟨i.val, List.mem_range.2 i.isLt, beq_self_eq_true _⟩
  obtain ⟨j, hj⟩ := Option.isSome_iff_exists.1 hsome
  have hji : j = i.val := by simpa using List.find?_some hj
  subst hji
  simp [Claims.Flock.fill, List.find?_map, hj]

/-- **`ListWithin` constrains the decoder**: the decoder that lists nothing fails it at every code with a trace of
nonnegative radius, whose own encoding, opened everywhere, agrees with it everywhere. -/
theorem not_listWithin_nil {Col T : Type} [DecidableEq Col] [Zero Col] (len : T → ℕ) (enc : T → ℕ → Col)
    (θ : T → ℝ) (M : T) (hθ : 0 ≤ θ M) : ¬ Claims.Flock.ListWithin len enc θ fun _ => [] := by
  intro h
  refine absurd (h ((List.range (len M)).map fun j => (j, enc M j)) M ?_) List.not_mem_nil
  rw [Finset.filter_true_of_mem fun i _ => (fill_enc enc M (len M) i).symm, Finset.card_univ, Fintype.card_fin]
  have : (0 : ℝ) ≤ len M := Nat.cast_nonneg _
  nlinarith

/-- **`ListWithin` fails** at the decoder that lists nothing, at a code of one trace of length `1`. -/
theorem ListWithin_fails : ¬ Claims.Flock.ListWithin (fun _ : Unit => 1) (fun _ _ => ()) (fun _ => 0) fun _ => [] :=
  not_listWithin_nil _ _ _ () le_rfl

end Claims.Witness.Flock
