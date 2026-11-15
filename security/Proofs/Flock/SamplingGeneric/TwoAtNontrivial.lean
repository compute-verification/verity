module

public import Proofs.Flock.SamplingGeneric.TwoAt
public import Proofs.Flock.Claims.Witnesses.SamplingTwo

/-!
# `SampledKSTwo`'s conclusion fails at an instance it speaks of

`Flock.Guarantees.SampledKSTwo` takes the domain hypothesis `Code.ReplayCap` at SHA-512 and concludes
`CompiledTwo.KS` at `compiledTwo` over Flock's masked record and the two-stage code at a stage order. Its nontrivial
witness states both at one instance of its own binders: the replay cap holds there, and `CompiledTwo.KS` fails.

* `accCode`: a code that accepts every run after two rounds, parses the toy statement into its one proof block
  (`toyStmt`, `toyPart`), opens nothing (`at_ = none`, so its commitment opens no value), has no session, and holds
  its proof block's session at `N₀ = 1000`; `accStage` its stage order, round `0` then the one round `1`.
* `accRs`: its replay sample of record, with no registration, is empty, so `ReplayCap` holds at the cap `0`
  (`accReplayCap`).
* `accTwo`: the two-stage code at that order, under the two-stage design `toyTwo`. No witness is in its relation at
  `δ = 0` (`accTwo_noR`): its one proof block is incorrect at every interior, so `P̄_a = 0`.
* `κ_lt_one`: a compiled verifier whose sessions' bound `κs` is in `[0, 1/10]` has `κ 0 < 1`, its waits' tails at
  most `1/50` each (`out_le`, `in_le`); `accκF`: `κ_F` at `N₀ = 1000` and Flock's inner numbers (`waits = 210`,
  `cap = 2^256`, no table) is at most `1/10`.
* `Flock.SecurityProofs.SampledKSTwo_nontrivial`: the silent prover is accepted on every run while the extractor
  returns no witness in the relation, and the sessions' finder finds nothing, so `1 ≤ κ 0 < 1`.
-/

@[expose] public section

namespace SamplingDeploy

open Claims Claims.Sampling
open Claims.Witness (silentP)
open Claims.Witness.Sampling (toyStmt toyPart toyPart_blocks exp_neg_le_inv)
open Claims.SamplingGeneric (κFAt)
open Claims.Witness.SamplingGeneric (toyTwo inScopeTwo_toy toyTwo_Pa bext_pr toy_correct_val)

/-! ## A verifier of two rounds that accepts every run -/

/-- A verifier of two rounds that accepts every run. -/
def accV2 : Verifier FlockDeploy.Coin := ⟨fun _ => 2, fun _ _ _ => .ret .acc⟩

theorem run_accV2_out {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (accV2.k x) → FlockDeploy.Coin) : (run H P accV2 x ω r).out = .acc := by
  simp [run, accV2, stepWith, OC.eval]

theorem run_accV2_vLog {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (accV2.k x) → FlockDeploy.Coin) : (run H P accV2 x ω r).vLog = [] := by
  simp [run, accV2, stepWith, OC.eval]

theorem prB_accV2 {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (P : Prover Ω) (x : Msg) :
    PrB H accV2 P x = 0 := by
  unfold PrB
  split_ifs with hk
  · unfold Finder.Pr
    rw [Pr_congr fun s => iff_false_intro (by simp [B, run_accV2_vLog, findColl])]
    exact Pr_false
  · rfl

/-- At `accV2`, on a statement with no witness, every extractor fails on every run. -/
theorem prBad_accV2 {W Ω : Type} [Fintype Ω] (H : Msg → Msg) (R : Rel W) (E : Extractor FlockDeploy.Coin W accV2)
    (x : Msg) (P : Prover Ω) [Nonempty Ω] (hR : ∀ w, ¬ R x w) : PrBad H accV2 R E x P = 1 := by
  have hb : ∀ s : Ω × (Fin (accV2.k x) → FlockDeploy.Coin) × E.C x,
      ((run H P accV2 x s.1 s.2.1).out = .acc ∧ ¬ R.ext x (E.eval H P x s.1 s.2.1 s.2.2).1) ↔ True := by
    intro s
    refine iff_true_intro ⟨run_accV2_out H P x s.1 s.2.1, ?_⟩
    cases (E.eval H P x s.1 s.2.1 s.2.2).1 with
    | none => exact id
    | some w => exact hR w
  unfold PrBad
  rw [Pr_congr hb]
  exact ZKMal.Pr_true

/-! ## The accepting code -/

/-- **The accepting code**: it accepts every run after two rounds, parses the toy statement into its one proof block,
opens nothing, has no session, and holds its proof block's session at `N₀ = 1000`. -/
def accCode : Code where
  V := accV2
  parse _ := some toyStmt
  scopeCheck _ := true
  partition _ := some toyPart
  secondStage _ := ∅
  checked _ _ := []
  rounds _ _ := ∅
  session _ _ _ := none
  reveal _ _ _ _ := []
  own := #[]
  at_ _ _ := none
  layer _ _ _ := []
  place _ := ⟨1, 0⟩
  bits _ _ := false
  salt _ := []
  π _ := []
  innerN₀ _ := 1000
  tables _ := 0
  reveals _ := 0
  sessions _ := 0
  step _ _ := .say []
  reads _ _ := []
  wit _ _ _ := ⟨[]⟩

/-- The accepting code's stage order: round `0`, then the one round `1`. -/
def accStage (x : Msg) : StageOrder (accCode.V.k x) 1 :=
  (⟨⟨0, Nat.zero_lt_two⟩, ⟨1, Nat.one_lt_two⟩, Nat.zero_lt_one⟩ : Order (accCode.V.k x)).stage

/-- The accepting code's two-stage code at its stage order, with no registration and byte maps that read no bytes. -/
abbrev accTwo : Code :=
  codeTwoStage accCode (fun _ => none) (fun _ => 0) (fun _ => 0) toyBytes (fun _ => 1) accStage toyStageBytes

/-- The accepting code's replay sample of record. -/
abbrev accRs : (x : Msg) → (Fin ((accCode.atOrder (orderOf accStage)).V.k x) → FlockDeploy.Coin) → List ℕ :=
  replayTwo (accCode.atOrder (orderOf accStage)) (fun _ => none) (fun _ => 0)
    (bytesFirst accCode (fun _ => 0) (orderOf accStage) toyBytes)

/-- **The replay cap holds at `0`**: with no registration, the replay sample of record is empty. -/
theorem accReplayCap :
    (accCode.atOrder (orderOf accStage)).ReplayCap FlockDeploy.sha512 (fun _ => some toyTwo) accRs (fun _ => 0) := by
  intro x s Pt d _ _ _ _ Ω P ω r e _
  show (sampleBlocks Pt d.part []).card ≤ 0
  simp [sampleBlocks]

/-! ## No witness is in its relation -/

theorem commit_accTwo_opens (H : Msg → Msg) (roots : List Msg) (o : Opening Bool) :
    (commit accTwo).Opens H roots o = false :=
  rfl

/-- The accepting code's commitment opens no value. -/
theorem commit_accTwo_val (H : Msg → Msg) (w : Wit) (s : AuditStatement) (g : ℕ) :
    w.val H (commit accTwo) s g = none := by
  unfold Wit.val
  rw [List.find?_eq_none.2 fun o _ => by simp [commit_accTwo_opens]]
  rfl

/-- The proof block is incorrect under any witness at any interior values: the commitment opens nothing. -/
theorem acc_incorrectIn (H : Msg → Msg) (w : Wit) (ι : ℕ → Bool) :
    0 ∈ incorrectIn H (commit accTwo) toyStmt toyPart toyPart w 0 0 ι := by
  classical
  unfold incorrectIn
  refine Finset.mem_filter.2 ⟨by decide, ?_⟩
  rintro ⟨u, r, hu, hr, hc⟩
  rw [toyPart_blocks] at hu hr
  simp only [List.getElem?_cons_zero, Option.some.injEq] at hu hr
  subst hu hr
  obtain ⟨b, hb⟩ := toy_correct_val _ ι hc
  rw [commit_accTwo_val] at hb
  cases hb

theorem acc_footprint (H : Msg → Msg) (w : Wit) (ι : ℕ → ℕ → Bool) :
    1 ≤ footprint H (commit accTwo) toyStmt toyPart toyPart w ι 0 0 :=
  Finset.card_pos.2 ⟨0, acc_incorrectIn H w (ι 0)⟩

/-- No witness is in `accTwo`'s relation at `δ = 0`. -/
theorem accTwo_noR (H : Msg → Msg) (x : Msg) (w : Wit) :
    ¬ RTwo H (commit accTwo) accTwo.parse accTwo.partition (fun _ => some toyTwo) 0 x w := by
  rintro ⟨s, Pt, d, hs, hPt, hd, ι, hM⟩
  obtain rfl : s = toyStmt := (Option.some.inj hs).symm
  obtain rfl : Pt = toyPart := (Option.some.inj hPt).symm
  obtain rfl : d = toyTwo := (Option.some.inj hd).symm
  have hlt : (0 : ℝ) < toyTwo.Pa toyPart (footprint H (commit accTwo) toyStmt toyPart toyPart w ι) := hM
  rw [toyTwo_Pa (acc_footprint H w ι)] at hlt
  exact lt_irrefl _ hlt

/-! ## `κ` at Flock's numbers -/

section Kappa

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Claims.SamplingGeneric.Arg Coin}

/-- A wait of `k = 2^10·(2n + 1)` accepted runs and `2^192·k` reruns has tails at most `1/50`. -/
theorem wait_le (k : ℝ) (hk : 1024 ≤ k) : 2 * k / (2 ^ 192 * k) + Real.exp (-k / 8) ≤ 1 / 50 := by
  have hk0 : 0 < k := by linarith
  have h1 : 2 * k / (2 ^ 192 * k) ≤ 1 / 100 := by
    rw [div_le_iff₀ (mul_pos (by positivity) hk0)]
    nlinarith
  have h2 : Real.exp (-k / 8) ≤ 1 / 100 := by
    rw [neg_div]
    refine (exp_neg_le_inv (by positivity)).trans ?_
    rw [inv_le_comm₀ (by positivity) (by norm_num)]
    linarith
  linarith

theorem le_k (n : ℕ) : (1024 : ℝ) ≤ ((2 ^ slackExp * (2 * n + 1) : ℕ) : ℝ) := by
  have : 1024 ≤ 2 ^ slackExp * (2 * n + 1) := by
    rw [show 2 ^ slackExp = 1024 from rfl]
    omega
  exact_mod_cast this

theorem out_le (L : Claims.SamplingGeneric.Compiled Coin A) (x : Msg) :
    2 * (L.kOut x : ℝ) / (L.capOut x : ℝ) + Real.exp (-(L.kOut x : ℝ) / 8) ≤ 1 / 50 := by
  have hc : (L.capOut x : ℝ) = 2 ^ 192 * (L.kOut x : ℝ) := by
    simp only [Claims.SamplingGeneric.Compiled.capOut, Nat.cast_mul, Nat.cast_pow, Nat.cast_ofNat]
  rw [hc]
  exact wait_le _ (le_k _)

theorem in_le (L : Claims.SamplingGeneric.Compiled Coin A) (x : Msg) :
    2 * (L.kIn x : ℝ) / (L.capIn x : ℝ) + Real.exp (-(L.kIn x : ℝ) / 8) ≤ 1 / 50 := by
  have hc : (L.capIn x : ℝ) = 2 ^ 192 * (L.kIn x : ℝ) := by
    simp only [Claims.SamplingGeneric.Compiled.capIn, Nat.cast_mul, Nat.cast_pow, Nat.cast_ofNat]
  rw [hc]
  exact wait_le _ (le_k _)

/-- **`κ 0 < 1`** wherever the sessions' bound `κs` is in `[0, 1/10]`. -/
theorem κ_lt_one (L : Claims.SamplingGeneric.Compiled Coin A) (x : Msg) (h0 : 0 ≤ L.κs x) (h : L.κs x ≤ 1 / 10) :
    L.κ 0 x < 1 := by
  have hs := Claims.Witness.Sampling.slack_le_two
  have hs0 := Claims.Witness.Sampling.slack_pos
  have ho := out_le L x
  have hi := in_le L x
  have ho0 : 0 ≤ 2 * (L.kOut x : ℝ) / (L.capOut x : ℝ) + Real.exp (-(L.kOut x : ℝ) / 8) := by positivity
  have hi0 : 0 ≤ 2 * (L.kIn x : ℝ) / (L.capIn x : ℝ) + Real.exp (-(L.kIn x : ℝ) / 8) := by positivity
  have hsq : slack ^ 2 ≤ 4 := by nlinarith
  have h1 : slack ^ 2 * (0 + L.κs x) ≤ 4 * (1 / 10) := by
    rw [zero_add]
    exact mul_le_mul hsq h (by positivity) (by norm_num)
  have h2 : slack * (2 * (L.kIn x : ℝ) / (L.capIn x : ℝ) + Real.exp (-(L.kIn x : ℝ) / 8)) ≤ 2 * (1 / 50) :=
    mul_le_mul hs hi hi0 (by norm_num)
  unfold Claims.SamplingGeneric.Compiled.κ
  linarith

end Kappa

section Flock

/-- **`κ_F` at `N₀ = 1000`** and Flock's inner numbers, at no table: `2^{−210} + 2k′/2^{256} + e^{−k′/8}`
with `k′ = ⌈4000/e⌉ ≥ 1000`, at most `1/10`. -/
theorem accκF {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (x : Msg) :
    κFAt (deployOf F accTwo) x 1000 ≤ 1 / 10 := by
  have hε : (deployOf F accTwo).εtable x = 0 := by
    show ((0 : ℕ) : ℝ) * 2 ^ (-(205 : ℝ)) = 0
    simp
  have he : Real.exp 1 ≤ 4 := by linarith [Real.exp_one_lt_d9]
  have he1 : 1 ≤ Real.exp 1 := by linarith [Real.add_one_le_exp (1 : ℝ)]
  have hK1 : (1000 : ℝ) ≤ (⌈4 * (1000 : ℝ) / Real.exp 1⌉₊ : ℝ) := by
    refine le_trans ?_ (Nat.le_ceil _)
    rw [le_div_iff₀ (Real.exp_pos 1)]
    nlinarith
  have hK2 : (⌈4 * (1000 : ℝ) / Real.exp 1⌉₊ : ℝ) ≤ 4001 := by
    have h := Nat.ceil_lt_add_one (show (0 : ℝ) ≤ 4 * 1000 / Real.exp 1 by positivity)
    have h' : 4 * (1000 : ℝ) / Real.exp 1 ≤ 4000 := by
      rw [div_le_iff₀ (Real.exp_pos 1)]
      nlinarith
    linarith
  show 4 * (deployOf F accTwo).εtable x + (2 : ℝ) ^ (-((Claims.Flock.Numbers.waits : ℕ) : ℝ)) +
      2 * (⌈4 * (1000 : ℝ) / Real.exp 1⌉₊ : ℝ) / ((Claims.Flock.Numbers.cap : ℕ) : ℝ) +
      Real.exp (-(⌈4 * (1000 : ℝ) / Real.exp 1⌉₊ : ℝ) / 8) ≤ 1 / 10
  rw [hε]
  generalize (⌈4 * (1000 : ℝ) / Real.exp 1⌉₊ : ℝ) = K at hK1 hK2 ⊢
  have hw : (2 : ℝ) ^ (-((Claims.Flock.Numbers.waits : ℕ) : ℝ)) ≤ 1 / 100 := by
    rw [Real.rpow_neg (by norm_num), Real.rpow_natCast, inv_le_comm₀ (by positivity) (by norm_num)]
    simp only [Claims.Flock.Numbers.waits]
    norm_num
  have hc : 2 * K / ((Claims.Flock.Numbers.cap : ℕ) : ℝ) ≤ 1 / 100 := by
    simp only [Claims.Flock.Numbers.cap]
    push_cast
    rw [div_le_iff₀ (by positivity)]
    nlinarith
  have hx : Real.exp (-K / 8) ≤ 1 / 100 := by
    rw [neg_div]
    refine (exp_neg_le_inv (by positivity)).trans ?_
    rw [inv_le_comm₀ (by positivity) (by norm_num)]
    linarith
  linarith

/-- The accepting code's compiled two-stage verifier over `F`. -/
noncomputable abbrev accL {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) :
    Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin (Claims.SamplingGeneric.Arg.ofFlock F) :=
  compiledTwo F accTwo (fun _ _ _ => []) (fun _ => some toyTwo) accRs (fun _ => 0)

theorem accL_scope {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (x : Msg) :
    (accL F).Scope x :=
  ⟨toyStmt, toyPart, toyTwo, rfl, rfl, rfl, inScopeTwo_toy⟩

theorem accL_κ {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (x : Msg) :
    (accL F).κ 0 x < 1 := by
  have hb : (accL F).κb x 0 = κFAt (deployOf F accTwo) x 1000 := rfl
  have hs : (accL F).κs x = max (κFAt (deployOf F accTwo) x 1000) 0 := by
    show ((List.range 1).map ((accL F).κb x)).foldr max 0 = _
    rw [show List.range 1 = [0] by simp]
    exact congrArg (fun t => max t 0) hb
  refine κ_lt_one (accL F).toCompiled x ?_ ?_
  · rw [hs]
    exact le_max_right _ _
  · rw [hs]
    exact max_le (accκF F x) (by norm_num)

/-- **`CompiledTwo.KS` fails at the accepting code**: its silent prover is accepted on every run, no witness is in its
relation, its sessions' finder finds nothing, and `κ 0 < 1`. -/
theorem accL_KS_fails {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)
    (H : Msg → Msg) : ¬ (accL F).KS H := by
  intro h
  have h1 := h 0 le_rfl [] (accL_scope F []) Unit silentP
  have hp : PrBad H (accL F).V (RTwo H (accL F).commit (accL F).parse (accL F).partition (accL F).two 0)
      (accL F).E [] silentP = 1 :=
    prBad_accV2 H _ _ [] silentP (accTwo_noR H [])
  have hb : PrB H (accL F).V silentP [] = 0 := prB_accV2 H silentP []
  have hx : (Claims.Sampling.Bext H (accL F).V (accL F).E Unit silentP []).Pr H = 0 :=
    bext_pr H (accL F).toCompiled (fun _ _ _ => rfl) (fun _ => Nat.two_pos) Unit silentP []
  have hκ := accL_κ F []
  change PrBad H (accL F).V (RTwo H (accL F).commit (accL F).parse (accL F).partition (accL F).two 0) (accL F).E []
      silentP ≤ (accL F).κ 0 [] + Real.sqrt ((accL F).γ [] * PrB H (accL F).V silentP []) +
        1 * (Claims.Sampling.Bext H (accL F).V (accL F).E Unit silentP []).Pr H at h1
  rw [hp, hb, mul_zero, Real.sqrt_zero, add_zero, hx, mul_zero, add_zero] at h1
  linarith

end Flock

/-- **`SampledKSTwo`'s conclusion fails at an instance of its binders where its domain hypothesis `ReplayCap` holds**:
at the accepting code, its stage order, no registration, byte maps that read no bytes, the toy's two-stage design and
the cap `0`, the replay sample of record is empty, so `ReplayCap` holds, and `CompiledTwo.KS` fails at Flock's masked
record over the readable relation and SHA-512 (`accL_KS_fails`). -/
theorem _root_.Flock.SecurityProofs.SampledKSTwo_nontrivial :
    ∃ (c : Code) (reg : Msg → Option RegTwo) (two : AuditStatement → Option Claims.Sampling.Design.Two)
      (L₁ L₂ R : Msg → ℕ) (s : (x : Msg) → StageOrder (c.V.k x) (R x))
      (b₁ : (x : Msg) → FlockDeploy.Coin → Fin (L₁ x) → Fin 256)
      (b₂ : (x : Msg) → (Fin (R x) → FlockDeploy.Coin) → Fin (L₂ x) → Fin 256)
      (iroots : Msg → Seen → ℕ → List Msg) (cap : Msg → ℕ),
      ∃ _ : (c.atOrder (orderOf s)).ReplayCap FlockDeploy.sha512 two
          (replayTwo (c.atOrder (orderOf s)) reg L₁ (bytesFirst c L₁ (orderOf s) b₁)) cap,
        ¬ (compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation)
          (codeTwoStage c reg L₁ L₂ b₁ R s b₂) iroots two
          (replayTwo (c.atOrder (orderOf s)) reg L₁ (bytesFirst c L₁ (orderOf s) b₁)) cap).KS
          FlockDeploy.sha512 :=
  ⟨accCode, fun _ => none, fun _ => some toyTwo, fun _ => 0, fun _ => 0, fun _ => 1, accStage, toyBytes,
    toyStageBytes, fun _ _ _ => [], fun _ => 0, accReplayCap,
    accL_KS_fails (FlockDeploy.deployZk _root_.Flock.Readable.relation) FlockDeploy.sha512⟩

end SamplingDeploy
