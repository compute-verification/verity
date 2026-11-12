module

public import Proofs.Flock.Claims.Sampling.Runs
public import Proofs.Flock.Claims.Sampling.RunsExp
public import Proofs.Flock.SamplingGeneric.Session
public import Proofs.Flock.SamplingGeneric.Bits

/-!
# The compiled extractor's costs

`Compiled.Efficient`'s three bounds at a compiled sampled verifier that meets the contract's facts its cost reads
(`CostContract`: `RunsFixed`, `slots` and `keys`, which `Compiled.Sessions` and `CompiledTwo.Sessions` give), from the
opening checks' count (`Compiled.OpenCallsLe`): `Compiled.E` extracts session by session at their first slots
(`Compiled.runExt`), so it is the sampling extractor at the step `runExt` (`E_eq_Esamp`).

* `Arg.reruns_le`, `Arg.rerunsExp_le`, `Arg.own_le`: every proof system's extractor is `E^rep` at waits, truncation
  and accepted reruns its shape fixes, so it reruns at most `1 + m′·K` times, `1 + m′·k` in expectation (`ERep_cost`,
  on its coins cast along the statement's rounds), and makes no SHA-512 call of its own.

* `Esamp`: the sampling extractor with what it does at each accepted inner run a parameter (`step`), and its three
  bounds from the step's (`Esamp_runs_le`, `Esamp_runs_exp`, `Esamp_own_le`): the replay, the outer wait over
  `K_out` samples and the inner wait over `K_in` at each kept run, then the step. In expectation the step's reruns
  average at most `W` of the inner run, and acceptance times `W` averages at most `B` over the second stage, against
  which the inner wait amortizes (`level_ex`, `level_amortized`).
* `sessExt_runs_eq`, `sessExt_own_le`: under `runs`, a session's extraction reruns exactly as `A`'s extractor against
  the session's prover at the session's coins (each served rerun is one rerun, seen as the session's view), and its own
  SHA-512 calls are `A`'s extractor's and its reveal's checks at each of the session's blocks.
* `slotStep`: a run's extractions session by session of the sample (its first `n_PU` slots, `runExt`), each at its
  first slot's coins. Its strict reruns are at most `n_sess` sessions' and its own calls at most `n_PU` blocks'
  (`runExt_runs_le`, `runExt_own_le`); in expectation, slot `i` weighs at most `e_s` given acceptance where it is its
  session's first, and nothing elsewhere (`slot_exp_le`): resampling its block's session's rounds keeps the sample
  (`fixed`, `Ex_resample`), so the session's coins and the block's coins are uniform beside one prover `P'`, where
  `A.rerunsExp_le` holds.
* `slots_runs_le`, `slots_runs_exp`, `slots_own_le`: `Efficient`'s three bounds at `Esamp` of `slotStep`, and
  `efficient_of`, the three at `Compiled.E` through `E_eq_Esamp`.
* `compilerEfficient`: **`CompilerEfficient`**, `efficient_of` at `Sessions.toCostContract`, so `compilerBits_of`
  gives `CompilerBits`.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Claims.Sampling
open Claims.Recursion (serve viewing)

/-! ## Reruns and own calls read the runs only through what they show -/

theorem eval_runs_congr {R α : Type} (H : Msg → Msg) {run₁ run₂ : R → Transcript}
    (h : ∀ r, (run₁ r).seen = (run₂ r).seen) (p : RP R α) : (p.eval H run₁).2.2 = (p.eval H run₂).2.2 := by
  induction p with
  | ret a => rfl
  | hash q k ih => exact ih (H q)
  | rerun r k ih =>
    simp only [RP.eval]
    rw [h r, ih]

theorem ownLog_congr {R α : Type} (H : Msg → Msg) {run₁ run₂ : R → Transcript}
    (h : ∀ r, (run₁ r).seen = (run₂ r).seen) (p : RP R α) : ownLog H run₁ p = ownLog H run₂ p := by
  induction p with
  | ret a => rfl
  | hash q k ih => simp only [ownLog, ih]
  | rerun r k ih =>
    simp only [ownLog]
    rw [h r, ih]

/-- Served by one rerun each, seen through `f`, a program makes the inner stage's own SHA-512 calls against the served
runs. -/
theorem ownLog_serve_viewing {R₁ R₂ α : Type} (H : Msg → Msg) (f : Seen → Seen) (g : R₁ → R₂) (run : R₂ → Transcript)
    (p : RP R₁ α) :
    ownLog H run (RP.serve (fun r => viewing f (g r)) p) = ownLog H (servedRun H (fun r => viewing f (g r)) run) p := by
  induction p with
  | ret a => rfl
  | hash m k ih => simp only [RP.serve, ownLog, ih]
  | rerun r k ih =>
    simp only [RP.serve, ownLog]
    have hv : ownLog H run (viewing f (g r)) = [] :=
      ownLog_of_noHash H run (NoHash.rerun (g r) _ fun _ => NoHash.ret _)
    rw [ownLog_bind, hv, List.nil_append]
    exact ih _

/-- Served by one rerun each, a program reruns as often as the inner stage against the served runs. -/
theorem serve_place_runs {R₁ R₂ α : Type} (H : Msg → Msg) (f : Seen → Seen) (g : R₁ → R₂) (run : R₂ → Transcript)
    (p : RP R₁ α) :
    ((RP.serve (fun r => viewing f (g r)) p).eval H run).2.2 =
      (p.eval H (servedRun H (fun r => viewing f (g r)) run)).2.2 := by
  rw [(RP.serve_eval H _ run p).2.2, RP.eval_runs_eq H _ p]
  have h1 : ∀ r, ((viewing f (g r) : RP R₂ Seen).eval H run).2.2 = 1 := fun _ => rfl
  simp only [h1, List.map_const', List.sum_replicate, smul_eq_mul, mul_one]

/-- Checking what a witness reveals at each of a list's proof blocks reruns nothing. -/
theorem flatMapRP_checkRevealed_runs {R β : Type} (H : Msg → Msg) (run : R → Transcript) (Cm : CommitmentScheme Bool)
    (f : β → List (List Msg × Opening Bool)) (l : List β) :
    ((flatMapRP (fun b => checkRevealed (R := R) Cm (f b)) l).eval H run).2.2 = 0 :=
  Nat.le_zero.1 ((flatMapRP_runs_le H run _ 0 l fun _ _ => (checkRevealed_runs H run Cm _).le).trans_eq
    (Nat.mul_zero _))

theorem sum_zip_finRange_nat {α : Type} : ∀ (n : ℕ) (F : Fin n × α → ℕ) (l : List α),
    (((List.finRange n).zip l).map F).sum = ∑ i : Fin n, (l[i.val]?.map fun a => F (i, a)).getD 0
  | 0, _, _ => by simp
  | _ + 1, _, [] => by simp
  | n + 1, F, a :: l => by
    rw [List.finRange_succ, List.zip_cons_cons, List.map_cons, List.sum_cons, List.zip_map_left, List.map_map,
      sum_zip_finRange_nat n (F ∘ Prod.map Fin.succ id) l, Fin.sum_univ_succ]
    simp

/-- A wait's kept runs show what their runs show. -/
theorem seen_of_mem_zip {R : Type} (H : Msg → Msg) (run : R → Transcript) (k : ℕ) (rs : List R) {n : ℕ}
    {li : Fin n × (R × Seen)} (h : li ∈ (List.finRange n).zip ((waitAccC k rs []).eval H run).1) :
    li.2.2 = (run li.2.1).seen := by
  obtain ⟨j, q⟩ := li
  have hq := (List.of_mem_zip h).2
  rw [waitAccC_eval_nil] at hq
  obtain ⟨r, -, rfl⟩ := List.mem_map.1 hq
  rfl

theorem le_foldr_max_nat {a : ℕ} : ∀ {l : List ℕ}, a ∈ l → a ≤ l.foldr max 0
  | [], h => by simp at h
  | b :: l, h => by
    rw [List.foldr_cons]
    rcases List.mem_cons.1 h with rfl | h
    · exact le_max_left _ _
    · exact le_max_of_le_right (le_foldr_max_nat h)

theorem le_maxBelowN {n : ℕ} (f : ℕ → ℕ) {u : ℕ} (hu : u < n) : f u ≤ maxBelowN n f :=
  le_foldr_max_nat (List.mem_map.2 ⟨u, List.mem_range.2 hu, rfl⟩)

/-- **One coordinate of uniform dependent coins is uniform**, in expectation. -/
theorem Ex_pi_eval_dep {ι : Type} [Fintype ι] [DecidableEq ι] {E : ι → Type} [∀ i, Fintype (E i)]
    [∀ i, Nonempty (E i)] (i : ι) (f : E i → ℝ) : Ex (fun c : (j : ι) → E j => f (c i)) = Ex f := by
  rw [← Claims.Sampling.Ex_fst (β := (j : {j // j ≠ i}) → E j) f]
  exact Ex_equiv (Equiv.piSplitAt i E) fun p => f p.1

/-- **Uniform coins restricted along an injective map, beside `A`'s extractor's coins cast along an equation of
shapes, are uniform**, in expectation. -/
theorem Ex_restrict_shape {Coin : Type} [Fintype Coin] [Nonempty Coin] {A : Arg Coin} {m n : ℕ}
    (emb : Fin m → Fin n) (hemb : Function.Injective emb) {s s' : A.Sh} (hs : s = s')
    (F : (Fin m → Coin) → A.EC s → ℝ) :
    Ex (fun p : (Fin n → Coin) × A.EC s' => F (fun i => p.1 (emb i)) (cast (congrArg A.EC hs.symm) p.2)) =
      Ex (fun p : (Fin m → Coin) × A.EC s => F p.1 p.2) := by
  subst hs
  rw [Ex_prod, Ex_prod]
  exact Ex_comp_inj emb hemb fun a => Ex (F a)

/-! ## Every proof system's extractor's strict reruns and own calls -/

section NoHash

variable {Coin : Type} [Fintype Coin] [Nonempty Coin]

/-- **A proof system's `E^rep` makes no SHA-512 call of its own**: it replays the run, then its waits rerun. -/
theorem noHash_ext (A : Arg Coin) (x : Msg) (r₀ : Fin (A.V.k x) → Coin) (c : A.EC (A.shape x)) :
    Claims.Sampling.NoHash (A.ext x r₀ c) := by
  refine Claims.Sampling.NoHash.rerun _ _ fun v => ?_
  split
  · exact Claims.Sampling.NoHash.firstWait _ _ _ _ _
  · exact Claims.Sampling.NoHash.ret _

end NoHash

namespace Arg

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (H : Msg → Msg)

/-- **A proof system's extractor reruns at most `1 + m′·K` times**: it is `E^rep` (`ERep_cost`). -/
theorem reruns_le : ExtractRunsLe H A.E fun x => 1 + A.waits (A.shape x) * A.cap (A.shape x) := by
  intro x Ω P ω r c
  exact (ERep_cost H A.V (fun x => A.waits (A.shape x)) (fun x => A.k (A.shape x)) (fun x => A.cap (A.shape x))
    A.opens (fun x os => (A.assemble x os).1) A.check).1 x Ω P ω r fun i j t => c i j (Fin.cast (A.wf x) t)

/-- The extractor's coins at a statement's shape are `E^rep`'s at the statement, cast along its rounds (`wf`). -/
def ecEquiv (x : Msg) :
    A.EC (A.shape x) ≃ (Fin (A.waits (A.shape x)) → Fin (A.cap (A.shape x)) → Fin (A.V.k x) → Coin) where
  toFun c i j t := c i j (Fin.cast (A.wf x) t)
  invFun c i j t := c i j (Fin.cast (A.wf x).symm t)
  left_inv _ := rfl
  right_inv _ := rfl

/-- **A proof system's extractor reruns at most `1 + m′·k` times in expectation**: it is `E^rep` (`ERep_cost`) on its
coins cast along the statement's rounds (`ecEquiv`). -/
theorem rerunsExp_le : ExtractRunsExpLe H A.E fun x => 1 + A.waits (A.shape x) * A.k (A.shape x) := by
  intro x Ω P ω
  exact le_of_eq_of_le (Ex_equiv ((Equiv.refl (Fin (A.V.k x) → Coin)).prodCongr (A.ecEquiv x)) fun s =>
    (((ERep A.V (fun x => A.waits (A.shape x)) (fun x => A.k (A.shape x)) (fun x => A.cap (A.shape x)) A.opens
      (fun x os => (A.assemble x os).1) A.check).eval H P x ω s.1 s.2).2.2 : ℝ))
    ((ERep_cost H A.V (fun x => A.waits (A.shape x)) (fun x => A.k (A.shape x)) (fun x => A.cap (A.shape x))
      A.opens (fun x os => (A.assemble x os).1) A.check).2 x Ω P ω)

/-- **A proof system's extractor makes no SHA-512 call of its own**: `E^rep`'s waits rerun, and its assemble step and
check are pure. -/
theorem own_le : ExtractOwnLe H A.E fun _ => 0 := by
  intro x Ω P ω r₀ c
  have hn : NoHash (A.E.ext x r₀ c) := by
    refine NoHash.rerun _ _ fun v => ?_
    split
    · exact NoHash.firstWait _ _ _ _ _
    · exact NoHash.ret _
  exact (congrArg List.length (ownLog_of_noHash H _ hn)).le

end Arg

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## The sampling extractor at a step -/

/-- What the sampling extractor does at an accepted inner run `(r, v)`, on its coins `Y x`, at the outer run's index
`j` and the inner run's `l`. -/
abbrev Step (L : Compiled Coin A) (Y : Msg → Type) : Type :=
  (x : Msg) → AuditStatement → Y x → Fin (L.kOut x) → Fin (L.kIn x) → (Fin (L.V.k x) → Coin) → Seen →
    RP (Fin (L.V.k x) → Coin) (List (List Msg × Opening Bool))

/-- At an accepted outer run `jo`, on the inner waits' coins and the steps' (`c`): the inner wait fresh on the second
stage, and the step at each kept inner run. -/
noncomputable def sampInner (L : Compiled Coin A) (Y : Msg → Type) (step : L.Step Y) (x : Msg) (s : AuditStatement)
    (c : (Fin (L.kOut x) → Fin (L.capIn x) → Fin (L.V.k x) → Coin) × Y x)
    (jo : Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen)) :
    RP (Fin (L.V.k x) → Coin) (List (List Msg × Opening Bool)) :=
  (waitAccC (L.kIn x) ((List.finRange (L.capIn x)).map fun l => forkOn (L.stage2 x) jo.2.1 (c.1 jo.1 l)) []).bind
    fun inner => flatMapRP (fun li : Fin (L.kIn x) × ((Fin (L.V.k x) → Coin) × Seen) =>
      step x s c.2 jo.1 li.1 li.2.1 li.2.2) ((List.finRange (L.kIn x)).zip inner)

/-- **The sampling extractor at a step** (`Compiled.E` with its innermost part `step`): replay the judged run; if it
accepted, rerun on fresh coins until `k_out` accepted runs are in hand (giving up after `K_out`); at each, rerun fresh
on the second stage until `k_in` accepted runs are in hand (giving up after `K_in`); at each of those, the step; and
assemble. -/
noncomputable def Esamp (L : Compiled Coin A) (Y : Msg → Type) [∀ x, Fintype (Y x)] [∀ x, Nonempty (Y x)]
    (step : L.Step Y) : Extractor Coin Sampling.Wit L.V where
  C x := (Fin (L.capOut x) → Fin (L.V.k x) → Coin) × (Fin (L.kOut x) → Fin (L.capIn x) → Fin (L.V.k x) → Coin) × Y x
  ext x r₀ c := .rerun r₀ fun v₀ =>
    match L.parse x with
    | none => .ret none
    | some s =>
      if v₀.out = .acc then
        (waitAccC (L.kOut x) ((List.finRange (L.capOut x)).map c.1) []).bind fun outer =>
          (flatMapRP (L.sampInner Y step x s c.2) ((List.finRange (L.kOut x)).zip outer)).bind fun found =>
            .ret (some (assemble s found))
      else .ret none

section Samp

variable (H : Msg → Msg) (L : Compiled Coin A) (Y : Msg → Type) [∀ x, Fintype (Y x)] [∀ x, Nonempty (Y x)]
  (step : L.Step Y)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [∀ x, Fintype (Y x)] [∀ x, Nonempty (Y x)] in
theorem sampInner_runs_le {x : Msg} {s : AuditStatement} (run' : (Fin (L.V.k x) → Coin) → Transcript) (B : ℕ)
    (hstep : ∀ c j l r, ((step x s c j l r (run' r).seen).eval H run').2.2 ≤ B)
    (c : (Fin (L.kOut x) → Fin (L.capIn x) → Fin (L.V.k x) → Coin) × Y x)
    (jo : Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen)) :
    ((L.sampInner Y step x s c jo).eval H run').2.2 ≤ L.capIn x + L.kIn x * B := by
  unfold sampInner
  rw [(RP.eval_bind H run' _ _).2, waitAccC_runs_eq]
  refine Nat.add_le_add ((waitRuns_le_length _ _ _ _).trans_eq (by simp)) ?_
  refine (flatMapRP_runs_le H run' _ B _ fun li hli => ?_).trans
    (Nat.mul_le_mul_right _ (length_zip_finRange_le _ _))
  rw [seen_of_mem_zip H run' _ _ hli]
  exact hstep _ _ _ _

/-- **`ExtractRunsLe` at the sampling extractor**: at most `1 + K_out + k_out·(K_in + k_in·B)` reruns, when each step
at an inner run reruns at most `B`. -/
theorem Esamp_runs_le (B : Msg → ℕ)
    (hstep : ∀ x s (Ω : Type) (P : Prover Ω) (ω : Ω), L.parse x = some s → ∀ c j l r,
      ((step x s c j l r (run H P L.V x ω r).seen).eval H (run H P L.V x ω)).2.2 ≤ B x) :
    ExtractRunsLe H (L.Esamp Y step) (fun x => 1 + L.capOut x + L.kOut x * (L.capIn x + L.kIn x * B x)) := by
  intro x Ω P ω r₀ c
  simp only [Extractor.eval, Esamp, RP.eval]
  split
  · simp only [RP.eval, Nat.zero_add]
    rw [Nat.add_assoc]
    exact Nat.le_add_right 1 _
  · rename_i s hs
    split
    · have h := outer_runs_le H (run H P L.V x ω) (L.kOut x) (L.capOut x) (L.capIn x + L.kIn x * B x)
        ((List.finRange (L.capOut x)).map c.1) (by rw [List.length_map, List.length_finRange])
        (L.sampInner Y step x s c.2) (fun found => some (assemble s found))
        (fun jo => sampInner_runs_le H L Y step (run H P L.V x ω) (B x) (hstep x s Ω P ω hs) c.2 jo)
      exact (Nat.add_le_add_right h 1).trans_eq (by ring)
    · simp only [RP.eval, Nat.zero_add]
      rw [Nat.add_assoc]
      exact Nat.le_add_right 1 _

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [∀ x, Fintype (Y x)] [∀ x, Nonempty (Y x)] in
theorem sampInner_own_le {x : Msg} {s : AuditStatement} (run' : (Fin (L.V.k x) → Coin) → Transcript) (B : ℕ)
    (hstep : ∀ c j l r, (ownLog H run' (step x s c j l r (run' r).seen)).length ≤ B)
    (c : (Fin (L.kOut x) → Fin (L.capIn x) → Fin (L.V.k x) → Coin) × Y x)
    (jo : Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen)) :
    (ownLog H run' (L.sampInner Y step x s c jo)).length ≤ L.kIn x * B := by
  unfold sampInner
  rw [ownLog_bind, ownLog_of_noHash H run' (NoHash.waitAccC _ _ _), List.nil_append]
  refine (ownLog_flatMapRP_le H run' _ B _ fun li hli => ?_).trans
    (Nat.mul_le_mul_right _ (length_zip_finRange_le _ _))
  rw [seen_of_mem_zip H run' _ _ hli]
  exact hstep _ _ _ _

/-- **`ExtractOwnLe` at the sampling extractor**: at most `k_out·k_in·B` own calls, when each step at an inner run makes
at most `B`; the waits make none. -/
theorem Esamp_own_le (B : Msg → ℕ)
    (hstep : ∀ x s (Ω : Type) (P : Prover Ω) (ω : Ω), L.parse x = some s → ∀ c j l r,
      (ownLog H (run H P L.V x ω) (step x s c j l r (run H P L.V x ω r).seen)).length ≤ B x) :
    ExtractOwnLe H (L.Esamp Y step) (fun x => L.kOut x * (L.kIn x * B x)) := by
  intro x Ω P ω r₀ c
  simp only [Esamp, ownLog]
  split
  · simp [ownLog]
  · rename_i s hs
    split
    · rw [ownLog_bind, ownLog_of_noHash _ _ (NoHash.waitAccC _ _ _), List.nil_append, ownLog_bind]
      simp only [ownLog, List.append_nil]
      exact (ownLog_flatMapRP_le H _ _ _ _ fun jo _ =>
        sampInner_own_le H L Y step _ (B x) (hstep x s Ω P ω hs) c.2 jo).trans
        (Nat.mul_le_mul_right _ (length_zip_finRange_le _ _))
    · simp [ownLog]

/-- An outer body's reruns at outer index `j` and run `r`, on its inner wait's coins `d` and the steps' `ξ`: the inner
wait's, and the steps' at its kept runs. -/
noncomputable def sampB {x : Msg} (s : AuditStatement) (run' : (Fin (L.V.k x) → Coin) → Transcript) (j : Fin (L.kOut x))
    (r : Fin (L.V.k x) → Coin) (d : Fin (L.capIn x) → Fin (L.V.k x) → Coin) (ξ : Y x) : ℝ :=
  (waitRuns run' (L.kIn x) ((List.finRange (L.capIn x)).map fun l => forkOn (L.stage2 x) r (d l)) 0 : ℝ) +
    (((List.finRange (L.kIn x)).zip ((waitAccC (L.kIn x) ((List.finRange (L.capIn x)).map fun l =>
      forkOn (L.stage2 x) r (d l)) []).eval H run').1).map fun li =>
        (((step x s ξ j li.1 li.2.1 (run' li.2.1).seen).eval H run').2.2 : ℝ)).sum

/-- An outer body's weight given its run `r`: the inner wait's expected reruns fresh on the second stage, and its kept
runs' weights `W`. -/
noncomputable def sampG {x : Msg} (run' : (Fin (L.V.k x) → Coin) → Transcript) (W : (Fin (L.V.k x) → Coin) → ℝ)
    (r : Fin (L.V.k x) → Coin) : ℝ :=
  waitExp (run' ∘ forkOn (L.stage2 x) r) (L.kIn x) (L.capIn x) 0 +
    accExp (run' ∘ forkOn (L.stage2 x) r) (L.kIn x) (W ∘ forkOn (L.stage2 x) r) (L.capIn x) 0

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [∀ x, Fintype (Y x)] [∀ x, Nonempty (Y x)] in
theorem sampInner_runs_eq {x : Msg} (s : AuditStatement) (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (c : (Fin (L.kOut x) → Fin (L.capIn x) → Fin (L.V.k x) → Coin) × Y x)
    (jo : Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen)) :
    (((L.sampInner Y step x s c jo).eval H run').2.2 : ℝ) = sampB H L Y step s run' jo.1 jo.2.1 (c.1 jo.1) c.2 := by
  unfold sampInner sampB
  rw [wait_flat_runs, Nat.cast_add, Nat.cast_list_sum, List.map_map, add_right_inj]
  refine congrArg List.sum (List.map_congr_left fun li hli => ?_)
  simp only [Function.comp_apply]
  rw [seen_of_mem_zip H run' _ _ hli]

omit [Nonempty Coin] [Inhabited Coin] in
theorem sampG_nonneg {x : Msg} (run' : (Fin (L.V.k x) → Coin) → Transcript) {W : (Fin (L.V.k x) → Coin) → ℝ}
    (hW : ∀ r, 0 ≤ W r) (r : Fin (L.V.k x) → Coin) : 0 ≤ sampG L run' W r :=
  add_nonneg (waitExp_nonneg _ _ _ _) (accExp_nonneg _ _ (fun _ => hW _) _ _)

omit [Inhabited Coin] in
/-- An outer body's reruns average at most `sampG` of its run: `level_ex` at the inner wait, fresh on the second
stage. -/
theorem Ex_sampB_le {x : Msg} (s : AuditStatement) (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (W : (Fin (L.V.k x) → Coin) → ℝ) (hW : ∀ r, 0 ≤ W r)
    (hstep : ∀ j l r, Ex (fun ξ : Y x => (((step x s ξ j l r (run' r).seen).eval H run').2.2 : ℝ)) ≤ W r)
    (j : Fin (L.kOut x)) (r : Fin (L.V.k x) → Coin) :
    Ex (fun c2 : (Fin (L.kOut x) → Fin (L.capIn x) → Fin (L.V.k x) → Coin) × Y x =>
      sampB H L Y step s run' j r (c2.1 j) c2.2) ≤ sampG L run' W r := by
  have hslice : Ex (fun c2 : (Fin (L.kOut x) → Fin (L.capIn x) → Fin (L.V.k x) → Coin) × Y x =>
        sampB H L Y step s run' j r (c2.1 j) c2.2) =
      Ex (fun c : (Fin (L.capIn x) → Fin (L.V.k x) → Coin) × Y x => sampB H L Y step s run' j r c.1 c.2) := by
    rw [Ex_prod, Ex_prod]
    exact Ex_pi_eval j fun d => Ex fun ξ => sampB H L Y step s run' j r d ξ
  rw [hslice]
  exact level_ex (D' := Fin (L.V.k x) → Coin) (R := Fin (L.V.k x) → Coin) (X := Y x) H run'
    (forkOn (L.stage2 x) r) (L.kIn x) (L.capIn x)
    (fun l r' ξ => (((step x s ξ j l r' (run' r').seen).eval H run').2.2 : ℝ)) W (hstep j) hW

omit [Inhabited Coin] in
/-- The extractor's reruns at a statement that parses: the replay, and at an accepted judged run the outer wait and
each kept run's body. -/
theorem Esamp_runs_le_outer {x : Msg} {s : AuditStatement} (hs : L.parse x = some s)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (r₀ : Fin (L.V.k x) → Coin) (c : (L.Esamp Y step).C x) :
    ((((L.Esamp Y step).ext x r₀ c).eval H run').2.2 : ℝ) ≤
      1 + (if (run' r₀).out = .acc then (1 : ℝ) else 0) *
        ((waitRuns run' (L.kOut x) ((List.finRange (L.capOut x)).map c.1) 0 : ℝ) +
          (((List.finRange (L.kOut x)).zip ((waitAccC (L.kOut x) ((List.finRange (L.capOut x)).map c.1) []).eval H
            run').1).map fun jo => sampB H L Y step s run' jo.1 jo.2.1 (c.2.1 jo.1) c.2.2).sum) := by
  simp only [Esamp, RP.eval]
  split
  · rename_i hn
    rw [hs] at hn
    exact absurd hn (Option.some_ne_none s)
  · rename_i s' hs'
    obtain rfl : s = s' := Option.some.inj (hs.symm.trans hs')
    split
    · rename_i ha
      have ha' : (run' r₀).out = .acc := ha
      have h := outer_runs_le_sum H run' (L.kOut x) ((List.finRange (L.capOut x)).map c.1)
        (L.sampInner Y step x s c.2)
        (fun found => some (assemble s found)) (fun jo => sampB H L Y step s run' jo.1 jo.2.1 (c.2.1 jo.1) c.2.2)
        (fun jo => (sampInner_runs_eq H L Y step s run' c.2 jo).le)
      rw [ite_eq_left ha', one_mul, Nat.cast_add, Nat.cast_one, add_comm (1 : ℝ)]
      exact add_le_add h le_rfl
    · rename_i ha
      have ha' : ¬ (run' r₀).out = .acc := ha
      rw [ite_eq_right ha', zero_mul, add_zero, Nat.cast_add, Nat.cast_one]
      simp only [RP.eval, Nat.cast_zero, zero_add, le_refl]

/-- **`ExtractRunsExpLe` at the sampling extractor**, at one statement, prover and tape: at most
`1 + k_out + k_out·k_in·(1 + B)` reruns in expectation, when each step at an inner run `r` averages at most `W r` reruns
over its coins, and acceptance times `W` averages at most `B` over the second stage at every run. -/
theorem Esamp_runs_exp (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (W : AuditStatement → (Fin (L.V.k x) → Coin) → ℝ) (B : ℝ) (hB0 : 0 ≤ B) (hW : ∀ s r, 0 ≤ W s r)
    (hstep : ∀ s, L.parse x = some s → ∀ j l r,
      Ex (fun ξ : Y x => (((step x s ξ j l r (run H P L.V x ω r).seen).eval H (run H P L.V x ω)).2.2 : ℝ)) ≤ W s r)
    (hB : ∀ s, L.parse x = some s → ∀ r, Ex (fun d : Fin (L.V.k x) → Coin =>
      (if (run H P L.V x ω (forkOn (L.stage2 x) r d)).out = .acc then (1 : ℝ) else 0) *
        W s (forkOn (L.stage2 x) r d)) ≤ B) :
    Ex (fun p : (Fin (L.V.k x) → Coin) × (L.Esamp Y step).C x =>
      (((L.Esamp Y step).eval H P x ω p.1 p.2).2.2 : ℝ)) ≤ 1 + L.kOut x + (L.kOut x : ℝ) * L.kIn x * (1 + B) := by
  have hk0 : (0 : ℝ) ≤ L.kOut x := Nat.cast_nonneg _
  have hki0 : (0 : ℝ) ≤ L.kIn x := Nat.cast_nonneg _
  cases hp : L.parse x with
  | none =>
    have h1 : ∀ p : (Fin (L.V.k x) → Coin) × (L.Esamp Y step).C x,
        (((L.Esamp Y step).eval H P x ω p.1 p.2).2.2 : ℝ) = 1 := by
      intro p
      simp only [Extractor.eval, Esamp, RP.eval]
      split
      · simp [RP.eval]
      · rename_i s' hs'
        rw [hp] at hs'
        exact absurd hs'.symm (Option.some_ne_none s')
    simp only [h1, Ex_const]
    nlinarith [mul_nonneg (mul_nonneg hk0 hki0) (add_nonneg zero_le_one hB0)]
  | some s =>
    let run' := run H P L.V x ω
    let Rout : (L.Esamp Y step).C x → ℝ := fun c =>
      (waitRuns run' (L.kOut x) ((List.finRange (L.capOut x)).map c.1) 0 : ℝ) +
        (((List.finRange (L.kOut x)).zip ((waitAccC (L.kOut x) ((List.finRange (L.capOut x)).map c.1) []).eval H
          run').1).map fun jo => sampB H L Y step s run' jo.1 jo.2.1 (c.2.1 jo.1) c.2.2).sum
    have hpt : ∀ p : (Fin (L.V.k x) → Coin) × (L.Esamp Y step).C x,
        (((L.Esamp Y step).eval H P x ω p.1 p.2).2.2 : ℝ) ≤
          1 + (if (run' p.1).out = .acc then (1 : ℝ) else 0) * Rout p.2 :=
      fun p => Esamp_runs_le_outer H L Y step hp run' p.1 p.2
    have hsplit : Ex (fun p : (Fin (L.V.k x) → Coin) × (L.Esamp Y step).C x =>
        1 + (if (run' p.1).out = .acc then (1 : ℝ) else 0) * Rout p.2) =
        1 + Pr (fun r => (run' r).out = .acc) * Ex Rout := by
      rw [Ex_add, Ex_const, Ex_prod]
      congr 1
      have hin : ∀ a : Fin (L.V.k x) → Coin, (Ex fun b : (L.Esamp Y step).C x =>
          (if (run' a).out = .acc then (1 : ℝ) else 0) * Rout b) =
          (if (run' a).out = .acc then (1 : ℝ) else 0) * Ex Rout := fun a => Ex_mul_left _ Rout
      refine (congrArg Ex (funext hin)).trans ?_
      rw [Ex_mul_right, Ex_ind]
    have hW' : ∀ r, 0 ≤ W s r := hW s
    have hout : Pr (fun r => (run' r).out = .acc) * Ex Rout ≤
        L.kOut x + L.kOut x * Ex (fun r => (if (run' r).out = .acc then (1 : ℝ) else 0) * sampG L run' (W s) r) := by
      have hle := level_ex (D' := Fin (L.V.k x) → Coin) (R := Fin (L.V.k x) → Coin)
        (X := (Fin (L.kOut x) → Fin (L.capIn x) → Fin (L.V.k x) → Coin) × Y x) H run' id (L.kOut x) (L.capOut x)
        (fun j r c2 => sampB H L Y step s run' j r (c2.1 j) c2.2) (sampG L run' (W s))
        (fun j r => Ex_sampB_le H L Y step s run' (W s) hW' (hstep s hp) j r) (sampG_nonneg L run' hW')
      refine (mul_le_mul_of_nonneg_left hle (Pr_nonneg _)).trans ?_
      exact level_amortized run' id (L.kOut x) (L.capOut x) (sampG L run' (W s)) (sampG_nonneg L run' hW')
    have hacc : Ex (fun r => (if (run' r).out = .acc then (1 : ℝ) else 0) * sampG L run' (W s) r) ≤
        L.kIn x + L.kIn x * B := by
      rw [Ex_acc_forkOn (L.stage2 x) (fun r => (run' r).out = .acc) (sampG L run' (W s)) fun r d => by
        unfold sampG
        rw [forkOn_forkOn']]
      refine (Ex_mono fun r => ?_).trans_eq (Ex_const _)
      exact (level_amortized run' (forkOn (L.stage2 x) r) (L.kIn x) (L.capIn x) (W s) hW').trans
        (add_le_add le_rfl (mul_le_mul_of_nonneg_left (hB s hp r) hki0))
    calc Ex (fun p : (Fin (L.V.k x) → Coin) × (L.Esamp Y step).C x =>
          (((L.Esamp Y step).eval H P x ω p.1 p.2).2.2 : ℝ))
        ≤ 1 + Pr (fun r => (run' r).out = .acc) * Ex Rout := (Ex_mono hpt).trans_eq hsplit
      _ ≤ 1 + (L.kOut x + L.kOut x * (L.kIn x + L.kIn x * B)) :=
          add_le_add le_rfl (hout.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_left hacc hk0)))
      _ = 1 + L.kOut x + (L.kOut x : ℝ) * L.kIn x * (1 + B) := by ring

end Samp

/-! ## A sampled proof block's extraction -/

omit [Fintype Coin] [Nonempty Coin] in
/-- Under `runs`' session at `r0`, the served runs at coins `r` agreeing with `r0` off the session's rounds show what the
session's prover's runs show. -/
theorem served_seen (H : Msg → Msg) (L : Compiled Coin A) {x : Msg} {Ω : Type} (P : Prover Ω) (ω : Ω) {u : ℕ}
    {xS : Msg} (P' : Prover Unit) (r0 r : Fin (L.V.k x) → Coin) (hcard : A.V.k xS = (L.rounds x u).card)
    (hview : ∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, (∀ t' ∈ L.rounds x u, t < t') → r' t = r0 t) →
      L.session x u (run H P L.V x ω r').seen = some (xS, (run H P' A.V xS () (L.coinsOf x u xS r')).seen))
    (hr : ∀ t ∉ L.rounds x u, r t = r0 t) (d : Fin (A.V.k xS) → Coin) :
    (servedRun H (fun r' => viewing (L.view x u) (L.place x u xS r r')) (run H P L.V x ω) d).seen =
      (run H P' A.V xS () d).seen := by
  rw [servedRun_seen]
  show L.view x u (run H P L.V x ω (L.place x u xS r d)).seen = (run H P' A.V xS () d).seen
  unfold view
  rw [hview _ fun t ht _ => (L.place_of_not_mem r d ht).trans (hr t ht), L.coinsOf_place hcard]
  rfl

/-- **A session's extraction reruns as `A`'s extractor** against `runs`' session prover `P'`, at the session's coins of
the run and its first block's coins cast to the session's shape: each served rerun is one rerun, and checking what the
witness reveals at the session's blocks reruns nothing. -/
theorem sessExt_runs_eq (H : Msg → Msg) (L : Compiled Coin A) {x : Msg} {Ω : Type} (P : Prover Ω) (ω : Ω) {u : ℕ}
    {xS : Msg} (P' : Prover Unit) (r0 : Fin (L.V.k x) → Coin) (hshape : A.shape xS = L.sh x u)
    (hcard : A.V.k xS = (L.rounds x u).card)
    (hview : ∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, (∀ t' ∈ L.rounds x u, t < t') → r' t = r0 t) →
      L.session x u (run H P L.V x ω r').seen = some (xS, (run H P' A.V xS () (L.coinsOf x u xS r')).seen))
    {r : Fin (L.V.k x) → Coin} (hr : ∀ t ∉ L.rounds x u, r t = r0 t) (s : AuditStatement) (us : List ℕ)
    (c : A.EC (L.sh x u)) :
    ((L.sessExt x s us u r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).2.2 =
      (A.E.eval H P' xS () (L.coinsOf x u xS r) (cast (congrArg A.EC hshape.symm) c)).2.2 := by
  have hv := hview r fun t ht _ => hr t ht
  unfold sessExt sessWit
  simp only [hv]
  rw [dite_eq_left hshape, (RP.eval_bind H _ _ _).2, Claims.Recursion.serve_eq, serve_place_runs,
    eval_runs_congr H (L.served_seen H P ω P' r0 r hcard hview hr)]
  split
  · rfl
  · rw [flatMapRP_checkRevealed_runs, Nat.add_zero]
    rfl

/-- **A session's extraction's own SHA-512 calls** are `A`'s extractor's against `runs`' session prover `P'`, and its
reveal's checks at the session's blocks `us`, at most `openCalls` a block. -/
theorem sessExt_own_le (H : Msg → Msg) (L : Compiled Coin A) {x : Msg} {Ω : Type} (P : Prover Ω) (ω : Ω) {u : ℕ}
    {xS : Msg} (P' : Prover Unit) (r0 : Fin (L.V.k x) → Coin) (hshape : A.shape xS = L.sh x u)
    (hcard : A.V.k xS = (L.rounds x u).card)
    (hview : ∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, (∀ t' ∈ L.rounds x u, t < t') → r' t = r0 t) →
      L.session x u (run H P L.V x ω r').seen = some (xS, (run H P' A.V xS () (L.coinsOf x u xS r')).seen))
    {r : Fin (L.V.k x) → Coin} (hr : ∀ t ∉ L.rounds x u, r t = r0 t) (hopen : L.OpenCallsLe H) {s : AuditStatement}
    (hs : L.parse x = some s) (us : List ℕ) (c : A.EC (L.sh x u)) :
    (ownLog H (run H P L.V x ω) (L.sessExt x s us u r (run H P L.V x ω r).seen c)).length ≤
      (ownLog H (run H P' A.V xS ()) (A.ext xS (L.coinsOf x u xS r) (cast (congrArg A.EC hshape.symm) c))).length +
        us.length * L.openCalls x := by
  have hv := hview r fun t ht _ => hr t ht
  unfold sessExt sessWit
  simp only [hv]
  rw [dite_eq_left hshape, ownLog_bind, Claims.Recursion.serve_eq, ownLog_serve_viewing,
    ownLog_congr H (L.served_seen H P ω P' r0 r hcard hview hr), List.length_append]
  refine Nat.add_le_add_left ?_ _
  split
  · exact Nat.zero_le _
  · refine ownLog_flatMapRP_le H _ _ _ us fun u' _ => ?_
    rw [ownLog_checkRevealed_length]
    exact hopen x s hs u' _ _

/-! ## Slot by slot -/

/-- The slots' coins at each accepted inner run, indexed (outer, inner, slot). -/
abbrev SlotY (L : Compiled Coin A) (x : Msg) : Type := Fin (L.kOut x) × Fin (L.kIn x) × Fin (L.nPU x) → L.BlockEC x

/-- A run's extractions (`runExt`, session by session) as the sampling extractor's step. -/
noncomputable def slotStep (L : Compiled Coin A) : L.Step L.SlotY := fun x s c j l r v =>
  L.runExt x s r v fun i => c (j, l, i)

/-- **The compiled extractor is the sampling extractor at `slotStep`**, the one fact about `Compiled.E` this file's
bounds read. -/
theorem E_eq_Esamp (L : Compiled Coin A) : L.E = L.Esamp L.SlotY L.slotStep := rfl

/-- **The contract's facts the extractor's cost reads**: the sessions' runs on their fixed rounds (`RunsFixed`), and
the sample's slots, at most `n_PU` (`slots`) holding at most `n_sess` keys (`keys`). Both contracts give them
(`Sessions.toCostContract`, `CompiledTwo.Sessions.toCostContract`). -/
structure CostContract (H : Msg → Msg) (L : Compiled Coin A) : Prop extends RunsFixed H L where
  slots : ∀ x r, (L.checked x r).length ≤ L.nPU x
  keys : ∀ x r, ((Finset.range (L.checked x r).length).image (L.skey x (L.checked x r))).card ≤ L.nSess x

omit [Nonempty Coin] in
theorem Sessions.toCostContract {H : Msg → Msg} {L : Compiled Coin A} (hS : L.Sessions H) : L.CostContract H :=
  { toRunsFixed := hS.toRunsFixed, slots := hS.slots, keys := hS.keys }

variable {H : Msg → Msg} {L : Compiled Coin A}

/-- **Strict reruns of a run's extractions**: a session's first slot reruns at most `ρ_s`, any other slot nothing, and
a sample has at most `n_sess` first slots (`card_first_le`). -/
theorem runExt_runs_le (hS : L.CostContract H) {x : Msg} {Ω : Type} (P : Prover Ω) (ω : Ω) (s : AuditStatement)
    (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) :
    ((L.runExt x s r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).2.2 ≤ L.nSess x * L.ρs x := by
  unfold runExt
  rw [flatMapRP_runs, sum_zip_finRange_nat]
  refine (Finset.sum_le_sum (g := fun i : Fin (L.nPU x) =>
    if (L.sessSlots x (L.checked x r) i).head? = some i.val then L.ρs x else 0) fun i _ => ?_).trans ?_
  · cases hl : (L.checked x r)[i.val]? with
    | none => exact Nat.zero_le _
    | some u =>
      simp only [Option.map_some, Option.getD_some]
      by_cases hu : u < L.nPU x ∧ (L.sessSlots x (L.checked x r) i).head? = some i.val
      · rw [dite_eq_left hu, ite_eq_left hu.2]
        obtain ⟨xS, P', hshape, -, -, hcard, -, hview⟩ := hS.runs x Ω P ω r u (List.mem_of_getElem? hl)
        rw [L.sessExt_runs_eq H P ω P' r hshape hcard (fun r' h => (hview r' h).1) (fun _ _ => rfl) s _ _]
        refine (A.reruns_le H xS Unit P' () _ _).trans ?_
        show 1 + A.waits (A.shape xS) * A.cap (A.shape xS) ≤ L.ρs x
        rw [hshape]
        exact le_maxBelowN (fun u => 1 + A.waits (L.sh x u) * A.cap (L.sh x u)) hu.1
      · rw [dite_eq_right hu]
        exact Nat.zero_le _
  · rw [← Finset.sum_filter, Finset.sum_const, smul_eq_mul]
    exact Nat.mul_le_mul_right _ (card_first_le hS.keys x r)

/-- **Own calls of a run's extractions**: a session's first slot checks what its witness reveals at each of the
session's blocks, so the sample's slots, at most `n_PU` (`sum_first_le`, `CostContract.slots`), each at most
`openCalls`. -/
theorem runExt_own_le (hS : L.CostContract H) (hopen : L.OpenCallsLe H) {x : Msg} {s : AuditStatement}
    (hs : L.parse x = some s) {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) :
    (ownLog H (run H P L.V x ω) (L.runExt x s r (run H P L.V x ω r).seen c)).length ≤ L.nPU x * L.openCalls x := by
  unfold runExt
  rw [ownLog_flatMapRP, sum_zip_finRange_nat]
  refine (Finset.sum_le_sum (g := fun i : Fin (L.nPU x) =>
    if (L.sessSlots x (L.checked x r) i).head? = some i.val then
      (L.sessSlots x (L.checked x r) i).length * L.openCalls x else 0) fun i _ => ?_).trans ?_
  · cases hl : (L.checked x r)[i.val]? with
    | none => exact Nat.zero_le _
    | some u =>
      simp only [Option.map_some, Option.getD_some]
      by_cases hu : u < L.nPU x ∧ (L.sessSlots x (L.checked x r) i).head? = some i.val
      · rw [dite_eq_left hu, ite_eq_left hu.2]
        obtain ⟨xS, P', hshape, -, -, hcard, -, hview⟩ := hS.runs x Ω P ω r u (List.mem_of_getElem? hl)
        refine (L.sessExt_own_le H P ω P' r hshape hcard (fun r' h => (hview r' h).1) (fun _ _ => rfl) hopen hs _
          _).trans ?_
        refine (Nat.add_le_add_right (A.own_le H xS Unit P' () _ _) _).trans ?_
        exact (Nat.zero_add _).trans_le (Nat.mul_le_mul_right _ (List.length_filterMap_le _ _))
      · rw [dite_eq_right hu]
        exact Nat.zero_le _
  · rw [← Finset.sum_filter, ← Finset.sum_mul]
    exact Nat.mul_le_mul_right _ ((L.sum_first_le x (L.checked x r)).trans (hS.slots x r))

/-- Slot `i`'s extraction's reruns at the run `r` on the slot's coins `ρ`: its session's, when it is the session's first
slot; `0` otherwise, and when fewer blocks are sampled. -/
noncomputable def slotRuns (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (i : Fin (L.nPU x)) (r : Fin (L.V.k x) → Coin) (ρ : L.BlockEC x) : ℝ :=
  (((L.checked x r)[i.val]?).map fun u =>
    if hu : u < L.nPU x ∧ (L.sessSlots x (L.checked x r) i).head? = some i.val then
      (((L.sessExt x s (L.sessBlocks x (L.checked x r) i) u r (run' r).seen (ρ ⟨u, hu.1⟩)).eval H run').2.2 : ℝ)
    else 0).getD 0

theorem runExt_runs_eq (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) :
    (((L.runExt x s r (run' r).seen c).eval H run').2.2 : ℝ) = ∑ i, L.slotRuns H x s run' i r (c i) := by
  unfold runExt
  rw [flatMapRP_runs, Nat.cast_list_sum, List.map_map]
  refine Eq.trans ?_ (sum_zip_finRange (L.nPU x) (fun iu =>
    if hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val then
      (((L.sessExt x s (L.sessBlocks x (L.checked x r) iu.1) iu.2 r (run' r).seen (c iu.1 ⟨iu.2, hu.1⟩)).eval H
        run').2.2 : ℝ)
    else 0) (L.checked x r))
  refine congrArg List.sum (List.map_congr_left fun iu _ => ?_)
  simp only [Function.comp_apply]
  split
  · rfl
  · exact Nat.cast_zero

theorem slotRuns_nonneg (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (i : Fin (L.nPU x)) (r : Fin (L.V.k x) → Coin) (ρ : L.BlockEC x) :
    0 ≤ L.slotRuns H x s run' i r ρ := by
  unfold slotRuns
  cases (L.checked x r)[i.val]? with
  | none => exact le_rfl
  | some u =>
    simp only [Option.map_some, Option.getD_some]
    split
    · exact Nat.cast_nonneg _
    · exact le_rfl

/-- Slot `i`, as a selection of the sample's slots (`Compiled.selSlot`), when it is its session's first. -/
def firstSel (L : Compiled Coin A) (x : Msg) (i : Fin (L.nPU x)) : List ℕ → Option (Fin (L.nPU x)) := fun l =>
  if (L.sessSlots x l i).head? = some i.val then some i else none

theorem slotRuns_none (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (i : Fin (L.nPU x)) {r : Fin (L.V.k x) → Coin}
    (h : L.selSlot x (L.firstSel x i) r = none) (ρ : L.BlockEC x) : L.slotRuns H x s run' i r ρ = 0 := by
  unfold slotRuns
  cases hr : (L.checked x r)[i.val]? with
  | none => rfl
  | some u =>
    rw [Option.map_some, Option.getD_some]
    by_cases hu : u < L.nPU x ∧ (L.sessSlots x (L.checked x r) i).head? = some i.val
    · have hs := selSlot_eq_some.2 (show L.firstSel x i (L.checked x r) = some i ∧
        (L.checked x r)[i.val]? = some (⟨u, hu.1⟩ : Fin (L.nPU x)).val from ⟨ite_eq_left hu.2, hr⟩)
      rw [h] at hs
      exact (Option.some_ne_none _ hs.symm).elim
    · exact dite_eq_right hu

theorem slotRuns_some (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) {i u : Fin (L.nPU x)} {r : Fin (L.V.k x) → Coin}
    (h : (L.checked x r)[i.val]? = some u.val) (hf : (L.sessSlots x (L.checked x r) i).head? = some i.val)
    (ρ : L.BlockEC x) :
    L.slotRuns H x s run' i r ρ =
      (((L.sessExt x s (L.sessBlocks x (L.checked x r) i) u r (run' r).seen (ρ u)).eval H run').2.2 : ℝ) := by
  unfold slotRuns
  rw [h, Option.map_some, Option.getD_some]
  exact dite_eq_left ⟨u.isLt, hf⟩

/-- **Slot `i`'s expected reruns**: over the second stage `d` at `r`, slot `i`'s expected reruns is at most `e_s` where
slot `i` is its session's first, and `0` elsewhere. Resampling its block's session's rounds keeps the sample (`fixed`),
so slot `i` stays first, and on them the session's extraction is `A`'s extractor against one prover `P'` at uniform
coins, which reruns at most `1 + m′·k` of the block's shape in expectation (`Arg.rerunsExp_le`). -/
theorem slot_exp_le_un (hS : L.CostContract H) {x : Msg} (s : AuditStatement)
    {Ω : Type} (P : Prover Ω) (ω : Ω) (i : Fin (L.nPU x)) (r : Fin (L.V.k x) → Coin) :
    Ex (fun d : Fin (L.V.k x) → Coin =>
        Ex fun ρ : L.BlockEC x => L.slotRuns H x s (run H P L.V x ω) i (forkOn (L.stage2 x) r d) ρ) ≤
      Ex fun d : Fin (L.V.k x) → Coin =>
        if (L.sessSlots x (L.checked x (forkOn (L.stage2 x) r d)) i).head? = some i.val then L.es x else 0 := by
  have he0 : 0 ≤ L.es x := maxBelow_nonneg _ _
  have hres := Ex_resample (fun d => L.selRounds x (L.firstSel x i) (forkOn (L.stage2 x) r d))
    (selRounds_resample hS.toRunsFixed (L.firstSel x i) r) fun d =>
      Ex fun ρ : L.BlockEC x => L.slotRuns H x s (run H P L.V x ω) i (forkOn (L.stage2 x) r d) ρ
  beta_reduce at hres
  rw [← hres]
  refine Ex_mono fun d => ?_
  cases h : L.selSlot x (L.firstSel x i) (forkOn (L.stage2 x) r d) with
  | none =>
    rw [selRounds_none h]
    simp only [forkOn_empty, slotRuns_none H L x s (run H P L.V x ω) i h, Ex_const]
    split_ifs
    · exact he0
    · exact le_rfl
  | some iu =>
    obtain ⟨i', u⟩ := iu
    obtain ⟨hsel, hm⟩ := selSlot_eq_some.1 h
    have hfi : (L.sessSlots x (L.checked x (forkOn (L.stage2 x) r d)) i).head? = some i.val ∧ i' = i := by
      unfold firstSel at hsel
      by_cases hf : (L.sessSlots x (L.checked x (forkOn (L.stage2 x) r d)) i).head? = some i.val
      · rw [ite_eq_left hf] at hsel
        exact ⟨hf, (Option.some.inj hsel).symm⟩
      · rw [ite_eq_right hf] at hsel
        cases hsel
    rw [hfi.2] at h hm
    rw [ite_eq_left hfi.1]
    have hmem : u.val ∈ L.checked x (forkOn (L.stage2 x) r d) := List.mem_of_getElem? hm
    rw [selRounds_some h]
    simp only [forkOn_sub (hS.fixed x _ u hmem).1]
    obtain ⟨xS, P', hshape, -, -, hcard, -, hview⟩ := hS.runs x Ω P ω (forkOn (L.stage2 x) r d) u hmem
    have hag : ∀ σ : Fin (L.V.k x) → Coin, ∀ t ∉ L.rounds x u,
        forkOn (L.rounds x u) (forkOn (L.stage2 x) r d) σ t = forkOn (L.stage2 x) r d t :=
      fun σ t ht => by simp [forkOn, ht]
    have hunit : ∀ σ (ρ : L.BlockEC x),
        L.slotRuns H x s (run H P L.V x ω) i (forkOn (L.rounds x u) (forkOn (L.stage2 x) r d) σ) ρ =
          ((A.E.eval H P' xS () (fun t => σ (L.emb x u xS hcard t)) (cast (congrArg A.EC hshape.symm) (ρ u))).2.2 :
            ℝ) := by
      intro σ ρ
      have hcσ := (hS.fixed x _ u hmem).2 σ
      have hmσ : (L.checked x (forkOn (L.rounds x u) (forkOn (L.stage2 x) r d) σ))[i.val]? = some u.val := by
        rw [hcσ]
        exact hm
      have hfσ : (L.sessSlots x (L.checked x (forkOn (L.rounds x u) (forkOn (L.stage2 x) r d) σ)) i).head? =
          some i.val := by
        rw [hcσ]
        exact hfi.1
      rw [slotRuns_some H L x s _ hmσ hfσ ρ, L.sessExt_runs_eq H P ω P' (forkOn (L.stage2 x) r d) hshape hcard
        (fun r' h => (hview r' h).1) (hag σ) s _ (ρ u), L.coinsOf_forkOn hcard]
    calc Ex (fun σ : Fin (L.V.k x) → Coin => Ex fun ρ : L.BlockEC x =>
            L.slotRuns H x s (run H P L.V x ω) i (forkOn (L.rounds x u) (forkOn (L.stage2 x) r d) σ) ρ)
        = Ex (fun σ : Fin (L.V.k x) → Coin => Ex fun c : A.EC (L.sh x u) =>
            ((A.E.eval H P' xS () (fun t => σ (L.emb x u xS hcard t)) (cast (congrArg A.EC hshape.symm) c)).2.2 :
              ℝ)) := by
          refine congrArg Ex (funext fun σ => ?_)
          simp only [hunit σ]
          exact Ex_pi_eval_dep (E := fun v : Fin (L.nPU x) => A.EC (L.sh x v)) u fun c : A.EC (L.sh x u) =>
            ((A.E.eval H P' xS () (fun t => σ (L.emb x u xS hcard t)) (cast (congrArg A.EC hshape.symm) c)).2.2 : ℝ)
      _ = Ex (fun p : (Fin (L.V.k x) → Coin) × A.EC (L.sh x u) => ((A.E.eval H P' xS ()
            (fun t => p.1 (L.emb x u xS hcard t)) (cast (congrArg A.EC hshape.symm) p.2)).2.2 : ℝ)) :=
          (Ex_prod fun p : (Fin (L.V.k x) → Coin) × A.EC (L.sh x u) => ((A.E.eval H P' xS ()
            (fun t => p.1 (L.emb x u xS hcard t)) (cast (congrArg A.EC hshape.symm) p.2)).2.2 : ℝ)).symm
      _ = Ex (fun p : (Fin (A.V.k xS) → Coin) × A.EC (A.shape xS) => ((A.E.eval H P' xS () p.1 p.2).2.2 : ℝ)) :=
          Ex_restrict_shape (L.emb x u xS hcard) (L.emb x u xS hcard).injective hshape
            fun ρ' c' => ((A.E.eval H P' xS () ρ' c').2.2 : ℝ)
      _ ≤ 1 + A.waits (A.shape xS) * A.k (A.shape xS) := A.rerunsExp_le H xS Unit P' ()
      _ = 1 + A.waits (L.sh x u) * A.k (L.sh x u) := by rw [hshape]
      _ ≤ L.es x := le_maxBelow (fun v => 1 + (A.waits (L.sh x v) : ℝ) * A.k (L.sh x v)) u.isLt

/-- **Slot `i`'s expected reruns given acceptance**: acceptance times slot `i`'s expected reruns is at most slot `i`'s
expected reruns (`slot_exp_le_un`). -/
theorem slot_exp_le (hS : L.CostContract H) {x : Msg} (s : AuditStatement)
    {Ω : Type} (P : Prover Ω) (ω : Ω) (i : Fin (L.nPU x)) (r : Fin (L.V.k x) → Coin) :
    Ex (fun d : Fin (L.V.k x) → Coin => (if (run H P L.V x ω (forkOn (L.stage2 x) r d)).out = .acc then (1 : ℝ) else 0) *
        Ex fun ρ : L.BlockEC x => L.slotRuns H x s (run H P L.V x ω) i (forkOn (L.stage2 x) r d) ρ) ≤
      Ex fun d : Fin (L.V.k x) → Coin =>
        if (L.sessSlots x (L.checked x (forkOn (L.stage2 x) r d)) i).head? = some i.val then L.es x else 0 :=
  calc Ex (fun d : Fin (L.V.k x) → Coin =>
          (if (run H P L.V x ω (forkOn (L.stage2 x) r d)).out = .acc then (1 : ℝ) else 0) *
            Ex fun ρ : L.BlockEC x => L.slotRuns H x s (run H P L.V x ω) i (forkOn (L.stage2 x) r d) ρ)
      ≤ Ex (fun d : Fin (L.V.k x) → Coin =>
          Ex fun ρ : L.BlockEC x => L.slotRuns H x s (run H P L.V x ω) i (forkOn (L.stage2 x) r d) ρ) :=
        Ex_mono fun d => by
          have hnn : 0 ≤ Ex fun ρ : L.BlockEC x =>
              L.slotRuns H x s (run H P L.V x ω) i (forkOn (L.stage2 x) r d) ρ :=
            Ex_nonneg fun ρ => slotRuns_nonneg H L x s _ i _ ρ
          split_ifs
          · exact (one_mul _).le
          · exact (zero_mul _).le.trans hnn
    _ ≤ _ := slot_exp_le_un hS s P ω i r

/-- **Expected reruns of a run's extractions**: over the run's coins and the extractor's, at most `n_sess·e_s`. Each
slot weighs at most `e_s` where it is its session's first and `0` elsewhere (`slot_exp_le_un`, the second stage's coins
fresh at uniform coins, `Ex_forkOn`), and a sample has at most `n_sess` first slots (`card_first_le`). -/
theorem runExt_runs_exp (hS : L.CostContract H) {x : Msg} {Ω : Type} (P : Prover Ω) (ω : Ω) (s : AuditStatement) :
    Ex (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (((L.runExt x s p.1 (run H P L.V x ω p.1).seen p.2).eval H (run H P L.V x ω)).2.2 : ℝ)) ≤
      L.nSess x * L.es x := by
  have he0 : 0 ≤ L.es x := maxBelow_nonneg _ _
  have hslot : ∀ i : Fin (L.nPU x), Ex (fun r : Fin (L.V.k x) → Coin =>
      Ex fun ρ : L.BlockEC x => L.slotRuns H x s (run H P L.V x ω) i r ρ) ≤
        Ex fun r : Fin (L.V.k x) → Coin =>
          if (L.sessSlots x (L.checked x r) i).head? = some i.val then L.es x else 0 := by
    intro i
    rw [← Ex_forkOn (L.stage2 x) (fun r => Ex fun ρ : L.BlockEC x => L.slotRuns H x s (run H P L.V x ω) i r ρ),
      ← Ex_forkOn (L.stage2 x) (fun r =>
        if (L.sessSlots x (L.checked x r) i).head? = some i.val then L.es x else 0), Ex_prod, Ex_prod]
    exact Ex_mono fun r => slot_exp_le_un hS s P ω i r
  calc Ex (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
          (((L.runExt x s p.1 (run H P L.V x ω p.1).seen p.2).eval H (run H P L.V x ω)).2.2 : ℝ))
      = Ex (fun r : Fin (L.V.k x) → Coin =>
          ∑ i, Ex fun ρ : L.BlockEC x => L.slotRuns H x s (run H P L.V x ω) i r ρ) := by
        rw [Ex_prod]
        refine congrArg Ex (funext fun r => ?_)
        simp only [runExt_runs_eq]
        rw [Ex_finsum]
        exact Finset.sum_congr rfl fun i _ => Ex_pi_eval i fun ρ => L.slotRuns H x s (run H P L.V x ω) i r ρ
    _ = ∑ i, Ex (fun r : Fin (L.V.k x) → Coin =>
          Ex fun ρ : L.BlockEC x => L.slotRuns H x s (run H P L.V x ω) i r ρ) := by
        rw [Ex_finsum]
    _ ≤ ∑ i : Fin (L.nPU x), Ex (fun r : Fin (L.V.k x) → Coin =>
          if (L.sessSlots x (L.checked x r) i).head? = some i.val then L.es x else 0) :=
        Finset.sum_le_sum fun i _ => hslot i
    _ = Ex (fun r : Fin (L.V.k x) → Coin => ∑ i : Fin (L.nPU x),
          if (L.sessSlots x (L.checked x r) i).head? = some i.val then L.es x else 0) := by
        rw [Ex_finsum]
    _ ≤ Ex (fun _ : Fin (L.V.k x) → Coin => (L.nSess x : ℝ) * L.es x) := Ex_mono fun r => by
        rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
        exact mul_le_mul_of_nonneg_right (Nat.cast_le.2 (card_first_le hS.keys x _)) he0
    _ = L.nSess x * L.es x := Ex_const _

theorem Ex_slotStep (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (j : Fin (L.kOut x)) (l : Fin (L.kIn x)) (r : Fin (L.V.k x) → Coin) :
    Ex (fun c : L.SlotY x => (((L.slotStep x s c j l r (run' r).seen).eval H run').2.2 : ℝ)) =
      ∑ i, Ex fun ρ : L.BlockEC x => L.slotRuns H x s run' i r ρ := by
  simp only [slotStep, runExt_runs_eq]
  rw [Ex_finsum]
  exact Finset.sum_congr rfl fun i _ => Ex_pi_eval (j, l, i) fun ρ => L.slotRuns H x s run' i r ρ

/-! ## `Efficient` at the slot-by-slot extractor -/

/-- **Strict reruns**: at most `1 + K_out + k_out·(K_in + k_in·n_sess·ρ_s)`. -/
theorem slots_runs_le (hS : L.CostContract H) :
    ExtractRunsLe H (L.Esamp L.SlotY L.slotStep)
      (fun x => 1 + L.capOut x + L.kOut x * (L.capIn x + L.kIn x * (L.nSess x * L.ρs x))) :=
  Esamp_runs_le H L L.SlotY L.slotStep (fun x => L.nSess x * L.ρs x) fun _ s _ P ω _ _ _ _ r =>
    runExt_runs_le hS P ω s r _

/-- **Expected reruns**: at most `1 + k_out + k_out·k_in·(1 + n_sess·e_s)`: a sample has at most `n_sess` first slots
(`card_first_le`), each weighing at most `e_s` given acceptance (`slot_exp_le`). -/
theorem slots_runs_exp (hS : L.CostContract H) :
    ExtractRunsExpLe H (L.Esamp L.SlotY L.slotStep)
      (fun x => 1 + L.kOut x + (L.kOut x : ℝ) * L.kIn x * (1 + L.nSess x * L.es x)) := by
  intro x Ω P ω
  have he0 : 0 ≤ L.es x := maxBelow_nonneg _ _
  refine Esamp_runs_exp H L L.SlotY L.slotStep x P ω
    (fun s r => ∑ i, Ex fun ρ : L.BlockEC x => L.slotRuns H x s (run H P L.V x ω) i r ρ) (L.nSess x * L.es x)
    (mul_nonneg (Nat.cast_nonneg _) he0)
    (fun s r => Finset.sum_nonneg fun i _ => Ex_nonneg fun ρ => slotRuns_nonneg H L x s _ i r ρ)
    (fun s _ j l r => (Ex_slotStep H L x s _ j l r).le) fun s _ r => ?_
  simp only [Finset.mul_sum]
  rw [Ex_finsum]
  calc ∑ i, Ex (fun d : Fin (L.V.k x) → Coin =>
        (if (run H P L.V x ω (forkOn (L.stage2 x) r d)).out = .acc then (1 : ℝ) else 0) *
          Ex fun ρ : L.BlockEC x => L.slotRuns H x s (run H P L.V x ω) i (forkOn (L.stage2 x) r d) ρ)
      ≤ ∑ i : Fin (L.nPU x), Ex (fun d : Fin (L.V.k x) → Coin =>
          if (L.sessSlots x (L.checked x (forkOn (L.stage2 x) r d)) i).head? = some i.val then L.es x else 0) :=
        Finset.sum_le_sum fun i _ => slot_exp_le hS s P ω i r
    _ = Ex (fun d : Fin (L.V.k x) → Coin => ∑ i : Fin (L.nPU x),
          if (L.sessSlots x (L.checked x (forkOn (L.stage2 x) r d)) i).head? = some i.val then L.es x else 0) := by
        rw [Ex_finsum]
    _ ≤ Ex (fun _ : Fin (L.V.k x) → Coin => (L.nSess x : ℝ) * L.es x) := Ex_mono fun d => by
        rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
        exact mul_le_mul_of_nonneg_right (Nat.cast_le.2 (card_first_le hS.keys x _)) he0
    _ = L.nSess x * L.es x := Ex_const _

/-- **Own SHA-512 calls**: at most `k_out·k_in·n_PU·openCalls`. -/
theorem slots_own_le (hS : L.CostContract H) (hopen : L.OpenCallsLe H) :
    ExtractOwnLe H (L.Esamp L.SlotY L.slotStep) (fun x => L.kOut x * L.kIn x * L.nPU x * L.openCalls x) := by
  intro x Ω P ω r₀ c
  exact (Esamp_own_le H L L.SlotY L.slotStep (fun x => L.nPU x * L.openCalls x)
    (fun _ _ _ P ω hs c j l r => runExt_own_le hS hopen hs P ω r _) x Ω P ω r₀ c).trans_eq (by ring)

/-- **`Efficient` at the contract's cost facts**: the compiled extractor's three bounds (`E_eq_Esamp`, then
`slots_runs_le`, `slots_runs_exp`, `slots_own_le`). -/
theorem efficient_of (hS : L.CostContract H) (hopen : L.OpenCallsLe H) : L.Efficient H := by
  unfold Compiled.Efficient
  rw [L.E_eq_Esamp]
  exact ⟨slots_runs_le hS, slots_runs_exp hS, slots_own_le hS hopen⟩

end Compiled

/-! ## The compiler's efficiency -/

/-- **`CompilerEfficient`**: the compiled extractor's three bounds at the contract's cost facts
(`Compiled.efficient_of`, `Compiled.Sessions.toCostContract`). -/
theorem compilerEfficient : CompilerEfficient := by
  intro Coin _ _ _ A L H hS hopen
  exact Compiled.efficient_of hS.toCostContract hopen

end Claims.SamplingGeneric
