module

public import Proofs.Flock.SamplingGeneric.PhasesOnce
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Bext
import Proofs.Flock.Claims.Sampling.Extract
import Proofs.Flock.Claims.Sampling.Wait

/-!
# The first-hit bound over a monitoring period, at the period's averages

`PhasesOnce.lean` charges a period's calls once where the sample's three counts hold at every sample of every phase.
A verifier whose refused selection samples every proof block has no such count below the whole partition's. Here
the counts enter as `Compiled.CallsAvg` reads them, as averages of a run's calls times its sample's counts, and the
period pays those averages' totals once:

`Σ_i (3·κ_s + 2·√(k·γ_s·Pr[B_i]))`
`+ 2·(2·(T + e·A_s + n·c) + 4·(T + e·A_s + A_k + e·A_m + (1 + K_s)·n·c))/2^256`

with `T` at least the period's expected calls in total, and `A_s`, `A_k` and `A_m` at least the period's expected
totals of a phase's calls times the number of sessions in its sample, of classes of the gates its sample needs, and
of the classes' representatives its sessions need. `K_s` bounds each phase's expected number of classes, and `e` and
`c` each phase's expected reruns a session and its opening checks' calls. A refused run then costs the whole
partition's counts only on the calls spent in refused runs.

* `sum_lin4`: a sum of four scaled terms and a constant.
* `sum_hash_avg_le`: the phases' parts that count calls, summed, at the averages' totals.
* `pr_phases_avg_le`: **the period's step at averages**, for phases whose bounds have that shape.
* `CompiledTwo.phases_oneWait_le_avg_of_phase`: the period's step from any bound of a phase at its averages.
* `CompiledTwo.phases_oneWait_le_avg`: the period's bound at two stages with one wait, each phase under
  `eventTwoX_oneWait_le_cls_avg`'s premises at its own averages.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeTwo)

/-- A sum over the phases of four scaled terms and a constant. -/
theorem sum_lin4 {n : ℕ} (x₁ x₂ x₃ x₄ : Fin n → ℝ) (c₁ c₂ c₃ c₄ κ : ℝ) :
    ∑ i, (c₁ * x₁ i + c₂ * x₂ i + c₃ * x₃ i + c₄ * x₄ i + κ) =
      c₁ * ∑ i, x₁ i + c₂ * ∑ i, x₂ i + c₃ * ∑ i, x₃ i + c₄ * ∑ i, x₄ i + n * κ := by
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_add_distrib,
    ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]

/-- **The phases' parts that count calls, summed, at averages.** Phase `i` pays `a` times
`T_i + e_i·A_s,i + c_i` and `f` times `T_i + e_i·A_s,i + A_k,i + e_i·A_m,i + (1 + K_i)·c_i`. With `e_i ≤ e`,
`K_i ≤ K`, `c_i ≤ c` and the phases' `T_i`, `A_s,i`, `A_k,i` and `A_m,i` summing to at most `T`, `A_s`, `A_k` and
`A_m`, the phases together pay at most the same expression at those totals, at `e`, at `K` and at `n·c`. -/
theorem sum_hash_avg_le {n : ℕ} (Tq As Ak Am es Ks c : Fin n → ℝ) {a f T As' Ak' Am' e K cc : ℝ} (ha : 0 ≤ a)
    (hf : 0 ≤ f) (hAs0 : ∀ i, 0 ≤ As i) (hAm0 : ∀ i, 0 ≤ Am i) (hc0 : ∀ i, 0 ≤ c i) (he0 : 0 ≤ e) (hK0 : 0 ≤ K)
    (he : ∀ i, es i ≤ e) (hKs : ∀ i, Ks i ≤ K) (hc : ∀ i, c i ≤ cc) (hT : ∑ i, Tq i ≤ T) (hAs : ∑ i, As i ≤ As')
    (hAk : ∑ i, Ak i ≤ Ak') (hAm : ∑ i, Am i ≤ Am') :
    ∑ i, (a * (Tq i + es i * As i + c i) +
        f * (Tq i + es i * As i + Ak i + es i * Am i + (1 + Ks i) * c i)) ≤
      a * (T + e * As' + n * cc) + f * (T + e * As' + Ak' + e * Am' + (1 + K) * (n * cc)) := by
  have hterm : ∀ i, a * (Tq i + es i * As i + c i) +
        f * (Tq i + es i * As i + Ak i + es i * Am i + (1 + Ks i) * c i) ≤
      (a + f) * Tq i + (a + f) * e * As i + f * Ak i + f * e * Am i + (a * cc + f * ((1 + K) * cc)) := fun i => by
    have h1 : es i * As i ≤ e * As i := mul_le_mul_of_nonneg_right (he i) (hAs0 i)
    have h2 : es i * Am i ≤ e * Am i := mul_le_mul_of_nonneg_right (he i) (hAm0 i)
    have h3 : (1 + Ks i) * c i ≤ (1 + K) * cc :=
      mul_le_mul (by linarith [hKs i]) (hc i) (hc0 i) (by linarith)
    have h4 : Tq i + es i * As i + c i ≤ Tq i + e * As i + cc := by linarith [hc i]
    have h5 : Tq i + es i * As i + Ak i + es i * Am i + (1 + Ks i) * c i ≤
        Tq i + e * As i + Ak i + e * Am i + (1 + K) * cc := by linarith
    exact (add_le_add (mul_le_mul_of_nonneg_left h4 ha) (mul_le_mul_of_nonneg_left h5 hf)).trans
      (le_of_eq (by ring))
  have haf : 0 ≤ a + f := add_nonneg ha hf
  have m1 := mul_le_mul_of_nonneg_left hT haf
  have m2 := mul_le_mul_of_nonneg_left hAs (mul_nonneg haf he0)
  have m3 := mul_le_mul_of_nonneg_left hAk hf
  have m4 := mul_le_mul_of_nonneg_left hAm (mul_nonneg hf he0)
  have hring : (a + f) * T + (a + f) * e * As' + f * Ak' + f * e * Am' + n * (a * cc + f * ((1 + K) * cc)) =
      a * (T + e * As' + n * cc) + f * (T + e * As' + Ak' + e * Am' + (1 + K) * (n * cc)) := by
    ring
  exact ((Finset.sum_le_sum fun i _ => hterm i).trans
    ((sum_lin4 Tq As Ak Am (a + f) ((a + f) * e) f (f * e) (a * cc + f * ((1 + K) * cc))).le.trans
      (add_le_add (add_le_add (add_le_add (add_le_add m1 m2) m3) m4) le_rfl))).trans hring.le

/-- **The period's step at averages.** A period's sample space split at each of its `n` phases (`φ`), phase `i` with
an event `E i`, a count of calls and three nonnegative weights over its prover's tape and its own coins. Where
phase `i`'s event has probability at most `b_i` plus, over `2^256`, `a` times `T_i + e_i·A_s,i + c_i` and `f` times
`T_i + e_i·A_s,i + A_k,i + e_i·A_m,i + (1 + K_i)·c_i`, with `T_i`, `A_s,i`, `A_k,i` and `A_m,i` the phase's averages
of its calls and its weights, and the period's calls and weights average at most `T`, `A_s`, `A_k` and `A_m` in
total, some phase has its event with probability at most `Σ_i b_i` plus the same expression at those totals, at
`e ≥ e_i`, at `K ≥ K_i` and at `n·c`, `c ≥ c_i`. -/
theorem pr_phases_avg_le {S : Type} [Fintype S] {n : ℕ} {Ω C : Fin n → Type} [∀ i, Fintype (Ω i)]
    [∀ i, Fintype (C i)] (φ : ∀ i, S ≃ Ω i × C i) (E : ∀ i, Ω i × C i → Prop)
    (calls ws wk wm : ∀ i, Ω i × C i → ℝ) (b es Ks c : Fin n → ℝ) {a f T As Ak Am e K cc : ℝ} (ha : 0 ≤ a)
    (hf : 0 ≤ f) (hws : ∀ i q, 0 ≤ ws i q) (hwm : ∀ i q, 0 ≤ wm i q) (hc0 : ∀ i, 0 ≤ c i) (he0 : 0 ≤ e)
    (hK0 : 0 ≤ K) (he : ∀ i, es i ≤ e) (hKs : ∀ i, Ks i ≤ K) (hc : ∀ i, c i ≤ cc)
    (hT : Ex (fun σ : S => ∑ i, calls i (φ i σ)) ≤ T) (hAs : Ex (fun σ : S => ∑ i, ws i (φ i σ)) ≤ As)
    (hAk : Ex (fun σ : S => ∑ i, wk i (φ i σ)) ≤ Ak) (hAm : Ex (fun σ : S => ∑ i, wm i (φ i σ)) ≤ Am)
    (hph : ∀ i, Pr (E i) ≤ b i + (a * (Ex (calls i) + es i * Ex (ws i) + c i) +
      f * (Ex (calls i) + es i * Ex (ws i) + Ex (wk i) + es i * Ex (wm i) + (1 + Ks i) * c i)) /
        2 ^ (256 : ℝ)) :
    Pr (fun σ : S => ∃ i, E i (φ i σ)) ≤
      ∑ i, b i + (a * (T + e * As + n * cc) + f * (T + e * As + Ak + e * Am + (1 + K) * (n * cc))) /
        2 ^ (256 : ℝ) := by
  have hh := sum_hash_avg_le (fun i => Ex (calls i)) (fun i => Ex (ws i)) (fun i => Ex (wk i))
    (fun i => Ex (wm i)) es Ks c ha hf (fun i => Ex_nonneg (hws i)) (fun i => Ex_nonneg (hwm i)) hc0 he0 hK0 he
    hKs hc ((Ex_phases_sum φ calls).symm.le.trans hT) ((Ex_phases_sum φ ws).symm.le.trans hAs)
    ((Ex_phases_sum φ wk).symm.le.trans hAk) ((Ex_phases_sum φ wm).symm.le.trans hAm)
  have hp : (0 : ℝ) < 2 ^ (256 : ℝ) := by positivity
  have hsplit : ∑ i, (b i + (a * (Ex (calls i) + es i * Ex (ws i) + c i) +
        f * (Ex (calls i) + es i * Ex (ws i) + Ex (wk i) + es i * Ex (wm i) + (1 + Ks i) * c i)) /
          2 ^ (256 : ℝ)) =
      ∑ i, b i + (∑ i, (a * (Ex (calls i) + es i * Ex (ws i) + c i) +
        f * (Ex (calls i) + es i * Ex (ws i) + Ex (wk i) + es i * Ex (wm i) + (1 + Ks i) * c i))) /
          2 ^ (256 : ℝ) := by
    rw [Finset.sum_add_distrib, Finset.sum_div]
  exact (Pr_phases_le φ E).trans (((Finset.sum_le_sum fun i _ => hph i).trans hsplit.le).trans
    (add_le_add le_rfl (div_le_div_of_nonneg_right hh hp.le)))

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

namespace CompiledTwo

/-- **The period's step for the one-wait bound at averages, from a phase's bound.** A period of `n` phases, phase
`i` at its statement `x i` and its prover `P i`, whose tape and the phase's coins split the period's sample space
(`φ`). Where each phase's event has probability at most `εclsOneWaitAvg` at any averages its calls and its sample's
counts are within (`Compiled.CallsAvg`), and the period's totals are at most `T`, `A_s`, `A_k` and `A_m`, with each
phase's expected classes at most `K_s`, its expected reruns a session at most `e` and its opening checks' calls at
most `c`, some phase has its event with probability at most

`Σ_i (3·κ_s + 2·√(k·γ_s·Pr[B_i]))`
`+ 2·(2·(T + e·A_s + n·c) + 4·(T + e·A_s + A_k + e·A_m + (1 + K_s)·n·c))/2^256`. -/
theorem phases_oneWait_le_avg_of_phase {H : Msg → Msg} {L : CompiledTwo Coin A} {n : ℕ} {x : Fin n → Msg}
    {S : Type} [Fintype S] {Ω : Fin n → Type} [∀ i, Fintype (Ω i)] [∀ i, Nonempty (Ω i)]
    (φ : ∀ i, S ≃ Ω i × (Fin (L.V.k (x i)) → Coin)) (P : ∀ i, Prover (Ω i)) (needs : Fin n → ℕ → Finset ℕ)
    (cls : ∀ i, Fin (L.nG (x i)) → Fin (L.nG (x i)))
    (hphase : ∀ i (T As Ak Am Ks : ℝ), L.toCompiled.CallsAvg H (P i) (x i) (needs i) (cls i) T As Ak Am Ks →
      Pr (fun p : Ω i × (Fin (L.V.k (x i)) → Coin) => (run H (P i) L.V (x i) p.1 p.2).out = .acc ∧
          ∃ u ∈ L.checked (x i) p.2, u ∈ L.oneWait.FregTwoX H (P i) (x i) p.1 p.2) ≤
        L.εclsOneWaitAvg H (x i) (P i) T As Ak Am Ks)
    {T As Ak Am Ks e c : ℝ} (he0 : 0 ≤ e) (hKs0 : 0 ≤ Ks)
    (hT : Ex (fun σ : S => ∑ i, (L.toCompiled.partiesCalls H (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)) ≤ T)
    (hAs : Ex (fun σ : S => ∑ i, ((L.toCompiled.firstSlots (x i) (L.checked (x i) (φ i σ).2)).card : ℝ) *
      (L.toCompiled.partiesCalls H (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)) ≤ As)
    (hAk : Ex (fun σ : S => ∑ i,
      (((L.toCompiled.sampleNeeds (x i) (needs i) (L.checked (x i) (φ i σ).2)).image (cls i)).card : ℝ) *
        (L.toCompiled.partiesCalls H (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)) ≤ Ak)
    (hAm : Ex (fun σ : S => ∑ i,
      (L.toCompiled.sessCountIn (x i) (needs i)
        (fun l => (L.toCompiled.sampleNeeds (x i) (needs i) l).image (cls i)) (L.checked (x i) (φ i σ).2) : ℝ) *
          (L.toCompiled.partiesCalls H (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)) ≤ Am)
    (hKs : ∀ i, Ex (fun r : Fin (L.V.k (x i)) → Coin =>
      (((L.toCompiled.sampleNeeds (x i) (needs i) (L.checked (x i) r)).image (cls i)).card : ℝ)) ≤ Ks)
    (he : ∀ i, L.oneWait.toCompiled.es (x i) ≤ e)
    (hc : ∀ i, (L.oneWait.toCompiled.nPU (x i) : ℝ) * L.oneWait.toCompiled.openCalls (x i) ≤ c) :
    Pr (fun σ : S => ∃ i, (run H (P i) L.V (x i) (φ i σ).1 (φ i σ).2).out = .acc ∧
        ∃ u ∈ L.checked (x i) (φ i σ).2, u ∈ L.oneWait.FregTwoX H (P i) (x i) (φ i σ).1 (φ i σ).2) ≤
      ∑ i, (3 * L.κs (x i) + 2 * Real.sqrt (L.V.k (x i) * L.γs (x i) * PrB H L.V (P i) (x i))) +
        2 * (2 * (T + e * As + n * c) + 4 * (T + e * As + Ak + e * Am + (1 + Ks) * (n * c))) /
          2 ^ (256 : ℝ) := by
  have key : ∀ b u v p : ℝ, b + 2 * (2 * u + 4 * v) / p = b + (4 * u + 8 * v) / p := by
    intro b u v p
    ring
  have hph : ∀ i, Pr (fun p : Ω i × (Fin (L.V.k (x i)) → Coin) => (run H (P i) L.V (x i) p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked (x i) p.2, u ∈ L.oneWait.FregTwoX H (P i) (x i) p.1 p.2) ≤
      3 * L.κs (x i) + 2 * Real.sqrt (L.V.k (x i) * L.γs (x i) * PrB H L.V (P i) (x i)) +
        (4 * (Ex (fun q : Ω i × (Fin (L.V.k (x i)) → Coin) =>
              (L.toCompiled.partiesCalls H (P i) (x i) q.1 q.2 : ℝ)) +
            L.oneWait.toCompiled.es (x i) * Ex (fun q : Ω i × (Fin (L.V.k (x i)) → Coin) =>
              ((L.toCompiled.firstSlots (x i) (L.checked (x i) q.2)).card : ℝ) *
                (L.toCompiled.partiesCalls H (P i) (x i) q.1 q.2 : ℝ)) +
            (L.oneWait.toCompiled.nPU (x i) : ℝ) * L.oneWait.toCompiled.openCalls (x i)) +
          8 * (Ex (fun q : Ω i × (Fin (L.V.k (x i)) → Coin) =>
              (L.toCompiled.partiesCalls H (P i) (x i) q.1 q.2 : ℝ)) +
            L.oneWait.toCompiled.es (x i) * Ex (fun q : Ω i × (Fin (L.V.k (x i)) → Coin) =>
              ((L.toCompiled.firstSlots (x i) (L.checked (x i) q.2)).card : ℝ) *
                (L.toCompiled.partiesCalls H (P i) (x i) q.1 q.2 : ℝ)) +
            Ex (fun q : Ω i × (Fin (L.V.k (x i)) → Coin) =>
              (((L.toCompiled.sampleNeeds (x i) (needs i) (L.checked (x i) q.2)).image (cls i)).card : ℝ) *
                (L.toCompiled.partiesCalls H (P i) (x i) q.1 q.2 : ℝ)) +
            L.oneWait.toCompiled.es (x i) * Ex (fun q : Ω i × (Fin (L.V.k (x i)) → Coin) =>
              (L.toCompiled.sessCountIn (x i) (needs i)
                (fun l => (L.toCompiled.sampleNeeds (x i) (needs i) l).image (cls i)) (L.checked (x i) q.2) : ℝ) *
                  (L.toCompiled.partiesCalls H (P i) (x i) q.1 q.2 : ℝ)) +
            (1 + Ex (fun r : Fin (L.V.k (x i)) → Coin =>
              (((L.toCompiled.sampleNeeds (x i) (needs i) (L.checked (x i) r)).image (cls i)).card : ℝ))) *
              ((L.oneWait.toCompiled.nPU (x i) : ℝ) * L.oneWait.toCompiled.openCalls (x i)))) /
          2 ^ (256 : ℝ) := by
    intro i
    have hC : L.toCompiled.CallsAvg H (P i) (x i) (needs i) (cls i)
        (Ex (fun q : Ω i × (Fin (L.V.k (x i)) → Coin) => (L.toCompiled.partiesCalls H (P i) (x i) q.1 q.2 : ℝ)))
        (Ex (fun q : Ω i × (Fin (L.V.k (x i)) → Coin) =>
          ((L.toCompiled.firstSlots (x i) (L.checked (x i) q.2)).card : ℝ) *
            (L.toCompiled.partiesCalls H (P i) (x i) q.1 q.2 : ℝ)))
        (Ex (fun q : Ω i × (Fin (L.V.k (x i)) → Coin) =>
          (((L.toCompiled.sampleNeeds (x i) (needs i) (L.checked (x i) q.2)).image (cls i)).card : ℝ) *
            (L.toCompiled.partiesCalls H (P i) (x i) q.1 q.2 : ℝ)))
        (Ex (fun q : Ω i × (Fin (L.V.k (x i)) → Coin) =>
          (L.toCompiled.sessCountIn (x i) (needs i)
            (fun l => (L.toCompiled.sampleNeeds (x i) (needs i) l).image (cls i)) (L.checked (x i) q.2) : ℝ) *
              (L.toCompiled.partiesCalls H (P i) (x i) q.1 q.2 : ℝ)))
        (Ex (fun r : Fin (L.V.k (x i)) → Coin =>
          (((L.toCompiled.sampleNeeds (x i) (needs i) (L.checked (x i) r)).image (cls i)).card : ℝ))) :=
      ⟨le_refl _, le_refl _, le_refl _, le_refl _, le_refl _⟩
    have hi := hphase i _ _ _ _ _ hC
    unfold εclsOneWaitAvg Compiled.runCostAvg Compiled.firstHitCostAvg at hi
    exact hi.trans (le_of_eq (key _ _ _ _))
  have h := pr_phases_avg_le φ
    (fun i p => (run H (P i) L.V (x i) p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked (x i) p.2, u ∈ L.oneWait.FregTwoX H (P i) (x i) p.1 p.2)
    (fun i q => (L.toCompiled.partiesCalls H (P i) (x i) q.1 q.2 : ℝ))
    (fun i q => ((L.toCompiled.firstSlots (x i) (L.checked (x i) q.2)).card : ℝ) *
      (L.toCompiled.partiesCalls H (P i) (x i) q.1 q.2 : ℝ))
    (fun i q => (((L.toCompiled.sampleNeeds (x i) (needs i) (L.checked (x i) q.2)).image (cls i)).card : ℝ) *
      (L.toCompiled.partiesCalls H (P i) (x i) q.1 q.2 : ℝ))
    (fun i q => (L.toCompiled.sessCountIn (x i) (needs i)
      (fun l => (L.toCompiled.sampleNeeds (x i) (needs i) l).image (cls i)) (L.checked (x i) q.2) : ℝ) *
        (L.toCompiled.partiesCalls H (P i) (x i) q.1 q.2 : ℝ))
    (fun i => 3 * L.κs (x i) + 2 * Real.sqrt (L.V.k (x i) * L.γs (x i) * PrB H L.V (P i) (x i)))
    (fun i => L.oneWait.toCompiled.es (x i))
    (fun i => Ex (fun r : Fin (L.V.k (x i)) → Coin =>
      (((L.toCompiled.sampleNeeds (x i) (needs i) (L.checked (x i) r)).image (cls i)).card : ℝ)))
    (fun i => (L.oneWait.toCompiled.nPU (x i) : ℝ) * L.oneWait.toCompiled.openCalls (x i))
    (a := 4) (f := 8) (by norm_num) (by norm_num)
    (fun _ _ => mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))
    (fun _ _ => mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))
    (fun _ => mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)) he0 hKs0 he hKs hc hT hAs hAk hAm hph
  exact h.trans (le_of_eq (key _ _ _ _).symm)


/-- **The event at the one-wait extraction's traces over a monitoring period, at the period's averages.** A period
of `n` phases, phase `i` at its statement `x i` in the two-stage scope and its prover `P i`, whose tape and the
phase's coins split the period's sample space (`φ`). Each phase is under `eventTwoX_oneWait_le_cls_avg`'s premises.
Where the period's phases call `H` at most `T` times on average in total, both parties' calls counted, a phase's
calls times the number of sessions in its sample, of classes of the gates its sample needs, and of the classes'
representatives its sessions need average at most `A_s`, `A_k` and `A_m` in total, each phase's sample needs gates
of at most `K_s` classes on average, the one-wait compiled verifier's expected reruns a session are at most `e` and
the opening checks' calls at most `c` at every phase, some phase's accepted run samples a proof block of the
one-wait compiled verifier's `FregTwoX` with probability at most

`Σ_i (3·κ_s + 2·√(k·γ_s·Pr[B_i]))`
`+ 2·(2·(T + e·A_s + n·c) + 4·(T + e·A_s + A_k + e·A_m + (1 + K_s)·n·c))/2^256`.

No count is held at every sample, and no run of a prover to a budget. -/
theorem phases_oneWait_le_avg {H : Msg → Msg} {L : CompiledTwo Coin A} (hF : A.OneWaitFinds H) (hA : A.KS H)
    (hbind : L.commit.Binds H) (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {n : ℕ}
    {x : Fin n → Msg} {s : Fin n → AuditStatement} {Pt : Fin n → Partition} {d : Fin n → Sampling.Design.Two}
    (hs : ∀ i, L.parse (x i) = some (s i)) (hPt : ∀ i, L.partition (s i) = some (Pt i))
    (hd : ∀ i, L.two (s i) = some (d i)) (hin : ∀ i, InScopeTwo (Pt i) (d i) (s i)) {S : Type} [Fintype S]
    {Ω : Fin n → Type} [∀ i, Fintype (Ω i)] [∀ i, Nonempty (Ω i)]
    (φ : ∀ i, S ≃ Ω i × (Fin (L.V.k (x i)) → Coin)) (P : ∀ i, Prover (Ω i)) (needs : Fin n → ℕ → Finset ℕ)
    (hpos : ∀ i u v w, ∀ p ∈ L.reveal (s i) u v w, p.2.pos ∈ needs i u)
    (cls : ∀ i, Fin (L.nG (x i)) → Fin (L.nG (x i)))
    (hallX : ∀ i (a : Ω i × ((Fin (L.V.k (x i)) → Coin) × L.oneWait.SlotEC (x i))) (g g' : Fin (L.nG (x i))),
      cls i g = cls i g' → (L.oneWait.opensAt H (P i) (x i) a.1 a.2.1 a.2.2 g).isSome →
        (L.oneWait.opensAt H (P i) (x i) a.1 a.2.1 a.2.2 g').isSome)
    (hallI : ∀ i (t : Ω i × (Fin (L.V.k (x i)) → Coin))
      (ec : (Fin (L.V.k (x i)) → Coin) × L.oneWait.SlotEC (x i)) (g g' : Fin (L.nG (x i))),
      cls i g = cls i g' → (L.oneWait.iopTable H (P i) (x i) t.1 (d i).part t.2 ec g).isSome →
        (L.oneWait.iopTable H (P i) (x i) t.1 (d i).part t.2 ec g').isSome)
    {T As Ak Am Ks e c : ℝ} (he0 : 0 ≤ e) (hKs0 : 0 ≤ Ks)
    (hT : Ex (fun σ : S => ∑ i, (L.toCompiled.partiesCalls H (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)) ≤ T)
    (hAs : Ex (fun σ : S => ∑ i, ((L.toCompiled.firstSlots (x i) (L.checked (x i) (φ i σ).2)).card : ℝ) *
      (L.toCompiled.partiesCalls H (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)) ≤ As)
    (hAk : Ex (fun σ : S => ∑ i,
      (((L.toCompiled.sampleNeeds (x i) (needs i) (L.checked (x i) (φ i σ).2)).image (cls i)).card : ℝ) *
        (L.toCompiled.partiesCalls H (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)) ≤ Ak)
    (hAm : Ex (fun σ : S => ∑ i,
      (L.toCompiled.sessCountIn (x i) (needs i)
        (fun l => (L.toCompiled.sampleNeeds (x i) (needs i) l).image (cls i)) (L.checked (x i) (φ i σ).2) : ℝ) *
          (L.toCompiled.partiesCalls H (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)) ≤ Am)
    (hKs : ∀ i, Ex (fun r : Fin (L.V.k (x i)) → Coin =>
      (((L.toCompiled.sampleNeeds (x i) (needs i) (L.checked (x i) r)).image (cls i)).card : ℝ)) ≤ Ks)
    (he : ∀ i, L.oneWait.toCompiled.es (x i) ≤ e)
    (hc : ∀ i, (L.oneWait.toCompiled.nPU (x i) : ℝ) * L.oneWait.toCompiled.openCalls (x i) ≤ c)
    (hB : ∀ i, ExpectedCR H (Compiled.BconfX H L.oneWait.toCompiled (Ω i) (P i) (x i)))
    (hX : ∀ i (K : ℕ),
      ExpectedCR H (L.oneWait.toCompiled.finderXCls H (P i) (x i) (s i) (needs i) (cls i) K))
    (hI : ∀ i (K : ℕ),
      ExpectedCR H (L.oneWait.finderICls H (P i) (x i) (s i) (d i).part (needs i) (cls i) K)) :
    Pr (fun σ : S => ∃ i, (run H (P i) L.V (x i) (φ i σ).1 (φ i σ).2).out = .acc ∧
        ∃ u ∈ L.checked (x i) (φ i σ).2, u ∈ L.oneWait.FregTwoX H (P i) (x i) (φ i σ).1 (φ i σ).2) ≤
      ∑ i, (3 * L.κs (x i) + 2 * Real.sqrt (L.V.k (x i) * L.γs (x i) * PrB H L.V (P i) (x i))) +
        2 * (2 * (T + e * As + n * c) + 4 * (T + e * As + Ak + e * Am + (1 + Ks) * (n * c))) /
          2 ^ (256 : ℝ) :=
  phases_oneWait_le_avg_of_phase φ P needs cls
    (fun i _ _ _ _ _ hC => eventTwoX_oneWait_le_cls_avg hF hA hbind hS hopen (hs i) (hPt i) (hd i) (hin i) (P i)
      (needs i) (hpos i) (cls i) (hallX i) (hallI i) hC (hB i) (hX i) (hI i))
    he0 hKs0 hT hAs hAk hAm hKs he hc

end CompiledTwo

end Claims.SamplingGeneric
