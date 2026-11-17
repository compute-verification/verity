module

public import Proofs.Flock.SamplingGeneric.PhasesOnce
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Bext
import Proofs.Flock.Claims.Sampling.Extract
import Proofs.Flock.Claims.Sampling.Wait

/-! The first-hit bound over a monitoring period, with the sample's counts entering as the period's averages. -/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeTwo)

theorem sum_lin4 {n : ℕ} (x₁ x₂ x₃ x₄ : Fin n → ℝ) (c₁ c₂ c₃ c₄ κ : ℝ) :
    ∑ i, (c₁ * x₁ i + c₂ * x₂ i + c₃ * x₃ i + c₄ * x₄ i + κ) =
      c₁ * ∑ i, x₁ i + c₂ * ∑ i, x₂ i + c₃ * ∑ i, x₃ i + c₄ * ∑ i, x₄ i + n * κ := by
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_add_distrib,
    ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]

theorem sum_two_le {n : ℕ} (a f r C N : Fin n → ℝ) {C' N' T R : ℝ} (ha : ∀ i, a i ≤ C i * f i + N i * r i)
    (hf : ∀ i, 0 ≤ f i) (hr : ∀ i, 0 ≤ r i) (hC : ∀ i, C i ≤ C') (hN : ∀ i, N i ≤ N') (hC0 : 0 ≤ C')
    (hN0 : 0 ≤ N') (hT : ∑ i, f i ≤ T) (hR : ∑ i, r i ≤ R) : ∑ i, a i ≤ C' * T + N' * R :=
  calc ∑ i, a i ≤ ∑ i, (C' * f i + N' * r i) :=
        Finset.sum_le_sum fun i _ => (ha i).trans
          (add_le_add (mul_le_mul_of_nonneg_right (hC i) (hf i)) (mul_le_mul_of_nonneg_right (hN i) (hr i)))
    _ = C' * ∑ i, f i + N' * ∑ i, r i := by
        rw [Finset.sum_add_distrib, Finset.mul_sum, Finset.mul_sum]
    _ ≤ C' * T + N' * R :=
        add_le_add (mul_le_mul_of_nonneg_left hT hC0) (mul_le_mul_of_nonneg_left hR hN0)

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


/-- No count is held at every sample, and no run of a prover to a budget. -/
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
