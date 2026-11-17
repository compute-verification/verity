module

public import Proofs.Flock.SpotFlock.OneWaitClassAt
public import Proofs.Flock.SpotFlock.ClassCounts
public import Proofs.Flock.SamplingGeneric.TwoOneWaitAvg
public import Proofs.Flock.SamplingGeneric.PhasesAvg
import Proofs.Flock.SamplingGeneric
import Proofs.Flock.Masked
import Proofs.Flock.Claims.KS.FindsFlock
import Proofs.Flock.Claims.KS.ZkExtend
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Sampling.Assembled
import Proofs.Flock.SpotFlock.RevealsLe

/-! The one-wait by-class first-hit bound at SPOT[Flock]'s code at the prover's expected calls, the whole partition's
count paid only on the calls spent in refused runs. -/

@[expose] public section

namespace SamplingDeploy

open Claims

theorem RegTwo.ex_checked_mul_le {Ω : Type} [Fintype Ω] (g : RegTwo) (N : ℕ) (src₁ src₂ : Ω → ByteArray)
    (F : List ℕ → ℝ) (p : Ω → ℝ) {B : ℝ} (hB : 0 ≤ B) (hp : ∀ ω, 0 ≤ p ω)
    (hok : ∀ ω sel n, Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω) = .ok (sel, n) →
      F (g.checked N (src₁ ω) (src₂ ω)) ≤ B) :
    Ex (fun ω => F (g.checked N (src₁ ω) (src₂ ω)) * p ω) ≤
      B * Ex p + F (List.range N) * Ex (fun ω =>
        if Except.toBool (Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω)) = false then p ω
        else 0) := by
  have hpt : ∀ ω, F (g.checked N (src₁ ω) (src₂ ω)) * p ω ≤ B * p ω + F (List.range N) *
      (if Except.toBool (Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω)) = false then p ω
        else 0) := by
    intro ω
    by_cases hR : Except.toBool (Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω)) = false
    · rw [if_pos hR, g.checked_of_refused N _ _ hR]
      have h0 := mul_nonneg hB (hp ω)
      linarith
    · rw [if_neg hR, mul_zero, add_zero]
      refine mul_le_mul_of_nonneg_right ?_ (hp ω)
      cases hs : Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω) with
      | ok d => exact hok ω d.1 d.2 hs
      | error e => exact absurd (by rw [hs]; rfl) hR
  calc Ex (fun ω => F (g.checked N (src₁ ω) (src₂ ω)) * p ω)
      ≤ Ex (fun ω => B * p ω + F (List.range N) *
          (if Except.toBool (Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω)) = false then p ω
            else 0)) := Ex_mono hpt
    _ = Ex (fun ω => B * p ω) + Ex (fun ω => F (List.range N) *
          (if Except.toBool (Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω)) = false then p ω
            else 0)) := Ex_add _ _
    _ = B * Ex p + F (List.range N) * Ex (fun ω =>
          if Except.toBool (Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω)) = false then p ω
          else 0) := by rw [Ex_mul_left, Ex_mul_left]

end SamplingDeploy

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

local notation "𝓛[" F "]" => compiledTwo F (code h) iroots (two h) (rs h) (cap h)

theorem code_ex_mul_le_strata (x : Msg) (G : List ℕ → ℝ) (w M : ℕ → ℕ)
    (hM : ∀ ρ t i, t < h.reg.ns.size → i < h.reg.ns.getD t 0 → w (h.reg.blockOf ρ t i) ≤ M t)
    (hG : ∀ l, G l ≤ ((l.map w).sum : ℝ)) {K : ℕ} (hK : firstK h.reg.first = some K) {Ω : Type} [Fintype Ω]
    (p : Ω × (Fin ((code h).V.k x) → FlockDeploy.Coin) → ℝ) (hp : ∀ q, 0 ≤ p q) :
    Ex (fun q : Ω × (Fin ((code h).V.k x) → FlockDeploy.Coin) => G ((code h).checked x q.2) * p q) ≤
      ((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
            w).sum + K * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum : ℕ) * Ex p +
        (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map w).sum : ℝ) *
          Ex (fun q : Ω × (Fin ((code h).V.k x) → FlockDeploy.Coin) =>
            if Except.toBool (Spot.Select.selectSecond (h.reg.sample (src (q.2 (order h x).first))) h.reg.ns
              h.reg.ks (stageSrc (stageCoins h x q.2))) = false then p q else 0) := by
  have hc : ∀ r : Fin ((code h).V.k x) → FlockDeploy.Coin, (code h).checked x r =
      ((reg h x).map fun g => g.checked (SamplingDeploy.units ((base h).atOrder (order h)) x)
        (src (r (order h x).first)) (stageSrc (stageCoins h x r))).getD [] := fun _ => rfl
  cases hd : decode x with
  | none =>
    have h0 : ∀ r, (code h).checked x r = [] := fun r => by rw [hc]; simp [reg, hd]
    have hG0 : G [] ≤ 0 := by simpa using hG []
    calc Ex (fun q : Ω × (Fin ((code h).V.k x) → FlockDeploy.Coin) => G ((code h).checked x q.2) * p q)
        ≤ Ex (fun q : Ω × (Fin ((code h).V.k x) → FlockDeploy.Coin) => (0 : ℝ) * p q) :=
          Ex_mono fun q => by
            rw [h0]
            exact mul_le_mul_of_nonneg_right hG0 (hp q)
      _ = 0 := by rw [Ex_mul_left, zero_mul]
      _ ≤ _ := add_nonneg (mul_nonneg (Nat.cast_nonneg _) (Ex_nonneg hp))
          (mul_nonneg (Nat.cast_nonneg _) (Ex_nonneg fun q => by
            split_ifs
            · exact hp q
            · exact le_rfl))
  | some _ =>
    have h1 : ∀ r, (code h).checked x r = h.reg.checked (SamplingDeploy.units ((base h).atOrder (order h)) x)
        (src (r (order h x).first)) (stageSrc (stageCoins h x r)) := fun r => by rw [hc]; simp [reg, hd]
    calc Ex (fun q : Ω × (Fin ((code h).V.k x) → FlockDeploy.Coin) => G ((code h).checked x q.2) * p q)
        ≤ Ex (fun q : Ω × (Fin ((code h).V.k x) → FlockDeploy.Coin) =>
            (((h.reg.checked (SamplingDeploy.units ((base h).atOrder (order h)) x)
              (src (q.2 (order h x).first)) (stageSrc (stageCoins h x q.2))).map w).sum : ℝ) * p q) :=
          Ex_mono fun q => by
            rw [h1]
            exact mul_le_mul_of_nonneg_right (hG _) (hp q)
      _ ≤ _ := h.reg.ex_checked_mul_le (Ω := Ω × (Fin ((code h).V.k x) → FlockDeploy.Coin))
          (SamplingDeploy.units ((base h).atOrder (order h)) x) (fun q => src (q.2 (order h x).first))
          (fun q => stageSrc (stageCoins h x q.2)) (fun l => ((l.map w).sum : ℝ)) p (Nat.cast_nonneg _) hp
          fun q _ _ hsel => by
            exact_mod_cast h.reg.sum_checked_le_strata _ _ _ hsel w M hM (h.reg.sample_size_le hK _)

section Counts

variable {Tr : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Tr)

theorem code_hAs (x : Msg) {K : ℕ} (hK : firstK h.reg.first = some K) {Ω : Type} [Fintype Ω]
    (p : Ω × (Fin (𝓛[F].V.k x) → FlockDeploy.Coin) → ℝ) (hp : ∀ q, 0 ≤ p q) :
    Ex (fun q : Ω × (Fin (𝓛[F].V.k x) → FlockDeploy.Coin) =>
      ((𝓛[F].toCompiled.firstSlots x (𝓛[F].checked x q.2)).card : ℝ) * p q) ≤
      (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).length +
          K * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0).sum : ℕ) * Ex p +
        ((SamplingDeploy.units ((base h).atOrder (order h)) x) : ℝ) *
          Ex (fun q : Ω × (Fin (𝓛[F].V.k x) → FlockDeploy.Coin) =>
            if Except.toBool (Spot.Select.selectSecond (h.reg.sample (src (q.2 (order h x).first))) h.reg.ns
              h.reg.ks (stageSrc (stageCoins h x q.2))) = false then p q else 0) := by
  have := code_ex_mul_le_strata h x
    (fun l => ((𝓛[F].toCompiled.firstSlots x (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))).card : ℝ))
    (fun _ => 1) (fun _ => 1) (fun _ _ _ _ _ => le_rfl)
    (fun l => by
      rw [sum_map_one]
      exact_mod_cast (𝓛[F].toCompiled.card_firstSlots_le x (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))
        (𝓛[F].toCompiled.nPU x)).trans (List.length_take_le' _ _)) hK p hp
  rw [sum_map_one, sum_map_one, List.length_range] at this
  simp only [mul_one] at this
  exact this

theorem code_hAk_cls (x : Msg) (needs : ℕ → Finset ℕ) (cls : Fin (𝓛[F].nG x) → Fin (𝓛[F].nG x)) (M : ℕ → ℕ)
    (hM : ∀ ρ t i, t < h.reg.ns.size → i < h.reg.ns.getD t 0 →
      (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[F].nG x) cls needs (h.reg.blockOf ρ t i)).card ≤ M t) {K : ℕ}
    (hK : firstK h.reg.first = some K) {Ω : Type} [Fintype Ω]
    (p : Ω × (Fin (𝓛[F].V.k x) → FlockDeploy.Coin) → ℝ) (hp : ∀ q, 0 ≤ p q) :
    Ex (fun q : Ω × (Fin (𝓛[F].V.k x) → FlockDeploy.Coin) =>
      (((𝓛[F].toCompiled.sampleNeeds x needs (𝓛[F].checked x q.2)).image cls).card : ℝ) * p q) ≤
      ((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[F].nG x) cls needs u).card).sum +
          K * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum : ℕ) * Ex p +
        (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[F].nG x) cls needs u).card).sum : ℝ) *
          Ex (fun q : Ω × (Fin (𝓛[F].V.k x) → FlockDeploy.Coin) =>
            if Except.toBool (Spot.Select.selectSecond (h.reg.sample (src (q.2 (order h x).first))) h.reg.ns
              h.reg.ks (stageSrc (stageCoins h x q.2))) = false then p q else 0) :=
  code_ex_mul_le_strata h x
    (fun l => (((𝓛[F].toCompiled.sampleNeeds x needs
      (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))).image cls).card : ℝ))
    (fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[F].nG x) cls needs u).card) M hM
    (fun l => by
      have h1 : (((𝓛[F].toCompiled.sampleNeeds x needs
          (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))).image cls).card : ℝ) ≤
          ((𝓛[F].toCompiled.sampleNeeds x (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[F].nG x) cls needs)
            (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))).card : ℝ) :=
        Nat.cast_le.2 (Finset.card_le_card (𝓛[F].toCompiled.image_sampleNeeds_subset x needs cls
          (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))))
      have hc := Claims.SamplingGeneric.Compiled.card_filter_needs_le
        (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))
        (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[F].nG x) cls needs) (𝓛[F].toCompiled.nG x)
      refine h1.trans ?_
      exact_mod_cast hc.trans (sum_map_take_le l _ fun u =>
        (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[F].nG x) cls needs u).card)) hK p hp

theorem code_hAm_cls (x : Msg) (needs : ℕ → Finset ℕ) (cls : Fin (𝓛[F].nG x) → Fin (𝓛[F].nG x))
    (hidem : ∀ g, cls (cls g) = cls g) (M : ℕ → ℕ)
    (hM : ∀ ρ t i, t < h.reg.ns.size → i < h.reg.ns.getD t 0 →
      (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[F].nG x) cls needs (h.reg.blockOf ρ t i)).card ≤ M t) {K : ℕ}
    (hK : firstK h.reg.first = some K) {Ω : Type} [Fintype Ω]
    (p : Ω × (Fin (𝓛[F].V.k x) → FlockDeploy.Coin) → ℝ) (hp : ∀ q, 0 ≤ p q) :
    Ex (fun q : Ω × (Fin (𝓛[F].V.k x) → FlockDeploy.Coin) =>
      (𝓛[F].toCompiled.sessCountIn x needs (fun l => (𝓛[F].toCompiled.sampleNeeds x needs l).image cls)
        (𝓛[F].checked x q.2) : ℝ) * p q) ≤
      ((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[F].nG x) cls needs u).card).sum +
          K * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum : ℕ) * Ex p +
        (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[F].nG x) cls needs u).card).sum : ℝ) *
          Ex (fun q : Ω × (Fin (𝓛[F].V.k x) → FlockDeploy.Coin) =>
            if Except.toBool (Spot.Select.selectSecond (h.reg.sample (src (q.2 (order h x).first))) h.reg.ns
              h.reg.ks (stageSrc (stageCoins h x q.2))) = false then p q else 0) :=
  code_ex_mul_le_strata h x
    (fun l => (𝓛[F].toCompiled.sessCountIn x needs (fun l => (𝓛[F].toCompiled.sampleNeeds x needs l).image cls)
      (l.take (Claims.Sampling.nPU (deployOf F (code h)) x)) : ℝ))
    (fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[F].nG x) cls needs u).card) M hM
    (fun l => by
      have h1 : (𝓛[F].toCompiled.sessCountIn x needs
          (fun l => (𝓛[F].toCompiled.sampleNeeds x needs l).image cls)
          (l.take (Claims.Sampling.nPU (deployOf F (code h)) x)) : ℝ) ≤
          (𝓛[F].toCompiled.sessCount x (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[F].nG x) cls needs)
            (l.take (Claims.Sampling.nPU (deployOf F (code h)) x)) : ℝ) :=
        Nat.cast_le.2 (𝓛[F].toCompiled.sessCountIn_cls_le x needs cls hidem
          (l.take (Claims.Sampling.nPU (deployOf F (code h)) x)))
      have hc := 𝓛[F].toCompiled.sum_first_sessNeeds_le x (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))
        (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[F].nG x) cls needs) (𝓛[F].toCompiled.nPU x)
        (𝓛[F].toCompiled.nG x)
      refine h1.trans ?_
      exact_mod_cast hc.trans (sum_map_take_le l _ fun u =>
        (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[F].nG x) cls needs u).card)) hK p hp

theorem code_callsAvg (x : Msg) (needs : ℕ → Finset ℕ) (cls : Fin (𝓛[F].nG x) → Fin (𝓛[F].nG x))
    (hidem : ∀ g, cls (cls g) = cls g) (M : ℕ → ℕ)
    (hM : ∀ ρ t i, t < h.reg.ns.size → i < h.reg.ns.getD t 0 →
      (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[F].nG x) cls needs (h.reg.blockOf ρ t i)).card ≤ M t) {K : ℕ}
    (hK : firstK h.reg.first = some K) {Ω : Type} [Fintype Ω] (P : Prover Ω) {T TR Ks : ℝ}
    (hT : Ex (fun q : Ω × (Fin (𝓛[F].V.k x) → FlockDeploy.Coin) =>
      (𝓛[F].toCompiled.partiesCalls FlockDeploy.sha512 P x q.1 q.2 : ℝ)) ≤ T)
    (hR : Ex (fun q : Ω × (Fin (𝓛[F].V.k x) → FlockDeploy.Coin) =>
      if Except.toBool (Spot.Select.selectSecond (h.reg.sample (src (q.2 (order h x).first))) h.reg.ns
        h.reg.ks (stageSrc (stageCoins h x q.2))) = false then
        (𝓛[F].toCompiled.partiesCalls FlockDeploy.sha512 P x q.1 q.2 : ℝ)
      else 0) ≤ TR)
    (hKs : Ex (fun r : Fin (𝓛[F].V.k x) → FlockDeploy.Coin =>
      (((𝓛[F].toCompiled.sampleNeeds x needs (𝓛[F].checked x r)).image cls).card : ℝ)) ≤ Ks) :
    𝓛[F].toCompiled.CallsAvg FlockDeploy.sha512 P x needs cls T
      ((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).length +
          K * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0).sum : ℕ) * T +
        ((SamplingDeploy.units ((base h).atOrder (order h)) x) : ℝ) * TR)
      (((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[F].nG x) cls needs u).card).sum +
          K * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum : ℕ) * T +
        (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[F].nG x) cls needs u).card).sum : ℝ) * TR)
      (((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[F].nG x) cls needs u).card).sum +
          K * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum : ℕ) * T +
        (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[F].nG x) cls needs u).card).sum : ℝ) * TR)
      Ks := by
  have hp : ∀ q : Ω × (Fin (𝓛[F].V.k x) → FlockDeploy.Coin),
      (0 : ℝ) ≤ (𝓛[F].toCompiled.partiesCalls FlockDeploy.sha512 P x q.1 q.2 : ℝ) :=
    fun q => Nat.cast_nonneg _
  exact ⟨hT,
    (code_hAs h F x hK _ hp).trans (add_le_add (mul_le_mul_of_nonneg_left hT (Nat.cast_nonneg _))
      (mul_le_mul_of_nonneg_left hR (Nat.cast_nonneg _))),
    (code_hAk_cls h F x needs cls M hM hK _ hp).trans (add_le_add (mul_le_mul_of_nonneg_left hT (Nat.cast_nonneg _))
      (mul_le_mul_of_nonneg_left hR (Nat.cast_nonneg _))),
    (code_hAm_cls h F x needs cls hidem M hM hK _ hp).trans
      (add_le_add (mul_le_mul_of_nonneg_left hT (Nat.cast_nonneg _))
        (mul_le_mul_of_nonneg_left hR (Nat.cast_nonneg _))), hKs⟩

end Counts

local notation "Fzk" => FlockDeploy.deployZk _root_.Flock.Readable.relation

/-- No budget is held on every run. -/
theorem code_eventTwoX_oneWait_le_cls_avg (hB : Built h) (hS : Addresses h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin (Claims.SamplingGeneric.Arg.ofFlock Fzk)}
    (hL : L = 𝓛[Fzk]) {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hx : parse h x = some s)
    (hP : partition h s = some Pt) (hd : two h s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {needs : ℕ → Finset ℕ}
    (hneeds : needs = fun u => (Reveal.needs (partition h) s u).toFinset) {cls : Fin (L.nG x) → Fin (L.nG x)}
    (hcls : cls = Verity.FirstHit.clsOf fun g : Fin (L.nG x) => gateKey h s Pt d g.val) {T As Ak Am Ks : ℝ}
    (hC : L.toCompiled.CallsAvg FlockDeploy.sha512 P x needs cls T As Ak Am Ks)
    (hBc : ExpectedCR FlockDeploy.sha512
      (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 L.oneWait.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512
      (L.oneWait.toCompiled.finderXCls FlockDeploy.sha512 P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512
      (L.oneWait.finderICls FlockDeploy.sha512 P x s d.part needs cls K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → FlockDeploy.Coin) =>
        (run FlockDeploy.sha512 P L.V x p.1 p.2).out = .acc ∧
          ∃ u ∈ L.checked x p.2, u ∈ L.oneWait.FregTwoX FlockDeploy.sha512 P x p.1 p.2) ≤
      L.εclsOneWaitAvg FlockDeploy.sha512 x P T As Ak Am Ks := by
  have hA : (Claims.SamplingGeneric.Arg.ofFlock Fzk).KS FlockDeploy.sha512 :=
    Claims.SamplingGeneric.flockArgKS FlockDeploy.sha512 _ _root_.Flock.SecurityProofs.MaskedKS
      (Claims.Sampling.listWithin_deployZk _)
  have hbind : L.commit.Binds FlockDeploy.sha512 := by
    subst hL
    exact SamplingDeploy.commit_binds (code h) _
  have hSess : L.Sessions FlockDeploy.sha512 := by
    subst hL
    exact sessionsTwo_spot h
      (sessionOpensTwo_of_frameIs h (layoutFor h) (embeds_layoutFor h) (frameIs_built h hB hS))
  have hopen : L.toCompiled.OpenCallsLe FlockDeploy.sha512 := by
    subst hL
    exact SamplingDeploy.openCallsLe_compiled _ (code h) iroots _ (revealsLe h)
  have hs : L.parse x = some s := by
    subst hL
    exact hx
  have hPt : L.partition s = some Pt := by
    subst hL
    exact hP
  have hd' : L.two s = some d := by
    subst hL
    exact hd
  have hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u := by
    subst hL hneeds
    exact fun u v w => code_hpos h s u v w
  have hblock : ∀ g g' : Fin (L.nG x), cls g = cls g' → gateBlock d.part g.val = gateBlock d.part g'.val := by
    subst hcls
    intro g g' hgg
    exact gateKey_block h ((Verity.FirstHit.clsOf_eq_iff _ g g').1 hgg)
  have hkept : ∀ (ω : Ω) (r : Fin (L.V.k x) → FlockDeploy.Coin) (c : L.oneWait.SlotEC x),
      L.oneWait.toCompiled.ClassWhole FlockDeploy.sha512 x cls (L.oneWait.kept FlockDeploy.sha512 P x s ω r c) := by
    subst hcls
    exact fun ω r c => L.oneWait.toCompiled.kept_classWhole_of_wit FlockDeploy.sha512 P x s ω _ r c
      (code_hwit_oneWait h hB hS hL hx hP hd hin P ω r c)
  exact Claims.SamplingGeneric.CompiledTwo.eventTwoX_oneWait_le_cls_avg oneWaitFinds_zk hA hbind hSess hopen
    hs hPt hd' hin P needs hpos cls
    (fun a g g' hgg hg =>
      L.oneWait.toCompiled.opensAt_isSome_cls FlockDeploy.sha512 P x a.1 hs cls a.2.1 a.2.2
        (hkept a.1 a.2.1 a.2.2) g g' hgg hg)
    (fun t ec g g' hgg hg =>
      L.oneWait.iopTable_isSome_cls FlockDeploy.sha512 P x t.1 hs d.part cls hblock t.2 ec (hkept t.1 _ ec.2) g g'
        hgg hg)
    hC hBc hX hI

theorem code_eventTwoX_oneWait_le_cls_avg_strata (hB : Built h) (hS : Addresses h) {x : Msg} {s : AuditStatement}
    {Pt : Partition} {d : Design.Two} (hx : parse h x = some s) (hP : partition h s = some Pt)
    (hd : two h s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    {needs : ℕ → Finset ℕ} (hneeds : needs = fun u => (Reveal.needs (partition h) s u).toFinset)
    {cls : Fin (𝓛[Fzk].nG x) → Fin (𝓛[Fzk].nG x)}
    (hcls : cls = Verity.FirstHit.clsOf fun g : Fin (𝓛[Fzk].nG x) => gateKey h s Pt d g.val) (M : ℕ → ℕ)
    (hM : ∀ ρ t i, t < h.reg.ns.size → i < h.reg.ns.getD t 0 →
      (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[Fzk].nG x) cls needs (h.reg.blockOf ρ t i)).card ≤ M t) {K₀ : ℕ}
    (hK : firstK h.reg.first = some K₀) {T TR Ks : ℝ}
    (hT : Ex (fun q : Ω × (Fin (𝓛[Fzk].V.k x) → FlockDeploy.Coin) =>
      (𝓛[Fzk].toCompiled.partiesCalls FlockDeploy.sha512 P x q.1 q.2 : ℝ)) ≤ T)
    (hR : Ex (fun q : Ω × (Fin (𝓛[Fzk].V.k x) → FlockDeploy.Coin) =>
      if Except.toBool (Spot.Select.selectSecond (h.reg.sample (src (q.2 (order h x).first))) h.reg.ns
        h.reg.ks (stageSrc (stageCoins h x q.2))) = false then
        (𝓛[Fzk].toCompiled.partiesCalls FlockDeploy.sha512 P x q.1 q.2 : ℝ)
      else 0) ≤ TR)
    (hKs : Ex (fun r : Fin (𝓛[Fzk].V.k x) → FlockDeploy.Coin =>
      (((𝓛[Fzk].toCompiled.sampleNeeds x needs (𝓛[Fzk].checked x r)).image cls).card : ℝ)) ≤ Ks)
    (hBc : ExpectedCR FlockDeploy.sha512
      (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 𝓛[Fzk].oneWait.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512
      (𝓛[Fzk].oneWait.toCompiled.finderXCls FlockDeploy.sha512 P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512
      (𝓛[Fzk].oneWait.finderICls FlockDeploy.sha512 P x s d.part needs cls K)) :
    Pr (fun p : Ω × (Fin (𝓛[Fzk].V.k x) → FlockDeploy.Coin) =>
        (run FlockDeploy.sha512 P 𝓛[Fzk].V x p.1 p.2).out = .acc ∧
          ∃ u ∈ 𝓛[Fzk].checked x p.2, u ∈ 𝓛[Fzk].oneWait.FregTwoX FlockDeploy.sha512 P x p.1 p.2) ≤
      𝓛[Fzk].εclsOneWaitAvg FlockDeploy.sha512 x P T
        ((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v =>
              h.reg.always v).length +
            K₀ * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0).sum : ℕ) * T +
          ((SamplingDeploy.units ((base h).atOrder (order h)) x) : ℝ) * TR)
        (((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v =>
              h.reg.always v).map
              fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[Fzk].nG x) cls needs u).card).sum +
            K₀ * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum : ℕ) * T +
          (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map
              fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[Fzk].nG x) cls needs u).card).sum : ℝ) * TR)
        (((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v =>
              h.reg.always v).map
              fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[Fzk].nG x) cls needs u).card).sum +
            K₀ * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum : ℕ) * T +
          (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map
              fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[Fzk].nG x) cls needs u).card).sum : ℝ) * TR)
        Ks :=
  code_eventTwoX_oneWait_le_cls_avg h hB hS rfl hx hP hd hin P hneeds hcls
    (code_callsAvg h Fzk x needs cls (by subst hcls; exact Claims.SamplingGeneric.Compiled.clsOf_idem _) M hM hK P
      hT hR hKs) hBc hX hI

/-- No count is held at every sample, and no run or phase of a prover to a budget. -/
theorem code_phases_oneWait_le_avg (hB : Built h) (hS : Addresses h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin (Claims.SamplingGeneric.Arg.ofFlock Fzk)}
    (hL : L = 𝓛[Fzk]) {n : ℕ} {x : Fin n → Msg} {s : Fin n → AuditStatement} {Pt : Fin n → Partition}
    {d : Fin n → Design.Two} (hx : ∀ i, parse h (x i) = some (s i))
    (hP : ∀ i, partition h (s i) = some (Pt i)) (hd : ∀ i, two h (s i) = some (d i))
    (hin : ∀ i, InScopeTwo (Pt i) (d i) (s i)) {S : Type} [Fintype S] {Ω : Fin n → Type} [∀ i, Fintype (Ω i)]
    [∀ i, Nonempty (Ω i)] (φ : ∀ i, S ≃ Ω i × (Fin (L.V.k (x i)) → FlockDeploy.Coin)) (P : ∀ i, Prover (Ω i))
    {needs : Fin n → ℕ → Finset ℕ}
    (hneeds : ∀ i, needs i = fun u => (Reveal.needs (partition h) (s i) u).toFinset)
    {cls : ∀ i, Fin (L.nG (x i)) → Fin (L.nG (x i))}
    (hcls : ∀ i, cls i = Verity.FirstHit.clsOf fun g : Fin (L.nG (x i)) => gateKey h (s i) (Pt i) (d i) g.val)
    {T As Ak Am Ks e c : ℝ} (he0 : 0 ≤ e) (hKs0 : 0 ≤ Ks)
    (hT : Ex (fun σ : S =>
      ∑ i, (L.toCompiled.partiesCalls FlockDeploy.sha512 (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)) ≤ T)
    (hAs : Ex (fun σ : S => ∑ i, ((L.toCompiled.firstSlots (x i) (L.checked (x i) (φ i σ).2)).card : ℝ) *
      (L.toCompiled.partiesCalls FlockDeploy.sha512 (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)) ≤ As)
    (hAk : Ex (fun σ : S => ∑ i,
      (((L.toCompiled.sampleNeeds (x i) (needs i) (L.checked (x i) (φ i σ).2)).image (cls i)).card : ℝ) *
        (L.toCompiled.partiesCalls FlockDeploy.sha512 (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)) ≤ Ak)
    (hAm : Ex (fun σ : S => ∑ i,
      (L.toCompiled.sessCountIn (x i) (needs i)
        (fun l => (L.toCompiled.sampleNeeds (x i) (needs i) l).image (cls i)) (L.checked (x i) (φ i σ).2) : ℝ) *
          (L.toCompiled.partiesCalls FlockDeploy.sha512 (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)) ≤ Am)
    (hKs : ∀ i, Ex (fun r : Fin (L.V.k (x i)) → FlockDeploy.Coin =>
      (((L.toCompiled.sampleNeeds (x i) (needs i) (L.checked (x i) r)).image (cls i)).card : ℝ)) ≤ Ks)
    (he : ∀ i, L.oneWait.toCompiled.es (x i) ≤ e)
    (hc : ∀ i, (L.oneWait.toCompiled.nPU (x i) : ℝ) * L.oneWait.toCompiled.openCalls (x i) ≤ c)
    (hBc : ∀ i, ExpectedCR FlockDeploy.sha512
      (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 L.oneWait.toCompiled (Ω i) (P i) (x i)))
    (hX : ∀ i (K : ℕ), ExpectedCR FlockDeploy.sha512
      (L.oneWait.toCompiled.finderXCls FlockDeploy.sha512 (P i) (x i) (s i) (needs i) (cls i) K))
    (hI : ∀ i (K : ℕ), ExpectedCR FlockDeploy.sha512
      (L.oneWait.finderICls FlockDeploy.sha512 (P i) (x i) (s i) (d i).part (needs i) (cls i) K)) :
    Pr (fun σ : S => ∃ i, (run FlockDeploy.sha512 (P i) L.V (x i) (φ i σ).1 (φ i σ).2).out = .acc ∧
        ∃ u ∈ L.checked (x i) (φ i σ).2,
          u ∈ L.oneWait.FregTwoX FlockDeploy.sha512 (P i) (x i) (φ i σ).1 (φ i σ).2) ≤
      ∑ i, (3 * L.κs (x i) +
          2 * Real.sqrt (L.V.k (x i) * L.γs (x i) * PrB FlockDeploy.sha512 L.V (P i) (x i))) +
        2 * (2 * (T + e * As + n * c) + 4 * (T + e * As + Ak + e * Am + (1 + Ks) * (n * c))) /
          2 ^ (256 : ℝ) :=
  Claims.SamplingGeneric.CompiledTwo.phases_oneWait_le_avg_of_phase φ P needs cls
    (fun i _ _ _ _ _ hC => code_eventTwoX_oneWait_le_cls_avg h hB hS hL (hx i) (hP i) (hd i) (hin i) (P i)
      (hneeds i) (hcls i) hC (hBc i) (hX i) (hI i))
    he0 hKs0 hT hAs hAk hAm hKs he hc

theorem code_phases_oneWait_le_avg_strata (hB : Built h) (hS : Addresses h) {n : ℕ} {x : Fin n → Msg}
    {s : Fin n → AuditStatement} {Pt : Fin n → Partition} {d : Fin n → Design.Two}
    (hx : ∀ i, parse h (x i) = some (s i)) (hP : ∀ i, partition h (s i) = some (Pt i))
    (hd : ∀ i, two h (s i) = some (d i)) (hin : ∀ i, InScopeTwo (Pt i) (d i) (s i)) {S : Type} [Fintype S]
    {Ω : Fin n → Type} [∀ i, Fintype (Ω i)] [∀ i, Nonempty (Ω i)]
    (φ : ∀ i, S ≃ Ω i × (Fin (𝓛[Fzk].V.k (x i)) → FlockDeploy.Coin)) (P : ∀ i, Prover (Ω i))
    {needs : Fin n → ℕ → Finset ℕ}
    (hneeds : ∀ i, needs i = fun u => (Reveal.needs (partition h) (s i) u).toFinset)
    {cls : ∀ i, Fin (𝓛[Fzk].nG (x i)) → Fin (𝓛[Fzk].nG (x i))}
    (hcls : ∀ i, cls i =
      Verity.FirstHit.clsOf fun g : Fin (𝓛[Fzk].nG (x i)) => gateKey h (s i) (Pt i) (d i) g.val)
    (M : ℕ → ℕ)
    (hM : ∀ i ρ t j, t < h.reg.ns.size → j < h.reg.ns.getD t 0 →
      (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[Fzk].nG (x i)) (cls i) (needs i) (h.reg.blockOf ρ t j)).card ≤ M t)
    {K₀ : ℕ} (hK : firstK h.reg.first = some K₀) {Cs N Ck Sk : ℕ}
    (hCs : ∀ i, ((List.range (SamplingDeploy.units ((base h).atOrder (order h)) (x i))).filter fun v =>
        h.reg.always v).length +
      K₀ * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0).sum ≤ Cs)
    (hN : ∀ i, SamplingDeploy.units ((base h).atOrder (order h)) (x i) ≤ N)
    (hCk : ∀ i, (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) (x i))).filter fun v =>
        h.reg.always v).map
        fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[Fzk].nG (x i)) (cls i) (needs i) u).card).sum +
      K₀ * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum ≤ Ck)
    (hSk : ∀ i, ((List.range (SamplingDeploy.units ((base h).atOrder (order h)) (x i))).map
      fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (𝓛[Fzk].nG (x i)) (cls i) (needs i) u).card).sum ≤ Sk)
    {T TR Ks e c : ℝ} (he0 : 0 ≤ e) (hKs0 : 0 ≤ Ks)
    (hT : Ex (fun σ : S =>
      ∑ i, (𝓛[Fzk].toCompiled.partiesCalls FlockDeploy.sha512 (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)) ≤ T)
    (hR : Ex (fun σ : S => ∑ i,
      if Except.toBool (Spot.Select.selectSecond (h.reg.sample (src ((φ i σ).2 (order h (x i)).first)))
        h.reg.ns h.reg.ks (stageSrc (stageCoins h (x i) (φ i σ).2))) = false then
        (𝓛[Fzk].toCompiled.partiesCalls FlockDeploy.sha512 (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)
      else 0) ≤ TR)
    (hKs : ∀ i, Ex (fun r : Fin (𝓛[Fzk].V.k (x i)) → FlockDeploy.Coin =>
      (((𝓛[Fzk].toCompiled.sampleNeeds (x i) (needs i) (𝓛[Fzk].checked (x i) r)).image (cls i)).card : ℝ)) ≤ Ks)
    (he : ∀ i, 𝓛[Fzk].oneWait.toCompiled.es (x i) ≤ e)
    (hc : ∀ i, (𝓛[Fzk].oneWait.toCompiled.nPU (x i) : ℝ) * 𝓛[Fzk].oneWait.toCompiled.openCalls (x i) ≤ c)
    (hBc : ∀ i, ExpectedCR FlockDeploy.sha512
      (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 𝓛[Fzk].oneWait.toCompiled (Ω i) (P i) (x i)))
    (hX : ∀ i (K : ℕ), ExpectedCR FlockDeploy.sha512
      (𝓛[Fzk].oneWait.toCompiled.finderXCls FlockDeploy.sha512 (P i) (x i) (s i) (needs i) (cls i) K))
    (hI : ∀ i (K : ℕ), ExpectedCR FlockDeploy.sha512
      (𝓛[Fzk].oneWait.finderICls FlockDeploy.sha512 (P i) (x i) (s i) (d i).part (needs i) (cls i) K)) :
    Pr (fun σ : S => ∃ i, (run FlockDeploy.sha512 (P i) 𝓛[Fzk].V (x i) (φ i σ).1 (φ i σ).2).out = .acc ∧
        ∃ u ∈ 𝓛[Fzk].checked (x i) (φ i σ).2,
          u ∈ 𝓛[Fzk].oneWait.FregTwoX FlockDeploy.sha512 (P i) (x i) (φ i σ).1 (φ i σ).2) ≤
      ∑ i, (3 * 𝓛[Fzk].κs (x i) +
          2 * Real.sqrt (𝓛[Fzk].V.k (x i) * 𝓛[Fzk].γs (x i) * PrB FlockDeploy.sha512 𝓛[Fzk].V (P i) (x i))) +
        2 * (2 * (T + e * (Cs * T + N * TR) + n * c) +
            4 * (T + e * (Cs * T + N * TR) + (Ck * T + Sk * TR) + e * (Ck * T + Sk * TR) +
              (1 + Ks) * (n * c))) /
          2 ^ (256 : ℝ) := by
  have hp : ∀ i (q : Ω i × (Fin (𝓛[Fzk].V.k (x i)) → FlockDeploy.Coin)),
      (0 : ℝ) ≤ (𝓛[Fzk].toCompiled.partiesCalls FlockDeploy.sha512 (P i) (x i) q.1 q.2 : ℝ) :=
    fun i q => Nat.cast_nonneg _
  have hpr : ∀ i (q : Ω i × (Fin (𝓛[Fzk].V.k (x i)) → FlockDeploy.Coin)), (0 : ℝ) ≤
      if Except.toBool (Spot.Select.selectSecond (h.reg.sample (src (q.2 (order h (x i)).first))) h.reg.ns
        h.reg.ks (stageSrc (stageCoins h (x i) q.2))) = false then
        (𝓛[Fzk].toCompiled.partiesCalls FlockDeploy.sha512 (P i) (x i) q.1 q.2 : ℝ)
      else 0 := fun i q => by
    split_ifs
    · exact Nat.cast_nonneg _
    · exact le_rfl
  have hidem : ∀ i g, cls i (cls i g) = cls i g := fun i g => by
    rw [hcls i]
    exact Claims.SamplingGeneric.Compiled.clsOf_idem _ g
  have hTs : ∑ i, Ex (fun q : Ω i × (Fin (𝓛[Fzk].V.k (x i)) → FlockDeploy.Coin) =>
      (𝓛[Fzk].toCompiled.partiesCalls FlockDeploy.sha512 (P i) (x i) q.1 q.2 : ℝ)) ≤ T :=
    (Claims.SamplingGeneric.Ex_phases_sum φ fun i (q : Ω i × (Fin (𝓛[Fzk].V.k (x i)) → FlockDeploy.Coin)) =>
      (𝓛[Fzk].toCompiled.partiesCalls FlockDeploy.sha512 (P i) (x i) q.1 q.2 : ℝ)).symm.le.trans hT
  have hRs : ∑ i, Ex (fun q : Ω i × (Fin (𝓛[Fzk].V.k (x i)) → FlockDeploy.Coin) =>
      if Except.toBool (Spot.Select.selectSecond (h.reg.sample (src (q.2 (order h (x i)).first))) h.reg.ns
        h.reg.ks (stageSrc (stageCoins h (x i) q.2))) = false then
        (𝓛[Fzk].toCompiled.partiesCalls FlockDeploy.sha512 (P i) (x i) q.1 q.2 : ℝ)
      else 0) ≤ TR :=
    (Claims.SamplingGeneric.Ex_phases_sum φ fun i (q : Ω i × (Fin (𝓛[Fzk].V.k (x i)) → FlockDeploy.Coin)) =>
      if Except.toBool (Spot.Select.selectSecond (h.reg.sample (src (q.2 (order h (x i)).first))) h.reg.ns
        h.reg.ks (stageSrc (stageCoins h (x i) q.2))) = false then
        (𝓛[Fzk].toCompiled.partiesCalls FlockDeploy.sha512 (P i) (x i) q.1 q.2 : ℝ)
      else 0).symm.le.trans hR
  have hAs : Ex (fun σ : S =>
      ∑ i, ((𝓛[Fzk].toCompiled.firstSlots (x i) (𝓛[Fzk].checked (x i) (φ i σ).2)).card : ℝ) *
        (𝓛[Fzk].toCompiled.partiesCalls FlockDeploy.sha512 (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)) ≤ Cs * T + N * TR :=
    (Claims.SamplingGeneric.Ex_phases_sum φ fun i (q : Ω i × (Fin (𝓛[Fzk].V.k (x i)) → FlockDeploy.Coin)) =>
      ((𝓛[Fzk].toCompiled.firstSlots (x i) (𝓛[Fzk].checked (x i) q.2)).card : ℝ) *
        (𝓛[Fzk].toCompiled.partiesCalls FlockDeploy.sha512 (P i) (x i) q.1 q.2 : ℝ)).le.trans
      (Claims.SamplingGeneric.sum_two_le _ _ _ _ _ (fun i => code_hAs h Fzk (x i) hK _ (hp i))
        (fun i => Ex_nonneg (hp i)) (fun i => Ex_nonneg (hpr i)) (fun i => by exact_mod_cast hCs i)
        (fun i => by exact_mod_cast hN i) (Nat.cast_nonneg _) (Nat.cast_nonneg _) hTs hRs)
  have hAk : Ex (fun σ : S => ∑ i,
      (((𝓛[Fzk].toCompiled.sampleNeeds (x i) (needs i) (𝓛[Fzk].checked (x i) (φ i σ).2)).image (cls i)).card : ℝ) *
        (𝓛[Fzk].toCompiled.partiesCalls FlockDeploy.sha512 (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)) ≤ Ck * T + Sk * TR :=
    (Claims.SamplingGeneric.Ex_phases_sum φ fun i (q : Ω i × (Fin (𝓛[Fzk].V.k (x i)) → FlockDeploy.Coin)) =>
      (((𝓛[Fzk].toCompiled.sampleNeeds (x i) (needs i) (𝓛[Fzk].checked (x i) q.2)).image (cls i)).card : ℝ) *
        (𝓛[Fzk].toCompiled.partiesCalls FlockDeploy.sha512 (P i) (x i) q.1 q.2 : ℝ)).le.trans
      (Claims.SamplingGeneric.sum_two_le _ _ _ _ _
        (fun i => code_hAk_cls h Fzk (x i) (needs i) (cls i) M (hM i) hK _ (hp i))
        (fun i => Ex_nonneg (hp i)) (fun i => Ex_nonneg (hpr i)) (fun i => by exact_mod_cast hCk i)
        (fun i => by exact_mod_cast hSk i) (Nat.cast_nonneg _) (Nat.cast_nonneg _) hTs hRs)
  have hAm : Ex (fun σ : S => ∑ i,
      (𝓛[Fzk].toCompiled.sessCountIn (x i) (needs i)
        (fun l => (𝓛[Fzk].toCompiled.sampleNeeds (x i) (needs i) l).image (cls i))
          (𝓛[Fzk].checked (x i) (φ i σ).2) : ℝ) *
        (𝓛[Fzk].toCompiled.partiesCalls FlockDeploy.sha512 (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)) ≤ Ck * T + Sk * TR :=
    (Claims.SamplingGeneric.Ex_phases_sum φ fun i (q : Ω i × (Fin (𝓛[Fzk].V.k (x i)) → FlockDeploy.Coin)) =>
      (𝓛[Fzk].toCompiled.sessCountIn (x i) (needs i)
        (fun l => (𝓛[Fzk].toCompiled.sampleNeeds (x i) (needs i) l).image (cls i))
          (𝓛[Fzk].checked (x i) q.2) : ℝ) *
        (𝓛[Fzk].toCompiled.partiesCalls FlockDeploy.sha512 (P i) (x i) q.1 q.2 : ℝ)).le.trans
      (Claims.SamplingGeneric.sum_two_le _ _ _ _ _
        (fun i => code_hAm_cls h Fzk (x i) (needs i) (cls i) (hidem i) M (hM i) hK _ (hp i))
        (fun i => Ex_nonneg (hp i)) (fun i => Ex_nonneg (hpr i)) (fun i => by exact_mod_cast hCk i)
        (fun i => by exact_mod_cast hSk i) (Nat.cast_nonneg _) (Nat.cast_nonneg _) hTs hRs)
  exact code_phases_oneWait_le_avg h hB hS rfl hx hP hd hin φ P hneeds hcls he0 hKs0 hT hAs hAk hAm hKs he hc hBc
    hX hI

end SpotFlock
