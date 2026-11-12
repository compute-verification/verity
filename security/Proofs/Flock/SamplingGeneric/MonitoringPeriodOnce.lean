module

public import Proofs.Flock.SamplingGeneric.TwoOneWaitAvg
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Bext
import Proofs.Flock.Claims.Sampling.Extract
import Proofs.Flock.Claims.Sampling.Wait

/-!
# The first-hit bound over a monitoring period, the prover's calls charged once

`eventTwoX_oneWait_le_cls_exp` and `Compiled.eventX_le_cls_exp` bound one execution phase's event, an accepted run
sampling a proof block of the committed trace's incorrect set, with the prover's SHA-512 calls as an average over its
tape and the phase's coins. Here a monitoring period is `n` execution phases, and the event is that **some** phase of
it has that event. Its probability is at most the phases' bounds summed, and since the part of a phase's bound that
counts calls is linear in the phase's expected calls, that part is paid once, at the period's expected total:

`Σ_i (3·κ_s + 2·√(k·γ_s·Pr[B_i]))`
`+ 2·(2·(T·(1 + e·N_s) + n·c) + 4·(T·(1 + e·N_s + K_s + e·K_m) + (1 + K_s)·n·c))/2^256`

with `T` at least the expected number of calls of all `n` phases together, both parties' counted. A prover that
spends its calls in the phases where the earlier phases' coins fell its way is charged what it spends on average over
the period, not its largest spending in every phase. The opening checks' own calls `c` and the two terms that count
no call of the prover's stay one a phase.

* `monitoringPeriodSplit`: **a period's sample space, as one of its phases reads it.** The period's sample space is
  an adversary's tape and every phase's coins. Phase `i`'s prover reads the tape and the other phases' coins as its
  own tape, and phase `i`'s coins are uniform and independent of that. The theorems below take any sample space with
  such a splitting at each phase (`φ`); this is the one where every phase's coins are independent.
* `Pr_monitoringPeriod_le`, `Ex_monitoringPeriod_sum`: an event of some phase has at most the phases' probabilities
  summed, and the average of a sum over the phases is the phases' averages summed.
* `sum_hash_le`: the phases' parts that count calls, summed, are at most that part at the summed expected calls and
  at `n` times the opening checks' own.
* `pr_monitoringPeriod_once_le`: **the period's step**, for phases whose bounds have that shape.
* `CompiledTwo.monitoringPeriod_oneWait_le_once`: the period's bound at two stages with one wait, each phase under
  `eventTwoX_oneWait_le_cls_exp`'s premises. `Compiled.monitoringPeriod_eventX_le_once`: the same at one stage, each
  phase under `Compiled.eventX_le_cls_exp`'s premises, with `κ_s + √(k·γ_s·Pr[B_i])` a phase and `2·… + 2·…` over
  `2^256`.

Each phase's statement is fixed before the period. The sample's three counts are bounded at every sample, by one
bound for all phases, as are the sessions' expected reruns `e` and the opening checks' calls `c`. No statement of the
definitions tree changes.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeTwo)

/-! ## A period's sample space -/

/-- **A period's sample space, split at phase `i`**: an adversary's tape `Θ` and every phase's coins, as the pair of
phase `i`'s prover's tape (the adversary's tape and the other phases' coins) and phase `i`'s own coins. -/
def monitoringPeriodSplit (Θ : Type) {n : ℕ} (C : Fin n → Type) (i : Fin n) :
    Θ × ((j : Fin n) → C j) ≃ (Θ × ((j : {j : Fin n // j ≠ i}) → C j)) × C i :=
  ((Equiv.refl Θ).prodCongr (Equiv.piSplitAt i C)).trans
    ⟨fun p => ((p.1, p.2.2), p.2.1), fun q => (q.1.1, (q.2, q.1.2)), fun _ => rfl, fun _ => rfl⟩

/-- **Some phase's event has at most the phases' probabilities summed**, each phase's over its prover's tape and its
own coins. -/
theorem Pr_monitoringPeriod_le {S : Type} [Fintype S] {n : ℕ} {Ω C : Fin n → Type} [∀ i, Fintype (Ω i)]
    [∀ i, Fintype (C i)] (φ : ∀ i, S ≃ Ω i × C i) (E : ∀ i, Ω i × C i → Prop) :
    Pr (fun σ : S => ∃ i, E i (φ i σ)) ≤ ∑ i, Pr (E i) := by
  have h := Claims.Sampling.Pr_exists_le (α := S) (Finset.univ : Finset (Fin n)) fun i σ => E i (φ i σ)
  have h1 : Pr (fun σ : S => ∃ i, E i (φ i σ)) = Pr (fun σ : S => ∃ i ∈ (Finset.univ : Finset (Fin n)), E i (φ i σ)) :=
    Pr_congr fun σ => by simp
  have h2 : ∑ i ∈ (Finset.univ : Finset (Fin n)), Pr (fun σ : S => E i (φ i σ)) = ∑ i, Pr (E i) :=
    Finset.sum_congr rfl fun i _ => Claims.Sampling.Pr_equiv (φ i) (E i)
  exact h1.le.trans (h.trans h2.le)

/-- **The average of a sum over the phases is the phases' averages summed**, each phase's over its prover's tape and
its own coins. -/
theorem Ex_monitoringPeriod_sum {S : Type} [Fintype S] {n : ℕ} {Ω C : Fin n → Type} [∀ i, Fintype (Ω i)]
    [∀ i, Fintype (C i)] (φ : ∀ i, S ≃ Ω i × C i) (f : ∀ i, Ω i × C i → ℝ) :
    Ex (fun σ : S => ∑ i, f i (φ i σ)) = ∑ i, Ex (f i) :=
  (Claims.Sampling.Ex_finsum (Finset.univ : Finset (Fin n)) fun i (σ : S) => f i (φ i σ)).trans
    (Finset.sum_congr rfl fun i _ => Claims.Sampling.Ex_equiv (φ i) (f i))

/-! ## The phases' parts that count calls, summed -/

/-- **The phases' parts that count calls, summed.** Phase `i` pays `a` times `T_i·(1 + e_i·N_s) + c_i` and `f` times
`T_i·(1 + e_i·N_s + K_s + e_i·K_m) + (1 + K_s)·c_i`. With `e_i ≤ e`, `c_i ≤ c` and `Σ_i T_i ≤ T`, the phases together
pay at most the same expression at `T`, at `e` and at `n·c`. -/
theorem sum_hash_le {n : ℕ} (Tq es c : Fin n → ℝ) {a f T e cc Ns Ks Km : ℝ} (ha : 0 ≤ a) (hf : 0 ≤ f)
    (hT0 : ∀ i, 0 ≤ Tq i) (he0 : 0 ≤ e) (hNs0 : 0 ≤ Ns) (hKs0 : 0 ≤ Ks) (hKm0 : 0 ≤ Km) (he : ∀ i, es i ≤ e)
    (hc : ∀ i, c i ≤ cc) (hT : ∑ i, Tq i ≤ T) :
    ∑ i, (a * (Tq i * (1 + es i * Ns) + c i) +
        f * (Tq i * (1 + es i * Ns + Ks + es i * Km) + (1 + Ks) * c i)) ≤
      a * (T * (1 + e * Ns) + n * cc) + f * (T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * (n * cc)) := by
  have h0 : 0 ≤ e * Ns := mul_nonneg he0 hNs0
  have h0' : 0 ≤ e * Km := mul_nonneg he0 hKm0
  have hα : 0 ≤ a * (1 + e * Ns) + f * (1 + e * Ns + Ks + e * Km) :=
    add_nonneg (mul_nonneg ha (by linarith)) (mul_nonneg hf (by linarith))
  have hterm : ∀ i, a * (Tq i * (1 + es i * Ns) + c i) +
        f * (Tq i * (1 + es i * Ns + Ks + es i * Km) + (1 + Ks) * c i) ≤
      (a * (1 + e * Ns) + f * (1 + e * Ns + Ks + e * Km)) * Tq i + (a * cc + f * ((1 + Ks) * cc)) := fun i => by
    have h1 : es i * Ns ≤ e * Ns := mul_le_mul_of_nonneg_right (he i) hNs0
    have h2 : es i * Km ≤ e * Km := mul_le_mul_of_nonneg_right (he i) hKm0
    have h3 : Tq i * (1 + es i * Ns) ≤ Tq i * (1 + e * Ns) := mul_le_mul_of_nonneg_left (by linarith) (hT0 i)
    have h4 : Tq i * (1 + es i * Ns + Ks + es i * Km) ≤ Tq i * (1 + e * Ns + Ks + e * Km) :=
      mul_le_mul_of_nonneg_left (by linarith) (hT0 i)
    have h5 : (1 + Ks) * c i ≤ (1 + Ks) * cc := mul_le_mul_of_nonneg_left (hc i) (by linarith)
    have h6 := mul_le_mul_of_nonneg_left (add_le_add h3 (hc i)) ha
    have h7 := mul_le_mul_of_nonneg_left (add_le_add h4 h5) hf
    exact (add_le_add h6 h7).trans (le_of_eq (by ring))
  have hsplit : ∑ i : Fin n, ((a * (1 + e * Ns) + f * (1 + e * Ns + Ks + e * Km)) * Tq i +
        (a * cc + f * ((1 + Ks) * cc))) =
      (a * (1 + e * Ns) + f * (1 + e * Ns + Ks + e * Km)) * ∑ i, Tq i + n * (a * cc + f * ((1 + Ks) * cc)) := by
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul]
  have hmul := mul_le_mul_of_nonneg_left hT hα
  have hring : (a * (1 + e * Ns) + f * (1 + e * Ns + Ks + e * Km)) * T + n * (a * cc + f * ((1 + Ks) * cc)) =
      a * (T * (1 + e * Ns) + n * cc) + f * (T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * (n * cc)) := by
    ring
  exact ((Finset.sum_le_sum fun i _ => hterm i).trans (hsplit.le.trans (add_le_add hmul le_rfl))).trans hring.le

/-- **The period's step.** A period's sample space split at each of its `n` phases (`φ`), phase `i` with an event
`E i` and a nonnegative count of calls over its prover's tape and its own coins. Where phase `i`'s event has
probability at most `b_i` plus, over `2^256`, `a` times `T_i·(1 + e_i·N_s) + c_i` and `f` times
`T_i·(1 + e_i·N_s + K_s + e_i·K_m) + (1 + K_s)·c_i`, with `T_i` the phase's expected calls, and the period's calls
average at most `T` in total, some phase has its event with probability at most `Σ_i b_i` plus the same expression
at `T`, at `e ≥ e_i` and at `n·c`, `c ≥ c_i`. -/
theorem pr_monitoringPeriod_once_le {S : Type} [Fintype S] {n : ℕ} {Ω C : Fin n → Type} [∀ i, Fintype (Ω i)]
    [∀ i, Fintype (C i)] (φ : ∀ i, S ≃ Ω i × C i) (E : ∀ i, Ω i × C i → Prop) (calls : ∀ i, Ω i × C i → ℝ)
    (b es c : Fin n → ℝ) {a f T e cc Ns Ks Km : ℝ} (ha : 0 ≤ a) (hf : 0 ≤ f) (hcalls : ∀ i q, 0 ≤ calls i q)
    (he0 : 0 ≤ e) (hNs0 : 0 ≤ Ns) (hKs0 : 0 ≤ Ks) (hKm0 : 0 ≤ Km) (he : ∀ i, es i ≤ e) (hc : ∀ i, c i ≤ cc)
    (hT : Ex (fun σ : S => ∑ i, calls i (φ i σ)) ≤ T)
    (hph : ∀ i, Pr (E i) ≤ b i + (a * (Ex (calls i) * (1 + es i * Ns) + c i) +
      f * (Ex (calls i) * (1 + es i * Ns + Ks + es i * Km) + (1 + Ks) * c i)) / 2 ^ (256 : ℝ)) :
    Pr (fun σ : S => ∃ i, E i (φ i σ)) ≤
      ∑ i, b i + (a * (T * (1 + e * Ns) + n * cc) +
        f * (T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * (n * cc))) / 2 ^ (256 : ℝ) := by
  have hsum : ∑ i, Ex (calls i) ≤ T := (Ex_monitoringPeriod_sum φ calls).symm.le.trans hT
  have hh := sum_hash_le (fun i => Ex (calls i)) es c ha hf (fun i => Ex_nonneg (hcalls i)) he0 hNs0 hKs0 hKm0 he
    hc hsum
  have hp : (0 : ℝ) < 2 ^ (256 : ℝ) := by positivity
  have hsplit : ∑ i, (b i + (a * (Ex (calls i) * (1 + es i * Ns) + c i) +
        f * (Ex (calls i) * (1 + es i * Ns + Ks + es i * Km) + (1 + Ks) * c i)) / 2 ^ (256 : ℝ)) =
      ∑ i, b i + (∑ i, (a * (Ex (calls i) * (1 + es i * Ns) + c i) +
        f * (Ex (calls i) * (1 + es i * Ns + Ks + es i * Km) + (1 + Ks) * c i))) / 2 ^ (256 : ℝ) := by
    rw [Finset.sum_add_distrib, Finset.sum_div]
  exact (Pr_monitoringPeriod_le φ E).trans (((Finset.sum_le_sum fun i _ => hph i).trans hsplit.le).trans
    (add_le_add le_rfl (div_le_div_of_nonneg_right hh hp.le)))

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

namespace Compiled

/-- **The one-stage event over a monitoring period, the prover's calls charged once.** A period of `n` phases, phase
`i` at its statement `x i` in scope and its prover `P i`, whose tape and the phase's coins split the period's sample
space (`φ`). Each phase is under `eventX_le_cls_exp`'s premises, with the sample's three counts at most `N_s`, `K_s`
and `K_m` at every sample of every phase, the sessions' expected reruns at most `e` and the opening checks' calls at
most `c` at every phase. Where the period's phases call `H` at most `T` times on average in total, both parties'
calls counted, some phase's accepted run samples a proof block of its `FregX` with probability at most

`Σ_i (κ_s + √(k·γ_s·Pr[B_i])) + (2·(T·(1 + e·N_s) + n·c) + 2·(T·(1 + e·N_s + K_s + e·K_m) + (1 + K_s)·n·c))/2^256`. -/
theorem monitoringPeriod_eventX_le_once {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H)
    (hS : L.Sessions H) (hSK : L.SessionKS H) (hopen : L.OpenCallsLe H) {n : ℕ} {x : Fin n → Msg}
    {s : Fin n → AuditStatement} {Pt : Fin n → Partition} (hs : ∀ i, L.parse (x i) = some (s i))
    (hPt : ∀ i, L.partition (s i) = some (Pt i)) (hin : ∀ i, Sampling.InScope (Pt i) (s i)) {S : Type} [Fintype S]
    {Ω : Fin n → Type} [∀ i, Fintype (Ω i)] [∀ i, Nonempty (Ω i)]
    (φ : ∀ i, S ≃ Ω i × (Fin (L.V.k (x i)) → Coin)) (P : ∀ i, Prover (Ω i)) (needs : Fin n → ℕ → Finset ℕ)
    (hpos : ∀ i u v w, ∀ p ∈ L.reveal (s i) u v w, p.2.pos ∈ needs i u)
    (cls : ∀ i, Fin (L.nG (x i)) → Fin (L.nG (x i)))
    (hallX : ∀ i (a : Ω i × ((Fin (L.V.k (x i)) → Coin) × L.SlotEC (x i))) (g g' : Fin (L.nG (x i))),
      cls i g = cls i g' → (L.opensAt H (P i) (x i) a.1 a.2.1 a.2.2 g).isSome →
        (L.opensAt H (P i) (x i) a.1 a.2.1 a.2.2 g').isSome)
    {T e c Ns Ks Km : ℝ} (he0 : 0 ≤ e) (hNs0 : 0 ≤ Ns) (hKs0 : 0 ≤ Ks) (hKm0 : 0 ≤ Km)
    (hT : Ex (fun σ : S => ∑ i, (L.partiesCalls H (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)) ≤ T)
    (hNs : ∀ i (r : Fin (L.V.k (x i)) → Coin), ((L.firstSlots (x i) (L.checked (x i) r)).card : ℝ) ≤ Ns)
    (hKs : ∀ i (r : Fin (L.V.k (x i)) → Coin),
      (((L.sampleNeeds (x i) (needs i) (L.checked (x i) r)).image (cls i)).card : ℝ) ≤ Ks)
    (hKm : ∀ i (r : Fin (L.V.k (x i)) → Coin),
      (L.sessCountIn (x i) (needs i) (fun l => (L.sampleNeeds (x i) (needs i) l).image (cls i))
        (L.checked (x i) r) : ℝ) ≤ Km)
    (he : ∀ i, L.es (x i) ≤ e) (hc : ∀ i, (L.nPU (x i) : ℝ) * L.openCalls (x i) ≤ c)
    (hB : ∀ i, ExpectedCR H (BconfX H L (Ω i) (P i) (x i)))
    (hX : ∀ i (K : ℕ), ExpectedCR H (L.finderXCls H (P i) (x i) (s i) (needs i) (cls i) K)) :
    Pr (fun σ : S => ∃ i, (run H (P i) L.V (x i) (φ i σ).1 (φ i σ).2).out = .acc ∧
        ∃ u ∈ L.checked (x i) (φ i σ).2, u ∈ L.FregX H (P i) (x i)) ≤
      ∑ i, (L.κs (x i) + Real.sqrt (L.V.k (x i) * L.γs (x i) * PrB H L.V (P i) (x i))) +
        (2 * (T * (1 + e * Ns) + n * c) + 2 * (T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * (n * c))) /
          2 ^ (256 : ℝ) := by
  have h := pr_monitoringPeriod_once_le φ
    (fun i p => (run H (P i) L.V (x i) p.1 p.2).out = .acc ∧ ∃ u ∈ L.checked (x i) p.2, u ∈ L.FregX H (P i) (x i))
    (fun i q => (L.partiesCalls H (P i) (x i) q.1 q.2 : ℝ))
    (fun i => L.κs (x i) + Real.sqrt (L.V.k (x i) * L.γs (x i) * PrB H L.V (P i) (x i)))
    (fun i => L.es (x i)) (fun i => (L.nPU (x i) : ℝ) * L.openCalls (x i)) (a := 2) (f := 2) (by norm_num)
    (by norm_num) (fun _ _ => Nat.cast_nonneg _) he0 hNs0 hKs0 hKm0 he hc hT fun i =>
      eventX_le_cls_exp hbind hS hSK hopen (hs i) (hPt i) (hin i) (P i) (needs i) (hpos i) (cls i) (hallX i)
        (le_refl _) (hNs i) (hKs i) (hKm i) (hB i) (hX i)
  exact h

end Compiled

namespace CompiledTwo

/-- **The event at the one-wait extraction's traces over a monitoring period, the prover's calls charged once.** A
period of `n` phases, phase `i` at its statement `x i` in the two-stage scope and its prover `P i`, whose tape and
the phase's coins split the period's sample space (`φ`). Each phase is under `eventTwoX_oneWait_le_cls_exp`'s
premises, with the sample's three counts at most `N_s`, `K_s` and `K_m` at every sample of every phase, the one-wait
compiled verifier's expected reruns a session at most `e` and the opening checks' calls at most `c` at every phase.
Where the period's phases call `H` at most `T` times on average in total, both parties' calls counted, some phase's
accepted run samples a proof block of the one-wait compiled verifier's `FregTwoX` with probability at most

`Σ_i (3·κ_s + 2·√(k·γ_s·Pr[B_i]))`
`+ 2·(2·(T·(1 + e·N_s) + n·c) + 4·(T·(1 + e·N_s + K_s + e·K_m) + (1 + K_s)·n·c))/2^256`:

a phase's expression at the period's expected calls and at `n` times the opening checks' own, with the two terms
that count no call of the prover's once a phase. -/
theorem monitoringPeriod_oneWait_le_once {H : Msg → Msg} {L : CompiledTwo Coin A} (hF : A.OneWaitFinds H)
    (hA : A.KS H) (hbind : L.commit.Binds H) (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {n : ℕ}
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
    {T e c Ns Ks Km : ℝ} (he0 : 0 ≤ e) (hNs0 : 0 ≤ Ns) (hKs0 : 0 ≤ Ks) (hKm0 : 0 ≤ Km)
    (hT : Ex (fun σ : S => ∑ i, (L.toCompiled.partiesCalls H (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)) ≤ T)
    (hNs : ∀ i (r : Fin (L.V.k (x i)) → Coin),
      ((L.toCompiled.firstSlots (x i) (L.checked (x i) r)).card : ℝ) ≤ Ns)
    (hKs : ∀ i (r : Fin (L.V.k (x i)) → Coin),
      (((L.toCompiled.sampleNeeds (x i) (needs i) (L.checked (x i) r)).image (cls i)).card : ℝ) ≤ Ks)
    (hKm : ∀ i (r : Fin (L.V.k (x i)) → Coin),
      (L.toCompiled.sessCountIn (x i) (needs i)
        (fun l => (L.toCompiled.sampleNeeds (x i) (needs i) l).image (cls i)) (L.checked (x i) r) : ℝ) ≤ Km)
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
        2 * (2 * (T * (1 + e * Ns) + n * c) + 4 * (T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * (n * c))) /
          2 ^ (256 : ℝ) := by
  have key : ∀ b u v p : ℝ, b + 2 * (2 * u + 4 * v) / p = b + (4 * u + 8 * v) / p := by
    intro b u v p
    ring
  have h := pr_monitoringPeriod_once_le φ
    (fun i p => (run H (P i) L.V (x i) p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked (x i) p.2, u ∈ L.oneWait.FregTwoX H (P i) (x i) p.1 p.2)
    (fun i q => (L.toCompiled.partiesCalls H (P i) (x i) q.1 q.2 : ℝ))
    (fun i => 3 * L.κs (x i) + 2 * Real.sqrt (L.V.k (x i) * L.γs (x i) * PrB H L.V (P i) (x i)))
    (fun i => L.oneWait.toCompiled.es (x i))
    (fun i => (L.oneWait.toCompiled.nPU (x i) : ℝ) * L.oneWait.toCompiled.openCalls (x i))
    (a := 4) (f := 8) (by norm_num) (by norm_num) (fun _ _ => Nat.cast_nonneg _) he0 hNs0 hKs0 hKm0 he hc hT
    fun i => (eventTwoX_oneWait_le_cls_exp hF hA hbind hS hopen (hs i) (hPt i) (hd i) (hin i) (P i) (needs i)
      (hpos i) (cls i) (hallX i) (hallI i) (le_refl _) (hNs i) (hKs i) (hKm i) (hB i) (hX i) (hI i)).trans
        (le_of_eq (key _ _ _ _))
  exact h.trans (le_of_eq (key _ _ _ _).symm)

end CompiledTwo

end Claims.SamplingGeneric
