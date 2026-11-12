module

public import Proofs.Flock.SamplingGeneric.PhasesOnce
public import Proofs.Flock.SamplingGeneric.TwoFirstHitClassNumbers
import Proofs.Flock.Claims.Prob

/-!
# The first-hit bound over a monitoring period, at numbers

`CompiledTwo.phases_oneWait_le_once` bounds the event of some phase of a monitoring period by

`Σ_i (3·κ_s + 2·√(k·γ_s·Pr[B_i]))`
`+ 2·(2·(T·(1 + e·N_s) + n·c) + 4·(T·(1 + e·N_s + K_s + e·K_m) + (1 + K_s)·n·c))/2^256`.

Here the expression is at rational numbers: `phasesBound n b T e N_s K_s K_m c` is `n·b` plus twice `hashBound` at
`T` and at `n·c`, a period of at most `n` phases each paying at most `b` for the two terms that count no call of the
prover's.

* `real_phases_le`: the expression over the reals is at most a bound of `phasesBound` at numbers it is within.
* `sqrt_le_of_le_sq`, `fork_le_sq`: a phase's fork term is at most `q` where `k·γ_s` and `Pr[B]` are within rational
  numbers whose product is at most `q²`.
* `CompiledTwo.phases_oneWait_le_of_phasesBound`: **the period's bound at numbers**, under
  `phases_oneWait_le_once`'s premises, with `κ_s ≤ κ` and `k·γ_s·Pr[B_i] ≤ q²` at every phase.
* The figures: `phasesBound` at a period's expected `2^80` SHA-512 calls in total, sessions rerun `12343987` times in
  expectation, `N_s ≤ 2^26`, `K_s, K_m ≤ 2^38`, opening checks of `2^60` calls a phase, `κ_s ≤ 2^-202`, and the fork
  term of a sample of one session (`q = 49234135/2^176`) or of `4096` sessions (`q = 49234135/2^170`), at six
  numbers of phases. That `κ_s`, `k·γ_s` and `Pr[B]` are within those numbers at a deployment is each figure's
  premise, not proved here; `forkSq_one` and `forkSq_4096` say which numbers the two values of `q` cover.
-/

@[expose] public section

namespace Claims.Flock.Numbers

/-- **The period's expression at rational numbers**: at most `n` phases each paying at most `b`, and twice
`hashBound` at the period's expected calls `T` and at `n` times the opening checks' own calls `c`. -/
def phasesBound (n b T e Ns Ks Km c : ℚ) : ℚ :=
  n * b + 2 * hashBound T e Ns Ks Km (n * c)

/-- **The expression over the reals**, `2^256` a real power. -/
theorem phasesBound_cast (n b T e Ns Ks Km c : ℚ) :
    ((phasesBound n b T e Ns Ks Km c : ℚ) : ℝ) =
      (n : ℝ) * (b : ℝ) + 2 * (2 * ((T : ℝ) * (1 + (e : ℝ) * (Ns : ℝ)) + (n : ℝ) * (c : ℝ)) +
        4 * ((T : ℝ) * (1 + (e : ℝ) * (Ns : ℝ) + (Ks : ℝ) + (e : ℝ) * (Km : ℝ)) +
          (1 + (Ks : ℝ)) * ((n : ℝ) * (c : ℝ)))) / 2 ^ (256 : ℝ) := by
  rw [show (256 : ℝ) = ((256 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  simp only [phasesBound, hashBound, firstHitCalls]
  push_cast
  all_goals ring

/-- **The period's expression over the reals is at most a bound of it at numbers**: `n ≤ n'` phases whose terms
`β i` are each at most `b'`, at the numbers `T'`, `e'`, `N_s'`, `K_s'`, `K_m'` and `c'`. -/
theorem real_phases_le {n : ℕ} {β : Fin n → ℝ} {n' b' T' e' Ns' Ks' Km' c' B : ℚ} (hn : (n : ℝ) ≤ n')
    (hβ : ∀ i, β i ≤ b') (hb0 : 0 ≤ b') (hKs0 : 0 ≤ Ks') (hc0 : 0 ≤ c')
    (h : phasesBound n' b' T' e' Ns' Ks' Km' c' ≤ B) :
    ∑ i, β i + 2 * (2 * ((T' : ℝ) * (1 + (e' : ℝ) * (Ns' : ℝ)) + (n : ℝ) * (c' : ℝ)) +
        4 * ((T' : ℝ) * (1 + (e' : ℝ) * (Ns' : ℝ) + (Ks' : ℝ) + (e' : ℝ) * (Km' : ℝ)) +
          (1 + (Ks' : ℝ)) * ((n : ℝ) * (c' : ℝ)))) / 2 ^ (256 : ℝ) ≤ (B : ℝ) := by
  have hb0' : (0 : ℝ) ≤ (b' : ℝ) := by exact_mod_cast hb0
  have hKs0' : (0 : ℝ) ≤ (Ks' : ℝ) := by exact_mod_cast hKs0
  have hc0' : (0 : ℝ) ≤ (c' : ℝ) := by exact_mod_cast hc0
  have hp : (0 : ℝ) ≤ 2 ^ (256 : ℝ) := (Real.rpow_pos_of_pos (by norm_num) _).le
  have hsum : ∑ i, β i ≤ (n' : ℝ) * (b' : ℝ) :=
    calc ∑ i, β i ≤ ∑ _i : Fin n, (b' : ℝ) := Finset.sum_le_sum fun i _ => hβ i
      _ = (n : ℝ) * (b' : ℝ) := by simp
      _ ≤ (n' : ℝ) * (b' : ℝ) := mul_le_mul_of_nonneg_right hn hb0'
  have hnc : (n : ℝ) * (c' : ℝ) ≤ (n' : ℝ) * (c' : ℝ) := mul_le_mul_of_nonneg_right hn hc0'
  have hnc' : (1 + (Ks' : ℝ)) * ((n : ℝ) * (c' : ℝ)) ≤ (1 + (Ks' : ℝ)) * ((n' : ℝ) * (c' : ℝ)) :=
    mul_le_mul_of_nonneg_left hnc (by linarith)
  have hnum : 2 * (2 * ((T' : ℝ) * (1 + (e' : ℝ) * (Ns' : ℝ)) + (n : ℝ) * (c' : ℝ)) +
        4 * ((T' : ℝ) * (1 + (e' : ℝ) * (Ns' : ℝ) + (Ks' : ℝ) + (e' : ℝ) * (Km' : ℝ)) +
          (1 + (Ks' : ℝ)) * ((n : ℝ) * (c' : ℝ)))) ≤
      2 * (2 * ((T' : ℝ) * (1 + (e' : ℝ) * (Ns' : ℝ)) + (n' : ℝ) * (c' : ℝ)) +
        4 * ((T' : ℝ) * (1 + (e' : ℝ) * (Ns' : ℝ) + (Ks' : ℝ) + (e' : ℝ) * (Km' : ℝ)) +
          (1 + (Ks' : ℝ)) * ((n' : ℝ) * (c' : ℝ)))) := by
    linarith
  have hdiv := div_le_div_of_nonneg_right hnum hp
  have hB : ((phasesBound n' b' T' e' Ns' Ks' Km' c' : ℚ) : ℝ) ≤ (B : ℝ) := Rat.cast_le.2 h
  rw [phasesBound_cast] at hB
  exact (add_le_add hsum hdiv).trans hB

/-- A square root is at most `q` where its argument is at most `q²`. -/
theorem sqrt_le_of_le_sq {y q : ℝ} (hq : 0 ≤ q) (h : y ≤ q ^ 2) : Real.sqrt y ≤ q :=
  (Real.sqrt_le_left hq).2 h

/-- **A phase's fork term's argument at numbers**: `k·γ_s ≤ G` and `Pr[B] ≤ P` with `G·P ≤ q²` give
`k·γ_s·Pr[B] ≤ q²`. -/
theorem fork_le_sq {g p : ℝ} {G P q : ℚ} (hp0 : 0 ≤ p) (hG0 : 0 ≤ G) (hg : g ≤ G) (hp : p ≤ P)
    (h : G * P ≤ q ^ 2) : g * p ≤ (q : ℝ) ^ 2 := by
  have h1 : g * p ≤ (G : ℝ) * (P : ℝ) := mul_le_mul hg hp hp0 (by exact_mod_cast hG0)
  have h2 : ((G * P : ℚ) : ℝ) ≤ ((q ^ 2 : ℚ) : ℝ) := Rat.cast_le.2 h
  push_cast at h2
  exact h1.trans h2

/-! ## The figures

`T = 2^80` expected calls in the whole period, `e = 12343987`, `N_s = 2^26`, `K_s = K_m = 2^38`, `c = 2^60` a phase,
and a phase's own terms `3/2^202 + 2·q`. -/

/-- A sample of one session: `k·γ_s ≤ 1212·10^12` and `Pr[B] ≤ 2^-351` are within `q = 49234135/2^176`. -/
theorem forkSq_one : (1212 * 10 ^ 12 : ℚ) * (1 / 2 ^ 351) ≤ (49234135 / 2 ^ 176) ^ 2 := by
  rw [mul_one_div, div_pow, div_le_div_iff₀ (by positivity) (by positivity)]
  norm_num

/-- A sample of `4096` sessions: `k·γ_s ≤ 4096·1212·10^12` and `Pr[B] ≤ 2^-351` are within
`q = 49234135/2^170`. -/
theorem forkSq_4096 : (4096 * (1212 * 10 ^ 12) : ℚ) * (1 / 2 ^ 351) ≤ (49234135 / 2 ^ 170) ^ 2 := by
  rw [mul_one_div, div_pow, div_le_div_iff₀ (by positivity) (by positivity)]
  norm_num

/-- One session a sample, `59,100,000` phases: at most `2^-111`. -/
theorem phasesBound_one_59100000 :
    phasesBound 59100000 (3 * (1 / 2 ^ 202) + 2 * (49234135 / 2 ^ 176)) (2 ^ 80) 12343987 (2 ^ 26) (2 ^ 38)
      (2 ^ 38) (2 ^ 60) ≤ 1 / 2 ^ 111 := by
  norm_num [phasesBound, hashBound, firstHitCalls]

/-- One session a sample, `253,000,000` phases: at most `2^-111`. -/
theorem phasesBound_one_253000000 :
    phasesBound 253000000 (3 * (1 / 2 ^ 202) + 2 * (49234135 / 2 ^ 176)) (2 ^ 80) 12343987 (2 ^ 26) (2 ^ 38)
      (2 ^ 38) (2 ^ 60) ≤ 1 / 2 ^ 111 := by
  norm_num [phasesBound, hashBound, firstHitCalls]

/-- One session a sample, `3,080,000,000` phases: at most `2^-111`. -/
theorem phasesBound_one_3080000000 :
    phasesBound 3080000000 (3 * (1 / 2 ^ 202) + 2 * (49234135 / 2 ^ 176)) (2 ^ 80) 12343987 (2 ^ 26) (2 ^ 38)
      (2 ^ 38) (2 ^ 60) ≤ 1 / 2 ^ 111 := by
  norm_num [phasesBound, hashBound, firstHitCalls]

/-- One session a sample, `221,000,000,000` phases: at most `2^-110`. -/
theorem phasesBound_one_221000000000 :
    phasesBound 221000000000 (3 * (1 / 2 ^ 202) + 2 * (49234135 / 2 ^ 176)) (2 ^ 80) 12343987 (2 ^ 26) (2 ^ 38)
      (2 ^ 38) (2 ^ 60) ≤ 1 / 2 ^ 110 := by
  norm_num [phasesBound, hashBound, firstHitCalls]

/-- One session a sample, `949,000,000,000` phases: at most `2^-109`. -/
theorem phasesBound_one_949000000000 :
    phasesBound 949000000000 (3 * (1 / 2 ^ 202) + 2 * (49234135 / 2 ^ 176)) (2 ^ 80) 12343987 (2 ^ 26) (2 ^ 38)
      (2 ^ 38) (2 ^ 60) ≤ 1 / 2 ^ 109 := by
  norm_num [phasesBound, hashBound, firstHitCalls]

/-- One session a sample, `11,550,000,000,000` phases: at most `2^-105`. -/
theorem phasesBound_one_11550000000000 :
    phasesBound 11550000000000 (3 * (1 / 2 ^ 202) + 2 * (49234135 / 2 ^ 176)) (2 ^ 80) 12343987 (2 ^ 26) (2 ^ 38)
      (2 ^ 38) (2 ^ 60) ≤ 1 / 2 ^ 105 := by
  norm_num [phasesBound, hashBound, firstHitCalls]

/-- `4096` sessions a sample, `59,100,000` phases: at most `2^-111`. -/
theorem phasesBound_4096_59100000 :
    phasesBound 59100000 (3 * (1 / 2 ^ 202) + 2 * (49234135 / 2 ^ 170)) (2 ^ 80) 12343987 (2 ^ 26) (2 ^ 38)
      (2 ^ 38) (2 ^ 60) ≤ 1 / 2 ^ 111 := by
  norm_num [phasesBound, hashBound, firstHitCalls]

/-- `4096` sessions a sample, `253,000,000` phases: at most `2^-111`. -/
theorem phasesBound_4096_253000000 :
    phasesBound 253000000 (3 * (1 / 2 ^ 202) + 2 * (49234135 / 2 ^ 170)) (2 ^ 80) 12343987 (2 ^ 26) (2 ^ 38)
      (2 ^ 38) (2 ^ 60) ≤ 1 / 2 ^ 111 := by
  norm_num [phasesBound, hashBound, firstHitCalls]

/-- `4096` sessions a sample, `3,080,000,000` phases: at most `2^-110`. -/
theorem phasesBound_4096_3080000000 :
    phasesBound 3080000000 (3 * (1 / 2 ^ 202) + 2 * (49234135 / 2 ^ 170)) (2 ^ 80) 12343987 (2 ^ 26) (2 ^ 38)
      (2 ^ 38) (2 ^ 60) ≤ 1 / 2 ^ 110 := by
  norm_num [phasesBound, hashBound, firstHitCalls]

/-- `4096` sessions a sample, `221,000,000,000` phases: at most `2^-105`. -/
theorem phasesBound_4096_221000000000 :
    phasesBound 221000000000 (3 * (1 / 2 ^ 202) + 2 * (49234135 / 2 ^ 170)) (2 ^ 80) 12343987 (2 ^ 26) (2 ^ 38)
      (2 ^ 38) (2 ^ 60) ≤ 1 / 2 ^ 105 := by
  norm_num [phasesBound, hashBound, firstHitCalls]

/-- `4096` sessions a sample, `949,000,000,000` phases: at most `2^-103`. -/
theorem phasesBound_4096_949000000000 :
    phasesBound 949000000000 (3 * (1 / 2 ^ 202) + 2 * (49234135 / 2 ^ 170)) (2 ^ 80) 12343987 (2 ^ 26) (2 ^ 38)
      (2 ^ 38) (2 ^ 60) ≤ 1 / 2 ^ 103 := by
  norm_num [phasesBound, hashBound, firstHitCalls]

/-- `4096` sessions a sample, `11,550,000,000,000` phases: at most `2^-100`. -/
theorem phasesBound_4096_11550000000000 :
    phasesBound 11550000000000 (3 * (1 / 2 ^ 202) + 2 * (49234135 / 2 ^ 170)) (2 ^ 80) 12343987 (2 ^ 26) (2 ^ 38)
      (2 ^ 38) (2 ^ 60) ≤ 1 / 2 ^ 100 := by
  norm_num [phasesBound, hashBound, firstHitCalls]

end Claims.Flock.Numbers

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeTwo)
open Claims.Flock.Numbers (phasesBound real_phases_le sqrt_le_of_le_sq)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The event at the one-wait extraction's traces over a monitoring period, at numbers.** Under
`phases_oneWait_le_once`'s premises at rational numbers: a period of `n ≤ n'` phases that call `H` at most `T'`
times on average in total, sessions' expected reruns at most `e'`, the sample's three counts at most `N_s'`, `K_s'`
and `K_m'`, opening checks of at most `c'` calls a phase, and at every phase `κ_s ≤ κ'` and
`k·γ_s·Pr[B_i] ≤ q'²`. Where `phasesBound` at those numbers, with `3·κ' + 2·q'` a phase, is at most `B`, some
phase's accepted run samples a proof block of the one-wait compiled verifier's `FregTwoX` with probability at most
`B`. -/
theorem phases_oneWait_le_of_phasesBound {H : Msg → Msg} {L : CompiledTwo Coin A} (hF : A.OneWaitFinds H)
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
    {n' κ' q' T' e' Ns' Ks' Km' c' B : ℚ} (hκ0 : 0 ≤ κ') (hq0 : 0 ≤ q') (he0 : 0 ≤ e') (hNs0 : 0 ≤ Ns')
    (hKs0 : 0 ≤ Ks') (hKm0 : 0 ≤ Km') (hc0 : 0 ≤ c') (hn : (n : ℝ) ≤ n')
    (hT : Ex (fun σ : S => ∑ i, (L.toCompiled.partiesCalls H (P i) (x i) (φ i σ).1 (φ i σ).2 : ℝ)) ≤ (T' : ℝ))
    (hNs : ∀ i (r : Fin (L.V.k (x i)) → Coin),
      ((L.toCompiled.firstSlots (x i) (L.checked (x i) r)).card : ℝ) ≤ (Ns' : ℝ))
    (hKs : ∀ i (r : Fin (L.V.k (x i)) → Coin),
      (((L.toCompiled.sampleNeeds (x i) (needs i) (L.checked (x i) r)).image (cls i)).card : ℝ) ≤ (Ks' : ℝ))
    (hKm : ∀ i (r : Fin (L.V.k (x i)) → Coin),
      (L.toCompiled.sessCountIn (x i) (needs i)
        (fun l => (L.toCompiled.sampleNeeds (x i) (needs i) l).image (cls i)) (L.checked (x i) r) : ℝ) ≤
          (Km' : ℝ))
    (he : ∀ i, L.oneWait.toCompiled.es (x i) ≤ (e' : ℝ))
    (hc : ∀ i, (L.oneWait.toCompiled.nPU (x i) : ℝ) * L.oneWait.toCompiled.openCalls (x i) ≤ (c' : ℝ))
    (hκ : ∀ i, L.κs (x i) ≤ (κ' : ℝ))
    (hfork : ∀ i, (L.V.k (x i) : ℝ) * L.γs (x i) * PrB H L.V (P i) (x i) ≤ (q' : ℝ) ^ 2)
    (hB : ∀ i, ExpectedCR H (Compiled.BconfX H L.oneWait.toCompiled (Ω i) (P i) (x i)))
    (hX : ∀ i (K : ℕ),
      ExpectedCR H (L.oneWait.toCompiled.finderXCls H (P i) (x i) (s i) (needs i) (cls i) K))
    (hI : ∀ i (K : ℕ),
      ExpectedCR H (L.oneWait.finderICls H (P i) (x i) (s i) (d i).part (needs i) (cls i) K))
    (h : phasesBound n' (3 * κ' + 2 * q') T' e' Ns' Ks' Km' c' ≤ B) :
    Pr (fun σ : S => ∃ i, (run H (P i) L.V (x i) (φ i σ).1 (φ i σ).2).out = .acc ∧
        ∃ u ∈ L.checked (x i) (φ i σ).2, u ∈ L.oneWait.FregTwoX H (P i) (x i) (φ i σ).1 (φ i σ).2) ≤
      (B : ℝ) := by
  have he0' : (0 : ℝ) ≤ (e' : ℝ) := by exact_mod_cast he0
  have hNs0' : (0 : ℝ) ≤ (Ns' : ℝ) := by exact_mod_cast hNs0
  have hKs0' : (0 : ℝ) ≤ (Ks' : ℝ) := by exact_mod_cast hKs0
  have hKm0' : (0 : ℝ) ≤ (Km' : ℝ) := by exact_mod_cast hKm0
  have hq0' : (0 : ℝ) ≤ (q' : ℝ) := by exact_mod_cast hq0
  have hb0 : (0 : ℚ) ≤ 3 * κ' + 2 * q' := by linarith
  have hβ : ∀ i, 3 * L.κs (x i) + 2 * Real.sqrt (L.V.k (x i) * L.γs (x i) * PrB H L.V (P i) (x i)) ≤
      ((3 * κ' + 2 * q' : ℚ) : ℝ) := by
    intro i
    have h1 := sqrt_le_of_le_sq hq0' (hfork i)
    have h2 := hκ i
    push_cast
    linarith
  have h1 := phases_oneWait_le_once hF hA hbind hS hopen hs hPt hd hin φ P needs hpos cls hallX hallI he0' hNs0'
    hKs0' hKm0' hT hNs hKs hKm he hc hB hX hI
  exact h1.trans (real_phases_le
    (β := fun i => 3 * L.κs (x i) + 2 * Real.sqrt (L.V.k (x i) * L.γs (x i) * PrB H L.V (P i) (x i)))
    hn hβ hb0 hKs0 hc0 h)

end CompiledTwo

end Claims.SamplingGeneric
