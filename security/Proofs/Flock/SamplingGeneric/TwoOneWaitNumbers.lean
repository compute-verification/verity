module

public import Proofs.Flock.SamplingGeneric.TwoOneWaitAt
public import Proofs.Flock.SamplingGeneric.TwoFirstHitClassNumbers
import Proofs.Flock.SamplingGeneric.Deployed
import Proofs.Flock.SamplingGeneric.Fails
import Proofs.Flock.Claims.Prob

/-!
# The one-wait by-class first-hit bound at a budget a run, at numbers

`CompiledTwo.eventTwoX_oneWait_le_cls_at_budget` bounds the event at the one-wait extraction's committed traces by
`3·κ_s + 2·√(k·γ_s·Pr[B])` and twice a part that counts calls of SHA-512 over `2^256`,

`(2·(T·(1 + e_s·N_s) + c) + 4·(T·(1 + e_s·N_s + K_s + e_s·K_m) + (1 + K_s)·c))/2^256`, `c = n_PU·openCalls`:

`Claims.Flock.Numbers.hashBound`, the sum `εclsAt` reads, with `e_s` the one-wait compiled verifier's: a session's
extraction reruns `1 + k` times in expectation where the proof system's reruns `1 + m′·k` times. This file bounds
that part by a number where what it reads is within given numbers:

* `Compiled.oneWait_es_le`: where every proof block's shape collects at most `k` accepting runs a wait, the one-wait
  compiled verifier's `e_s` is at most `1 + k`;
* `CompiledTwo.εclsOneWait_le_of_hashBound`: at a budget, an `e_s`, an `N_s`, a `K_s`, a `K_m` and an
  `n_PU·openCalls` within rational numbers at which `hashBound` is at most `b`, the right-hand side `εclsOneWait` is
  at most `3·κ_s + 2·√(k·γ_s·Pr[B]) + 2·b`;
* the instances, each at sessions of one table (`k ≤ 12343986`, so `e_s ≤ 12343987`) with every call charged as
  `513`: `hashBound_oneWait_513` at `N_s ≤ 2^26` and `K_s, K_m ≤ 2^38`, at most `2^-103`, and
  `εclsOneWait_le_513`, the right-hand side there at most `3·κ_s + 2·√(k·γ_s·Pr[B]) + 2^-102`; and over the
  rationals `hashBound_oneWait_513_k40`, at `N_s ≤ 2^30` and `K_s, K_m ≤ 2^40` at most `2^-101`;
* `CompiledTwo.eventTwoX_oneWait_le_cls_of_hashBound`: `eventTwoX_oneWait_le_cls_at_budget` with its right-hand
  side replaced by `3·κ_s + 2·√(k·γ_s·Pr[B]) + 2·b` at any such numbers, and
  `CompiledTwo.eventTwoX_oneWait_le_cls_513`, the same at the first instance's numerals: at most
  `3·κ_s + 2·√(k·γ_s·Pr[B]) + 2^-102`.

That each is within its number is a hypothesis, as in `εclsAt_le_of_hashBound`: on the budget `T`, on the accepting
runs a wait collects, on the three counts and on `n_PU·openCalls`. A call is one of SHA-512 on a whole message;
`513·2^80` in place of `2^80` is the same expression at numbers `513` times larger, a call charged as the `513`
compressions of a row of `2^16` bytes.
-/

@[expose] public section

namespace Claims.Flock.Numbers

/-! ## The sum at sessions of one table, one wait a session -/

/-- **At most `2^-103` at one wait a session, with every call charged as `513`**: the sum at `513·2^80` in place of a
run's `2^80` calls and of its own opening checks', at `e = 1 + 12343986`, `2^26` sessions and `2^38` in place of
`K_s` and of `K_m`. -/
theorem hashBound_oneWait_513 :
    hashBound (513 * 2 ^ 80) 12343987 (2 ^ 26) (2 ^ 38) (2 ^ 38) (513 * 2 ^ 80) ≤ 1 / 2 ^ 103 := by
  norm_num [hashBound, firstHitCalls]

/-- **At most `2^-101` at one wait a session, at `2^30` sessions and `2^40`** in place of `K_s` and of `K_m`, with
every call charged as `513`. -/
theorem hashBound_oneWait_513_k40 :
    hashBound (513 * 2 ^ 80) 12343987 (2 ^ 30) (2 ^ 40) (2 ^ 40) (513 * 2 ^ 80) ≤ 1 / 2 ^ 101 := by
  norm_num [hashBound, firstHitCalls]

/-- Twice `1/2^103` over the rationals is `2^-102`, a real power. -/
theorem two_mul_cast_two_pow_neg_103 : 2 * ((1 / 2 ^ 103 : ℚ) : ℝ) = 2 ^ (-(102 : ℝ)) := by
  rw [Real.rpow_neg (by norm_num), show (102 : ℝ) = ((102 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  push_cast
  all_goals ring

end Claims.Flock.Numbers

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeTwo)
open Claims.Flock.Numbers (hashBound real_hash_le hashBound_oneWait_513 two_mul_cast_two_pow_neg_103)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **The sessions' expected reruns at one wait are at most `1 + k`**, where every proof block's shape collects at
most `k` accepting runs a wait. -/
theorem Compiled.oneWait_es_le (L : Compiled Coin A) (x : Msg) {k : ℕ}
    (hk : ∀ u < L.nPU x, A.k (L.sh x u) ≤ k) : L.oneWait.es x ≤ 1 + (k : ℝ) := by
  show maxBelow (L.nPU x) (fun u => 1 + ((1 : ℕ) : ℝ) * (A.k (L.sh x u) : ℝ)) ≤ 1 + (k : ℝ)
  refine maxBelow_le (by positivity) fun u hu => ?_
  have h : (A.k (L.sh x u) : ℝ) ≤ k := by exact_mod_cast hk u hu
  simp only [Nat.cast_one, one_mul]
  linarith

namespace CompiledTwo

/-- **The one-wait by-class right-hand side at numbers.** Where the budget `T`, the one-wait compiled verifier's
expected reruns a session `e_s`, `N_s`, `K_s`, `K_m` and `n_PU·openCalls` are within rational numbers at which
`hashBound` is at most `b`, `εclsOneWait` is at most `3·κ_s + 2·√(k·γ_s·Pr[B]) + 2·b`: its two costs over `2^256`
are bounded together, and it reads them twice. -/
theorem εclsOneWait_le_of_hashBound (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {T : ℕ} {Ns Ks Km : ℝ} {T' e' Ns' Ks' Km' c' b : ℚ} (hNs0 : 0 ≤ Ns)
    (hKs0 : 0 ≤ Ks) (hKm0 : 0 ≤ Km) (hT : (T : ℝ) ≤ T') (he : L.oneWait.toCompiled.es x ≤ e') (hNs : Ns ≤ Ns')
    (hKs : Ks ≤ Ks') (hKm : Km ≤ Km')
    (hown : (L.oneWait.toCompiled.nPU x : ℝ) * L.oneWait.toCompiled.openCalls x ≤ c')
    (h : hashBound T' e' Ns' Ks' Km' c' ≤ b) :
    L.εclsOneWait H x P T Ns Ks Km ≤
      3 * L.κs x + 2 * Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + 2 * (b : ℝ) := by
  have he0 : 0 ≤ L.oneWait.toCompiled.es x := maxBelow_nonneg _ _
  have hc0 : (0 : ℝ) ≤ (L.oneWait.toCompiled.nPU x : ℝ) * L.oneWait.toCompiled.openCalls x :=
    mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  unfold εclsOneWait Compiled.runCost Compiled.firstHitCost
  rw [mul_div_assoc]
  exact add_le_add (le_refl _) (mul_le_mul_of_nonneg_left
    (real_hash_le (Nat.cast_nonneg T) he0 hNs0 hKs0 hKm0 hc0 hT he hNs hKs hKm hown h) (by norm_num))

/-- **The one-wait by-class right-hand side at sessions of one table, with every call charged as `513`**: where
`T ≤ 513·2^80`, every proof block's shape collects at most `12343986` accepting runs a wait, `N_s ≤ 2^26`,
`K_s ≤ 2^38`, `K_m ≤ 2^38` and `n_PU·openCalls ≤ 513·2^80`, `εclsOneWait` is at most
`3·κ_s + 2·√(k·γ_s·Pr[B]) + 2^-102`. -/
theorem εclsOneWait_le_513 (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {T : ℕ} {Ns Ks Km : ℝ} (hNs0 : 0 ≤ Ns) (hKs0 : 0 ≤ Ks) (hKm0 : 0 ≤ Km)
    (hT : T ≤ 513 * 2 ^ 80) (hk : ∀ u < L.toCompiled.nPU x, A.k (L.toCompiled.sh x u) ≤ 12343986)
    (hNs : Ns ≤ 2 ^ 26) (hKs : Ks ≤ 2 ^ 38) (hKm : Km ≤ 2 ^ 38)
    (hown : L.toCompiled.nPU x * L.toCompiled.openCalls x ≤ 513 * 2 ^ 80) :
    L.εclsOneWait H x P T Ns Ks Km ≤
      3 * L.κs x + 2 * Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + 2 ^ (-(102 : ℝ)) := by
  have he : L.oneWait.toCompiled.es x ≤ 1 + ((12343986 : ℕ) : ℝ) := L.toCompiled.oneWait_es_le x hk
  have hown' : L.oneWait.toCompiled.nPU x * L.oneWait.toCompiled.openCalls x ≤ 513 * 2 ^ 80 := hown
  rw [← two_mul_cast_two_pow_neg_103]
  refine εclsOneWait_le_of_hashBound H L x P (T' := 513 * 2 ^ 80) (e' := 12343987) (Ns' := 2 ^ 26) (Ks' := 2 ^ 38)
    (Km' := 2 ^ 38) (c' := 513 * 2 ^ 80) hNs0 hKs0 hKm0 ?_ ?_ ?_ ?_ ?_ ?_ hashBound_oneWait_513
  · exact_mod_cast hT
  · refine he.trans (le_of_eq ?_)
    push_cast
    norm_num
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hNs
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hKs
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hKm
  · exact_mod_cast hown'

/-- **The event at the one-wait extraction's traces, by class of gates, at numbers.** Under
`eventTwoX_oneWait_le_cls_at_budget`'s hypotheses, where the budget `T`, the one-wait compiled verifier's expected
reruns a session `e_s`, the bounds `N_s`, `K_s` and `K_m` on the sample's counts and `n_PU·openCalls` are within
rational numbers at which `hashBound` is at most `b`, an accepted run samples a proof block of the one-wait compiled
verifier's `FregTwoX` with probability at most `3·κ_s + 2·√(k·γ_s·Pr[B]) + 2·b`. -/
theorem eventTwoX_oneWait_le_cls_of_hashBound {H : Msg → Msg} {L : CompiledTwo Coin A} (hF : A.OneWaitFinds H)
    (hA : A.KS H) (hbind : L.commit.Binds H) (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.oneWait.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.oneWait.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    (hallI : ∀ (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x)
      (g g' : Fin (L.nG x)), cls g = cls g' → (L.oneWait.iopTable H P x t.1 d.part t.2 ec g).isSome →
        (L.oneWait.iopTable H P x t.1 d.part t.2 ec g').isSome)
    (T : ℕ) (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      (((L.toCompiled.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin =>
      (L.toCompiled.sessCountIn x needs (fun l => (L.toCompiled.sampleNeeds x needs l).image cls)
        (L.checked x r) : ℝ)) ≤ Km)
    (hB : ExpectedCR H (Compiled.BconfX H L.oneWait.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR H (L.oneWait.toCompiled.finderXCls H P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.oneWait.finderICls H P x s d.part needs cls K))
    {T' e' Ns' Ks' Km' c' b : ℚ} (hTcap : (T : ℝ) ≤ T') (he : L.oneWait.toCompiled.es x ≤ e')
    (hNscap : Ns ≤ Ns') (hKscap : Ks ≤ Ks') (hKmcap : Km ≤ Km')
    (hown : (L.oneWait.toCompiled.nPU x : ℝ) * L.oneWait.toCompiled.openCalls x ≤ c')
    (h : hashBound T' e' Ns' Ks' Km' c' ≤ b) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.oneWait.FregTwoX H P x p.1 p.2) ≤
      3 * L.κs x + 2 * Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + 2 * (b : ℝ) := by
  have hNs0 : 0 ≤ Ns := le_trans (Ex_nonneg (by intro r; exact Nat.cast_nonneg _)) hNs
  have hKs0 : 0 ≤ Ks := le_trans (Ex_nonneg (by intro r; exact Nat.cast_nonneg _)) hKs
  have hKm0 : 0 ≤ Km := le_trans (Ex_nonneg (by intro r; exact Nat.cast_nonneg _)) hKm
  exact (eventTwoX_oneWait_le_cls_at_budget hF hA hbind hS hopen hs hPt hd hin P needs hpos cls hallX hallI T hT
    hNs hKs hKm hB hX hI).trans
      (εclsOneWait_le_of_hashBound H L x P hNs0 hKs0 hKm0 hTcap he hNscap hKscap hKmcap hown h)

/-- **The event at the one-wait extraction's traces, by class of gates, at sessions of one table with every call
charged as `513`.** Under `eventTwoX_oneWait_le_cls_at_budget`'s hypotheses, where the budget is `T ≤ 513·2^80`,
every proof block's shape collects at most `12343986` accepting runs a wait, the bounds on the sample's counts are
`N_s ≤ 2^26`, `K_s ≤ 2^38` and `K_m ≤ 2^38`, and the opening checks of the statement's proof blocks call SHA-512
`n_PU·openCalls ≤ 513·2^80` times, an accepted run samples a proof block of the one-wait compiled verifier's
`FregTwoX` with probability at most

`3·κ_s + 2·√(k·γ_s·Pr[B]) + 2^-102`:

twice the conflict finder's cost and the first-hit finders', summed, is at most `2^-102` of `2^256`. -/
theorem eventTwoX_oneWait_le_cls_513 {H : Msg → Msg} {L : CompiledTwo Coin A} (hF : A.OneWaitFinds H)
    (hA : A.KS H) (hbind : L.commit.Binds H) (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.oneWait.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.oneWait.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    (hallI : ∀ (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x)
      (g g' : Fin (L.nG x)), cls g = cls g' → (L.oneWait.iopTable H P x t.1 d.part t.2 ec g).isSome →
        (L.oneWait.iopTable H P x t.1 d.part t.2 ec g').isSome)
    (T : ℕ) (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      (((L.toCompiled.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin =>
      (L.toCompiled.sessCountIn x needs (fun l => (L.toCompiled.sampleNeeds x needs l).image cls)
        (L.checked x r) : ℝ)) ≤ Km)
    (hB : ExpectedCR H (Compiled.BconfX H L.oneWait.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR H (L.oneWait.toCompiled.finderXCls H P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.oneWait.finderICls H P x s d.part needs cls K))
    (hTcap : T ≤ 513 * 2 ^ 80) (hk : ∀ u < L.toCompiled.nPU x, A.k (L.toCompiled.sh x u) ≤ 12343986)
    (hNscap : Ns ≤ 2 ^ 26) (hKscap : Ks ≤ 2 ^ 38) (hKmcap : Km ≤ 2 ^ 38)
    (hown : L.toCompiled.nPU x * L.toCompiled.openCalls x ≤ 513 * 2 ^ 80) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.oneWait.FregTwoX H P x p.1 p.2) ≤
      3 * L.κs x + 2 * Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + 2 ^ (-(102 : ℝ)) := by
  have hNs0 : 0 ≤ Ns := le_trans (Ex_nonneg (by intro r; exact Nat.cast_nonneg _)) hNs
  have hKs0 : 0 ≤ Ks := le_trans (Ex_nonneg (by intro r; exact Nat.cast_nonneg _)) hKs
  have hKm0 : 0 ≤ Km := le_trans (Ex_nonneg (by intro r; exact Nat.cast_nonneg _)) hKm
  exact (eventTwoX_oneWait_le_cls_at_budget hF hA hbind hS hopen hs hPt hd hin P needs hpos cls hallX hallI T hT
    hNs hKs hKm hB hX hI).trans (εclsOneWait_le_513 H L x P hNs0 hKs0 hKm0 hTcap hk hNscap hKscap hKmcap hown)

end CompiledTwo

end Claims.SamplingGeneric
