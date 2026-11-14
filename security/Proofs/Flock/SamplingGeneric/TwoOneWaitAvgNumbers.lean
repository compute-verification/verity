module

public import Proofs.Flock.SamplingGeneric.TwoOneWaitAvg
public import Proofs.Flock.SamplingGeneric.TwoOneWaitNumbers
import Proofs.Flock.SamplingGeneric.Fails
import Proofs.Flock.Claims.Prob

/-!
# The one-wait by-class first-hit bound at the prover's expected calls, at numbers

`CompiledTwo.eventTwoX_oneWait_le_cls_exp` bounds the event at the one-wait extraction's committed traces by
`3·κ_s + 2·√(k·γ_s·Pr[B])` and twice a part that counts calls of SHA-512 over `2^256`,

`(2·(T·(1 + e_s·N_s) + c) + 4·(T·(1 + e_s·N_s + K_s + e_s·K_m) + (1 + K_s)·c))/2^256`, `c = n_PU·openCalls`:

`Claims.Flock.Numbers.hashBound`, with `T` a run's expected calls, both parties', and `N_s`, `K_s` and `K_m` the
sample's three counts at every sample. This file bounds that part by a number where what it reads is within given
numbers, as `TwoOneWaitNumbers.lean` does for the bound at a budget on every run:

* `hashBound_oneWait_513_n29`: over the rationals, at sessions of one table (`e = 12343987`) with every call charged
  as `513`, at `N_s ≤ 2^29` and `K_s, K_m ≤ 2^38`, at most `2^-103`;
* `CompiledTwo.eventTwoX_oneWait_le_cls_exp_of_hashBound`: `eventTwoX_oneWait_le_cls_exp` with its right-hand side
  replaced by `3·κ_s + 2·√(k·γ_s·Pr[B]) + 2·b` at any rational numbers at which `hashBound` is at most `b`;
* `CompiledTwo.eventTwoX_oneWait_le_cls_exp_513`: the same at the instance's numerals, at most
  `3·κ_s + 2·√(k·γ_s·Pr[B]) + 2^-102`.

That each is within its number is a hypothesis: on a run's expected calls, on the accepting runs a wait collects, on
the three counts and on `n_PU·openCalls`. `κ_s`, `k·γ_s` and `Pr[B]` are left as the theorem reads them.
-/

@[expose] public section

namespace Claims.Flock.Numbers

/-- **At most `2^-103` at one wait a session and `2^29` sessions, with every call charged as `513`**: the sum at
`513·2^80` in place of a run's expected `2^80` calls and of its own opening checks', at `e = 1 + 12343986`, and
`2^38` in place of `K_s` and of `K_m`. -/
theorem hashBound_oneWait_513_n29 :
    hashBound (513 * 2 ^ 80) 12343987 (2 ^ 29) (2 ^ 38) (2 ^ 38) (513 * 2 ^ 80) ≤ 1 / 2 ^ 103 := by
  norm_num [hashBound, firstHitCalls]

end Claims.Flock.Numbers

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeTwo)
open Claims.Flock.Numbers (hashBound real_hash_le hashBound_oneWait_513_n29 two_mul_cast_two_pow_neg_103)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

namespace CompiledTwo

/-- **The event at the one-wait extraction's traces, by class of gates, at the prover's expected calls, at
numbers.** Under `eventTwoX_oneWait_le_cls_exp`'s premises, where a run's expected calls `T`, the one-wait compiled
verifier's expected reruns a session `e_s`, the sample's counts `N_s`, `K_s` and `K_m` and `n_PU·openCalls` are
within rational numbers at which `hashBound` is at most `b`, an accepted run samples a proof block of the one-wait
compiled verifier's `FregTwoX` with probability at most `3·κ_s + 2·√(k·γ_s·Pr[B]) + 2·b`. -/
theorem eventTwoX_oneWait_le_cls_exp_of_hashBound {H : Msg → Msg} {L : CompiledTwo Coin A}
    (hF : A.OneWaitFinds H) (hA : A.KS H) (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hopen : L.toCompiled.OpenCallsLe H) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Sampling.Design.Two} (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.oneWait.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.oneWait.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    (hallI : ∀ (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x)
      (g g' : Fin (L.nG x)), cls g = cls g' → (L.oneWait.iopTable H P x t.1 d.part t.2 ec g).isSome →
        (L.oneWait.iopTable H P x t.1 d.part t.2 ec g').isSome)
    {T Ns Ks Km : ℝ}
    (hT : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) => (L.toCompiled.partiesCalls H P x q.1 q.2 : ℝ)) ≤ T)
    (hNs : ∀ r : Fin (L.V.k x) → Coin, ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ) ≤ Ns)
    (hKs : ∀ r : Fin (L.V.k x) → Coin,
      (((L.toCompiled.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ) ≤ Ks)
    (hKm : ∀ r : Fin (L.V.k x) → Coin,
      (L.toCompiled.sessCountIn x needs (fun l => (L.toCompiled.sampleNeeds x needs l).image cls)
        (L.checked x r) : ℝ) ≤ Km)
    (hB : ExpectedCR H (Compiled.BconfX H L.oneWait.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR H (L.oneWait.toCompiled.finderXCls H P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.oneWait.finderICls H P x s d.part needs cls K))
    {T' e' Ns' Ks' Km' c' b : ℚ} (hTcap : T ≤ T') (he : L.oneWait.toCompiled.es x ≤ e')
    (hNscap : Ns ≤ Ns') (hKscap : Ks ≤ Ks') (hKmcap : Km ≤ Km')
    (hown : (L.oneWait.toCompiled.nPU x : ℝ) * L.oneWait.toCompiled.openCalls x ≤ c')
    (h : hashBound T' e' Ns' Ks' Km' c' ≤ b) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.oneWait.FregTwoX H P x p.1 p.2) ≤
      3 * L.κs x + 2 * Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + 2 * (b : ℝ) := by
  have hT0 : 0 ≤ T := le_trans (Ex_nonneg (by intro q; exact Nat.cast_nonneg _)) hT
  have hNs0 : 0 ≤ Ns := le_trans (Nat.cast_nonneg _) (hNs fun _ => default)
  have hKs0 : 0 ≤ Ks := le_trans (Nat.cast_nonneg _) (hKs fun _ => default)
  have hKm0 : 0 ≤ Km := le_trans (Nat.cast_nonneg _) (hKm fun _ => default)
  have he0 : 0 ≤ L.oneWait.toCompiled.es x := maxBelow_nonneg _ _
  have hc0 : (0 : ℝ) ≤ (L.oneWait.toCompiled.nPU x : ℝ) * L.oneWait.toCompiled.openCalls x :=
    mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  have h1 := eventTwoX_oneWait_le_cls_exp hF hA hbind hS hopen hs hPt hd hin P needs hpos cls hallX hallI hT hNs
    hKs hKm hB hX hI
  have h2 := real_hash_le hT0 he0 hNs0 hKs0 hKm0 hc0 hTcap he hNscap hKscap hKmcap hown h
  have key : ∀ a u p : ℝ, a + 2 * u / p = a + 2 * (u / p) := by
    intro a u p
    ring
  rw [key] at h1
  exact h1.trans (add_le_add (le_refl _) (mul_le_mul_of_nonneg_left h2 (by norm_num)))

/-- **The event at the one-wait extraction's traces, by class of gates, at the prover's expected calls, at sessions
of one table with every call charged as `513`.** Under `eventTwoX_oneWait_le_cls_exp`'s premises, where a run's
expected calls are `T ≤ 513·2^80`, every proof block's shape collects at most `12343986` accepting runs a wait, the
sample's counts are `N_s ≤ 2^29`, `K_s ≤ 2^38` and `K_m ≤ 2^38`, and the opening checks of the statement's proof
blocks call SHA-512 `n_PU·openCalls ≤ 513·2^80` times, an accepted run samples a proof block of the one-wait
compiled verifier's `FregTwoX` with probability at most

`3·κ_s + 2·√(k·γ_s·Pr[B]) + 2^-102`. -/
theorem eventTwoX_oneWait_le_cls_exp_513 {H : Msg → Msg} {L : CompiledTwo Coin A} (hF : A.OneWaitFinds H)
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
    {T Ns Ks Km : ℝ}
    (hT : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) => (L.toCompiled.partiesCalls H P x q.1 q.2 : ℝ)) ≤ T)
    (hNs : ∀ r : Fin (L.V.k x) → Coin, ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ) ≤ Ns)
    (hKs : ∀ r : Fin (L.V.k x) → Coin,
      (((L.toCompiled.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ) ≤ Ks)
    (hKm : ∀ r : Fin (L.V.k x) → Coin,
      (L.toCompiled.sessCountIn x needs (fun l => (L.toCompiled.sampleNeeds x needs l).image cls)
        (L.checked x r) : ℝ) ≤ Km)
    (hB : ExpectedCR H (Compiled.BconfX H L.oneWait.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR H (L.oneWait.toCompiled.finderXCls H P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.oneWait.finderICls H P x s d.part needs cls K))
    (hTcap : T ≤ 513 * 2 ^ 80) (hk : ∀ u < L.toCompiled.nPU x, A.k (L.toCompiled.sh x u) ≤ 12343986)
    (hNscap : Ns ≤ 2 ^ 29) (hKscap : Ks ≤ 2 ^ 38) (hKmcap : Km ≤ 2 ^ 38)
    (hown : L.toCompiled.nPU x * L.toCompiled.openCalls x ≤ 513 * 2 ^ 80) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.oneWait.FregTwoX H P x p.1 p.2) ≤
      3 * L.κs x + 2 * Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + 2 ^ (-(102 : ℝ)) := by
  have he : L.oneWait.toCompiled.es x ≤ 1 + ((12343986 : ℕ) : ℝ) := L.toCompiled.oneWait_es_le x hk
  have hown' : L.oneWait.toCompiled.nPU x * L.oneWait.toCompiled.openCalls x ≤ 513 * 2 ^ 80 := hown
  rw [← two_mul_cast_two_pow_neg_103]
  refine eventTwoX_oneWait_le_cls_exp_of_hashBound hF hA hbind hS hopen hs hPt hd hin P needs hpos cls hallX
    hallI hT hNs hKs hKm hB hX hI (T' := 513 * 2 ^ 80) (e' := 12343987) (Ns' := 2 ^ 29) (Ks' := 2 ^ 38)
    (Km' := 2 ^ 38) (c' := 513 * 2 ^ 80) ?_ ?_ ?_ ?_ ?_ ?_ hashBound_oneWait_513_n29
  · push_cast
    exact hTcap
  · refine he.trans (le_of_eq ?_)
    push_cast
    norm_num
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hNscap
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hKscap
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hKmcap
  · exact_mod_cast hown'

end CompiledTwo

end Claims.SamplingGeneric
