import Definitions.Accounting.Responder
import Definitions.Accounting.Window
import Definitions.Pous.P2.Cost
import Definitions.Pous.Model.SampledAudit
import Definitions.Pous.Assumptions.P2

/-!
# B2: a location as PoUS's responder (trusted)

Obligation B2 of the actual-circuit model, on PoUS's ring model, for a location that does no arithmetic modulo `p`
during a window (so no ring answer is in flight at a challenge, the row's `f = 0`; where it does, `B2Window` states the
oracle-location form's core). A strategy `S` maps PoUS's outcome (the primitive `ω`, the encoder's coins, the `k`
challenges) to a run, causal from every tick. PoUS sends each challenge when the last answer arrives, so the challenge
ticks are read off the run: challenge `j` reaches the location `X` at tick `t x j`, the ticks increasing in each run,
and `t x (j + 1)` is a function of what happened before it (`ArrivalRule`). Its answer is read off what `X` does during
window `j`. What `X` does during a window includes its state at the window's end, so its state at each challenge tick is
a function of its state at the previous one and of what reached it in between.

* `LengthCode`, the coding lemma: strings of at most `M` bits inject into `M + 1` bits (`lengthCode`). With `Recall`'s
  `length (content X t) ≤ mem X t`, a state of `mem ≤ M` bits fits in an `(M + 1)`-bit code.
* `TimelyTicks`: on a run whose windows last at most `span` ticks, challenge `j`'s tick is at most `t 0 + j·span`.
  With the first challenge at a tick `t₀` the verifier fixes, every challenge of a `k`-challenge audit that passes is
  in `I = [t₀, t₀ + k·span)`, and the last one at most `t₀ + (k − 1)·span` (reads by determination's window).
* `B2`: if `X`'s state at the first challenge, that tick and a *help* (that takes at most `2^H` values) depend on the
  outcome only through the primitive and the encoding (`SetupFixed`), what reaches `X` during window `j` only through
  the help, challenge `j` and its tick (`IngressFixed`), the next tick only through those and what `X` does
  (`ArrivalRule`), and challenge `j`'s tick takes at most `2^τ` values on the runs that pass (`TicksIn`), then a
  preprocessing and a sequential responder with `N ≥ M + 1 + τ + H` bits of state answer every challenge exactly as `X`
  does, at every outcome whose windows last at most `span` and where `mem_X(t_j) ≤ M` before each challenge. Its state
  before challenge `j` is the `(M + 1)`-bit code of `content X t_j`, a `τ`-bit code of `t_j` and an `H`-bit code of
  the help. It makes no query (`RingBounded 0 0 0`): with `ω` among the coins and causality at `X`, what `X` does in a
  window is a function of its state's content and of what reaches it, so nothing gives `X` the ring but its state.
* `B2Audit`: so when PoUS's sequential audit accepts every `N`-bit responder that makes no query with probability at
  most `δ + ε` (`AuditSecureOn … N 0 0 0`), the run passes with `mem_X(t_j) + 1 + τ + H ≤ N` before every challenge
  with probability at most `δ + ε`: on an accept, some `j` has `mem_X(t_j) + 1 + τ + H > N`, except with PoUS's error.
* `B2P2Ring`: the same at P2's deployed point under `P2RingAuditSecure 0 0 0`, which `P2RingAuditSecure D Q L` implies
  for every `(D, Q, L)`, with the first challenge at a fixed tick and `105·span ≤ 2^τ`: some `j` has
  `mem_X(t_j) + 1 + τ + H > ⌊ρ·|C|⌋`, except with `δ + 2^-129 + 2^-130`.
* `B2Determines`, the hookup to lemma M (`Accounting.Guarantees.Memory.AddsOn`): for a family of strategies, one for
  each value `z` the location keeps, each meeting `B2`'s hypotheses with every window at most `span` and
  `mem_X(t_j) ≤ M` in every run, and `z` read off `X`'s state's content at every challenge, B2's responders are lemma
  M's adversaries: an `(M + 1) + τ + H`-bit state, from which a map reads `z` before every challenge on every outcome.
  The conjunct `∀ z x j, j < k → π j (seqState …) = z` is lemma M's `Determines sch k W π A₁ (fun z => (A₂ z).step)`
  unfolded.
* `B2Adds`, lemma M through `B2Determines`: under `AuditSecureOn … N 0 0 0 c δ ε`, a location that keeps a uniform
  `q`-bit `z` passes while `q + N ≥ M + 1 + τ + H + k` with probability at most `δ + ε + c·2^-k`; `B2AddsP2Ring`, the
  same at P2's deployed point under `P2RingAuditSecure 0 0 0`, with `δ + 2^-129 + 2^-130 + 105·2^-k`.
* `B2Window`, the core of the oracle-location form (B2's row): where the location does use the ring, at oracle locations
  of its own (`RingResponder`), a window whose multiplications take at least `ℓ > 0` ticks to answer and which spans at
  most `D·ℓ` ticks is a step of an `(N, D, Q, L)` sequential responder, which asks the additions, `one` and the keys at
  once and holds each multiplication until the earliest held one could be due. With `ℓ = ⌈(Δ + RTT)/D_p⌉` (the latency
  form of sequential speed) and one challenge every `Δ + RTT` ticks, `D = D_p`. `B2WindowAudit`: so under
  `AuditSecureOn … N D Q L`, a location that starts in an `N`-bit state and answers with such windows, each within
  `(Q, L)`, answers every challenge with probability at most `δ + ε`; `B2WindowP2Ring`, the same at P2's deployed point
  under `P2RingAuditSecure D Q L`, with `δ + 2^-129 + 2^-130`, paced: one challenge every `T` ticks and every
  multiplication at least `ℓ` ticks, with `T ≤ D·ℓ`. What is not stated yet: the windows read off a strategy's runs
  (oracle locations on `Actual.Run`, `Wd` from `CausalFrom` with the state's content, the help and the answers in flight
  coded into `N` bits).

A fixed schedule is the case `t x = t` for every `x`: `ArrivalRule`, `SetupFixed`'s tick and `TicksIn` at `τ = 0` hold,
`IngressFixed` is the fixed window's ingress, and every run's windows last at most `t k`, so these statements give the
fixed-schedule statements at `τ = 0`. The help is whatever the setup fixes that, with the challenge, determines what
reaches `X`: under isolation nothing, and when the rest of the prover learns nothing that depends on the challenges or
their ticks (the model's `e = 0` across the audit), what the rest sends `X` during the whole audit, with `H` the log of
the number of values it takes. The proofs are in `Security/Proofs/Accounting/Responder.lean`, `B2Window`'s in
`Security/Proofs/Accounting/Responder/Latency.lean`.
-/

namespace Accounting.Guarantees

open scoped ENNReal

/-- **The coding lemma.** The `(M + 1)`-bit length code is injective on the strings of at most `M` bits. -/
def LengthCode : Prop := ∀ M : ℕ, Set.InjOn (lengthCode M) {l : List Bool | l.length ≤ M}

/-- **The ticks of a run on time.** On a run whose `k` windows each last at most `span` ticks, challenge `j`'s tick is
at most `span·j` past the first, for every `j ≤ k`. -/
def TimelyTicks : Prop := ∀ (t : ℕ → ℕ) (span k : ℕ), Timely t span k → ∀ j ≤ k, t j ≤ t 0 + j * span

/-- **B2.** For a ring-model scheme, `k` challenges at `W`, a strategy `S` causal from every tick over PoUS's outcomes,
a location `X`, challenge ticks `t` read off each run, a window bound `span`, answers read off what `X` does in each
window (`ans`) and a help `help` with at most `2^H` values: if the ticks increase in each run, the setup fixes `X`'s
state at the first challenge, that tick and the help, the help, challenge `j` and its tick fix what reaches `X` during
window `j`, and with what `X` does the next tick, and each challenge's tick takes at most `2^τ` values on the runs on
time, then some `N`-bit preprocessing and sequential responder that makes no query (`N ≥ M + 1 + τ + H`) answer, at
every outcome on time where `X`'s state has at most `M` bits at each challenge tick, exactly what `X` answers. -/
def B2 : Prop :=
  ∀ {L : Type*} [LinearOrder L] {B w p : ℕ} [NeZero p] (sch : Pous.Scheme (Pous.ringRO B w p)) (k : ℕ)
    (W : Pous.Bits sch.n) (S : Pous.Outcome sch k → Actual.Run L) (X : Finset L) (t : Pous.Outcome sch k → ℕ → ℕ)
    (span : ℕ) (ans : ℕ → Actual.Acts L → Pous.Bits sch.ℓ) {α : Type*} (help : Pous.Outcome sch k → α)
    (M τ H N : ℕ),
    CausalFrom S → (∀ x (j : Fin k), t x j < t x (j + 1)) → SetupFixed W S X t help → IngressFixed S X t help →
    ArrivalRule S X t help → TicksIn t span τ → (Set.range help).encard ≤ ((2 ^ H : ℕ) : ℕ∞) →
    M + 1 + τ + H ≤ N →
    ∃ (A₁ : Pous.Preprocessor sch N) (A₂ : Pous.SeqResponderOn sch N 0 0 0),
      ∀ x, Timely (t x) span k → (∀ j : Fin k, (S x).mem X (t x j) ≤ M) →
        Pous.seqAnswers ((Pous.ringRO B w p).answer x.1) (A₂.step W (sch.enc W x.1 x.2.1).1) 0 (List.ofFn x.2.2)
            (A₁ W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1) =
          List.ofFn fun j : Fin k => ans j ((S x).acts X (window (t x) j))

/-- **B2, against PoUS's guarantee.** When PoUS's sequential audit with `k` challenges accepts every `N`-bit responder
that makes no query with probability at most `δ + ε`, a strategy meeting `B2`'s hypotheses passes the audit (every
window at most `span`, every answer right) with `mem_X(t_j) + 1 + τ + H ≤ N` at every challenge with probability at
most `δ + ε`. -/
def B2Audit : Prop :=
  ∀ {L : Type*} [LinearOrder L] {B w p : ℕ} [NeZero p] (sch : Pous.Scheme (Pous.ringRO B w p)) (N k : ℕ)
    (δ ε : ℝ≥0∞), Pous.AuditSecureOn sch .sequential N 0 0 0 k δ ε →
    ∀ (W : Pous.Bits sch.n) (S : Pous.Outcome sch k → Actual.Run L) (X : Finset L) (t : Pous.Outcome sch k → ℕ → ℕ)
      (span : ℕ) (ans : ℕ → Actual.Acts L → Pous.Bits sch.ℓ) {α : Type*} (help : Pous.Outcome sch k → α) (τ H : ℕ),
      CausalFrom S → (∀ x (j : Fin k), t x j < t x (j + 1)) → SetupFixed W S X t help → IngressFixed S X t help →
      ArrivalRule S X t help → TicksIn t span τ → (Set.range help).encard ≤ ((2 ^ H : ℕ) : ℕ∞) →
      Pous.pr (fun x => Passes sch W S X t span ans x ∧ ∀ j : Fin k, (S x).mem X (t x j) + 1 + τ + H ≤ N) ≤ δ + ε

/-- **B2 at P2's deployed point.** Under `P2RingAuditSecure 0 0 0` (P2 in the ring model, 2,054 payload bytes per block,
105 challenges, state bound `⌊ρ·|C|⌋`), a strategy meeting `B2`'s hypotheses, with the first challenge at a tick `t₀`
the verifier fixes and `105·span ≤ 2^τ` in place of `TicksIn`, passes with `mem_X(t_j) + 1 + τ + H ≤ ⌊ρ·|C|⌋` at
every challenge with probability at most `δ + 2^-129 + 2^-130`. -/
def B2P2Ring : Prop :=
  Pous.P2.Assumptions.P2RingAuditSecure 0 0 0 →
    ∀ {L : Type*} [LinearOrder L] (W : Pous.Bits (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).n)
      (S : Pous.Outcome (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) 105 → Actual.Run L) (X : Finset L)
      (t : Pous.Outcome (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) 105 → ℕ → ℕ) (t₀ span : ℕ)
      (ans : ℕ → Actual.Acts L → Pous.Bits (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).ℓ)
      {α : Type*} (help : Pous.Outcome (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) 105 → α) (τ H : ℕ),
      CausalFrom S → (∀ x (j : Fin 105), t x j < t x (j + 1)) → SetupFixed W S X t help →
      IngressFixed S X t help → ArrivalRule S X t help → (∀ x, t x 0 = t₀) → 105 * span ≤ 2 ^ τ →
      (Set.range help).encard ≤ ((2 ^ H : ℕ) : ℕ∞) →
      Pous.pr (fun x => Passes (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) W S X t span ans x ∧
          ∀ j : Fin 105, (S x).mem X (t x j) + 1 + τ + H ≤
            (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).stateBound Pous.Params.ρ) ≤
        ENNReal.ofReal Pous.Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130)

/-- **B2's responders are lemma M's adversaries.** For a family of strategies `S z`, one for each `q`-bit value `z` that
`X` keeps, each meeting `B2`'s hypotheses, with every window at most `span` and `mem_X(t_j) ≤ M` at every challenge in
every run and `z` read off `X`'s state's content at every challenge (`ζ j`): some `(M + 1 + τ + H)`-bit
preprocessings and responders that make no query, one for each `z`, answer as `X` does, and some `π` reads `z` from the
state each holds before every challenge `j < k` on every outcome (lemma M's `Determines`, unfolded). -/
def B2Determines : Prop :=
  ∀ {L : Type*} [LinearOrder L] {B w p : ℕ} [NeZero p] (sch : Pous.Scheme (Pous.ringRO B w p)) (k q : ℕ)
    (W : Pous.Bits sch.n) (S : Pous.Bits q → Pous.Outcome sch k → Actual.Run L) (X : Finset L)
    (t : Pous.Bits q → Pous.Outcome sch k → ℕ → ℕ) (span : ℕ) (ans : ℕ → Actual.Acts L → Pous.Bits sch.ℓ)
    {α : Type*} (help : Pous.Bits q → Pous.Outcome sch k → α) (ζ : ℕ → List Bool → Pous.Bits q) (M τ H : ℕ),
    (∀ z, CausalFrom (S z)) → (∀ z x (j : Fin k), t z x j < t z x (j + 1)) →
    (∀ z, SetupFixed W (S z) X (t z) (help z)) → (∀ z, IngressFixed (S z) X (t z) (help z)) →
    (∀ z, ArrivalRule (S z) X (t z) (help z)) → (∀ z, TicksIn (t z) span τ) →
    (∀ z, (Set.range (help z)).encard ≤ ((2 ^ H : ℕ) : ℕ∞)) →
    (∀ z x, Timely (t z x) span k) → (∀ z x (j : Fin k), (S z x).mem X (t z x j) ≤ M) →
    (∀ z x (j : Fin k), ζ j ((S z x).content X (t z x j)) = z) →
    ∃ (A₁ : Pous.Bits q → Pous.Preprocessor sch (M + 1 + τ + H))
      (A₂ : Pous.Bits q → Pous.SeqResponderOn sch (M + 1 + τ + H) 0 0 0)
      (π : ℕ → Pous.Bits (M + 1 + τ + H) → Pous.Bits q),
      (∀ (z : Pous.Bits q) (x : Pous.Outcome sch k) (j : ℕ), j < k →
        π j (Pous.Sampled.seqState ((Pous.ringRO B w p).answer x.1) ((A₂ z).step W (sch.enc W x.1 x.2.1).1) 0
          ((List.ofFn x.2.2).take j) (A₁ z W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1)) = z) ∧
      ∀ z x, Pous.seqAnswers ((Pous.ringRO B w p).answer x.1) ((A₂ z).step W (sch.enc W x.1 x.2.1).1) 0
          (List.ofFn x.2.2) (A₁ z W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1) =
        List.ofFn fun j : Fin k => ans j ((S z x).acts X (window (t z x) j))

/-- **B2 with lemma M: what a location keeps adds to what it holds.** With `z` uniform over `q` bits and a strategy
`S z` for each, each meeting `B2Determines`' hypotheses: when PoUS's sequential audit with `c` challenges accepts every
`N`-bit responder that makes no query with probability at most `δ + ε`, the run passes while
`q + N ≥ M + 1 + τ + H + k` with probability at most `δ + ε + c·2^-k`. So on an accept `q < M + 1 + τ + H − N + k`,
except with that probability. -/
def B2Adds : Prop :=
  ∀ {L : Type*} [LinearOrder L] {B w p : ℕ} [NeZero p] (sch : Pous.Scheme (Pous.ringRO B w p)) (N c : ℕ)
    (δ ε : ℝ≥0∞), Pous.AuditSecureOn sch .sequential N 0 0 0 c δ ε →
    ∀ (q k : ℕ) (W : Pous.Bits sch.n) (S : Pous.Bits q → Pous.Outcome sch c → Actual.Run L) (X : Finset L)
      (t : Pous.Bits q → Pous.Outcome sch c → ℕ → ℕ) (span : ℕ) (ans : ℕ → Actual.Acts L → Pous.Bits sch.ℓ)
      {α : Type*} (help : Pous.Bits q → Pous.Outcome sch c → α) (ζ : ℕ → List Bool → Pous.Bits q) (M τ H : ℕ),
      (∀ z, CausalFrom (S z)) → (∀ z x (j : Fin c), t z x j < t z x (j + 1)) →
      (∀ z, SetupFixed W (S z) X (t z) (help z)) → (∀ z, IngressFixed (S z) X (t z) (help z)) →
      (∀ z, ArrivalRule (S z) X (t z) (help z)) → (∀ z, TicksIn (t z) span τ) →
      (∀ z, (Set.range (help z)).encard ≤ ((2 ^ H : ℕ) : ℕ∞)) →
      (∀ z x, Timely (t z x) span c) → (∀ z x (j : Fin c), (S z x).mem X (t z x j) ≤ M) →
      (∀ z x (j : Fin c), ζ j ((S z x).content X (t z x j)) = z) →
      Pous.pr (fun x : Pous.Bits q × Pous.Outcome sch c =>
          Passes sch W (S x.1) X (t x.1) span ans x.2 ∧ ¬ q + N < M + 1 + τ + H + k) ≤
        δ + ε + c * (2⁻¹ : ℝ≥0∞) ^ k

/-- **`B2Adds` at P2's deployed point.** Under `P2RingAuditSecure 0 0 0` (P2 in the ring model, 2,054 payload bytes per
block, 105 challenges, state bound `⌊ρ·|C|⌋`), a location meeting `B2Adds`' hypotheses, with the first challenge at a
tick `t₀` the verifier fixes and `105·span ≤ 2^τ` in place of `TicksIn`, passes while
`q + ⌊ρ·|C|⌋ ≥ M + 1 + τ + H + k` with probability at most `δ + 2^-129 + 2^-130 + 105 · 2^-k`. -/
def B2AddsP2Ring : Prop :=
  Pous.P2.Assumptions.P2RingAuditSecure 0 0 0 →
    ∀ {L : Type*} [LinearOrder L] (q k : ℕ) (W : Pous.Bits (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).n)
      (S : Pous.Bits q → Pous.Outcome (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) 105 → Actual.Run L)
      (X : Finset L) (t : Pous.Bits q → Pous.Outcome (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) 105 → ℕ → ℕ)
      (t₀ span : ℕ) (ans : ℕ → Actual.Acts L → Pous.Bits (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).ℓ)
      {α : Type*} (help : Pous.Bits q → Pous.Outcome (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) 105 → α)
      (ζ : ℕ → List Bool → Pous.Bits q) (M τ H : ℕ),
      (∀ z, CausalFrom (S z)) → (∀ z x (j : Fin 105), t z x j < t z x (j + 1)) →
      (∀ z, SetupFixed W (S z) X (t z) (help z)) → (∀ z, IngressFixed (S z) X (t z) (help z)) →
      (∀ z, ArrivalRule (S z) X (t z) (help z)) → (∀ z x, t z x 0 = t₀) → 105 * span ≤ 2 ^ τ →
      (∀ z, (Set.range (help z)).encard ≤ ((2 ^ H : ℕ) : ℕ∞)) →
      (∀ z x, Timely (t z x) span 105) → (∀ z x (j : Fin 105), (S z x).mem X (t z x j) ≤ M) →
      (∀ z x (j : Fin 105), ζ j ((S z x).content X (t z x j)) = z) →
      Pous.pr (fun x : Pous.Bits q × Pous.Outcome (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) 105 =>
          Passes (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) W (S x.1) X (t x.1) span ans x.2 ∧
            ¬ q + (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).stateBound Pous.Params.ρ <
              M + 1 + τ + H + k) ≤
        ENNReal.ofReal Pous.Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130) + 105 * (2⁻¹ : ℝ≥0∞) ^ k

/-- **B2's oracle-location form: a window as a ring responder.** For a ring-model scheme and a window for each
encoder input `W`, public parameters, challenge index, `N`-bit state and challenged block (`Wd`), each yielding an
answer and the next state: when every multiplication takes at least `ℓ > 0` ticks to answer and every window spans at most
`D·ℓ` ticks, some `(N, D, Q, L)` sequential responder's step yields what the window yields, whatever the oracle
locations answer, wherever the window asks at most `Q` key queries and `L` ring queries. -/
def B2Window : Prop :=
  ∀ {B w p : ℕ} [NeZero p] (sch : Pous.Scheme (Pous.ringRO B w p)) (N D Q L ℓ : ℕ)
    (Wd : Pous.Bits sch.n → Pous.Bits sch.t → ℕ → Pous.Bits N → Fin sch.B →
      RingResponder B w (Pous.Bits sch.ℓ × Pous.Bits N)),
    0 < ℓ → (∀ W pp j σ i (q : Pous.RingQry B w), q.isMul = true → ℓ ≤ (Wd W pp j σ i).lat q) →
    (∀ W pp j σ i, (Wd W pp j σ i).span ≤ D * ℓ) →
    ∃ A₂ : Pous.SeqResponderOn sch N D Q L, ∀ W pp j σ i (f : Pous.RingQry B w → Pous.Bits w),
      (Wd W pp j σ i).keys f ≤ Q → (Wd W pp j σ i).ops f ≤ L →
      (A₂.step W pp j σ i).run f = (Wd W pp j σ i).result f

/-- **B2's oracle-location form, against PoUS's guarantee.** When PoUS's sequential audit with `k` challenges accepts
every `N`-bit `(D, Q, L)` responder with probability at most `δ + ε`: a location that starts the audit in an `N`-bit
state set up from the codeword (`A₁`) and answers each challenge with a window of `Wd`, whose multiplications take at
least `ℓ > 0` ticks to answer, which spans at most `D·ℓ` ticks and asks at most `Q` key and `L` ring queries, answers
every challenge with its block with probability at most `δ + ε`. Its answers are `seqAnswers` of the windows' results:
each window yields the answer and the state the next one starts in. -/
def B2WindowAudit : Prop :=
  ∀ {B w p : ℕ} [NeZero p] (sch : Pous.Scheme (Pous.ringRO B w p)) (N D Q L ℓ k : ℕ) (δ ε : ℝ≥0∞),
    Pous.AuditSecureOn sch .sequential N D Q L k δ ε →
    ∀ (W : Pous.Bits sch.n) (A₁ : Pous.Preprocessor sch N)
      (Wd : Pous.Bits sch.n → Pous.Bits sch.t → ℕ → Pous.Bits N → Fin sch.B →
        RingResponder B w (Pous.Bits sch.ℓ × Pous.Bits N)),
      0 < ℓ → (∀ W pp j σ i (q : Pous.RingQry B w), q.isMul = true → ℓ ≤ (Wd W pp j σ i).lat q) →
      (∀ W pp j σ i, (Wd W pp j σ i).span ≤ D * ℓ) →
      (∀ (ω : (Pous.ringRO B w p).Ω) W pp j σ i, (Wd W pp j σ i).keys ((Pous.ringRO B w p).answer ω) ≤ Q ∧
        (Wd W pp j σ i).ops ((Pous.ringRO B w p).answer ω) ≤ L) →
      Pous.pr (fun x : Pous.Outcome sch k =>
        Pous.seqAnswers ((Pous.ringRO B w p).answer x.1)
            (fun j σ i => Pous.Prog.ret ((Wd W (sch.enc W x.1 x.2.1).1 j σ i).result ((Pous.ringRO B w p).answer x.1)))
            0 (List.ofFn x.2.2) (A₁ W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1) =
          List.ofFn fun j => (sch.enc W x.1 x.2.1).2 (x.2.2 j)) ≤ δ + ε

/-- **`B2WindowAudit` at P2's deployed point, paced.** Under `P2RingAuditSecure D Q L` (P2 in the ring model, 2,054
payload bytes per block, 105 challenges, state bound `⌊ρ·|C|⌋`), with one challenge every `T` ticks (`T = Δ + RTT`;
each window spans at most `T`) and every multiplication reaching its querier at least `ℓ > 0` ticks after it is asked,
where `T ≤ D·ℓ` (the latency form of sequential speed, `ℓ ≥ (Δ + RTT)/D_p` at `D = D_p`): a location that starts in a
`⌊ρ·|C|⌋`-bit state and answers with such windows, each within `(Q, L)`, answers every challenge with its block with
probability at most `δ + 2^-129 + 2^-130`. -/
def B2WindowP2Ring : Prop :=
  ∀ D Q L T ℓ : ℕ, Pous.P2.Assumptions.P2RingAuditSecure D Q L →
    ∀ (W : Pous.Bits (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).n)
      (A₁ : Pous.Preprocessor (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432)
        ((Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).stateBound Pous.Params.ρ))
      (Wd : Pous.Bits (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).n →
        Pous.Bits (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).t → ℕ →
        Pous.Bits ((Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).stateBound Pous.Params.ρ) →
        Fin (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).B →
        RingResponder (2 ^ 19) 16448 (Pous.Bits (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).ℓ ×
          Pous.Bits ((Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).stateBound Pous.Params.ρ))),
      0 < ℓ → T ≤ D * ℓ →
      (∀ W pp j σ i (q : Pous.RingQry (2 ^ 19) 16448), q.isMul = true → ℓ ≤ (Wd W pp j σ i).lat q) →
      (∀ W pp j σ i, (Wd W pp j σ i).span ≤ T) →
      (∀ (ω : (Pous.ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω) W pp j σ i,
        (Wd W pp j σ i).keys ((Pous.ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).answer ω) ≤ Q ∧
        (Wd W pp j σ i).ops ((Pous.ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).answer ω) ≤ L) →
      Pous.pr (fun x : Pous.Outcome (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) 105 =>
        Pous.seqAnswers ((Pous.ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).answer x.1)
            (fun j σ i => Pous.Prog.ret ((Wd W ((Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).enc W x.1
              x.2.1).1 j σ i).result ((Pous.ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).answer x.1)))
            0 (List.ofFn x.2.2)
            (A₁ W ((Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).enc W x.1 x.2.1).1
              ((Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).enc W x.1 x.2.1).2 x.1) =
          List.ofFn fun j => ((Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).enc W x.1 x.2.1).2 (x.2.2 j)) ≤
        ENNReal.ofReal Pous.Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130)

end Accounting.Guarantees
