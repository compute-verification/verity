import FlockSoundness.Protocol.Audit.Closure

/-!
# Windows of calls

A window of calls is one population and one law. Its strata are the pairs (call, template), `call s` being stratum `s`'s
call, and `Law.stratified` draws a uniform `k_s`-subset of every stratum independently: the product of the calls' draws.

Two modelling lines sit outside the statements:
* M1: each call is drawn on its own coins, the window's key derived under a **distinct context** for every call. Two calls
  that share a context draw the same subsets, and the window's draw is then not this product law.
* M2: the window's audit is one `Analysis` over the window's session, with the window's `ε_ks` and `δ_link`. How those
  compose from the calls' is not stated here.

`audit_window_of_le` takes any law whose escapes are at most the product law's, such as an executable draw proved against
it, so the ideal law here need not be the one run. `audit_window_of_le_slack` allows those escapes an additive `η`, which
a keyed draw's pseudorandomness assumption supplies, and adds `η` to the bound; at `η = 0` it is `audit_window_of_le`.
`extraction_audit_window_of_le_slack` and `extraction_audit_window_split_of_record_of_le_slack` are the same at the
compiled layer.
`audit_window_split_of_record_of_le_slack` is its record form for the per-call split with y's floors.

Two weightings of the same draw: `w`, the work (each tile's W_ref on the tile strata, 0 elsewhere), and `v`, the y cells
(1 on the dequantization strata, 0 elsewhere).

* `Covers σ k w K`: every stratum draws at least its share `K · w_s · n_s / W`, or all its units.
  `stratified_escape_le_of_covers`, **the work bound for any law whose sizes cover the weights**:
  `escape B ≤ ((W − W_B)/W)^K`. `covers_work`: the work law covers its own work.
* `windowK`, **the per-call split** (X-SPC-106): call `c` draws by its own work law at its share
  `K_c = min(N_c, max(1, ⌈K · W_c / W⌉))` of the window's `K`. `covers_window`: that covers the window's work when each
  call's work is in one stratum (`hone`). `covers_of_floor`, **y's floors** (X-SPC-105, option B): sizes that draw at
  least `min(n_s, f_s)`, with `K_y · v_s · n_s ≤ f_s · N`, cover `v` at `K_y`.
* `audit_window`, `extraction_audit_window`: the audit accepts while the unsound work is at least `T` or the unsound
  weight (`unsoundWork` under `v`) at least `T_y` with probability at most `max(((W − T)/W)^K, ((N − T_y)/N)^(K_y)) + ε_ks + δ_link`. The larger,
  not the sum: both bound the escape of the one wrong set the prover committed before the draw (`audit_profile`).
  `audit_window_of_record`: at `K = K_y = 27,713`, a 0.1% share of either with at most `2^-40 + ε_ks + δ_link`, and
  `audit_window_split_of_record` for the per-call split with y's floors, the law #364 runs.

**`hone` is needed, and not only by this route.** When a call's share `⌈K · W_c / W⌉` is capped at its units `N_c`, a
second stratum doing work in the call can draw well below its share, and then the window bound itself fails. With call 0
of 1,000 units at work 2 and 10 at work 1,000, call 1 of 100,000 units at work 0 and 1 at work 1, and `K = 5,000`, call
0's share caps at `K_0 = 1,010`, its work-2 stratum draws 169 of its 1,000 units, and ten wrong units there escape with
0.156 against the bound's 2.4·10⁻⁴. Uncapped it would draw 834. A call with several work strata must have a cap that
does not bind, or prove those strata whole.
-/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal NNReal

namespace Law

variable {n m C : ℕ}

/-- **Sizes that cover the weights `w` at budget `K`**: every stratum draws at least its share `K · w_s · n_s / W` of
the weight, or all its units. -/
def Covers (σ : Fin n → Fin m) (k w : Fin m → ℕ) (K : ℕ) : Prop :=
  ∀ s, K * w s * (stratum σ s).card ≤ k s * totalWork σ w ∨ k s = (stratum σ s).card

/-- Call `c`'s work `W_c`, its strata's. -/
def callWork (σ : Fin n → Fin m) (call : Fin m → Fin C) (w : Fin m → ℕ) (c : Fin C) : ℕ :=
  ∑ s ∈ univ.filter (call · = c), w s * (stratum σ s).card

/-- Call `c`'s units `N_c`. -/
def callUnits (σ : Fin n → Fin m) (call : Fin m → Fin C) (c : Fin C) : ℕ :=
  ∑ s ∈ univ.filter (call · = c), (stratum σ s).card

/-- Call `c`'s budget, its share of the window's `K`: `K_c = min(N_c, max(1, ⌈K · W_c / W⌉))`. -/
def callK (σ : Fin n → Fin m) (call : Fin m → Fin C) (w : Fin m → ℕ) (K : ℕ) (c : Fin C) : ℕ :=
  min (callUnits σ call c) (max 1 ((K * callWork σ call w c + totalWork σ w - 1) / totalWork σ w))

/-- The window's sizes: each call's work law at its own budget, `k_s = min(n_s, max(f_s, ⌈K_c · w_s · n_s / W_c⌉))`. -/
def windowK (σ : Fin n → Fin m) (call : Fin m → Fin C) (w f : Fin m → ℕ) (K : ℕ) (s : Fin m) : ℕ :=
  workRule (callK σ call w K (call s)) (w s) (stratum σ s).card (callWork σ call w (call s)) (f s)

end Law

section Audits

variable {n m : ℕ} {σ : Fin n → Fin m} {k : Fin m → ℕ} {hk : ∀ s, k s ≤ (Law.stratum σ s).card}
  {cl : Fin n → Finset (Fin n)} {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}

/-- At `η = 0`, `audit_window_of_le_slack` is `audit_window_of_le`. -/

end Audits

end FlockSoundness.Audit
