import Definitions.Pouw.Game.Params
import Models.Pouw.Assumptions.Game
import Models.Pouw.Assumptions.NCP.Assumptions
import Proofs.Flock.Soundness.Audit.Window

/-!
# PoUW's work bound at C-Flock's window audit (trusted)

`EndToEndWindow` is `Pouw.Guarantees.EndToEnd` with two changes:
* its ideal audit (`t` work-weighted tickets, each sampled unit recomputed) is C-Flock's window audit at the law of
  record, which `FlockSoundness.Audit.extraction_audit_window_split_of_record_of_le_slack` bounds;
* its work step is a parameter, `WindowWork.TileWork`, in place of TT.

It is a statement with no proof yet; its proof is to be
`Pouw.SecurityProofs.EndToEndWindow : Pouw.Guarantees.EndToEndWindow`.

The work step's instances:
* NCP's and Pearl's, from their conjectures TT_NCP and TT through `G_γ`'s reduction: `WindowWork.TileWorkOfTT` and
  `WindowWork.TileWorkOfTTNCP`, statements to be proved as `tileWork_of_TT` and `tileWork_of_TTNCP`.
* EPS1's (sm_120 H1 + fx1, duo, LI-CP-W), outside TT's domain: from compute accounting's tensor–scalar model,
  `Accounting.Guarantees.Compose.CircuitCredit`, through a bridge from its play to PoUW's cost model. That bridge is
  PoUW's gap (3), and it isn't stated here.

C-Flock's audit definitions are still in the proofs package, so this module imports
`Proofs.Flock.Soundness.Audit.Window`, as `Flock.Guarantees.EndToEnd`'s statement does C-Flock's; `Models.lean`, which the
`security` package builds without `security_proofs`, does not import it. `TileWork` and `RegisteredIsOnline` are named
premises awaiting their card.
-/

namespace Pouw.WindowWork

open Finset FlockSoundness.Audit

/-- **The window's law of record** (#364): every call drawn by its own work law at its share of `K = 27,713`, each
stratum at least its floor `f`, with closures `cl`. It is the law `audit_window_split_of_record` bounds. -/
noncomputable def lawOfRecord {n m C : ℕ} (σ : Fin n → Fin m) (call : Fin m → Fin C) (w f : Fin m → ℕ)
    (cl : Fin n → Finset (Fin n)) : Law n :=
  (Law.stratified σ (Law.windowK σ call w f 27713) (Law.windowK_le σ call w f 27713)).closure cl

/-- **The work step, TileWork(γ_u)** (a named premise, open), for a cost model `CM`, a protocol `P`, a domain `D`, a
gap `γ_u` and an error `η(q, N)`. A correct tile was paid for: the prover spent at least `(1 − γ_u)·W_ref(u)` on it.
A cost model prices a whole run, not a tile, so it is stated summed over the window, in `G_γ`'s game (the same
adversaries, budgets and oracle-call bounds). For every adversary within `(T, q)`,
$$\Pr_{H,s}\Big[T < (1-\gamma_u)\sum_{u \in K} W_{ref}(u)\Big] \le \eta(q, N),$$
`K` being the correct units (`correctSet`: checked values and useful output correct). It is `G_γ` at `γ = γ_u` with
the gap multiplied out, so it needs no `γ_u < 1`: at `γ_u ≥ 1` its event is empty.

Its instances: NCP's and Pearl's from TT_NCP and TT (`TileWorkOfTTNCP`, `TileWorkOfTT`), both conjectures; EPS1's
from `Accounting.Guarantees.Compose.CircuitCredit`, PoUW's gap (3). -/
def TileWork {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (D : Layout → Prop) (γu : ℝ) (η : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T < (1 - γu) * ∑ u ∈ correctSet P U x.1 x.2 (A.transcript U x.1 x.2), (P.Wref U.layout u : ℝ))
      ≤ η q U.layout.N

/-- **`TileWork` from TT** (statement, no proof yet; to be proved as `tileWork_of_TT`): Pearl's instance, `G_γ`'s
reduction (`Pinned.GammaFromTT`) read as the work step. If `W_ref(u) ≤ Ω_max·m·k·n` on the domain and TT(γ₀) holds
with error `ε`, then `TileWork` holds at `γ_u = 1 − (1 − γ₀)/Ω_max` with error `ε`. -/
def TileWorkOfTT : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (γ₀ Ωmax : ℝ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM P D γ₀ ε →
    TileWork CM P D (reductionGamma γ₀ Ωmax) ε

/-- **`TileWork` from TT_NCP** (statement, no proof yet; to be proved as `tileWork_of_TTNCP`): NCP's instance, at
the credit `3·m·k·n` (`GamePinned.GammaFromTTNCP` read as the work step). -/
def TileWorkOfTTNCP : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (γ₀ Ωmax : ℝ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * (3 * ((U.layout.shape u).Wmm : ℝ))) →
    NCP.Assumptions.TTNCP CM P D γ₀ ε →
    TileWork CM P D (reductionGamma γ₀ Ωmax) ε

/-- **The window realizes PoUW's units** (open obligation, to be proved), at the oracle `H` and the salt `s`, for the
wrong units `wrong` of an analysis's transcripts and the map `tx` from them to PoUW transcripts. At every committed
transcript `X`, the declared work of the units that are not correct in `tx X` is at most the window's unsound work at
`X`'s wrong units:
$$\sum_{u < N,\ u \notin K(tx\,X)} W_{ref}(u) \le \mathrm{unsoundWork}_w(cl, \mathrm{wrong}\,X).$$
`K` is PoUW's `correctSet`: int7 activations, checked values equal to `Y` at the noise from `H` and `s`, and the useful
output `A·B` at the registered weights.

For compiled PoUW it holds by construction (compute accounting, 18 Oct): under decision 106 a PoUW unit is one
self-contained `r × c` tile, one template instance in the Program's partition, and member splits fall between tiles.
So it is to be discharged from the compiler, with the work table (each tile's `W_ref` is its audit unit's work). It
fails when a PoUW unit spans several audit units that the closures don't tie together: one wrong tile makes the whole
PoUW unit incorrect, but puts only that tile's work in the unsound work. -/
def Realizes {Q R S : Type} (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) {n m : ℕ}
    (σ : Fin n → Fin m) (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) {Tr : Type} (wrong : Tr → Finset (Fin n))
    (tx : Tr → Pouw.Transcript) : Prop :=
  ∀ X : Tr, ∑ u ∈ Finset.range U.layout.N \ correctSet P U H s (tx X), (P.Wref U.layout u : ℝ) ≤
    (Law.unsoundWork σ w cl (wrong X) : ℝ)

/-- **The registered transcript is the online one** (a named premise, open; it needs a card). At every oracle `H` and
salt `s`, the transcript the analysis fixes at registration carries, through `tx`, the PoUW adversary's online
transcript `A.transcript U H s`, the one whose cost is `T`. That fixed transcript is the committed transcript of the
prover's audit strategy `σ' H s`, which in the compiled analysis is the extractor's plurality behind the registered roots.

It is an extractability statement: whatever stands behind the registered roots was computed by the online phase before
registration, not found later. For the frame-v3-sha512 trees of hm96 rows that holds in the random-oracle model or under
a knowledge assumption, and neither is stated in Lean. Nor can this game prove it: PoUW's game has ideal commitments,
and C-Flock's audit game has no oracle. It is carded with `Accounting.Assumptions.Compose.SeedGrind`, so that Daniel
rules on the random oracle once. -/
def RegisteredIsOnline {Q R S : Type} {CM : CostModel Q R S} (A : Adversary CM) (U : Workload) {n : ℕ} {L : Law n}
    {Reg : Type} {session : (Q → R) → S → Finset (Fin n) → Reg → FlockSoundness.Game Bool}
    (An : (H : Q → R) → (s : S) → ExtractionAnalysis L Reg (session H s))
    (tx : (H : Q → R) → (s : S) → (An H s).Tr → Pouw.Transcript)
    (σ' : (H : Q → R) → (s : S) → FlockSoundness.Game.Strategy (audit L Reg (session H s))) : Prop :=
  ∀ H s, tx H s ((An H s).committedOf (σ' H s)) = A.transcript U H s

end Pouw.WindowWork

namespace Pouw.Guarantees

open Finset FlockSoundness.Audit
open scoped ENNReal

/-- **PoUW's work bound at C-Flock's window audit** (statement only, no proof yet). A prover that passes C-Flock's
window audit did at least a `(1 − γ_u)(1 − 1/1000)` share of the window's declared work, except with probability
`2^-40 + κ + E[coll + δ_link] + ε(q, N)`, the expectation taken at the prover's own state.

**What it bounds.** The prover's work `T` is the cost of its online phase, from the salt to registration, in the cost
model `CM`'s units (`CM.cost`, which the model fixes, not the prover). It is compared with the window's declared work
`W = ∑_{u < N} W_ref(u)` in the same units, `W_ref(u)` being the honest reference's cost of unit `u` without hash calls.
Averaged over the oracle `H` and the salt `s`, the window audit accepts while `T < (1 − γ_u)(1 − 1/1000)·W` with
probability at most
$$\mathbb{E}_{H,s}\Pr[\text{accept} \wedge T < (1-\gamma_u)(1-10^{-3})\,W] \le 2^{-40} + \kappa +
  \mathbb{E}_{H,s}\big[\mathrm{coll}(R, \tau) + \delta_{link}(R, \tau)\big] + \varepsilon(q, N),$$
`(R, τ)` being the prover's state at `(H, s)`: its registration and its session strategy for every draw.
It has `EndToEnd`'s form. The audit is C-Flock's one-stage audit (`FlockSoundness.Audit.audit`) at the window's law of
record (`WindowWork.lawOfRecord`), the prover's strategy `σ'` may depend on `H` and `s`, `ε_s = 1/1000` is the law's
sizing share, and `2^-40 + κ + coll + δ_link` takes the place of `(1 − ε_s)^t`.

**What it assumes**, each a hypothesis:
* the work step `WindowWork.TileWork γ_u` with error `ε` (a named premise; `TileWorkOfTT` and `TileWorkOfTTNCP` are its
  instances from NCP's and Pearl's conjectures, EPS1's is gap (3));
* knowledge soundness and the link term at the prover's state, in the compiled tables' shape (`ksJoint_le`): `κ`, a
  constant, plus `coll(R, τ)`, the one window-level collision term, which is about 1 for a prover holding a collision
  and small for one whose finders meet `SHA512CRExpected`; and `δ_link(R, τ)`. A constant over every state can't be
  met: collisions of SHA-512 exist, and a prover that knows one wins;
* the law of record's sizing, as `audit_window_split_of_record` takes it, and the work table: the window's work is at
  most `W`;
* `WindowWork.Realizes` at every oracle and salt (an open obligation, by construction for compiled PoUW);
* `WindowWork.RegisteredIsOnline` (a named premise, carded with `SeedGrind`).

**What it composes**, at every oracle and salt: either the work step's event holds (the prover's correct tiles are
worth more than `T/(1 − γ_u)`), whose average is at most `ε`; or the incorrect work exceeds `W/1000`, and by `Realizes`
and the work table the window's unsound work exceeds its 0.1% share, which
`FlockSoundness.Audit.extraction_audit_window_split_of_record_of_le_slack` bounds (at the ideal law, `η = 0`). -/
def EndToEndWindow : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (γu : ℝ) (ε : ℕ → ℕ → ℝ),
    WindowWork.TileWork CM P D γu ε →
    ∀ U : Workload, U.InDomain D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary CM, A.Bounded U T q →
    -- the window: `n` audit units in `m` strata, stratum `t` in call `call t`, at work `w t` and y weight `v t`
    ∀ (n m C : ℕ) (σ : Fin n → Fin m) (call : Fin m → Fin C) (w v f : Fin m → ℕ) (cl : Fin n → Finset (Fin n)),
    -- the law of record's sizing
    0 < Law.totalWork σ w →
    (∀ t, 0 < w t → Law.callWork σ call w (call t) = w t * (Law.stratum σ t).card) →
    0 < Law.totalWork σ v →
    (∀ t, 27713 * v t * (Law.stratum σ t).card ≤ f t * Law.totalWork σ v) →
    -- the work table
    (Law.totalWork σ w : ℝ) ≤ totalWref P U.layout →
    -- the session at each oracle and salt, its analysis, and the PoUW transcript a committed transcript carries
    ∀ (Reg : Type) (session : (Q → R) → S → Finset (Fin n) → Reg → FlockSoundness.Game Bool)
      (An : (H : Q → R) → (s : S) →
        ExtractionAnalysis (WindowWork.lawOfRecord σ call w f cl) Reg (session H s))
      (tx : (H : Q → R) → (s : S) → (An H s).Tr → Transcript),
    (∀ H s, WindowWork.Realizes P U H s σ w cl (An H s).wrong (tx H s)) →
    -- knowledge soundness and the link term at the prover's state: `κ` plus the window's collision term, and `δ_link`
    ∀ (κ : ℝ≥0∞) (coll δlink : (H : Q → R) → (s : S) → (R' : Reg) →
      Cont (WindowWork.lawOfRecord σ call w f cl) Reg (session H s) R' → ℝ≥0∞),
    (∀ H s, (An H s).KnowledgeSound fun R' τ => κ + coll H s R' τ) →
    (∀ H s, (An H s).LinkSound (δlink H s)) →
    -- the prover's audit strategy, whose registration commits its online transcript
    ∀ σ' : (H : Q → R) → (s : S) →
      FlockSoundness.Game.Strategy (audit (WindowWork.lawOfRecord σ call w f cl) Reg (session H s)),
    WindowWork.RegisteredIsOnline A U An tx σ' →
      FlockSoundness.Game.avg (fun x : (Q → R) × S =>
        FlockSoundness.Game.prob
          (fun o => o.1 = true ∧ T < (1 - γu) * (1 - 1 / 1000) * totalWref P U.layout)
          (audit (WindowWork.lawOfRecord σ call w f cl) Reg (session x.1 x.2)) (σ' x.1 x.2))
        ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ + κ +
          FlockSoundness.Game.avg (fun x : (Q → R) × S =>
            coll x.1 x.2 (reg (σ' x.1 x.2)) (cont (σ' x.1 x.2)) +
              δlink x.1 x.2 (reg (σ' x.1 x.2)) (cont (σ' x.1 x.2))) +
          ENNReal.ofReal (ε q U.layout.N)

end Pouw.Guarantees
