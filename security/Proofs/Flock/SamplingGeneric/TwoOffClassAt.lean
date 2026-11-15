module

public import Proofs.Flock.SamplingGeneric.TwoFirstHitClassAt
import Proofs.Flock.Claims.Prob
import Proofs.Flock.SamplingGeneric.RegTrace

/-!
# The two off-trace masses by class of gates, at a budget a run

`eventTwoX_le_of_off` bounds the event of `KSRegTwoX` from three masses of one prover's runs and their extractions:
that `B_conf,X` finds a collision, that a run's extractions open a gate away from the boundary trace read at the
statement (`Compiled.OffRegX`), and that they open one away from its replay block's committed interior
(`CompiledTwo.OffIreg`). `Compiled.bconfX_le_at` bounds the first by `2·runCost/2^256` where `B_conf,X` meets the
linear expected-time bound. This file bounds the other two by `2·firstHitCost/2^256` where the by-class finders meet
it at every `K`:

* `Compiled.pr_offRegX_le_cls_at`: `Pr[off the boundary trace] ≤ 2·firstHitCost/2^256`;
* `CompiledTwo.pr_offIreg_le_cls_at`: `Pr[off a committed interior] ≤ 2·firstHitCost/2^256`,

with `firstHitCost = T·(1 + e_s·N_s + K_s + e_s·K_m) + (1 + K_s)·n_PU·openCalls` at a budget `T` on a run's calls of
`H`. Each is the by-class bound at the finder that stops at `K`, taken at every `K`, with the finder's expected calls
in closed form (`Compiled.ex_calls_le_at_cls`). Neither event asks that the run accepted, so each bounds the accepted
runs' mass too (`CompiledTwo.pr_acc_offRegX_le_cls_at`, `CompiledTwo.pr_acc_offIreg_le_cls_at`).

So none of the three masses is a premise of its own: each follows from the linear expected-time bound on `H`
(`ExpectedCR`), read at `B_conf,X` and at the two by-class finders at every `K`.

The budget `T` need not be a hypothesis either: `Compiled.runCalls` is the most calls of `H` a run of the prover
makes, and every run stays within it (`Compiled.runCalls_le`), so each bound holds at `T = runCalls`.

At one stage the event has the first two masses only (`Compiled.event_le_of_off` at the trace read at the statement),
so it is at most `κ_s + √(k·γ_s·Pr[B]) + (2·runCost + 2·firstHitCost)/2^256` (`Compiled.eventX_le_cls_at`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition)

namespace Compiled

section RunCalls

variable {Coin : Type} [Fintype Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The most calls of `H` a run of the prover makes** at the statement `x`, both parties' calls counted, over the
prover's tapes and the run's coins: the budget `T` a bound at a budget reads, as a function of the prover. -/
noncomputable def runCalls (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) {Ω : Type} [Fintype Ω] (P : Prover Ω) :
    ℕ :=
  Finset.univ.sup fun p : Ω × (Fin (L.V.k x) → Coin) =>
    (run H P L.V x p.1 p.2).pLog.length + (run H P L.V x p.1 p.2).vLog.length

/-- Every run calls `H` at most `runCalls` times. -/
theorem runCalls_le (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) {Ω : Type} [Fintype Ω] (P : Prover Ω) (ω : Ω)
    (r : Fin (L.V.k x) → Coin) :
    (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ L.runCalls H x P :=
  Finset.le_sup (f := fun p : Ω × (Fin (L.V.k x) → Coin) =>
    (run H P L.V x p.1 p.2).pLog.length + (run H P L.V x p.1 p.2).vLog.length) (Finset.mem_univ (ω, r))

end RunCalls

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The mass off the boundary trace read at the statement, by class, at a budget `T` a run**: where a run's
extractions open a class of gates whole or not at all under the statement's roots (`hallX`), every run calls `H` at
most `T` times, both parties' calls counted (`hT`), the opening checks call `H` at most `openCalls` times a slot
(`hopen`), `N_s`, `K_s` and `K_m` are `ex_calls_le_at_cls`'s three counts, and the by-class finder at every `K`
meets the linear expected-time bound, a run's extractions open a gate away from the boundary trace read at the
statement with probability at most `2·firstHitCost/2^256`. -/
theorem pr_offRegX_le_cls_at {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H)
    (hS : L.CostContract H) (hopen : L.OpenCallsLe H) {x : Msg} {s : AuditStatement} (hs : L.parse x = some s)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    (T : ℕ) (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      (((L.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin =>
      (L.sessCountIn x needs (fun l => (L.sampleNeeds x needs l).image cls) (L.checked x r) : ℝ)) ≤ Km)
    (hX : ∀ K : ℕ, ExpectedCR H (L.finderXCls H P x s needs cls K)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffRegX H P x q.1 q.2.1 q.2.2) ≤
      2 * L.firstHitCost x T Ns Ks Km / 2 ^ (256 : ℝ) := by
  have hall : ∀ K : ℕ,
      Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffRegX H P x q.1 q.2.1 q.2.2) ≤
        2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
            ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
              ∑ c ∈ (univ.filter (fun g : Fin (L.nG x) =>
                  decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))).image cls,
                L.callsBy H P x s needs c.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
            2 ^ (256 : ℝ) +
          (L.nG x : ℝ) / ((K : ℝ) + 1) :=
    fun K => pr_offRegX_le_cls_cap hbind hs P needs hpos cls hallX K (hX K)
  have h := Verity.FirstHit.le_of_forall_cap hall
  have hc := ex_calls_le_at_cls hS hopen hs P needs cls T hT hNs hKs hKm
  exact h.trans (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hc (by norm_num : (0 : ℝ) ≤ 2))
    (by positivity : (0 : ℝ) ≤ 2 ^ (256 : ℝ)))

set_option synthInstance.maxSize 1024 in
/-- **The one-stage event at the trace read at the statement, by class, at a budget `T` a run**: on a statement in
scope, under `pr_offRegX_le_cls_at`'s premises and the linear expected-time bound at `B_conf,X`, an accepted run
samples a proof block of `FregX` with probability at most

`κ_s + √(k·γ_s·Pr[B]) + (2·(T·(1 + e_s·N_s) + n_PU·openCalls) + 2·(T·(1 + e_s·N_s + K_s + e_s·K_m)`
`+ (1 + K_s)·n_PU·openCalls))/2^256`:

`event_le_of_off` at the trace read at the statement, with one run's own disagreement at `B_conf,X`
(`bconfX_le_at`) and the gates opened off the trace at the by-class finders (`pr_offRegX_le_cls_at`), where `εregX`
has `Pr[B_conf,X]` and `N·√Pr[B_conf,X]`. -/
theorem eventX_le_cls_at {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) (hopen : L.OpenCallsLe H) {x : Msg} {s : AuditStatement} {Pt : Partition}
    (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hin : Sampling.InScope Pt s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    (T : ℕ) (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      (((L.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin =>
      (L.sessCountIn x needs (fun l => (L.sampleNeeds x needs l).image cls) (L.checked x r) : ℝ)) ≤ Km)
    (hB : ExpectedCR H (BconfX H L Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR H (L.finderXCls H P x s needs cls K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregX H P x) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
        (2 * L.runCost x T Ns + 2 * L.firstHitCost x T Ns Ks Km) / 2 ^ (256 : ℝ) := by
  have hI := (pr_inConfX_le hbind hs P).trans (bconfX_le_at hS.toCostContract hopen x P T hT hNs hB)
  have hO := (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffRegX H P x q.1 q.2.1 q.2.2) fun _ h => h.2).trans
    (pr_offRegX_le_cls_at hbind hS.toCostContract hopen hs P needs hpos cls hallX T hT hNs hKs hKm hX)
  simp only [FregX_eq_FregAt]
  refine (event_le_of_off hS hSK hs hPt hin P (fun _ _ => L.regValX H P x) hI hO).trans ?_
  rw [add_div]
  linarith

end Compiled

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The mass off the committed interiors, by class, at a budget `T` a run**: where a second stage's extractions
open a class of gates whole or not at all under a replay block's interior roots (`hallI`), every run calls `H` at
most `T` times, both parties' calls counted (`hT`), the opening checks call `H` at most `openCalls` times a slot
(`hopen`), `N_s`, `K_s` and `K_m` are `Compiled.ex_calls_le_at_cls`'s three counts, and the by-class finder at every
`K` meets the linear expected-time bound, a run's extractions open a gate away from the committed interior of the
replay block of `Pρ` that holds it with probability at most `2·firstHitCost/2^256`. -/
theorem pr_offIreg_le_cls_at {H : Msg → Msg} {L : CompiledTwo Coin A} (hbind : L.commit.Binds H)
    (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {x : Msg} {s : AuditStatement}
    (hs : L.parse x = some s) (Pρ : Partition) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (needs : ℕ → Finset ℕ) (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u)
    (cls : Fin (L.nG x) → Fin (L.nG x))
    (hallI : ∀ (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g g' : Fin (L.nG x)),
      cls g = cls g' → (L.iopTable H P x t.1 Pρ t.2 ec g).isSome → (L.iopTable H P x t.1 Pρ t.2 ec g').isSome)
    (T : ℕ) (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      (((L.toCompiled.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin =>
      (L.toCompiled.sessCountIn x needs (fun l => (L.toCompiled.sampleNeeds x needs l).image cls)
        (L.checked x r) : ℝ)) ≤ Km)
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderICls H P x s Pρ needs cls K)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffIreg H P x q.1 Pρ q.2.1 q.2.2) ≤
      2 * L.toCompiled.firstHitCost x T Ns Ks Km / 2 ^ (256 : ℝ) := by
  have hall : ∀ K : ℕ,
      Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffIreg H P x q.1 Pρ q.2.1 q.2.2) ≤
        2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
            ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
              ∑ c ∈ (univ.filter (fun g : Fin (L.nG x) =>
                  decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))).image cls,
                L.toCompiled.callsBy H P x s needs c.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
            2 ^ (256 : ℝ) +
          (L.nG x : ℝ) / ((K : ℝ) + 1) :=
    fun K => pr_offIreg_le_cls_cap hbind hS hs Pρ P needs hpos cls hallI K (hI K)
  have h := Verity.FirstHit.le_of_forall_cap hall
  have hc := Compiled.ex_calls_le_at_cls hS.toCostContract hopen hs P needs cls T hT hNs hKs hKm
  exact h.trans (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hc (by norm_num : (0 : ℝ) ≤ 2))
    (by positivity : (0 : ℝ) ≤ 2 ^ (256 : ℝ)))

/-- **An accepted run is off the boundary trace read at the statement with probability at most
`2·firstHitCost/2^256`**, under `Compiled.pr_offRegX_le_cls_at`'s hypotheses at the compiled two-stage verifier:
the bound a first-hit error takes on its second mass, at `c_F = firstHitCost`. -/
theorem pr_acc_offRegX_le_cls_at {H : Msg → Msg} {L : CompiledTwo Coin A} (hbind : L.commit.Binds H)
    (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {x : Msg} {s : AuditStatement}
    (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    (T : ℕ) (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      (((L.toCompiled.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin =>
      (L.toCompiled.sessCountIn x needs (fun l => (L.toCompiled.sampleNeeds x needs l).image cls)
        (L.checked x r) : ℝ)) ≤ Km)
    (hX : ∀ K : ℕ, ExpectedCR H (L.toCompiled.finderXCls H P x s needs cls K)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (run H P L.V x q.1 q.2.1).out = .acc ∧
      L.OffRegX H P x q.1 q.2.1 q.2.2) ≤ 2 * L.toCompiled.firstHitCost x T Ns Ks Km / 2 ^ (256 : ℝ) :=
  (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffRegX H P x q.1 q.2.1 q.2.2) fun _ h => h.2).trans
    (Compiled.pr_offRegX_le_cls_at (L := L.toCompiled) hbind hS.toCostContract hopen hs P needs hpos cls hallX T hT
      hNs hKs hKm hX)

/-- **An accepted run is off a committed interior with probability at most `2·firstHitCost/2^256`**, under
`pr_offIreg_le_cls_at`'s hypotheses: the bound a first-hit error takes on its third mass, at
`c_F = firstHitCost`. -/
theorem pr_acc_offIreg_le_cls_at {H : Msg → Msg} {L : CompiledTwo Coin A} (hbind : L.commit.Binds H)
    (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {x : Msg} {s : AuditStatement}
    (hs : L.parse x = some s) (Pρ : Partition) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (needs : ℕ → Finset ℕ) (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u)
    (cls : Fin (L.nG x) → Fin (L.nG x))
    (hallI : ∀ (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g g' : Fin (L.nG x)),
      cls g = cls g' → (L.iopTable H P x t.1 Pρ t.2 ec g).isSome → (L.iopTable H P x t.1 Pρ t.2 ec g').isSome)
    (T : ℕ) (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      (((L.toCompiled.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin =>
      (L.toCompiled.sessCountIn x needs (fun l => (L.toCompiled.sampleNeeds x needs l).image cls)
        (L.checked x r) : ℝ)) ≤ Km)
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderICls H P x s Pρ needs cls K)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (run H P L.V x q.1 q.2.1).out = .acc ∧
      L.OffIreg H P x q.1 Pρ q.2.1 q.2.2) ≤ 2 * L.toCompiled.firstHitCost x T Ns Ks Km / 2 ^ (256 : ℝ) :=
  (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffIreg H P x q.1 Pρ q.2.1 q.2.2) fun _ h => h.2).trans
    (pr_offIreg_le_cls_at hbind hS hopen hs Pρ P needs hpos cls hallI T hT hNs hKs hKm hI)

end CompiledTwo

end Claims.SamplingGeneric
