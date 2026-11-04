import Definitions.NetworkCertifier.TiedLeak
import Definitions.NetworkCertifier.Assumptions.Tied
import Definitions.NetworkCertifier.Tap
import Definitions.Core.Assumptions
import Mathlib.Data.Set.Card
import Mathlib.Algebra.BigOperators.Fin

/-!
# The leak through the network certifier over a monitoring period of rows-form phases

`PeriodLeakAccepted` with each window a rows-form phase, counted as its joint Program (`Phase.prog`): its scan's and
its tied compute audits' Programs side by side, with the ties among its wires (`Definitions.NetworkCertifier.Tied`). A
tied gate is no free input, so the content term is the joint Program's free inputs less the registered ones, which the
period check's count bounds (`CircuitCounts`).

`PeriodLeakTied` has two parts:

* **The tie.** At a phase registered as `pw i` whose ties are placed together (`RowsStateHeld`, `RowsProgWired`), the
  sampling relation at each statement's own incorrect set (`RWired`, at its own wires and its `δ`) gives the phase's leak
  relation at the network certifier's rows (`RLeakTied`), or two of the traces' openings, or one of them and one of the
  network certifier's, collide. A tie is right by binding: its two ends are committed at one position under one root
  (`Assumptions.SharedRootsParsed`, at the deployed parse), and a trace's transfer gate is the network certifier's
  (`CommitmentScheme.Binds`). So a wrong tie is no new error: each statement fails `RWired` with probability at most
  `δ + ε_ks + ε_bind` at its session (`KSWired`), and a collision is found on the openings the extractions check.
* **The count.** Where the network certifier registers its observations at every phase's scan, the secrets of a covert
  sender whose every phase meets its good event (`PhaseGood`) number at most
  `#Ss · ∏_c (cap_c + 1) · ∏_i 2^free_i · ∏_j #faultySets_ij(δ_ij) · 2^β_ij · Dj^k_i`, `free_i` the period check's
  count of phase `i`'s free inputs (`CircuitCounts`), and `j` over the phase's statements.

The union of the sessions' and the windows' failures over the monitoring period's live coins is not stated here, nor is
the finder for the collision over two sessions' extractions. A wrong scan proof block pays as in `PeriodLeakAccepted`;
a wrong tied proof block pays through its statement's `faultySets` and `β`.
-/

namespace NetTiming.Guarantees

open Claims.Sampling Finset

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak through the network certifier over a monitoring period of rows-form phases.** At every deployment `D` of
the sampled verifier whose parse puts each tie's two ends at one position (`Assumptions.SharedRootsParsed`, at the
registrations' placement `place`) and whose trace commitment binds (`CommitmentScheme.Binds`), over a monitoring period
of `n` phases registered as `pw i` (its statements' partitions `Pw i`, ties `t i`, each statement `j` read at `δ i j`
with `β i j` bounding the committed bits of each of its possible incorrect sets, the joint Program's registered inputs
`reg i` at the values `ρ i`, and operator `op i`), where each phase's layout fits its joint Program (`RowsTiesOrdered`)
and the period check counts at least its free inputs less the registered ones (`CircuitCounts`, at `free i`), with the
network certifier's openings `tops s` at each statement, and a network certifier that runs its code by `trusted-host`
on schedules in `Ss`, each held by its record format `f` and giving connection `c` at most `cap c` cells:

* at every phase whose statements parse to a phase registered as `pw i` with its ties placed together (`RowsStateHeld`,
  `RowsProgWired`), traces at which each statement meets the sampling relation at its own incorrect set (`RWired`)
  meet the phase's leak relation at the network certifier's rows (`RLeakTied`), or the `H` calls that check their
  openings and the network certifier's collide (`LogColl`);
* where at every phase's scan statement the network certifier registers its observations at that phase's operator
  (`TapOpens`), a covert sender in the server, whose secrets a decoder recovers from the client's view of the server
  side: its secrets whose every phase meets `PhaseGood`, at that phase's statements `x z i` and extracted witnesses
  `wit z i`, number at most `#Ss · ∏_c (cap c + 1) · ∏_i 2^free_i · charge_i · Dj^k_i`, `charge_i` the product over
  the phase's statements of `#faultySets · 2^β` (`Phase.charge`). -/
def PeriodLeakTied : Prop :=
  ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Claims.Msg → Claims.Msg) (D : Deploy Coin Col T) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (Dj : ℕ)
    (place : Placement) (hsr : Assumptions.SharedRootsParsed H D place) (hb : D.commit.Binds H)
    {n : ℕ} (pw : Fin n → Phase) (t : Fin n → Ties) (Pw : Fin n → List Partition)
    (hPw : ∀ i, List.Forall₂ (fun s P => D.partition s = some P) (pw i).stmts (Pw i))
    (δ : Fin n → ℕ → ℝ) (hδ : ∀ i j, 0 ≤ δ i j) (reg : Fin n → Finset ℕ) (ρ : Fin n → ℕ → Bool)
    (hord : ∀ i, RowsTiesOrdered (pw i) lay (t i) (k (pw i).scan)) (free : Fin n → ℕ)
    (hcount : ∀ i, CircuitCounts (pw i) lay (t i) (k (pw i).scan) (reg i) (free i)) (β : Fin n → ℕ → ℕ)
    (hβ : ∀ i j s P, (pw i).stmts[j]? = some s → (Pw i)[j]? = some P →
      ∀ F ∈ faultySets s.design P (δ i j), faultyBits s.prog P F ≤ β i j)
    (op : Fin n → ℕ → List (Obs L K S) → List Bool)
    {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : TapRecord Conn Cell (Fin n → List (Obs L K S))) (tap : Dev)
    (run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => tapCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (WardenSchedule Conn)) (hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c)
    (hfit : ∀ s ∈ Ss, f.fits s)
    (Zs : Set Z) (schedOf : Z → WardenSchedule Conn) (srvOf : Z → Srv) (hs : ∀ z ∈ Zs, schedOf z ∈ Ss)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : ∀ z ∈ Zs, dec (run tap (schedOf z, srvOf z)).1 = z)
    (x : Z → Fin n → List Claims.Msg) (wit : Z → Fin n → List Wit) (tops : AuditStatement → List (Opening Bool)),
    (∀ i (ph : Phase) (xs : List Claims.Msg) (ws : List Wit), SamePhase D lay k (pw i) ph →
      List.Forall₂ (fun x s => D.parse x = some s) xs ph.stmts → ws.length = xs.length →
      RowsStateHeld ph place (t i) → RowsProgWired ph place (t i) →
      (∀ j x w, xs[j]? = some x → ws[j]? = some w →
        RWired H D.commit D.parse D.partition (fun s => (lay s).wires) (δ i j) x w) →
      RLeakTied H D.commit D.parse D.partition L K S lay (k ph.scan) ph (t i) (δ i)
          (AsReg ph.prog (ph.layout lay (t i)) (k ph.scan) (reg i)) Dj (tops ph.scan) xs ws ∨
        Claims.Sampling.LogColl (phaseLog H D.commit ph ws ++ tapLog H D.commit ph.scan.roots (tops ph.scan))) ∧
    ((∀ z ∈ Zs, ∀ i x₀ s, (x z i).head? = some x₀ → D.parse x₀ = some s →
        TapOpens H D.commit (lay s) (k s) (op i) s (tops s) ((run tap (schedOf z, srvOf z)).2 i)) →
      {z | z ∈ Zs ∧ ∀ i, PhaseGood H D L K S lay k (δ i) Dj (pw i) (t i) (reg i) (ρ i) tops (x z i) (wit z i)
          ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤
        Ss.encard * (((∏ c, (cap c + 1)) * ∏ i, (2 ^ free i * (pw i).charge (Pw i) (δ i) (β i) *
          Dj ^ k (pw i).scan) : ℕ) : ℕ∞))

end NetTiming.Guarantees
