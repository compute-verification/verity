import Definitions.NetworkCertifier.TiedLeak
import Definitions.NetworkCertifier.Tap
import Definitions.Core.Assumptions
import Mathlib.Data.Set.Card
import Mathlib.Algebra.BigOperators.Fin

/-!
# The leak through the network certifier over a monitoring period of rows-form phases: the tie and the count

`PeriodLeakRegistered` with each window a rows-form phase, counted as its joint Program (`Phase.prog`): its scan's and
its tied compute audits' Programs side by side, with the ties among its wires (`Definitions.NetworkCertifier.Tied`). A
tied gate is no free input, so the content term is the joint Program's free inputs less the registered ones, which the
period check's count bounds (`CircuitCounts`). The registered inputs are pinned at the values the registration opens
(`Phase.regVal`), and a phase's good event has its state reads at row `0` among them at the empty state
(`RowsStateEmpty`).

`PeriodTiedCount` is the part that holds on every run, in two parts:

* **The tie.** At a phase registered as `pw i` whose ties are placed together (`RowsStateHeld`, `RowsProgWired`) and
  whose statements' registrations open their registered inputs (`RegOpensAt`), the sampling relation at each
  statement's own incorrect set (`RWired`, at its own wires and its `δ`) gives the phase's leak relation at the network
  certifier's rows with the registered inputs at the registration's values (`RLeakTied`, at `AsReg`), or two of the
  traces' openings, or one of them and one of the registration's or the network certifier's, collide. The tie holds at
  whatever values the registration opens, so it needs no empty state. A tie is right by binding: its two ends are
  committed as one bit of one leaf under one root (`SharedRootsParsed`, at the deployed parse), and a trace's transfer
  gate is the network certifier's (`CommitmentScheme.Binds`). So a wrong tie is no new error: each statement fails
  `RWired` with probability at most `δ + ε_ks + ε_bind` at its session (`KSWired`), and a collision is found on the
  openings the extractions and the registration check.
* **The count.** Where the network certifier registers its observations at every phase's scan, the secrets of a covert
  sender whose every phase meets its good event (`PhaseGood`) number at most
  `#Ss · ∏_c (cap_c + 1) · ∏_i 2^free_i · #faultySets_i(δ_0) · 2^(β_i + Σ_j βt_ij) · Dj^k_i`, `free_i` the period
  check's count of phase `i`'s free inputs less the registered ones (`CircuitCounts`), `β_i` bounding the committed bits
  of the scan's incorrect sets (`WindowsPlaced`) and `βt_ij` the gates tied audit `j` shows the rest of the phase when it
  can be wrong (`TiedExposed`). A tied audit's choice of incorrect set is not charged: the rest of the phase reads it
  only at the gates it shows.

The probability over the monitoring period's live coins is not stated here: each statement's session's failure, and the
phase's collisions over two sessions' extractions, whose finder is not stated either.
-/

namespace NetTiming.Guarantees

open Claims.Sampling Finset

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak through the network certifier over a monitoring period of rows-form phases: the tie and the count.** At
every deployment `D` of the sampled verifier whose parse puts each tie's two ends at one bit (`SharedRootsParsed`, at the
registrations' placement `place`) and whose trace commitment binds (`CommitmentScheme.Binds`), with the registered
inputs `reg s` of each statement and the registration's openings `ops s`, over a monitoring period of `n` phases
registered as `pw i` (ties `t i`, the phase's statement `j` read at `δ j`, the scans placed with partitions `Pw i` and
`β i` bounding the committed bits of each of their possible incorrect sets (`WindowsPlaced`), each tied audit `j` that
can be wrong showing the rest of the phase at most `βt i j` gates (`TiedExposed`), the registration opening `ρ i` at
the joint Program's registered free inputs, and operator `op i`), where each phase's layout fits its joint Program
(`RowsTiesOrdered`) and the period check counts at least its free inputs less the registered ones (`CircuitCounts`, at
`free i`), with the network certifier's openings `tops s` at each statement, and a network certifier that runs its code
by `trusted-host` on schedules in `Ss` its record format holds (`SchedulesHeld`), the sender's among them
(`SchedulesCovered`):

* at every phase whose statements' bytes and traces parse to a phase registered as `pw i` (`PhaseParsed`), with its ties
  placed together (`RowsStateHeld`, `RowsProgWired`) and its statements' registrations opening their registered inputs
  (`RegOpensAt`), traces at which
  each statement meets the sampling relation at its own incorrect set (`RWired`) meet the phase's leak relation at the
  network certifier's rows with the registered inputs at the registration's values (`RLeakTied`, at `AsReg` of
  `Phase.regVal`), or the `H` calls that check their openings, the registration's and the network certifier's collide
  (`LogColl`);
* where at every phase's scan statement `x z i` the network certifier registers its observations at that phase's
  operator (`TapsOpen`), a covert sender in the server, whose secrets a decoder recovers from the client's view of the
  server side (`SenderDecodes`): its secrets whose every phase meets `PhaseGood`, at that phase's statements, the scan's
  `x z i` and the tied audits' `xt z i`, and extracted witnesses `wit z i`, number at most
  `#Ss · ∏_c (cap c + 1) · ∏_i 2^free_i · #faultySets_i · 2^(β_i + Σ_j βt_ij) · Dj^k_i`. -/
def PeriodTiedCount : Prop :=
  ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Claims.Msg → Claims.Msg) (D : Deploy Coin Col T) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (Dj : ℕ)
    (place : GatePlaces) (hsr : SharedRootsParsed H D place) (hb : D.commit.Binds H)
    (reg : AuditStatement → Finset ℕ) (ops : AuditStatement → List (Opening Bool))
    {n : ℕ} (pw : Fin n → Phase) (t : Fin n → Ties) (Pw : Fin n → Partition)
    (δ : ℕ → ℝ) (hδ : ∀ j, 0 ≤ δ j) (ρ : Fin n → ℕ → Bool)
    (hord : ∀ i, RowsTiesOrdered (pw i) lay (t i) (k (pw i).scan)) (free : Fin n → ℕ)
    (hcount : ∀ i, CircuitCounts (pw i) lay (t i) (k (pw i).scan) ((pw i).reg reg) (free i)) (β : Fin n → ℕ)
    (hplaced : WindowsPlaced D (δ 0) (fun i => (pw i).scan) Pw β)
    (βt : Fin n → ℕ → ℕ) (hexp : TiedExposed D lay δ pw t βt)
    (op : Fin n → ℕ → List (Obs L K S) → List Bool)
    {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : TapRecord Conn Cell (Fin n → List (Obs L K S))) (tap : Dev)
    (run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => tapCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (WardenSchedule Conn)) (hheld : SchedulesHeld f cap Ss)
    (Zs : Set Z) (schedOf : Z → WardenSchedule Conn) (srvOf : Z → Srv) (hcov : SchedulesCovered Ss Zs schedOf)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : SenderDecodes run tap Zs schedOf srvOf dec)
    (x : Z → Fin n → Claims.Msg) (xt : Z → Fin n → List Claims.Msg) (wit : Z → Fin n → List Wit)
    (tops : AuditStatement → List (Opening Bool)),
    (∀ i (ph : Phase) (xs : List Claims.Msg) (ws : List Wit), SamePhase D lay k reg (pw i) ph → PhaseParsed D ph xs ws →
      RowsStateHeld ph place (t i) → RowsProgWired ph place (t i) → (∀ x ∈ xs, RegOpensAt H D reg ops x) →
      (∀ j x w, xs[j]? = some x → ws[j]? = some w →
        RWired H D.commit D.parse D.partition (fun s => (lay s).wires) (δ j) x w) →
      RLeakTied H D.commit D.parse D.partition L K S lay (k ph.scan) ph (t i) δ
          (fun _ => AsReg ph.prog (ph.layout lay (t i)) (k ph.scan) (ph.reg reg) (ph.regVal ops)) Dj (tops ph.scan)
          xs ws ∨
        Claims.Sampling.LogColl (phaseLog H D.commit ph ws ++ phaseRegLog H D.commit ph ops ++
          tapLog H D.commit ph.scan.roots (tops ph.scan))) ∧
    (TapsOpen H D lay k op tops Zs x (fun z => (run tap (schedOf z, srvOf z)).2) →
      {z | z ∈ Zs ∧ ∀ i, PhaseGood H D L K S lay k δ reg ops Dj (pw i) (t i) (ρ i) tops (x z i :: xt z i) (wit z i)
          ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤
        Ss.encard * (((∏ c, (cap c + 1)) * ∏ i, (2 ^ free i * (faultySets (pw i).scan.design (Pw i) (δ 0)).card *
          2 ^ (β i + ∑ j ∈ range (pw i).tied.length, βt i j) * Dj ^ k (pw i).scan) : ℕ) : ℕ∞))

end NetTiming.Guarantees
