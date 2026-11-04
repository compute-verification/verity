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
  `#Ss · ∏_c (cap_c + 1) · ∏_i 2^free_i · #faultySets_i(δ_0) · 2^(β_i + Σ_j βt_ij) · Dj^k_i`, `free_i` the period
  check's count of phase `i`'s free inputs (`CircuitCounts`), `β_i` bounding the committed bits of the scan's incorrect
  sets (as `PeriodLeakAccepted`'s `β`) and `βt_ij` the gates tied audit `j` shows the rest of the phase when it can be
  wrong (`TiedExposed`). A tied audit's choice of incorrect set is not charged: the rest of the phase reads it only at
  the gates it shows.

The union of the sessions' and the windows' failures over the monitoring period's live coins is not stated here, nor is
the finder for the collision over two sessions' extractions.
-/

namespace NetTiming.Guarantees

open Claims.Sampling Finset

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak through the network certifier over a monitoring period of rows-form phases.** At every deployment `D` of
the sampled verifier whose parse puts each tie's two ends at one position (`Assumptions.SharedRootsParsed`, at the
registrations' placement `place`) and whose trace commitment binds (`CommitmentScheme.Binds`), over a monitoring period
of `n` phases registered as `pw i` (ties `t i`, the phase's statement `j` read at `δ j`, the scan's partition `Pw i`
with `β i` bounding the committed bits of each of its possible incorrect sets, each tied audit `j` that can be wrong
showing the rest of the phase at most `βt i j` gates (`TiedExposed`), the joint Program's registered inputs `reg i` at
the values `ρ i`, and operator `op i`), where each phase's layout fits its joint Program (`RowsTiesOrdered`) and the
period check counts at least its free inputs less the registered ones (`CircuitCounts`, at `free i`), with the network
certifier's openings `tops s` at each statement, and a network certifier that runs its code by `trusted-host` on
schedules in `Ss`, each held by its record format `f` and giving connection `c` at most `cap c` cells:

* at every phase whose statements' bytes and traces parse to a phase registered as `pw i` (`PhaseParsed`) with its ties
  placed together (`RowsStateHeld`, `RowsProgWired`), traces at which each statement meets the sampling relation at its
  own incorrect set (`RWired`) meet the phase's leak relation at the network certifier's rows (`RLeakTied`), or the `H`
  calls that check their openings and the network certifier's collide (`LogColl`);
* where at every phase's scan statement `x z i` the network certifier registers its observations at that phase's
  operator (`TapOpens`), a covert sender in the server, whose secrets a decoder recovers from the client's view of the
  server side: its secrets whose every phase meets `PhaseGood`, at that phase's statements, the scan's `x z i` and the
  tied audits' `xt z i`, and extracted witnesses `wit z i`, number at most
  `#Ss · ∏_c (cap c + 1) · ∏_i 2^free_i · #faultySets_i · 2^(β_i + Σ_j βt_ij) · Dj^k_i`. -/
def PeriodLeakTied : Prop :=
  ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Claims.Msg → Claims.Msg) (D : Deploy Coin Col T) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (Dj : ℕ)
    (place : Placement) (hsr : Assumptions.SharedRootsParsed H D place) (hb : D.commit.Binds H)
    {n : ℕ} (pw : Fin n → Phase) (t : Fin n → Ties) (Pw : Fin n → Partition)
    (δ : ℕ → ℝ) (hδ : ∀ j, 0 ≤ δ j) (reg : Fin n → Finset ℕ) (ρ : Fin n → ℕ → Bool)
    (hord : ∀ i, RowsTiesOrdered (pw i) lay (t i) (k (pw i).scan)) (free : Fin n → ℕ)
    (hcount : ∀ i, CircuitCounts (pw i) lay (t i) (k (pw i).scan) (reg i) (free i)) (β : Fin n → ℕ)
    (hPw : ∀ i, D.partition (pw i).scan = some (Pw i))
    (hβ : ∀ i, ∀ F ∈ faultySets (pw i).scan.design (Pw i) (δ 0), faultyBits (pw i).scan.prog (Pw i) F ≤ β i)
    (βt : Fin n → ℕ → ℕ) (hexp : TiedExposed D lay δ pw t βt)
    (op : Fin n → ℕ → List (Obs L K S) → List Bool)
    {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : TapRecord Conn Cell (Fin n → List (Obs L K S))) (tap : Dev)
    (run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => tapCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (WardenSchedule Conn)) (hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c)
    (hfit : ∀ s ∈ Ss, f.fits s)
    (Zs : Set Z) (schedOf : Z → WardenSchedule Conn) (srvOf : Z → Srv) (hs : ∀ z ∈ Zs, schedOf z ∈ Ss)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : ∀ z ∈ Zs, dec (run tap (schedOf z, srvOf z)).1 = z)
    (x : Z → Fin n → Claims.Msg) (xt : Z → Fin n → List Claims.Msg) (wit : Z → Fin n → List Wit)
    (tops : AuditStatement → List (Opening Bool)),
    (∀ i (ph : Phase) (xs : List Claims.Msg) (ws : List Wit), SamePhase D lay k (pw i) ph → PhaseParsed D ph xs ws →
      RowsStateHeld ph place (t i) → RowsProgWired ph place (t i) →
      (∀ j x w, xs[j]? = some x → ws[j]? = some w →
        RWired H D.commit D.parse D.partition (fun s => (lay s).wires) (δ j) x w) →
      RLeakTied H D.commit D.parse D.partition L K S lay (k ph.scan) ph (t i) δ
          (AsReg ph.prog (ph.layout lay (t i)) (k ph.scan) (reg i)) Dj (tops ph.scan) xs ws ∨
        Claims.Sampling.LogColl (phaseLog H D.commit ph ws ++ tapLog H D.commit ph.scan.roots (tops ph.scan))) ∧
    ((∀ z ∈ Zs, ∀ i s, D.parse (x z i) = some s →
        TapOpens H D.commit (lay s) (k s) (op i) s (tops s) ((run tap (schedOf z, srvOf z)).2 i)) →
      {z | z ∈ Zs ∧ ∀ i, PhaseGood H D L K S lay k δ Dj (pw i) (t i) (reg i) (ρ i) tops (x z i :: xt z i) (wit z i)
          ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤
        Ss.encard * (((∏ c, (cap c + 1)) * ∏ i, (2 ^ free i * (faultySets (pw i).scan.design (Pw i) (δ 0)).card *
          2 ^ (β i + ∑ j ∈ range (pw i).tied.length, βt i j) * Dj ^ k (pw i).scan) : ℕ) : ℕ∞))

end NetTiming.Guarantees
