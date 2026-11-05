import Definitions.NetworkCertifier.Guarantees.SampledCharge
import Definitions.NetworkCertifier.Period
import Definitions.NetworkCertifier.Tap
import Definitions.Core.Assumptions
import Mathlib.Data.Set.Card
import Mathlib.Algebra.BigOperators.Fin

/-!
# The leak through the network certifier over a monitoring period of sampled audits

A monitoring period is `n` windows, each audited by the sampled verifier at a statement it registered (`sw i`, with its
partition `Pw i` and the values `ρ i` the registration commits at its registered inputs). The network certifier is one
device on the server's uplink, and it functions as an active network certifier (`Definitions.NetworkCertifier.Tap`): one
run of its code gives the frames the client receives, what crosses the wire with its padding (`laneWire`), and the
observations the audits read, its record of those frames, so on every schedule its record format holds the client's
view reads back from the observations. The network certifier
registers its records at the transfer gates: its openings `tops s` under the statement's roots, its signed root for the
window, so they vary with what the server did
(`TapOpens`). `PeriodLeakAccepted` composes the sampled leak bound at the network certifier's rows (`RLeakTap`, with the
prover's registered inputs pinned: `AsReg`) with that read-back, so the content term of the bound comes from the
audits, not from a premise:

* each window's audit, at a statement in its scope, fails the leak relation at the network certifier's rows with
  probability at most `δ + ε_ks + ε_bind + Pr[B_tap]` at that statement (the errors of sampling at the incorrect set,
  `Claims.Sampling.KSBy` at the wired check `wiredOk`, and `B_tap` the collisions between the extracted trace's openings and the network certifier's, under
  `CommitmentScheme.Binds`);
* where the network certifier registers its observations, the secrets of a covert sender in the server whose every
  window meets its good event (`WindowGood`) number at most
  `#Ss · ∏_c (cap_c + 1) · ∏_i 2^#(freeInputs_i \ reg_i) · #faultySets_i(δ) · 2^β_i · Dj^k_i`: the sender moves at most
  `log2 #Ss + Σ_c log2 (cap_c + 1) + Σ_i (#(freeInputs_i \ reg_i) + log2 #faultySets_i(δ) + β_i + k_i·log2 Dj)` bits
  through the network certifier in the monitoring period.

The union of the windows' failures over the monitoring period's live coins is not stated here. `WindowGood` is the
event the first part bounds, plus one condition no proof here gives: the registration's values at the prover's
registered inputs, which the registration's commitment binds. The network certifier's rows are not pinned: they vary
with the secret. The first part covers an extracted trace that leaves them outside the incorrect proof blocks, where
every transfer gate is opened; the count charges an incorrect proof block's rows through the Program, from the free
inputs, the possible incorrect sets, their committed bits and the timing. A proof block the sample missed is in the
incorrect set.

**Wired rows.** The layout's wiring (`BoundaryGates.wires`) carries each wired input gate's writer's value into it, so a
wired gate is no free input, and the content term `2^#(freeInputs_i \ reg_i)` leaves out the wired rows as it leaves out
the ingress. A proof block that holds or reads a wired row whose committed value is not its writer's is wire-wrong, and
the sets the count ranges over are incorrect sets, the faulty and the wire-wrong proof blocks (`incorrect`): a drawn
proof block's session shows its wired rows right (`KSBy` at `wiredOk`), so an undrawn wire-wrong one is in the incorrect
set at the same error. Each incorrect proof block is charged in full, at all its committed gates (`faultyBits`): its inputs, the
wired rows among them, its returned gates and the gates another proof block reads, so a wrong pass-through proof block,
which returns an input it holds, pays in full too. An incorrect proof block reaches the rest of the circuit only through
its committed gates, each held at one bit: those it returns or another proof block reads, and its inputs, among them a
first copy of an input bit that later copies in other proof blocks read through their wires. A free input that writes a
wire stays a free input.

`Properties.NetworkPeriodLeak.PeriodLeakRegistered` removes that condition. Its first part bounds the leak relation with
the registered inputs at the values the registration opens under the statement's roots (`RegOpens`), adding
`Pr[B_reg]`, and its count is over `WindowGoodReg`, which holds the registration's values, not the trace's, at the
registered inputs. Its network certifier records in its own rows format (`cellRows`), which reads the client's view back
from the record and the schedule's `(r, D)` alone, so its count charges the schedule only for its `(r, D)` among the
registered set `Θ`: `log2 #Θ` bits in place of `log2 #Ss + Σ_c log2 (cap_c + 1)`. It is stated on the period check's
acceptance, which refuses a schedule whose frames don't read back (`NetworkCheck.Period.rowsWf`).
-/

namespace NetTiming.Guarantees

open Claims.Sampling Finset

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak through the network certifier over a monitoring period of accepted sampled audits.** At every
deployment `D` of the sampled verifier where sampling at the incorrect set holds at the layout's wiring
(`Claims.Sampling.KSBy` at the wired check `wiredOk`, which is `Claims.Sampling.KS` with no wires; at a wired layout it
follows from `KS_of`'s hypotheses with `WiresRevealed`, `ksWired_of_revealed`, at the deployed record `WiredReads`,
`ksWired_of_record`, which the verifier of record's check of each session's wired reads is to give) and the trace's
commitment binds
(`CommitmentScheme.Binds`), with a decoder that lists every trace within Flock's radius, over a monitoring period of `n`
windows registered as `sw i` (partition `Pw i`, registered values `ρ i`, operator `op i`, and `β i` bounding the
committed bits of each of its possible incorrect sets), with the network certifier's openings `tops s` at each statement,
and a network certifier that runs its code by `trusted-host` on schedules in `Ss`, each held by its record format `f`
(`TapRecord.fits`) and giving connection `c` at most `cap c` cells, its one run giving the frames the client receives
(`laneWire`) and the observations of the monitoring period's windows (its record of those frames, in the format `f`):

* each window's audit, at a statement in its scope, has `Pr[acc ∧ ¬RLeakTap δ] ≤ δ + ε_ks + ε_bind + Pr[B_tap]`;
* where at every window's statement the network certifier registers its observations at that window's operator
  (`TapOpens`), a covert sender in the server, whose secrets a decoder recovers from the client's view of the server
  side: its secrets whose every window meets `WindowGood`, at that window's statement `x z i` and extracted witness
  `wit z i`, number at most
  `#Ss · ∏_c (cap c + 1) · ∏_i 2^#(freeInputs_i \ reg_i) · #faultySets_i(δ) · 2^β_i · Dj^k_i`.

A wired gate is no free input, so `freeInputs_i` leaves out the wired rows, while a free input that writes a wire stays
in it; and the sets `faultySets_i(δ)` counts are incorrect sets, the faulty and the wire-wrong proof blocks. Each
incorrect proof block is charged in full, at all its committed gates (`faultyBits`): its inputs, the wired rows among
them, its returned gates and the gates another proof block reads, so a wrong pass-through proof block, which returns an
input it holds, pays in full too. An incorrect proof block reaches the rest of the circuit only through its committed
gates, each held at one bit: those it returns or another proof block reads, and its inputs, among them a first copy of
an input bit that later copies in other proof blocks read through their wires. -/
def PeriodLeakAccepted : Prop :=
  ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Claims.Msg → Claims.Msg) (D : Deploy Coin Col T) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (hks : KSBy H D (wiredOk fun s => (lay s).wires))
    (k : AuditStatement → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (reg : AuditStatement → Finset ℕ) (Dj : ℕ)
    {n : ℕ} (sw : Fin n → AuditStatement) (Pw : Fin n → Partition) (hPw : ∀ i, D.partition (sw i) = some (Pw i))
    (ρ : Fin n → ℕ → Bool) (op : Fin n → ℕ → List (Obs L K S) → List Bool) (β : Fin n → ℕ)
    (hβ : ∀ i, ∀ F ∈ faultySets (sw i).design (Pw i) δ, faultyBits (sw i).prog (Pw i) F ≤ β i)
    {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : TapRecord Conn Cell (Fin n → List (Obs L K S))) (tap : Dev)
    (run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => tapCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (WardenSchedule Conn)) (hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c)
    (hfit : ∀ s ∈ Ss, f.fits s)
    (Zs : Set Z) (schedOf : Z → WardenSchedule Conn) (srvOf : Z → Srv) (hs : ∀ z ∈ Zs, schedOf z ∈ Ss)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : ∀ z ∈ Zs, dec (run tap (schedOf z, srvOf z)).1 = z)
    (x : Z → Fin n → Claims.Msg) (wit : Z → Fin n → Wit) (tops : AuditStatement → List (Opening Bool)) (hb : D.commit.Binds H),
    (∀ m, ScopeAcc D lay k m → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
      Claims.PrBad H D.V
          (RLeakTap H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj tops)
          (E D decode) m P ≤
        κ D δ m + Real.sqrt (γ D m * Claims.PrB H D.V P m) + (Bext H D.V (E D decode) Ω P m).Pr H +
          (Btap H D.V (E D decode) D.commit D.parse tops Ω P m).Pr H) ∧
      ((∀ z ∈ Zs, ∀ i s, D.parse (x z i) = some s → TapOpens H D.commit (lay s) (k s) (op i) s (tops s)
          ((run tap (schedOf z, srvOf z)).2 i)) →
        {z | z ∈ Zs ∧ ∀ i, WindowGood H D L K S lay k δ reg Dj (sw i) (ρ i) tops (x z i) (wit z i)
            ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤
          Ss.encard * (((∏ c, (cap c + 1)) * ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
            reg (sw i)).card * (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i)) : ℕ) : ℕ∞))

end NetTiming.Guarantees
