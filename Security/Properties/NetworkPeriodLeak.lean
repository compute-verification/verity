import NetworkCheck.Period
import CoreProtocol.Site
import Definitions.NetworkCertifier.Period
import Definitions.NetworkCertifier.RecordRows
import Definitions.Core.Assumptions
import Mathlib.Data.Set.Card
import Mathlib.Algebra.BigOperators.Fin

/-!
# The leak through the network certifier over a monitoring period, on the period check's acceptance

A monitoring period is `n` windows, each audited by the sampled verifier at a statement it registered (`sw i`, with its
partition `Pw i` and the values `ρ i` the registration commits at its registered inputs), and its link-windows'
records are the network certifier's frames in the rows format. `PeriodLeakRegistered` composes the sampled leak bound at
the network certifier's rows, with the registered inputs at the values the registration opens under the statement's
roots (`RegOpens`), with the rows format's read-back, from the record alone, of the client's view, which is what
crossed the wire (`NetTiming.laneWire`), padding included:

* each window's audit, at a statement in its scope whose registration opens its registered inputs (`RegOpensAt`), has
  `Pr[acc ∧ ¬RLeakTap δ] ≤ δ + ε_ks + ε_bind + Pr[B_tap] + Pr[B_reg]` (the errors of sampling at the incorrect set,
  `Claims.Sampling.KSBy` at the wired check `wiredOk`, and the collisions between the extracted trace's openings and
  the network certifier's, `B_tap`, and the registration's, `B_reg`, under `CommitmentScheme.Binds`);
* where at every window's statement the network certifier registers its observations at that window's operator
  (`CertifiersOpen`), a covert sender's secrets whose every window meets `WindowGoodReg` number at most
  `#Θ · ∏_i 2^#(freeInputs_i \ reg_i) · #faultySets_i(δ) · 2^β_i · Dj^k_i`: the sender moves at most
  `log2 #Θ + Σ_i (#(freeInputs_i \ reg_i) + log2 #faultySets_i(δ) + β_i + k_i·log2 Dj)` bits through the network
  certifier in the monitoring period.

The rows format is the network certifier's own (`NetTiming.cellRows`): frames of the schedule's `B` bytes for its
`lanes · windows · window` connections, which read back (`NetTiming.RowsFormat.Wf`) when both are under 2^32. That is a
decidable condition on public parameters, so the period check refuses a schedule that fails it
(`NetworkCheck.Period.rowsWf`, `rowsWfCheck`), and the statement takes the check's acceptance where
`NetTiming.RowsFormat.Wf` was a premise: whatever rows-form period the check accepts, its frames read back.

The schedule `schedOf z` the network certifier runs for a secret `z`, the client's connections' arrivals (their
admissions and cells), is any function of `z`: the client's strategy run against the node's egress, which the secret
steers, never a list of arrivals fixed across secrets. The client's randomness is fixed, or quantified independently
of `z`: a strategy under randomness drawn before `z` is one function of the client's view, and the bound holds at
every one (`Proofs.Properties.NetworkPeriodLeak`'s `periodLeakRegistered_strategy`).

The network certifier's LMS parameters, key and session, against which the check verifies its certificates, are the
period's `p.lms`, `p.key` and `p.session`. The statement says where they come from: the auditor's record of its
network certifier's deployment (`NetTiming.NetworkCertifierRecord`, `certifier-key/v1`), whose key the auditor's
registration `A` holds for the device `cert` (`Core.Site`'s `Owners`: `A.key x.id = some x.key`), the device whose
run `trusted-host` covers. `A` is read from the auditor's record `aj` as the checks read it
(`Core.Site.Registration.Auditor.ofJson`, the registration's `auditor`), never supplied beside it.

Its other conditions are named: the windows' registrations (`WindowsPlaced`), the sender's schedules among those at
`Θ` that fit the format (`SchedulesCovered`, a property of the compiled Lean audit over the network certifier's signed
records) and the sender's secrets recovered from the client's view (`SenderDecodes`, the count's own terms).
`RegOpensAt` restricts which openings `ops` the first part speaks of, those that open the registered roots: a choice of
`ops`, not a claim about the prover's honesty. Two different openings of one registered input are a collision `B_reg`
covers. A wired gate is no free input, so `freeInputs_i` leaves out the wired rows, while a free input that writes a
wire stays in it; the sets `faultySets_i(δ)` counts are incorrect sets, the faulty and the wire-wrong proof blocks, and
each incorrect proof block is charged in full, at all its committed gates (`faultyBits`).

`PeriodLeakRegisteredNontrivial` says the per-window bound is not true of every deployment: some deployment, layout,
statement and prover break it.
-/

namespace NetTiming

/-- **The auditor's record of its network certifier's deployment** (`certifier-key/v1`): the LMS parameters, public key
and session it read off its own device, never a field of the evidence or the prover's copy of the run. The period
check's `Params` carry them (`NetworkCheck.Period.Params`: its key, LMS parameters and session are this record's). -/
structure NetworkCertifierRecord where
  lms : Lms.Params
  key : ByteArray
  session : ByteArray

end NetTiming

namespace Properties.NetworkPeriodLeak

open NetTiming Claims.Sampling Finset

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak through the network certifier over a monitoring period, on the period check's acceptance.** For every
rows-form period the check accepts, on its schedule `s` over `n = p.windows` link-windows, the network certifier
recording its `N = lanes · n · window` connections in its frames of `B = s.cellBytes` bytes (`NetTiming.cellRows`, on
the egress link `link`, `window` buckets a window, with any jitter classes and statuses), and at every deployment `D`
of the sampled verifier where sampling at the incorrect set holds at the layout's wiring (`KSBy` at the wired check)
and the trace's commitment binds (`CommitmentScheme.Binds`), with a decoder that lists every trace within Flock's
radius, the windows registered as `sw i` (`WindowsPlaced`: partition `Pw i`, and `β i` bounding the committed bits of
each of its possible incorrect sets), the network certifier's openings `tops s` at each statement, and a network
certifier `cert` that runs its code by `trusted-host`, its one run giving what crosses the wire to the client, padding
included (`laneWire`), and its record of it, whose LMS parameters, key and session are the auditor's record `r` of it,
the key the auditor's registration `A` (read from its record `aj`) holds for `cert`:

* each window's audit, at a statement in its scope whose registration opens its registered inputs (`RegOpensAt`), has
  `Pr[acc ∧ ¬RLeakTap δ] ≤ δ + ε_ks + ε_bind + Pr[B_tap] + Pr[B_reg]` with the registered inputs at the registration's
  values;
* where at every window's statement the network certifier registers its observations at that window's operator
  (`CertifiersOpen`), a covert sender's secrets whose every window meets `WindowGoodReg`, on schedules among those whose
  `(r, D)` is in the registered set `Θ` and which fit the format (`SchedulesCovered`), recovered from the client's view
  (`SenderDecodes`), number at most `#Θ · ∏_i 2^#(freeInputs_i \ reg_i) · #faultySets_i(δ) · 2^β_i · Dj^k_i`.

The schedule `schedOf z`, the client's arrivals, is any function of the secret: the client's strategy against the
node's egress, under randomness fixed or quantified independently of `z`, never arrivals fixed across secrets. -/
def PeriodLeakRegistered : Prop :=
  ∀ (p : NetworkCheck.Period.Params) (c : NetworkCheck.Period.Commitment) (e : NetworkCheck.Period.Evidence)
    (pub : NetworkCheck.Period.Public) (prof : NetworkCheck.Period.Profile),
    NetworkCheck.Period.check p c e = .accept pub prof →
    ∀ (s : NetworkCheck.Period.Schedule) (sd : ByteArray) (scan : NetworkCheck.Period.Scan),
    p.form = .rows s sd scan →
    ∀ (aj : Lean.Json) (A : Core.Site.Registration.Auditor) (r : NetTiming.NetworkCertifierRecord),
    Core.Site.Registration.Auditor.ofJson aj = .ok A →
    p.lms = r.lms ∧ p.key = r.key ∧ p.session = r.session →
    ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
      (H : Claims.Msg → Claims.Msg) (D : Deploy Coin Col T) (L K S : ℕ)
      (lay : AuditStatement → BoundaryGates L S) (hks : KSBy H D (wiredOk fun s => (lay s).wires))
      (k : AuditStatement → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
      (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (reg : AuditStatement → Finset ℕ)
      (ops : AuditStatement → List (Opening Bool)) (Dj : ℕ)
      (sw : Fin p.windows → AuditStatement) (Pw : Fin p.windows → Partition)
      (ρ : Fin p.windows → ℕ → Bool) (op : Fin p.windows → ℕ → List (Obs L K S) → List Bool) (β : Fin p.windows → ℕ)
      (hplaced : WindowsPlaced D δ sw Pw β)
      {N : ℕ} (hN : N = s.lanes * (p.windows * s.window)) {Srv Z : Type}
      (arr : Srv → Fin N → Arrivals (FrameCell (s.cellBytes - 9))) (link : Fin L)
      (jitter : WardenSchedule (Fin N) → (Fin N → Arrivals (FrameCell (s.cellBytes - 9))) → ℕ → ℕ)
      (status : WardenSchedule (Fin N) → (Fin N → Arrivals (FrameCell (s.cellBytes - 9))) → ℕ → Fin K → Status)
      (cert : Core.Site.Loc) (hkey : A.key cert.id = some (Flock.hexOf r.key))
      (run : Core.Site.Loc → WardenSchedule (Fin N) × Srv →
        (Fin N → List (ℕ × CellView (FrameCell (s.cellBytes - 9)))) × (Fin p.windows → List (Obs L K S)))
      (htrust : Verity.Assumptions.TrustedHost cert run fun q =>
        (laneWire q.1 (wardenCode q.1 (arr q.2)),
          (cellRows s.cellBytes N link s.window jitter status).record S p.windows q.1 (arr q.2)
            (wardenCode q.1 (arr q.2))))
      (Θ : Set (ℕ × ℕ)) (Zs : Set Z) (schedOf : Z → WardenSchedule (Fin N)) (srvOf : Z → Srv)
      (hcov : SchedulesCovered
        {σ | (σ.r, σ.D) ∈ Θ ∧ (cellRows s.cellBytes N link s.window jitter status).Fits S p.windows σ} Zs schedOf)
      (dec : (Fin N → List (ℕ × CellView (FrameCell (s.cellBytes - 9)))) → Z)
      (hdz : SenderDecodes run cert Zs schedOf srvOf dec)
      (x : Z → Fin p.windows → Claims.Msg) (wit : Z → Fin p.windows → Wit) (tops : AuditStatement → List (Opening Bool))
      (hb : D.commit.Binds H),
      (∀ m, ScopeAcc D lay k m → RegOpensAt H D reg ops m →
        ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
        Claims.PrBad H D.V
            (RLeakTap H D.commit D.parse D.partition L K S lay k δ
              (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (regVal (ops s))) Dj tops)
            (E D decode) m P ≤
          κ D δ m + Real.sqrt (γ D m * Claims.PrB H D.V P m) + (Bext H D.V (E D decode) Ω P m).Pr H +
            (Btap H D.V (E D decode) D.commit D.parse tops Ω P m).Pr H +
            (Breg H D.V (E D decode) D.commit D.parse ops Ω P m).Pr H) ∧
        (CertifiersOpen H D lay k op tops Zs x (fun z => (run cert (schedOf z, srvOf z)).2) →
          {z | z ∈ Zs ∧ ∀ i, WindowGoodReg H D L K S lay k δ reg ops Dj (sw i) (ρ i) tops (x z i) (wit z i)
              ((run cert (schedOf z, srvOf z)).2 i)}.encard ≤
            Θ.encard * ((∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
              reg (sw i)).card * (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i)) : ℕ) : ℕ∞))

/-- **`PeriodLeakRegistered`'s per-window bound can fail**: some deployment of the sampled verifier, layout, decoder,
registration and prover has `Pr[acc ∧ ¬RLeakTap δ]` above `δ + ε_ks + ε_bind + Pr[B_tap] + Pr[B_reg]` at one
statement, so the bound is not true of every deployment. -/
def PeriodLeakRegisteredNontrivial : Prop :=
  ∃ (H : Claims.Msg → Claims.Msg) (D : Deploy Unit Unit Unit) (lay : AuditStatement → BoundaryGates 1 1)
    (k : AuditStatement → ℕ) (δ : ℝ) (decode : List (ℕ × Unit) → List Unit) (reg : AuditStatement → Finset ℕ)
    (ops tops : AuditStatement → List (Opening Bool)) (m : Claims.Msg) (P : Claims.Prover Unit),
    ¬ Claims.PrBad H D.V
        (RLeakTap H D.commit D.parse D.partition 1 1 1 lay k δ
          (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (regVal (ops s))) 1 tops)
        (E D decode) m P ≤
      κ D δ m + Real.sqrt (γ D m * Claims.PrB H D.V P m) + (Bext H D.V (E D decode) Unit P m).Pr H +
        (Btap H D.V (E D decode) D.commit D.parse tops Unit P m).Pr H +
        (Breg H D.V (E D decode) D.commit D.parse ops Unit P m).Pr H

end Properties.NetworkPeriodLeak
