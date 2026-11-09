import Properties.NetworkPeriodLeak
import Proofs.Properties.NetworkPeriod
import Proofs.NetworkCertifier.PeriodReg
import Proofs.NetworkCertifier.WitnessToys

/-!
# The leak through the network certifier over a monitoring period, on the period check's acceptance: its proofs

`periodLeakRegistered` proves `Properties.NetworkPeriodLeak.PeriodLeakRegistered`: a rows-form period the check accepts
passes `rowsWf` (`Properties.NetworkPeriod.rowsWf_of_accept`), so the network certifier's frames read back at its
schedule (`Properties.NetworkPeriod.cellRows_wf_of_rowsWf`: `NetTiming.RowsFormat.Wf` at `NetTiming.cellRows`), and
the bound at every rows format that reads back (`NetTiming.periodLeak_registered`) gives it there.

`periodLeakRegistered_strategy`, unlisted, is it with the client's arrivals its strategy: the schedule the network
certifier runs for the secret `z` is `strategy (view z)`, a function of what the client saw of the node's egress, which
the secret steers. `rowsWf_served`, unlisted, is the deployment's side of the format: the served schedule `w63` passes
`rowsWf` over every period of fewer than 2^25 link-windows, so the check doesn't refuse it there and its frames read
back.

`periodLeakRegistered_nontrivial`, in the lock's `witnesses`: the per-window bound fails at the accepting verifier
`tapD`, at SHA-512, with no input registered and a registration with no openings
(`NetTiming.Witness.tapD_reg_bound_fails`). Its `honest` half, which the lock's `unwitnessed` exempts, is the sampled
leg's guarantee at the honest deployment.
-/

namespace Properties.NetworkPeriodLeak

open NetTiming Claims.Sampling Finset

/-- **The leak through the network certifier over a monitoring period, on the period check's acceptance**: the check's
acceptance gives `rowsWf` (`rowsWf_of_accept`), so the network certifier's frames read back
(`cellRows_wf_of_rowsWf`), and `NetTiming.periodLeak_registered` bounds the leak at that format. -/
theorem periodLeakRegistered : PeriodLeakRegistered := by
  intro p c e pub prof h s sd scan hform aj A r hA hr Coin Col T _ _ _ _ _ H D L K S lay hks k δ decode hδ hdec reg ops
    Dj sw Pw ρ op β hplaced N hN Srv Sec arr link jitter status cert hkey run htrust Θ Zs schedOf srvOf hcov dec hdz x
    wit tops hb
  subst hN
  exact NetTiming.periodLeak_registered H D L K S lay hks k δ decode hδ hdec reg ops Dj sw Pw ρ op β hplaced arr _
    (Properties.NetworkPeriod.cellRows_wf_of_rowsWf (Properties.NetworkPeriod.rowsWf_of_accept h hform) link s.window
      jitter status) cert run htrust Θ Zs schedOf srvOf hcov dec hdz x wit tops hb

/-- **The client's arrivals as its strategy**: `PeriodLeakRegistered` with the schedule the network certifier runs for
the secret `z`, each connection's admission and cells, the client's strategy `strategy` at what it saw, `view z`: a
function of its view of the node's egress, which the secret steers, and so of `z`, never arrivals fixed across secrets.
The client's randomness is fixed before `z`: a strategy that draws it is one `strategy` for each draw, and the bound
holds at each. -/
theorem periodLeakRegistered_strategy (p : NetworkCheck.Period.Params) (c : NetworkCheck.Period.Commitment)
    (e : NetworkCheck.Period.Evidence) (pub : NetworkCheck.Period.Public) (prof : NetworkCheck.Period.Profile)
    (h : NetworkCheck.Period.check p c e = .accept pub prof) (s : NetworkCheck.Period.Schedule) (sd : ByteArray)
    (scan : NetworkCheck.Period.Scan) (hform : p.form = .rows s sd scan) (aj : Lean.Json)
    (A : Core.Site.Registration.Auditor) (r : NetTiming.NetworkCertifierRecord)
    (hA : Core.Site.Registration.Auditor.ofJson aj = .ok A)
    (hr : p.lms = r.lms ∧ p.key = r.key ∧ p.session = r.session)
    {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Claims.Msg → Claims.Msg) (D : Deploy Coin Col T) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (hks : KSBy H D (wiredOk fun s => (lay s).wires))
    (k : AuditStatement → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (reg : AuditStatement → Finset ℕ)
    (ops : AuditStatement → List (Opening Bool)) (Dj : ℕ)
    (sw : Fin p.windows → AuditStatement) (Pw : Fin p.windows → Partition)
    (ρ : Fin p.windows → ℕ → Bool) (op : Fin p.windows → ℕ → List (Obs L K S) → List Bool) (β : Fin p.windows → ℕ)
    (hplaced : WindowsPlaced D δ sw Pw β)
    {N : ℕ} (hN : N = s.lanes * (p.windows * s.window)) {Srv Sec V : Type}
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
    (Θ : Set (ℕ × ℕ)) (Zs : Set Sec) (view : Sec → V) (strategy : V → WardenSchedule (Fin N)) (srvOf : Sec → Srv)
    (hcov : SchedulesCovered
      {σ | (σ.r, σ.D) ∈ Θ ∧ (cellRows s.cellBytes N link s.window jitter status).Fits S p.windows σ} Zs
      fun z => strategy (view z))
    (dec : (Fin N → List (ℕ × CellView (FrameCell (s.cellBytes - 9)))) → Sec)
    (hdz : SenderDecodes run cert Zs (fun z => strategy (view z)) srvOf dec)
    (x : Sec → Fin p.windows → Claims.Msg) (wit : Sec → Fin p.windows → Wit)
    (tops : AuditStatement → List (Opening Bool))
    (hb : D.commit.Binds H) :
    (∀ m, ScopeAcc D lay k m → RegOpensAt H D reg ops m →
      ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
      Claims.PrBad H D.V
          (RLeakTap H D.commit D.parse D.partition L K S lay k δ
            (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (regVal (ops s))) Dj tops)
          (E D decode) m P ≤
        κ D δ m + Real.sqrt (γ D m * Claims.PrB H D.V P m) + (Bext H D.V (E D decode) Ω P m).Pr H +
          (Btap H D.V (E D decode) D.commit D.parse tops Ω P m).Pr H +
          (Breg H D.V (E D decode) D.commit D.parse ops Ω P m).Pr H) ∧
      (CertifiersOpen H D lay k op tops Zs x (fun z => (run cert (strategy (view z), srvOf z)).2) →
        {z | z ∈ Zs ∧ ∀ i, WindowGoodReg H D L K S lay k δ reg ops Dj (sw i) (ρ i) tops (x z i) (wit z i)
            ((run cert (strategy (view z), srvOf z)).2 i)}.encard ≤
          Θ.encard * ((∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
            reg (sw i)).card * (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i)) : ℕ) : ℕ∞)) :=
  periodLeakRegistered p c e pub prof h s sd scan hform aj A r hA hr H D L K S lay hks k δ decode hδ hdec reg ops Dj sw
    Pw ρ op β hplaced hN arr link jitter status cert hkey run htrust Θ Zs (fun z => strategy (view z)) srvOf hcov dec
    hdz x wit tops hb

/-- **The served schedule passes `rowsWf`**: `w63`'s, 4 lanes of 32-bucket link-windows and 512-byte cells, over any
period of fewer than 2^25 link-windows, so the period check doesn't refuse it there (`rowsWfCheck`) and the network
certifier's frames read back (`cellRows_wf_of_rowsWf`). -/
theorem rowsWf_served {windows : ℕ} (h : windows < 2 ^ 25) :
    NetworkCheck.Period.rowsWf ⟨100, 512, 27, 65, 16, 12, 47, 4, 4, 4, 32⟩ windows = true := by
  have h25 : (2 : ℕ) ^ 25 = 33554432 := rfl
  have h32 : (2 : ℕ) ^ 32 = 4294967296 := rfl
  rw [h25] at h
  simp only [NetworkCheck.Period.rowsWf, Bool.and_eq_true, decide_eq_true_eq, h32]
  omega

open Claims Claims.Witness NetTiming.Witness in
/-- **`PeriodLeakRegistered`'s per-window bound fails at the accepting verifier** `tapD`, at SHA-512, at one step at
`δ = 1/8` with no input registered and a registration with no openings, which opens every registered input
(`tapD_reg_bound_fails`). -/
theorem periodLeakRegistered_nontrivial : PeriodLeakRegisteredNontrivial :=
  ⟨FlockDeploy.sha512, tapD Unit, fun _ => notLay, fun _ => 1, 1 / 8, fun _ => [()], fun _ => ∅, fun _ => [],
    fun _ => tapOps, [], silentP, tapD_reg_bound_fails FlockDeploy.sha512⟩

end Properties.NetworkPeriodLeak
