import Proofs.Properties.Verity.Subprotocols
import Proofs.Properties.Verity.Network.Window

/-!
# The network certifier's refinement at the code, from the named steps

`Verity.Subprotocol.NetworkWindowLe R A` (the network certifier's refinement) follows from `Verity.Network.windowLe` at any verdict
`A` under which onsite's network component accepts what the prover sent and S5's guard holds (`networkWindowLe`). In
the game, `A` is onsite's network coupling's verdict at the code (`Vocab.Coupling.verdict`), whose first conjunct is
that acceptance and whose other two, the releases logged and the break absent, are the guard
(`networkWindowLe_verdict`); remote has no network component. So the network certifier at the code
(`Verity.Subprotocol.network_atCode`, at the coupling's run, `Vocab.Coupling.atCode`) rests on the bridge's assumptions
(`KS`, `Commitment.Binds`, `Flock.ListWithin`, and `trusted-host` in the realization) and the named steps
(`network_atCode_of_steps`): S2–S6, the ρ gap and S8's two premises (`Proofs.Properties.Verity.Network.Window`). At a
coupling, that is the per-period guarantee `Vocab.networkSound_of_coupling` reads at onsite
(`coupling_atCode_of_steps`).
-/

namespace Verity.Network

open Accounting Core Claims Claims.Sampling NetTiming Accounting.JointBridge Verity.Subprotocol

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}
  [DecidableEq I] [∀ i, Fintype (Ω i)] {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
  [DecidableEq Col] [Zero Col] {H : Msg → Msg} {D : Deploy Coin Col T} {L Kₒ S : ℕ} {lay : Stmt → Layout L S}
  {k : Stmt → ℕ} {δ : ℝ} {decode : List (ℕ × Col) → List T} {reg : Stmt → Finset ℕ} {Dj n : ℕ}

/-- **The network certifier's refinement at a verdict `A` under which onsite's network component accepts what the prover sent
(`s`) and S5's guard `G` holds, from its named steps.** -/
theorem networkWindowLe {r : PeriodRun d α C Ω} {K : ℝ}
    {R : RealizesCommOfPeriodLeakAccepted r K H D L Kₒ S lay k δ decode reg Dj n}
    {A : Audit → ((i : I) → Ω i) → Prop} {s : ((i : I) → Ω i) → Sent} {G : ((i : I) → Ω i) → Prop}
    (hA : ∀ ξ, A .network ξ → ComponentAccepts .onsite .network (s ξ) ∧ G ξ) (W : WindowRuns R)
    (h2 : PeriodRunsV2 W s) (h3 : V2AcceptMeans W) (h4 : WindowCoinsLive W) (h5 : SessionIsSampledRun W G)
    (h6 : NetworkCouples R s W) (hρ : WindowValuesRegistered R s) (hreg : WindowRegistersAs W s)
    (hcov : RecordCovers W s) :
    NetworkWindowLe R A := fun ξ i =>
  le_trans (Pouw.pr_mono fun _ hx => ⟨(hA _ hx.1).1, (hA _ hx.1).2, hx.2⟩) (windowLe W h2 h3 h4 h5 h6 hρ hreg hcov ξ i)

/-- **The refinement in the game**: the network certifier's refinement at a coupling's verdict at the code, with the steps at what the prover
sent at the outcome the coins lead to, S5 guarded by the verdict's releases logged and break absent. -/
theorem networkWindowLe_verdict {V : Vocab d α} {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)}
    (q : V.Coupling .onsite .network θ F c σ) {K : ℝ}
    {R : RealizesCommOfPeriodLeakAccepted q.run K H D L Kₒ S lay k δ decode reg Dj n} (W : WindowRuns R)
    (h2 : PeriodRunsV2 W fun ξ => (V.outcome θ F c σ q.coins ξ).sent) (h3 : V2AcceptMeans W)
    (h4 : WindowCoinsLive W)
    (h5 : SessionIsSampledRun W fun ξ => V.Logged (q.pos .network) .network (V.outcome θ F c σ q.coins ξ) ∧
      ¬ q.brk ξ)
    (h6 : NetworkCouples R (fun ξ => (V.outcome θ F c σ q.coins ξ).sent) W)
    (hρ : WindowValuesRegistered R fun ξ => (V.outcome θ F c σ q.coins ξ).sent)
    (hreg : WindowRegistersAs W fun ξ => (V.outcome θ F c σ q.coins ξ).sent)
    (hcov : RecordCovers W fun ξ => (V.outcome θ F c σ q.coins ξ).sent) : NetworkWindowLe R q.verdict :=
  networkWindowLe (fun _ h => h) W h2 h3 h4 h5 h6 hρ hreg hcov

/-- **The network certifier at the code from the named steps**: `Verity.Subprotocol.network_atCode` with
`NetworkWindowLe` from `networkWindowLe`. -/
theorem network_atCode_of_steps {r : PeriodRun d α C Ω} (c : Certificates) (p : Price)
    (R : RealizesCommOfPeriodLeakAccepted r c.K H D L Kₒ S lay k δ decode reg Dj n) (hks : KS H D)
    (hb : D.commit.Binds H) (hδ : 0 ≤ δ) (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode)
    (εks εbind εtap : ℝ)
    (herr : ∀ ξ i, Claims.Sampling.κ D δ (R.stmt ξ i) - δ +
      Real.sqrt (Claims.Sampling.γ D (R.stmt ξ i) * PrB H D.V (R.prover ξ i) (R.stmt ξ i)) ≤ εks)
    (hbind : ∀ ξ i, (Bext H D.V (E D decode) (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤ εbind)
    (htap : ∀ ξ i, (Btap H D.V (E D decode) D.commit D.parse R.tops (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤
      εtap)
    {A : Audit → ((i : I) → Ω i) → Prop} {s : ((i : I) → Ω i) → Sent} {G : ((i : I) → Ω i) → Prop}
    (hA : ∀ ξ, A .network ξ → ComponentAccepts .onsite .network (s ξ) ∧ G ξ) (W : WindowRuns R)
    (h2 : PeriodRunsV2 W s) (h3 : V2AcceptMeans W) (h4 : WindowCoinsLive W) (h5 : SessionIsSampledRun W G)
    (h6 : NetworkCouples R s W) (hρ : WindowValuesRegistered R s) (hreg : WindowRegistersAs W s)
    (hcov : RecordCovers W s) :
    (atCode r A).Sound c p .network (n * (δ + εks + εbind + εtap)) :=
  network_atCode c p R hks hb hδ hdec εks εbind εtap herr hbind htap
    (networkWindowLe hA W h2 h3 h4 h5 h6 hρ hreg hcov)

/-- **A network coupling's guarantee at the code from the named steps**: the coupled run at the verdict at the code
(`Vocab.Coupling.atCode`) meets `CommOfPeriodLeakAccepted`'s bound at the published certificates and the coupling's
price, the per-period guarantee `Vocab.networkSound_of_coupling` reads, from the bridge's assumptions and the steps at
what the prover sent at the outcome the coins lead to. -/
theorem coupling_atCode_of_steps {V : Vocab d α} {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)}
    (q : V.Coupling .onsite .network θ F c σ)
    (R : RealizesCommOfPeriodLeakAccepted q.run (V.certOf θ).K H D L Kₒ S lay k δ decode reg Dj n) (hks : KS H D)
    (hb : D.commit.Binds H) (hδ : 0 ≤ δ) (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode)
    (εks εbind εtap : ℝ)
    (herr : ∀ ξ i, Claims.Sampling.κ D δ (R.stmt ξ i) - δ +
      Real.sqrt (Claims.Sampling.γ D (R.stmt ξ i) * PrB H D.V (R.prover ξ i) (R.stmt ξ i)) ≤ εks)
    (hbind : ∀ ξ i, (Bext H D.V (E D decode) (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤ εbind)
    (htap : ∀ ξ i, (Btap H D.V (E D decode) D.commit D.parse R.tops (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤
      εtap)
    (W : WindowRuns R) (h2 : PeriodRunsV2 W fun ξ => (V.outcome θ F c σ q.coins ξ).sent) (h3 : V2AcceptMeans W)
    (h4 : WindowCoinsLive W)
    (h5 : SessionIsSampledRun W fun ξ => V.Logged (q.pos .network) .network (V.outcome θ F c σ q.coins ξ) ∧
      ¬ q.brk ξ)
    (h6 : NetworkCouples R (fun ξ => (V.outcome θ F c σ q.coins ξ).sent) W)
    (hρ : WindowValuesRegistered R fun ξ => (V.outcome θ F c σ q.coins ξ).sent)
    (hreg : WindowRegistersAs W fun ξ => (V.outcome θ F c σ q.coins ξ).sent)
    (hcov : RecordCovers W fun ξ => (V.outcome θ F c σ q.coins ξ).sent) :
    q.atCode.Sound (V.certOf θ) q.price .network (n * (δ + εks + εbind + εtap)) :=
  network_atCode_of_steps (V.certOf θ) q.price R hks hb hδ hdec εks εbind εtap herr hbind htap (fun _ h => h) W h2
    h3 h4 h5 h6 hρ hreg hcov

end Verity.Network
