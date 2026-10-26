import Proofs.Properties.Verity.Soundness.Legs
import Proofs.Properties.Verity.Network.Window

/-!
# The communication leg at the code, from NET-85's steps

`Verity.Legs.NetworkWindowLe R A` (ONE-12's refinement) follows from `Verity.Network.windowLe` at any verdict `A` under
which the compiled component accepts what the prover sent (`networkWindowLe`); in the game `A` is `Vocab.Logged`, whose
first conjunct is that acceptance (`networkWindowLe_logged`). So the communication leg at `atCode r A`
(`Verity.Legs.network`) rests on the bridge's assumptions (`KS`, `Commitment.Binds`, `Flock.ListWithin`, and
`trusted-host` in the realization) and NET-85's named steps (`network_of_steps`): S2–S6, the ρ gap and S8's two
premises (`Proofs.Properties.Verity.Network.Window`).
-/

namespace Verity.Legs

open Accounting Core Claims Claims.Sampling NetTiming Accounting.JointBridge Verity.Network

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}
  [DecidableEq I] [∀ i, Fintype (Ω i)] {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
  [DecidableEq Col] [Zero Col] {H : Msg → Msg} {D : Deploy Coin Col T} {L Kₒ S : ℕ} {lay : Stmt → Layout L S}
  {k : Stmt → ℕ} {δ : ℝ} {decode : List (ℕ × Col) → List T} {reg : Stmt → Finset ℕ} {Dj n : ℕ}
  {r : PeriodRun d α C Ω}

/-- **NET-85: ONE-12's refinement at a verdict `A` under which the compiled component accepts what the prover sent
(`s`), from its named steps.** -/
theorem networkWindowLe {K : ℝ} {R : RealizesCommOfPeriodLeakAccepted r K H D L Kₒ S lay k δ decode reg Dj n}
    {A : Audit → ((i : I) → Ω i) → Prop} {s : ((i : I) → Ω i) → Sent}
    (hA : ∀ ξ, A .network ξ → ComponentAccepts .network (s ξ)) (W : WindowRuns R) (h2 : PeriodRunsV2 W s)
    (h3 : V2AcceptMeans W) (h4 : WindowCoinsLive W) (h5 : SessionIsSampledRun W) (h6 : NetworkCouples R s W)
    (hρ : WindowValuesRegistered R s) (hreg : WindowRegistersAs W s) (hcov : RecordCovers W s) :
    NetworkWindowLe R A := fun ξ i =>
  le_trans (Pouw.pr_mono fun _ hx => ⟨hA _ hx.1, hx.2⟩) (windowLe W h2 h3 h4 h5 h6 hρ hreg hcov ξ i)

/-- **NET-85 in the game**: ONE-12's refinement at `Vocab.Logged`, the verdict `bound_atCode` reads, with the steps at
what the prover sent at the outcome the coins lead to. -/
theorem networkWindowLe_logged [Fintype I] {V : Vocab d α} {θ F c : ByteArray} {out : ((i : I) → Ω i) → Outcome}
    {K : ℝ} {R : RealizesCommOfPeriodLeakAccepted r K H D L Kₒ S lay k δ decode reg Dj n} (W : WindowRuns R)
    (h2 : PeriodRunsV2 W fun ξ => (out ξ).sent) (h3 : V2AcceptMeans W) (h4 : WindowCoinsLive W)
    (h5 : SessionIsSampledRun W) (h6 : NetworkCouples R (fun ξ => (out ξ).sent) W)
    (hρ : WindowValuesRegistered R fun ξ => (out ξ).sent) (hreg : WindowRegistersAs W fun ξ => (out ξ).sent)
    (hcov : RecordCovers W fun ξ => (out ξ).sent) : NetworkWindowLe R (V.Logged θ F c out) :=
  networkWindowLe (fun _ h => h.1) W h2 h3 h4 h5 h6 hρ hreg hcov

/-- **The communication leg at the code from NET-85's steps**: `Verity.Legs.network` with `NetworkWindowLe` from
`networkWindowLe`. -/
theorem network_of_steps (c : Certificates) (p : Price)
    (R : RealizesCommOfPeriodLeakAccepted r c.K H D L Kₒ S lay k δ decode reg Dj n) (hks : KS H D)
    (hb : D.commit.Binds H) (hδ : 0 ≤ δ) (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode)
    (εks εbind εtap : ℝ)
    (herr : ∀ ξ i, Claims.Sampling.κ D δ (R.stmt ξ i) - δ +
      Real.sqrt (Claims.Sampling.γ D (R.stmt ξ i) * PrB H D.V (R.prover ξ i) (R.stmt ξ i)) ≤ εks)
    (hbind : ∀ ξ i, (Bext H D.V (E D decode) (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤ εbind)
    (htap : ∀ ξ i, (Btap H D.V (E D decode) D.commit D.parse R.tops (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤
      εtap)
    {A : Audit → ((i : I) → Ω i) → Prop} {s : ((i : I) → Ω i) → Sent}
    (hA : ∀ ξ, A .network ξ → ComponentAccepts .network (s ξ)) (W : WindowRuns R) (h2 : PeriodRunsV2 W s)
    (h3 : V2AcceptMeans W) (h4 : WindowCoinsLive W) (h5 : SessionIsSampledRun W) (h6 : NetworkCouples R s W)
    (hρ : WindowValuesRegistered R s) (hreg : WindowRegistersAs W s) (hcov : RecordCovers W s) :
    (atCode r A).Sound c p .network (n * (δ + εks + εbind + εtap)) :=
  network c p R hks hb hδ hdec εks εbind εtap herr hbind htap
    (networkWindowLe hA W h2 h3 h4 h5 h6 hρ hreg hcov)

end Verity.Legs
