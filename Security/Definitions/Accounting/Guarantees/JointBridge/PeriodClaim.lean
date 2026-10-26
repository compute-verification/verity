import Definitions.NetworkCertifier.PeriodClaim
import Definitions.Accounting.JointBridge.CommOfPeriodLeakAccepted

/-!
# The network audit's per-period guarantee on a run whose claim is its monitoring period's

`PeriodClaimSound` is the network audit's `Accounting.PeriodRun.Sound` for a run whose communication claim is the
network certifier's claim over a monitoring period of sampled audits (`NetTiming.NetworkPeriod.commWithin`), at a `K`
whose bits name the period's count (`NetworkPeriod.Bits`), at the error of `PeriodLeakAccepted` over the period's `n`
windows. It is `Accounting.Guarantees.Bridge.CommOfPeriodLeakAccepted` at the realization the period gives.
-/

namespace NetTiming.Guarantees

open Claims Claims.Sampling Accounting

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The network audit's per-period guarantee on a run whose claim is its monitoring period's.** On a run `r` whose
claim at `c.K` is the network certifier's claim over a monitoring period `N` of `n` sampled audits (`hr`), at a `c.K`
whose bits name the period's count (`hK`), with every window's statement in the sampled audit's scope (`scope`), the
period's network certifier running its code on its host (`trusted`, `trusted-host`), and the network audit's
acceptance with a window failing its good event bounded by a run of the sampled verifier at that window's statement
and prover (`window_le`), at a deployment `D` where `Claims.Sampling.KS` holds and the trace's commitment binds
(`Commitment.Binds`), at `δ ≥ 0` and a decoder that lists every trace within Flock's radius, where each window's
prover has `κ − δ + √(γ·Pr[B]) ≤ ε_ks`, `Pr[B_ext] ≤ ε_bind` and `Pr[B_tap] ≤ ε_tap`: the network audit's
`PeriodRun.Sound` holds at `n · (δ + ε_ks + ε_bind + ε_tap)`. -/
def PeriodClaimSound : Prop :=
  ∀ {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}
    [DecidableEq I] [∀ i, Fintype (Ω i)] {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] {r : PeriodRun d α C Ω} (c : Certificates) (p : Price) (H : Msg → Msg)
    {D : Deploy Coin Col T} {L Kₒ S : ℕ} {lay : Stmt → Layout L S} {k : Stmt → ℕ} {δ : ℝ}
    (decode : List (ℕ × Col) → List T) {reg : Stmt → Finset ℕ} {Dj n : ℕ} {Conn : Type} [Fintype Conn]
    (N : NetworkPeriod D L Kₒ S lay k δ reg Dj n Conn ((i : I) → Ω i)) (hK : N.Bits c.K)
    (hr : ∀ ξ, r.commWithin c.K ξ ↔ N.commWithin H c.K ξ) (scope : ∀ ξ i, ScopeAcc D lay k (N.stmt ξ i))
    (Tape : ((i : I) → Ω i) → Fin n → Type) [∀ ξ i, Fintype (Tape ξ i)] [∀ ξ i, Nonempty (Tape ξ i)]
    (prover : (ξ : (i : I) → Ω i) → (i : Fin n) → Prover (Tape ξ i))
    (trusted : Verity.Assumptions.TrustedHost N.device N.run fun p => tapCode N.fmt p.1 (N.arr p.2))
    (window_le : ∀ ξ i, Pouw.pr (fun x : Ω (r.own .network) =>
      r.accepts .network (Function.update ξ (r.own .network) x) ∧
        ¬ WindowGood H D L Kₒ S lay k δ reg Dj (N.sw i) (N.ρ i) N.tops
          (N.stmt (Function.update ξ (r.own .network) x) i) (N.wit (Function.update ξ (r.own .network) x) i)
          (N.obs (Function.update ξ (r.own .network) x) i)) ≤
      PrBad H D.V
        (RLeakTap H D.commit D.parse D.partition L Kₒ S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj
          N.tops)
        (E D decode) (N.stmt ξ i) (prover ξ i))
    (hks : KS H D) (hb : D.commit.Binds H) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (εks εbind εtap : ℝ)
    (herr : ∀ ξ i, Claims.Sampling.κ D δ (N.stmt ξ i) - δ +
      Real.sqrt (Claims.Sampling.γ D (N.stmt ξ i) * PrB H D.V (prover ξ i) (N.stmt ξ i)) ≤ εks)
    (hbind : ∀ ξ i, (Bext H D.V (E D decode) (Tape ξ i) (prover ξ i) (N.stmt ξ i)).Pr H ≤ εbind)
    (htap : ∀ ξ i, (Btap H D.V (E D decode) D.commit D.parse N.tops (Tape ξ i) (prover ξ i) (N.stmt ξ i)).Pr H ≤
      εtap),
    r.Sound c p .network (n * (δ + εks + εbind + εtap))

end NetTiming.Guarantees
