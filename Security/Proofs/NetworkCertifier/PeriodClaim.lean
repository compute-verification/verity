import Definitions.NetworkCertifier.PeriodClaim
import Proofs.Accounting.JointBridge.CommOfPeriodLeakAccepted

/-!
# The network audit's per-period guarantee on a run whose claim is its monitoring period's

A run whose communication claim is `NetworkPeriod.commWithin` of a monitoring period, at a `K` whose bits name the
period's count (`NetworkPeriod.Bits`), realizes the network certifier's period of sampled audits
(`Accounting.JointBridge.RealizesCommOfPeriodLeakAccepted`, `realizes`) once each window's statement is in the sampled
audit's scope and the network audit's acceptance with a window failing its good event is bounded by a run of the
sampled verifier at that window (`window_le`). `sound` is `CommOfPeriodLeakAccepted` there: the network audit's
`PeriodRun.Sound` at `n · (δ + ε_ks + ε_bind + ε_tap)`.
-/

namespace NetTiming.SecurityProofs.PeriodClaim

open Claims Claims.Sampling Accounting Accounting.JointBridge

/-- A run whose claim at `K` is the monitoring period's, at a `K` whose bits name the period's count, realizes the
network certifier's period of sampled audits. -/
def realizes {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}
    [DecidableEq I] [∀ i, Fintype (Ω i)] {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] {r : PeriodRun d α C Ω} (H : Msg → Msg) {D : Deploy Coin Col T} {L Kₒ S : ℕ}
    {lay : Stmt → Layout L S} {k : Stmt → ℕ} {δ : ℝ} (decode : List (ℕ × Col) → List T) {reg : Stmt → Finset ℕ}
    {Dj n : ℕ} {Conn : Type} [Fintype Conn] (N : NetworkPeriod D L Kₒ S lay k δ reg Dj n Conn ((i : I) → Ω i))
    {K : ℝ} (hK : N.Bits K) (hr : ∀ ξ, r.commWithin K ξ ↔ N.commWithin H K ξ)
    (scope : ∀ ξ i, ScopeAcc D lay k (N.stmt ξ i))
    (Tape : ((i : I) → Ω i) → Fin n → Type) [hT : ∀ ξ i, Fintype (Tape ξ i)] [hN : ∀ ξ i, Nonempty (Tape ξ i)]
    (prover : (ξ : (i : I) → Ω i) → (i : Fin n) → Prover (Tape ξ i))
    (window_le : ∀ ξ i, Pouw.pr (fun x : Ω (r.own .network) =>
      r.accepts .network (Function.update ξ (r.own .network) x) ∧
        ¬ WindowGood H D L Kₒ S lay k δ reg Dj (N.sw i) (N.ρ i) N.tops
          (N.stmt (Function.update ξ (r.own .network) x) i) (N.wit (Function.update ξ (r.own .network) x) i)
          (N.obs (Function.update ξ (r.own .network) x) i)) ≤
      PrBad H D.V
        (RLeakTap H D.commit D.parse D.partition L Kₒ S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj
          N.tops)
        (E D decode) (N.stmt ξ i) (prover ξ i)) :
    RealizesCommOfPeriodLeakAccepted r K H D L Kₒ S lay k δ decode reg Dj n where
  sw := N.sw
  ρ := N.ρ
  tops := N.tops
  stmt := N.stmt
  wit := N.wit
  obs := N.obs
  comm_iff ξ := (hr ξ).trans (and_iff_right hK)
  scope := scope
  Tape := Tape
  tapeFintype := hT
  tapeNonempty := hN
  prover := prover
  window_le := window_le

/-- **The network audit's per-period guarantee on a run whose claim is its monitoring period's**, at certificates `c`
whose `K` names the period's count: `PeriodRun.Sound` at `n · (δ + ε_ks + ε_bind + ε_tap)`, under
`CommOfPeriodLeakAccepted`'s conditions at each window's statement and prover. -/
theorem sound {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}
    [DecidableEq I] [∀ i, Fintype (Ω i)] {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] {r : PeriodRun d α C Ω} (c : Certificates) (p : Price) (H : Msg → Msg)
    {D : Deploy Coin Col T} {L Kₒ S : ℕ} {lay : Stmt → Layout L S} {k : Stmt → ℕ} {δ : ℝ}
    (decode : List (ℕ × Col) → List T) {reg : Stmt → Finset ℕ} {Dj n : ℕ} {Conn : Type} [Fintype Conn]
    (N : NetworkPeriod D L Kₒ S lay k δ reg Dj n Conn ((i : I) → Ω i)) (hK : N.Bits c.K)
    (hr : ∀ ξ, r.commWithin c.K ξ ↔ N.commWithin H c.K ξ) (scope : ∀ ξ i, ScopeAcc D lay k (N.stmt ξ i))
    (Tape : ((i : I) → Ω i) → Fin n → Type) [hT : ∀ ξ i, Fintype (Tape ξ i)] [hN : ∀ ξ i, Nonempty (Tape ξ i)]
    (prover : (ξ : (i : I) → Ω i) → (i : Fin n) → Prover (Tape ξ i))
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
      εtap) :
    r.Sound c p .network (n * (δ + εks + εbind + εtap)) :=
  Accounting.SecurityProofs.Bridge.comm_of_periodLeak NetTiming.periodLeak c p
    (realizes H decode N hK hr scope Tape prover window_le) hks hb hδ hdec εks εbind εtap herr hbind htap

end NetTiming.SecurityProofs.PeriodClaim
