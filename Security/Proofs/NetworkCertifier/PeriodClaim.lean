import Definitions.NetworkCertifier.PeriodClaim
import Proofs.Accounting.JointBridge.CommOfPeriodLeakAccepted
import Proofs.Core.UnionBound

/-!
# The network audit's per-period guarantee on a run whose claim is its monitoring period's

A run whose communication claim is `NetworkPeriod.commWithin` of a monitoring period, at a `K` whose bits name the
period's count (`NetworkPeriod.Bits`), realizes the network certifier's period of sampled audits
(`Accounting.JointBridge.RealizesCommOfPeriodLeakAccepted`, `realizes`) once each window's statement is in the sampled
audit's scope, the period's network certifier runs its code on its host (`trusted`, `trusted-host`, carried into the
realization as its premise) and the network audit's acceptance with a window failing its good event is bounded by a
run of the sampled verifier at that window (`window_le`). `sound` is `CommOfPeriodLeakAccepted` there: the network
audit's `PeriodRun.Sound` at `n · (δ + ε_ks + ε_bind + ε_tap)`.

**The union over network certifier devices.** Over a fleet of `m` devices, each with its own monitoring period `N j`
and its own run `r j` on one coin space, at certificates `c j`:

* `fleet_sound`: if each device's network audit has its per-period guarantee at `ε j` (`sound`), then some device's
  audit accepts while its claim fails with probability at most `∑ j, ε j` over the coins (`Pouw.union_bound_update`);
* `fleet_bits`: if `K j` bits name device `j`'s count (`NetworkPeriod.Bits`), then `∑ j, K j` bits name the fleet's
  (`NetworkPeriod.fleetCount`);
* `fleet_encard_le`: the fleet's secrets whose parts on the devices determine them, each part one of the device's at
  most `count` (`PeriodLeakAccepted`'s count at `N j`), number at most `fleetCount`.

Outside `fleet_sound`'s event, every device whose audit accepted holds its claim at `K j`, so the fleet moves at most
`∑ j, K j` bits through the network certifiers.
-/

namespace NetTiming.SecurityProofs.PeriodClaim

open Claims Claims.Sampling Accounting Accounting.JointBridge

/-- A run whose claim at `K` is the monitoring period's, at a `K` whose bits name the period's count, realizes the
network certifier's period of sampled audits, with the period's network certifier and its `trusted-host` premise. -/
def realizes {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}
    [DecidableEq I] [∀ i, Fintype (Ω i)] {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] {r : PeriodRun d α C Ω} (H : Msg → Msg) {D : Deploy Coin Col T} {L Kₒ S : ℕ}
    {lay : Stmt → Layout L S} {k : Stmt → ℕ} {δ : ℝ} (decode : List (ℕ × Col) → List T) {reg : Stmt → Finset ℕ}
    {Dj n : ℕ} {Conn : Type} [Fintype Conn] (N : NetworkPeriod D L Kₒ S lay k δ reg Dj n Conn ((i : I) → Ω i))
    {K : ℝ} (hK : N.Bits K) (hr : ∀ ξ, r.commWithin K ξ ↔ N.commWithin H K ξ)
    (scope : ∀ ξ i, ScopeAcc D lay k (N.stmt ξ i))
    (Tape : ((i : I) → Ω i) → Fin n → Type) [hT : ∀ ξ i, Fintype (Tape ξ i)] [hN : ∀ ξ i, Nonempty (Tape ξ i)]
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
  Conn := Conn
  Cell := N.Cell
  Srv := N.Srv
  Dev := N.Dev
  arr := N.arr
  fmt := N.fmt
  device := N.device
  run := N.run
  trusted := trusted
  window_le := window_le

/-- **The network audit's per-period guarantee on a run whose claim is its monitoring period's**, at certificates `c`
whose `K` names the period's count: `PeriodRun.Sound` at `n · (δ + ε_ks + ε_bind + ε_tap)`, under `trusted-host` for
the period's network certifier and `CommOfPeriodLeakAccepted`'s conditions at each window's statement and prover. -/
theorem sound {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}
    [DecidableEq I] [∀ i, Fintype (Ω i)] {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] {r : PeriodRun d α C Ω} (c : Certificates) (p : Price) (H : Msg → Msg)
    {D : Deploy Coin Col T} {L Kₒ S : ℕ} {lay : Stmt → Layout L S} {k : Stmt → ℕ} {δ : ℝ}
    (decode : List (ℕ × Col) → List T) {reg : Stmt → Finset ℕ} {Dj n : ℕ} {Conn : Type} [Fintype Conn]
    (N : NetworkPeriod D L Kₒ S lay k δ reg Dj n Conn ((i : I) → Ω i)) (hK : N.Bits c.K)
    (hr : ∀ ξ, r.commWithin c.K ξ ↔ N.commWithin H c.K ξ) (scope : ∀ ξ i, ScopeAcc D lay k (N.stmt ξ i))
    (Tape : ((i : I) → Ω i) → Fin n → Type) [hT : ∀ ξ i, Fintype (Tape ξ i)] [hN : ∀ ξ i, Nonempty (Tape ξ i)]
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
      εtap) :
    r.Sound c p .network (n * (δ + εks + εbind + εtap)) :=
  Accounting.SecurityProofs.Bridge.comm_of_periodLeak NetTiming.periodLeak c p
    (realizes H decode N hK hr scope Tape prover trusted window_le) hks hb hδ hdec εks εbind εtap herr hbind htap

/-- **The union over network certifier devices.** Over `m` devices, device `j`'s run `r j` on one coin space with its
network audit's per-period guarantee at certificates `c j`, price `p` and error `ε j` (`sound`): the probability over
the coins that some device's network audit accepts while its claim fails (more than `(c j).K` bits through it) is at
most `∑ j, ε j`. -/
theorem fleet_sound {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} [Fintype I]
    [DecidableEq I] {Ω : I → Type} [∀ i, Fintype (Ω i)] [∀ i, Nonempty (Ω i)] {m : ℕ}
    (r : Fin m → PeriodRun d α C Ω) (c : Fin m → Certificates) (p : Price) (ε : Fin m → ℝ)
    (h : ∀ j, (r j).Sound (c j) p .network (ε j)) :
    Pouw.pr (fun ξ => ∃ j, (r j).accepts .network ξ ∧ ¬ (r j).commWithin (c j).K ξ) ≤ ∑ j, ε j :=
  Pouw.union_bound_update (fun j => (r j).own .network)
    (fun j ξ => (r j).accepts .network ξ ∧ ¬ (r j).commWithin (c j).K ξ) ε h

/-- **A fleet's bits are its devices' bits summed.** If `K j` bits name device `j`'s count, `∑ j, K j` bits name the
fleet's: `fleetCount ≤ 2^b` for a whole number of bits `b ≤ ∑ j, K j`. -/
theorem fleet_bits {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    {D : Deploy Coin Col T} {L Kₒ S : ℕ} {lay : Stmt → Layout L S} {k : Stmt → ℕ} {δ : ℝ} {reg : Stmt → Finset ℕ}
    {Dj : ℕ} {X : Type} {m : ℕ} {n : Fin m → ℕ} {Conn : Fin m → Type} [∀ j, Fintype (Conn j)]
    (N : ∀ j, NetworkPeriod D L Kₒ S lay k δ reg Dj (n j) (Conn j) X) {K : Fin m → ℝ} (hK : ∀ j, (N j).Bits (K j)) :
    ∃ b : ℕ, (b : ℝ) ≤ ∑ j, K j ∧ NetworkPeriod.fleetCount N ≤ 2 ^ b := by
  have hK' : ∀ j, ∃ b : ℕ, (b : ℝ) ≤ K j ∧ (N j).count ≤ 2 ^ b := hK
  choose b hb hc using hK'
  refine ⟨∑ j, b j, by push_cast; exact Finset.sum_le_sum fun j _ => hb j, ?_⟩
  unfold NetworkPeriod.fleetCount
  rw [← Finset.prod_pow_eq_pow_sum]
  exact Finset.prod_le_prod fun j _ => hc j

/-- **A fleet's secrets are at most its count.** Over `m` devices, the fleet's secrets `A` whose parts on the devices
(`π j`) determine them (`hinj`), each part in device `j`'s secrets `A' j` (`hA`), which number at most device `j`'s
count (`hc`, `PeriodLeakAccepted`'s count at `N j`), number at most `fleetCount`. -/
theorem fleet_encard_le {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Zero Col] {D : Deploy Coin Col T} {L Kₒ S : ℕ} {lay : Stmt → Layout L S} {k : Stmt → ℕ} {δ : ℝ}
    {reg : Stmt → Finset ℕ} {Dj : ℕ} {X : Type} {m : ℕ} {n : Fin m → ℕ} {Conn : Fin m → Type}
    [∀ j, Fintype (Conn j)] (N : ∀ j, NetworkPeriod D L Kₒ S lay k δ reg Dj (n j) (Conn j) X) {Z : Type}
    {Z' : Fin m → Type} (A : Set Z) (A' : ∀ j, Set (Z' j)) (π : ∀ j, Z → Z' j)
    (hinj : Set.InjOn (fun z j => π j z) A) (hA : ∀ z ∈ A, ∀ j, π j z ∈ A' j)
    (hc : ∀ j, (A' j).encard ≤ (N j).count) : A.encard ≤ NetworkPeriod.fleetCount N := by
  rw [← hinj.encard_image]
  calc ((fun z j => π j z) '' A).encard ≤ (Set.univ.pi A').encard :=
        Set.encard_le_encard (Set.image_subset_iff.2 fun z hz => Set.mem_univ_pi.2 (hA z hz))
    _ = ∏ j, (A' j).encard := Set.encard_pi_eq_prod_encard
    _ ≤ ∏ j, (N j).count := Finset.prod_le_prod fun j _ => hc j

end NetTiming.SecurityProofs.PeriodClaim
