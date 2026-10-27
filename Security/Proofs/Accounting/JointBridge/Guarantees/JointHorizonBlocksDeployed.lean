import Proofs.Accounting.JointBridge.Guarantees.IntegrityOfIntegrityBlock
import Definitions.Accounting.JointBridge.CreditedWords
import Definitions.Accounting.Assumptions.CreditedWords
import Definitions.Accounting.JointBridge.CommOfPeriodLeakAccepted
import Definitions.Accounting.JointBridge.BandwidthOfAuditPerRunPeriod
import Definitions.Accounting.JointBridge.SpaceOfPeriodHeld
import Properties.Pouw.Period
import Properties.Pous.Period

/-!
# The joint guarantee over a horizon of server-time blocks, every leg from its protocol's own guarantee (trusted)

`Accounting.Guarantees.JointHorizonBlocks` takes each audit's per-period guarantee in each block
(`Accounting.PeriodRun.Sound`) as a hypothesis. Each protocol's bridge gives that guarantee, at its source's error, for
a period that realizes the protocol's audit, from the source's own premises:

* sampled proofs: `Accounting.Guarantees.Bridge.IntegrityOfIntegrityBlock`, C-Flock's integrity audit of the
  server-time block (`Flock.Guarantees.IntegrityBlock`), at `δ + Σ_{t ∈ b} (ε_ks,t + δ_link,t)`;
* PoUW: `Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod`, H1's checked credit over the torch loop
  (`Properties.Pouw.Period.H1CheckedCreditPeriod`), at `δ_s + n·(δ_bind + δ_ks)`;
* PoUS: `Accounting.Bridge.SpaceOfPeriodHeld`, the ring audit's period (`Pous.Period.PeriodHeld`), at `δ + ε_max`;
* PoUB: `Accounting.Guarantees.Bridge.BandwidthOfAuditPerRunPeriod`, the bandwidth audit's game
  (`Bandwidth.Guarantees.AuditRO`), at `((k − 1)/N)^s + s/2^h + C(q, k)/2^λ`;
* the network certifier: `Accounting.Guarantees.Bridge.CommOfPeriodLeakAccepted`, its period of sampled audits
  (`NetTiming.Guarantees.PeriodLeakAccepted`), at `n·(δ + ε_ks + ε_bind + ε_tap)`.

`JointHorizonBlocksDeployed` is `JointHorizonBlocks` with each block's five legs given that way: its hypotheses are each
block's five realizations and their sources' premises, and its bound is each source's error summed over the blocks.
`JointHorizonBlocksDeployedA` is Demo A's (`Accounting.Guarantees.JointHorizonBlocksA`): sampled proofs, PoUW and the
network certifier. The protocol's parameters (the scheme and the declared circuit, the deployment of the sampled
verifier, the store's shape, the ring game's bounds) are one across the horizon; what the device does in a block (its
windows, its play, its strategy, its game device, its joint run) is that block's.

The statement reads C-Flock's integrity audit, whose definitions are still in the proofs package, and PoUW's and PoUS's
period statements, which are in `security_properties`, so it is in the proofs package, beside
`Proofs.Accounting.JointBridge.Guarantees.IntegrityOfIntegrityBlock`.
-/

namespace Accounting.Guarantees

open Accounting
open scoped ENNReal

-- A guarantee keeps its sources' hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

open FlockSoundness.Audit FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.ZkLink
  FlockSoundness.Discharge.ZkHidden Claims Claims.Sampling NetTiming in
/-- **The joint guarantee over a horizon of `B` server-time blocks, every leg from its protocol's own guarantee.** On
any device and any horizon of `B` blocks of a node's run over a circuit on one coin space (as
`Accounting.Guarantees.JointHorizonBlocks`), suppose each block's five audits realize their protocols' audits under
their sources' premises, block by block:

* sampled proofs: block `b`'s sampled audit is C-Flock's integrity audit of block `b` of the server's sessions `S`
  (`Accounting.RealizesIntegrityOfIntegrityBlock`), each draw covering at least a fraction `q > 0` of its sampling
  block's units, under A3 for each draw's bytes and `cr/sha-512` (`TableCRZ`) and `ecr/sha-512` (`LinkCRZC`) at every
  strategy the node plays a sampling block with, at an integrity certificate of at least `ln(1/δ_I)/q` units;
* PoUW: block `b`'s PoUW audit credits H1's checked words over the torch loop (`Accounting.RealizesCreditedWords` at
  `m = minWordMacs Sc Cd macs` multiply-adds a word, `u` a unit and `b_unit` bits a unit), each right word of a unit
  the device didn't outsource costing `m` multiply-adds of priced work (`Accounting.Assumptions.WordCostsMacs`) and at
  most `K / b_unit` units outsourced (`Accounting.Assumptions.HelpWithinCertifier`), under
  `Properties.Pouw.Period.H1CheckedCreditPeriod`'s premises at the block's windows, openings, salts, live tiles, play
  and folded words;
* PoUS: block `b`'s PoUS audit is the capacity audit of a joint run on a ring-model server
  (`Accounting.RealizesSpaceOfPeriodHeld`, `Pous.Period.RingDevice`), under `Pous.Period.PeriodHeld`'s premises, with a
  space certificate within the ring game's state bound;
* PoUB: block `b`'s PoUB audit is the bandwidth audit's game (`Accounting.RealizesBandwidthOfAuditPerRunPeriod`), with
  a bandwidth certificate of at most `k·b − S_chip − c − G·b − λ` bits;
* the network certifier: block `b`'s network audit is the network certifier's period of `n_c` sampled audits
  (`Accounting.JointBridge.RealizesCommOfPeriodLeakAccepted`), under `Claims.Sampling.KS`, the trace commitment's
  binding (`Commitment.Binds`) and the decoder's radius, each window's prover within `ε_ks`, `ε_bind` and `ε_tap`;

and, as in `JointHorizonBlocks`, each wipe's survivor bound holds at its error `ε_wipe b` and each block's memory term
at `δ_PoUS b`. Then the probability that some block's accepted audit's claim fails, some accepted wipe's survivor bound
fails, or some block's memory term fails while its PoUS accepted is at most
$$\sum_b \Big(\delta_I + \sum_{t \in b} (\varepsilon_{ks,t} + \delta_{link,t}) + \delta_s + n(\delta_{bind} + \delta_{ks})
  + (\delta + \varepsilon_{max}) + \big(\tfrac{k-1}{N}\big)^s + \tfrac{s}{2^h} + \tfrac{C(q_b, k)}{2^\lambda}
  + n_c(\delta_c + \varepsilon_{ks} + \varepsilon_{bind} + \varepsilon_{tap})\Big) + \sum_b \varepsilon_{\mathrm{wipe},b}
  + \sum_b \delta_{\mathrm{PoUS},b},$$
`q_b = rate_oracle · T_b` the device's oracle queries over block `b`'s window.

**The wipe and the memory term keep their hypotheses.** Each wipe's survivor bound (`ε_wipe b`) and each block's memory
term (`δ_PoUS b`) are taken as in `JointHorizonBlocks`: memory accounting's wipe bridge, from
`Pous.Guarantees.SecureErasureMeetsFamily` through DEVICE-WIPE's `Verity.Wipe.SurvivorBound` to `BlockWipe.Survives`,
doesn't exist yet, and neither does a bridge from weights-only PoUS to `BlocksRun.MemoryTerm`. Each bridge's own gap
stays as its realization's field or its source's premise: the program-from-circuit correspondence (M1, `wrong_eq`), the
words fixed right as device-model work (RT4-LOGIC-6, now the premises `WordCostsMacs` and `HelpWithinCertifier`), PoUS
at its challenge times only, and the
trace's values at the registered inputs and the records' coverage on the network certifier's acceptance
(`window_le`). As in `JointHorizonBlocks`, the state a block
carries across a wipe is outside the integrity bound (PRF-167). -/
def JointHorizonBlocksDeployed : Prop :=
  ∀ (ι α I : Type) [Fintype ι] [Fintype α] [Fintype I] [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)]
    [∀ i, Nonempty (Ω i)] (d : Device ι) (C : Verity.Circuit) (B : ℕ) (s : BlocksRun d α C Ω B)
    (c : Fin B → Certificates) (p : Price) (f : ℝ) (εWipe δPous : Fin B → ℝ)
    -- sampled proofs: C-Flock's integrity audit of each server-time block
    (S : Flock.Guarantees.ServerBlocks) (q δI : ℝ), 0 < q → 0 < δI →
    (∀ b : Fin B, ∀ t ∈ S.sessions b, q ≤ ((S.W t).kd : ℝ) / (S.W t).n) →
    (∀ b : Fin B, ∀ t ∈ S.sessions b, FlockSoundness.Assumptions.UniformRandomBytes (S.W t).Lb (S.W t).src) →
    ∀ RI : (b : Fin B) → RealizesIntegrityOfIntegrityBlock (s.block b) S b,
    (∀ b ξ t (τ : FlockSoundness.Game.Strategy (S.W t).game),
      Flock.Guarantees.Period.Plays S.W S.T ((RI b).strat ξ) t τ → ∀ ω j,
      TableCRZ FlockSoundness.execArith hs512 (planZHJ (S.W t).dj (S.W t).y₀ ((S.W t).L.draw ω) (reg ((S.W t).sim τ))) j
        (zcStrat FlockSoundness.execArith hs512 (planZHJ (S.W t).dj (S.W t).y₀) (cont ((S.W t).sim τ) ω)) (S.W t).qF (S.W t).qS) →
    (∀ b ξ t (τ : FlockSoundness.Game.Strategy (S.W t).game),
      Flock.Guarantees.Period.Plays S.W S.T ((RI b).strat ξ) t τ →
      LinkCRZC FlockSoundness.execArith hs512 (S.W t).P (planZHJ (S.W t).dj (S.W t).y₀) (tabZHJ (S.W t).dj (S.W t).y₀) (S.W t).vb
        (reg ((S.W t).sim τ)) (cont ((S.W t).sim τ)) (S.W t).k (S.W t).Rw (S.W t).M (S.W t).t') →
    (∀ b, Real.log (1 / δI) / q ≤ (c b).ε * (s.block b).units) →
    (∀ b, ENNReal.ofReal δI + S.err b ≠ ⊤) →
    -- PoUW: H1's checked credit over the torch loop
    ∀ {ι' K T L U Cn Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype Cn] [Fintype Ξ]
      [Nonempty V] (Sc : Accounting.Compose.Scheme K) (Cd : Accounting.Compose.Declared ι') {n : ℕ} (hn : 0 < n)
      (hN : Cd.units ≤ 144) (hr : 0 < Cd.rounds) (hM : Cd.rounds = 1) (hK : 24 ≤ 144) (prm : PouwCheck.Params)
      (enc : Cn → ByteArray) (staged flat : ByteArray → Fin 144 → (ℕ → V) → Prop) (δs δbind δks : ℝ)
      (ws : Fin B → ((i : I) → Ω i) →
        Properties.Pouw.Period.Windows (Pouw.RelativeSecurity.Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n)
      (opened : Fin B → ((i : I) → Ω i) →
        Pouw.RelativeSecurity.Paths (Pouw.RelativeSecurity.Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n →
          Fin n → ℕ → ℕ → V)
      (salt : Fin B → ((i : I) → Ω i) →
        Pouw.RelativeSecurity.Paths (Pouw.RelativeSecurity.Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n →
          Fin n → ByteArray)
      (Wp : Fin B → ((i : I) → Ω i) →
        Accounting.Compose.Period Cn (Pouw.RelativeSecurity.Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ))
          (Fin 144) Cd.rounds n)
      (Pl : Fin B → ((i : I) → Ω i) →
        Accounting.TS.Play (Pouw.RelativeSecurity.Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n K T L U)
      (folded : Fin B → ((i : I) → Ω i) →
        Pouw.RelativeSecurity.Paths (Pouw.RelativeSecurity.Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → ℝ)
      (macs : Accounting.Compose.Kind → ℕ) (u bUnit : ℕ)
      (help : Fin B → ((i : I) → Ω i) →
        Pouw.RelativeSecurity.Paths (Pouw.RelativeSecurity.Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → ℝ),
    ∀ RW : (b : Fin B) → Accounting.RealizesCreditedWords (s.block b) (c b) p (Accounting.Compose.minWordMacs Sc Cd macs) u
      bUnit (Pouw.RelativeSecurity.Paths (Pouw.RelativeSecurity.Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n)
      (fun ξ => (Pl b ξ).accepts)
      (fun ξ x => Accounting.Compose.creditPrefix Sc Cd 144 ((ws b ξ).prefixLen prm x)
        (Properties.Pouw.Period.checkedLive Sc Cd (Wp b ξ) prm (ws b ξ) x) 1 1 (67 / 8) δs),
    (∀ b, Accounting.Assumptions.WordCostsMacs (s.block b) p (Accounting.Compose.minWordMacs Sc Cd macs) u (RW b).coin
      (fun ξ => (Pl b ξ).accepts) (fun ξ x => n * ((Pl b ξ).run x).fixed) (help b)) →
    (∀ b, Accounting.Assumptions.HelpWithinCertifier (s.block b) (c b) bUnit (RW b).coin (fun ξ => (Pl b ξ).accepts)
      (help b)) →
    Accounting.Compose.Scope Sc Cd δs →
    (∀ t : Fin 144, (8 / 9 : ℝ) * (144 : ℕ) * Accounting.Compose.shareCone Sc Cd 144 t ≤ 1) →
    (∀ j, Accounting.Compose.stageShare (Accounting.Compose.tileStage Cd 144)
      (Accounting.Compose.shareCone Sc Cd 144) j ≤ 1) →
    (∀ t : Fin 144, 0 < Accounting.Compose.shareCone Sc Cd 144 t →
      Accounting.Compose.tileLuck Sc Cd 144 t ≤ 7 / 8 ∧ Accounting.Compose.tileLuck0 Sc Cd 144 t ≤ 1 / 2) →
    (∀ b ξ, (Pl b ξ).game = Accounting.Compose.periodGame (Accounting.Compose.tileStage Cd 144)
      (Accounting.Compose.liveCone Sc Cd 144) 1 1 24 (Wp b ξ)) →
    (∀ b ξ, (Wp b ξ).EveryWindow (fun live _ => Accounting.Compose.DeadConeDead Sc Cd 144 live)) →
    (∀ b ξ, Accounting.Compose.PrefixOf
      (Properties.Pouw.Period.checkedLayer Cd hr hK prm (ws b ξ) (opened b ξ)
        (Properties.Pouw.Period.stagedAt (salt b ξ) staged) (Properties.Pouw.Period.flatAtCoin hr enc flat))
      (Pl b ξ).game) →
    (∀ b ξ w, Pouw.RelativeSecurity.prob (fun x =>
      (Properties.Pouw.Period.checkedLayer Cd hr hK prm (ws b ξ) (opened b ξ)
        (Properties.Pouw.Period.stagedAt (salt b ξ) staged)
        (Properties.Pouw.Period.flatAtCoin hr enc flat)).MovedIn x w) ≤ δbind) →
    (∀ b ξ, Properties.Pouw.Period.RegisteredBeforeDraw (ws b ξ)) →
    (∀ b ξ, Properties.Pouw.Period.DrawIsCoin hK prm (ws b ξ)) →
    (∀ b ξ, Properties.Pouw.Period.ServiceSound prm (ws b ξ) (opened b ξ)
      (Properties.Pouw.Period.stagedAt (salt b ξ) staged) δks) →
    (∀ b ξ, Properties.Pouw.Period.SaltIsLiveCoin hr (salt b ξ) enc) → Properties.Pouw.Period.StagedPins staged flat →
    (∀ b ξ, (Wp b ξ).EveryWindow (fun live W => Accounting.Compose.PerLiveTile Sc Cd live W
      (fun k G _ => Accounting.Assumptions.TS.OperandHMu (Sc.mu k) G))) →
    (∀ b ξ, (Wp b ξ).EveryWindow (fun live W => Accounting.Compose.PerLiveTile Sc Cd live W
      (fun k G _ => Accounting.Assumptions.TS.ReceiptLuckAt (Sc.rB k) G))) →
    (∀ b ξ, (Wp b ξ).EveryWindow (fun live W => Accounting.Compose.PerLiveTile Sc Cd live W
      (fun k _ G0 => Accounting.Assumptions.Compose.GuessLuckAt (Sc.guess k) G0))) →
    (∀ b ξ, Accounting.Assumptions.TS.UnitFixed (Pl b ξ)
      (Properties.Pouw.Period.checkedLayer Cd hr hK prm (ws b ξ) (opened b ξ)
        (Properties.Pouw.Period.stagedAt (salt b ξ) staged) (Properties.Pouw.Period.flatAtCoin hr enc flat))
      (Accounting.Compose.periodMode Cd (Wp b ξ)
        (Properties.Pouw.Period.checkedRun hK prm (ws b ξ) (opened b ξ)
          (Properties.Pouw.Period.stagedAt (salt b ξ) staged) (Properties.Pouw.Period.flatAtCoin hr enc flat)))
      (fun _ t => (Accounting.Compose.tileWords Sc Cd 144 t : ℝ)) (folded b ξ)) →
    -- PoUS: the ring audit's period
    PousCheck.Properties.AcceptsDecodes → PousCheck.Properties.PublishesRoot →
    Pous.P2.Assumptions.CoinCommitHides Pous.EndToEnd.coinView ((2⁻¹ : ℝ≥0∞) ^ 131) →
    ∀ Dr Qr Lr : ℕ, Pous.P2.Assumptions.P2RingAuditSecure Dr Qr Lr →
    ∀ (ΩW ΩQ : Fin B → Type)
      (run : (b : Fin B) → Run d α (ΩW b) (ΩQ b)
        ((Pous.ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × Pous.EndToEnd.HmCoins × Pous.EndToEnd.Streams))
      (P : (b : Fin B) → Pous.Period.Audits (ΩW b) (ΩQ b)) (round : Fin B → ℕ),
    ∀ RS : (b : Fin B) → RealizesSpaceOfPeriodHeld (s.block b) (run b) (fun ω => (P b).Accepts ω ∧ ¬ (P b).Break ω),
    (∀ b, Pous.Period.RingDevice (run b) (P b) Dr Qr Lr (round b)) →
    (∀ b, (c b).S * ((run b).win .capacity).len + d.chip + ((∑ i, d.cap i * round b : ℕ) : ℝ) ≤ Pous.EndToEnd.S) →
    -- PoUB: the bandwidth audit's game
    ∀ (N bits h sB G k lam : ℕ), 0 < N →
    ∀ RB : (b : Fin B) → RealizesBandwidthOfAuditPerRunPeriod G (N := N) (b := bits) (h := h) (s := sB) (s.block b),
    (∀ b, (c b).Q ≤ ((k * bits : ℕ) : ℝ) - d.chip - ((∑ i, d.cap i * (s.block b).win.len : ℕ) : ℝ) -
      ((G * bits : ℕ) : ℝ) - lam) →
    -- the network certifier: its period of sampled audits
    ∀ {Coin Col Tn : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
      {H : Msg → Msg} {D : Deploy Coin Col Tn} {Ln Kₒ Sn : ℕ} {lay : Stmt → Layout Ln Sn} {kh : Stmt → ℕ} {δc : ℝ}
      {decode : List (ℕ × Col) → List Tn} {regN : Stmt → Finset ℕ} {Dj nc : ℕ},
    ∀ RN : (b : Fin B) →
      Accounting.JointBridge.RealizesCommOfPeriodLeakAccepted (s.block b) (c b).K H D Ln Kₒ Sn lay kh δc decode regN
        Dj nc,
    KS H D → D.commit.Binds H → 0 ≤ δc → Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode →
    ∀ εks εbind εtap : ℝ,
    (∀ b ξ i, Claims.Sampling.κ D δc ((RN b).stmt ξ i) - δc +
      Real.sqrt (Claims.Sampling.γ D ((RN b).stmt ξ i) * PrB H D.V ((RN b).prover ξ i) ((RN b).stmt ξ i)) ≤ εks) →
    (∀ b ξ i, (Bext H D.V (E D decode) ((RN b).Tape ξ i) ((RN b).prover ξ i) ((RN b).stmt ξ i)).Pr H ≤ εbind) →
    (∀ b ξ i, (Btap H D.V (E D decode) D.commit D.parse (RN b).tops ((RN b).Tape ξ i) ((RN b).prover ξ i)
      ((RN b).stmt ξ i)).Pr H ≤ εtap) →
    -- the wipes and the memory terms, as in `JointHorizonBlocks`
    (∀ b, Pouw.pr (fun ξ => (s.wipe b).accepts ξ ∧ ¬ (s.wipe b).Survives f ξ) ≤ εWipe b) →
    (∀ b, Pouw.pr (fun ξ => (s.block b).accepts .pous ξ ∧ ¬ s.MemoryTerm b ξ) ≤ δPous b) →
    Pouw.pr (fun ξ => (∃ b a, (s.block b).accepts a ξ ∧ ¬ (s.block b).Holds (c b) p a ξ) ∨
        (∃ b, (s.wipe b).accepts ξ ∧ ¬ (s.wipe b).Survives f ξ) ∨
        (∃ b, (s.block b).accepts .pous ξ ∧ ¬ s.MemoryTerm b ξ)) ≤
      ∑ b : Fin B, ((ENNReal.ofReal δI + S.err b).toReal + (δs + n * (δbind + δks)) +
          ((Pous.Params.δ : ℝ) + Pous.Params.εMax.toReal) +
          ((((k - 1 : ℕ) : ℝ) / N) ^ sB + (sB : ℝ) / 2 ^ h +
            (((d.rate .oracle * (s.block b).win.len).choose k : ℕ) : ℝ) / 2 ^ lam) +
          nc * (δc + εks + εbind + εtap)) +
        ∑ b, εWipe b + ∑ b, δPous b

open FlockSoundness.Audit FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.ZkLink
  FlockSoundness.Discharge.ZkHidden Claims Claims.Sampling NetTiming in
/-- **Demo A's composed theorem, every leg from its protocol's own guarantee: sampled proofs, PoUW and the network
certifier over a horizon of `B` server-time blocks.** As `Accounting.Guarantees.JointHorizonBlocksA`, with each block's
three legs given by their realizations and their sources' premises, block by block, as in
`JointHorizonBlocksDeployed`: block `b`'s sampled audit is C-Flock's integrity audit of block `b` of the server's
sessions `S` (`Accounting.RealizesIntegrityOfIntegrityBlock`), its PoUW audit H1's checked words credited over the
torch loop (`Accounting.RealizesCreditedWords`, with `WordCostsMacs` and `HelpWithinCertifier`), and its network audit the network certifier's period of `n_c`
sampled audits (`Accounting.JointBridge.RealizesCommOfPeriodLeakAccepted`). Then the probability that one of those three
audits in some block accepts while its claim fails is at most
$$\sum_b \Big(\delta_I + \sum_{t \in b} (\varepsilon_{ks,t} + \delta_{link,t}) + \delta_s + n(\delta_{bind} + \delta_{ks})
  + n_c(\delta_c + \varepsilon_{ks} + \varepsilon_{bind} + \varepsilon_{tap})\Big).$$
Demo A has no wipe or memory term (memory accounting joins at M4 through `JointHorizonBlocksDeployed`); each bridge's own
gap stays as its realization's field or its source's premise, as there. -/
def JointHorizonBlocksDeployedA : Prop :=
  ∀ (ι α I : Type) [Fintype α] [Fintype I] [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)]
    [∀ i, Nonempty (Ω i)] (d : Device ι) (C : Verity.Circuit) (B : ℕ) (r : Fin B → PeriodRun d α C Ω)
    (c : Fin B → Certificates) (p : Price)
    -- sampled proofs: C-Flock's integrity audit of each server-time block
    (S : Flock.Guarantees.ServerBlocks) (q δI : ℝ), 0 < q → 0 < δI →
    (∀ b : Fin B, ∀ t ∈ S.sessions b, q ≤ ((S.W t).kd : ℝ) / (S.W t).n) →
    (∀ b : Fin B, ∀ t ∈ S.sessions b, FlockSoundness.Assumptions.UniformRandomBytes (S.W t).Lb (S.W t).src) →
    ∀ RI : (b : Fin B) → RealizesIntegrityOfIntegrityBlock (r b) S b,
    (∀ b ξ t (τ : FlockSoundness.Game.Strategy (S.W t).game),
      Flock.Guarantees.Period.Plays S.W S.T ((RI b).strat ξ) t τ → ∀ ω j,
      TableCRZ FlockSoundness.execArith hs512 (planZHJ (S.W t).dj (S.W t).y₀ ((S.W t).L.draw ω) (reg ((S.W t).sim τ))) j
        (zcStrat FlockSoundness.execArith hs512 (planZHJ (S.W t).dj (S.W t).y₀) (cont ((S.W t).sim τ) ω)) (S.W t).qF (S.W t).qS) →
    (∀ b ξ t (τ : FlockSoundness.Game.Strategy (S.W t).game),
      Flock.Guarantees.Period.Plays S.W S.T ((RI b).strat ξ) t τ →
      LinkCRZC FlockSoundness.execArith hs512 (S.W t).P (planZHJ (S.W t).dj (S.W t).y₀) (tabZHJ (S.W t).dj (S.W t).y₀) (S.W t).vb
        (reg ((S.W t).sim τ)) (cont ((S.W t).sim τ)) (S.W t).k (S.W t).Rw (S.W t).M (S.W t).t') →
    (∀ b, Real.log (1 / δI) / q ≤ (c b).ε * (r b).units) →
    (∀ b, ENNReal.ofReal δI + S.err b ≠ ⊤) →
    -- PoUW: H1's checked credit over the torch loop
    ∀ {ι' K T L U Cn Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype Cn] [Fintype Ξ]
      [Nonempty V] (Sc : Accounting.Compose.Scheme K) (Cd : Accounting.Compose.Declared ι') {n : ℕ} (hn : 0 < n)
      (hN : Cd.units ≤ 144) (hr : 0 < Cd.rounds) (hM : Cd.rounds = 1) (hK : 24 ≤ 144) (prm : PouwCheck.Params)
      (enc : Cn → ByteArray) (staged flat : ByteArray → Fin 144 → (ℕ → V) → Prop) (δs δbind δks : ℝ)
      (ws : Fin B → ((i : I) → Ω i) →
        Properties.Pouw.Period.Windows (Pouw.RelativeSecurity.Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n)
      (opened : Fin B → ((i : I) → Ω i) →
        Pouw.RelativeSecurity.Paths (Pouw.RelativeSecurity.Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n →
          Fin n → ℕ → ℕ → V)
      (salt : Fin B → ((i : I) → Ω i) →
        Pouw.RelativeSecurity.Paths (Pouw.RelativeSecurity.Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n →
          Fin n → ByteArray)
      (Wp : Fin B → ((i : I) → Ω i) →
        Accounting.Compose.Period Cn (Pouw.RelativeSecurity.Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ))
          (Fin 144) Cd.rounds n)
      (Pl : Fin B → ((i : I) → Ω i) →
        Accounting.TS.Play (Pouw.RelativeSecurity.Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n K T L U)
      (folded : Fin B → ((i : I) → Ω i) →
        Pouw.RelativeSecurity.Paths (Pouw.RelativeSecurity.Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → ℝ)
      (macs : Accounting.Compose.Kind → ℕ) (u bUnit : ℕ)
      (help : Fin B → ((i : I) → Ω i) →
        Pouw.RelativeSecurity.Paths (Pouw.RelativeSecurity.Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → ℝ),
    ∀ RW : (b : Fin B) → Accounting.RealizesCreditedWords (r b) (c b) p (Accounting.Compose.minWordMacs Sc Cd macs) u
      bUnit (Pouw.RelativeSecurity.Paths (Pouw.RelativeSecurity.Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n)
      (fun ξ => (Pl b ξ).accepts)
      (fun ξ x => Accounting.Compose.creditPrefix Sc Cd 144 ((ws b ξ).prefixLen prm x)
        (Properties.Pouw.Period.checkedLive Sc Cd (Wp b ξ) prm (ws b ξ) x) 1 1 (67 / 8) δs),
    (∀ b, Accounting.Assumptions.WordCostsMacs (r b) p (Accounting.Compose.minWordMacs Sc Cd macs) u (RW b).coin
      (fun ξ => (Pl b ξ).accepts) (fun ξ x => n * ((Pl b ξ).run x).fixed) (help b)) →
    (∀ b, Accounting.Assumptions.HelpWithinCertifier (r b) (c b) bUnit (RW b).coin (fun ξ => (Pl b ξ).accepts)
      (help b)) →
    Accounting.Compose.Scope Sc Cd δs →
    (∀ t : Fin 144, (8 / 9 : ℝ) * (144 : ℕ) * Accounting.Compose.shareCone Sc Cd 144 t ≤ 1) →
    (∀ j, Accounting.Compose.stageShare (Accounting.Compose.tileStage Cd 144)
      (Accounting.Compose.shareCone Sc Cd 144) j ≤ 1) →
    (∀ t : Fin 144, 0 < Accounting.Compose.shareCone Sc Cd 144 t →
      Accounting.Compose.tileLuck Sc Cd 144 t ≤ 7 / 8 ∧ Accounting.Compose.tileLuck0 Sc Cd 144 t ≤ 1 / 2) →
    (∀ b ξ, (Pl b ξ).game = Accounting.Compose.periodGame (Accounting.Compose.tileStage Cd 144)
      (Accounting.Compose.liveCone Sc Cd 144) 1 1 24 (Wp b ξ)) →
    (∀ b ξ, (Wp b ξ).EveryWindow (fun live _ => Accounting.Compose.DeadConeDead Sc Cd 144 live)) →
    (∀ b ξ, Accounting.Compose.PrefixOf
      (Properties.Pouw.Period.checkedLayer Cd hr hK prm (ws b ξ) (opened b ξ)
        (Properties.Pouw.Period.stagedAt (salt b ξ) staged) (Properties.Pouw.Period.flatAtCoin hr enc flat))
      (Pl b ξ).game) →
    (∀ b ξ w, Pouw.RelativeSecurity.prob (fun x =>
      (Properties.Pouw.Period.checkedLayer Cd hr hK prm (ws b ξ) (opened b ξ)
        (Properties.Pouw.Period.stagedAt (salt b ξ) staged)
        (Properties.Pouw.Period.flatAtCoin hr enc flat)).MovedIn x w) ≤ δbind) →
    (∀ b ξ, Properties.Pouw.Period.RegisteredBeforeDraw (ws b ξ)) →
    (∀ b ξ, Properties.Pouw.Period.DrawIsCoin hK prm (ws b ξ)) →
    (∀ b ξ, Properties.Pouw.Period.ServiceSound prm (ws b ξ) (opened b ξ)
      (Properties.Pouw.Period.stagedAt (salt b ξ) staged) δks) →
    (∀ b ξ, Properties.Pouw.Period.SaltIsLiveCoin hr (salt b ξ) enc) → Properties.Pouw.Period.StagedPins staged flat →
    (∀ b ξ, (Wp b ξ).EveryWindow (fun live W => Accounting.Compose.PerLiveTile Sc Cd live W
      (fun k G _ => Accounting.Assumptions.TS.OperandHMu (Sc.mu k) G))) →
    (∀ b ξ, (Wp b ξ).EveryWindow (fun live W => Accounting.Compose.PerLiveTile Sc Cd live W
      (fun k G _ => Accounting.Assumptions.TS.ReceiptLuckAt (Sc.rB k) G))) →
    (∀ b ξ, (Wp b ξ).EveryWindow (fun live W => Accounting.Compose.PerLiveTile Sc Cd live W
      (fun k _ G0 => Accounting.Assumptions.Compose.GuessLuckAt (Sc.guess k) G0))) →
    (∀ b ξ, Accounting.Assumptions.TS.UnitFixed (Pl b ξ)
      (Properties.Pouw.Period.checkedLayer Cd hr hK prm (ws b ξ) (opened b ξ)
        (Properties.Pouw.Period.stagedAt (salt b ξ) staged) (Properties.Pouw.Period.flatAtCoin hr enc flat))
      (Accounting.Compose.periodMode Cd (Wp b ξ)
        (Properties.Pouw.Period.checkedRun hK prm (ws b ξ) (opened b ξ)
          (Properties.Pouw.Period.stagedAt (salt b ξ) staged) (Properties.Pouw.Period.flatAtCoin hr enc flat)))
      (fun _ t => (Accounting.Compose.tileWords Sc Cd 144 t : ℝ)) (folded b ξ)) →
    -- the network certifier: its period of sampled audits
    ∀ {Coin Col Tn : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
      {H : Msg → Msg} {D : Deploy Coin Col Tn} {Ln Kₒ Sn : ℕ} {lay : Stmt → Layout Ln Sn} {kh : Stmt → ℕ} {δc : ℝ}
      {decode : List (ℕ × Col) → List Tn} {regN : Stmt → Finset ℕ} {Dj nc : ℕ},
    ∀ RN : (b : Fin B) →
      Accounting.JointBridge.RealizesCommOfPeriodLeakAccepted (r b) (c b).K H D Ln Kₒ Sn lay kh δc decode regN Dj nc,
    KS H D → D.commit.Binds H → 0 ≤ δc → Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode →
    ∀ εks εbind εtap : ℝ,
    (∀ b ξ i, Claims.Sampling.κ D δc ((RN b).stmt ξ i) - δc +
      Real.sqrt (Claims.Sampling.γ D ((RN b).stmt ξ i) * PrB H D.V ((RN b).prover ξ i) ((RN b).stmt ξ i)) ≤ εks) →
    (∀ b ξ i, (Bext H D.V (E D decode) ((RN b).Tape ξ i) ((RN b).prover ξ i) ((RN b).stmt ξ i)).Pr H ≤ εbind) →
    (∀ b ξ i, (Btap H D.V (E D decode) D.commit D.parse (RN b).tops ((RN b).Tape ξ i) ((RN b).prover ξ i)
      ((RN b).stmt ξ i)).Pr H ≤ εtap) →
    Pouw.pr (fun ξ => ∃ b, ∃ a ∈ ({.sampled, .pouw, .network} : Finset Audit),
        (r b).accepts a ξ ∧ ¬ (r b).Holds (c b) p a ξ) ≤
      ∑ b : Fin B, ((ENNReal.ofReal δI + S.err b).toReal + (δs + n * (δbind + δks)) + nc * (δc + εks + εbind + εtap))

end Accounting.Guarantees
