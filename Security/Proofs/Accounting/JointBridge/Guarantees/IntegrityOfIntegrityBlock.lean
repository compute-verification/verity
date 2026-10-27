import Definitions.Accounting.JointBlocks
import Proofs.Flock.EndToEnd.Blocks.Statement

/-!
# The integrity audit of a server-time block is C-Flock's integrity audit of that block (trusted)

A server-time block's sampled-proofs audit (`Audit.sampled` of the block's period, `Accounting.PeriodRun`) is C-Flock's
integrity audit of block `b` of a server's horizon of sessions (`Flock.Guarantees.IntegrityBlock`): the sessions `t` in
`S.sessions b`, each a sampling block `S.W t` (`Flock.Guarantees.SamplingBlock`), a `--zk` session under hidden outputs
with a fresh live subset draw, played by one adaptive prover over the whole horizon
(`Flock.Guarantees.ServerBlocks.run`). `RealizesIntegrityOfIntegrityBlock` ties the block's period to it: the period's
sampled audit's own coins drive the horizon's game at the prover's strategy, as far as block `b`'s sampling blocks'
records go, its verdict is flock-verify's on every sampling block of the server-time block, and the claim's units and
wrong units are those sampling blocks'.

`IntegrityOfIntegrityBlock` is `PeriodRun.Sound` for the server-time block's sampled audit at error
`δ + Σ_{t ∈ b} (ε_ks,t + δ_link,t)`, `IntegrityBlock`'s: the coins model's per-period guarantee, which the one theorem's
sampled-proofs target couples into the causal game (`Accounting.PeriodLive`).

C-Flock's definitions are still in its proofs package, so, like `Flock.Guarantees.IntegrityBlock`, this module imports
them.
-/

namespace Accounting

open FlockSoundness FlockSoundness.Game Flock.Guarantees
open scoped ENNReal

/-- The period `r` of server-time block `b` has as its sampled-proofs audit C-Flock's integrity audit of block `b` of
the server's horizon of sessions `S` (`Flock.Guarantees.ServerBlocks`), each session a sampling block `S.W t`. Every
field is a property of the deployed run:

* `strat`, `record`, `law`: the node is one prover strategy of the horizon's game at each value of the other coins
  (every other audit's, and every other server-time block's), and the period's sampled audit's own coins
  (`r.own .sampled`), drawn uniformly, drive that game as far as block `b`'s sampling blocks go: every event of their
  records has the probability, over those coins, that the game's run gives the records of the sessions in
  `S.sessions b`, at that strategy. This is the live coin flow: each of the block's subset draws comes from the
  verifier's byte source, which the period's sampled coordinate is (A3,
  `FlockSoundness.Assumptions.UniformRandomBytes`, stays a hypothesis of the guarantee). The sessions of the
  server-time blocks before `b` are played at the coins they were, which the strategy sees through `ξ`.
* `accepts_iff`: the period's sampled audit accepts exactly when flock-verify accepts every sampling block of the
  server-time block.
* `wrong_eq`: the wrong units of the period's partition in its committed transcript are, in number, the block's
  sampling blocks' wrong units (`SamplingBlock.wrongSet`: `wrongRegZ` in the transcript fixed at registration, over the
  key program's partition). This is the program-from-circuit correspondence (M1: the key program and its partition are
  the circuit `C`'s lowering and `r.parts`), which the bridge PRs land as `Lowers`/`progOf`/`partOf`; until then it is
  this field.
* `units_eq`, `gates_nonempty`: the partition's public unit-size property. Its units are the block's sampling blocks'
  units, and every unit has at least one of the circuit's gates, so no empty unit, trivially correct, dilutes the share
  `ε` the claim bounds. -/
structure RealizesIntegrityOfIntegrityBlock {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit}
    {I : Type} {Ω : I → Type} [DecidableEq I] [∀ i, Fintype (Ω i)] (r : PeriodRun d α C Ω)
    (S : Flock.Guarantees.ServerBlocks) (b : ℕ) where
  /-- the prover strategy the node plays the horizon's game with, at the coins (it reads only the other coins) -/
  strat : ((i : I) → Ω i) → FlockSoundness.Game.Strategy S.game
  /-- block `b`'s sampling blocks' records at the coins: flock-verify's verdict and the wrong units of each -/
  record : ((i : I) → Ω i) → List Flock.Guarantees.WindowRec
  /-- the period's sampled audit's own coins drive the horizon's game at the strategy, on block `b`'s sampling blocks -/
  law : ∀ (ξ : (i : I) → Ω i) (E : List Flock.Guarantees.WindowRec → Prop),
    ENNReal.ofReal (Pouw.pr fun x : Ω (r.own .sampled) => E (record (Function.update ξ (r.own .sampled) x))) =
      FlockSoundness.Game.value (fun l => E (l.filter fun w => w.idx ∈ S.sessions b)) (S.run (strat ξ))
  /-- the sampled audit accepts exactly when flock-verify accepts every sampling block of the server-time block -/
  accepts_iff : ∀ ξ, r.accepts .sampled ξ ↔ ∀ w ∈ record ξ, w.accepted
  /-- the period's wrong units are its sampling blocks' (M1, the program-from-circuit correspondence) -/
  wrong_eq : ∀ ξ, (r.parts.wrong (r.transcript ξ)).card = ((record ξ).map Flock.Guarantees.WindowRec.wrong).sum
  /-- the partition's units are the block's sampling blocks' units -/
  units_eq : r.units = ∑ t ∈ S.sessions b, (S.W t).n
  /-- every unit of the partition has at least one gate -/
  gates_nonempty : ∀ u, (r.parts.gates u).Nonempty

namespace Guarantees.Bridge

open FlockSoundness.Audit FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.ZkLink
  FlockSoundness.Discharge.ZkHidden in
/-- **The integrity audit of a server-time block, from C-Flock's integrity audit of that block.** For the period `r` of
server-time block `b` whose sampled-proofs audit realizes C-Flock's integrity audit of block `b` of the server's
sessions `S` (`RealizesIntegrityOfIntegrityBlock`), each of the block's draws covering at least a fraction `q > 0` of
its sampling block's units, under A3 for each of the block's draw bytes and `cr/sha-512` (`TableCRZ`) and
`ecr/sha-512` (`LinkCRZC`) at every strategy the node plays a sampling block with on some run (`Period.Plays`), at
certificates whose share `ε` of the block's units is at least `ln(1/δ)/q`, the period's sampled audit has its
per-period guarantee (`PeriodRun.Sound`) at error
$$\delta + \sum_{t \in b} (\varepsilon_{ks,t} + \delta_{link,t}),$$
`Flock.Guarantees.IntegrityBlock`'s, when that is finite. -/
def IntegrityOfIntegrityBlock : Prop :=
  ∀ (ι : Type) (d : Device ι) (α : Type) [Fintype α] (C : Verity.Circuit) (I : Type) (Ω : I → Type)
    [DecidableEq I] [∀ i, Fintype (Ω i)] (r : PeriodRun d α C Ω) (S : Flock.Guarantees.ServerBlocks) (b : ℕ)
    (q δ : ℝ), 0 < q → 0 < δ →
    (∀ t ∈ S.sessions b, q ≤ ((S.W t).kd : ℝ) / (S.W t).n) →
    (∀ t ∈ S.sessions b, FlockSoundness.Assumptions.UniformRandomBytes (S.W t).Lb (S.W t).src) →
    ∀ R : RealizesIntegrityOfIntegrityBlock r S b,
    (∀ ξ t (τ : Strategy (S.W t).game), Flock.Guarantees.Period.Plays S.W S.T (R.strat ξ) t τ → ∀ ω j,
      TableCRZ execArith hs512 (planZHJ (S.W t).dj (S.W t).y₀ ((S.W t).L.draw ω) (reg ((S.W t).sim τ))) j
        (zcStrat execArith hs512 (planZHJ (S.W t).dj (S.W t).y₀) (cont ((S.W t).sim τ) ω)) (S.W t).qF (S.W t).qS) →
    (∀ ξ t (τ : Strategy (S.W t).game), Flock.Guarantees.Period.Plays S.W S.T (R.strat ξ) t τ →
      LinkCRZC execArith hs512 (S.W t).P (planZHJ (S.W t).dj (S.W t).y₀) (tabZHJ (S.W t).dj (S.W t).y₀) (S.W t).vb
        (reg ((S.W t).sim τ)) (cont ((S.W t).sim τ)) (S.W t).k (S.W t).Rw (S.W t).M (S.W t).t') →
    ∀ (c : Certificates) (p : Price), Real.log (1 / δ) / q ≤ c.ε * r.units →
    ENNReal.ofReal δ + S.err b ≠ ⊤ →
    r.Sound c p .sampled (ENNReal.ofReal δ + S.err b).toReal

end Guarantees.Bridge

end Accounting
