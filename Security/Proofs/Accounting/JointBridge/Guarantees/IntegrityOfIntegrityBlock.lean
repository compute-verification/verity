import Definitions.Accounting.JointBlocks
import Proofs.Flock.EndToEnd.Blocks.Statement

/-!
# The integrity audit of a server-time block is C-Flock's integrity audit of that block (trusted)

Block `b`'s sampled-proofs audit (`Audit.sampled` of `BlocksRun.block b`) is C-Flock's integrity audit of block `b` of a
server's horizon of sessions (`Flock.Guarantees.IntegrityBlock`): the sessions `W t` for `t` in `S.sessions b`, each a
`--zk` session under hidden outputs with a fresh live subset draw, played by one adaptive prover over the whole horizon
(`Flock.Guarantees.ServerBlocks.run`). `RealizesIntegrityOfIntegrityBlock` ties block `b` of a `BlocksRun` to it: the
block's sampled audit's own coins drive the horizon's game at the prover's strategy, as far as block `b`'s sessions'
records go, its verdict is flock-verify's on every session of the block, and the claim's units and wrong units are the
block's sessions'.

* `IntegrityOfIntegrityBlock`: `PeriodRun.Sound` for block `b`'s sampled audit at error `δ + Σ_{t ∈ b} (ε_ks,t +
  δ_link,t)`, `IntegrityBlock`'s.
* `JointHorizonBlocksOfIntegrityBlock`: `Accounting.Guarantees.JointHorizonBlocks` with every block's integrity leg
  given by its realization, so its sampled audit's error is that block's `IntegrityBlock` error.

C-Flock's definitions are still in its proofs package, so, like `Flock.Guarantees.IntegrityBlock`, this module imports
them.
-/

namespace Accounting

open FlockSoundness FlockSoundness.Game Flock.Guarantees
open scoped ENNReal

/-- Block `b`'s sampled-proofs audit is C-Flock's integrity audit of block `b` of the server's horizon of sessions `S`
(`Flock.Guarantees.ServerBlocks`). Every field is a property of the deployed run:

* `strat`, `record`, `law`: the node is one prover strategy of the horizon's game at each value of the other coins
  (every other audit's, and every other block's), and block `b`'s sampled audit's own coins
  (`(s.block b).own .sampled`), drawn uniformly, drive that game as far as block `b`'s sessions go: every event of
  block `b`'s sessions' records has the probability, over those coins, that the game's run gives the records of the
  sessions in `S.sessions b`, at that strategy. This is the live coin flow: each of the block's subset draws comes from
  the verifier's byte source, which the block's sampled coordinate is (A3,
  `FlockSoundness.Assumptions.UniformRandomBytes`, stays a hypothesis of the guarantee). The sessions of the blocks
  before `b` are played at the coins they were, which the strategy sees through `ξ`.
* `accepts_iff`: block `b`'s sampled audit accepts exactly when flock-verify accepts every session of the block.
* `wrong_eq`: the wrong units of block `b`'s partition in its committed transcript are, in number, the block's
  sessions' wrong units (`HiddenWindow.wrongSet`: `wrongRegZ` in the transcript fixed at registration, over the key
  program's partition). This is the program-from-circuit correspondence (M1: the key program and its partition are the
  circuit `C`'s lowering and the block's `parts`), which the bridge PRs land as `Lowers`/`progOf`/`partOf`; until then
  it is this field.
* `units_eq`, `gates_nonempty`: the partition's public unit-size property. Its units are the block's sessions' units,
  and every unit has at least one of the circuit's gates, so no empty unit, trivially correct, dilutes the share `ε` the
  claim bounds. -/
structure RealizesIntegrityOfIntegrityBlock {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit}
    {I : Type} {Ω : I → Type} [DecidableEq I] [∀ i, Fintype (Ω i)] {B : ℕ} (s : BlocksRun d α C Ω B)
    (S : Flock.Guarantees.ServerBlocks) (b : Fin B) where
  /-- the prover strategy the node plays the horizon's game with, at the coins (it reads only the other coins) -/
  strat : ((i : I) → Ω i) → FlockSoundness.Game.Strategy S.game
  /-- block `b`'s sessions' records at the coins: flock-verify's verdict and the wrong units of each session -/
  record : ((i : I) → Ω i) → List Flock.Guarantees.WindowRec
  /-- block `b`'s sampled audit's own coins drive the horizon's game at the strategy, on block `b`'s sessions -/
  law : ∀ (ξ : (i : I) → Ω i) (E : List Flock.Guarantees.WindowRec → Prop),
    ENNReal.ofReal (Pouw.pr fun x : Ω ((s.block b).own .sampled) =>
        E (record (Function.update ξ ((s.block b).own .sampled) x))) =
      FlockSoundness.Game.value (fun l => E (l.filter fun w => w.idx ∈ S.sessions b)) (S.run (strat ξ))
  /-- block `b`'s sampled audit accepts exactly when flock-verify accepts every session of the block -/
  accepts_iff : ∀ ξ, (s.block b).accepts .sampled ξ ↔ ∀ w ∈ record ξ, w.accepted
  /-- the block's wrong units are its sessions' (M1, the program-from-circuit correspondence) -/
  wrong_eq : ∀ ξ, ((s.block b).parts.wrong ((s.block b).transcript ξ)).card =
    ((record ξ).map Flock.Guarantees.WindowRec.wrong).sum
  /-- the partition's units are the block's sessions' units -/
  units_eq : (s.block b).units = ∑ t ∈ S.sessions b, (S.W t).n
  /-- every unit of the partition has at least one gate -/
  gates_nonempty : ∀ u, ((s.block b).parts.gates u).Nonempty

namespace Guarantees.Bridge

/-- **The integrity audit of a server-time block, from C-Flock's integrity audit of that block.** For block `b` of a
horizon of server-time blocks `s` whose sampled-proofs audit realizes C-Flock's integrity audit of block `b` of the
server's sessions `S` (`RealizesIntegrityOfIntegrityBlock`), each of the block's draws covering at least a fraction
`q > 0` of its session's units, under A3 for each of the block's draw bytes and `cr/sha-512`, `ecr/sha-512` at the
strategy the node plays (`Period.CR`), at certificates whose share `ε` of the block's units is at least `ln(1/δ)/q`,
block `b`'s sampled audit has its per-period guarantee (`PeriodRun.Sound`) at error
$$\delta + \sum_{t \in b} (\varepsilon_{ks,t} + \delta_{link,t}),$$
`Flock.Guarantees.IntegrityBlock`'s, when that is finite. -/
def IntegrityOfIntegrityBlock : Prop :=
  ∀ (ι : Type) (d : Device ι) (α : Type) [Fintype α] (C : Verity.Circuit) (I : Type) (Ω : I → Type)
    [DecidableEq I] [∀ i, Fintype (Ω i)] (B : ℕ) (s : BlocksRun d α C Ω B) (S : Flock.Guarantees.ServerBlocks)
    (b : Fin B) (q δ : ℝ), 0 < q → 0 < δ →
    (∀ t ∈ S.sessions b, q ≤ ((S.W t).kd : ℝ) / (S.W t).n) →
    (∀ t ∈ S.sessions b, FlockSoundness.Assumptions.UniformRandomBytes (S.W t).Lb (S.W t).src) →
    ∀ R : RealizesIntegrityOfIntegrityBlock s S b, (∀ ξ, Flock.Guarantees.Period.CR S.W S.T (R.strat ξ)) →
    ∀ (c : Fin B → Certificates) (p : Price), Real.log (1 / δ) / q ≤ (c b).ε * (s.block b).units →
    ENNReal.ofReal δ + S.err b ≠ ⊤ →
    (s.block b).Sound (c b) p .sampled (ENNReal.ofReal δ + S.err b).toReal

/-- **The joint guarantee over a horizon of server-time blocks, with every block's integrity leg from C-Flock's
integrity audit of that block.** As `Accounting.Guarantees.JointHorizonBlocks`, with block `b`'s sampled audit given
by its realization of block `b` of the server's sessions `S` (`RealizesIntegrityOfIntegrityBlock`), under the premises
of `IntegrityOfIntegrityBlock` at every block, in place of its per-period guarantee: the probability that some block's
accepted audit's claim fails, some accepted wipe's survivor bound fails, or some block's memory term fails while its
PoUS accepted is at most
$$\sum_b \Big(\delta + \sum_{t \in b} (\varepsilon_{ks,t} + \delta_{link,t}) + \sum_{a \ne \mathrm{sampled}} \delta_{b,a}\Big)
  + \sum_b \varepsilon_{\mathrm{wipe},b} + \sum_b \delta_{\mathrm{PoUS},b}.$$ -/
def JointHorizonBlocksOfIntegrityBlock : Prop :=
  ∀ (ι α I : Type) [Fintype ι] [Fintype α] [Fintype I] [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)]
    [∀ i, Nonempty (Ω i)] (d : Device ι) (C : Verity.Circuit) (B : ℕ) (s : BlocksRun d α C Ω B)
    (S : Flock.Guarantees.ServerBlocks) (q δI : ℝ), 0 < q → 0 < δI →
    (∀ b : Fin B, ∀ t ∈ S.sessions b, q ≤ ((S.W t).kd : ℝ) / (S.W t).n) →
    (∀ b : Fin B, ∀ t ∈ S.sessions b, FlockSoundness.Assumptions.UniformRandomBytes (S.W t).Lb (S.W t).src) →
    ∀ R : (b : Fin B) → RealizesIntegrityOfIntegrityBlock s S b,
    (∀ b ξ, Flock.Guarantees.Period.CR S.W S.T ((R b).strat ξ)) →
    ∀ (c : Fin B → Certificates) (p : Price) (f : ℝ) (δ : Fin B → Audit → ℝ) (εWipe δPous : Fin B → ℝ),
    (∀ b, Real.log (1 / δI) / q ≤ (c b).ε * (s.block b).units) →
    (∀ b, ENNReal.ofReal δI + S.err b ≠ ⊤) →
    (∀ b a, a ≠ .sampled → (s.block b).Sound (c b) p a (δ b a)) →
    (∀ b, Pouw.pr (fun ξ => (s.wipe b).accepts ξ ∧ ¬ (s.wipe b).Survives f ξ) ≤ εWipe b) →
    (∀ b, Pouw.pr (fun ξ => (s.block b).accepts .pous ξ ∧ ¬ s.MemoryTerm b ξ) ≤ δPous b) →
    Pouw.pr (fun ξ => (∃ b a, (s.block b).accepts a ξ ∧ ¬ (s.block b).Holds (c b) p a ξ) ∨
        (∃ b, (s.wipe b).accepts ξ ∧ ¬ (s.wipe b).Survives f ξ) ∨
        (∃ b, (s.block b).accepts .pous ξ ∧ ¬ s.MemoryTerm b ξ)) ≤
      ∑ b, ((ENNReal.ofReal δI + S.err b).toReal + ∑ a ∈ Finset.univ.erase .sampled, δ b a) +
        ∑ b, εWipe b + ∑ b, δPous b

end Guarantees.Bridge

end Accounting
