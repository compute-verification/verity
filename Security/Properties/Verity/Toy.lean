import CoreToy
import Properties.Verity.Game

/-!
# The toy deployment's verifier as a `Deployment`, its vocabulary, and its honest lemma's statement

The toy deployment is `integrations/torch_loop` served under sampled proofs plus the network certifier. Its verifier of
record is `Core.Toy.protocol` (`verity/toy/v0`): NCI's registration check on the registration sampled proofs'
one-stage audit commits to, then the network certifier's period check and sampled proofs' one-stage audit v2 over the
blobs the prover sent. As a `Deployment` (`deployment`) it decides two claims, sampled proofs' (`.sampled`) and the
network certifier's (`.network`); PoUW's, PoUS's and the bandwidth audit's it decides by no verdict.

**`Given`** holds the `Vocab` fields that no owner has defined on `main` yet, each with its owner in its docstring;
each field leaves `Given` in the PR that defines it. `vocab G` is the toy's `Vocab` from them. Its fields indexed by
`Audit` are the toy's at the claims it decides (`ofAudit`) and a fixed inert value at the others (no releases read, the
finder that outputs nothing, no calls, no collision bound, error `⊤`), and PoUW's gates are none (`computed := ∅`). Its
honest prover (`honest`) sends, in each of the device's rounds, the bytes the honest prover computes from the releases
so far (`Given.answer`), and then the evidence from the device's log (`Given.evidence`): the prover of
`benchmarks/prove_path/records_audit.py` and `serve.py`, as a function of the coins. That prover runs C-Flock and reads
files through `partial` parses, so its outputs are taken as `Given` fields and the toy's bytes are never exhibited.

**The honest lemma's statement (`SoundnessHonest`).** `Vocab.HonestHolds a` holds the honest prover to `a`'s claim at
every commitment `c`, and sampled proofs' claim reads `c` (`Vocab.Opens`), so at a commitment that opens nothing it
fails on every play (`Verity.Vocab.not_honestHolds_sampled_of_unopened`, in the proofs). The toy's statement is at the
honest deployed instance instead: at registered files, the commitment the honest prover sends (`Given.commit`), then
the honest strategy (`HonestHoldsAt`, the restatement proposed to ONETHM on PRF-193). It takes two named premises
until their owners define the fields they read: `SampledHonest` (the extractor's transcript at the honest strategy is
correct and the honest commitment opens it: ONE-8's extractor, ONE-7's positions) and `NetworkHonest`
(network-accounting's `Verity.networkComponentBound_honest`, stated exactly as named on PRF-193 until it lands).
-/

namespace Verity.Toy

open Accounting Core
open scoped ENNReal

/-- The claims the toy decides. -/
inductive Index
  | sampled
  | network
  deriving DecidableEq

/-- The claim each index certifies. -/
def audit : Index → Audit
  | .sampled => .sampled
  | .network => .network

/-- The index of a claim the toy decides. -/
def ofAudit : Audit → Option Index
  | .sampled => some .sampled
  | .network => some .network
  | _ => none

/-- The toy's component deciding each claim. -/
def comp : Index → BlobProtocol
  | .sampled => SampledProofs.OneStageZk.protocol
  | .network => NetworkCheck.Period.protocol

/-- **The toy deployment** (`Core.Toy.protocol`): its two components, deciding sampled proofs' and the network
certifier's claims. -/
def deployment : Deployment where
  id := "verity/toy/v0"
  components := Core.Toy.components
  comp a := (ofAudit a).map comp
  comp_mem a P h := by
    cases a <;> simp [ofAudit] at h <;> subst h <;> simp [comp, Core.Toy.components]

example : deployment.protocol = Core.Toy.protocol := rfl

/-- The finder that outputs nothing, on one coin, at no calls: the inert finder at the claims the toy doesn't decide. -/
def inertFinder : Claims.Finder where
  S := Unit
  out _ := none
  cost _ := 0

/-- **The `Vocab` fields no owner has defined on `main` yet**, and the toy's prover's outputs, which have no Lean value.

* `read`: v2's reading of the committed blob (ONE-3b, ONE3B).
* `place`, `programDigest`, `registered`: the positions, the circuit file's digest in the parameters and the registered
  blob (ONE-7; #4754 gives `Flock.Bridge.circuitOf`, none of these).
* `rounds`, `coinsRead`: the device's rounds and the releases each component's check read (ONE-4, #4888).
* `extract`: the extractor applied to a strategy (ONE-8, `XplurZC`).
* `finder`, `callsMul`, `callsAdd`, `finder_calls`: each decided claim's collision finder and its SHA-512 calls in
  those of the strategy (ONE-8 for sampled proofs, network-accounting for the network certifier).
* `cost`: the strategy's SHA-512 calls, declared, as `Vocab.cost` is (`Verity.Game` has no call model; ONETHM).
* `world`: the world each play realizes (ONE-12, the network certifier's realized run, `commWithin`).
* `certOf`, `crBound`, `delta`, `loss`: what the product publishes, from the parameters alone (ONE-19; its `deltaPub`
  is in the proofs tree, which a statement can't read).
* `commit`, `answer`, `evidence`: the honest prover's commitment, each round's bytes from the releases so far, and its
  evidence from the device's log (`records_audit.py`, `serve.py`: C-Flock's prover and the integration's commit). -/
structure Given {ι : Type} (d : Device ι) (α : Type) [Fintype α] where
  read : ByteArray → FileRead
  place : (θ F c : ByteArray) → Fin (read F).circuit.N → Option Place
  programDigest : ByteArray → Option Hash.Digest
  registered : ByteArray → ByteArray
  rounds : ByteArray → List ℕ
  coinsRead : Index → Sent → List Release
  extract : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → (read F).circuit.Transcript
  finder : Index → (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → Claims.Finder
  cost : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → ℕ
  callsMul : ByteArray → Index → ℕ
  callsAdd : ByteArray → Index → ℕ
  finder_calls : ∀ k θ F c σ (x : (finder k θ F c σ).S),
    (finder k θ F c σ).cost x ≤ callsMul θ k * cost θ F c σ + callsAdd θ k
  world : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → Outcome → World d α
  certOf : ByteArray → Certificates
  crBound : ByteArray → Index → ℕ → ℝ
  delta : ByteArray → Index → ℕ → ℝ≥0∞
  loss : Index → ByteArray → ℝ≥0∞ → ℝ≥0∞
  commit : ByteArray → ByteArray → ByteArray
  answer : (θ F c : ByteArray) → List Release → ByteArray
  evidence : (θ F c : ByteArray) → List Release → ByteArray

/-- A strategy of `g.bind f`: `s` in `g`, then `t a` in `f a` once `g` ended with `a`. -/
def bindStrategy {α β : Type} : (g : Game α) → (f : α → Game β) → Game.Strategy g →
    ((a : α) → Game.Strategy (f a)) → Game.Strategy (g.bind f)
  | .ret a, _, _, t => t a
  | .send _ k, f, s, t => ⟨s.1, bindStrategy (k s.1) f s.2 t⟩
  | @Game.coin _ _ _ _ k, f, s, t => fun c => bindStrategy (k c) f (s c) t

/-- **The device's rounds against a prover that answers from the releases so far**: before each release, the prover
sends `ans pre`, where `pre` is the releases before it. -/
def releasesBy (ans : List Release → ByteArray) (pre : List Release) :
    (ns : List ℕ) → Game.Strategy (releases ns)
  | [] => PUnit.unit
  | _ :: ns => ⟨ans pre, fun x => Game.Strategy.toMap _ (releases ns)
      (releasesBy ans (pre ++ [⟨ans pre, List.ofFn x⟩]) ns)⟩

/-- **The toy's honest prover**: each round's bytes from the releases so far, then the evidence from the device's
log. -/
def honest {ι : Type} {d : Device ι} {α : Type} [Fintype α] (G : Given d α) (θ F c : ByteArray) :
    Game.Strategy (play (G.rounds θ) θ F c) :=
  bindStrategy (releases (G.rounds θ)) (fun log => .send ByteArray fun e => .ret (⟨θ, F, c, log, e⟩ : Outcome))
    (releasesBy (G.answer θ F c) [] (G.rounds θ)) fun log => ⟨G.evidence θ F c log, PUnit.unit⟩

/-- **The toy's vocabulary**: `G`'s fields, the toy's at the claims it decides and inert at the others, no gates
computed, and the toy's honest prover. -/
noncomputable def vocab {ι : Type} {d : Device ι} {α : Type} [Fintype α] (G : Given d α) : Vocab d α where
  read := G.read
  place := G.place
  programDigest := G.programDigest
  registered := G.registered
  rounds := G.rounds
  coinsRead a s := match ofAudit a with
    | some k => G.coinsRead k s
    | none => []
  extract := G.extract
  finder a θ F c σ := match ofAudit a with
    | some k => G.finder k θ F c σ
    | none => inertFinder
  cost := G.cost
  callsMul θ a := match ofAudit a with
    | some k => G.callsMul θ k
    | none => 0
  callsAdd θ a := match ofAudit a with
    | some k => G.callsAdd θ k
    | none => 0
  finder_calls a θ F c σ x := by
    cases a
    · exact G.finder_calls .sampled θ F c σ x
    · exact Nat.zero_le _
    · exact Nat.zero_le _
    · exact Nat.zero_le _
    · exact G.finder_calls .network θ F c σ x
  computed _ _ _ _ _ := ∅
  world := G.world
  honest := honest G
  certOf := G.certOf
  crBound θ a n := match ofAudit a with
    | some k => G.crBound θ k n
    | none => 0
  delta θ a n := match ofAudit a with
    | some k => G.delta θ k n
    | none => ⊤
  loss a θ p := match ofAudit a with
    | some k => G.loss k θ p
    | none => ⊤

/-- **Claim `a`'s honest lemma at the honest deployed instance**: at registered files, the honest prover, after
sending its commitment `commit θ F`, meets `a`'s claim on almost every play. `Vocab.HonestHolds` with the commitment
the honest prover sends in place of every commitment. -/
def HonestHoldsAt {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)
    (commit : ByteArray → ByteArray → ByteArray) (a : Audit) : Prop :=
  ∀ θ F : ByteArray, V.Registered θ F →
    Game.prob (fun o => ¬ V.Holds θ F (commit θ F) (V.honest θ F (commit θ F)) a o) (V.game θ F (commit θ F))
      (V.honest θ F (commit θ F)) = 0

/-- **The honest prover's sampled proofs case** (premise; owners: ONE-8 for the extractor, ONE-7 for the positions):
at registered files and the honest commitment, the extractor's transcript at the honest strategy is correct on every
unit, the commitment opens every gate to it (a registered constant to its constant), and the published share `ε` is
not negative. It leaves the toy's premises once `extract` and `place` are defined and it is proved, from
`Flock.Bridge.card_wrong_lt_of_relS_honest` (the honest witness opens every committed value and no unit is wrong). -/
def SampledHonest {ι : Type} {d : Device ι} {α : Type} [Fintype α] (G : Given d α) : Prop :=
  ∀ θ F : ByteArray, (vocab G).Registered θ F →
    (∀ u, ((vocab G).partsOf F).Correct (G.extract θ F (G.commit θ F) (honest G θ F (G.commit θ F))) u) ∧
      (∀ g, (vocab G).Opens θ F (G.commit θ F) g (G.extract θ F (G.commit θ F) (honest G θ F (G.commit θ F)) g)) ∧
      0 ≤ (G.certOf θ).ε

/-- **The network certifier's honest lemma at the toy** (premise; owner network-accounting, ONE-12):
`Verity.networkComponentBound_honest` (`Properties.Verity.Network`) as named on PRF-193, `Vocab.HonestHolds` at the
network certifier's claim: on every play of the honest strategy at a registered file, what the network certifiers saw
the node move is within the published `K`. Its use here becomes a citation once it lands. -/
def NetworkHonest {ι : Type} {d : Device ι} {α : Type} [Fintype α] (G : Given d α) : Prop :=
  (vocab G).HonestHolds (audit .network)

/-- **The toy's honest lemma**: on any device, at any `Given`, where the honest prover's sampled proofs case and the
network certifier's honest lemma hold, every claim the toy deployment decides holds on almost every play of the honest
prover at registered files and the honest commitment. -/
def SoundnessHonest : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (G : Given d α), SampledHonest G → NetworkHonest G →
    ∀ a, deployment.Decides a → HonestHoldsAt (vocab G) G.commit a

end Verity.Toy
