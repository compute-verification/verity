import Properties.Verity.Game

/-!
# Onsite's honest instance: its vocabulary and its honest lemma's statement

Protocol soundness's honest instance is onsite itself (`Verity.onsite`, `verity/onsite/v0`, `Core.Onsite.protocol`),
the verifier of record, which decides every claim: sampled proofs', PoUW's, PoUS's, the bandwidth audit's and the
network certifier's. Its wipe decides none.

**`Given`** holds the `Vocab` fields that no owner has defined on `main` yet, each with its owner in its docstring;
each field leaves `Given` in the PR that defines it. `vocab G` is onsite's `Vocab` from them. Its honest prover
(`honest`) sends, in each of the device's rounds, the bytes the honest prover computes from the releases so far
(`Given.answer`), and then the evidence from the device's log with the blobs the components fetch (`Given.evidence`,
`Given.blobs`): the served path's prover as a function of the coins. That prover runs C-Flock and reads files through
`partial` parses, so its outputs are taken as `Given` fields and the served bytes are never exhibited.

**The honest lemma's statement (`SoundnessHonest`).** Each claim's `Vocab.HonestHolds` at the commitment the honest
prover sends (`Given.commit`): never an arbitrary one, since sampled proofs' claim reads the commitment
(`Vocab.Opens`) and at one that opens nothing it fails on every play
(`Verity.Vocab.not_honestHolds_sampled_of_unopened`, in the proofs). It takes one named premise per claim until its
owner proves it:

* `SampledHonest` (soundness): the extractor's transcript at the honest strategy is correct and the honest commitment
  opens it, from which sampled proofs' claim follows (`Vocab.holds_sampled_of_opens`);
* `NetworkHonest` (network-accounting): the network certifier's honest lemma, `Verity.networkComponentBound_honest`;
* `PouwHonest` (compute-accounting): PoUW's period honest lemma;
* `PousHonest` and `BandwidthHonest` (memory-accounting): PoUS's run check's honest lemma (`runAccepts_meets`) and
  the bandwidth audit's.
-/

namespace Verity.Onsite

open Accounting Core
open scoped ENNReal

/-- **The `Vocab` fields no owner has defined on `main` yet**, and the honest prover's outputs, which have no Lean
value.

* `read`: v2's reading of the committed blob.
* `place`, `programDigest`, `registered`: the positions, the circuit file's digest in the parameters and the registered
  blob (`Flock.Bridge.circuitOf` gives none of these).
* `rounds`, `sign`, `key`, `coinsRead`: the device's rounds, its signer and registered LMS key, and the releases each
  component's check read.
* `extract`: the extractor applied to a strategy (`XplurZC`).
* `finder`, `callsMul`, `callsAdd`, `finder_calls`: each claim's collision finder and its SHA-512 calls in those of
  the strategy (each subprotocol's owner).
* `forger`: each claim's LMS forger on the play's outcome (what a forgery is, `Vocab.forgery` defines).
* `cost`: the strategy's SHA-512 calls, declared, as `Vocab.cost` is (`Verity.Game` has no call model).
* `computed`: the gates of the committed circuit the play computed after their window's coin (compute-accounting).
* `world`: the world each play realizes (the runs PoUS, the bandwidth audit and the network certifier realize).
* `certOf`, `crBound`, `delta`, `loss`: what the product publishes, from the parameters alone (the model's `deltaPub`
  is in the proofs tree, which a statement can't read).
* `commit`, `answer`, `evidence`, `blobs`: the honest prover's commitment, each round's bytes from the releases so far,
  and its evidence and the blobs the components fetch by digest (v2's circuit and public files, each session's record
  and rep files), from the device's log. -/
structure Given {ι : Type} (d : Device ι) (α : Type) [Fintype α] where
  read : ByteArray → FileRead
  place : (θ F c : ByteArray) → Fin (read F).circuit.N → Option Place
  programDigest : ByteArray → Option Hash.Digest
  registered : ByteArray → ByteArray
  rounds : ByteArray → List ℕ
  sign : ByteArray → ByteArray → List Bool → ByteArray × ByteArray
  key : ByteArray → Lms.Params × ByteArray
  coinsRead : Audit → Sent → List Release
  extract : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) (sign θ) θ F c) → (read F).circuit.Transcript
  finder : Audit → (θ F c : ByteArray) → Game.Strategy (play (rounds θ) (sign θ) θ F c) → Claims.Finder
  forger : Audit → ByteArray → Outcome → Option (ByteArray × ByteArray)
  cost : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) (sign θ) θ F c) → ℕ
  callsMul : ByteArray → Audit → ℕ
  callsAdd : ByteArray → Audit → ℕ
  finder_calls : ∀ a θ F c σ (x : (finder a θ F c σ).S),
    (finder a θ F c σ).cost x ≤ callsMul θ a * cost θ F c σ + callsAdd θ a
  computed : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) (sign θ) θ F c) → Outcome →
    Finset (Fin (read F).circuit.N)
  world : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) (sign θ) θ F c) → Outcome → World d α
  certOf : ByteArray → Certificates
  crBound : ByteArray → Audit → ℕ → ℝ
  delta : ByteArray → Audit → ℕ → ℝ≥0∞
  loss : Audit → ByteArray → ℝ≥0∞ → ℝ≥0∞
  commit : ByteArray → ByteArray → ByteArray
  answer : (θ F c : ByteArray) → List Release → ByteArray
  evidence : (θ F c : ByteArray) → List Release → ByteArray
  blobs : (θ F c : ByteArray) → List Release → Array ByteArray

/-- A strategy of `g.bind f`: `s` in `g`, then `t a` in `f a` once `g` ended with `a`. -/
def bindStrategy {α β : Type} : (g : Game α) → (f : α → Game β) → Game.Strategy g →
    ((a : α) → Game.Strategy (f a)) → Game.Strategy (g.bind f)
  | .ret a, _, _, t => t a
  | .send _ k, f, s, t => ⟨s.1, bindStrategy (k s.1) f s.2 t⟩
  | @Game.coin _ _ _ _ k, f, s, t => fun c => bindStrategy (k c) f (s c) t

/-- **The device's rounds against a prover that answers from the releases so far**: before each release, the prover
sends `ans pre`, where `pre` is the releases before it. -/
def releasesBy (sign : ByteArray → List Bool → ByteArray × ByteArray) (ans : List Release → ByteArray)
    (pre : List Release) : (ns : List ℕ) → Game.Strategy (releases sign ns)
  | [] => PUnit.unit
  | _ :: ns => ⟨ans pre, fun x => Game.Strategy.toMap _ (releases sign ns)
      (releasesBy sign ans
        (pre ++ [⟨ans pre, List.ofFn x, (sign (ans pre) (List.ofFn x)).1, (sign (ans pre) (List.ofFn x)).2⟩]) ns)⟩

/-- **Onsite's honest prover**: each round's bytes from the releases so far, then the evidence from the device's log,
with the blobs the components fetch. -/
def honest {ι : Type} {d : Device ι} {α : Type} [Fintype α] (G : Given d α) (θ F c : ByteArray) :
    Game.Strategy (play (G.rounds θ) (G.sign θ) θ F c) :=
  bindStrategy (releases (G.sign θ) (G.rounds θ))
    (fun log => .send (ByteArray × Array ByteArray) fun eb => .ret (⟨θ, F, c, log, eb.1, eb.2⟩ : Outcome))
    (releasesBy (G.sign θ) (G.answer θ F c) [] (G.rounds θ))
    fun log => ⟨(G.evidence θ F c log, G.blobs θ F c log), PUnit.unit⟩

/-- **Onsite's vocabulary**: `G`'s fields and onsite's honest prover. -/
noncomputable def vocab {ι : Type} {d : Device ι} {α : Type} [Fintype α] (G : Given d α) : Vocab d α where
  read := G.read
  place := G.place
  programDigest := G.programDigest
  registered := G.registered
  rounds := G.rounds
  sign := G.sign
  key := G.key
  coinsRead := G.coinsRead
  extract := G.extract
  finder := G.finder
  forger := G.forger
  cost := G.cost
  callsMul := G.callsMul
  callsAdd := G.callsAdd
  finder_calls := G.finder_calls
  computed := G.computed
  world := G.world
  honest := honest G
  certOf := G.certOf
  crBound := G.crBound
  delta := G.delta
  loss := G.loss

/-- **The honest prover's sampled proofs case** (premise; owner soundness, the extractor and the positions): at
registered files and the honest commitment, the extractor's transcript at the honest strategy is correct on every
unit, the commitment opens every gate to it (a registered constant to its constant), and the published share `ε` is
not negative. It leaves the premises once `extract` and `place` are defined and it is proved, from
`Flock.Bridge.card_wrong_lt_of_relS_honest` (the honest witness opens every committed value and no unit is wrong). -/
def SampledHonest {ι : Type} {d : Device ι} {α : Type} [Fintype α] (G : Given d α) : Prop :=
  ∀ θ F : ByteArray, (vocab G).Registered θ F →
    (∀ u, ((vocab G).partsOf F).Correct (G.extract θ F (G.commit θ F) (honest G θ F (G.commit θ F))) u) ∧
      (∀ g, (vocab G).Opens θ F (G.commit θ F) g (G.extract θ F (G.commit θ F) (honest G θ F (G.commit θ F)) g)) ∧
      0 ≤ (G.certOf θ).ε

/-- **The network certifier's honest lemma at onsite** (premise; owner network-accounting):
`Verity.networkComponentBound_honest` (`Properties.Verity.Network`), `Vocab.HonestHolds` at the network certifier's
claim at the honest commitment: on every play of the honest strategy at a registered file, what the network
certifiers saw the node move is within the published `K`. Its use here becomes a citation once it lands. -/
def NetworkHonest {ι : Type} {d : Device ι} {α : Type} [Fintype α] (G : Given d α) : Prop :=
  (vocab G).HonestHolds G.commit .network

/-- **PoUW's honest lemma at onsite** (premise; owner compute-accounting): `Vocab.HonestHolds` at PoUW's claim at the
honest commitment: on every play of the honest strategy at a registered file, the period's credited work is gates the
play computed. Its use here becomes a citation once PoUW's period honest lemma lands. -/
def PouwHonest {ι : Type} {d : Device ι} {α : Type} [Fintype α] (G : Given d α) : Prop :=
  (vocab G).HonestHolds G.commit .pouw

/-- **PoUS's honest lemma at onsite** (premise; owner memory-accounting): `Vocab.HonestHolds` at PoUS's claim at the
honest commitment, PoUS's run check's honest lemma (`runAccepts_meets`) at onsite's vocabulary. Its use here becomes a
citation once it is stated at `Vocab`. -/
def PousHonest {ι : Type} {d : Device ι} {α : Type} [Fintype α] (G : Given d α) : Prop :=
  (vocab G).HonestHolds G.commit .pous

/-- **The bandwidth audit's honest lemma at onsite** (premise; owner memory-accounting): `Vocab.HonestHolds` at the
bandwidth audit's claim at the honest commitment. Its use here becomes a citation once the bandwidth audit's honest
lemma is stated at `Vocab`. -/
def BandwidthHonest {ι : Type} {d : Device ι} {α : Type} [Fintype α] (G : Given d α) : Prop :=
  (vocab G).HonestHolds G.commit .poub

/-- **Onsite's honest lemma**: on any device, at any `Given`, where each claim's honest premise holds, every claim
onsite decides holds on almost every play of the honest prover at registered files and the honest commitment. -/
def SoundnessHonest : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (G : Given d α), SampledHonest G → NetworkHonest G → PouwHonest G →
    PousHonest G → BandwidthHonest G → ∀ a, onsite.Decides a → (vocab G).HonestHolds G.commit a

end Verity.Onsite
