module

public import CoreComposites
public import Properties.Composition
public import Properties.Flock.Circuit.CircuitOf
public import Definitions.Accounting.JointLive
public import Definitions.Core.Blocks
public import Definitions.Core.Game
public import Definitions.Core.Site
public import Definitions.Hash.Assumptions
public import Definitions.Hash.RowLeaf
public import Definitions.Nci
public import Flock.CoinTree
public import Flock.Hash
public import Flock.Hm96
public import Lms.Verify
public import Properties.Verity.Sha512

/-!
# Soundness as a causal game at a composite's compiled verifier, and each subprotocol's target

The verifier is one compiled Lean function per composite (`Verity.Registration.Composite`), the `answer` of
the composite `Core.Composites.composites` lists for it (`verifier`): onsite's `Core.Onsite.protocol`
(`verity/onsite/v0`, `Core.Composition.tiedSameB`) and remote's `Core.Remote.protocol` (`verity/remote/v0`,
`Core.Composition.tiedSameB`), each composing its subprotocols over the blobs the prover sent. Onsite has seven
subprotocols and remote three of them, each subprotocol at the compiled check its composite names for it (`component`);
the five that decide a period are one per `Accounting.Audit` (`Subprotocol.ofAudit`):

| subprotocol | `Audit` | onsite's compiled check | remote's compiled check |
| --- | --- | --- | --- |
| SPOT | `.sampled` | `VStar.Recursive.Sampled.protocol` (`verity/sampled-proofs/v1`: one-stage v2's audit; one-stage v1 and the recursive audit refused) | the same |
| PoUW | `.pouw` | `PouwCheck.Period.protocol` (its period credit) | the same |
| the inference-only policy | | `NciCheck.Protocol.protocol` (`verity/inference-only/v0`) | the same |
| PoUS | `.pous` | `PousCheck.RunProtocol.protocol` | |
| PoUB-P2 | `.poub` | `RemoteAudit.P2Protocol.protocol` (`verity/remote-bandwidth/p2/v0`) | |
| the network certifier | `.network` | `NetworkCheck.Period.protocol` | |
| the wipe | | `MemoryChallenger.WipeProtocol.protocol` (`verity/wipe/v0`) | |

The statement's form is the causal game, the committed files, one fixed extractor and one adversary form:

* **A causal game.** The prover sends the parameters `θ`, the committed files `F` (any number of blobs) and the
  commitment `c` first, before any coin, so every statement is `∀ θ F c`, and then plays
  `play (families θ) (signer θ) θ F c`, a `Verity.Game` in which the prover drives the device. It sends its request
  (`Exchange.request`), then its frames (`releases`): the parameters fix only the coin commitment's families and the
  device's signer, and the frames are the prover's, in any order and number. Each frame names what the device releases
  and carries the bytes the release answers (a root, a transcript digest), and then the device samples and releases its
  coins: a family's next coin (`Frame.fresh`), a coin tree sampled at its `Hello` with its root and key
  (`Frame.hello`), or an opening of a tree the device holds open, its stream's next unopened round (`Frame.opening`), so
  no round is opened twice. A frame the device refuses ends its releases, as the prover's close does. At the close
  (`finish`) the device flips each coin's salt and the coins it committed to and didn't release, and releases its
  signatures on what the play fixed (`Signer`): its record, its close over its log and its ledger lines, as the
  deployed device does once a withheld session's signed close is written. The evidence comes last, with its own blobs
  (the record, its rep files, the proofs). A strategy (`Verity.Game.Strategy`) is a function of the coins released so
  far and never of later ones. The composite reads the committed files and then the evidence's blobs
  (`Exchange.sent`), so every input whose blobs begin with the committed files is what some strategy sends
  (`Verity.sent_reachable`).
* **The claims read the committed files the parameters name.** The circuit file and the public file are the committed
  files the parameters name by their digests (`programDigest`, `publicDigest`, `namedFile`, as v2 looks them up). The
  circuit, its units and its constants are what the verifier reads from the two (`Flock.Bridge.circuitOfFile`,
  `unitsOfFile`, `partsOfFile` at v2's reading of the files, `read`), so the units are the public file's population,
  and the positions are the files' and the commitment's (`place`), never the prover's world. Committed files without a
  file of each digest are refused (`Registered`, conjoined beside the composite's acceptance).
* **Sampled proofs' claim reads one fixed extractor's transcript.** The extractor (`extract`, the one
  `SampledKSLive*` is stated with, `XplurZC`) is a construction applied to the strategy, never chosen per strategy.
  The claim is that at most an `ε` share of the file's units are unbound in its transcript (`Accounting.unbound`): wrong
  in it, or with a committed input or output the commitment doesn't open to its value there (`Opens`: a registered
  constant to its constant, any other wire by an opening of its row leaf that `Hash.RowLeaf.accepts` passes under the
  compiled SHA-512).
* **One adversary form: every prover, and collisions a fixed finder outputs.** There is no strategy class and no break
  in the event. The finder `B` (`finder a`, a `Claims.Finder`) is a fixed construction applied to the strategy whose
  SHA-512 calls are at most `c · calls(τ) + d` (`finder_calls`). Subprotocol `a`'s error at a strategy (`error`) is,
  where `B^τ` meets `ecr/sha-512` (`FinderCR`, `Claims.ExpectedCR` at the finder), its published `δ` at the strategy's
  calls plus its loss at `Pr[B^τ]`, the form of `Accounting.PeriodLive.error`; elsewhere `⊤`. There `B^τ` outputs a
  SHA-512 collision with probability at most its budget over `2^256` (`crBound`), which a bound reads as `ε_cr`. That
  admits Flock's conditional bound (`loss = 0`). A bound that reads further finders built from the strategy lists them (`finders`, each a
  `FinderSpec`: the finder, and its budget as a function of the strategy's calls that bounds its calls), and the error
  is `⊤` where one of them outputs a collision with probability above its budget over `2^256` (`finderCR`, built by
  `finderCROf`), the bound `ecr/sha-512` gives it. Two openings of one position
  and a second file with a registered digest (`fileFinder`, `publicFinder`) each enter that way. The named assumption
  `ecr/sha-512` is asked only of the finder. A forged device release enters the same way through a fixed forger
  (`forger a`, a `Hash.Assumptions.LmsForger` applied to the strategy, its calls bounded as the finder's are). It sees
  the device's signatures a play releases, on each of its coins on at most the published `Q` messages, the most the
  device signs in a play (`signs_le`, `forger_signed`), so it is a forger of the standard EUF-CMA game, its queries
  chosen adaptively by the strategy's moves. Where it meets `Hash.Assumptions.LmsUnforgeable` at the compiled LMS
  verifier (`lmsVerify`), the error adds the published `ε_forge(t, Q)` (`epsForge`).
* **The one vocabulary.** The period is `Accounting.PeriodLive` (`live`), and the five period claims are its `Holds`.
  The inference-only policy's claim is `Nci.CompliesRate` at the published `Γ` and `α` of the declaration over the committed
  circuit (`declaration`: its registered inputs `θ`, `theta`). The wipe's claim is that the run, read at the wipe's
  boundary as the wipe's game (`Verity.Wipe.OfRun`, `wipeRun`), doesn't keep foreign content past the survivor threshold
  the wipe publishes (`Verity.Wipe.OfRun.Kept` at `foreignBits`).
* **Each compiled check's acceptance in what the prover sent.** Subprotocol `a`'s verdict at composite `k` (`Verdict`) is the
  file check, the composite's own acceptance as a `same` composition (`SameAccepts`: an accepted composition of its
  subprotocols whose record subprotocols committed to one record, `Properties.Composition.AcceptSame`'s conclusion, and
  every window of PoUW's period on the record and at onsite PoUS and PoUB-P2 on one store, `AcceptTied`'s),
  and `a`'s compiled check's acceptance of its own case. The device's log as each compiled check read it
  (`coinsRead`) is the targets' to bind to the device's whole log (`Logged`).
* **PoUW counts gates** of the committed circuit the play computed after their window's coin, over the period's windows
  (`computed`), with no pricing.

Each subprotocol's target (`SpotSound`, `PouwSound`, `InferenceOnlySound`, `PousSound`, `BandwidthAuditSound`,
`NetworkCertifierSound`, `WipeSound`) is `Target k a`: at every `θ`, `F`, `c` and strategy, the probability that `a`'s
verdict at `k` accepts while its claim fails is at most its error there. `Verity.Soundness` and `Verity.RemoteSoundness`
(`Properties.Verity.Soundness`) compose them, onsite's seven and remote's three.

The names a subprotocol defines are `Vocab`'s fields: `read` (v2's reading of the committed files), `place`,
`programDigest`, `publicDigest` and `registered`, `families`, `signer` and `coinsRead`, `extract`, the finders and the
forgers, `computed`, `world` (the runs PoUS, PoUB-P2 and the network certifier realize), the declaration's registered
inputs and the policy's budgets, the wipe at the period's boundary and the run read as its game, and `certOf`,
`foreignBits`, `signatures`, `delta`, `epsForge` and `loss` (what the composite publishes, functions of the parameters
alone).
-/

@[expose] public section

namespace Verity

open Accounting Core
open scoped ENNReal

/-- The subprotocols a composite is made of. -/
inductive Subprotocol
  /-- sampled proofs: the integrity of the circuit's committed transcript -/
  | sampled
  /-- PoUW: work -/
  | pouw
  /-- the inference-only policy: the registered circuit divides into inference blocks within their budgets -/
  | inferenceOnly
  /-- PoUS: space -/
  | pous
  /-- PoUB-P2, the remote bandwidth audit's P2 session on PoUS's store: bits read from device memory -/
  | poub
  /-- the network certifier: communication through the network certifiers -/
  | network
  /-- the wipe: the memory challenger's streamed-label wipe that opens the period -/
  | wipe
  deriving DecidableEq

instance instFintypeSubprotocol : Fintype Subprotocol where
  elems := {.sampled, .pouw, .inferenceOnly, .pous, .poub, .network, .wipe}
  complete a := by cases a <;> simp

/-- The subprotocol that decides `Audit` `a`. -/
def Subprotocol.ofAudit : Audit → Subprotocol
  | .sampled => .sampled
  | .pouw => .pouw
  | .pous => .pous
  | .poub => .poub
  | .network => .network

/-- The compiled verifier of composite `k`. -/
def verifier : Verity.Registration.Composite → BlobProtocol
  | .onsite => Onsite.protocol
  | .remote => Remote.protocol

/-- Composite `k`'s subprotocols, as its verifier composes them, in the order of their ids. -/
def componentsOf : Verity.Registration.Composite → List BlobProtocol
  | .onsite => Onsite.components
  | .remote => Remote.components

/-- The subprotocols of composite `k` that must commit to one record. -/
def tied : Verity.Registration.Composite → List String
  | .onsite => Onsite.tied
  | .remote => Remote.tied

/-- The ties of composite `k`: at both, every window of PoUW's period on the policy's record; at onsite, also PoUS and
PoUB-P2 on one store. -/
def ties : Verity.Registration.Composite → List Composition.Tie
  | .onsite => Onsite.ties
  | .remote => [Remote.windows]

/-- **The compiled check of composite `k` that decides subprotocol `a`**; `none` when `k` has no such
subprotocol. -/
def component : Verity.Registration.Composite → Subprotocol → Option BlobProtocol
  | .onsite, .sampled => some VStar.Recursive.Sampled.protocol
  | .onsite, .pouw => some PouwCheck.Period.protocol
  | .onsite, .inferenceOnly => some NciCheck.Protocol.protocol
  | .onsite, .pous => some PousCheck.RunProtocol.protocol
  | .onsite, .poub => some RemoteAudit.P2Protocol.protocol
  | .onsite, .network => some NetworkCheck.Period.protocol
  | .onsite, .wipe => some MemoryChallenger.WipeProtocol.protocol
  | .remote, .sampled => some VStar.Recursive.Sampled.protocol
  | .remote, .pouw => some PouwCheck.Period.protocol
  | .remote, .inferenceOnly => some NciCheck.Protocol.protocol
  | .remote, _ => none

/-- Composite `k`'s subprotocols: those it has a compiled check for. Remote's are a subset of onsite's. -/
def subprotocols (k : Verity.Registration.Composite) : Finset Subprotocol :=
  Finset.univ.filter fun a => (component k a).isSome

/-- What the prover sends a composite's verifier: the parameters, the commitment and the evidence, as bytes,
and the blobs its subprotocols read. -/
structure Sent where
  params : ByteArray
  commitment : ByteArray
  evidence : ByteArray
  blobs : Array ByteArray

/-- Composite `k`'s subprotocols on the blobs the prover sent. -/
def Sent.components (k : Verity.Registration.Composite) (s : Sent) : List Protocol :=
  (componentsOf k).map (·.onBlobs (Blobs.ofArray s.blobs))

/-- **Composite `k`'s verifier accepts what the prover sent.** -/
def Accepts (k : Verity.Registration.Composite) (s : Sent) : Prop :=
  ((verifier k).answer s.params s.commitment s.evidence s.blobs).verdict = .accept

/-- The three byte strings decode under the composition's formats (`Core.Composition.product`'s, which both composites
read): the subprotocols' parameters `ps`, the binding `c` of their digests, and their cases `ev`. -/
def Decoded (s : Sent) (ps : List Composition.Named) (c : ByteArray) (ev : List Composition.Case) : Prop :=
  (canonicalJson s.params >>= Composition.decodeParams) = .ok ps ∧ Composition.decodeCommitment s.commitment = .ok c ∧
    Composition.decodeEvidence s.evidence = .ok ev

/-- The protocol `P` accepts its own case in what the prover sent composite `k`: the three byte strings decode, and
one of the subprotocols' cases is `P`'s, as one of `k`'s subprotocols, which decodes under the parameters `k` names for
it, and `P`'s check accepts it. -/
def CaseAccepts (k : Verity.Registration.Composite) (P : BlobProtocol) (s : Sent) : Prop :=
  ∃ ps c ev, Decoded s ps c ev ∧
    ∃ x ∈ Properties.Composition.triples (s.components k) ps ev, x.1 = P.onBlobs (Blobs.ofArray s.blobs) ∧
      Properties.Composition.DecodedAccepted x.1 x.2.1.params.toUTF8 x.2.2.commitment x.2.2.evidence

/-- Subprotocol `a`'s compiled check at composite `k` accepts its own case in what the prover sent. -/
def ComponentAccepts (k : Verity.Registration.Composite) (a : Subprotocol) (s : Sent) : Prop :=
  ∃ P, component k a = some P ∧ CaseAccepts k P s

/-- **Composite `k` accepted what the prover sent as a `same` composition**: the three byte strings decode to an
accepted composition of `k`'s subprotocols on the blobs sent, whose record subprotocols (`tied k`) committed to one
record (`Properties.Composition.SameAccepted`), and whose ties (`ties k`: at both, every window of PoUW's period on the
policy's record; at onsite, also PoUS and PoUB-P2 on one store) each named one value (`Properties.Composition.Tied`). -/
def SameAccepts (k : Verity.Registration.Composite) (s : Sent) : Prop :=
  ∃ ps c ev, Decoded s ps c ev ∧ ∃ pub prof,
    Properties.Composition.SameAccepted (verifier k).id (tied k) (s.components k) ps c ev pub prof ∧
      ∀ t ∈ ties k, Properties.Composition.Tied t ev

/-- **A frame the prover sends the device**, beside the bytes the device's release answers. -/
inductive Frame where
  /-- the next coin of the coin commitment's family `label`, after the root the prover sends -/
  | fresh (label : String)
  /-- a coin tree's `Hello`: the device samples a tree of shape `spec` (`coin-tree/hm96-sha512/v2`) under the nonce the
  prover sends, and releases its root and key -/
  | hello (spec : Flock.CoinTree.Spec)
  /-- an opening of the `i`-th coin tree the device sampled: its stream `s`'s next unopened round, the round's first `n`
  coins, after the bytes the opening answers. An opening names no round, so no round is opened twice: the device opens a
  stream's rounds in order, each once, as the challenger refuses any other round -/
  | opening (i s n : ℕ)

/-- One release of the challenger device: the frame it answers, the bytes the prover sent with it, and the bits it
released after them, a family's coin, a coin tree's root and key at its `Hello`, or a round's opened coins. -/
structure Release where
  frame : Frame
  answers : ByteArray
  coins : List Bool

/-- **A coin tree's hash under the nonce `ν`**: the compiled SHA-512 of `ν ‖ m` (`coin_tree.Rules.h`), every hash of the
tree but the key's in its leaf prefix. -/
def treeHash (ν m : ByteArray) : ByteArray := Flock.Sha512.hash (ν ++ m)

/-- The coin tree's tag `verity/coin-tree/name/v2`, with its zero byte. -/
def treeTag (name : String) : ByteArray := (s!"verity/coin-tree/{name}/v2").toUTF8.push 0

/-- **The leaf of stream `s`'s round `r`** under the nonce `ν` and the key `key`, at the round's block of coins `cs` (16
bytes each) and its salt `y` (`coin_tree.Rules.leaf`): the block's chain `T_0` (`T_block = H(TAIL)`,
`T_i = H(CHAIN ‖ c_i ‖ T_{i+1})`), the round digest `x = H(ROUND ‖ u32be(s) ‖ u32be(r) ‖ T_0)`, and hm96-sha512's leaf
of `x` at the key, `H(LEAF_PREFIX(key) ‖ x ⊕ M(key)·y ‖ H(SALT_PREFIX ‖ y))`. -/
def treeLeaf (ν key : ByteArray) (s r : ℕ) (cs : List ByteArray) (y : ByteArray) : ByteArray :=
  let t0 := cs.foldr (fun c t => treeHash ν (treeTag "chain" ++ c ++ t)) (treeHash ν (treeTag "tail"))
  let x := treeHash ν (treeTag "round" ++ Flock.be32 s.toUInt32 ++ Flock.be32 r.toUInt32 ++ t0)
  treeHash ν (Flock.Hm96.sha512.leafPrefix key ++
    ({ Flock.Hm96.sha512 with hash := treeHash ν } : Flock.Hm96).commitString key x y)

/-- One level up a Merkle tree under the nonce `ν`: adjacent nodes hashed in pairs (`coin_tree._levels`). -/
def treeUp (ν : ByteArray) : List ByteArray → List ByteArray
  | a :: b :: t => treeHash ν (a ++ b) :: treeUp ν t
  | _ => []

/-- The root of a Merkle tree of `2^k` nodes under the nonce `ν`. -/
def treeTop (ν : ByteArray) : ℕ → List ByteArray → ByteArray
  | 0, xs => xs.headD .empty
  | k + 1, xs => treeTop ν k (treeUp ν xs)

/-- The `n` bytes of the bits `f`: bit `t` is bit `t % 8` of byte `t / 8`. -/
def bytesOfBits (n : ℕ) (f : ℕ → Bool) : ByteArray :=
  ⟨Array.ofFn fun i : Fin n => ((List.range 8).foldr (fun b acc => 2 * acc + (f (8 * i.val + b)).toNat) 0).toUInt8⟩

/-- The bits of the bytes `b`, bit `t` bit `t % 8` of byte `t / 8`. -/
def bitsOfBytes (b : ByteArray) : List Bool := (List.range (8 * b.size)).map (Flock.Hm96.bit b)

/-- **A coin tree's round as the device samples it**: its block of `block` coins of 128 bits each, and its salt's 1536
bits. -/
abbrev TreeDraw (block : ℕ) : Type := (Fin block → Fin 128 → Bool) × (Fin 1536 → Bool)

/-- **A coin tree's sample at its `Hello`**: the key's 2047 bits (bit 2047 of its 256 bytes is clear), and every round's
sample. -/
abbrev TreeSample (spec : Flock.CoinTree.Spec) : Type :=
  (Fin 2047 → Bool) × (Fin spec.streams.size → Fin (2 ^ spec.roundsLog) → TreeDraw spec.block)

/-- The key's 256 bytes. -/
def treeKey (κ : Fin 2047 → Bool) : ByteArray := bytesOfBits 256 fun t => if h : t < 2047 then κ ⟨t, h⟩ else false

/-- A sample's coins, 16 bytes each. -/
def TreeDraw.coins {block : ℕ} (d : TreeDraw block) : List ByteArray :=
  List.ofFn fun i => bytesOfBits 16 fun t => if h : t < 128 then d.1 i ⟨t, h⟩ else false

/-- A sample's leaf as stream `s`'s round `r` under the nonce `ν` and the key `key`. -/
def TreeDraw.leaf {block : ℕ} (ν key : ByteArray) (s r : ℕ) (d : TreeDraw block) : ByteArray :=
  treeLeaf ν key s r d.coins (bytesOfBits 192 fun t => if h : t < 1536 then d.2 ⟨t, h⟩ else false)

/-- A sample's first `n` coins, the bits the device releases when it opens them. -/
def TreeDraw.opened {block : ℕ} (d : TreeDraw block) (n : ℕ) : List Bool :=
  ((List.ofFn fun i => List.ofFn (d.1 i)).take n).flatten

/-- **A coin tree's root** at the sample `t` under the nonce `ν` (`coin_tree.CoinTree.root`): each stream's Merkle tree
over its rounds' leaves, then the tree over the streams' roots, padded to a power of two with `H(PAD)`. -/
def treeRoot (ν : ByteArray) (spec : Flock.CoinTree.Spec) (t : TreeSample spec) : ByteArray :=
  let w := if spec.streams.size ≤ 1 then 0 else (spec.streams.size - 1).log2 + 1
  let tops := List.ofFn fun s : Fin spec.streams.size => treeTop ν spec.roundsLog
    (List.ofFn fun r : Fin (2 ^ spec.roundsLog) => TreeDraw.leaf ν (treeKey t.1) s r (t.2 s r))
  treeTop ν w (tops ++ List.replicate (2 ^ w - spec.streams.size) (treeHash ν (treeTag "pad")))

/-- **The flips a coin tree's opening samples among**: stream `s`'s round `r` with the leaf the sample `t` committed to
there under the nonce `ν`. -/
abbrev TreeOpenable (ν : ByteArray) (spec : Flock.CoinTree.Spec) (t : TreeSample spec) (s : Fin spec.streams.size)
    (r : Fin (2 ^ spec.roundsLog)) : Type :=
  {d : TreeDraw spec.block //
    (TreeDraw.leaf ν (treeKey t.1) s r d).data = (TreeDraw.leaf ν (treeKey t.1) s r (t.2 s r)).data}

/-- **A coin tree the device holds open**: its `Hello`'s nonce, its shape, what the device sampled at its `Hello`, and how
many of each stream's rounds it has opened. -/
structure OpenTree where
  nonce : ByteArray
  spec : Flock.CoinTree.Spec
  sample : TreeSample spec
  opened : Fin spec.streams.size → ℕ

/-- **The round an opening of stream `s` of the tree `T` opens, for its first `n` coins**: the stream's next unopened
round; `none` where the device refuses the opening (`open_tree`): `s` is no stream of `T`, every round of it is opened,
or `n` is more than a block. -/
def OpenTree.next (T : OpenTree) (s n : ℕ) : Option (Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)) :=
  if hs : s < T.spec.streams.size then
    if hr : T.opened ⟨s, hs⟩ < 2 ^ T.spec.roundsLog ∧ n ≤ T.spec.block then some (⟨s, hs⟩, ⟨T.opened ⟨s, hs⟩, hr.1⟩)
    else none
  else none

/-- The tree `T` once its stream `s`'s next round is opened. -/
def OpenTree.bump (T : OpenTree) (s : Fin T.spec.streams.size) : OpenTree :=
  ⟨T.nonce, T.spec, T.sample, fun j => if j = s then T.opened s + 1 else T.opened j⟩

/-- **The coin commitment's families once a coin of family `l` is released**: its count one less; `none` where the device
refuses the coin (`LiveRecord`): no family is `l`, or every coin of it is released. -/
def takeCoin (l : String) : List (String × ℕ) → Option (List (String × ℕ))
  | [] => none
  | (l', n) :: fs =>
    if l' = l then (match n with | 0 => none | n + 1 => some ((l', n) :: fs)) else (takeCoin l fs).map ((l', n) :: ·)

/-- The most coin-tree lines the device's log holds, each tree it sampled and each round it opened (`MAX_TREE_LINES`). -/
def maxTreeLines : ℕ := 2 ^ 23

/-- **The most coins a coin tree the device samples holds**, every stream's every round (`MAX_TREE_COINS`). -/
def maxTreeCoins : ℕ := 2 ^ 26

/-- A coin tree's coins: `2^roundsLog` rounds a stream, `block` coins a round. -/
def treeCoins (spec : Flock.CoinTree.Spec) : ℕ := spec.streams.size * 2 ^ spec.roundsLog * spec.block

/-- **The specs the device samples a coin tree of**: a coin or more a round, and at most `maxTreeCoins` coins. The
deployed device refuses every other spec: `LiveChallenger.sample_tree`
(`verity/core/protocols/spot/challenger.py`) refuses a spec `tree_spec_reason` refuses, a round of no coin
among them, and then a tree of more than `MAX_TREE_COINS` coins. So a tree the device samples holds at most
`maxTreeCoins` leaves: a leaf is a round, of a coin or more. -/
def TreeSpecOk (spec : Flock.CoinTree.Spec) : Prop := 1 ≤ spec.block ∧ treeCoins spec ≤ maxTreeCoins

instance instDecidableTreeSpecOk (spec : Flock.CoinTree.Spec) : Decidable (TreeSpecOk spec) :=
  inferInstanceAs (Decidable (_ ∧ _))

/-- **The coin-tree lines the device's log holds** with the coin trees `ts` open: one for each tree it sampled and one
for each round it opened. The deployed device counts them (`LiveChallenger._tree_lines`) and refuses a tree or an
opening once its log holds `MAX_TREE_LINES` of them (`LiveChallenger._tree_line`). -/
def treeLines (ts : List OpenTree) : ℕ := (ts.map fun T => 1 + ∑ s, T.opened s).sum

/-- **The most leaves the device's coin trees hold in a play**: at most `maxTreeLines` trees (`treeLines`), each of at
most `maxTreeCoins` leaves (`TreeSpecOk`), `2^49`. -/
def maxLeaves : ℕ := maxTreeLines * maxTreeCoins

/-- **The most frames the device serves** under the coin commitment's families `fs`: each frame it serves releases one of
their coins or writes one of its coin-tree lines. -/
-- The literal comes first: `Nat.add` recurses on its second argument, so the kernel compares two plays' games at the
-- families' sums instead of counting the literal down.
def frames (fs : List (String × ℕ)) : ℕ := maxTreeLines + (fs.map Prod.snd).sum

/-- **The device's releases as the prover drives it**, with the coins `fs` of the coin commitment's families still to
release, the coin trees `ts` open, and at most `k` frames more. The prover sends a frame with the bytes its release
answers, or closes (`none`): the order, number and kind of the frames are the prover's. At a family's coin
(`Frame.fresh`) the device samples the family's next coin, its 64 bytes (`COIN_BYTES`), and releases it. At a coin tree's
`Hello` the bytes are its nonce `ν`: the device samples the tree (`TreeSample`: a fresh key and every round's sample),
releases its root and key (`treeRoot`, `treeKey`) and holds it open. At an opening of the `i`-th tree it holds, the
device samples the stream's next unopened round anew among the samples with the leaf the tree's `Hello` committed to there
(`TreeOpenable`), releases its first `n` coins and counts the round opened (`OpenTree.next`, `OpenTree.bump`). A frame
the device refuses (a family with no coin left, a `Hello` whose spec it doesn't sample (`TreeSpecOk`), a tree frame once
its log holds `maxTreeLines` coin-tree lines (`treeLines`), no `i`-th tree, or `OpenTree.next`'s refusal) ends its
releases, as a refused frame ends the challenger's session, and so does the prover's close. The output is the device's
log. The two caps are the deployed device's own refusals (`LiveChallenger.sample_tree` and `LiveChallenger._tree_line`
in `verity/core/protocols/spot/challenger.py`), so a play here samples at most `maxLeaves` leaves.

The challenger refuses further frames this device serves (a spec past its other limits, a nonce or a root of another
length, a tree's root or key sampled twice), so each of its sessions, a refused frame read as the prover's close, is a
play here with the same log, and the plays it refuses only add strategies. Each frame it serves
releases one of the families' coins or writes one of its coin-tree lines, so it serves at most `frames fs`. A withheld
session, the one a sampled-proofs window's sample reads, answers its hello with the families alone and each coin with its
raw bytes, as the device here releases them, and its close with the commitment and every coin's salt and path, which
the play samples after the last release (`finish`). A nonce the prover leaves to the challenger is sampled apart from
everything else, so a prover that reads it is a mixture of strategies here.

An opening samples its round anew among the samples with the leaf the `Hello` committed to. That gives the opened coins and
salts, jointly with the key and every leaf, the sampling design they have when the device keeps the samples it sampled at `Hello`, as
the deployed one does. A strategy sees every coin it's given, so the prover here sees the whole `Hello` sample, every
leaf among it, and each opening's whole round, where the deployed device answers the root and the key at `Hello` and an
opening's coins, the chain's link after them, the salt and the path: a deployed prover is a strategy that reads only
those, and its plays have the same law here. What the leaves tell of a round's coins before it's opened is what the
key's hm96 hiding gap bounds, at every leaf the device's trees hold, since the prover sees each before it chooses what
to open. -/
def releases : ℕ → List (String × ℕ) → List OpenTree → Game (List Release)
  | 0, _, _ => .ret []
  | k + 1, fs, ts => .send (Option (Frame × ByteArray)) fun
    | none => .ret []
    | some (.fresh l, b) =>
      match takeCoin l fs with
      | none => .ret []
      | some fs' => .coin (Fin 512 → Bool) fun x => (releases k fs' ts).map fun log => ⟨.fresh l, b, List.ofFn x⟩ :: log
    | some (.hello spec, ν) =>
      if treeLines ts < maxTreeLines ∧ TreeSpecOk spec then
        .coin (TreeSample spec) fun t =>
          (releases k fs (ts ++ [⟨ν, spec, t, fun _ => 0⟩])).map fun log =>
            ⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec t ++ treeKey t.1)⟩ :: log
      else .ret []
    | some (.opening i s n, b) =>
      if treeLines ts < maxTreeLines then
        match ts[i]? with
        | none => .ret []
        | some T =>
          match T.next s n with
          | none => .ret []
          | some (s', r) =>
            haveI : Nonempty (TreeOpenable T.nonce T.spec T.sample s' r) := ⟨⟨T.sample.2 s' r, rfl⟩⟩
            .coin (TreeOpenable T.nonce T.spec T.sample s' r) fun d =>
              (releases k fs (ts.set i (T.bump s'))).map fun log => ⟨.opening i s n, b, TreeDraw.opened d.1 n⟩ :: log
      else .ret []

/-- **The device's signer** in a play: the messages it signs under its LMS key, in the order it signs them, and its
signature on each. Before any coin it signs what the prover's request fixes (`first`: the ledger's reserve and open,
and the period's open). At the close it samples `draws` bits apart from every release, each coin's salt and the coins it
committed to and didn't release, and signs what the request, its log, the bytes the close leaves to it and its sample fix
(`last`: its record over its commitment to every coin and salt, its close over its log's line count and chain, the
ledger's end and the period's close). `sign i m` is its signature on `m` as its `i`-th in the play. -/
structure Signer where
  first : ByteArray → List Hash.Msg
  draws : ℕ
  last : ByteArray → List Release → ByteArray → List Bool → List Hash.Msg
  sign : ℕ → Hash.Msg → Hash.Msg

/-- The messages the device signs in a play, in order: what the request fixes, then the rest. -/
def Signer.signs (D : Signer) (a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) : List Hash.Msg :=
  D.first a ++ D.last a log b ρ

/-- The device's signatures in a play: each message it signs, with its signature at its place among them. -/
def Signer.signatures (D : Signer) (a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) :
    List (Hash.Msg × Hash.Msg) :=
  (D.signs a log b ρ).mapIdx fun i m => (m, D.sign i m)

/-- How a play ends: the parameters, the committed files and the commitment sent first, the prover's request, the
device's log, the bytes the prover's close leaves to the device, the device's sample after it and its signatures, and the
evidence sent last with its blobs. -/
structure Exchange where
  params : ByteArray
  files : Array ByteArray
  commitment : ByteArray
  request : ByteArray
  log : List Release
  close : ByteArray
  drawn : List Bool
  signed : List (Hash.Msg × Hash.Msg)
  evidence : ByteArray
  blobs : Array ByteArray

/-- What the prover sent the composite in the play that ended with `o`: the blobs its subprotocols read are the committed
files, then the evidence's blobs. -/
def Exchange.sent (o : Exchange) : Sent := ⟨o.params, o.commitment, o.evidence, o.files ++ o.blobs⟩

/-- **The play from the prover's close**, after its request `a` and the device's log `log`: the bytes `b` the close
leaves to the device (how the session ended, the times its lines carry), the device's sample `ρ` and its signatures on
what the play fixed (`Signer.signatures`), then the evidence with its blobs.

A withheld session releases its commitment and every opened coin's salt and path only once its signed close is written,
so sampling the salts and the unreleased coins at the close gives them, jointly with the released coins, the sampling design the
deployed device's sample before the session gives them. The prover here sees the whole sample, where the deployed one sees
the salts and no unreleased coin, and chooses the bytes its close leaves to the device, where the deployed device's
times and ending follow the prover's frames: a deployed prover is a mixture of strategies here. Each signature is the
device's on its message at its place among the play's, so the signatures on what the request fixes (`Signer.first`)
are fixed before any coin. The deployed device signs the ledger's reserve and open before any coin and releases those
signatures then, where the play here releases them at the close: they are functions of the request alone, so a
deployed prover that reads them before its first frame is a strategy here that hard-codes them after its request, and
the mixture covers it. A forger built from a strategy has the device sign the play's
messages on its coin (`Hash.Assumptions.LmsForger.signed`), each chosen by the strategy's moves before it: the
adaptive queries of the EUF-CMA game. -/
def finish (D : Signer) (θ : ByteArray) (F : Array ByteArray) (c a : ByteArray) (log : List Release) : Game Exchange :=
  .send ByteArray fun b => .coin (Fin D.draws → Bool) fun ρ => .send (ByteArray × Array ByteArray) fun e =>
    .ret ⟨θ, F, c, a, log, b, List.ofFn ρ, D.signatures a log b (List.ofFn ρ), e.1, e.2⟩

/-- **The causal game after the parameters `θ`, the committed files `F` and the commitment `c`**: the prover's request
`a`, then the device the prover drives under the coin commitment's families `fs`, from no coin tree open, each release
after the bytes it answers, then the close under the device's signer `D` (`finish`). -/
def play (fs : List (String × ℕ)) (D : Signer) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) : Game Exchange :=
  .send ByteArray fun a => (releases (frames fs) fs []).bind (finish D θ F c a)

/-- **The prover that closes the device at once**, then plays `t` in what follows the device's empty log. -/
def closeThen {β : Type} (f : List Release → Game β) (t : Game.Strategy (f [])) :
    (k : ℕ) → (fs : List (String × ℕ)) → (ts : List OpenTree) → Game.Strategy ((releases k fs ts).bind f)
  | 0, _, _ => t
  | _ + 1, _, _ => ⟨none, t⟩

/-- **The prover that sends the request `a` and closes the device at once**, then plays `t` from the close. -/
-- Stated at every `fs`: there `frames fs` stops at the families' sum, where at given families the kernel would unfold
-- `play`'s game frame by frame to check the pair against it.
def requestThen (fs : List (String × ℕ)) (D : Signer) (θ : ByteArray) (F : Array ByteArray) (c a : ByteArray)
    (t : Game.Strategy (finish D θ F c a [])) : Game.Strategy (play fs D θ F c) :=
  ⟨a, closeThen (finish D θ F c a) t (frames fs) fs []⟩

/-- SHA-512, the compiled one, as the message-to-message hash `Claims.Finder.Pr` and `ecr/sha-512`
(`Claims.ExpectedCR`) read. -/
def sha512Msg (m : Claims.Msg) : Claims.Msg := (Flock.Sha512.hash ⟨m.toArray⟩).data.toList

open Classical in
/-- **The committed file a digest the parameters hold names**: the first of the committed files `F` whose SHA-512 is
`h`, as v2 looks the circuit file and the public file up by their digests; empty when none is. -/
noncomputable def namedFile (h : Option Hash.Digest) (F : Array ByteArray) : ByteArray :=
  (F.find? fun f => decide (h = some (sha512 f.data.toList))).getD .empty

/-- **The finder for a second file with the digest `h`**: on its one coin it hashes the file of the committed files `F`
that `h` names and the one of the registered files `R`, and outputs them when they differ, a SHA-512 collision when both
have the digest. -/
noncomputable def namedFinder (h : Option Hash.Digest) (F R : Array ByteArray) : Claims.Finder where
  S := Unit
  out _ := if namedFile h F = namedFile h R then none
    else some ((namedFile h F).data.toList, (namedFile h R).data.toList)
  cost _ := 2

/-- **The compiled LMS verifier** at the deployed parameters (`Lms.deployed`), on a public key, a message and a
signature's bytes: the signature decodes and verifies under the key. The device's releases are signed under it. -/
def lmsVerify (pub m sig : Hash.Msg) : Bool :=
  match Lms.Signature.decode Lms.deployed ⟨sig.toArray⟩ >>= Lms.verify Lms.deployed ⟨pub.toArray⟩ ⟨m.toArray⟩ with
  | .ok b => b
  | .error _ => false

/-- **How v2 reads the committed files**: its statement's tags (derived from the circuit file), the circuit file, its
lookup tables and the public file. -/
structure FileRead where
  tags : Flock.Tags
  circuitFile : ByteArray
  tables : List (String × ByteArray)
  publicFile : ByteArray

namespace FileRead

/-- The committed files as core's circuit, as the verifier reads them (`Flock.Bridge.circuitOfFile`). -/
noncomputable def circuit (r : FileRead) : Verity.Circuit :=
  Flock.Bridge.circuitOfFile r.tags r.circuitFile r.tables r.publicFile

/-- The committed files' number of units, the public file's population. -/
noncomputable def units (r : FileRead) : ℕ := Flock.Bridge.unitsOfFile r.tags r.circuitFile r.tables r.publicFile

/-- The committed files' units, as core's partition of `circuit`. -/
noncomputable def parts (r : FileRead) : Verity.Partition r.circuit r.units :=
  Flock.Bridge.partsOfFile r.tags r.circuitFile r.tables r.publicFile

end FileRead

/-- **A parse's constants**: the key program's constant input gate holds `1` and its zero gate `0`
(`FlockSoundness.Discharge.Placed.oneGate`, `zeroGate`), public circuit structure that no proof block opens. -/
noncomputable def parsedConsts : (o : Option Flock.Bridge.Parsed) → Fin (Flock.Bridge.Parsed.circuit o).N → Option Bool
  | some r, g =>
    if g.val = (FlockSoundness.Discharge.Placed.oneGate r.n (FlockSoundness.Discharge.Layout.keyB r.D.unroll r.hU)).val
    then some true
    else if g.val =
        (FlockSoundness.Discharge.Placed.zeroGate r.n (FlockSoundness.Discharge.Layout.keyB r.D.unroll r.hU)).val
    then some false
    else none
  | none, _ => none

/-- The committed files' constants. -/
noncomputable def FileRead.consts (r : FileRead) : Fin r.circuit.N → Option Bool :=
  parsedConsts (Flock.Bridge.parsedOf r.tags r.circuitFile r.tables r.publicFile)

/-- A value's registered position: the row-leaf verifier (the frame-v3 domain, the hm96 key and the row format), the
commitment's root it opens against, the leaf's rank, and the value's first bit in the row. -/
structure Place where
  verifier : Hash.RowLeaf.Verifier
  root : Hash.Digest
  rank : ℕ
  offset : ℕ

/-- **A further collision finder a subprotocol's bound reads, with its budget**: a finder built from the strategy, and
its SHA-512 budget as a function of the strategy's calls alone (`budget`, as `q_F = 2t + 2v` and `q_S = t + v` are of
`t`), which bounds the finder's calls on every coin (`calls`). -/
structure FinderSpec (G : Game Exchange) (cost : Game.Strategy G → ℕ) where
  finder : Game.Strategy G → Claims.Finder
  budget : ℕ → ℕ
  calls : ∀ σ (x : (finder σ).S), (finder σ).cost x ≤ budget (cost σ)

/-- **A further finder's collision premise, at its budget**: applied to the strategy, it outputs a SHA-512 collision with
probability at most its budget at the strategy's calls over `2^256`, what `ecr/sha-512` gives a finder of that many
calls. -/
def FinderSpec.CR {G : Game Exchange} {cost : Game.Strategy G → ℕ} (f : FinderSpec G cost) (σ : Game.Strategy G) :
    Prop :=
  (f.finder σ).Pr sha512Msg ≤ (f.budget (cost σ) : ℝ) / 2 ^ (256 : ℝ)

/-- **The collision premises at a list of further finders**, each at its own budget (`FinderSpec.CR`): `True` at no
finder. -/
def finderCROf {G : Game Exchange} {cost : Game.Strategy G → ℕ} (fs : List (FinderSpec G cost))
    (σ : Game.Strategy G) : Prop :=
  fs.Forall fun f => f.CR σ

/-- The fields of a period's run that the prover's world realizes in a play: whose account each resource is charged to,
the window, the device schedule, and what the network certifiers saw (`Accounting.PeriodWorld`, the one definition). -/
abbrev World {ι : Type} (d : Device ι) (α : Type) [Fintype α] := Accounting.PeriodWorld d α

/-- **The names the subprotocols define.** v2's reading of the committed files at the parameters; the positions from
the files and the commitment, the circuit file's and the public file's digests in the parameters and the registered
files; the coin commitment's families and the device's signer, the device's only parts the parameters fix (its frames
are the prover's: `Frame`), and the device's log as each compiled check read it, decoded from the device's signed
log in what the prover sent; the extractor applied to a strategy, and each subprotocol's collision finder and forger of
the device's signature, with their SHA-512 calls bounded by the strategy's (`finder_calls`, `forger_calls`), and the
device's messages in a play and the forger's on each of its coins bounded by the published `Q` (`signs_le`,
`forger_signed`), and the other finders each subprotocol's bound reads,
each with its budget in the strategy's calls (`finders`); the committed circuit's gates a play computed after
their window's coin; the world each play realizes; the declaration's registered inputs and the inference-only policy's
budgets; the wipe that opens the period, at a boundary of a schedule of the committed circuit's blocks, and each play
read at that boundary as the wipe's game;
the deployment's honest prover; and the certificates, survivor threshold, errors, forgery bounds and losses the
composite publishes, from the parameters and the strategy's SHA-512 calls alone.

`cost θ F c σ` is the strategy's SHA-512 calls in one run (`t' τ`): `Verity.Game` has no call model of
strategies, so it is declared, as `Claims.Finder.cost` is. -/
structure Vocab {ι : Type} (d : Device ι) (α : Type) [Fintype α] where
  /-- v2's reading of the committed files at the parameters: the circuit file and the public file their digests name -/
  read : ByteArray → Array ByteArray → FileRead
  /-- the circuit file's digest the parameters hold -/
  programDigest : ByteArray → Option Hash.Digest
  /-- the public file's digest the parameters hold -/
  publicDigest : ByteArray → Option Hash.Digest
  place : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Fin (read θ F).circuit.N → Option Place
  /-- the registered files, fixed by the parameters -/
  registered : ByteArray → Array ByteArray
  /-- the coin commitment's families, each its label and its number of coins, fixed by the parameters before any coin -/
  families : ByteArray → List (String × ℕ)
  /-- the device's signer, what it signs in a play and its signatures, fixed by the parameters before any coin -/
  signer : ByteArray → Signer
  /-- the device's log as subprotocol `a`'s check read it, from the signed log in what the prover sent; `none` where its
  check reads none -/
  coinsRead : Subprotocol → Sent → Option (List Release)
  extract : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Game.Strategy (play (families θ) (signer θ) θ F c) → (read θ F).circuit.Trace
  finder : Subprotocol → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Game.Strategy (play (families θ) (signer θ) θ F c) → Claims.Finder
  cost : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Game.Strategy (play (families θ) (signer θ) θ F c) → ℕ
  callsMul : ByteArray → Subprotocol → ℕ
  callsAdd : ByteArray → Subprotocol → ℕ
  /-- **The finder's SHA-512 calls are at most `c · calls(τ) + d`**, on every coin of the finder. -/
  finder_calls : ∀ a θ F c σ (x : (finder a θ F c σ).S),
    (finder a θ F c σ).cost x ≤ callsMul θ a * cost θ F c σ + callsAdd θ a
  /-- **The other finders subprotocol `a`'s bound reads**, each built from the strategy, with its budget a function of
  the strategy's calls (`cost`) that bounds its calls (`FinderSpec`); none where the bound reads only `finder`. -/
  finders : Subprotocol → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    List (FinderSpec (play (families θ) (signer θ) θ F c) (cost θ F c))
  /-- each subprotocol's forger of the device's signature, applied to the strategy -/
  forger : Subprotocol → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Game.Strategy (play (families θ) (signer θ) θ F c) → Hash.Assumptions.LmsForger
  /-- **The forger's SHA-512 calls are at most `c · calls(τ) + d`**, the finder's budget, on every coin of the
  forger. -/
  forger_calls : ∀ a θ F c σ (x : (forger a θ F c σ).S),
    (forger a θ F c σ).cost x ≤ callsMul θ a * cost θ F c σ + callsAdd θ a
  /-- **`Q`**, fixed by the parameters: the most messages the device signs in a play, its records, closes and ledger
  lines -/
  signatures : ByteArray → ℕ
  /-- the device signs at most `Q` messages in every play -/
  signs_le : ∀ θ a log b ρ, ((signer θ).signs a log b ρ).length ≤ signatures θ
  /-- the forger has at most `Q` of the device's messages signed on each of its coins -/
  forger_signed : ∀ a θ F c σ (x : (forger a θ F c σ).S), ((forger a θ F c σ).signed x).length ≤ signatures θ
  computed : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Game.Strategy (play (families θ) (signer θ) θ F c) → Exchange → Finset (Fin (read θ F).circuit.N)
  world : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Game.Strategy (play (families θ) (signer θ) θ F c) → Exchange → World d α
  /-- the window PoUS's and PoUB-P2's claims read, fixed by the parameters before any coin -/
  window : ByteArray → Accounting.Window
  /-- the account PoUS's claim charges device memory to, fixed by the parameters before any coin -/
  capacity : ByteArray → α
  /-- every play's world is read over the parameters' window, never one PoUS's own coins chose -/
  world_window : ∀ θ F c σ o, (world θ F c σ o).win = window θ
  /-- every play's world charges device memory to the parameters' account, never one PoUS's own coins chose -/
  world_capacity : ∀ θ F c σ o, (world θ F c σ o).acct Accounting.Resource.capacity = capacity θ
  /-- the declaration's registered inputs `θ`, input gates of the committed files' circuit -/
  theta : (θ : ByteArray) → (F : Array ByteArray) → Finset (Fin (read θ F).circuit.N)
  theta_input : ∀ θ F, ∀ g ∈ theta θ F, (read θ F).circuit.isInput g = true
  /-- the inference-only policy's maximum inference block size `Γ`, in gates -/
  gates : ByteArray → ℕ
  /-- the inference-only policy's limit `α` on fresh input, in bits per gate -/
  fresh : ByteArray → ℝ
  /-- the number of blocks of the committed circuit's schedule -/
  spans : ByteArray → Array ByteArray → ℕ
  /-- the schedule of the committed circuit's blocks the wipe sits in -/
  schedule : (θ : ByteArray) → (F : Array ByteArray) → Verity.Blocks (read θ F).circuit (spans θ F)
  /-- the boundary the wipe opens -/
  boundary : (θ : ByteArray) → (F : Array ByteArray) → Fin (spans θ F)
  /-- the wipe's seeds -/
  Seed : Type
  /-- the wipe's spot reads -/
  Spots : Type
  /-- the wipe that opens the period, on the device -/
  wipe : (θ : ByteArray) → (F : Array ByteArray) → Verity.Wipe d (schedule θ F) (boundary θ F) Seed Spots
  /-- each play, read at the wipe's boundary as the wipe's game, with the strategy's moves there -/
  wipeRun : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    (σ : Game.Strategy (play (families θ) (signer θ) θ F c)) → (wipe θ F).OfRun (play (families θ) (signer θ) θ F c)
  /-- the survivor threshold the wipe publishes with an accepted verdict -/
  foreignBits : ByteArray → ℕ
  honest : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Game.Strategy (play (families θ) (signer θ) θ F c)
  certOf : ByteArray → Certificates
  delta : ByteArray → Subprotocol → ℕ → ℝ≥0∞
  /-- **The published `ε_forge(t, Q)`**: the bound `Hash.Assumptions.LmsUnforgeable` gives a forger of at most `t`
  SHA-512 calls that has at most `Q` of the device's messages signed on each of its coins -/
  epsForge : ByteArray → ℕ → ℕ → ℝ
  loss : Subprotocol → ByteArray → ℝ≥0∞ → ℝ≥0∞

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- The game after `θ`, `F` and `c`. -/
def game (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) : Game Exchange :=
  play (V.families θ) (V.signer θ) θ F c

/-- **The circuit file**: the committed file the parameters name by the circuit file's digest (`namedFile`). -/
noncomputable def circuitFile (θ : ByteArray) (F : Array ByteArray) : ByteArray := namedFile (V.programDigest θ) F

/-- **The public file**: the committed file the parameters name by the public file's digest (`namedFile`). -/
noncomputable def publicFile (θ : ByteArray) (F : Array ByteArray) : ByteArray := namedFile (V.publicDigest θ) F

/-- The committed files' circuit. -/
noncomputable def circuitOf (θ : ByteArray) (F : Array ByteArray) : Verity.Circuit := (V.read θ F).circuit

/-- The committed files' number of units. -/
noncomputable def units (θ : ByteArray) (F : Array ByteArray) : ℕ := (V.read θ F).units

/-- The committed files' units. -/
noncomputable def partsOf (θ : ByteArray) (F : Array ByteArray) : Verity.Partition (V.circuitOf θ F) (V.units θ F) :=
  (V.read θ F).parts

/-- **The declaration the inference-only policy judges**: the committed circuit, with its registered inputs. -/
noncomputable def declaration (θ : ByteArray) (F : Array ByteArray) : RegisteredCircuit :=
  .ofCircuit (V.circuitOf θ F) (V.theta θ F) (V.theta_input θ F)

/-- **The commit phase's file check**: the committed files hold a file with each digest the parameters hold, the
circuit file's and the public file's. -/
def Registered (θ : ByteArray) (F : Array ByteArray) : Prop :=
  (∃ f ∈ F, V.programDigest θ = some (sha512 f.data.toList)) ∧ ∃ f ∈ F, V.publicDigest θ = some (sha512 f.data.toList)

/-- **The commitment `c` opens gate `g` to `v`**: a registered constant to its constant; any other gate by an opening
of the row leaf at its registered position that the row-leaf verifier accepts under SHA-512, whose row holds `v`'s bits
from the position's offset. -/
def Opens (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (g : Fin (V.circuitOf θ F).N)
    (v : BitVec ((V.circuitOf θ F).w g)) : Prop :=
  match (V.read θ F).consts g with
  | some b => ∀ i : Fin ((V.circuitOf θ F).w g), v.getLsbD i.val = b
  | none => ∃ p, V.place θ F c g = some p ∧ ∃ o : Hash.RowLeaf.Opening,
      Hash.RowLeaf.accepts sha512 sha512 p.verifier p.root p.rank o ∧
        ∀ i : Fin ((V.circuitOf θ F).w g), o.row[p.offset + i.val]? = some (v.getLsbD i.val)

/-- **Subprotocol `a`'s verdict at composite `k`**: the committed files passed the commit phase's check, `k` accepted
what the prover sent as a `same` composition, and `a`'s compiled check accepts its own case in it. -/
def Verdict (k : Verity.Registration.Composite) (a : Subprotocol) (o : Exchange) : Prop :=
  V.Registered o.params o.files ∧ SameAccepts k o.sent ∧ ComponentAccepts k a o.sent

/-- **The period after `θ`, `F` and `c` at composite `k`, as the one vocabulary's `Accounting.PeriodLive`**: the
committed files' units, the commitment's openings, the extractor, the finders, the gates computed, the world and the
verdicts. -/
noncomputable def live (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    PeriodLive d α (V.circuitOf θ F) (V.game θ F c) where
  units := V.units θ F
  parts := V.partsOf θ F
  opens := V.Opens θ F c
  extract σ _ := V.extract θ F c σ
  finder a σ := V.finder (.ofAudit a) θ F c σ
  computed := V.computed θ F c
  world := V.world θ F c
  accepts a := V.Verdict k (.ofAudit a)

/-- **The claim subprotocol `a` certifies** in the play of `σ` that ended with `o`. The five period claims are
`Accounting.PeriodLive.Holds` at the period and the published certificates: sampled proofs' on the extractor's
transcript at `σ` (at most an `ε` share of the units unbound in it under the commitment's openings), PoUW's in gates (at
least `W` gates of the committed circuit computed after their window's coin), and the others' on the world the play
realized (at least `S` bits of device memory held on average, at least `Q` bits read, at most `K` bits through the
network certifiers), whose window and capacity account are the parameters' (`world_window`, `world_capacity`), so PoUS's
own coins choose neither. The inference-only policy's: the declaration complies with `Nci.CompliesRate` at the published `Γ` and
`α`. The wipe's: the play, read at the wipe's boundary as the wipe's game, doesn't keep foreign content past the
published survivor threshold (`Verity.Wipe.OfRun.Kept`). -/
def Holds (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)) :
    Subprotocol → Exchange → Prop
  | .sampled, o => (V.live k θ F c).Holds (V.certOf θ) σ .sampled o
  | .pouw, o => (V.live k θ F c).Holds (V.certOf θ) σ .pouw o
  | .pous, o => (V.live k θ F c).Holds (V.certOf θ) σ .pous o
  | .poub, o => (V.live k θ F c).Holds (V.certOf θ) σ .poub o
  | .network, o => (V.live k θ F c).Holds (V.certOf θ) σ .network o
  | .inferenceOnly, _ => Nci.CompliesRate (V.gates θ) (V.fresh θ) (V.declaration θ F)
  | .wipe, o => ¬ (V.wipeRun θ F c σ).Kept (V.foreignBits θ) o

/-- **The log subprotocol `a`'s check read is the device's whole log**: every release, in order, and no other
(`coinsRead`; a check that reads none binds nothing). The device signs its log's line count and the lines' SHA-512 chain
at its close (`close_session`), and v2 refuses a session whose signed close is missing or doesn't match its log
(`Spot.Challenger.sessionOf`), so any other log a check read is a forgery of the device's LMS signature, on a
message the device didn't sign in the play (`Exchange.signed`), which `a`'s error bounds where its forger meets
`Hash.Assumptions.LmsUnforgeable` (`Vocab.Unforgeable`, `Vocab.epsForge`), or a SHA-512 collision of the chain, which
its finder outputs. -/
def Logged (a : Subprotocol) (o : Exchange) : Prop := ∀ log ∈ V.coinsRead a o.sent, log = o.log

/-- **The finder for a second circuit file with the registered digest** (`namedFinder`): it outputs the circuit file and
the registered one when they differ, a SHA-512 collision when the files passed the file check. -/
noncomputable def fileFinder (θ : ByteArray) (F : Array ByteArray) : Claims.Finder :=
  namedFinder (V.programDigest θ) F (V.registered θ)

/-- **The finder for a second public file with the registered digest** (`namedFinder`): it outputs the public file and
the registered one when they differ, a SHA-512 collision when the files passed the file check. -/
noncomputable def publicFinder (θ : ByteArray) (F : Array ByteArray) : Claims.Finder :=
  namedFinder (V.publicDigest θ) F (V.registered θ)

/-- **The SHA-512 premises at the other finders subprotocol `a`'s bound reads** (`finders`), each at its own budget at
the strategy's calls (`finderCROf`); `True` where the bound reads only `finder`. -/
def finderCR (a : Subprotocol) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)) : Prop :=
  finderCROf (V.finders a θ F c) σ

/-- **Subprotocol `a`'s finder with its budget** (`finder_calls`): `callsMul θ a * t + callsAdd θ a` SHA-512 calls at
the strategy's `t`, a defined function of the strategy's `cost` as `qF`/`qS` are of `t'`. -/
def finderSpec (a : Subprotocol) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    FinderSpec (V.game θ F c) (V.cost θ F c) where
  finder := V.finder a θ F c
  budget t := V.callsMul θ a * t + V.callsAdd θ a
  calls σ x := V.finder_calls a θ F c σ x

/-- **`ε_cr`, subprotocol `a`'s finder's collision bound at `t` calls**: its budget over `2^256`, what `ecr/sha-512`
gives a finder of that many calls. -/
noncomputable def crBound (θ : ByteArray) (a : Subprotocol) (t : ℕ) : ℝ :=
  ((V.callsMul θ a * t + V.callsAdd θ a : ℕ) : ℝ) / 2 ^ (256 : ℝ)

/-- **`B^τ`'s collision premise**: `ecr/sha-512` (`Claims.ExpectedCR`) at subprotocol `a`'s finder applied to the
strategy. Its calls are within its budget on every coin (`finderSpec`), so it outputs a SHA-512 collision with
probability at most `crBound` at the strategy's calls. -/
def FinderCR (a : Subprotocol) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)) : Prop :=
  Claims.ExpectedCR sha512Msg (V.finder a θ F c σ)

/-- **`B^τ`'s forgery premise**: subprotocol `a`'s forger applied to the strategy meets
`Hash.Assumptions.LmsUnforgeable` at the compiled LMS verifier, at the finder's budget of SHA-512 calls
(`callsMul θ a * cost θ F c σ + callsAdd θ a`, `forger_calls`) and at most `Q` of the device's messages signed on each
of its coins (`forger_signed`). -/
def Unforgeable (a : Subprotocol) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)) : Prop :=
  Hash.Assumptions.LmsUnforgeable lmsVerify (V.epsForge θ) (V.callsMul θ a * V.cost θ F c σ + V.callsAdd θ a)
    (V.signatures θ) (V.forger a θ F c σ)

/-- **Subprotocol `a`'s error at a strategy**: where its finders meet their collision premises (`finderCR`, `FinderCR`)
and its forger its forgery premise (`Unforgeable`), its published `δ` at the strategy's SHA-512 calls, plus the published
`ε_forge(t, Q)` at the forger's budget and the device's signatures (`epsForge`), plus its loss at the probability that
the finder outputs a collision, the form of `Accounting.PeriodLive.error` with the forgery term beside `δ`; elsewhere no
bound (`⊤`), as `Accounting.condError` reads it. Flock's conditional bound is the case `loss = 0`. -/
noncomputable def error (a : Subprotocol) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)) : ℝ≥0∞ :=
  condError (fun σ => V.finderCR a θ F c σ ∧ V.FinderCR a θ F c σ ∧ V.Unforgeable a θ F c σ)
    (fun σ => V.delta θ a (V.cost θ F c σ) +
      ENNReal.ofReal (V.epsForge θ (V.callsMul θ a * V.cost θ F c σ + V.callsAdd θ a) (V.signatures θ)) +
      V.loss a θ (ENNReal.ofReal ((V.finder a θ F c σ).Pr sha512Msg))) σ

/-- **Soundness at composite `k` for the acceptance `acc`, in the causal game.** At all parameters `θ`, committed files
`F` and commitments `c`, and for every strategy, the probability that `acc` holds of what the prover sent for files
that passed the commit phase's check while the claim of one of `k`'s subprotocols fails is at most their errors at that
strategy summed. -/
def SoundnessFor (k : Verity.Registration.Composite) (acc : Sent → Prop) : Prop :=
  ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => acc o.sent ∧ V.Registered o.params o.files ∧ ∃ a ∈ subprotocols k, ¬ V.Holds k θ F c σ a o)
        (V.game θ F c) σ ≤
      ∑ a ∈ subprotocols k, V.error a θ F c σ

/-- **Soundness at composite `k`'s compiled verifier** (`Accepts k`). -/
def SoundnessAt (k : Verity.Registration.Composite) : Prop := V.SoundnessFor k (Accepts k)

/-- **Subprotocol `a`'s target at composite `k`**: at all `θ`, `F` and `c`, for every strategy, the probability that
`a`'s verdict at `k` accepts while its claim fails is at most its error at that strategy. For a subprotocol that decides
`Audit` `b` it is the per-period guarantee `Accounting.PeriodLive.Sound` at `live` (`Vocab.target_audit`). -/
def Target (k : Verity.Registration.Composite) (a : Subprotocol) : Prop :=
  ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => V.Verdict k a o ∧ ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ ≤ V.error a θ F c σ

/-- **Sampled proofs' target**: at most an `ε` share of the units unbound in the extractor's transcript. -/
def SpotSound (k : Verity.Registration.Composite) : Prop := V.Target k .sampled

/-- **PoUW's target**: at least `W` gates of the committed circuit computed after their window's coin. -/
def PouwSound (k : Verity.Registration.Composite) : Prop := V.Target k .pouw

/-- **The inference-only policy's target**: the declaration over the committed circuit complies with `Nci.CompliesRate`
at the published `Γ` and `α`. -/
def InferenceOnlySound (k : Verity.Registration.Composite) : Prop := V.Target k .inferenceOnly

/-- **PoUS's target**: at least `S` bits of device memory held on average. -/
def PousSound (k : Verity.Registration.Composite) : Prop := V.Target k .pous

/-- **PoUB-P2's target** (`verity/remote-bandwidth/p2/v0`, on PoUS's store): at least `Q` bits read from device
memory. -/
def BandwidthAuditSound (k : Verity.Registration.Composite) : Prop := V.Target k .poub

/-- **The network certifier's target**: at most `K` bits through the network certifiers. -/
def NetworkCertifierSound (k : Verity.Registration.Composite) : Prop := V.Target k .network

/-- **The wipe's target**: the play doesn't keep foreign content past the survivor threshold the wipe publishes. -/
def WipeSound (k : Verity.Registration.Composite) : Prop := V.Target k .wipe

/-- **Subprotocol `a`'s honest lemma at the deployment**: at registered files, the deployment's honest prover meets
`a`'s claim at composite `k` on almost every play. No theorem proves it for any subprotocol: the soundness theorem
takes it as a hypothesis, one per subprotocol (`Proofs.Properties.Verity.Soundness`). -/
def HonestHolds (k : Verity.Registration.Composite) (a : Subprotocol) : Prop :=
  ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray), V.Registered θ F →
    Game.prob (fun o => ¬ V.Holds k θ F c (V.honest θ F c) a o) (V.game θ F c) (V.honest θ F c) = 0

end Vocab

end Verity
