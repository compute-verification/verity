module

public import CoreProtocol.Protocol
public import Poub.Receipts

/-!
# A window's receipt checks: `audit.py`'s `Session.close` sample and `Session.verify`

`verifyWindow` takes what the verifier holds of window t once its roots are final (its challenge, each pass's root,
plan and the appends' head received before it, the root before its first pass, the session's appends with their coins)
and its openings coin, samples `s` distinct receipts uniformly from the window's `U_t = Σ_i L_i`, and checks the
prover's openings of them, in the order sampled. Each opening, in `_check`'s order, fails with the first of:

1. `malformed`: not the sampled (pass, position), a block not 4096 bytes, a tile (none is bound), a salt not 192 bytes, a
   path sibling not 64 bytes, or a payload digest `d_j` present for a registered block or absent for an appended one;
   an origin position present for a block not of a re-key append, absent for one, or not its origin's leaf (`originLeaf`);
   or receipts not the size of the position's leaf;
2. `receipt-path`: the leaf, under its salt and path, not that leaf of the pass's receipt tree (bound to the pass's seed
   and plan) under the pass's root;
3. `codec`: the block not canonical, or its decoding under the seed's key (a registered block) or under the bound key
   of `d_j` and its append's coin `c_a` (an appended block) not a payload of that digest;
4. `payload-path`: the decoded payload, under its salt and path, not leaf j of the payload tree (a registered block) or
   of its append's tree, under that root; for a block of a re-key append, `origin-path` instead: the decoded payload,
   under its salt and path, not the origin's leaf under the origin's root (`originLeaf`);
5. `receipt`: the receipt recomputed from the block under the registration's kind and the pass's seed not the opened one.

A session with refresh also rejects a re-key off the rota with `origin` and a plan with `probation`
(`Session.receive`).

A window with another number of openings than samples is `malformed`. `tile` is never reached: a registration that binds
PoUW's tiles is refused (`Registration.make?`), a form no proof here covers.
-/

@[expose] public section

namespace Poub

def REGISTRATION_DOMAIN : ByteArray := tag0 "verity/remote-bandwidth/registration/v0"
def PAYLOAD_DOMAIN : ByteArray := tag0 "verity/remote-bandwidth/payload/v0"
def APPEND_DOMAIN : ByteArray := tag0 "verity/remote-bandwidth/append/v0"
def OPENINGS_DOMAIN : String := "verity/remote-bandwidth/openings/v0"
def LAYOUT : String := "store-order/v0"

/-- `audit.FAILURES`. -/
inductive Failure
  | late | order | step | plan | lifetime | append | coin | release | few | malformed
  | receiptPath | codec | payloadPath | tile | receipt | probation | originPath | origin
deriving BEq, Repr, DecidableEq

def Failure.name : Failure → String
  | .late => "late" | .order => "order" | .step => "step" | .plan => "plan" | .lifetime => "lifetime"
  | .append => "append" | .coin => "coin" | .release => "release" | .few => "few" | .malformed => "malformed"
  | .receiptPath => "receipt-path" | .codec => "codec" | .payloadPath => "payload-path" | .tile => "tile"
  | .receipt => "receipt" | .probation => "probation" | .originPath => "origin-path" | .origin => "origin"

/-- What the store is (`audit.Registration`), with its codec key. -/
structure Registration where
  node : ByteArray
  blocks : Nat
  payloadRoot : ByteArray
  codecSeed : ByteArray
  codec : Codec.Codec
  receipt : Kind
  rows : Nat
  key : Codec.Key

/-- A registration the port covers: a node of 1 to 64 bytes, `1 ≤ N < 2^64` and at most the codec's `max_blocks`, a
64-byte payload root, a 32-byte codec seed, the store-order/v0 layout, sketch rows the kind allows, and receipts of
blocks alone: `bind` refused. -/
def Registration.make? (node : ByteArray) (blocks : Nat) (payloadRoot codecSeed : ByteArray) (codec : Codec.Codec)
    (receipt : Kind) (bind : Bool) (layout : String) (rows : Nat) : Except String Registration := do
  if !(1 ≤ node.size && node.size ≤ 64) then throw "a node's id is 1 to 64 bytes"
  if !(storeBlocks blocks && blocks ≤ codec.maxBlocks) then throw "a store is 1 to max_blocks blocks"
  if payloadRoot.size ≠ 64 then throw "the payload root is 64 bytes"
  if codecSeed.size ≠ Codec.SEED_BYTES then throw "a codec seed is 32 bytes"
  if bind then throw "a registration that binds PoUW's tile outputs is not covered: the port refuses it"
  if layout ≠ LAYOUT then throw "the block layout is store-order/v0"
  if !receipt.rowsOk rows then throw "the sketch rows are not the receipt kind's"
  pure { node, blocks, payloadRoot, codecSeed, codec, receipt, rows, key := Codec.key codecSeed codec }

def CODEC_DOMAIN : String := "verity/remote-bandwidth/codec/v0"
def NONCE_BYTES : Nat := 32

/-- `audit.register`: the codec seed the verifier samples from its secret `source` and a fresh 32-byte `nonce` once the
payload root has arrived, `derive(source, CODEC_DOMAIN, {blocks, codec, node, nonce, payload}).stream("codec", 32)`,
and the registration under it (`Registration.make?`). The receipt kind, `bind`, the layout and the rows are not in
the seed's context: the store depends on none of them. -/
def Registration.register (source node : ByteArray) (blocks : Nat) (payloadRoot : ByteArray) (codec : Codec.Codec)
    (receipt : Kind) (bind : Bool) (layout : String) (rows : Nat) (nonce : ByteArray) : Except String Registration := do
  if source.size = 0 then throw "the source is a key holder's secret: nonempty bytes"
  if nonce.size ≠ NONCE_BYTES then throw "a registration's nonce is 32 bytes"
  let key := FlockSoundness.Randomness.derive source CODEC_DOMAIN [("blocks", .int blocks), ("codec", .str codec.name),
    ("node", .bytes node), ("nonce", .bytes nonce), ("payload", .bytes payloadRoot)]
  Registration.make? node blocks payloadRoot (FlockSoundness.Randomness.stream key [.str "codec"] Codec.SEED_BYTES)
    codec receipt bind layout rows

/-- `Registration.digest`: SHA512(REGISTRATION_DOMAIN ‖ frame(node) ‖ u64be N ‖ payload root ‖ codec seed ‖
frame(codec name) ‖ frame(receipt kind) ‖ u8 bind ‖ frame(layout) ‖ u32be rows), bind 0. -/
def Registration.digest (r : Registration) : ByteArray :=
  sha512 (REGISTRATION_DOMAIN ++ frameBytes r.node ++ be 8 r.blocks ++ r.payloadRoot ++ r.codecSeed
    ++ frameStr r.codec.name ++ frameStr r.receipt.name ++ be 1 0 ++ frameStr LAYOUT ++ be 4 r.rows)

/-- `payload_domain(node, N)`. -/
def payloadDomain (node : ByteArray) (blocks : Nat) : Flock.FV3.Sha512.Domain :=
  .make (sha512 (PAYLOAD_DOMAIN ++ frameBytes node)) 0 0 blocks

def Registration.payloadDomain (r : Registration) : Flock.FV3.Sha512.Domain := Poub.payloadDomain r.node r.blocks

/-- Where an append's payloads were first committed: by the append itself (`fresh`); by the registration
(`registered a`: its block `first + m` re-keys registered block `a + m`); or by the fresh append in slot `k` of the
session's schedule (`rekey k`: block `first + m` re-keys that append's block `first_k + m`, the whole append). A re-key
encodes those payloads again under its own coin and commits none of its own. -/
inductive Origin
  | fresh
  | registered (a : Nat)
  | rekey (k : Nat)
deriving BEq, Repr, DecidableEq, Inhabited

/-- A fresh append of a session with refresh holds at most this many blocks, so its tree, a re-key's origin path, is at
most 9 levels and an opening of a block of its re-key stays one proof block. -/
def REKEY_MAX : Nat := 512

def REKEY_DOMAIN : ByteArray := tag0 "verity/remote-bandwidth/rekey/v0"

/-- `u8 0 ‖ u64be a` for `registered a`, `u8 1 ‖ u64be k` for `rekey k`. -/
def Origin.bytes : Origin → ByteArray
  | .fresh => ByteArray.empty
  | .registered a => be 1 0 ++ be 8 a
  | .rekey k => be 1 1 ++ be 8 k

/-- A re-key's declaration, the root it sends in place of a payload root: SHA512(REKEY_DOMAIN ‖ registration digest ‖
u64be first ‖ u64be count ‖ origin bytes). -/
def rekeyRoot (registration : ByteArray) (first count : Nat) (origin : Origin) : ByteArray :=
  sha512 (REKEY_DOMAIN ++ registration ++ be 8 first ++ be 8 count ++ origin.bytes)

/-- An append the verifier received: store indices `first … first + count − 1`, its payload tree's root (a re-key's
declaration, `rekeyRoot`), the coin `c_a` it answered with, where its payloads were first committed (`fresh` in a
session without refresh), and its slot: with refresh, its index in the session's schedule (`Session.Params.slotAt`);
without, its place among the appends. -/
structure Append where
  first : Nat
  count : Nat
  root : ByteArray
  coin : ByteArray
  origin : Origin := .fresh
  slot : Nat := 0
deriving Inhabited

/-- `append_domain(registration digest, first, count)`. -/
def appendDomain (registration : ByteArray) (first count : Nat) : Flock.FV3.Sha512.Domain :=
  .make (sha512 (APPEND_DOMAIN ++ registration)) 0 first count

/-- A pass of a window as the verifier received it: its root, its plan, and the chain over the appends received before
it (`none` without one). -/
structure PassRecord where
  root : ByteArray
  plan : Plan
  head : Option ByteArray
deriving Inhabited

/-- Window t as the verifier holds it once its roots are final. -/
structure WindowRecord where
  index : Nat
  challenge : ByteArray
  /-- The root before the window's first pass: the previous window's last, or the registration's digest. -/
  previous : ByteArray
  passes : Array PassRecord

/-- `Session._previous(t, i)`: the root before pass i, chained to the appends received before pass i's root. -/
def WindowRecord.previousOf (w : WindowRecord) (i : Nat) : ByteArray :=
  chained (if i = 0 then w.previous else w.passes[i - 1]!.root) w.passes[i]!.head

/-- Pass i's seed. -/
def WindowRecord.seedOf (w : WindowRecord) (i : Nat) : ByteArray := seed w.challenge i (w.previousOf i)

/-- Receipt (i, p) of a window opened (`audit.Opening`). Indices arrive as integers. -/
structure Opening where
  pass : Int
  position : Int
  block : ByteArray
  leaf : ByteArray
  leafSalt : ByteArray
  receiptPath : Array ByteArray
  payloadSalt : ByteArray
  payloadPath : Array ByteArray
  tile : ByteArray
  digest : ByteArray
  /-- For a block of a re-key append, the store index of its payload's leaf under the origin's root, whose salt and
  path `payloadSalt` and `payloadPath` are; `none` for every other block. -/
  originPosition : Option Nat := none
deriving Inhabited

/-- What rejected a window: its failure and the sample it failed at, `none` for the window as a whole. -/
structure Rejection where
  failure : Failure
  at? : Option (Nat × Nat)
deriving BEq, Repr, DecidableEq

/-- What a window's check found besides its verdict, all witness: what rejected it, `none` when it accepts. -/
abbrev Findings := Option Rejection

/-- A window's outcome as `Core.Verdict`, with its findings (`audit.Verdict`): `accept` with none (`Verdict.accept`), or
`reject` with the failure and the sample it failed at (`Verdict.reject`); never `undecided`. -/
abbrev Verdict := Findings × Core.Verdict

def Verdict.accept : Verdict := (none, Core.Verdict.accept)

def Verdict.reject (failure : Failure) (at? : Option (Nat × Nat)) : Verdict :=
  (some ⟨failure, at?⟩, Core.Verdict.reject)

/-- What rejected the window, for a verdict other than `accept`: its findings' rejection, `malformed` when they hold
none, which no check here gives. -/
def Verdict.rejection (v : Verdict) : Rejection := v.1.getD ⟨.malformed, none⟩

/-- `Session.close`'s samples: `min(s, U)` distinct receipts (pass, position) uniform over the window's U receipts, from
its openings coin, in order. -/
def receiptSample (registration : ByteArray) (t : Nat) (coin : ByteArray) (samples : Nat) (lengths : Array Nat) :
    Option (Array (Nat × Nat)) := do
  let ends := (lengths.foldl (fun (acc : Array Nat × Nat) n => (acc.1.push (acc.2 + n), acc.2 + n)) (#[], 0)).1
  let total := ends.back?.getD 0
  let key := FlockSoundness.Randomness.derive coin OPENINGS_DOMAIN [("registration", .bytes registration), ("window", .int t)]
  let us ← FlockSoundness.Randomness.subsetExact key (min samples total) total [.str "openings"]
  pure (us.map fun u => let i := bisectRight ends u; (i, u - (if i = 0 then 0 else ends[i - 1]!)))

/-- The append holding store index `j ≥ N`: the last with `first ≤ j`. -/
def appendOf (appends : Array Append) (j : Nat) : Option Append :=
  let k := bisectRight (appends.map (·.first)) j
  if k = 0 then none else appends[k - 1]?

/-- Where the payload of store block `j` of append `a` was first committed, if `a` is a re-key: its leaf's store index,
and the root and domain of the tree holding it (the registration's payload tree, or that of the fresh append in slot k). -/
def originLeaf (reg : Registration) (digest : ByteArray) (appends : Array Append) (a : Append) (j : Nat) :
    Option (Nat × ByteArray × Flock.FV3.Sha512.Domain) :=
  match a.origin with
  | .fresh => none
  | .registered x => some (x + (j - a.first), reg.payloadRoot, reg.payloadDomain)
  | .rekey k => (appends.find? (·.slot == k)).map fun b =>
    (b.first + (j - a.first), b.root, appendDomain digest b.first b.count)

/-- `Session._check`: sample (i, p)'s opening, `none` if it checks. -/
def checkOpening (reg : Registration) (digest : ByteArray) (appends : Array Append) (w : WindowRecord)
    (receipter : Receipter) (i p : Nat) (o : Opening) : Option Failure := Id.run do
  let N := reg.blocks
  if o.pass ≠ i || o.position ≠ p || o.block.size ≠ Codec.BLOCK_BYTES || o.tile.size ≠ 0
      || o.leafSalt.size ≠ SALT_BYTES || o.payloadSalt.size ≠ SALT_BYTES
      || (o.receiptPath ++ o.payloadPath).any (·.size ≠ DIGEST_BYTES) then
    return some .malformed
  let pr := w.passes[i]!
  let j := pr.plan.block p
  if o.digest.size ≠ (if j < N then 0 else DIGEST_BYTES) then return some .malformed
  let origin := if j < N then none else (appendOf appends j).bind (originLeaf reg digest appends · j)
  if o.originPosition != origin.map (·.1) then return some .malformed
  let s := w.seedOf i
  let some y := leafReceipt o.leaf p pr.plan.length | return some .malformed
  if !verifyLeaf s pr.plan pr.root p o.leaf o.leafSalt o.receiptPath then return some .receiptPath
  let (decoded, dom, root) :=
    if j < N then (Codec.decode o.block j reg.key none, reg.payloadDomain, reg.payloadRoot)
    else match appendOf appends j with
      | some a => (Codec.decode o.block j reg.key (some (o.digest, some a.coin)), appendDomain digest a.first a.count, a.root)
      | none => (.error "no append holds the block", reg.payloadDomain, reg.payloadRoot)
  let .ok (some payload) := decoded | return some .codec
  match origin with
  | some (q, oroot, odom) =>
    if !odom.verifyOpening oroot odom.count q (hidden payload o.payloadSalt) SCHEMA o.payloadPath then
      return some .originPath
  | none =>
    if !dom.verifyOpening root dom.count j (hidden payload o.payloadSalt) SCHEMA o.payloadPath then
      return some .payloadPath
  if receipter.receipt j o.block != y then return some .receipt
  return none

/-- `Session.verify` on sampled receipts: the first opening that fails, with its sample, or accept. -/
def checkWindow (reg : Registration) (digest : ByteArray) (appends : Array Append) (w : WindowRecord)
    (sampled : Array (Nat × Nat)) (openings : Array Opening) : Verdict := Id.run do
  if openings.size ≠ sampled.size then return .reject .malformed none
  let mut receipters : Std.HashMap Nat Receipter := {}
  for k in [0:sampled.size] do
    let (i, p) := sampled[k]!
    let r := match receipters.get? i with
      | some r => r
      | none => receipter reg.receipt (w.seedOf i) reg.rows
    receipters := receipters.insert i r
    if let some f := checkOpening reg digest appends w r i p openings[k]! then return .reject f (some (i, p))
  return .accept

/-- `checkWindow` over contiguous chunks of `size` samples, each chunk on a thread of its own: the first chunk's
rejection in sample order, or accept. A chunk's verdict is `checkWindow`'s on its samples and their openings, so a
rejection names the sample and failure `checkWindow` on the whole window would (its earlier chunks accept, and its first
failing opening is the window's first), and an acceptance is every chunk's, each chunk's openings passing `checkOpening`
under the receipter of their pass's seed (`checkWindow_accept`). -/
def checkWindowChunked (size : Nat) (reg : Registration) (digest : ByteArray) (appends : Array Append)
    (w : WindowRecord) (sampled : Array (Nat × Nat)) (openings : Array Opening) : Verdict := Id.run do
  if openings.size ≠ sampled.size then return .reject .malformed none
  let size := max 1 size
  let tasks := (List.range ((sampled.size + size - 1) / size)).map fun c =>
    Task.spawn (prio := .dedicated) fun _ =>
      checkWindow reg digest appends w (sampled.extract (c * size) ((c + 1) * size))
        (openings.extract (c * size) ((c + 1) * size))
  for t in tasks do
    if t.get ≠ .accept then return t.get
  return .accept

/-- Window t's verdict: its samples from the openings coin, then each opening checked in the order sampled. `none` only
if the sample runs past its doubled reads, which it does with probability below 2^-256. -/
def verifyWindow (reg : Registration) (appends : Array Append) (w : WindowRecord) (openingsCoin : ByteArray)
    (samples : Nat) (openings : Array Opening) : Option Verdict := do
  let digest := reg.digest
  let sampled ← receiptSample digest w.index openingsCoin samples (w.passes.map (·.plan.length))
  pure (checkWindow reg digest appends w sampled openings)

end Poub
