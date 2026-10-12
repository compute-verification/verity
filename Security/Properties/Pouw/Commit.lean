import PouwCommit.Coins
import PouwCommit.Draws
import PouwCommit.Rows

/-!
# PoUW's commitments: properties of the port (`PouwCommit`)

What the executable Lean of PoUW's commitments does, stated against its own functions, for every input. The numbers are
the protocol's: digests of 64 bytes (SHA-512) or 32 (SHA-256), hm96 salts of 192 bytes, coins of 32, two coins an epoch
and so a one-sibling path, and derived keys and epoch salts of 32. Where binding rests on SHA-512, the statement's other
disjunct is a collision among the messages actually hashed: two different ones, one from each opening's check (or the two
framed messages), with one digest. That is collision resistance's content, taken as a disjunct rather than assumed away, and
bounded to the inputs, since some collision of SHA-512 exists outright.
-/

namespace Properties.Pouw.Commit

open Flock PouwCommit

/-! ## Sizes -/

/-- Every binding `tagged` gives, and so every weight, activation and transcript binding and the weights' root, is 64 bytes:
what every frame takes as a SHA-512 identity. -/
def TaggedSize : Prop := ∀ tag parts, (tagged tag parts).size = 64

/-- A frame-v3 domain's id, leaves, nodes, pads and empty root are digests of the frame's hash: 64 bytes under SHA-512,
32 under SHA-256. -/
def FrameSizes : Prop :=
  ∀ h binding owner count (d : Domain), Domain.mk? h binding owner count = .ok d →
    d.id.size = h.bytes ∧ d.empty.size = h.bytes ∧ (∀ r, (d.pad r).size = h.bytes) ∧
      (∀ rank schema v, (d.leaf rank schema v).size = h.bytes) ∧ (∀ lv i l r, (d.node lv i l r).size = h.bytes)

/-- The epoch coins' numbers: two coins, `salt` then `draw`, of 32 bytes; a tree of depth 1, so an opening's path is one
sibling; and a committed coin value (`leafValue`) of 64 bytes. -/
def CoinNumbers : Prop :=
  LABELS = ["salt", "draw"] ∧ COIN_BYTES = 32 ∧ depth 2 = 1 ∧ (∀ e, (coinDomain e).count = 2) ∧
    ∀ coin salt, (leafValue coin salt).size = 64

/-- `merkle_depth` on the served sizes: a path's length is `⌈log₂ count⌉`, so 13 siblings for 8192 tiles and 14 for 9216
(a window of 8192 tiles padded to the grid), and 20 for a million rows. -/
def DepthServed : Prop :=
  depth 0 = 0 ∧ depth 1 = 0 ∧ depth 8192 = 13 ∧ depth 8193 = 14 ∧ depth 9216 = 14 ∧ depth (2 ^ 20) = 20

/-! ## Refusals -/

/-- A domain `CommitmentDomain` gives has a binding of 32 or 64 bytes and an owner of at least -2, for any count and
either hash. -/
def DomainRefusals : Prop :=
  ∀ h binding owner count d, Domain.mk? h binding owner count = .ok d →
    (binding.size = 32 ∨ binding.size = 64) ∧ -2 ≤ owner

/-- A leaf `Hm96Domain.leaf` gives is of a value at least its 192-byte salt long, and in a tile tree of a 64-byte digest
and its salt. -/
def LeafRefusals : Prop :=
  (∀ frame digests rank schema (v x : ByteArray),
      (RowsDomain.mk frame true digests).leaf rank schema v = .ok x → 192 ≤ v.size) ∧
    ∀ frame rank schema (v x : ByteArray), (RowsDomain.mk frame true true).leaf rank schema v = .ok x → v.size = 256

/-- `coin_reason` answers (a reason or none) only for a label in `LABELS`, and, once the epoch, the label and the sizes
match, only for a path whose every item is 32 or 64 bytes (`Opening`'s refusal). -/
def CoinRaises : Prop :=
  ∀ c o label after r, coinReason c o label after = .ok r →
    label ∈ LABELS ∧
      (o.epoch = c.epoch → o.label = label → o.coin.size = 32 → o.salt.size = 192 →
        ∀ p ∈ o.path, p.size = 32 ∨ p.size = 64)

/-! ## What an accepted coin is -/

/-- The root over a coin's value at `rank` and the other coin's leaf digest `sib`, at the tree's one level. -/
def coinRoot (epoch rank : Nat) (coin salt sib : ByteArray) : ByteArray :=
  let d := coinDomain epoch
  let l := d.leaf rank SCHEME (leafValue coin salt)
  if rank % 2 = 0 then d.node 0 (rank / 2) l sib else d.node 0 (rank / 2) sib l

/-- An opening is epoch `c.epoch`'s `label` coin, opened after `after`: the epoch and label agree, the coin is 32 bytes
and its salt 192, the path is one 64-byte sibling, the 64-byte root is the one over the coin's value and that sibling, and
the opening followed `after`. -/
def CoinOpens (c : CoinCommitment) (o : CoinOpening) (label : String) (after : ByteArray) : Prop :=
  ∃ rank sib, LABELS.idxOf? label = some rank ∧ o.epoch = c.epoch ∧ o.label = label ∧ o.coin.size = 32 ∧
    o.salt.size = 192 ∧ o.path = [sib] ∧ sib.size = 64 ∧ c.root.size = 64 ∧
    c.root = coinRoot c.epoch rank o.coin o.salt sib ∧ o.after = after

/-- An opening `coin_reason` accepts (gives no reason) is one `CoinOpens` describes. -/
def CoinAccepts : Prop := ∀ c o label after, coinReason c o label after = .ok none → CoinOpens c o label after

/-! ## Salts and draw keys -/

/-- A key PoUW derives (`derive` under its domains) is from a nonempty source. -/
def KeyRefusals : Prop :=
  ∀ source domain context k, key? source domain context = .ok k → source.size ≠ 0

/-- Every key PoUW derives is 32 bytes: the epoch's salt key, the audit key, and the Pearl-C draw and replay keys. -/
def KeySize : Prop :=
  ∀ source domain context k, key? source domain context = .ok k → k.size = 32

/-- An epoch's salt, from its salt coin (`salt_of`) or from a bare seed (`Epoch.unverified`), is 32 bytes. -/
def SaltSize : Prop :=
  (∀ c coin weightsRoot s, saltOf c coin weightsRoot = .ok s → s.size = SALT_BYTES) ∧
    ∀ seed weightsRoot s, unverifiedSalt seed weightsRoot = .ok s → s.size = SALT_BYTES

/-- A salt `salt_of` gives is from a nonempty coin, and one `Epoch.unverified` gives from a nonempty seed, whatever the
epoch, the coin commitment and the weights' root. -/
def SaltRefusals : Prop :=
  (∀ c coin weightsRoot s, saltOf c coin weightsRoot = .ok s → coin.size ≠ 0) ∧
    ∀ seed weightsRoot s, unverifiedSalt seed weightsRoot = .ok s → seed.size ≠ 0

/-- A salt `Epoch.start` gives is from the epoch's salt coin opened after the weights' root, and is `salt_of` of the
opened coin. -/
def EpochSaltAccepts : Prop :=
  ∀ c o weightsRoot s, epochSalt c o weightsRoot = .ok (.ok s) →
    CoinOpens c o "salt" weightsRoot ∧ saltOf c o.coin weightsRoot = .ok s

/-- A coin `draw_coin` gives is the opened one, from an epoch with a coin commitment whose draw coin the opening opens after
what the verifier recorded. -/
def DrawCoinAccepts : Prop :=
  ∀ cc o after coin, drawCoin cc o after = .ok (.ok coin) → ∃ c, cc = some c ∧ CoinOpens c o "draw" after ∧ coin = o.coin

/-! ## Draws -/

/-- A work-weighted draw (`Sampled.select`) gives `t` tickets, each a position of the ticket list. -/
def DrawShape : Prop :=
  ∀ key tickets t ids, weightedDraw key tickets t = .ok ids → ids.length = t ∧ ∀ i ∈ ids, i < tickets.length

/-! ## Binding -/

/-- The node frames `Domain.verify` hashes climbing from the leaf digest `leaf` at `rank` along `path`, bottom first. -/
def pathFrames (d : Domain) (rank : Nat) (leaf : ByteArray) (path : List ByteArray) : List ByteArray :=
  (path.zipIdx.foldl (fun (acc : List ByteArray × ByteArray × Nat) (sib, lv) =>
    let (ms, dg, cur) := acc
    let m := if cur % 2 == 0 then frameBytes "node" [d.id, natBytes lv, natBytes (cur / 2), dg, sib]
      else frameBytes "node" [d.id, natBytes lv, natBytes (cur / 2), sib, dg]
    (ms ++ [m], d.hash.run m, cur / 2)) ([], leaf, rank)).1

/-- What SHA-512 hashes when `coin_reason` checks the opening `o` of epoch `epoch`'s coin at `rank`, from the coin up: the
coin's inner digest's input, the hm96 leaf's salt-digest and leaf inputs, the tree leaf's frame, and the frame of each node
on its path. The domain's id, the hm96 key and the prefixes are the same for every opening of one commitment's coin. -/
def coinHashed (epoch rank : Nat) (o : CoinOpening) : List ByteArray :=
  let d := coinDomain epoch
  let value := leafValue o.coin o.salt
  [COIN_TAG ++ o.coin, Hm96.Default512.saltPrefix ++ o.salt,
    Hm96.Default512.leafPrefix ++ Hm96.Default512.commitString (Sha512.hash (COIN_TAG ++ o.coin)) o.salt,
    frameBytes "leaf" [d.id, natBytes rank, natBytes rank, SCHEME.toUTF8, value]] ++
    pathFrames d rank (d.leaf rank SCHEME value) o.path

/-- A commitment opens to one coin per label: two openings of the same label accepted against one commitment carry the
same coin, or two different messages, one hashed in checking each opening, have one SHA-512 digest. -/
def CoinBinding : Prop :=
  ∀ c o o' label after after', coinReason c o label after = .ok none → coinReason c o' label after' = .ok none →
    o.coin = o'.coin ∨
      ∃ x ∈ coinHashed c.epoch (LABELS.idxOf label) o, ∃ y ∈ coinHashed c.epoch (LABELS.idxOf label) o',
        x ≠ y ∧ Sha512.hash x = Sha512.hash y

/-- The trees' framing separates them: frame-v3 messages with different tags or parts differ, so equal digests of
different domain ids, leaves, nodes, pads or empty roots are a collision of the frame's hash on the two framed messages. A
tag's length is framed in 32 bits and a part's in 64, and the frame refuses longer ones (`frameHash?`, as Python's does):
two frames it accepts with one digest `d`. -/
def FrameInjective : Prop :=
  ∀ h tag tag' (parts parts' : List ByteArray) d, frameHash? h tag parts = .ok d → frameHash? h tag' parts' = .ok d →
    (tag = tag' ∧ parts = parts') ∨
      (frameBytes tag parts ≠ frameBytes tag' parts' ∧ h.run (frameBytes tag parts) = h.run (frameBytes tag' parts'))

end Properties.Pouw.Commit
