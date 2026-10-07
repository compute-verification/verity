import PouwCommit.Coins
import PouwCommit.Rows

/-!
# PoUW's commitments: properties of the port (`PouwCommit`)

What the executable Lean of PoUW's commitments does, stated against its own functions, for every input. The numbers are
the protocol's: digests of 64 bytes (SHA-512) or 32 (SHA-256), hm96 salts of 192 bytes, coins of 32, two coins an epoch
and so a one-sibling path. A collision of SHA-512 is written in where binding rests on it (`Sha512Collision`): the binding
statements say that two different openings accepted against one commitment give one, which is the assumption's content,
taken as a disjunct rather than assumed away.
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

/-- `CommitmentDomain` refuses exactly a binding of neither 32 nor 64 bytes and an owner below -2, for any count and
either hash. -/
def DomainRefusals : Prop :=
  ∀ h binding owner count,
    (Domain.mk? h binding owner count).isOk = decide ((binding.size = 32 ∨ binding.size = 64) ∧ -2 ≤ owner)

/-- `Hm96Domain.leaf` refuses exactly a value shorter than its 192-byte salt, and in a tile tree one other than a 64-byte
digest and its salt; a plain tree's leaf refuses nothing. -/
def LeafRefusals : Prop :=
  ∀ (d : RowsDomain) rank schema (v : ByteArray),
    (d.leaf rank schema v).isOk = decide (d.hm96 = true → 192 ≤ v.size ∧ (d.digests = true → v.size = 256))

/-- `coin_reason` raises exactly on a label outside `LABELS`, or, once the epoch, the label and the sizes match, on a path
holding an item of neither 32 nor 64 bytes (`Opening`'s refusal). -/
def CoinRaises : Prop :=
  ∀ c o label after, (coinReason c o label after).isOk = false ↔
    label ∉ LABELS ∨
      (o.epoch = c.epoch ∧ o.label = label ∧ o.coin.size = 32 ∧ o.salt.size = 192 ∧
        o.path.any (fun p => p.size != 32 && p.size != 64) = true)

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

/-- `coin_reason` accepts (gives no reason) exactly the openings `CoinOpens` describes. -/
def CoinAccepts : Prop := ∀ c o label after, coinReason c o label after = .ok none ↔ CoinOpens c o label after

/-! ## Binding -/

/-- Two different messages with one SHA-512 digest. -/
def Sha512Collision : Prop := ∃ m m' : ByteArray, m ≠ m' ∧ Sha512.hash m = Sha512.hash m'

/-- A commitment opens to one coin per label: two openings of the same label accepted against one commitment carry the
same coin, or SHA-512 has a collision. -/
def CoinBinding : Prop :=
  ∀ c o o' label after after', coinReason c o label after = .ok none → coinReason c o' label after' = .ok none →
    o.coin = o'.coin ∨ Sha512Collision

/-- The trees' framing separates them: two frame-v3 messages with different tags or parts differ, so equal digests of
different domain ids, leaves, nodes, pads or empty roots are a collision of the frame's hash. A tag's length is framed in 32
bits and a part's in 64, which bound them (Python's frame refuses longer ones). -/
def FrameInjective : Prop :=
  ∀ h tag tag' (parts parts' : List ByteArray), tag.toUTF8.size < 2 ^ 32 → tag'.toUTF8.size < 2 ^ 32 →
    (∀ p ∈ parts, p.size < 2 ^ 64) → (∀ p ∈ parts', p.size < 2 ^ 64) → frameHash h tag parts = frameHash h tag' parts' →
    (tag = tag' ∧ parts = parts') ∨ ∃ m m' : ByteArray, m ≠ m' ∧ h.run m = h.run m'

end Properties.Pouw.Commit
