module

public import Spot.HistoryTable
public import Definitions.Hash.HistoryTable
public import Properties.Verity.Sha512

/-!
# A history table's registered root, checked: the model's root (`Spot.HistoryTable.checkAt`)

`Spot.HistoryTable.checkAt` accepts a registered root only when it is `rootAt`, the frame-v3-sha512 root the verifier
computes over a history table's writers' `b ‖ c` at a power-of-two capacity, the placeholder's past the table's length.
`HistoryRootIsModel` says that root is the hash area's `Hash.HistoryTable.root` at SHA-512 (`Verity.sha512`), so
`Hash.Guarantees.HistoryReadBinding` speaks of the root the verifier holds:

* the domain is `domainAt`'s id, counting `cap` ranks, each its own position (`historyDomain`);
* the key is hm96-sha512's default key, the one `Flock.Hm96.Default512.treeLeaf` reads;
* the strings are the written rows' `b ‖ c` below the table's length and the placeholder of `words` words past it
  (`Hash.HistoryTable.string`).

It takes no premise: `checkAt` itself refuses a capacity that is not a power of two of at least 2, where the code's tree
would pad and the model's would not.
-/

@[expose] public section

namespace Properties.Spot

/-- A history table's domain in the hash area's terms: the id `dom`, counting `cap` ranks, each its own position. -/
def historyDomain (cap : ℕ) (dom : ByteArray) : Hash.Tree.Domain := ⟨fun i => dom.data.getD i.val 0, cap, id⟩

/-- **An accepted history table's registered root is the model's root**: when `checkAt` accepts `registered` for the
rows `bcs` at capacity `cap`, the table's domain id is `dom` and `registered`'s bytes are `Hash.HistoryTable.root` over
`bcs`' `b ‖ c` below `bcs.size` and the placeholder's past it, under `dom` at `cap` and the default key. -/
def HistoryRootIsModel : Prop :=
  ∀ (cap : ℕ) (program table : String) (words : ℕ) (bcs : Array ByteArray) (registered : ByteArray),
    _root_.Spot.HistoryTable.checkAt cap program table words bcs registered = .ok () →
      ∃ dom, _root_.Spot.HistoryTable.domainAt cap program table words = .ok dom ∧
        registered.data.toList =
          (Hash.HistoryTable.root Verity.sha512 (historyDomain cap dom) Flock.Hm96.Default512.key.data.toList
            (Hash.HistoryTable.string bcs.size (fun r => bcs[r]!.data.toList)
              (_root_.Spot.HistoryTable.placeholder words).data.toList)).bytes

end Properties.Spot
