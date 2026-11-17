module

public import Spot.HistoryTable
public import Definitions.Hash.HistoryTable
public import Properties.Verity.Sha512

/-! A history table's registered root, which `Spot.HistoryTable.checkAt` accepts, is the hash area's model root
(`HistoryRootIsModel`). -/

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
