import FlockSoundness.Discharge.Canonical.Format
import FlockSoundness.Discharge.Layout.KeysRow

set_option Elab.async false

/-!
# Every unit input a computed row reads is a row bit or the zero, at any number of input rows and groups

`Layout.keysRow_of_parse` gives `KeysRow` at the row widths `16 · words` under `inGroups.size = 1` (the unit reads its
leaf group only) and `PortReads`. A canonical circuit may read any number of input rows, through its leaves (group 0) or
its leaf cuts (another group's port read straight from a row word, `HmRow.leafCuts`), and its rows are bit rows of
`ℓ_q` bits. `keysRowC_of_parse` is `KeysRow` at the canonical widths `rowW c q = ℓ_q`, from `formatOk`.

Δ's key of a read input is `Aliased.srcAt` at its port and bit (`keysRow_of_parse`'s first half, unchanged). `srcAt`
gives `.zero`, `.msg q (2 (x - start_q) + t / 8) (t % 8)` exactly when `16 (x - start_q) + t < ℓ_q` (a leaf's bit, or a
leaf cut's with `t < 16`), `.wire` or `.lit`; `formatOk`'s `readsRows` excludes the last two. The message key's bit
`8 · byte + bit = 16 (x - start_q) + t` is below `ℓ_q`, so it is a row bit at the canonical width. Nothing about the
key depends on how many groups the unit has.

`HiddenExec.copy_siteH` (a message-keyed input holds its row's bit at `Typed.msgCol`) already holds for any rows.

**Proof work.** Small: `keysRow_of_parse`'s proof with `Aliased.srcAt_leaf` replaced by a three-way case on `srcAt`
(leaf, leaf cut, wire or literal) and the width bound read from `srcAt`'s own `if` (`16 (x - start) + t < bits`);
`q < ports.size` is `Layout.starts_find` as today.
-/

namespace FlockSoundness.Discharge.Canonical

open Flock FlockLevel3

/-- **`KeysRow` at the canonical widths, from the parse**: in a canonical circuit `HmRow.parse` accepts, every input a
computed row reads has the zero or a row bit below its row's `ℓ_q` bits as its key, for any number of input rows and
groups.

*Route*: the module docstring's. -/
theorem keysRowC_of_parse {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hU : Built c.unit) (h : HmRow.parse tags bytes tables = .ok c) (hfmt : formatOk c = true) :
    Layout.KeysRow c hU (rowW c) := by
  sorry

end FlockSoundness.Discharge.Canonical
