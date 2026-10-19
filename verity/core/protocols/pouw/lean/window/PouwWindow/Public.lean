/-!
# What the window makes public (`window.py`, "Public items")

`PUBLIC`, exactly the draft's list, the window's shape and its fixed length, in the service's `Public` form,
`REGISTERED`, the items a registration carries, and `KINDS`, the kinds of an entry.
-/

namespace PouwWindow

/-- One public item: its name, kind, source and the reason it is public. -/
structure PublicItem where
  item : String
  kind : String
  source : String
  reason : String
  deriving DecidableEq, Repr

/-- `PUBLIC`. -/
def publicItems : List PublicItem := [
  ⟨"verdict", "output", "outcome", "what the credit rests on: every drawn tile was proved and opened, or not"⟩,
  ⟨"work", "output", "window", "the credited work, the masked sum of credited tiles rounded down to a grid level, of the " ++
    "shape's N tiles the draw ranges over; one level, so at most log2 q bits"⟩,
  ⟨"check", "structure", "parameter", "the check's identity: the tile template, its work per tile and the commitment " ++
    "scheme fix the auditor's verifier"⟩,
  ⟨"sampling", "structure", "parameter", "ε, δ and the grid, and the law subset:K′ they fix with the level: the guarantee " ++
    "is stated in them"⟩,
  ⟨"shape", "structure", "parameter", "the window's calls, each its batch slots, weight rows and first tile in the pass: " ++
    "they fix N, every entry's range and rows, and the tile each index is"⟩,
  ⟨"length", "structure", "parameter", "each call's fixed length: its requests ran at max_tokens L ignoring " ++
    "end-of-sequence, so the window's useful work is public and done masks nothing"⟩,
  ⟨"drawn", "coins", "outcome", "the verifier's own draw, after the receipt"⟩,
  ⟨"root_W", "commitment", "parameter", "the weights' salted root: what the tiles' weight rows are committed to, hiding"⟩]

/-- `REGISTERED`: the items a registration carries; the verdict and the draw come after it. -/
def registered : List String := ["work", "check", "sampling", "shape", "length", "root_W"]

/-- `KINDS`. -/
def kinds : List String := ["call", "filler", "none"]

end PouwWindow
