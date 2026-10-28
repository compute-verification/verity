/-!
# What the window makes public (`window.py`, "Public items")

`PUBLIC`, the draft's list and the rows' fixed length, in the service's `Public` form, `REGISTERED`, the items
a registration carries, `LEGACY_REGISTERED`, what one made before the fixed-length rule carries, and `KINDS`, the kinds
of an entry.
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
  ⟨"work", "output", "window", "the credited work, as the tile count padded to a grid level: the draw ranges over its N " ++
    "tiles and the auditor credits its work; one level, so at most log2 q bits"⟩,
  ⟨"check", "structure", "parameter", "the check's identity: the tile template, its work per tile and the commitment " ++
    "scheme fix the auditor's verifier"⟩,
  ⟨"sampling", "structure", "parameter", "ε, δ and the grid, and the law subset:K′ they fix with the level: the guarantee " ++
    "is stated in them"⟩,
  ⟨"length", "structure", "parameter", "the rows of record's fixed length: each slot's max_tokens, served whatever it " ++
    "samples (ignore_eos), so the useful work is public"⟩,
  ⟨"drawn", "coins", "outcome", "the verifier's own draw, after the receipt"⟩,
  ⟨"root_W", "commitment", "parameter", "the weights' salted root: what the tiles' weight rows are committed to, hiding"⟩]

/-- `REGISTERED`: the items a registration carries; the verdict and the draw come after it. -/
def registered : List String := ["work", "check", "sampling", "length", "root_W"]

/-- `LEGACY_REGISTERED`: what a registration made before the fixed-length rule carries, `registered` without `length`,
so not of record. -/
def legacyRegistered : List String := registered.filter (· != "length")

/-- `KINDS`. -/
def kinds : List String := ["call", "filler", "none"]

end PouwWindow
