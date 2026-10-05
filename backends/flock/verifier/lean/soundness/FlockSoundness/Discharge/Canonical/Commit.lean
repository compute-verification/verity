import FlockSoundness.Discharge.Canonical.Format
import FlockSoundness.Discharge.Canonical.Chain
import FlockSoundness.Discharge.Canonical.OutCopy
import FlockSoundness.Discharge.Hidden.Wide.RowBc
import FlockSoundness.Discharge.Hidden.Wide.Upv
import FlockSoundness.Discharge.HiddenExec.SiteH
import FlockSoundness.Protocol.Discharge.Canonical.Commit
import FlockSoundness.Discharge.Hidden.Theorems
import FlockSoundness.Discharge.Hidden.Wide.Shared
import FlockSoundness.Discharge.HiddenExec.Ret
import FlockSoundness.Discharge.HiddenExec.Statements

set_option Elab.async false

/-!
# A canonical row's `b ‖ c` at its port

`Hidden.row_bc_tgt` and `HiddenExec.bc_site_commitH` for canonical rows: the `b ‖ c` at row `p`'s `hm96` output is the
commit string of `P(ℓ)` and `packBits` of the `ℓ` bits the witness holds at the row's message columns
(`Typed.msgCol`, bit `i` at byte `i / 8`'s bit `i % 8`), for any `ℓ`. `row_bc_tgt` takes `bits = 16 · words` and
`2 · words = 128 · chunks` (whole blocks); `bc_site_commitH` takes `sha512/row/v1` rows and `outNet = unitNet`.

* `rowBits`: the `ℓ` bits at row `p`'s message columns;
* `row_bc_C`: `row_bc_tgt` at any unsegmented bit row of the flat statement whose compressions are the ones its length
  needs (`chunks + 1 = ⌈(⌈ℓ / 8⌉ + 17) / 128⌉`, `HmRow.rowPorts`' count, `v2_of_parse`);
* `bc_site_commitC`: `bc_site_commitH` at a canonical circuit's accepted hidden-output statement.

`row_bc_C`: `row_copies_tgt`'s copies into the chain hold unchanged (`inBit_copyAll`); a message bit `HmRow.msgBit`
calls `.const v` (`msg_filterConstC`, `pad_filterAll`) holds `v`, which is `Chain.rowTail`'s bit there
(`constBit_rowC`: the fill bits of the last data byte are `packBits`' zeros, the rest `padTail(128 + B)`), and its
`.row i` bits are the message columns themselves (`msgCol_eq`, `msgCol_pad`), so `Chain.row_commitC` hashes
`P(ℓ) ‖ packBits (rowBits …)` (`port_commit_rowsC`). `bc_site_commitC`: the hidden statement's satisfying witness
satisfies its flat twin (`OutCopy`'s `sat_flatC`, at `outRowsC_of`'s output rows), then `row_bc_C`.
-/

namespace FlockSoundness.Discharge.Canonical

open Flock FlockLevel3 Model
open Flock.HmNets (wordBit)
open FlockSoundness.Discharge.Hm (blockBit bytesOf bcCol saltCol)
open FlockSoundness.Discharge.HiddenExec (AcceptedH siteAddrH)

/-! ## A bit row's bytes and its message after the prefix -/

open Hm in
/-- Block `k`'s bit `wordBit (8 o + t)` is a row bit when its index `8 (128 k + o) + t` is below the row's bits. -/
theorem constBit_none (p : Port) (tl : ByteArray) (k : ℕ) {o t : ℕ} (ho : o < 128) (ht : t < 8)
    (h : 8 * (128 * k + o) + t < p.bits) : HmRow.constBit p tl k (wordBit (8 * o + t)) = none := by
  simp only [HmRow.constBit, blockBit_wordBit ho ht, h, ↓reduceIte]

/-! ## Δ's copies at a bit row's message -/

section Msg

open Hm Hidden

variable {c : Circuit} (R : RowRanges c) (hu : ∀ p ∈ c.ports.toList, p.segs = none)
include R hu

end Msg

/-! ## The port -/

section Port

open Flock.HmNets hiding and
open Hm Hidden

end Port

/-! ## The flat statement -/

section Bc

open Hm Hidden

variable {tags : Tags} {st : Stmt} (L : FlockSoundness.Layout st) (F : Aliased.UnitFacts tags st)
include L F

end Bc

variable {T : Model.Statement}

end FlockSoundness.Discharge.Canonical
