import FlockSoundness.Discharge.Canonical.Format
import FlockSoundness.Discharge.Canonical.Chain
import FlockSoundness.Discharge.Hidden.Wide.RowBc
import FlockSoundness.Discharge.HiddenExec.SiteH

set_option Elab.async false

/-!
# A canonical row's `b ‖ c` at its port

`Hidden.row_bc_tgt` and `HiddenExec.bc_site_commitH` for canonical rows: the `b ‖ c` at row `p`'s `hm96` output is the
commit string of `P(ℓ)` and `packBits` of the `ℓ` bits the witness holds at the row's message columns
(`Typed.msgCol`, bit `i` at byte `i / 8`'s bit `i % 8`), for any `ℓ`. `row_bc_tgt` takes `bits = 16 · words` and
`2 · words = 128 · chunks` (whole blocks); `bc_site_commitH` takes `sha512/row/v1` rows and `outNet = unitNet`.

* `rowBits`: the `ℓ` bits at row `p`'s message columns;
* `row_bc_C`: `row_bc_tgt` at any unsegmented bit row of the flat statement;
* `bc_site_commitC`: `bc_site_commitH` at a canonical circuit's accepted hidden-output statement.

**Proof work.**
* `row_bc_C`: `row_copies_tgt`'s four copies hold unchanged (they are per compression and per input bit,
  `inBit_copyAll`); what changes is the message. Δ pins each message bit `HmRow.msgBit` calls `.const v` (`bitPairs` at
  `msgColOf` for `k ≥ bits / 1024`, and at the `hm96` slot's `padCol`) to the pin or the forced-zero row, which is
  `blockBit_constSrc`, so the block's bit there is `bitL (Chain.rowTail data) _` once `constBit` is read against
  `padTail(128 + B)` (the general `Hidden.constBit_row`, without `hw`: a `getD` split at byte `B`, the fill bits of
  byte `B - 1` zero). Its `.row i` bits are the message columns themselves (`Typed.msgCol` maps byte `b` to compression
  `b / 128` or, past the slots, to the `hm96` slot's `pad`). Then `row_commitC` replaces `port_commit_rowsP`, and
  `bytesOf … B msgCol = packBits (rowBits …)` because the fill bits are the zero (`packBits`'s `bitsVal` byte by byte).
  Moderate: `constBit_row`'s generalization and the last-block split carry the work; the rest is re-plumbing.
* `bc_site_commitC`: `AcceptedH.row_bc`'s route, `Hidden.sat_flat` then `row_bc_C`. `sat_flat` reads the output rows'
  copies (`Hidden.outPairs`) through `out_rows_v1`; at canonical output rows (runs of ports, exact bits) it needs the
  run form of `OutRows`, and `0 < upv` is `Hidden.setupHidden_upv_pos` (no `outNet = unitNet`). Small once
  `row_bc_C` and the run form of `OutRows` (the `Outputs` family's `outRows_runs`) hold.
-/

namespace FlockSoundness.Discharge.Canonical

open Flock FlockLevel3 Model
open FlockSoundness.Discharge.Hm (blockBit bytesOf bcCol saltCol)
open FlockSoundness.Discharge.HiddenExec (AcceptedH siteAddrH)

/-- **The bits at row `p`'s message columns** at VU `g`, under the bits `w`: bit `i < ℓ` is `w` at byte `i / 8`'s
bit `i % 8` (`Typed.msgCol`). -/
def rowBits (w : ℕ → Bool) (c : Circuit) (g p : ℕ) : List Bool :=
  List.ofFn fun i : Fin c.ports[p]!.bits => w (Typed.msgCol c g p (i.val / 8) (i.val % 8))

theorem rowBits_length (w : ℕ → Bool) (c : Circuit) (g p : ℕ) : (rowBits w c g p).length = c.ports[p]!.bits :=
  List.length_ofFn

section Bc

variable {tags : Tags} {st : Stmt} (L : FlockSoundness.Layout st) (F : Aliased.UnitFacts tags st)
include L F

/-- **`Hidden.row_bc_tgt` at any bit row**: in every block of a satisfying witness of the untyped (flat) statement, at
an unsegmented `sha512/row/v2` row `p` of any `ℓ` bits, the `b ‖ c` at its `hm96` output is the commit string of `P(ℓ)`
and `packBits` of the bits at its message columns, under the salt its `hm96` slot reads.

*Route*: the module docstring's. -/
theorem row_bc_C (hda : st.da = (HmRow.delta st.c st.pin).1) (hupv : 0 < st.c.upv)
    (hu : ∀ p ∈ st.c.ports.toList, p.segs = none)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) (hv : st.c.ports[p]!.v2 = true) :
    bytesOf (blockBit (L.model regs) z o) 128 (bcCol st.c g p) = Binding.commitString
      ((HmRow.rowPrefixV2 st.c.ports[p]!.bits).data.toList ++
        HmRow.packBits (rowBits (blockBit (L.model regs) z o) st.c g p))
      (bytesOf (blockBit (L.model regs) z o) 192 (saltCol st.c g p)) := by
  sorry

end Bc

variable {T : Model.Statement}

/-- **A canonical row's `b ‖ c` at a drawn unit's site** (`bc_site_commitH` for canonical circuits): in a satisfying
witness of an accepted hidden-output statement on a canonical circuit, at block `o`, VU `g` and row `p`, input or
output, the bytes at the address the regions pin (`Layout.bcAddr`) are the commit string of `P(ℓ_p)` and `packBits` of
the `ℓ_p` bits at the row's message columns, under the salt at `saltCol`. No condition on the out net.

*Route*: the module docstring's. -/
theorem bc_site_commitC (a : AcceptedH T) (hfmt : formatOk a.st.c = true) {z : Witness T.m} (hz : T.Satisfies z)
    (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g) (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddrH a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      Binding.commitString ((HmRow.rowPrefixV2 a.st.c.ports[p]!.bits).data.toList ++
        HmRow.packBits (rowBits (fun x => FlockSoundness.bit (z (siteAddrH a o x))) a.st.c g p))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => siteAddrH a o (saltCol a.st.c g p j.val t.val)) := by
  sorry

end FlockSoundness.Discharge.Canonical
