/-
A row's `b ‖ c` in a satisfying witness of the accepted template statement. `HmRow.rowSlots` puts port `p`'s compression
`k` at `sha512x3` slot `(at_p + k) / 3`, place `(at_p + k) % 3`, with `at_p` the earlier ports' chunks (`rowSlots_eq`).
So the satisfying witness holds the `sha512x3` and `hm96` rows at every compression of the row (`shaSlot_of_sat`,
`hmSlot_of_sat`), and `port_commit_rows` gives the row's commit string at its `hm96` slot's outputs (`row_bc_of_sat`),
given the port's words fill its chunks and the block holds Δ's copies into the slots' `cv` and `pad` inputs.
-/
import FlockSoundness.Discharge.Hm.Place
import FlockSoundness.ExecTemplateSetup
import FlockSoundness.Protocol.Discharge.Hm.Accept

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hm

open Compose Flock.DeriveAll Types.Dag

section Slots

variable {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D} {st : Stmt} {tb : TypedBlock}

set_option hygiene false in
local notation "𝒖" => done.getD unit default

/-- **A row's `b ‖ c` in a satisfying witness of the accepted template statement.** At VU `g`, port `p`'s row: if it is
not segmented (`hs`) and its words fill its chunks, at least one (`hn`, `hw`) and the block holds Δ's copies into the slots' inputs (the first
compression's `cv` the row prefix's midstate, each later `cv` the previous compression's output, the `hm96` slot's `cv`
the last output and its `pad` the padding block), then the 128 bytes the block holds at `bcCol` are the commit string of
the `2 · words` row bytes at `Typed.msgCol` and the 192 salt bytes at `saltCol`. -/
theorem row_bc_of_sat {tags : Tags} {t : Typed.Template} (hd : deriveChecked types ls info words unit = .ok done)
    (hul : unit < ls.size) (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖)
    (P : ParseFactsT tags t st.c) (C : CheckFactsT st.c tb)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (T.model regs).m}
    (hz : (T.model regs).Satisfies z) (o : Fin (2 ^ (T.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) (hs : st.c.ports[p]!.segs = none) (hn : 0 < st.c.ports[p]!.chunks)
    (hw : 2 * st.c.ports[p]!.words = 128 * st.c.ports[p]!.chunks)
    (hcv0 : ∀ t < 512, blockBit (T.model regs) z o (shaBase st.c g p 0 + shaIn st.c (2 * shaSub st.c p 0) + t) =
      wordsBit (HmRow.midstate st.c.ports[p]!.words) t)
    (hchain : ∀ k, k + 1 < st.c.ports[p]!.chunks → ∀ t < 512,
      blockBit (T.model regs) z o (shaBase st.c g p (k + 1) + shaIn st.c (2 * shaSub st.c p (k + 1)) + t) =
        blockBit (T.model regs) z o (shaBase st.c g p k + shaOut st.c (shaSub st.c p k) + t))
    (hcvh : ∀ t < 512, blockBit (T.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 0 + t) =
      blockBit (T.model regs) z o
        (shaBase st.c g p (st.c.ports[p]!.chunks - 1) + shaOut st.c (shaSub st.c p (st.c.ports[p]!.chunks - 1)) + t))
    (hpad : ∀ t < 1024, blockBit (T.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 1 + t) =
      wordsBit (HmRow.padWords st.c.ports[p]!.words) t) :
    bytesOf (blockBit (T.model regs) z o) 128 (bcCol st.c g p) = Binding.commitString
      ((HmRow.rowPrefix st.c.ports[p]!.words).data.toList ++
        bytesOf (blockBit (T.model regs) z o) (2 * st.c.ports[p]!.words) (Typed.msgCol st.c g p))
      (bytesOf (blockBit (T.model regs) z o) 192 (saltCol st.c g p)) := by
  refine port_commit_rows st.c _ g p hs hn hw (rowSlots_size hp).ge P.rowNets (fun k hk => (rowSlot_lt hp hk).2)
    (fun k hk => ?_) ?_ hcv0 hchain hcvh hpad
  · exact shaSlot_of_sat hd hul T U hz o P.rowNets hg (by rw [C.perVu_sha]; exact (rowSlot_lt hp hk).1)
  · exact hmSlot_of_sat hd hul T U hz o P.rowNets hg (by rw [C.perVu_hm]; exact lt_of_lt_of_le hp (le_nextPow2 _))

end Slots

end FlockSoundness.Discharge.Hm
