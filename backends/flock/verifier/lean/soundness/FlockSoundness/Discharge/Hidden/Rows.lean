import FlockSoundness.Discharge.Hidden.Facts
import FlockSoundness.Protocol.Discharge.Hidden.Rows

/-!
# An output row's shape

`HmOut.checkRows` holds output row `j` to one of two shapes (`checkRows_out`): a lone output port's words zero-padded to
whole SHA-512 blocks, all of them bits, or a bit row (`sha512/row/v2` or `sha512/row-seg/v1`) of exactly its run's words.
A `sha512/row/v1` row is the first (`out_rows_v1`), so its output rows are one per output port (`outRows_single`,
`outGroup_single`) and whole blocks (`ParseFacts.portWords`). `HmRow.rowPorts` derives a bit row's words and compressions
from its length in bits (`rowPorts_v2`).
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

/-- **Unsegmented output rows of whole blocks**: when the output rows are one per output port, each a multiple of 64
words, each is after the input rows, of `2 · words = 128 · chunks` bytes (`ParseFacts.portWords`; a `sha512/row/v2`
row's by `rowPorts_v2`) and `16 · words` bits (`checkRows_out`: a padded lone port's, or a bit row's of exactly its
port's words). -/
theorem out_rows_whole {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hp : HmRow.parse tags cf tables = .ok c) (hc : HmOut.checkRows c = .ok ())
    (hu : ∀ p ∈ c.ports.toList, p.segs = none) (hr : c.outRowPorts = Array.replicate c.outPorts.size 1)
    (h64 : ∀ j < c.outPorts.size, (c.ports.getD (HmOut.nIn c + j) default).words % 64 = 0) :
    ∀ j < c.outPorts.size, HmOut.nIn c + j < c.ports.size ∧
      2 * (c.ports.getD (HmOut.nIn c + j) default).words = 128 * (c.ports.getD (HmOut.nIn c + j) default).chunks ∧
      (c.ports.getD (HmOut.nIn c + j) default).bits = 16 * (c.ports.getD (HmOut.nIn c + j) default).words := by
  have P := parse_facts hp
  obtain ⟨⟨mj, typed, hrp⟩, -⟩ := parse_rowPorts hp
  intro j hj
  have hj' : j < c.outRowPorts.size := by rw [hr, Array.size_replicate]; exact hj
  have h6 := h64 j hj
  obtain ⟨hlt, hrow⟩ := checkRows_out hc hj'
  have hmem := getD_mem_toList (d := default) hlt
  generalize c.ports.getD (HmOut.nIn c + j) default = p at hmem hrow h6 ⊢
  have hb : p.bits = 16 * p.words := by
    rcases hrow with ⟨-, -, hb, -⟩ | ⟨hs, hb⟩
    · exact hb
    · have hv : p.v2 = true := by
        rcases hs with hs | hs
        · rw [hu p hmem] at hs; cases hs
        · exact hs
      rw [(rowPorts_v2 hrp p hmem hv).1, hb]
      omega
  refine ⟨hlt, ?_, hb⟩
  cases hv : p.v2
  · exact P.portWords p hmem hv
  · rw [(rowPorts_v2 hrp p hmem hv).2.1 (hu p hmem), hb]
    omega

end FlockSoundness.Discharge.Hidden
