import Flock.RecLinkTree

/-!
# `RecLink_v1{UB, PB}`'s reference (`verity_flock.rec_link.link_ref`)

What one instance of V*'s link check computes, on its ports' u16 words: the reader's row leaf
`leaf(commitString(xr, yr))` climbed by `raux`'s siblings to the position the table row names (port `PB` bits, unit `UB`
bits above), the writer's likewise by `waux`, each root's 32 words then 32 zero words, or all zero words when the row is
not well formed or, a link, its two digests differ (`okRef`).

`linkOut_meaning`: an output that is not all zero is the climb's root, and the row passed `okRef`. `okRef_link`: a link row
that passes has equal reader and writer digests; `okRef_filler`: a filler that passes names no row but leaf 0's; and
`okRef_fits`: every field of a row that passes fits its bits, so its position is the one it names. These are the unit's
outputs' meaning at its reference. The unit's netlist (`VStar.linkUnit`) is held to `rec_link.unit_circuit` by the
`link_unit` difftest, and `link_ref` to this reference by `link_ref`; that the netlist computes this reference is not
proved here.
-/

namespace Flock.RecLinkRef

open RecLinkTree RecLinks

def NODE_WORDS : Nat := 32
def OUT_WORDS : Nat := 64

/-- `rec_open.bytes_of`: u16 words, little-endian. -/
def bytesOf (ws : List Nat) : ByteArray := ⟨(ws.flatMap fun w => [(w % 256).toUInt8, (w / 256 % 256).toUInt8]).toArray⟩

/-- `rec_open.words_of`: bytes as u16 words, little-endian. -/
def wordsOf (b : ByteArray) : List Nat :=
  (List.range (b.size / 2)).map fun i => (b.get! (2 * i)).toNat + 256 * (b.get! (2 * i + 1)).toNat

def zeros : List Nat := List.replicate OUT_WORDS 0

/-- `rec_link.expected`: a root as a registered row, its words then zero words. -/
def expected (root : ByteArray) : List Nat := wordsOf root ++ List.replicate (OUT_WORDS - NODE_WORDS) 0

/-- `rec_link.aux_words`. -/
def auxWords (D : Nat) : Nat := max 1 ((NODE_WORDS * D + 63) / 64) * 64

/-- `rec_link.port_words`: `trow`, `xr`, `yr`, `raux`, `xw`, `yw`, `waux`. -/
def portWords (UB PB : Nat) : List Nat := let a := auxWords (UB + PB); [64, 32, 96, a, 32, 96, a]

/-- Field `k` of a table row: its `k`-th u32, little-endian. -/
def field (row : ByteArray) (k : Nat) : Nat := u32At row (4 * k)

def tailZero (row : ByteArray) : Bool := (List.range (row.size - FIELD_BYTES)).all fun i => row.get! (FIELD_BYTES + i) == 0

/-- The row's shape: kind 0 or 1, zero past the fields, units in `UB` bits and the port and row in `PB`. -/
def okShape (UB PB : Nat) (row : ByteArray) : Bool :=
  (field row 0 == 0 || field row 0 == KIND_LINK) && tailZero row && field row 1 < 2 ^ UB && field row 3 < 2 ^ UB &&
    field row 2 < 2 ^ PB && field row 4 < 2 ^ PB

/-- `rec_link.ok_ref`: the row's shape, and a filler's fields zero or a link's two digests equal. -/
def okRef (UB PB : Nat) (row xr xw : ByteArray) : Bool :=
  okShape UB PB row &&
    (if field row 0 == 0 then field row 1 == 0 && (field row 2 == 0 && (field row 3 == 0 && field row 4 == 0))
     else xr.data == xw.data)

/-- `rec_link._climb_ref`. -/
def climbRef (start : ByteArray) (sibs : List ByteArray) (pos : Nat) : ByteArray :=
  (List.range sibs.length).foldl (fun cur h =>
    let s := sibs.getD h EMPTY
    if (pos >>> h) % 2 == 1 then node s cur else node cur s) start

/-- An opening's `D` siblings, bottom up: 32 words each. -/
def sibsOf (aux : List Nat) (D : Nat) : List ByteArray :=
  (List.range D).map fun h => bytesOf ((aux.drop (NODE_WORDS * h)).take NODE_WORDS)

/-- The position the row's fields `unit` and `row` name, its low `D` bits: `unit · 2^PB + row`, which is `rec_link`'s
`row | unit << PB` wherever the row passes `okRef` (`row < 2^PB`), and where it doesn't the output is zero either way. -/
def posOf (trow : ByteArray) (unit row PB D : Nat) : Nat := (field trow unit * 2 ^ PB + field trow row) % 2 ^ D

/-- A row's leaf from its digest's and salt's words. -/
def leafOf (x y : List Nat) : ByteArray := leaf (Hm96.Default512.commitString (bytesOf x) (bytesOf y))

/-- The reader's climb: its leaf by `raux` to the position of fields 1 (unit) and 2 (port). -/
def rootR (UB PB : Nat) (trow xr yr raux : List Nat) : ByteArray :=
  climbRef (leafOf xr yr) (sibsOf raux (UB + PB)) (posOf (bytesOf trow) 1 2 PB (UB + PB))

/-- The writer's climb: its leaf by `waux` to the position of fields 3 (unit) and 4 (row). -/
def rootW (UB PB : Nat) (trow xw yw waux : List Nat) : ByteArray :=
  climbRef (leafOf xw yw) (sibsOf waux (UB + PB)) (posOf (bytesOf trow) 3 4 PB (UB + PB))

/-- `rec_link.link_ref` once its ports' sizes hold: `(rin, rout)`. -/
def linkOut (UB PB : Nat) (trow xr yr raux xw yw waux : List Nat) : List Nat × List Nat :=
  if okRef UB PB (bytesOf trow) (bytesOf xr) (bytesOf xw) then
    (expected (rootR UB PB trow xr yr raux), expected (rootW UB PB trow xw yw waux))
  else (zeros, zeros)

/-- `rec_link.link_ref`: the shape and the ports' sizes checked, then `linkOut`. -/
def linkRef (UB PB : Nat) (trow xr yr raux xw yw waux : List Nat) : Except String (List Nat × List Nat) := do
  checkShape UB PB
  let ports := [trow, xr, yr, raux, xw, yw, waux]
  let got := ports.map (·.length)
  unless got == portWords UB PB do throw s!"RecLink at UB {UB}, PB {PB} takes {portWords UB PB} words a port, not {got}"
  unless ports.all (·.all (· < 65536)) do throw "a port's words are u16s"
  pure (linkOut UB PB trow xr yr raux xw yw waux)

/-- **An output that is not all zero is a climb's root**: the row passed `okRef`, `rin` is the reader's climb and `rout`
the writer's, each as a registered row. -/
theorem linkOut_meaning {UB PB : Nat} {trow xr yr raux xw yw waux : List Nat}
    (hne : (linkOut UB PB trow xr yr raux xw yw waux).1 ≠ zeros ∨ (linkOut UB PB trow xr yr raux xw yw waux).2 ≠ zeros) :
    okRef UB PB (bytesOf trow) (bytesOf xr) (bytesOf xw) = true ∧
      linkOut UB PB trow xr yr raux xw yw waux =
        (expected (rootR UB PB trow xr yr raux), expected (rootW UB PB trow xw yw waux)) := by
  unfold linkOut at hne ⊢
  by_cases hok : okRef UB PB (bytesOf trow) (bytesOf xr) (bytesOf xw) = true
  · simp [hok]
  · simp [hok] at hne

theorem okRef_shape {UB PB : Nat} {row xr xw : ByteArray} (h : okRef UB PB row xr xw = true) : okShape UB PB row = true := by
  simp only [okRef, Bool.and_eq_true] at h
  exact h.1

/-- **A link row that passes joins equal digests**: the reader's and the writer's rows commit to one digest. -/
theorem okRef_link {UB PB : Nat} {row xr xw : ByteArray} (h : okRef UB PB row xr xw = true)
    (hk : field row 0 = KIND_LINK) : xr = xw := by
  simp only [okRef, Bool.and_eq_true] at h
  have h2 := h.2
  rw [hk] at h2
  have : xr.data = xw.data := by simpa [KIND_LINK] using h2
  exact congrArg ByteArray.mk this

/-- **A filler that passes names leaf 0 of each tree.** -/
theorem okRef_filler {UB PB : Nat} {row xr xw : ByteArray} (h : okRef UB PB row xr xw = true) (hk : field row 0 = 0) :
    field row 1 = 0 ∧ field row 2 = 0 ∧ field row 3 = 0 ∧ field row 4 = 0 := by
  simp only [okRef, Bool.and_eq_true] at h
  have h2 := h.2
  rw [hk] at h2
  simpa using h2

/-- **Every field of a row that passes fits its bits**, and its kind is a filler's or a link's. -/
theorem okRef_fits {UB PB : Nat} {row xr xw : ByteArray} (h : okRef UB PB row xr xw = true) :
    (field row 0 = 0 ∨ field row 0 = KIND_LINK) ∧ field row 1 < 2 ^ UB ∧ field row 3 < 2 ^ UB ∧ field row 2 < 2 ^ PB ∧
      field row 4 < 2 ^ PB := by
  have hs := okRef_shape h
  simp only [okShape, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at hs
  obtain ⟨⟨⟨⟨⟨hk, -⟩, h1⟩, h3⟩, h2⟩, h4⟩ := hs
  exact ⟨hk, h1, h3, h2, h4⟩

/-- **The position a passing row names**: its row (or port) in the low `PB` bits, its unit above, `index`'s. -/
theorem posOf_fits {UB PB : Nat} {trow : ByteArray} {unit row : Nat} (hu : field trow unit < 2 ^ UB)
    (hr : field trow row < 2 ^ PB) : posOf trow unit row PB (UB + PB) = index (field trow unit) (field trow row) PB := by
  unfold posOf index
  apply Nat.mod_eq_of_lt
  rw [Nat.pow_add]
  calc field trow unit * 2 ^ PB + field trow row < field trow unit * 2 ^ PB + 2 ^ PB := Nat.add_lt_add_left hr _
    _ = (field trow unit + 1) * 2 ^ PB := (Nat.succ_mul _ _).symm
    _ ≤ 2 ^ UB * 2 ^ PB := Nat.mul_le_mul_right _ hu

end Flock.RecLinkRef
