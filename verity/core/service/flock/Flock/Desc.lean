import Flock.HmRow

/-!
# A circuit's descriptor

A circuit file describes its circuit, and the verifier's statements read the circuit through its descriptor: `Desc.parse`
or `Desc.parseStmt` reads the file as a descriptor `D`, and `D.unroll` is the circuit it describes. The circuit's pin is
SHA-512 of the file, so it is SHA-512 of `D`'s bytes, whatever form `D` takes.

A descriptor takes one of two forms:

- a flat file, `single c`: `HmOut.parse` reads it, and it unrolls to `c` itself (`parse_flat`, `parse_single`);
- a templated statement (`Tags.isTmpl`), `tmpl c`: `HmOut.parseTmpl` reads it as the committed circuit `c`, which carries
  its unit's template, and it unrolls to the flat circuit that template unrolls to, `HmOut.unrolled c`. Its size is its
  template's and its shared rows', not the unrolled unit's.

`Desc.parse` reads flat files alone; `Desc.parseStmt` reads both forms, and agrees with `HmOut.parseStmt` on the circuit
(`parseStmt_stmt`) and with `Desc.parse` on every flat file (`parseStmt_of_parse`).
-/

namespace Flock

/-- A circuit descriptor: a flat circuit file, the circuit itself, or a templated statement, the committed circuit that
carries its unit's template. -/
inductive Desc where
  | single (c : Circuit)
  | tmpl (c : Circuit)

namespace Desc

/-- The circuit `D` describes. -/
def unroll : Desc → Circuit
  | single c => c
  | tmpl c => HmOut.unrolled c

/-- Read the circuit file `bytes` as a descriptor: a flat file, as `HmOut.parse` reads it. -/
def parse (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray) := []) : V Desc :=
  match HmOut.parse tags bytes tables with
  | .ok c => .ok (.single c)
  | .error e => .error e

/-- A descriptor `Desc.parse` reads unrolls to the circuit `HmOut.parse` reads from the same file. -/
theorem parse_flat {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {D : Desc}
    (h : parse tags bytes tables = .ok D) : HmOut.parse tags bytes tables = .ok D.unroll := by
  unfold parse at h
  split at h
  · next c hc => cases h; exact hc
  · cases h

/-- A file `HmOut.parse` reads as `c` is the flat descriptor `single c`. -/
theorem parse_single {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (h : HmOut.parse tags bytes tables = .ok c) : parse tags bytes tables = .ok (.single c) := by
  unfold parse
  rw [h]

/-- Read the statement file `bytes` as a descriptor: a templated statement (`Tags.isTmpl`) as `HmOut.parseTmpl` reads it,
any other as `Desc.parse` reads it. -/
def parseStmt (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray) := []) : V Desc :=
  if tags.isTmpl then
    match HmOut.parseTmpl tags bytes tables with
    | .ok c => .ok (.tmpl c)
    | .error e => .error e
  else parse tags bytes tables

/-- A descriptor `Desc.parseStmt` reads unrolls to the statement circuit `HmOut.parseStmt` reads from the same file. -/
theorem parseStmt_stmt {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {D : Desc}
    (h : parseStmt tags bytes tables = .ok D) : HmOut.parseStmt tags bytes tables = .ok D.unroll := by
  unfold parseStmt at h
  unfold HmOut.parseStmt
  cases ht : tags.isTmpl <;> simp only [ht, Bool.false_eq_true, ↓reduceIte] at h ⊢
  · exact parse_flat h
  · split at h
    · next c hc => cases h; simp only [hc, unroll]; rfl
    · cases h

/-- A file `Desc.parse` reads as `D`, `Desc.parseStmt` reads as `D` too. -/
theorem parseStmt_of_parse {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {D : Desc}
    (h : parse tags bytes tables = .ok D) : parseStmt tags bytes tables = .ok D := by
  have ht : tags.isTmpl = false := by
    unfold parse at h
    split at h
    · next c hc => exact HmOut.parse_isTmpl hc
    · cases h
  unfold parseStmt
  simp only [ht, Bool.false_eq_true, ↓reduceIte]
  exact h

/-- A descriptor `Desc.parseStmt` reads holds the circuit `HmOut.parseCommit` reads from the same file: flat, or the
committed templated statement. -/
theorem parseStmt_commit {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {D : Desc}
    (h : parseStmt tags bytes tables = .ok D) :
    ∃ c, HmOut.parseCommit tags bytes tables = .ok c ∧ (D = .single c ∨ D = .tmpl c) := by
  unfold parseStmt at h
  unfold HmOut.parseCommit
  cases ht : tags.isTmpl <;> simp only [ht, Bool.false_eq_true, ↓reduceIte] at h ⊢
  · unfold parse at h
    split at h
    · next c hc => cases h; exact ⟨c, hc, .inl rfl⟩
    · cases h
  · split at h
    · next c hc => cases h; exact ⟨c, hc, .inr rfl⟩
    · cases h

end Desc

end Flock
