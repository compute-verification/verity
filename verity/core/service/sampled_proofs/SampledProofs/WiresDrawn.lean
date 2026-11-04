import SampledProofs.WiresSound

/-!
The wire check on the drawn units alone: a reader unit is faulty in the wired flattening when its outputs are not its
Definition's on its inputs, its wired rows being its writers' committed output rows. So a wired row that is not its
writer's row makes its reader faulty, and drawing that reader catches it: the verifier holds the wired rows of the units it
draws, never every unit's (`Summary.drawnOk`).

`Summary.rowsAt` gives a drawn unit's wired rows and their writers from the wires alone, in work proportional to the wires
and the unit's rows (`Wire.rowsAt`). `drawnOk_iff`: the drawn check accepts exactly when every reader row the expanded
wires cover (`checkRef`'s entries) whose unit is drawn is its writer's row, `same` being the comparison of two rows' public
strings (`b ‖ c`) the leaf layer binds.
-/

namespace SampledProofs.Wires

/-- The rows of reader unit `u` of template `tid` that wire `w` covers, `((port, row), writer)` each, `writer` the
`(template, unit, out)` whose output row it is. -/
def Wire.rowsAt (w : Wire) (tid : String) (u : Nat) : List ((String × Nat) × (String × Int × String)) :=
  if w.reader == tid && w.start ≤ u && u < w.start + w.readerUnits then
    (List.range w.cuts).map fun j => ((w.port, w.cut0 + j), (w.writer, w.writerAt ((u - w.start) * w.cuts + j), w.out))
  else []

/-- Reader unit `u`'s wired rows, by the wires alone. -/
def Summary.rowsAt (s : Summary) (tid : String) (u : Nat) : List ((String × Nat) × (String × Int × String)) :=
  s.wires.flatMap (·.rowsAt tid u)

/-- **The drawn check**: every wired row of every drawn unit `(template, unit)` is its writer's row (`same`). -/
def Summary.drawnOk (s : Summary) (same : (String × Nat × String × Nat) → (String × Int × String) → Bool)
    (drawn : List (String × Nat)) : Bool :=
  drawn.all fun (tid, u) => (s.rowsAt tid u).all fun ((p, r), wr) => same (tid, u, p, r) wr

theorem mem_rowsAt {w : Wire} (hc : 1 ≤ w.cuts) (hd : w.units % w.cuts = 0) {tid : String} {u : Nat} {p : String}
    {r : Nat} {wr : String × Int × String} :
    ((p, r), wr) ∈ w.rowsAt tid u ↔ ((tid, u, p, r), wr) ∈ w.entries := by
  have h5 : w.units / w.cuts * w.cuts = w.units := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hd)
  unfold Wire.rowsAt Wire.entries Wire.readerAt
  constructor
  · intro h
    split at h
    · rename_i hcnd
      simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hcnd
      obtain ⟨⟨rfl, h1⟩, h2⟩ := hcnd
      simp only [List.mem_map, List.mem_range, Prod.mk.injEq] at h
      obtain ⟨j, hj, ⟨rfl, rfl⟩, rfl⟩ := h
      have e1 : (j + (u - w.start) * w.cuts) / w.cuts = u - w.start := by
        rw [Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hj, Nat.zero_add]
      have e2 : (j + (u - w.start) * w.cuts) % w.cuts = j := by
        rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj]
      rw [Nat.add_comm] at e1 e2
      have hk : (u - w.start + 1) * w.cuts ≤ w.readerUnits * w.cuts := Nat.mul_le_mul_right _ (by omega)
      unfold Wire.readerUnits at hk
      rw [h5, Nat.succ_mul] at hk
      simp only [List.mem_map, List.mem_range]
      refine ⟨(u - w.start) * w.cuts + j, by omega, ?_⟩
      rw [e1, e2, Nat.add_sub_of_le h1]
    · simp at h
  · intro h
    simp only [List.mem_map, List.mem_range, Prod.mk.injEq] at h
    obtain ⟨i, hi, ⟨rfl, rfl, rfl, rfl⟩, rfl⟩ := h
    have hu := readerUnit_lt hc hd hi
    have hm := Nat.mod_lt i (by omega : 0 < w.cuts)
    have hdm := Nat.div_add_mod' i w.cuts
    have hcnd : (w.reader == w.reader && decide (w.start ≤ w.start + i / w.cuts) &&
        decide (w.start + i / w.cuts < w.start + w.readerUnits)) = true := by
      simp [hu]
    simp only [hcnd, ↓reduceIte, Nat.add_sub_cancel_left, List.mem_map, List.mem_range]
    exact ⟨i % w.cuts, hm, by rw [hdm]⟩

/-- **The drawn check is the reference's on the drawn units**: it accepts exactly when every reader row the expanded
wires cover whose unit is drawn is its writer's row. -/
theorem drawnOk_iff {c : Wiring} {S : Summary} (h : check c = .ok S)
    (same : (String × Nat × String × Nat) → (String × Int × String) → Bool) (drawn : List (String × Nat)) :
    S.drawnOk same drawn = true ↔
      ∀ e ∈ c.wires.flatMap Wire.entries, (e.1.1, e.1.2.1) ∈ drawn → same e.1 e.2 = true := by
  have hw := check_wires h
  rw [check_summary h]
  simp only [Summary.drawnOk, Summary.rowsAt, List.all_eq_true, List.mem_flatMap]
  constructor
  · rintro hall ⟨⟨tid, u, p, r⟩, wr⟩ ⟨w, hwm, he⟩ hdr
    exact hall (tid, u) hdr ((p, r), wr) ⟨w, hwm, (mem_rowsAt (hw w hwm).1 (hw w hwm).2).2 he⟩
  · rintro hall ⟨tid, u⟩ hdr ⟨⟨p, r⟩, wr⟩ ⟨w, hwm, hx⟩
    exact hall ((tid, u, p, r), wr) ⟨w, hwm, (mem_rowsAt (hw w hwm).1 (hw w hwm).2).1 hx⟩ hdr

/-! ## JSON: the `wires_drawn` and `wires_drawn_unrolled` pieces

An instance is `{"templates", "cuts", "wires", "drawn", "reads", "writes"}`: the first three as the `wires` piece's;
`drawn` the drawn units `[tid, unit]` (repeats counted once); `reads` the public strings of reader rows,
`[tid, unit, port, row, s]`; `writes` those of writer output rows, `[tid, unit, out, s]` (the first of a repeated row
counts). A wired row is its writer's when both strings are given and equal. The output is the `wires` piece's refusal,
or `{"ok": true, "accepted", "checked", "bad"}`: the drawn units' wired rows checked, and the ones that are not their
writer's, sorted. -/

open Lean (Json toJson)

abbrev Key := String × Nat × String × Nat

def keyLt (a b : Key) : Bool :=
  a.1 < b.1 || (a.1 == b.1 && (a.2.1 < b.2.1 || (a.2.1 == b.2.1 && (a.2.2.1 < b.2.2.1 ||
    (a.2.2.1 == b.2.2.1 && a.2.2.2 < b.2.2.2)))))

def keyJson (k : Key) : Json := Json.arr #[toJson k.1, toJson k.2.1, toJson k.2.2.1, toJson k.2.2.2]

def drawnInstance (j : Json) : Except String (Wiring × List (String × Nat) ×
    (Key → (String × Int × String) → Bool)) := do
  sameKeys j ["templates", "cuts", "wires", "drawn", "reads", "writes"] "an instance"
  let c ← Wiring.ofJson j
  let drawn ← (← arrOf "drawn" (← field j "drawn")).mapM fun d => do
    match ← arrOf "a drawn unit" d with
    | [a, b] => pure (← strOf "a drawn unit's template" a, ← natOf "a drawn unit" b)
    | _ => throw "a drawn unit is [template, unit]"
  let reads ← (← arrOf "reads" (← field j "reads")).mapM fun d => do
    match ← arrOf "a read" d with
    | [a, b, c, e, f] => pure (((← strOf "template" a, ← natOf "unit" b, ← strOf "port" c, ← natOf "row" e) : Key),
                               ← strOf "a string" f)
    | _ => throw "a read is [template, unit, port, row, string]"
  let writes ← (← arrOf "writes" (← field j "writes")).mapM fun d => do
    match ← arrOf "a write" d with
    | [a, b, c, e] => pure (((← strOf "template" a, ← intOf "unit" b, ← strOf "out" c) : String × Int × String),
                            ← strOf "a string" e)
    | _ => throw "a write is [template, unit, out, string]"
  let same := fun (q : Key) (wr : String × Int × String) => match reads.lookup q, writes.lookup wr with
    | some a, some b => a == b
    | _, _ => false
  return (c, drawn.eraseDups, same)

def drawnJson (accepted : Bool) (rows : List (Key × (String × Int × String)))
    (same : Key → (String × Int × String) → Bool) : Json :=
  let bad := ((rows.filter fun (q, wr) => !same q wr).map (·.1)).mergeSort fun a b => !keyLt b a
  Json.mkObj [("ok", true), ("accepted", accepted), ("checked", toJson rows.length),
    ("bad", Json.arr (bad.map keyJson).toArray)]

/-- The `wires_drawn` piece: `check`, then `Summary.drawnOk` on the drawn units. -/
def pieceDrawn (j : Json) : Except String Json := do
  let (c, drawn, same) ← drawnInstance j
  match check c with
  | .error r => return refusalJson r
  | .ok s =>
    let rows := drawn.flatMap fun (t, u) => (s.rowsAt t u).map fun ((p, r), wr) => (((t, u, p, r) : Key), wr)
    return drawnJson (s.drawnOk same drawn) rows same

/-- The `wires_drawn_unrolled` piece: `checkRef`, then every expanded entry whose unit is drawn. -/
def pieceDrawnRef (j : Json) : Except String Json := do
  let (c, drawn, same) ← drawnInstance j
  match checkRef c with
  | .error r => return refusalJson r
  | .ok es =>
    let rows := es.filter fun e => drawn.contains (e.1.1, e.1.2.1)
    return drawnJson (rows.all fun (q, wr) => same q wr) rows same

end SampledProofs.Wires
