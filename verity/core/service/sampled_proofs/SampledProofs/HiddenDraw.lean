/-
A draw over hidden positions (`one_stage.hidden_draw`): a stratum's population is `M` positions whose layout is private.
The prover commits a calls table before the draw coin, call `c` an `offset` and its live `rows`, each row `width` units
(column tiles) wide, and positions `r < M` are drawn from the challenger's coins. A drawn position is proved by a witness
`(c, j, t)` that the opening checks inside the proof: call `c` covers `r` (`offset ≤ r < offset + rows·width`), and
`r − offset = j·width + t`, so `j` is the live row and `t` the tile (`witness`).

The verifier never sees the table, and soundness needs no proof that it is the honest prover's (`offsets`, the prefix
sums): whatever call the prover names for each position, two positions never share a witness (`witness_inj`), so the
positions a table's witnesses prove inject into its (call, row, tile) triples, each in range (`witness_bounds`). A table
whose calls overlap or leave gaps only gives up positions. `locate` is the honest prover's witness, the first call that
covers the position (`locate_witness`).
-/

namespace SampledProofs.HiddenDraw

structure Call where
  offset : Nat
  rows : Nat
  deriving Repr, DecidableEq

def covers (width : Nat) (c : Call) (r : Nat) : Bool := c.offset ≤ r && r < c.offset + c.rows * width

def row (width : Nat) (c : Call) (r : Nat) : Nat := (r - c.offset) / width

def tile (width : Nat) (c : Call) (r : Nat) : Nat := (r - c.offset) % width

/-- The witness `(c, j, t)` for position `r` when the prover names call `c`, or none when `c` is no call or doesn't cover
`r`. -/
def witness (width : Nat) (cs : List Call) (c r : Nat) : Option (Nat × Nat × Nat) :=
  match cs[c]? with
  | some call => if covers width call r then some (c, row width call r, tile width call r) else none
  | none => none

/-- The honest prover's offsets: call `c` starts after the positions of the calls before it. -/
def offsets (width : Nat) : List Nat → List Nat
  | ms => (ms.foldl (fun (acc : List Nat × Nat) m => (acc.1 ++ [acc.2], acc.2 + m * width)) ([], 0)).1

/-- The positions the honest table covers, `M = width · Σ rows`. -/
def total (width : Nat) (ms : List Nat) : Nat := ms.foldl (fun s m => s + m * width) 0

/-- The honest prover's table over the calls' live rows. -/
def table (width : Nat) (ms : List Nat) : List Call := (offsets width ms).zip ms |>.map fun (o, m) => ⟨o, m⟩

def locateFrom (width : Nat) (r : Nat) : Nat → List Call → Option (Nat × Nat × Nat)
  | _, [] => none
  | c, call :: cs => if covers width call r then some (c, row width call r, tile width call r)
    else locateFrom width r (c + 1) cs

/-- The first call that covers `r`, with its row and tile. -/
def locate (width : Nat) (cs : List Call) (r : Nat) : Option (Nat × Nat × Nat) := locateFrom width r 0 cs

theorem covers_iff (width : Nat) (c : Call) (r : Nat) :
    covers width c r = true ↔ c.offset ≤ r ∧ r < c.offset + c.rows * width := by
  simp [covers]

/-- Within one call, a position is its row and tile: `r = offset + row·width + tile`. -/
theorem decompose (width : Nat) (c : Call) (r : Nat) (h : covers width c r = true) :
    r = c.offset + width * row width c r + tile width c r := by
  have hc := (covers_iff width c r).mp h
  have hd := Nat.div_add_mod (r - c.offset) width
  unfold row tile
  omega

theorem witness_some {width : Nat} {cs : List Call} {c r : Nat} {w : Nat × Nat × Nat} (h : witness width cs c r = some w) :
    ∃ call, cs[c]? = some call ∧ covers width call r = true ∧ w = (c, row width call r, tile width call r) := by
  unfold witness at h
  split at h
  · rename_i call hcall
    by_cases hv : covers width call r = true
    · simp only [hv, ite_true, Option.some.injEq] at h
      exact ⟨call, hcall, hv, h.symm⟩
    · simp [hv] at h
  · simp at h

/-- Two positions never share a witness, whichever calls the prover names for them. -/
theorem witness_inj (width : Nat) (cs : List Call) (c c' r r' : Nat) (w : Nat × Nat × Nat)
    (h : witness width cs c r = some w) (h' : witness width cs c' r' = some w) : r = r' := by
  obtain ⟨call, hcall, hv, rfl⟩ := witness_some h
  obtain ⟨call', hcall', hv', hw⟩ := witness_some h'
  simp only [Prod.mk.injEq] at hw
  obtain ⟨rfl, hj, ht⟩ := hw
  rw [hcall] at hcall'
  cases hcall'
  rw [decompose width call r hv, decompose width call r' hv', hj, ht]

/-- A witness names a call of the table, one of its live rows and a tile of the row. -/
theorem witness_bounds (width : Nat) (cs : List Call) (c r : Nat) (w : Nat × Nat × Nat)
    (h : witness width cs c r = some w) :
    ∃ call, cs[c]? = some call ∧ w.1 = c ∧ c < cs.length ∧ w.2.1 < call.rows ∧ w.2.2 < width := by
  obtain ⟨call, hcall, hv, rfl⟩ := witness_some h
  have hc := (covers_iff width call r).mp hv
  have hw : 0 < width := by
    rcases Nat.eq_zero_or_pos width with h0 | h0
    · subst h0; omega
    · exact h0
  refine ⟨call, hcall, rfl, (List.getElem?_eq_some_iff.mp hcall).1, ?_, Nat.mod_lt _ hw⟩
  show (r - call.offset) / width < call.rows
  exact (Nat.div_lt_iff_lt_mul hw).mpr (by omega)

theorem locateFrom_witness (width r : Nat) :
    ∀ (pre cs : List Call) (w : Nat × Nat × Nat), locateFrom width r pre.length cs = some w →
      witness width (pre ++ cs) w.1 r = some w
  | _, [], _, h => by simp [locateFrom] at h
  | pre, call :: cs, w, h => by
    unfold locateFrom at h
    by_cases hv : covers width call r = true
    · simp only [hv, ite_true, Option.some.injEq] at h
      subst h
      simp [witness, hv]
    · simp only [hv, Bool.false_eq_true, ite_false] at h
      have := locateFrom_witness width r (pre ++ [call]) cs w (by simpa using h)
      simpa using this

/-- The honest prover's witness is a witness. -/
theorem locate_witness (width : Nat) (cs : List Call) (r : Nat) (w : Nat × Nat × Nat)
    (h : locate width cs r = some w) : witness width cs w.1 r = some w :=
  locateFrom_witness width r [] cs w h

end SampledProofs.HiddenDraw
