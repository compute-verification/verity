module

public import Proofs.Flock.SpotFlock.AddressFacts.Row

/-!
# Round 2's row is the read's

`second-read` (`Spot.SecondStage.readOk`) holds every read of a registered-reads entry at a port a link or a wire
names to the row `placedOk` names, and at a template's own port of the interior record to the row `rowsOk` names.
`placedOk_read` and `rowsOk_read` read that off the two checks at one slot (`readsAt`), and `readFact` assembles them
by the read's kind (`placedRow`): `AddressRow.ReadFact`.

Each proof cases on the entry's `reads_from` and `reads` so that its reader and the check's reduce together: the two
are separate matches on the same fields.
-/

@[expose] public section

namespace SpotFlock

namespace AddressRow

open Lean (Json)

/- The step both checks share once `reads` is `rs` and `reads_from` is `f`: the check's clause at slot `u`, `hall` at
`u - f`, read at the row `hr` names. Hygiene is off so that it reads the proof's own `hok`, `hrd`, `rs`, `f`, `u`. -/
set_option hygiene false in
local macro "slot_row " hr:ident : tactic => `(tactic| (
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range] at hok
  obtain ⟨-, hall⟩ := hok
  by_cases hfu : f ≤ u
  · simp only [hfu, ↓reduceIte] at hrd
    cases hx : rs[u - f]? with
    | none => simp [hx] at hrd
    | some x =>
      simp only [hx, Option.bind_some] at hrd
      have hk : u - f < rs.size := by
        by_contra hc
        simp [Nat.le_of_not_lt hc] at hx
      have hgx : rs.getD (u - f) Json.null = x := by
        first
          | simp [Array.getD_eq_getD_getElem?, hx]
          | simp [Array.getD, hk, hx]
          | (have hx' : rs[u - f] = x := by simpa [hk] using hx
             simp [Array.getD, hk, hx'])
      have h1 := hall (u - f) hk
      rw [hgx, show f + (u - f) = u by omega, $hr:ident] at h1
      cases x with
      | null => simp [Spot.nat?, Spot.int?] at hrd
      | bool _ => simp [Spot.nat?, Spot.int?] at hrd
      | str _ => simp [Spot.nat?, Spot.int?] at hrd
      | arr _ => simp [Spot.nat?, Spot.int?] at hrd
      | obj _ => simp [Spot.nat?, Spot.int?] at hrd
      | num _ =>
        simp only [hrd, Bool.and_eq_true, beq_iff_eq] at h1
        exact h1.2
  · simp [hfu] at hrd))

/-- **A read `placedOk` passes is its row**: where `placedOk` passes entry `e` against rows `row`, the read of slot `u`
(`readsAt`) is the row `row u` names. -/
theorem placedOk_read {q : ℕ} {p : Spot.SecondStage.Part} {e : Json} {leaves : ℕ} {row : ℕ → Option ℕ}
    {u pos r : ℕ} (hok : Spot.SecondStage.placedOk q p e leaves row = true) (hrd : readsAt e u = some pos)
    (hrow : row u = some r) : pos = r := by
  obtain ⟨F, hF⟩ : ∃ F, Spot.getD e "reads_from" = F := ⟨_, rfl⟩
  obtain ⟨R, hR⟩ : ∃ R, Spot.getD e "reads" = R := ⟨_, rfl⟩
  obtain ⟨L, hL⟩ : ∃ L, Spot.nat? (Spot.getD e "leaves") = L := ⟨_, rfl⟩
  simp only [readsAt, hF, hR] at hrd
  simp only [Spot.SecondStage.placedOk, hF, hR, hL] at hok
  cases R with
  | null => simp at hrd
  | bool _ => simp at hrd
  | num _ => simp at hrd
  | str _ => simp at hrd
  | obj _ => simp at hrd
  | arr rs =>
    cases L with
    | none => simp at hok
    | some l =>
      cases F with
      | bool _ => simp [Spot.nat?, Spot.int?] at hrd
      | str _ => simp [Spot.nat?, Spot.int?] at hrd
      | arr _ => simp [Spot.nat?, Spot.int?] at hrd
      | obj _ => simp [Spot.nat?, Spot.int?] at hrd
      | null =>
        simp only at hrd hok
        generalize (0 : ℕ) = f at hrd hok
        slot_row hrow
      | num n =>
        simp only at hrd hok
        generalize Spot.nat? (Json.num n) = fst at hrd hok
        cases fst with
        | none => simp at hrd
        | some f =>
          simp only at hrd hok
          slot_row hrow

/-- **A read `rowsOk` passes is its unit's first row**: where `rowsOk` passes entry `e` of `l` leaves over `units` units
against the units `unit` names, the read of slot `u` (`readsAt`) is unit `unit u`'s first row, `l / units` rows a
unit. -/
theorem rowsOk_read {q : ℕ} {p : Spot.SecondStage.Part} {e : Json} {units : ℕ} {unit : ℕ → Option ℕ}
    {u pos b l : ℕ} (hok : Spot.SecondStage.rowsOk q p e units unit = true)
    (hl : Spot.nat? (Spot.getD e "leaves") = some l) (hrd : readsAt e u = some pos) (hb : unit u = some b) :
    pos = b * (l / units) := by
  obtain ⟨F, hF⟩ : ∃ F, Spot.getD e "reads_from" = F := ⟨_, rfl⟩
  obtain ⟨R, hR⟩ : ∃ R, Spot.getD e "reads" = R := ⟨_, rfl⟩
  simp only [readsAt, hF, hR] at hrd
  simp only [Spot.SecondStage.rowsOk, hF, hR, hl] at hok
  cases R with
  | null => simp at hrd
  | bool _ => simp at hrd
  | num _ => simp at hrd
  | str _ => simp at hrd
  | obj _ => simp at hrd
  | arr rs =>
    cases F with
    | bool _ => simp [Spot.nat?, Spot.int?] at hrd
    | str _ => simp [Spot.nat?, Spot.int?] at hrd
    | arr _ => simp [Spot.nat?, Spot.int?] at hrd
    | obj _ => simp [Spot.nat?, Spot.int?] at hrd
    | null =>
      simp only at hrd hok
      generalize (0 : ℕ) = f at hrd hok
      slot_row hb
    | num n =>
      simp only at hrd hok
      generalize Spot.nat? (Json.num n) = fst at hrd hok
      cases fst with
      | none => simp at hrd
      | some f =>
        simp only at hrd hok
        slot_row hb

/-- **Round 2's row is the read's** (`ReadFact`): at each kind of read `placedRow` names a row for, `second-read`'s
case for it (`readOk`) passes the entry through `placedOk` or `rowsOk` at that row. -/
theorem readFact : ReadFact := by
  intro own named p port v e l u row pos hok hv hl hpl hrd
  simp only [Spot.SecondStage.readOk, hv] at hok
  simp only [placedRow, kindOf] at hpl
  cases hlk : Spot.SecondStage.linkOf own p.template port with
  | some lk =>
    simp only [hlk] at hok hpl
    cases hre : Spot.SecondStage.registeredEntry own.interior v with
    | some x =>
      cases hro : Spot.SecondStage.rowsOf own p.template port with
      | none => simp [hre, hro] at hpl
      | some rr =>
        simp [hre, hro] at hpl
        simp only [hre, hro, Bool.and_eq_true] at hok
        exact placedOk_read hok.2.2 hrd hpl
    | none =>
      cases hro : Spot.SecondStage.rowsOf own p.template port with
      | none => simp [hre, hro] at hpl
      | some rr =>
        simp [hre, hro] at hpl
        cases hb : (Spot.Record.byPort own.boundary).find? (·.1 == v) with
        | none => simp [hre, hb] at hok
        | some bx =>
          obtain ⟨bn, bj⟩ := bx
          cases hm : Spot.nat? (Spot.getD own.boundary "population") with
          | none => simp [hre, hb, hm] at hok
          | some m =>
            simp only [hre, hb, hm, hro, Bool.and_eq_true] at hok
            exact placedOk_read hok.2.2 hrd hpl
  | none =>
    cases hw : Spot.SecondStage.wireOf own p.template port with
    | some w =>
      simp only [hlk, hw] at hok hpl
      cases hro : Spot.SecondStage.rowsOf own p.template port with
      | none => simp [hro] at hpl
      | some rr =>
        simp [hro] at hpl
        cases hi : (Spot.Record.byPort own.interior).find? (·.1 == v) with
        | none => simp [hi] at hok
        | some ix =>
          obtain ⟨xn, xj⟩ := ix
          cases hpw : Spot.SecondStage.writerOf own w with
          | none => simp [hi, hpw] at hok
          | some pw =>
            simp only [hi, hpw, hro, Bool.and_eq_true] at hok
            exact placedOk_read hok.2.2 hrd hpl
    | none =>
      simp only [hlk, hw] at hok hpl
      cases hi : (Spot.Record.byPort own.interior).find? (·.1 == v) with
      | none => simp [hi] at hpl
      | some ix =>
        obtain ⟨xn, xj⟩ := ix
        simp [hi] at hpl
        simp only [hi, Bool.and_eq_true] at hok
        exact (rowsOk_read (b := u - p.lo) hok.2 hl hrd rfl).trans hpl

/-- **The header's position for an instance is the registered read's at its label** (`PositionFact`): `rows_open` at
the session's first table, read at the instance's label (`Registered.Port.readAt_eq`). -/
theorem positionFact : PositionFact := by
  intro xS ins spec sts own w σ hset hown hsc hck h0 hlab reg hreg q tb positions paths hq htb hent n hn
  have e0 : sts[0]! = sts[0] := getElem!_pos sts 0 h0
  obtain ⟨tb', idx, ht', hidx, hall⟩ := rows_open hset hown hsc hck 0 h0
  obtain ⟨q', positions', paths', hq', hent', hrows⟩ := hall reg hreg
  have hl := hlab idx (by rw [e0]; exact hidx) n hn
  rw [e0] at hq htb hent hn
  have hqq : q = q' := Option.some.inj (hq.symm.trans hq')
  have htt : tb = tb' := Option.some.inj (htb.symm.trans ht')
  obtain ⟨hpos, -⟩ := Prod.mk.inj (Except.ok.inj (hent.symm.trans hent'))
  subst hqq htt hpos
  obtain ⟨-, hreads, -⟩ := hrows n hn
  rw [← _root_.Flock.Registered.Port.readAt_eq, hl] at hreads
  rw [e0]
  exact hreads

end AddressRow

end SpotFlock
