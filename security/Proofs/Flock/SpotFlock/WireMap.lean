module

public import Proofs.Flock.Claims.Sampling.Rewire
public import Proofs.Flock.SpotFlock.AddressesAt
public import Proofs.Flock.SpotFlock.EntriesAt
public import Proofs.Flock.SpotFlock.Sessions

/-! The wiring round 2 checks, stated from the auditor's copies alone as a map on the members' gates (`wireMap`), and the rows it holds an accepted run's registered reads to (`wired_read`, `linked_read`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Spot (bitsProgs membersPart)
open Lean (Json)

variable (h : Held)

theorem writerOf_ownOf (st : Roots) (rec smp : Json) (c₀ : FlockDeploy.Coin) (w : Spot.SecondStage.Wire) :
    Spot.SecondStage.writerOf (ownOf h st rec smp c₀) w =
      (heldWriter h w.writer).map (partAt h (h.reg.sample (src c₀)).size) := by
  have key : (((List.range h.ms.length).map (partAt h (h.reg.sample (src c₀)).size)).toArray).find?
      (fun p => p.template == w.writer) =
        (heldWriter h w.writer).map (partAt h (h.reg.sample (src c₀)).size) := by
    rw [List.find?_toArray, List.find?_map]
    rfl
  exact key

def linkCell (t : ℕ) (port : String) (c : ℕ) : Option (String × ℕ × ℕ) :=
  match heldLink h (h.names.getD t "") port, heldRows h (h.names.getD t "") port with
  | some l, some r => (cellRow r c).map fun x => (l.value, r.per, x)
  | _, _ => none

theorem wireCell_eq {t : ℕ} {port : String} {w : Spot.SecondStage.Wire} {r : Spot.SecondStage.Rows} {tw : ℕ}
    (hl : heldLink h (h.names.getD t "") port = none) (hw : heldWire h (h.names.getD t "") port = some w)
    (hr : heldRows h (h.names.getD t "") port = some r) (htw : heldWriter h w.writer = some tw) (c : ℕ) :
    wireCell h t port c = (cellRow r c).map fun x => (tw, w.wport, x) := by
  have h1 : wireCell h t port c =
      (heldWriter h w.writer).bind fun tw => (cellRow r c).map fun x => (tw, w.wport, x) := by
    unfold wireCell
    rw [hl, hw, hr]
  rw [h1, htw]
  rfl

theorem linkCell_eq {t : ℕ} {port : String} {l : Spot.SecondStage.Link} {r : Spot.SecondStage.Rows}
    (hl : heldLink h (h.names.getD t "") port = some l) (hr : heldRows h (h.names.getD t "") port = some r) (c : ℕ) :
    linkCell h t port c = (cellRow r c).map fun x => (l.value, r.per, x) := by
  unfold linkCell
  rw [hl, hr]

def linkUnit (t : ℕ) (port : String) (u : ℕ) : Option (String × ℕ) :=
  (linkCell h t port (u % h.reg.ns.getD t 0)).map fun y => (y.1, u / h.reg.ns.getD t 0 * y.2.1 + y.2.2)

theorem wireUnit_block {t : ℕ} (port : String) (ρ : ℕ) {c : ℕ} (hc : c < h.reg.ns.getD t 0) :
    wireUnit h t port (c + ρ * h.reg.ns.getD t 0) =
      (wireCell h t port c).map fun y => (y.1, y.2.1, ρ * h.reg.ns.getD y.1 0 + y.2.2) := by
  have hpos : 0 < h.reg.ns.getD t 0 := Nat.lt_of_le_of_lt (Nat.zero_le c) hc
  unfold wireUnit
  rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hc, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hc,
    Nat.zero_add]

theorem linkUnit_block {t : ℕ} (port : String) (ρ : ℕ) {c : ℕ} (hc : c < h.reg.ns.getD t 0) :
    linkUnit h t port (c + ρ * h.reg.ns.getD t 0) =
      (linkCell h t port c).map fun y => (y.1, ρ * y.2.1 + y.2.2) := by
  have hpos : 0 < h.reg.ns.getD t 0 := Nat.lt_of_le_of_lt (Nat.zero_le c) hc
  unfold linkUnit
  rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hc, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hc,
    Nat.zero_add]

theorem wireCell_wire {t : ℕ} {port : String} {c : ℕ} {y : ℕ × String × ℕ} (hy : wireCell h t port c = some y) :
    ∃ w r, heldLink h (h.names.getD t "") port = none ∧ heldWire h (h.names.getD t "") port = some w ∧
      heldRows h (h.names.getD t "") port = some r ∧ heldWriter h w.writer = some y.1 ∧ y.2.1 = w.wport ∧
      cellRow r c = some y.2.2 := by
  unfold wireCell at hy
  cases hl : heldLink h (h.names.getD t "") port with
  | some l =>
    rw [hl] at hy
    cases hy
  | none =>
    cases hw : heldWire h (h.names.getD t "") port with
    | none =>
      rw [hl, hw] at hy
      cases hy
    | some w =>
      cases hr : heldRows h (h.names.getD t "") port with
      | none =>
        rw [hl, hw, hr] at hy
        cases hy
      | some r =>
        rw [hl, hw, hr] at hy
        have hy' : ((heldWriter h w.writer).bind fun tw => (cellRow r c).map fun x => (tw, w.wport, x)) = some y := hy
        obtain ⟨tw, htw, hy2⟩ := Option.bind_eq_some_iff.1 hy'
        obtain ⟨x, hx, hy3⟩ := Option.map_eq_some_iff.1 hy2
        subst hy3
        exact ⟨w, r, rfl, rfl, rfl, htw, rfl, hx⟩

def noSelfWire : Bool :=
  (List.range h.ms.length).all fun t => (h.wires.getD #[]).all fun w =>
    (List.range (h.reg.ns.getD t 0)).all fun c =>
      match wireCell h t w.port c with
      | some y => !(y.1 == t && y.2.2 == c)
      | none => true

theorem wireCell_ne_self (hc : noSelfWire h = true) {t : ℕ} (ht : t < h.ms.length) {port : String} {c : ℕ}
    (hcn : c < h.reg.ns.getD t 0) (wport : String) : wireCell h t port c ≠ some (t, wport, c) := by
  intro hcell
  obtain ⟨w, r, -, hw, -, -, -, -⟩ := wireCell_wire h hcell
  have hwm : w ∈ h.wires.getD #[] := Array.mem_of_find?_eq_some hw
  have hwp := Array.find?_some hw
  have hport : w.port = port := eq_of_beq (Bool.and_eq_true_iff.1 hwp).2
  have h1 := List.all_eq_true.1 hc t (List.mem_range.2 ht)
  have h2 := Array.all_eq_true_iff_forall_mem.1 h1 w hwm
  have h3 := List.all_eq_true.1 h2 c (List.mem_range.2 hcn)
  simp [hport, hcell] at h3

theorem wireUnit_ne_self (hc : noSelfWire h = true) {t : ℕ} (ht : t < h.ms.length) (port wport : String) {u : ℕ}
    (hn : 0 < h.reg.ns.getD t 0) : wireUnit h t port u ≠ some (t, wport, u) := by
  intro hu
  unfold wireUnit at hu
  obtain ⟨⟨tw, wp, x⟩, hcell, hy⟩ := Option.map_eq_some_iff.1 hu
  have h1 : tw = t := congrArg Prod.fst hy
  have h3 : u / h.reg.ns.getD t 0 * h.reg.ns.getD tw 0 + x = u := congrArg (fun z => z.2.2) hy
  rw [h1] at hcell h3
  have hx : x = u % h.reg.ns.getD t 0 :=
    Nat.add_left_cancel (h3.trans (Nat.div_add_mod' u (h.reg.ns.getD t 0)).symm)
  rw [hx] at hcell
  exact wireCell_ne_self h hc ht (Nat.mod_lt _ hn) wp hcell

/-- No wire's writer's output port is a port a link names, decided. Round 2 holds the writer's session to the link's
record at that port and a wired reader to the interior record's port, and does not hold the two entries to each other. -/
def noReturnedWired : Bool :=
  (h.wires.getD #[]).all fun w => (heldLink h w.writer w.wport).isNone

theorem writer_not_linked (hc : noReturnedWired h = true) {t port : String} {w : Spot.SecondStage.Wire}
    (hw : heldWire h t port = some w) : heldLink h w.writer w.wport = none := by
  have hm : w ∈ h.wires.getD #[] := Array.mem_of_find?_eq_some hw
  have hn : (heldLink h w.writer w.wport).isNone = true := Array.all_eq_true_iff_forall_mem.1 hc w hm
  exact Option.isNone_iff_eq_none.1 hn

def wiredProg : Prog := (bitsProgs h.ms).wired (wireMap h) (returned h)

theorem wireMap_of_target {x y : ℕ} (hxy : wireTarget h x = some y) : wireMap h x = y := by
  simp only [wireMap, hxy, Option.getD_some]

theorem wireMap_of_no_target {x : ℕ} (hx : wireTarget h x = none) : wireMap h x = x := by
  simp only [wireMap, hx, Option.getD_none]

theorem wireMap_of_not_source {x : ℕ} (hx : (sourceBits h.ms).find? (·.1 == x) = none) : wireMap h x = x :=
  wireMap_of_no_target h (by simp only [wireTarget, hx, Option.bind_none])

theorem wireTarget_spec {x y : ℕ} (hxy : wireTarget h x = some y) :
    ∃ t r tw wport u, (x, t, r) ∈ sourceBits h.ms ∧ wireUnit h t r.row r.leaf = some (tw, wport, u) ∧
      heldLink h (h.names.getD tw "") wport = none ∧
      (y, tw, Spot.Bits.RowPos.mk wport u r.bit) ∈ writtenBits h.ms ∧ returned h y = false := by
  unfold wireTarget at hxy
  obtain ⟨⟨sx, st, sr⟩, hs, hxy1⟩ := Option.bind_eq_some_iff.1 hxy
  obtain ⟨⟨tw, wport, u⟩, hu, hxy2⟩ := Option.bind_eq_some_iff.1 hxy1
  have hsp := List.find?_some hs
  have hsx : sx = x := eq_of_beq hsp
  have hsm : (x, st, sr) ∈ sourceBits h.ms := by
    rw [← hsx]
    exact List.mem_of_find?_eq_some hs
  have hxy3 : (if (heldLink h (h.names.getD tw "") wport).isSome then none
      else (((writtenBits h.ms).find? fun z =>
        z.2.1 == tw && z.2.2 == Spot.Bits.RowPos.mk wport u sr.bit).map (·.1)).filter fun g =>
          !returned h g) = some y := hxy2
  cases hv : heldLink h (h.names.getD tw "") wport with
  | some l =>
    rw [hv] at hxy3
    simp at hxy3
  | none =>
    rw [hv] at hxy3
    have hxy4 : (((writtenBits h.ms).find? fun z =>
        z.2.1 == tw && z.2.2 == Spot.Bits.RowPos.mk wport u sr.bit).map (·.1)).filter
          (fun g => !returned h g) = some y := by
      first
        | exact hxy3
        | simpa using hxy3
    obtain ⟨hxy5, hret⟩ := Option.filter_eq_some_iff.1 hxy4
    have hret' : returned h y = false := by
      cases hr : returned h y with
      | false => rfl
      | true =>
        have hret2 : (!returned h y) = true := hret
        rw [hr] at hret2
        cases hret2
    obtain ⟨⟨zx, zt, zr⟩, hz, hzy⟩ := Option.map_eq_some_iff.1 hxy5
    have hzx : zx = y := hzy
    have hzp := List.find?_some hz
    have hzt : zt = tw := eq_of_beq (Bool.and_eq_true_iff.1 hzp).1
    have hzr : zr = Spot.Bits.RowPos.mk wport u sr.bit := eq_of_beq (Bool.and_eq_true_iff.1 hzp).2
    have hm : (zx, zt, zr) ∈ writtenBits h.ms := List.mem_of_find?_eq_some hz
    rw [hzx, hzt, hzr] at hm
    exact ⟨st, sr, tw, wport, u, hsm, hu, hv, hm, hret'⟩

theorem returned_wireMap {x : ℕ} (hx : wireMap h x ≠ x) : returned h (wireMap h x) = false := by
  cases hv : wireTarget h x with
  | none => exact absurd (wireMap_of_no_target h hv) hx
  | some y =>
    rw [wireMap_of_target h hv]
    obtain ⟨_, _, _, _, _, _, _, _, _, hret⟩ := wireTarget_spec h hv
    exact hret

theorem wireTarget_of_wiresMove (hc : wiresMove h = true) {x t : ℕ} {r : Spot.Bits.RowPos}
    (hs : (x, t, r) ∈ sourceBits h.ms) (hl : heldLink h (h.names.getD t "") r.row = none)
    {w : Spot.SecondStage.Wire} (hw : heldWire h (h.names.getD t "") r.row = some w) :
    ∃ y, wireTarget h x = some y := by
  have hall : ((heldLink h (h.names.getD t "") r.row).isSome ||
      (heldWire h (h.names.getD t "") r.row).isNone || (wireTarget h x).isSome) = true :=
    List.all_eq_true.1 hc (x, t, r) hs
  rw [hl, hw] at hall
  cases hv : wireTarget h x with
  | some y => exact ⟨y, rfl⟩
  | none =>
    rw [hv] at hall
    simp at hall

def wireRow (t : ℕ) (port : String) (p : Spot.SecondStage.Part) (j : ℕ) : Option ℕ :=
  (wireCell h t port (p.unit j)).map fun y => p.span j * h.reg.ns.getD y.1 0 + y.2.2

theorem wireRow_unit {t : ℕ} {port : String} {p : Spot.SecondStage.Part} {j row : ℕ} (ρ : ℕ)
    (hc : p.unit j < h.reg.ns.getD t 0) (hrow : wireRow h t port p j = some row) :
    ∃ tw wport x, wireCell h t port (p.unit j) = some (tw, wport, x) ∧
      row = p.span j * h.reg.ns.getD tw 0 + x ∧
      wireUnit h t port (p.unit j + ρ * h.reg.ns.getD t 0) = some (tw, wport, ρ * h.reg.ns.getD tw 0 + x) := by
  unfold wireRow at hrow
  obtain ⟨⟨tw, wport, x⟩, hcell, hr⟩ := Option.map_eq_some_iff.1 hrow
  have hu := wireUnit_block h port ρ hc
  rw [hcell] at hu
  exact ⟨tw, wport, x, hcell, hr.symm, hu⟩

def linkRow (t : ℕ) (port : String) (S : Array ℕ) (p : Spot.SecondStage.Part) (j : ℕ) : Option ℕ :=
  (S[p.span j]?).bind fun b => (linkCell h t port (p.unit j)).map fun y => b * y.2.1 + y.2.2

def linkRowIn (t : ℕ) (port : String) (p : Spot.SecondStage.Part) (j : ℕ) : Option ℕ :=
  (linkCell h t port (p.unit j)).map fun y => p.span j * y.2.1 + y.2.2

theorem wiredRow_cell (p : Spot.SecondStage.Part) (r : Spot.SecondStage.Rows) (j : ℕ) :
    Spot.SecondStage.wiredRow p r j = (cellRow r (p.unit j)).map fun x => p.span j * r.per + x := by
  unfold Spot.SecondStage.wiredRow cellRow
  cases Spot.SecondStage.rowAt r.runs (p.unit j) with
  | none => rfl
  | some x =>
    show (if x < r.per then some (p.span j * r.per + x) else none) =
      (if x < r.per then some x else none).map fun x => p.span j * r.per + x
    by_cases hlt : x < r.per
    · rw [ite_eq_left hlt, ite_eq_left hlt]
      rfl
    · rw [ite_eq_right hlt, ite_eq_right hlt]
      rfl

theorem linkedRow_cell (own : Spot.SecondStage.Own) (p : Spot.SecondStage.Part) (r : Spot.SecondStage.Rows) (j : ℕ) :
    Spot.SecondStage.linkedRow own p r j =
      (own.sample[p.span j]?).bind fun b => (cellRow r (p.unit j)).map fun x => b * r.per + x := by
  unfold Spot.SecondStage.linkedRow cellRow
  cases own.sample[p.span j]? with
  | none => rfl
  | some b =>
    cases Spot.SecondStage.rowAt r.runs (p.unit j) with
    | none => rfl
    | some x =>
      show (if x < r.per then some (b * r.per + x) else none) =
        (if x < r.per then some x else none).map fun x => b * r.per + x
      by_cases hlt : x < r.per
      · rw [ite_eq_left hlt, ite_eq_left hlt]
        rfl
      · rw [ite_eq_right hlt, ite_eq_right hlt]
        rfl

theorem wireRow_eq {t : ℕ} {port : String} {w : Spot.SecondStage.Wire} {r : Spot.SecondStage.Rows} {tw : ℕ}
    (hl : heldLink h (h.names.getD t "") port = none) (hw : heldWire h (h.names.getD t "") port = some w)
    (hr : heldRows h (h.names.getD t "") port = some r) (htw : heldWriter h w.writer = some tw)
    (hn : h.reg.ns.getD tw 0 = r.per) (p : Spot.SecondStage.Part) :
    wireRow h t port p = Spot.SecondStage.wiredRow p r := by
  funext j
  unfold wireRow
  rw [wireCell_eq h hl hw hr htw, wiredRow_cell]
  cases cellRow r (p.unit j) with
  | none => rfl
  | some x =>
    show some (p.span j * h.reg.ns.getD tw 0 + x) = some (p.span j * r.per + x)
    rw [hn]

theorem linkRow_eq {t : ℕ} {port : String} {l : Spot.SecondStage.Link} {r : Spot.SecondStage.Rows}
    (hl : heldLink h (h.names.getD t "") port = some l) (hr : heldRows h (h.names.getD t "") port = some r)
    (own : Spot.SecondStage.Own) (p : Spot.SecondStage.Part) :
    linkRow h t port own.sample p = Spot.SecondStage.linkedRow own p r := by
  funext j
  unfold linkRow
  rw [linkCell_eq h hl hr, linkedRow_cell]
  cases own.sample[p.span j]? with
  | none => rfl
  | some b => cases cellRow r (p.unit j) <;> rfl

theorem linkRowIn_eq {t : ℕ} {port : String} {l : Spot.SecondStage.Link} {r : Spot.SecondStage.Rows}
    (hl : heldLink h (h.names.getD t "") port = some l) (hr : heldRows h (h.names.getD t "") port = some r)
    (p : Spot.SecondStage.Part) : linkRowIn h t port p = Spot.SecondStage.wiredRow p r := by
  funext j
  unfold linkRowIn
  rw [linkCell_eq h hl hr, wiredRow_cell]
  cases cellRow r (p.unit j) <;> rfl

theorem readOk_linked {own : Spot.SecondStage.Own} {named : Bool} {p : Spot.SecondStage.Part} {port : String}
    {e : Json} {l : Spot.SecondStage.Link} (hl : Spot.SecondStage.linkOf own p.template port = some l)
    (hok : Spot.SecondStage.readOk own named p port e = true) :
    Spot.getD e "value" = .str l.value ∧ ∃ r, Spot.SecondStage.rowsOf own p.template port = some r ∧
      ((∃ x, Spot.SecondStage.registeredEntry own.interior l.value = some x ∧
          Spot.SecondStage.sameAt e x = true ∧
          Spot.SecondStage.placedOk own.sample.size p e (own.sample.size * r.per)
            (Spot.SecondStage.wiredRow p r) = true) ∨
        (Spot.SecondStage.registeredEntry own.interior l.value = none ∧ ∃ b m,
          (l.value, b) ∈ Spot.Record.byPort own.boundary ∧
          Spot.nat? (Spot.getD own.boundary "population") = some m ∧ Spot.SecondStage.sameAt e b = true ∧
          Spot.SecondStage.placedOk own.sample.size p e (m * r.per)
            (Spot.SecondStage.linkedRow own p r) = true)) := by
  unfold Spot.SecondStage.readOk at hok
  rw [hl] at hok
  cases hv : Spot.getD e "value" with
  | str v =>
    rw [hv] at hok
    simp only [Bool.and_eq_true, beq_iff_eq] at hok
    obtain ⟨rfl, hm⟩ := hok
    refine ⟨rfl, ?_⟩
    cases hri : Spot.SecondStage.registeredEntry own.interior l.value with
    | some x =>
      cases hro : Spot.SecondStage.rowsOf own p.template port with
      | none => simp [hri, hro] at hm
      | some ro =>
        simp only [hri, hro, Bool.and_eq_true] at hm
        exact ⟨ro, rfl, Or.inl ⟨x, rfl, hm.1, hm.2⟩⟩
    | none =>
      cases hf : (Spot.Record.byPort own.boundary).find? (·.1 == l.value) with
      | none => simp [hri, hf] at hm
      | some kb =>
        obtain ⟨k, b⟩ := kb
        cases hpop : Spot.nat? (Spot.getD own.boundary "population") with
        | none => simp [hri, hf, hpop] at hm
        | some m =>
          cases hro : Spot.SecondStage.rowsOf own p.template port with
          | none => simp [hri, hf, hpop, hro] at hm
          | some ro =>
            simp only [hri, hf, hpop, hro, Bool.and_eq_true] at hm
            have hk : k = l.value := by simpa using Array.find?_some hf
            subst hk
            exact ⟨ro, rfl, Or.inr ⟨rfl, b, m, Array.mem_of_find?_eq_some hf, rfl, hm.1, hm.2⟩⟩
  | _ => rw [hv] at hok; simp at hok

theorem readOk_of_stmtOk {x m₁ m₂ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) {t : ℕ} (ht : t < h.ms.length) :
    ∃ st rec smp w S, decode x = some st ∧ jsonMsg m₁ = some rec ∧ secondDraw h c₀ c₁ = some smp ∧
      FlockDeploy.decode (stmtAt h m₂ t) = some w ∧ stmtJsonOf h t w = some S ∧
      ∀ port e, (port, e) ∈ Spot.entries (Spot.getD S.registered "ports") →
        Spot.SecondStage.readOk (ownOf h st rec smp c₀) (Spot.SecondStage.byTemplate rec)
          (partAt h (ownOf h st rec smp c₀).sample.size t) port e = true := by
  have hat := stmtOkAt_of h hok ht
  obtain ⟨s, hx⟩ := parse_of_stmtOkAt h hat
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨st, rec, smp, w, S, hst, hrec, hsmp, hw, hS, hc⟩ := readsOk_true h hat.2
  obtain ⟨-, -, -, -, -, -, hreads, -, -, -⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  refine ⟨st, rec, smp, w, S, hst, hrec, hsmp, hw, hS, fun port e hm => ?_⟩
  have hrd : Spot.SecondStage.readOk (ownOf h st rec smp c₀) (Spot.SecondStage.byTemplate rec)
      (partAt h (ownOf h st rec smp c₀).sample.size t) port e = true :=
    Array.all_eq_true_iff_forall_mem.1 hreads _ hm
  exact hrd

theorem wired_read {x m₁ m₂ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) {t : ℕ} (ht : t < h.ms.length) :
    ∃ st rec smp w S, decode x = some st ∧ jsonMsg m₁ = some rec ∧ secondDraw h c₀ c₁ = some smp ∧
      FlockDeploy.decode (stmtAt h m₂ t) = some w ∧ stmtJsonOf h t w = some S ∧
      ∀ port e wr, (port, e) ∈ Spot.entries (Spot.getD S.registered "ports") →
        heldLink h (h.names.getD t "") port = none → heldWire h (h.names.getD t "") port = some wr →
        ∃ tw r xe, heldWriter h wr.writer = some tw ∧ heldRows h (h.names.getD t "") port = some r ∧
          h.reg.ns.getD tw 0 = r.per ∧
          Spot.getD e "value" =
            .str (Spot.SecondStage.portOf (Spot.SecondStage.byTemplate rec) wr.writer wr.wport) ∧
          (Spot.SecondStage.portOf (Spot.SecondStage.byTemplate rec) wr.writer wr.wport, xe) ∈
            Spot.Record.byPort rec ∧
          Spot.SecondStage.sameAt e xe = true ∧
          Spot.SecondStage.placedOk (h.reg.sample (src c₀)).size (partAt h (h.reg.sample (src c₀)).size t) e
            ((h.reg.sample (src c₀)).size * h.reg.ns.getD tw 0)
            (wireRow h t port (partAt h (h.reg.sample (src c₀)).size t)) = true := by
  obtain ⟨st, rec, smp, w, S, hst, hrec, hsmp, hw, hS, hall⟩ := readOk_of_stmtOk h hok ht
  refine ⟨st, rec, smp, w, S, hst, hrec, hsmp, hw, hS, fun port e wr hm hl hwr => ?_⟩
  obtain ⟨xe, pw, r, hv, hmem, hpw, hr, hn, hsame, hpl⟩ :=
    Spot.SecondStage.readOk_wired (own := ownOf h st rec smp c₀)
      (named := Spot.SecondStage.byTemplate rec) (p := partAt h (ownOf h st rec smp c₀).sample.size t)
      (port := port) (e := e) (w := wr) hl hwr (hall port e hm)
  rw [writerOf_ownOf] at hpw
  obtain ⟨tw, htw, rfl⟩ := Option.map_eq_some_iff.1 hpw
  have hn' : h.reg.ns.getD tw 0 = r.per := hn
  have hr' : heldRows h (h.names.getD t "") port = some r := hr
  have hpl' : Spot.SecondStage.placedOk (h.reg.sample (src c₀)).size (partAt h (h.reg.sample (src c₀)).size t) e
      ((h.reg.sample (src c₀)).size * r.per)
      (Spot.SecondStage.wiredRow (partAt h (h.reg.sample (src c₀)).size t) r) = true := hpl
  refine ⟨tw, r, xe, htw, hr', hn', hv, hmem, hsame, ?_⟩
  rw [wireRow_eq h hl hwr hr' htw hn', hn']
  exact hpl'

theorem linked_read {x m₁ m₂ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) {t : ℕ} (ht : t < h.ms.length) :
    ∃ st rec smp w S, decode x = some st ∧ jsonMsg m₁ = some rec ∧ secondDraw h c₀ c₁ = some smp ∧
      FlockDeploy.decode (stmtAt h m₂ t) = some w ∧ stmtJsonOf h t w = some S ∧
      ∀ port e l, (port, e) ∈ Spot.entries (Spot.getD S.registered "ports") →
        heldLink h (h.names.getD t "") port = some l →
        Spot.getD e "value" = .str l.value ∧ ∃ r, heldRows h (h.names.getD t "") port = some r ∧
          ((∃ xe, Spot.SecondStage.registeredEntry rec l.value = some xe ∧ Spot.SecondStage.sameAt e xe = true ∧
              Spot.SecondStage.placedOk (h.reg.sample (src c₀)).size (partAt h (h.reg.sample (src c₀)).size t) e
                ((h.reg.sample (src c₀)).size * r.per)
                (linkRowIn h t port (partAt h (h.reg.sample (src c₀)).size t)) = true) ∨
            (Spot.SecondStage.registeredEntry rec l.value = none ∧ ∃ b m,
              (l.value, b) ∈ Spot.Record.byPort st.record ∧
              Spot.nat? (Spot.getD st.record "population") = some m ∧ Spot.SecondStage.sameAt e b = true ∧
              Spot.SecondStage.placedOk (h.reg.sample (src c₀)).size (partAt h (h.reg.sample (src c₀)).size t) e
                (m * r.per)
                (linkRow h t port (h.reg.sample (src c₀)) (partAt h (h.reg.sample (src c₀)).size t)) = true)) := by
  obtain ⟨st, rec, smp, w, S, hst, hrec, hsmp, hw, hS, hall⟩ := readOk_of_stmtOk h hok ht
  refine ⟨st, rec, smp, w, S, hst, hrec, hsmp, hw, hS, fun port e l hm hl => ?_⟩
  obtain ⟨hv, r, hr, hcase⟩ :=
    readOk_linked (own := ownOf h st rec smp c₀) (named := Spot.SecondStage.byTemplate rec)
      (p := partAt h (ownOf h st rec smp c₀).sample.size t) (port := port) (e := e) (l := l) hl (hall port e hm)
  have hr' : heldRows h (h.names.getD t "") port = some r := hr
  refine ⟨hv, r, hr', ?_⟩
  rcases hcase with ⟨xe, hxe, hsame, hpl⟩ | ⟨hnone, b, m, hb, hpop, hsame, hpl⟩
  · refine Or.inl ⟨xe, hxe, hsame, ?_⟩
    have hpl' : Spot.SecondStage.placedOk (h.reg.sample (src c₀)).size (partAt h (h.reg.sample (src c₀)).size t) e
        ((h.reg.sample (src c₀)).size * r.per)
        (Spot.SecondStage.wiredRow (partAt h (h.reg.sample (src c₀)).size t) r) = true := hpl
    rw [linkRowIn_eq h hl hr']
    exact hpl'
  · refine Or.inr ⟨hnone, b, m, hb, hpop, hsame, ?_⟩
    have hrow : linkRow h t port (h.reg.sample (src c₀)) (partAt h (h.reg.sample (src c₀)).size t) =
        Spot.SecondStage.linkedRow (ownOf h st rec smp c₀) (partAt h (h.reg.sample (src c₀)).size t) r :=
      linkRow_eq h hl hr' (ownOf h st rec smp c₀) (partAt h (h.reg.sample (src c₀)).size t)
    have hpl' : Spot.SecondStage.placedOk (h.reg.sample (src c₀)).size (partAt h (h.reg.sample (src c₀)).size t) e
        (m * r.per)
        (Spot.SecondStage.linkedRow (ownOf h st rec smp c₀) (partAt h (h.reg.sample (src c₀)).size t) r) =
          true := hpl
    rw [hrow]
    exact hpl'

theorem addresses_one_way (hA : Addresses h) (g : ℕ) : ∃ S : Address,
    (∀ r : CheckedRun h.stage, r.Ok h → r.s.prog.Needs r.b g → r.ReadsRegistered h S g) ∨
      ∀ r : CheckedRun h.stage, r.Ok h → r.s.prog.Needs r.b g → r.ReadsInterior h S g := by
  obtain ⟨S, hS⟩ := hA g
  refine ⟨S, ?_⟩
  by_cases hs : S.slot.sized = true
  · refine Or.inr fun r hr hg => ?_
    obtain ⟨-, -, hrd⟩ := hS r hr hg
    rcases hrd with ⟨reg, -, -, -, -, -, hf, -⟩ | hI
    · exact absurd (hf.symm.trans hs) (by decide)
    · exact hI
  · refine Or.inl fun r hr hg => ?_
    obtain ⟨-, -, hrd⟩ := hS r hr hg
    rcases hrd with hR | ⟨-, -, -, -, -, ht, -⟩
    · exact hR
    · exact absurd ht hs

end SpotFlock
