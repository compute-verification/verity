import Properties.NetworkCertifier.Wired
import Proofs.Properties.NetworkPeriod
import Proofs.Properties.NetworkScan.Accept

/-!
# What the period check holds of an audit's free inputs: the proofs of `RowsWiresChecked` and `RowsHeld`

`auditOf_ok` reads `auditOf`'s steps: an audit's `program` decodes through its canonical text to a Program `prog`,
whose SHA-512 it records, and its `free` is what `FreeOf` states: `prog`'s input gates (`inputGates`), or with a circuit
record the count `NetworkCheck.Circuit.check` made on its `circuit`, `order` and `pins`, whose digest and pins the audit
records, no pin the scan's state. `paramsOf_audits`: the parameters' audits are their `audits` entries, each through
`auditOf`. On acceptance no audit has a finding (`check_audits`), so each audit's check passed (`auditCheck_ok`): its
registration names that SHA-512, binds the record by its digest and records each pin as its parameter's registered
root. `rowsHeld` adds the rows form's `held` (`publicJson_held`) and `rowsCircuitHeld`.
-/

namespace Properties.NetworkPeriod

open Lean (Json)
open NetworkCheck.Period
open Properties.NetworkScan.Fits (bind_ok)

theorem get_ok {a : Json} {k ctx : String} {v : Json} (h : Flock.J.get a k ctx = .ok v) :
    a.getObjVal? k = .ok v := by
  unfold Flock.J.get at h
  split at h <;> simp_all

theorem mapError_ok {ε ε' α : Type} {f : ε → ε'} {x : Except ε α} {a : α} (h : x.mapError f = .ok a) :
    x = .ok a := by
  cases x <;> simp_all [Except.mapError]

theorem ite_ok {ε α : Type} {c : Prop} [Decidable c] {x y : Except ε α} {v : α}
    (h : (if c then x else y) = .ok v) : (c ∧ x = .ok v) ∨ (¬ c ∧ y = .ok v) := by
  by_cases hc : c
  · exact .inl ⟨hc, by rwa [if_pos hc] at h⟩
  · exact .inr ⟨hc, by rwa [if_neg hc] at h⟩

theorem isStr_eq {j : Json} {s : String} (h : isStr j s = true) : j = .str s := by
  cases j with
  | str x => simp only [isStr, beq_iff_eq] at h; rw [h]
  | _ => simp [isStr] at h

theorem ok_of_findingOf {α : Type} {f : Found α} (h : findingOf f = none) : ∃ v, f = .ok v := by
  cases f with
  | error x => simp [findingOf] at h
  | ok v => exact ⟨v, rfl⟩

private theorem getObjVal?_mkObj {l : List (String × Json)} (hd : l.Pairwise (fun a b => ¬ compare a.1 b.1 = .eq))
    {k : String} {v : Json} (hm : (k, v) ∈ l) : (Json.mkObj l).getObjVal? k = .ok v := by
  simp only [Json.mkObj, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?]
  rw [Std.TreeMap.Raw.getElem?_ofList_of_mem (Std.ReflCmp.compare_self) hd hm]
  rfl

/-- The rows form's public names `heldNames` as its `held`. -/
theorem publicJson_held {x : Public} (h : x.rows = true) :
    (publicJson x).getObjValD "held" = Lean.toJson heldNames := by
  have hv : (publicJson x).getObjVal? "held" = .ok (Lean.toJson heldNames) := by
    rw [publicJson, if_pos h]
    exact getObjVal?_mkObj (by simp) (by simp)
  rw [Json.getObjValD, hv] <;> rfl

/-- An accepted rows-form period publishes the rows form's terms. -/
theorem check_rows {p : Params} {c : Commitment} {e : Evidence} {pub : Public} {prof : Profile} {s : Schedule}
    {sd : ByteArray} {scan : Scan} (hform : p.form = .rows s sd scan) (h : check p c e = .accept pub prof) :
    pub.rows = true := by
  unfold check at h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  simp only [hform] at h
  split at h
  · cases h
  · cases h
  · cases h
    rfl

/-- An accepted period registers and answers one audit for each the parameters carry, and no audit has a finding. -/
theorem check_audits {p : Params} {c : Commitment} {e : Evidence} {pub : Public} {prof : Profile}
    (h : check p c e = .accept pub prof) :
    c.audits.size = p.audits.size ∧ e.audits.size = c.audits.size ∧
      ∀ f ∈ auditChecks p c e, findingOf f = none := by
  unfold check at h
  split at h
  · cases h
  split at h
  · cases h
  rename_i hca
  split at h
  · cases h
  split at h
  · cases h
  rename_i hea
  refine ⟨Decidable.not_not.1 hca, Decidable.not_not.1 hea, fun f hf => ?_⟩
  have hmem : findingOf f ∈ (auditChecks p c e).toList.map findingOf :=
    List.mem_map.2 ⟨f, Array.mem_toList_iff.2 hf, rfl⟩
  cases hform : p.form with
  | clock ss =>
    simp only [hform] at h
    split at h
    · cases h
    · cases h
    · rename_i hF
      exact firstFinding_none hF _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _
        (List.mem_append_right _ hmem))))
  | rows s sd scan =>
    simp only [hform] at h
    split at h
    · cases h
    · cases h
    · rename_i hF
      exact firstFinding_none hF _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _
        (List.mem_append_right _ hmem))))

theorem auditChecks_getElem? {p : Params} {c : Commitment} {e : Evidence} {i : ℕ} {a : Audit} {reg : Json}
    {x : Core.Answer} (ha : p.audits[i]? = some a) (hreg : c.audits[i]? = some reg) (hx : e.audits[i]? = some x) :
    (auditChecks p c e)[i]? = some (auditCheck i a reg x) := by
  have hz : ((p.audits.zip c.audits).zip e.audits)[i]? = some ((a, reg), x) :=
    Array.getElem?_zip_eq_some.2 ⟨Array.getElem?_zip_eq_some.2 ⟨ha, hreg⟩, hx⟩
  simp [auditChecks, Array.getElem?_zipIdx, hz]

/-- **What an audit's check holds of its registration**: it names the audit's Program, and with a circuit record its
window binds the record's digest and records each pin as its parameter's registered root. -/
theorem auditCheck_ok {i : ℕ} {a : Audit} {reg : Json} {x : Core.Answer} {f : Facts}
    (h : auditCheck i a reg x = .ok f) :
    isStr (reg.getObjValD "program") a.program = true ∧ ∀ d pins, a.record = some (d, pins) →
      isStr ((reg.getObjValD "window").getObjValD "circuit") d = true ∧ ∀ q ∈ pins, pinHeld reg q = true := by
  unfold auditCheck at h
  try simp only [] at h
  rcases ite_ok h with ⟨hp, h⟩ | ⟨-, h⟩
  swap
  · exact absurd h (by intro h'; injection h')
  refine ⟨hp, fun d pins hr => ?_⟩
  simp only [hr] at h
  rcases ite_ok h with ⟨hc, h⟩ | ⟨-, h⟩
  swap
  · exact absurd h (by intro h'; injection h')
  refine ⟨hc, fun q hq => ?_⟩
  cases hf : pins.find? (!pinHeld reg ·) with
  | none => simpa using List.find?_eq_none.1 hf q hq
  | some q' =>
    obtain ⟨p0, r0⟩ := q'
    simp only [hf] at h
    exact absurd h (by intro h'; injection h')

/-- What an audit's terms count its free inputs off, `RowsWiresCheckedAt`'s last part: its Program's input gates, or
with a circuit record the count `NetworkCheck.Circuit.check` made on its `circuit`, `order` and `pins`, whose digest and
pins the audit records, no pin the scan's state. -/
def FreeOf (a : Json) (prog : Flock.Program.Prog) (au : Audit) : Prop :=
  match au.record with
  | none => au.free = inputGates prog
  | some (d, pins) =>
    ∃ (record order pinsJ : Json) (cnt : NetworkCheck.Circuit.Counts),
      a.getObjVal? "circuit" = .ok record ∧ a.getObjVal? "order" = .ok order ∧ a.getObjVal? "pins" = .ok pinsJ ∧
      NetworkCheck.Circuit.check prog record order pinsJ = .ok cnt ∧ au.free = cnt.free ∧
      NetworkCheck.Circuit.digest record = .ok d ∧ NetworkCheck.Circuit.pinsOf pinsJ = .ok pins ∧
      ∀ q ∈ pins, q.1 ≠ stateValue

/-- **What `auditOf` decodes**: the audit's `program` decodes through its canonical text to the Program whose SHA-512
the audit records, and its free inputs are counted off that Program (`FreeOf`). -/
theorem auditOf_ok {a : Json} {ctx : String} {au : Audit} (h : auditOf a ctx = .ok au) :
    ∃ (pj : Json) (s : String) (prog : Flock.Program.Prog),
      a.getObjVal? "program" = .ok pj ∧ Flock.canon pj = .ok s ∧ Flock.Program.decode s.toUTF8 = .ok prog ∧
      au.program = Flock.hexOf prog.sha512 ∧ FreeOf a prog au := by
  unfold auditOf at h
  try simp only [] at h
  rcases ite_ok h with ⟨-, h⟩ | ⟨-, h⟩
  swap
  · exact absurd h (by intro h'; injection h')
  obtain ⟨pj, hpj, h⟩ := bind_ok h
  obtain ⟨s, hs, h⟩ := bind_ok h
  obtain ⟨prog, hdec⟩ : ∃ prog, Flock.Program.decode s.toUTF8 = .ok prog := by
    cases hdec : Flock.Program.decode s.toUTF8 with
    | error e =>
      simp only [hdec] at h
      exact absurd h (by intro h'; injection h')
    | ok prog => exact ⟨prog, rfl⟩
  simp only [hdec, pure_bind] at h
  obtain ⟨root, hroot⟩ : ∃ root, prog.defs[prog.root]? = some root := by
    cases hroot : prog.defs[prog.root]? with
    | none =>
      simp only [hroot] at h
      exact absurd h (by intro h'; injection h')
    | some root => exact ⟨root, rfl⟩
  simp only [hroot] at h
  rcases ite_ok h with ⟨-, h⟩ | ⟨-, h⟩
  · exact absurd h (by intro h'; injection h')
  obtain ⟨entries, -, h⟩ := bind_ok h
  cases hc : a.getObjVal? "circuit" with
  | error e =>
    simp only [hc, pure_bind] at h
    obtain ⟨steps, -, h⟩ := bind_ok h
    cases h
    exact ⟨pj, s, prog, get_ok hpj, hs, hdec, rfl, by simp [FreeOf]⟩
  | ok record =>
    simp only [hc, bind_assoc] at h
    obtain ⟨pinsJ, hpinsJ, h⟩ := bind_ok h
    obtain ⟨held, hheld, h⟩ := bind_ok h
    by_cases hany : held.any (·.1 == stateValue) = true
    · rw [if_pos hany] at h
      exact absurd h (by intro h'; injection h')
    rw [if_neg hany] at h
    try rw [pure_bind] at h
    obtain ⟨order, horder, h⟩ := bind_ok h
    obtain ⟨cnt, hcnt, h⟩ := bind_ok h
    obtain ⟨d, hd, h⟩ := bind_ok h
    try simp only [pure_bind] at h
    obtain ⟨steps, -, h⟩ := bind_ok h
    cases h
    refine ⟨pj, s, prog, get_ok hpj, hs, hdec, rfl, ?_⟩
    simp only [FreeOf]
    exact ⟨record, order, pinsJ, cnt, hc, get_ok horder, get_ok hpinsJ, mapError_ok hcnt, rfl, mapError_ok hd,
      mapError_ok hheld, fun q hq hq1 => hany (List.any_eq_true.2 ⟨q, hq, by simp [hq1]⟩)⟩

/-- **The parameters' audits are their `audits` entries**, each decoded by `auditOf`. -/
theorem paramsOf_audits {j : Json} {p : Params} (h : paramsOf j = .ok p) :
    ∃ xs, Flock.J.arr j "audits" "the object" = .ok xs ∧ xs.mapM (auditOf · "an audit") = .ok p.audits := by
  unfold paramsOf at h
  try simp only [] at h
  split at h <;> try exact absurd h (by intro h'; injection h')
  rcases ite_ok h with ⟨-, h⟩ | ⟨-, h⟩ <;> try exact absurd h (by intro h'; injection h')
  obtain ⟨xs, hxs, h⟩ := bind_ok h
  obtain ⟨audits, haud, h⟩ := bind_ok h
  suffices hp : p.audits = audits by exact ⟨xs, hxs, by rw [hp]; exact haud⟩
  clear hxs haud
  repeat' first
    | exact absurd h (by intro h'; injection h')
    | (cases h; rfl)
    | (obtain ⟨_, -, h⟩ := bind_ok h)
    | (rcases ite_ok h with ⟨-, h⟩ | ⟨-, h⟩)
    | split at h

theorem rowsWiresChecked : RowsWiresChecked := by
  intro j p c e pub prof hp h i au reg hau hreg
  obtain ⟨hca, hea, hfind⟩ := check_audits h
  obtain ⟨xs, hxs, hmap⟩ := paramsOf_audits hp
  obtain ⟨hic, -⟩ := Array.getElem?_eq_some_iff.1 hreg
  obtain ⟨hia, hau'⟩ := Array.getElem?_eq_some_iff.1 hau
  have hie : i < e.audits.size := by omega
  have hx : e.audits[i]? = some e.audits[i] := Array.getElem?_eq_some_iff.2 ⟨hie, rfl⟩
  obtain ⟨f, hf⟩ := ok_of_findingOf (hfind _ (Array.mem_of_getElem? (auditChecks_getElem? hau hreg hx)))
  obtain ⟨hprog, hrec⟩ := auditCheck_ok hf
  have hsize := NciCheck.Properties.SecurityProofs.BodiesOk.array_mapM_size _ xs p.audits hmap
  have hix : i < xs.size := by omega
  have hget : auditOf xs[i] "an audit" = .ok au := by
    have := NciCheck.Properties.SecurityProofs.BodiesOk.array_mapM_get _ xs p.audits hmap i hix hia
    rw [hau'] at this
    exact this
  obtain ⟨pj, s, prog, hpj, hs, hd, hpr, hfree⟩ := auditOf_ok hget
  refine ⟨xs, xs[i], pj, s, prog, hxs, Array.getElem?_eq_some_iff.2 ⟨hix, rfl⟩, hpj, hs, hd, ?_, ?_⟩
  · rw [← hpr]
    exact isStr_eq hprog
  · unfold FreeOf at hfree
    revert hfree hrec
    cases hr : au.record with
    | none => intro hfree _; exact hfree
    | some dp =>
      obtain ⟨d, pins⟩ := dp
      intro hfree hrec
      obtain ⟨hcirc, hpins⟩ := hrec d pins rfl
      obtain ⟨record, order, pinsJ, cnt, h1, h2, h3, h4, h5, h6, h7, h8⟩ := hfree
      exact ⟨record, order, pinsJ, cnt, h1, h2, h3, h4, h5, h6, h7, isStr_eq hcirc,
        fun q hq => ⟨hpins q hq, h8 q hq⟩⟩

theorem rowsHeld : RowsHeld := by
  intro j p c e pub prof hp h s sd scan hform
  exact ⟨publicJson_held (check_rows hform h), rfl, rowsCircuitHeld p c e pub prof h,
    rowsWiresChecked j p c e pub prof hp h⟩

set_option pp.deepTerms true in
set_option pp.maxSteps 1000000 in
#print NetworkCheck.Period.auditOf

set_option pp.deepTerms true in
set_option pp.maxSteps 1000000 in
#print NetworkCheck.Period.paramsOf

set_option pp.deepTerms true in
set_option pp.maxSteps 1000000 in
#print NetworkCheck.Period.auditCheck

end Properties.NetworkPeriod
