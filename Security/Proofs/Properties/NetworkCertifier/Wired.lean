import Properties.NetworkCertifier.Wired
import Proofs.Properties.NetworkPeriod
import Proofs.Properties.NetworkScan.Accept

/-!
# What the period check holds of an audit's free inputs: the proofs of `RowsWiresChecked` and `RowsHeld`

`auditOf_ok` reads `auditWith`'s steps, whose audit `auditOf` keeps: an audit's `program` decodes through its canonical
text to a Program `prog`, whose SHA-512 it records, and its free inputs are what `FreeOf` states: `prog`'s input gates
(`inputGates`), or with a circuit record `NetworkCheck.Circuit.countsOf` of the record `NetworkCheck.Circuit.checked`
read off its `circuit` and `order` against `prog`, at its `pins`' parameters (`free`) and at those and the scan's state
(`stateFree`), the record the audit keeps, its digest and pins the ones the audit records, no pin the scan's state.
`paramsOf_audits` walks `paramsWith`'s steps, whose parameters `paramsOf` keeps (`jp_ok` past each refusal,
`paramsCircuit_ok` past the circuit's two forms): the parameters' audits are their `audits` entries, each through
`auditWith`. On acceptance no audit has a finding (`check_audits`), so each audit's check passed (`auditCheck_ok`):
its registration names that SHA-512, binds the record by its digest and records each pin as its parameter's registered
root. `rowsHeld` adds the rows form's `held` (`publicJson_held`) and `rowsCircuitHeld`.

The honest lemmas show the check accepting a rows-form period and an audit's registration passing each registration
step; no audit `paramsOf` accepts is exhibited, since `Flock.Program.decode` reads `Lean.Json.parse`, a `partial def`
the kernel can't unfold. The nontrivial lemmas: a verifier that decodes every parameters to one audit's period
(`acceptAllParams`) and accepts every period (`acceptAllRows`) accepts one whose audit's registration names no Program.
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

theorem map_ok {ε α β : Type} {f : α → β} {x : Except ε α} {b : β} (h : f <$> x = .ok b) :
    ∃ a, x = .ok a ∧ f a = b := by
  cases x with
  | error e => first | cases h | simp [Functor.map, Except.map] at h
  | ok a => exact ⟨a, rfl, by first | injection h | simpa [Functor.map, Except.map] using h⟩

theorem ite_ok {ε α : Type} {c : Prop} [Decidable c] {x y : Except ε α} {v : α}
    (h : (if c then x else y) = .ok v) : (c ∧ x = .ok v) ∨ (¬ c ∧ y = .ok v) := by
  by_cases hc : c
  · exact .inl ⟨hc, by rwa [if_pos hc] at h⟩
  · exact .inr ⟨hc, by rwa [if_neg hc] at h⟩

/-- A refusal a do block passes (`unless c do throw …`, its `else` the refusal): the block continues. -/
theorem jp_ok {ε α : Type} {c : Prop} [Decidable c] {x y : Except ε α} {v : α}
    (h : (if c then x else y) = .ok v) (hy : y = .ok v → False) : x = .ok v := by
  rcases ite_ok h with ⟨-, h⟩ | ⟨-, h⟩
  · exact h
  · exact (hy h).elim

/-- `paramsOf`'s circuit: in either form of its keys, the parameters continue (`jc`) at some circuit. -/
theorem paramsCircuit_ok {α : Type} {c : Prop} [Decidable c] {jc : Option String → Except String α}
    {y : Except String α} {v : α} (h : (if c then (pure none >>= jc) else y) = .ok v)
    (hy : y = .ok v → ∃ o, jc o = .ok v) : ∃ o, jc o = .ok v := by
  rcases ite_ok h with ⟨-, h⟩ | ⟨-, h⟩
  · obtain ⟨o, -, h⟩ := bind_ok h
    exact ⟨o, h⟩
  · exact hy h

theorem isStr_eq {j : Json} {s : String} (h : isStr j s = true) : j = .str s := by
  cases j with
  | str x => simp only [isStr, beq_iff_eq] at h; rw [h]
  | _ => simp [isStr] at h

theorem ok_of_findingOf {α : Type} {f : Found α} (h : findingOf f = none) : ∃ v, f = .ok v := by
  cases f with
  | error x => simp [findingOf] at h
  | ok v => exact ⟨v, rfl⟩

/-- A registration whose window registers the root `q.2` for the parameter `q.1` holds the pin `q`. -/
theorem pinHeld_of {reg : Json} {q : String × String}
    (h : isStr (((reg.getObjValD "window").getObjValD "registered_roots").getObjValD q.1) q.2 = true) :
    pinHeld reg q = true := by
  simp [pinHeld, isStr_eq h]

private theorem getObjVal?_mkObj {l : List (String × Json)} (hd : l.Pairwise (fun a b => ¬ compare a.1 b.1 = .eq))
    {k : String} {v : Json} (hm : (k, v) ∈ l) : (Json.mkObj l).getObjVal? k = .ok v := by
  simp only [Json.mkObj, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?]
  rw [Std.TreeMap.Raw.getElem?_ofList_of_mem (Std.ReflCmp.compare_self) hd hm]
  rfl

/-- The rows form's public names `heldNames` as its `held`. -/
theorem publicJson_held {x : Public} (h : x.rows = true) :
    (publicJson x).getObjValD "held" = Lean.toJson heldNames := by
  have hv : (publicJson x).getObjVal? "held" = .ok (Lean.toJson heldNames) := by
    rw [publicJson, if_pos h, if_pos h]
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

/-- What an audit's terms count its free inputs off, `RowsWiresCheckedAt`'s last part less what its registration
binds: its Program's input gates, or with a circuit record `NetworkCheck.Circuit.countsOf` of the record
`NetworkCheck.Circuit.checked` read off its `circuit` and `order` against the Program, which the audit keeps, at its
`pins`' parameters (`free`) and at those and the scan's state (`stateFree`), the record's digest and the pins the ones
the audit records, no pin the scan's state. -/
def FreeOf (a : Json) (prog : Flock.Program.Prog) (au : Audit) : Prop :=
  match au.record with
  | none => au.free = inputGates prog ∧ au.stateFree = inputGates prog ∧ au.circuit = none
  | some (d, pins) =>
    ∃ (record order pinsJ : Json) (r : NetworkCheck.Circuit.Stated) (sm : SampledProofs.Wires.Summary),
      a.getObjVal? "circuit" = .ok record ∧ a.getObjVal? "order" = .ok order ∧ a.getObjVal? "pins" = .ok pinsJ ∧
      NetworkCheck.Circuit.checked prog record order = .ok (r, sm) ∧
      NetworkCheck.Circuit.pinsOf pinsJ = .ok pins ∧ au.circuit = some r ∧
      au.free = (NetworkCheck.Circuit.countsOf r sm (pins.map (·.1))).free ∧
      au.stateFree = (NetworkCheck.Circuit.countsOf r sm (stateValue :: pins.map (·.1))).free ∧
      NetworkCheck.Circuit.digest record = .ok d ∧ ∀ q ∈ pins, q.1 ≠ stateValue

/-- **What `auditOf` decodes**: the audit's `program` decodes through its canonical text to the Program whose SHA-512
the audit records, and its free inputs are counted off that Program (`FreeOf`). -/
theorem auditOf_ok {a : Json} {ctx : String} {au : Audit} (h : auditOf a ctx = .ok au) :
    ∃ (pj : Json) (s : String) (prog : Flock.Program.Prog),
      a.getObjVal? "program" = .ok pj ∧ Flock.canon pj = .ok s ∧ Flock.Program.decode s.toUTF8 = .ok prog ∧
      au.program = Flock.hexOf prog.sha512 ∧ FreeOf a prog au := by
  replace h : ∃ l, auditWith a ctx = .ok (au, l) := by
    obtain ⟨⟨au', l⟩, h', rfl⟩ := map_ok h
    exact ⟨l, h'⟩
  obtain ⟨l, h⟩ := h
  unfold auditWith at h
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
    simp only [hc] at h
    generalize a.getObjVal? "circuits" = cf at h
    cases cf
    all_goals
      repeat (obtain ⟨_, -, h⟩ := bind_ok h)
      cases h
      exact ⟨pj, s, prog, get_ok hpj, hs, hdec, rfl, by simp [FreeOf]⟩
  | ok record =>
    simp only [hc] at h
    obtain ⟨pinsJ, hpinsJ, h⟩ := bind_ok h
    obtain ⟨held, hheld, h⟩ := bind_ok h
    by_cases hany : held.any (·.1 == stateValue) = true
    · rw [if_pos hany] at h
      exact absurd h (by intro h'; injection h')
    rw [if_neg hany] at h
    try rw [pure_bind] at h
    obtain ⟨order, horder, h⟩ := bind_ok h
    obtain ⟨⟨r, sm⟩, hrs, h⟩ := bind_ok h
    obtain ⟨d, hd, h⟩ := bind_ok h
    try simp only at h
    generalize a.getObjVal? "circuits" = cf at h
    cases cf
    all_goals
      repeat (obtain ⟨_, -, h⟩ := bind_ok h)
      cases h
      refine ⟨pj, s, prog, get_ok hpj, hs, hdec, rfl, ?_⟩
      simp only [FreeOf]
      exact ⟨record, order, pinsJ, r, sm, hc, get_ok horder, get_ok hpinsJ, mapError_ok hrs, mapError_ok hheld, rfl,
        rfl, rfl, mapError_ok hd, fun q hq hq1 => hany (List.any_eq_true.2 ⟨q, hq, by simp [hq1]⟩)⟩

/-- **The parameters' audits are their `audits` entries**, each read by `auditWith`. `paramsOf` takes `paramsWith`'s
parameters where they list no circuit file, and `paramsWith`'s steps in order are its keys, `audits`, `cells`,
`first`, `jitter`, `link`, `lms`, `key`, `session`, `program`, `circuit`, the form (`sync_set`, or `schedule` and
`scan`), `windows`, the rows form's refusals and `scan_circuits`, each a bind or a refusal it passes, its join points
left unfolded so no step copies the rest. -/
theorem paramsOf_audits {j : Json} {p : Params} (h : paramsOf j = .ok p) :
    ∃ xs read, Flock.J.arr j "audits" "the object" = .ok xs ∧ xs.mapM (auditWith · "an audit") = .ok read ∧
      p.audits = read.map (·.1) := by
  unfold paramsOf at h
  obtain ⟨⟨p', l⟩, hw, h⟩ := bind_ok h
  have hp : p' = p := by
    try dsimp only at h
    rcases ite_ok h with ⟨-, h⟩ | ⟨-, h⟩
    · exact absurd h (by intro h'; injection h')
    · first | (cases h; rfl) | exact Except.ok.inj h | simpa using h
  subst hp
  unfold paramsWith at hw
  cases j
  case obj kv =>
    replace hw := jp_ok hw (by intro h'; injection h')
    obtain ⟨xs, hxs, hw⟩ := bind_ok hw
    obtain ⟨read, hread, hw⟩ := bind_ok hw
    refine ⟨xs, read, hxs, hread, ?_⟩
    clear hxs hread
    obtain ⟨_, -, hw⟩ := bind_ok hw
    obtain ⟨_, -, hw⟩ := bind_ok hw
    obtain ⟨_, -, hw⟩ := bind_ok hw
    obtain ⟨_, -, hw⟩ := bind_ok hw
    replace hw := jp_ok hw (by intro h'; injection h')
    obtain ⟨_, -, hw⟩ := bind_ok hw
    replace hw := jp_ok hw (by intro h'; injection h')
    obtain ⟨_, -, hw⟩ := bind_ok hw
    obtain ⟨_, -, hw⟩ := bind_ok hw
    obtain ⟨_, -, hw⟩ := bind_ok hw
    obtain ⟨_, -, hw⟩ := bind_ok hw
    obtain ⟨_, -, hw⟩ := bind_ok hw
    obtain ⟨_, -, hw⟩ := bind_ok hw
    obtain ⟨_, -, hw⟩ := bind_ok hw
    obtain ⟨_, -, hw⟩ := bind_ok hw
    obtain ⟨_, -, hw⟩ := bind_ok hw
    replace hw := jp_ok hw (by intro h'; injection h')
    obtain ⟨_, -, hw⟩ := bind_ok hw
    obtain ⟨_, -, hw⟩ := bind_ok hw
    replace hw := jp_ok hw (by intro h'; injection h')
    obtain ⟨_, -, hw⟩ := bind_ok hw
    replace hw := jp_ok hw (by intro h'; injection h')
    obtain ⟨_, hw⟩ := paramsCircuit_ok hw (fun hd => by
      obtain ⟨_, -, hd⟩ := bind_ok hd
      replace hd := jp_ok hd (by intro h'; injection h')
      obtain ⟨o, -, hd⟩ := bind_ok hd
      exact ⟨o, hd⟩)
    rcases ite_ok hw with ⟨-, hw⟩ | ⟨-, hw⟩
    · obtain ⟨_, -, hw⟩ := bind_ok hw
      obtain ⟨_, -, hw⟩ := bind_ok hw
      obtain ⟨_, hform, hw⟩ := bind_ok hw
      cases hform
      obtain ⟨_, -, hw⟩ := bind_ok hw
      replace hw := jp_ok hw (by intro h'; injection h')
      first
        | (generalize (Json.obj kv).getObjVal? "scan_circuits" = sc at hw
           cases sc
           all_goals
             repeat (obtain ⟨_, -, hw⟩ := bind_ok hw)
             cases hw
             rfl
           done)
        | (dsimp only at hw
           split at hw
           all_goals
             repeat (obtain ⟨_, -, hw⟩ := bind_ok hw)
             cases hw
             rfl
           done)
    · obtain ⟨_, -, hw⟩ := bind_ok hw
      obtain ⟨_, -, hw⟩ := bind_ok hw
      obtain ⟨_, -, hw⟩ := bind_ok hw
      obtain ⟨_, -, hw⟩ := bind_ok hw
      obtain ⟨_, -, hw⟩ := bind_ok hw
      obtain ⟨_, hform, hw⟩ := bind_ok hw
      cases hform
      obtain ⟨_, -, hw⟩ := bind_ok hw
      replace hw := jp_ok hw (by intro h'; injection h')
      replace hw := jp_ok hw (by intro h'; injection h')
      replace hw := jp_ok hw (by intro h'; injection h')
      first
        | (generalize (Json.obj kv).getObjVal? "scan_circuits" = sc at hw
           cases sc
           all_goals
             repeat (obtain ⟨_, -, hw⟩ := bind_ok hw)
             cases hw
             rfl
           done)
        | (dsimp only at hw
           split at hw
           all_goals
             repeat (obtain ⟨_, -, hw⟩ := bind_ok hw)
             cases hw
             rfl
           done)
  all_goals exact absurd hw (by intro h'; injection h')

theorem rowsWiresChecked : RowsWiresChecked := by
  intro j p c e pub prof hp h i au reg hau hreg
  obtain ⟨hca, hea, hfind⟩ := check_audits h
  obtain ⟨xs, read, hxs, hmap, haud⟩ := paramsOf_audits hp
  obtain ⟨hic, -⟩ := Array.getElem?_eq_some_iff.1 hreg
  obtain ⟨hia, -⟩ := Array.getElem?_eq_some_iff.1 hau
  have hie : i < e.audits.size := by omega
  have hx : e.audits[i]? = some e.audits[i] := Array.getElem?_eq_some_iff.2 ⟨hie, rfl⟩
  obtain ⟨f, hf⟩ := ok_of_findingOf (hfind _ (Array.mem_of_getElem? (auditChecks_getElem? hau hreg hx)))
  obtain ⟨hprog, hrec⟩ := auditCheck_ok hf
  have hsize := NciCheck.Properties.SecurityProofs.BodiesOk.array_mapM_size _ xs read hmap
  have hir : i < read.size := by
    have h1 := hia
    rw [haud, Array.size_map] at h1
    exact h1
  have hix : i < xs.size := by omega
  have hget : auditOf xs[i] "an audit" = .ok au := by
    have hr : auditWith xs[i] "an audit" = .ok read[i] :=
      NciCheck.Properties.SecurityProofs.BodiesOk.array_mapM_get _ xs read hmap i hix hir
    have hau2 : read[i].1 = au := by
      have h2 := hau
      rw [haud, Array.getElem?_map, Array.getElem?_eq_getElem hir, Option.map_some] at h2
      exact Option.some.inj h2
    unfold auditOf
    rw [hr, ← hau2]
    try rfl
  obtain ⟨pj, s, prog, hpj, hs, hd, hpr, hfree⟩ := auditOf_ok hget
  refine ⟨xs, xs[i], pj, s, prog, hxs, Array.getElem?_eq_some_iff.2 ⟨hix, rfl⟩, hpj, hs, hd, ?_, ?_⟩
  · rw [← hpr]
    exact isStr_eq hprog
  · unfold FreeOf at hfree
    revert hfree hrec
    cases hr : au.record with
    | none => intro _ hfree; exact hfree
    | some dp =>
      obtain ⟨d, pins⟩ := dp
      intro hrec hfree
      obtain ⟨hcirc, hpins⟩ := hrec d pins rfl
      obtain ⟨record, order, pinsJ, r, sm, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10⟩ := hfree
      exact ⟨record, order, pinsJ, r, sm, h1, h2, h3, h4, h5, h6, h7, h8, h9, isStr_eq hcirc,
        fun q hq => ⟨hpins q hq, h10 q hq⟩⟩

theorem rowsHeld : RowsHeld := by
  intro j p c e pub prof hp h s sd scan hform
  exact ⟨publicJson_held (check_rows hform h), rfl, rowsCircuitHeld p c e pub prof h,
    rowsWiresChecked j p c e pub prof hp h⟩

/-! ## Witnesses -/

/-- `slotPeriod` under a registration that names no Program fails `RowsWiresCheckedAt`, whatever its parameters. -/
theorem slotPeriod_unnamed (j : Json) : ¬ RowsWiresCheckedAt j slotPeriod ⟨#[.null], #[], .empty⟩ := by
  intro h
  obtain ⟨_, _, _, _, _, _, _, _, _, _, hreg, -⟩ := h 0 slotAudit .null rfl rfl
  first
    | cases hreg
    | injection hreg

/-- **`RowsWiresChecked`'s honest lemma**: the check accepts a rows-form period of no audit, and a circuit-record
audit's registration that names its Program, binds its record and registers its pin's root passes each registration
step of the audit check (`auditCheck`). `paramsOf`'s acceptance of an audit is not exhibited: it decodes the audit's
Program with `Flock.Program.decode`, which reads `Lean.Json.parse`, a `partial def` the kernel can't unfold. -/
theorem rowsWiresChecked_honest :
    check ⟨#[], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", some "c", .rows slotSchedule .empty ⟨0, 0, 0, 0, 0⟩, 0⟩
        ⟨#[], #[], .empty⟩ ⟨#[], #[]⟩ =
      .accept (rowsTerms #[] ⟨0, 0, 0, 0, 0⟩ #[] #[] 1 0 #[]) { link := "", windows := 0 } ∧
    isStr (recordReg.getObjValD "program") "x" = true ∧
    isStr ((recordReg.getObjValD "window").getObjValD "circuit") "d" = true ∧ pinHeld recordReg ("k", "r") = true :=
  ⟨rowsCircuitHeld_honest.1, by decide +kernel, by decide +kernel, pinHeld_of (by decide +kernel)⟩

/-- **`RowsWiresChecked`'s nontrivial lemma**: a verifier that decodes every parameters to `slotPeriod` and accepts
every period accepts one whose audit's registration names no Program, where `RowsWiresCheckedAt` fails. -/
theorem rowsWiresChecked_nontrivial : ¬ RowsWiresCheckedOf acceptAllParams acceptAllRows := fun h =>
  slotPeriod_unnamed .null (h .null slotPeriod ⟨#[.null], #[], .empty⟩ ⟨#[], #[]⟩ _ _ rfl rfl)

/-- **`RowsHeld`'s honest lemma**: the check accepts a rows-form period of no audit, and its public's `held` names the
checks `heldNames` names. `paramsOf`'s acceptance of an audit is not exhibited, as in `rowsWiresChecked_honest`. -/
theorem rowsHeld_honest :
    check ⟨#[], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", some "c", .rows slotSchedule .empty ⟨0, 0, 0, 0, 0⟩, 0⟩
        ⟨#[], #[], .empty⟩ ⟨#[], #[]⟩ =
      .accept (rowsTerms #[] ⟨0, 0, 0, 0, 0⟩ #[] #[] 1 0 #[]) { link := "", windows := 0 } ∧
    (publicJson (rowsTerms #[] ⟨0, 0, 0, 0, 0⟩ #[] #[] 1 0 #[])).getObjValD "held" = Lean.toJson heldNames :=
  ⟨rowsCircuitHeld_honest.1, publicJson_held rfl⟩

/-- **`RowsHeld`'s nontrivial lemma**: a verifier that decodes every parameters to `slotPeriod`, a rows-form period,
and accepts every period accepts one whose audit's registration names no Program, where `RowsHeldAt`'s `free` fails. -/
theorem rowsHeld_nontrivial : ¬ RowsHeldOf acceptAllParams acceptAllRows := fun h =>
  slotPeriod_unnamed .null (h .null slotPeriod ⟨#[.null], #[], .empty⟩ ⟨#[], #[]⟩ _ _ rfl rfl slotSchedule .empty
    ⟨0, 0, 0, 0, 0⟩ rfl).2.2.2

end Properties.NetworkPeriod
