module

public import Proofs.Properties.Nci.RecordBlocks
public import Proofs.Properties.Nci.RecordDeclared

/-!
# What the record check holds a row by

The record check counts a row as θ for one of three reasons (`Record.Case.heldAt`). Each reason is a test the check
made; these lemmas say what each test gives, in the check's own terms.

Of the trusted input and the run (`front_trusted`, `trusted_run`): `front` reads the trusted input, whose log is
chained and whose run is one of the log's records, named by its digest; and it holds the run's `program` and its
window's `circuit` to the SHA-512s of the Program and of the record's text.

Of θ (`isTheta_root`, `case_theta`, `heldAt_theta`): a parameter the case holds as θ is registered in the log's first
record, its last entry of that name, under the root the run's own record commits it under; and the parameter of
every reason `heldAt` gives is θ.

Of a reader's roles:

* `layoutOf_key`, `layoutOf_reads`: a layout's key is none of its opened rows, and a layout reads its Definition's
  parameters and return type alone.
* `pinOf_some`: a pin `pinOf` gives is the template's closure digest in the descriptor, and one of the pins.
* `rolesOf_spec`: of the declarations `rolesOf` took, a template it lists with a pin has that pin, and its
  Definition has a layout; every opening by that reader has the layout's key, opened rows and verdict, and every table
  read by it is in the layout's root row (of the root) or domain id row (of the domain id).
* `read_pin`: `read` asks `rolesOf` with `pinOf` on the Program's descriptor and the instance's pins.

Of a held opening (`covered_spec`, `case_held`): its reader is one `rolesOf` lists, its table lies in a θ parameter,
and a table read of the table's root and one of its domain id, by the same reader, each cover its units.

`heldAt_opened` puts them together for a row held as an opening's port: the reader is pinned, by the pin the reason
carries; its Definition has a layout, of which the row's port is an opened row; and at the row's own unit, every row
of the layout's root port and of its domain id port is held as a table read of the same table, of its root and of its
domain id. An opening's port is the rectangle the opening declares (`opening_of_fill`), so the row's unit is one of
the opening's, which the two table reads cover; a row of a table read's rectangle is filled by it (`table_fill`).
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Lean (Json)
open Flock.Program (Prog Defn)
open NciCheck.Baseline (getD strAt digest lastWith Trusted trustedOf)
open NciCheck.Properties (Chained)
open NciCheck.Record (Case Front Layout front finish rolesOf layoutOf pinOf isTheta openingHeld covered)
open NciCheck.Record.SecurityProofs (read_front finish_text mem_of_lookup)
open NciCheck.RecordText (Row Tmpl Decl Fill Inputs Table TableRead Opening Held Declared inputsOf fillAt)
open NciCheck.Properties.SecurityProofs.BaselineSteps (forIn_each trustedOf_ok findIdx_eq)

/-! ## The trusted input and the run -/

/-- **`front` reads the trusted input and holds the run to the Program and the record's text**: the run's `program`
is the SHA-512 of the Program, and its window's `circuit` the SHA-512 of the record's text. -/
theorem front_trusted {j tr : Json} {f : Front} (h : front j tr = .ok f) :
    trustedOf tr = .ok f.T ∧ strAt f.T.log[f.T.run]! "program" = some (Flock.hexOf f.P.sha512) ∧
      strAt (getD f.T.log[f.T.run]! "window") "circuit" =
        some (Flock.hexOf (Flock.Sha512.hash f.i.record.toUTF8)) := by
  unfold front at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat (split at h <;> try cases h)
  rename_i hT hprog hcirc _ _ _ _
  exact ⟨hT, by simpa using hprog, by simpa using hcirc⟩

/-- **A trusted input's log is chained and holds its run**, which the trusted input names by its digest. -/
theorem trusted_run {tr : Json} {T : Trusted} (h : trustedOf tr = .ok T) :
    getD tr "log" = .arr T.log ∧ Chained T.log ∧ T.run < T.log.size ∧
      ∃ r, digest T.log[T.run]! = .ok r ∧ strAt tr "run" = some r := by
  obtain ⟨hlog, hch, hsz, hdig, ⟨r, hr, hidx⟩, -⟩ := trustedOf_ok tr T h
  obtain ⟨hlt, hrun⟩ := findIdx_eq _ _ _ hidx
  rw [hsz] at hlt
  exact ⟨hlog, hch, hlt, r, by rw [hdig _ hlt, hrun], hr⟩

/-! ## θ -/

/-- **A θ parameter is registered under the run's root**: the log's first record registers a value of its name (its
last entry of that name) under the root the run's own record commits it under. -/
theorem isTheta_root {T : Trusted} {leaves : Nat} {p : String} (h : isTheta T leaves p = true) :
    ∃ e R, lastWith T.log[0]! "registered" "value" p = some e ∧ strAt e "root" = some R ∧
      strAt (getD (getD T.log[T.run]! "window") "registered_roots") p = some R := by
  unfold isTheta at h
  split at h
  · rename_i e R he hR
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    exact ⟨e, R, he, h.1, hR⟩
  · cases h

/-- A parameter the case holds as θ is one `isTheta` passes, at some count of leaves. -/
theorem case_theta {f : Front} {pin : String → Option String} {c : Case} (h : finish f pin = .ok c) {p : String}
    (hp : c.isTheta p = true) : ∃ leaves, isTheta f.T leaves p = true := by
  obtain ⟨-, -, -, -, hth, -⟩ := finish_case h
  unfold Case.isTheta at hp
  rw [hth] at hp
  cases hl : (c.x.params.map (fun (p, leaves) => (p, isTheta f.T leaves p))).lookup p with
  | none => rw [hl] at hp; cases hp
  | some b =>
    rw [hl] at hp
    obtain ⟨⟨p', leaves⟩, -, he⟩ := List.mem_map.1 (mem_of_lookup _ _ _ hl)
    cases he
    exact ⟨leaves, hp⟩

/-- The parameter a reason names. -/
def paramOf : Held → String
  | .read p => p
  | .table _ p _ => p
  | .opened _ p _ => p

/-- **The parameter of every reason `heldAt` gives is θ.** -/
theorem heldAt_theta {c : Case} {r : Row} {hd : Held} (hh : c.heldAt r = some hd) : c.isTheta (paramOf hd) = true := by
  unfold Case.heldAt at hh
  split at hh
  · cases hh
  split at hh
  · split at hh
    · cases hh
      assumption
    · cases hh
  · split at hh
    · cases hh
      assumption
    · cases hh
  · split at hh
    · rename_i hc
      simp only [Bool.and_eq_true] at hc
      obtain ⟨pin, -, rfl⟩ := Option.map_eq_some_iff.1 hh
      exact hc.2
    · cases hh
  · cases hh

/-! ## A reader's roles -/

/-- **A layout's key is none of its opened rows.** -/
theorem layoutOf_key {d : Defn} {l : Layout} (h : layoutOf d = some l) : l.key ∉ l.ports := by
  unfold layoutOf at h
  repeat' split at h
  all_goals try (cases h; done)
  rename_i hc
  cases h
  simp only [Bool.and_eq_true, Bool.not_eq_true'] at hc
  intro hmem
  have := hc.2
  rw [List.contains_eq_mem] at this
  simp [hmem] at this

/-- **A layout reads its Definition's parameters and return type alone.** -/
theorem layoutOf_reads {d d' : Defn} (hp : d.params = d'.params) (hr : d.ret = d'.ret) : layoutOf d = layoutOf d' := by
  unfold layoutOf
  rw [hp, hr]

/-- **A pin is the template's closure digest in the descriptor, and one of the pins.** -/
theorem pinOf_some {desc : Json} {pins : Array String} {tid d : String} (h : pinOf desc pins tid = some d) :
    NciCheck.DefinitionDigest.closureDigest desc tid = .ok d ∧ d ∈ pins.toList := by
  unfold pinOf at h
  repeat' split at h
  all_goals try (cases h; done)
  rename_i d' hd hc
  cases h
  exact ⟨hd, by simpa using hc⟩

/-- Where a `mapM` in `Except` keeps each element's key and gives it what `g` gives its key, the result's lookup at a
key of the list is `g`'s. -/
theorem mapM_lookup {α β : Type} (f : String × α → Except String (String × β)) (g : String → Option β)
    (hf : ∀ p q, f p = .ok q → q.1 = p.1 ∧ g p.1 = some q.2) :
    ∀ (rs : List (String × α)) (ls : List (String × β)), rs.mapM f = .ok ls →
      ∀ (tid : String) (d : α), (tid, d) ∈ rs → ∃ l, ls.lookup tid = some l ∧ g tid = some l
  | [], _, _, _, _, hm => by cases hm
  | a :: rs, ls, h, tid, d, hm => by
    rw [List.mapM_cons] at h
    simp only [bind, Except.bind, pure, Except.pure] at h
    repeat' split at h
    all_goals try (cases h; done)
    rename_i b hb _ bs hbs
    cases h
    obtain ⟨hk, hg⟩ := hf a b hb
    rw [List.lookup_cons]
    by_cases he : (tid == b.1) = true
    · rw [he]
      rw [beq_iff_eq.1 he, hk]
      exact ⟨_, rfl, hg⟩
    · rw [Bool.not_eq_true] at he
      rw [he]
      rcases List.mem_cons.1 hm with hm | hm
      · subst hm
        simp [hk] at he
      · exact mapM_lookup f g hf rs bs hbs tid d hm

/-- **What `rolesOf` took of a pinned reader.** A template `rolesOf` lists with a pin has that pin, and its Definition
has a layout; every opening by that reader has the layout's key, opened rows and verdict, and every table read by it
is in the layout's root row (of the root) or its domain id row (of the domain id). -/
theorem rolesOf_spec {P : Prog} {ts : List Tmpl} {pin : String → Option String} {x : Inputs}
    {pins : List (String × String)} (h : rolesOf P ts pin x = .ok pins) {tid d : String}
    (hp : pins.lookup tid = some d) :
    pin tid = some d ∧ ∃ t l, ts.find? (·.tid == tid) = some t ∧ layoutOf (P.defs[t.defn]!) = some l ∧
      (∀ o ∈ x.openings, o.reader = tid → o.key = l.key ∧ o.ports = l.ports ∧ o.verdict = l.verdict) ∧
      ∀ rd ∈ x.roots, rd.reader = tid → rd.port = if rd.root then l.root else l.domain := by
  unfold rolesOf at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat' split at h
  all_goals try (cases h; done)
  rename_i layouts hlay _ _ hop _ _ hrt
  cases h
  have hmem := mem_of_lookup _ _ _ hp
  obtain ⟨a, -, ha⟩ := List.mem_filterMap.1 hmem
  have hpin : pin tid = some d := by
    cases hpa : pin a with
    | none => rw [hpa] at ha; cases ha
    | some d' => rw [hpa] at ha; cases ha; exact hpa
  obtain ⟨l, hl, hg⟩ := mapM_lookup _
    (fun tid => (ts.find? (·.tid == tid)).bind fun t => layoutOf (P.defs[t.defn]!)) (by
      rintro ⟨tid', d'⟩ q hq
      dsimp only at hq
      split at hq
      · cases hq
        rename_i l hl
        exact ⟨rfl, hl⟩
      · cases hq) _ _ hlay tid d hmem
  obtain ⟨t, ht, htl⟩ := Option.bind_eq_some_iff.1 hg
  refine ⟨hpin, t, l, ht, htl, ?_, ?_⟩
  · intro o ho hor
    obtain ⟨_, s, hs⟩ := forIn_each _ (by
      intro o' b s hs
      repeat' split at hs
      all_goals try (cases hs; done)
      all_goals exact ⟨_, (Except.ok.inj hs).symm⟩) _ _ _ hop o ho
    rw [hor, hl] at hs
    dsimp only at hs
    split at hs
    · cases hs
    · rename_i hc
      simp only [Bool.not_eq_true', Bool.not_eq_false, Bool.and_eq_true, beq_iff_eq] at hc
      exact ⟨hc.1.1, hc.1.2, hc.2⟩
  · intro rd hrd hrr
    obtain ⟨_, s, hs⟩ := forIn_each _ (by
      intro o' b s hs
      repeat' split at hs
      all_goals try (cases hs; done)
      all_goals exact ⟨_, (Except.ok.inj hs).symm⟩) _ _ _ hrt rd hrd
    rw [hrr, hl] at hs
    dsimp only at hs
    by_cases hc : (rd.port != if rd.root then l.root else l.domain) = true
    · rw [ite_eq_left hc] at hs
      cases hs
    · simpa using hc

/-- **`read` asks for the roles with the Program's descriptor and the instance's pins**: its case is `finish` of its
front with `pinOf`, on what the Program's text parses to. -/
theorem read_pin {j tr : Json} {c : Case} (h : Record.read j tr = .ok c) :
    ∃ desc, (∀ pj, Flock.JsonParse.parse c.f.i.program = .ok pj → desc = pj) ∧ front j tr = .ok c.f ∧
      finish c.f (pinOf desc c.f.i.pins) = .ok c := by
  unfold Record.read at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i f hf
    obtain ⟨rfl, -⟩ := finish_text h
    refine ⟨_, ?_, hf, h⟩
    intro pj hpj
    rw [hpj]

/-! ## A held opening -/

/-- What `covered` found: a table read of the opening's table, of its root or its domain id, by the opening's reader,
over the opening's units. -/
theorem covered_spec {x : Inputs} {o : Opening} {root : Bool} (h : covered x o root = true) :
    ∃ rd ∈ x.roots, rd.reader = o.reader ∧ rd.table = o.table ∧ rd.root = root ∧ rd.start ≤ o.start ∧
      o.start + o.units ≤ rd.start + rd.units := by
  unfold covered at h
  obtain ⟨rd, hrd, hc⟩ := List.any_eq_true.1 h
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hc
  exact ⟨rd, hrd, hc.1.1.1.1, hc.1.1.1.2, hc.1.1.2, hc.1.2, hc.2⟩

/-- **A held opening**: its reader is one `rolesOf` lists, its table lies in a θ parameter, and a table read of the
table's root and one of its domain id each cover it. -/
theorem case_held {f : Front} {pin : String → Option String} {c : Case} (h : finish f pin = .ok c) {i : Nat}
    (hi : c.held[i]?.getD false = true) :
    ∃ o, c.x.openings[i]? = some o ∧ (c.pins.lookup o.reader).isSome = true ∧
      (∃ t, c.x.tables.find? (·.name == o.table) = some t ∧ c.isTheta t.param = true) ∧
      covered c.x o true = true ∧ covered c.x o false = true := by
  obtain ⟨-, -, -, -, -, hheld⟩ := finish_case h
  rw [hheld] at hi
  simp only [List.getElem?_toArray, List.getElem?_map] at hi
  cases ho : c.x.openings[i]? with
  | none => rw [ho] at hi; cases hi
  | some o =>
    rw [ho] at hi
    simp only [Option.map_some, Option.getD_some] at hi
    unfold openingHeld at hi
    simp only [Bool.and_eq_true] at hi
    obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := hi
    refine ⟨o, rfl, h1, ?_, h3, h4⟩
    cases ht : c.x.tables.find? (·.name == o.table) with
    | none => rw [ht] at h2; cases h2
    | some t => rw [ht] at h2; exact ⟨t, rfl, h2⟩

/-- **A row held as an opening's port.** The reader is pinned, by the pin the reason carries; its Definition has a
layout, of which the row's port is an opened row; and at the row's own unit, every row of the layout's root port and
of its domain id port is held as a table read of the same table, of its root and of its domain id. -/
theorem heldAt_opened {f : Front} {pin : String → Option String} {c : Case} (hc : finish f pin = .ok c) {r : Row}
    {name p d : String} (hh : c.heldAt r = some (.opened name p d)) :
    pin r.tid = some d ∧ ∃ t l, f.ts.find? (·.tid == r.tid) = some t ∧ layoutOf (f.P.defs[t.defn]!) = some l ∧
      r.port ∈ l.ports ∧
      (∀ row < c.W.cutsOf r.tid l.root, c.heldAt ⟨r.tid, r.unit, l.root, row⟩ = some (.table name p true)) ∧
      ∀ row < c.W.cutsOf r.tid l.domain, c.heldAt ⟨r.tid, r.unit, l.domain, row⟩ = some (.table name p false) := by
  obtain ⟨-, -, hroles, -⟩ := finish_case hc
  obtain ⟨-, -, text, -, hin, hcuts, -⟩ := finish_text hc
  have hth := heldAt_theta hh
  unfold Case.heldAt at hh
  split at hh
  · cases hh
  split at hh
  · split at hh <;> cases hh
  · split at hh <;> cases hh
  rotate_left
  · cases hh
  rename_i name' param i hfill hdecl
  split at hh
  rotate_left
  · cases hh
  rename_i hi
  simp only [Bool.and_eq_true] at hi
  obtain ⟨pin', hpin', he⟩ := Option.map_eq_some_iff.1 hh
  cases he
  obtain ⟨o, ho, hor, hport, hu, hu'⟩ := opening_of_fill hin hfill
  rw [ho] at hpin'
  simp only [Option.bind_some] at hpin'
  simp only [Inputs.declares] at hdecl
  rw [ho] at hdecl
  simp only [Option.bind_some, Option.map_eq_some_iff] at hdecl
  obtain ⟨t', ht', he⟩ := hdecl
  cases he
  rw [hor] at hpin'
  obtain ⟨hpin, t, l, ht, hl, hops, hrts⟩ := rolesOf_spec hroles hpin'
  obtain ⟨o', ho', -, -, hc1, hc2⟩ := case_held hc hi.1
  rw [ho] at ho'
  cases ho'
  have key : ∀ root : Bool, covered c.x o root = true → ∀ port, port = (if root then l.root else l.domain) →
      ∀ row < c.W.cutsOf r.tid port,
        c.heldAt ⟨r.tid, r.unit, port, row⟩ = some (.table o.table t'.param root) := by
    intro root hcov port hport' row hrow
    obtain ⟨rd, hrd, h1, h2, h3, h4, h5⟩ := covered_spec hcov
    have hp := hrts rd hrd (h1.trans hor)
    rw [h3, ← hport'] at hp
    have hcut : c.W.cutsOf r.tid port = text.cutsOf rd.reader rd.port := by
      unfold Spot.Wires.Wiring.cutsOf
      rw [hcuts, h1, hor, hp]
    have hf := table_fill hin hrd (u := r.unit) (row := row) (by omega) (by omega) (by rw [← hcut]; exact hrow)
    rw [h1, hor, hp, h2, h3] at hf
    unfold Case.heldAt
    rw [hf]
    simp only [Inputs.declares, ht', Option.map_some]
    simp only [paramOf] at hth
    rw [ite_eq_left hth]
  exact ⟨hpin, t, l, ht, hl, (hops o (List.mem_of_getElem? ho) hor).2.1 ▸ hport, key true hc1 _ rfl,
    key false hc2 _ rfl⟩

end NciCheck.Properties.Record.SecurityProofs
