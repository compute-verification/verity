/-
What round 2's statement check says: the theorems of `Spot.SecondStage`'s `check` and its parts, that a pass holds the
interior record, the statements, the headers, the ports and the values to what the module's docstring says
(`check_ok` and what follows from it), and that while `FORM` withholds the accept no run passes (`check_refused`).
-/
module

public import Spot.SecondStage

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.SecondStage

open Lean (Json)

theorem mem_fail (c : Check) (code : String) (d : Json) (x : String) :
    x ∈ (c.fail code d).codes ↔ x ∈ c.codes ∨ x = code := by
  unfold Check.fail
  split
  · rename_i h
    constructor
    · exact Or.inl
    · rintro (hx | rfl)
      · exact hx
      · simpa using h
  · simp

theorem mem_fold (l : List (String × Json)) (c : Check) (x : String) :
    x ∈ (l.foldl (fun c x => c.fail x.1 x.2) c).codes ↔ x ∈ c.codes ∨ ∃ d, (x, d) ∈ l := by
  induction l generalizing c with
  | nil => simp
  | cons a l ih =>
    rw [List.foldl_cons, ih, mem_fail]
    constructor
    · rintro ((h | rfl) | ⟨d, hd⟩)
      · exact Or.inl h
      · exact Or.inr ⟨a.2, by simp⟩
      · exact Or.inr ⟨d, by simp [hd]⟩
    · rintro (h | ⟨d, hd⟩)
      · exact Or.inl (Or.inl h)
      · rcases List.mem_cons.1 hd with rfl | hd
        · exact Or.inl (Or.inr rfl)
        · exact Or.inr ⟨d, hd⟩

/-- A fold of `Check.fail` has no code only over no failure. -/
theorem fold_codes_empty (l : List (String × Json)) :
    (l.foldl (fun c x => c.fail x.1 x.2) ({} : Check)).codes = #[] ↔ l = [] := by
  constructor
  · intro h
    cases l with
    | nil => rfl
    | cons a l =>
      have hm := (mem_fold (a :: l) {} a.1).2 (Or.inr ⟨a.2, by simp⟩)
      rw [h] at hm
      simp at hm
  · rintro rfl
    rfl

theorem failures_ne_nil (own : Own) (ps : Array Part) (ss : List Json) : failures false own ps ss ≠ [] := by
  simp only [failures]
  split
  · simp
  · rename_i h
    intro he
    simp_all

/-- A design not derived fails a code of its own (`second-links` or `second-wires`). -/
theorem designFailures_ne_nil {ps : Array Part} {own : Own} (h : ¬designOk ps own = true) :
    designFailures ps own ≠ [] := by
  revert h
  unfold designOk designFailures
  cases linksOk ps own <;> cases wiresOk ps own <;> simp

/-- **A wired port is no linked port**: where the wires are derived over parts `ps` (`wiresOk`), no link names a
wire's reader port, so a session's read of it is the wire's (`readOk_wired`), never a link's. -/
theorem wiresOk_unlinked {ps : Array Part} {own : Own} (hok : wiresOk ps own = true) {w : Wire}
    (hw : w ∈ own.wires.getD #[]) : linkOf own w.template w.port = none := by
  unfold wiresOk at hok
  cases hws : own.wires with
  | none =>
    rw [hws] at hok
    cases hok
  | some ws =>
    simp only [hws, Option.getD_some] at hok hw
    have h1 := Array.all_eq_true_iff_forall_mem.1 hok w hw
    simp only [Bool.and_eq_true] at h1
    exact Option.isNone_iff_eq_none.1 h1.1.2.2

/-- **Off, round 2 never passes**: with the two-stage form refused, every run's check fails. -/
theorem check_refused (own : Own) (stmts : Option (Array Json)) : (check false own stmts).codes ≠ #[] := by
  simp only [check, Ne, fold_codes_empty]
  split
  · simp
  · split
    · simp
    · exact fun he => designFailures_ne_nil ‹_› (List.append_eq_nil_iff.1 he).2
  · split
    · simp [failures_ne_nil]
    · exact fun he => designFailures_ne_nil ‹_› (List.append_eq_nil_iff.1 he).2

/-- **By position**: with no `second-order`, statement `i`, one for part `p` and no earlier statement's, is for the
part at its position. -/
theorem orderFailures_nil {ps : Array Part} {ss : List Json} (h : orderFailures ps ss = []) {i : Nat}
    (hi : i < ss.length) {s : Stmt} {p : Part} (hs : Stmt.ofJson? (ss.getD i .null) = some s)
    (hp : ps.find? (·.template == s.template) = some p) (hd : (templates (ss.take i)).contains s.template = false) :
    ps[i]?.map (·.template) = some s.template := by
  have h' := List.filterMap_eq_nil_iff.1 h i (List.mem_range.2 hi)
  simp only [hs] at h'
  by_cases he : ps[i]?.map (·.template) = some s.template
  · exact he
  · have hb : (ps[i]?.map (·.template) != some s.template) = true := bne_iff_ne.2 he
    simp only [hp, hd, hb, Option.isSome_some] at h'
    simp at h'

/-- With no `second-interior-order`, the interior record names the replay sample in its order. -/
theorem interiorFailures_nil {own : Own} (h : interiorFailures own = []) : interiorOrderOk own = true := by
  unfold interiorFailures at h
  have h := (List.append_eq_nil_iff.1 (List.append_eq_nil_iff.1 (List.append_eq_nil_iff.1
    (List.append_eq_nil_iff.1 h).1).1).1).1
  split at h
  · assumption
  · simp at h

/-- With no `second-interior-names`, no value the interior registration registers is of a name the boundary
registration's record has. -/
theorem interiorFailures_names {own : Own} (h : interiorFailures own = []) : namesOk own = true := by
  unfold interiorFailures at h
  have h := (List.append_eq_nil_iff.1 (List.append_eq_nil_iff.1 (List.append_eq_nil_iff.1
    (List.append_eq_nil_iff.1 h).1).1).1).2
  split at h
  · assumption
  · simp at h

/-- With no `second-return`, no value the interior registration registers is of a name the replay block's return has
in the boundary registration's record. -/
theorem interiorFailures_return {own : Own} (h : interiorFailures own = []) : returnsOk own = true := by
  unfold interiorFailures at h
  have h := (List.append_eq_nil_iff.1 (List.append_eq_nil_iff.1 (List.append_eq_nil_iff.1 h).1).1).2
  split at h
  · assumption
  · simp at h

/-- With no `second-schedule`, the boundary registration's record commits no schedule, or one of its replay blocks. -/
theorem interiorFailures_schedule {own : Own} (h : interiorFailures own = []) :
    Schedule.failure own.boundary = none := by
  unfold interiorFailures at h
  have h := (List.append_eq_nil_iff.1 (List.append_eq_nil_iff.1 h).1).2
  split at h
  · simp at h
  · assumption

/-- With no `second-inputs`, every value the interior registration registers is one a replay block's input may read. -/
theorem interiorFailures_inputs {own : Own} (h : interiorFailures own = []) : inputsFailure own = none := by
  unfold interiorFailures at h
  have h := (List.append_eq_nil_iff.1 h).2
  split at h
  · simp at h
  · assumption

/-- **A passing `second-interior-order`**: the interior record's `window.replay_sample` is a list whose entries are the
verifier's own replay sample's blocks, in its order. -/
theorem interiorOrderOk_sample {own : Own} (h : interiorOrderOk own = true) :
    ∃ bs, getD (getD own.interior "window") "replay_sample" = .arr bs ∧
      bs.toList.map nat? = own.sample.toList.map some := by
  unfold interiorOrderOk at h
  split at h
  · rename_i bs hb
    exact ⟨bs, hb, by simpa using h⟩
  · simp at h

/-- **What a passing check says**: the form is allowed, the interior record names the replay sample in its order
(`interiorOrderOk_sample`), the interior program's strata make parts, the replay block's Definition makes links, each
for a part's template, and wires, each between parts' templates, each statement is one for a part's template, no
earlier statement's, the part at its position (`orderFailures_nil`), and passes every check at its position
(`stmtCodes_nil`), every part has a statement, and the reads the verifiers report are the statements'. -/
theorem check_ok {allow : Bool} {own : Own} {ss : Array Json} (h : (check allow own (some ss)).codes = #[]) :
    allow = true ∧ interiorOrderOk own = true ∧ ∃ ps, own.parts = some ps ∧ linksOk ps own = true ∧
      wiresOk ps own = true ∧
      (∀ i < ss.size, ∃ s p, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧
        ps.find? (·.template == s.template) = some p ∧ (templates (ss.toList.take i)).contains s.template = false ∧
        ps[i]?.map (·.template) = some s.template ∧
        stmtCodes own (byTemplate own.interior) (posOf ps s.template) p s = []) ∧
      (∀ p ∈ ps.toList, (templates ss.toList).contains p.template = true) ∧
      Record.sameDict (own.reported.getD #[]) (statementReads ss.toList) = true := by
  simp only [check, fold_codes_empty] at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · exact absurd (List.append_eq_nil_iff.1 h).2 (designFailures_ne_nil ‹_›)
  · rename_i ps _ hp hs
    cases hs
    by_cases hl : designOk ps own = true
    case neg => rw [if_neg hl] at h; exact absurd (List.append_eq_nil_iff.1 h).2 (designFailures_ne_nil hl)
    rw [if_pos hl] at h
    obtain ⟨hl, hw⟩ : linksOk ps own = true ∧ wiresOk ps own = true := by simpa [designOk] using hl
    simp only [failures] at h
    split at h
    · simp at h
    · rename_i hne
      simp only [h, List.isEmpty_nil, Bool.true_and, Bool.not_eq_true', Bool.not_eq_false] at hne
      simp only [List.append_eq_nil_iff, List.flatMap_eq_nil_iff, List.mem_range, and_assoc] at h
      obtain ⟨hint, hst, hord, hcov, hrep, -⟩ := h
      refine ⟨by simpa using hne, interiorFailures_nil hint, ps, hp, hl, hw, ?_, ?_, ?_⟩
      · intro i hi
        have hi' : i < ss.toList.length := by simpa using hi
        have := hst i hi'
        simp only [stmtFailures] at this
        split at this
        · simp at this
        · rename_i s hs
          split at this
          · rename_i p hp'
            split at this
            · simp at this
            · rename_i hdup
              have hd : (templates (ss.toList.take i)).contains s.template = false := by simpa using hdup
              exact ⟨s, p, hs, hp', hd, orderFailures_nil hord hi' hs hp' hd, by simpa using this⟩
          · simp at this
      · intro p hp'
        have := List.filterMap_eq_nil_iff.1 hcov p hp'
        split at this
        · simp at this
        · rename_i hc
          simpa using hc
      · unfold reportedFailures at hrep
        split at hrep
        · assumption
        · simp at hrep

/-- A passing check has no failure of the interior record's own (`interiorFailures`). -/
theorem check_ok_interior {allow : Bool} {own : Own} {ss : Array Json} (h : (check allow own (some ss)).codes = #[]) :
    interiorFailures own = [] := by
  simp only [check, fold_codes_empty] at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · exact absurd (List.append_eq_nil_iff.1 h).2 (designFailures_ne_nil ‹_›)
  · rename_i ps _ hp hs
    cases hs
    by_cases hl : designOk ps own = true
    case neg => rw [if_neg hl] at h; exact absurd (List.append_eq_nil_iff.1 h).2 (designFailures_ne_nil hl)
    rw [if_pos hl] at h
    simp only [failures] at h
    split at h
    · simp at h
    · simp only [List.append_eq_nil_iff, List.flatMap_eq_nil_iff, List.mem_range, and_assoc] at h
      exact h.1

/-- **A passing `second-inputs`**: under a passing check, every value the interior registration registers is one a
replay block's input may read: one the verifier holds without the prover, or the opened row of a read against a root
round 1 committed (`inputsFailure`). -/
theorem check_ok_inputs {allow : Bool} {own : Own} {ss : Array Json} (h : (check allow own (some ss)).codes = #[]) :
    inputsFailure own = none :=
  interiorFailures_inputs (check_ok_interior h)

/-- A passing statement's checks: an accepted verdict of record checked its header, registered reads and sample, every
one that did checked its session as one table, and its header at its position, population, roots, sample and
registered reads, every port a link or a wire names among them, every port its header shares rows at read registered,
and its registered reads its template's reference statement's but for the run's fields. -/
theorem stmtCodes_nil {own : Own} {named : Bool} {k : Nat} {p : Part} {s : Stmt} (h : stmtCodes own named k p s = []) :
    verifiedOk own s = true ∧ tablesOk own s = true ∧
      headerOk own named k s.template s.header = true ∧ populationOk own.sample.size p s.header = true ∧
      rootsOk named own.interior s.template s.header ((entries (getD s.registered "ports")).map (·.1)) = true ∧
      jeq s.draw (ownSample own.sample.size p own.drawn) = true ∧
      (entries (getD s.registered "ports")).all (fun (port, e) => readOk own named p port e) = true ∧
      linkedRead own p.template (entries (getD s.registered "ports")) = true ∧
      wiredRead own p.template (entries (getD s.registered "ports")) = true ∧ refsOk s = true ∧
      valuesOk own s = true := by
  simpa [stmtCodes, List.filterMap_eq_nil_iff, and_assoc] using h

/-- The reference header `refOf` gives template `t` is one the auditor's params give it. -/
theorem refOf_mem {own : Own} {t : String} {r : Json} (h : refOf own t = some r) : (t, r) ∈ own.refs := by
  unfold refOf at h
  cases hf : own.refs.find? (·.1 == t) with
  | none => rw [hf] at h; simp at h
  | some e =>
    rw [hf] at h
    obtain ⟨t', r'⟩ := e
    have ht : t' = t := by simpa using Array.find?_some hf
    have hr : r' = r := by simpa using h
    subst ht hr
    exact Array.mem_of_find?_eq_some hf

/-- **A passing `second-header`** names the reference header the auditor's params give template `t`, which header `h`
of a statement for the `k`-th part is, placed in the run (`atRun`), but for the run's fields. -/
theorem headerOk_ref {own : Own} {named : Bool} {k : Nat} {t : String} {h : Json}
    (hh : headerOk own named k t h = true) :
    ∃ r r', (t, r) ∈ own.refs ∧ atRun own named k t r = some r' ∧ jeq (blankRun h) (blankRun r') = true := by
  unfold headerOk at hh
  cases hf : refOf own t with
  | none => simp [hf] at hh
  | some r =>
    cases ha : atRun own named k t r with
    | none => simp [hf, ha] at hh
    | some r' =>
      simp only [hf, ha] at hh
      exact ⟨r, r', refOf_mem hf, ha, hh⟩

/-- **Every header is its template's reference placed in the run** but for the run's fields (`RUN_FIELDS`): in a
passing check, the `i`-th statement is for the template of the part at its position, and its header with those fields
blanked is the reference header the auditor's params give that template placed in the run at the template's position
(`atRun` at `posOf`: its `set` the run's `served/<run>/m<k>`, its `frame_v3.bindings` the verifier's), blanked the same
(`second-header`). -/
theorem check_ok_header {allow : Bool} {own : Own} {ss : Array Json} (h : (check allow own (some ss)).codes = #[]) :
    ∃ ps, own.parts = some ps ∧ ∀ i < ss.size, ∃ s r r', Stmt.ofJson? (ss.toList.getD i .null) = some s ∧
      ps[i]?.map (·.template) = some s.template ∧ (s.template, r) ∈ own.refs ∧
      atRun own (byTemplate own.interior) (posOf ps s.template) s.template r = some r' ∧
      jeq (blankRun s.header) (blankRun r') = true := by
  obtain ⟨-, -, ps, hp, -, -, hst, -, -⟩ := check_ok h
  refine ⟨ps, hp, fun i hi => ?_⟩
  obtain ⟨s, p, hs, -, -, hi', hc⟩ := hst i hi
  obtain ⟨r, r', hr, ha, hj⟩ := headerOk_ref (stmtCodes_nil hc).2.2.1
  exact ⟨s, r, r', hs, hi', hr, ha, hj⟩

/-- **A passing `second-refs`**: every port statement `s`'s header shares rows at is one of its registered reads. -/
theorem refsOk_read {s : Stmt} (h : refsOk s = true) {port : String} {c : Json}
    (hp : (port, c) ∈ entries (getD s.header "shared_rows")) :
    ∃ e, (port, e) ∈ entries (getD s.registered "ports") := by
  dsimp only [refsOk] at h
  have h' : (entries (getD s.registered "ports")).any (·.1 == port) = true :=
    Array.all_eq_true_iff_forall_mem.mp h _ hp
  obtain ⟨⟨q, e⟩, hm, hq⟩ := Array.any_eq_true'.mp h'
  obtain rfl : q = port := by simpa using hq
  exact ⟨e, hm⟩

/-- **Every port a header shares rows at is a read round 2 holds**: in a passing check, every port the `i`-th
statement's header shares rows at (`shared_rows`) is one of its registered reads, each `readOk` at its part: for a port
a link names, the boundary port it names, each slot reading only its cell's replay block's rows; else its own port of
the interior record, each slot reading only its own rows, or a value either registration registers. The session's
verifier holds each instance's ref at such a port to the row at the position the read gives its unit
(`Flock.Registered.check`), so no ref there points a sampled proof block at another unit's rows. -/
theorem check_ok_refs {allow : Bool} {own : Own} {ss : Array Json} (h : (check allow own (some ss)).codes = #[]) :
    ∃ ps, own.parts = some ps ∧ ∀ i < ss.size, ∃ s p, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧
      ps.find? (·.template == s.template) = some p ∧ ∀ port c, (port, c) ∈ entries (getD s.header "shared_rows") →
        ∃ e, (port, e) ∈ entries (getD s.registered "ports") ∧ readOk own (byTemplate own.interior) p port e = true := by
  obtain ⟨-, -, ps, hp, -, -, hst, -, -⟩ := check_ok h
  refine ⟨ps, hp, fun i hi => ?_⟩
  obtain ⟨s, p, hs, hf, -, -, hc⟩ := hst i hi
  obtain ⟨-, -, -, -, -, -, hr, -, -, hrefs, -⟩ := stmtCodes_nil hc
  refine ⟨s, p, hs, hf, fun port c hm => ?_⟩
  obtain ⟨e, he⟩ := refsOk_read hrefs hm
  exact ⟨e, he, Array.all_eq_true_iff_forall_mem.mp hr _ he⟩

/-- **A wired read reads its writer unit's value** (one value a gate, one leaf a gate): at a port a wire names and no
link does, a read `readOk` passes is of the wire's writer's output port of the interior record, at that port's root,
domain and leaves, the writer's `Q·n_w` slots' (`pw`, the writer's part, `n_w` its rows' `per`), each slot `u` of part
`p` reading only the row of its unit's writer unit in its cell's replay block (`wiredRow`), as the port's rows give it
(`r`, the verifier's own). -/
theorem readOk_wired {own : Own} {named : Bool} {p : Part} {port : String} {e : Json} {w : Wire}
    (hl : linkOf own p.template port = none) (hw : wireOf own p.template port = some w)
    (h : readOk own named p port e = true) :
    ∃ x pw r, getD e "value" = .str (portOf named w.writer w.wport) ∧
      (portOf named w.writer w.wport, x) ∈ Record.byPort own.interior ∧ writerOf own w = some pw ∧
      rowsOf own p.template port = some r ∧ pw.n = r.per ∧ sameAt e x = true ∧
      placedOk own.sample.size p e (own.sample.size * r.per) (wiredRow p r) = true := by
  unfold readOk at h
  rw [hl, hw] at h
  cases hv : getD e "value" with
  | str v =>
    rw [hv] at h
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨rfl, h⟩ := h
    cases hf : (Record.byPort own.interior).find? (·.1 == portOf named w.writer w.wport) with
    | none => rw [hf] at h; simp at h
    | some kx =>
      obtain ⟨k, x⟩ := kx
      cases hp : writerOf own w with
      | none => rw [hf, hp] at h; simp at h
      | some pw =>
        cases hr : rowsOf own p.template port with
        | none => rw [hf, hp, hr] at h; simp at h
        | some r =>
          rw [hf, hp, hr] at h
          have hk : k = portOf named w.writer w.wport := by simpa using Array.find?_some hf
          subst hk
          simp only [Bool.and_eq_true, beq_iff_eq] at h
          exact ⟨x, pw, r, rfl, Array.mem_of_find?_eq_some hf, rfl, rfl, h.1.2, h.1.1, h.2⟩
  | _ => rw [hv] at h; simp at h

/-- **Every wired port reads its writer unit's value**: in a passing check, the `i`-th statement reads registered every
port a wire names for its template (`wiredRead`), and each of its reads at a port a wire names and no link does is of
that wire's writer's output port of the interior record, each slot at its unit's writer unit's row in its cell's
replay block (`readOk_wired`). -/
theorem check_ok_wired {allow : Bool} {own : Own} {ss : Array Json} (h : (check allow own (some ss)).codes = #[]) :
    ∃ ps, own.parts = some ps ∧ ∀ i < ss.size, ∃ s p, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧
      ps.find? (·.template == s.template) = some p ∧
      wiredRead own s.template (entries (getD s.registered "ports")) = true ∧
      ∀ port e w, (port, e) ∈ entries (getD s.registered "ports") → linkOf own s.template port = none →
        wireOf own s.template port = some w →
        ∃ x pw r, getD e "value" = .str (portOf (byTemplate own.interior) w.writer w.wport) ∧
          (portOf (byTemplate own.interior) w.writer w.wport, x) ∈ Record.byPort own.interior ∧
          writerOf own w = some pw ∧ rowsOf own p.template port = some r ∧ pw.n = r.per ∧ sameAt e x = true ∧
          placedOk own.sample.size p e (own.sample.size * r.per) (wiredRow p r) = true := by
  obtain ⟨-, -, ps, hp, -, -, hst, -, -⟩ := check_ok h
  refine ⟨ps, hp, fun i hi => ?_⟩
  obtain ⟨s, p, hs, hf, -, -, hc⟩ := hst i hi
  obtain ⟨-, -, -, -, -, -, hr, -, hwr, -, -⟩ := stmtCodes_nil hc
  have hpt : p.template = s.template := by simpa using Array.find?_some hf
  rw [hpt] at hwr
  refine ⟨s, p, hs, hf, hwr, fun port e w he hl hw => ?_⟩
  rw [← hpt] at hl hw
  exact readOk_wired hl hw (Array.all_eq_true_iff_forall_mem.mp hr _ he)

/-- **The build of record holds the wired key's switch off.** -/
theorem wiredKey_off : WIRED_KEY = false := rfl

/-- **With the switch off, a wired key opens no row**: where a wire names a read's key, `openedRow` fails, so round 2
refuses the row as it did before the switch (`inputsWhy`: `KEY_WHY`). -/
theorem openedRow_wired_off {own : Own} {registered : Array String}
    {mine : Array (String × Array (Array (String × String)))} {x : Link} {w : Wire}
    (hk : own.wiredKey = false) (hw : wireOf own x.template OPENING_KEY = some w) :
    openedRow own registered mine x = false := by
  simp [openedRow, hw, hk]

/-- **A wired key round 2 takes is whole**: where a wire names a read's key and `openedRow` passes, the switch is on and
the key is whole (`keyFailure`): one wire, no link, of the writer row's words, the writer's part first, each unit's
writer row among the writer's units. -/
theorem openedRow_wired {own : Own} {registered : Array String}
    {mine : Array (String × Array (Array (String × String)))} {x : Link} {w : Wire}
    (hw : wireOf own x.template OPENING_KEY = some w) (h : openedRow own registered mine x = true) :
    own.wiredKey = true ∧ keyFailure own x.template = none := by
  cases hk : own.wiredKey
  · simp [openedRow, hw, hk] at h
  · cases hf : keyFailure own x.template
    · exact ⟨rfl, rfl⟩
    · simp [openedRow, hw, hk, hf] at h

/-- The reference reads `refReadsOf` gives template `t` are ones the auditor's params give it. -/
theorem refReadsOf_mem {own : Own} {t : String} {r : Json} (h : refReadsOf own t = some r) :
    (t, r) ∈ own.refReads := by
  unfold refReadsOf at h
  cases hf : own.refReads.find? (·.1 == t) with
  | none => rw [hf] at h; simp at h
  | some e =>
    rw [hf] at h
    obtain ⟨t', r'⟩ := e
    have ht : t' = t := by simpa using Array.find?_some hf
    have hr : r' = r := by simpa using h
    subst ht hr
    exact Array.mem_of_find?_eq_some hf

/-- **A passing `second-values`** names the registered reads the auditor's params give statement `s`'s template's
reference statement, which `s`'s are but for the run's fields. -/
theorem valuesOk_ref {own : Own} {s : Stmt} (h : valuesOk own s = true) :
    ∃ r, (s.template, r) ∈ own.refReads ∧
      jeq (blankReads own s.template s.registered) (blankReads own s.template r) = true := by
  unfold valuesOk at h
  cases hf : refReadsOf own s.template with
  | none => simp [hf] at h
  | some r =>
    simp only [hf] at h
    exact ⟨r, refReadsOf_mem hf, h⟩

/-- **Every statement reads registered what its template's reference statement reads**: in a passing check, the `i`-th
statement's registered reads, each port's run fields blanked (`blankReads`: every port's root, domain and leaves, and
the rows of a port whose rows the run places, `runPlacesRows`), are those the auditor's params give its template's
reference statement, blanked the same (`second-values`). The ports it reads registered, the value each reads, and the
rows at which it reads a value a registration registers are fixed before the run. -/
theorem check_ok_values {allow : Bool} {own : Own} {ss : Array Json} (h : (check allow own (some ss)).codes = #[]) :
    ∀ i < ss.size, ∃ s r, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧ (s.template, r) ∈ own.refReads ∧
      jeq (blankReads own s.template s.registered) (blankReads own s.template r) = true := by
  obtain ⟨-, -, _ps, -, -, -, hst, -, -⟩ := check_ok h
  intro i hi
  obtain ⟨s, p, hs, -, -, -, hc⟩ := hst i hi
  obtain ⟨-, -, -, -, -, -, -, -, -, -, hv⟩ := stmtCodes_nil hc
  obtain ⟨r, hr, hj⟩ := valuesOk_ref hv
  exact ⟨s, r, hs, hr, hj⟩

/-- **A verified entry is a verdict's**: each of `verifiedOf verdicts` is read off an accepted verdict of record among
`verdicts`, its header's digest that verdict's `public_header_sha512`, its registered reads' its `registered_sha512`,
its sample its `unit_draw` and its tables its `session_tables`. -/
theorem mem_verifiedOf {verdicts : Array (String × Json)} {c : Checked} (h : c ∈ verifiedOf verdicts) :
    ∃ x ∈ verdicts, getD x.2 "public_header_sha512" = .str c.header ∧ truthy (getD x.2 "of_record") = true ∧
      jeq (getD x.2 "accepted") (.bool true) = true ∧ c.registered = getD x.2 "registered_sha512" ∧
      c.draw = getD x.2 "unit_draw" ∧ c.tables = getD x.2 "session_tables" := by
  obtain ⟨⟨k, v⟩, hm, hv⟩ := Array.mem_filterMap.mp h
  refine ⟨(k, v), hm, ?_⟩
  dsimp only at hv ⊢
  generalize hg : getD v "public_header_sha512" = g at hv
  cases g with
  | str hdr =>
    dsimp only at hv
    split at hv
    · next ha =>
      cases hv
      simp only [Bool.and_eq_true] at ha
      exact ⟨rfl, ha.1, ha.2, rfl, rfl, rfl⟩
    · cases hv
  | _ => simp at hv

/-- **What `second-verified` says**: a statement passing it is, by what a verdict names of it (`Stmt.checked`), one an
accepted verdict of record checked (`mem_verifiedOf`): its header's digest, its registered reads' and its sample are
that verdict's. -/
theorem verifiedOk_some {own : Own} {s : Stmt} (h : verifiedOk own s = true) :
    ∃ c, s.checked = some c ∧ ∃ v ∈ own.verified, v.header = c.header ∧ jeq v.registered c.registered = true ∧
      jeq v.draw c.draw = true := by
  unfold verifiedOk at h
  cases hc : s.checked with
  | none => simp [hc] at h
  | some c =>
    have h' : ∃ (i : Nat) (_ : i < own.verified.size), (own.verified[i].header = c.header ∧
        jeq own.verified[i].registered c.registered = true) ∧ jeq own.verified[i].draw c.draw = true := by
      simpa [hc, Bool.and_eq_true, beq_iff_eq] using h
    obtain ⟨i, hi, ⟨h1, h2⟩, h3⟩ := h'
    exact ⟨c, rfl, own.verified[i], Array.getElem_mem hi, h1, h2, h3⟩

/-- **What `second-tables` says**: every verified entry that names a statement passing it as `Stmt.checked` does, its
header's digest, its registered reads' and its sample, set its session up as one table (`mem_verifiedOf`: that verdict's
`session_tables` is 1). -/
theorem tablesOk_one {own : Own} {s : Stmt} {c : Checked} (h : tablesOk own s = true) (hc : s.checked = some c)
    {v : Checked} (hv : v ∈ own.verified) (h1 : v.header = c.header) (h2 : jeq v.registered c.registered = true)
    (h3 : jeq v.draw c.draw = true) : jeq v.tables ONE_TABLE = true := by
  unfold tablesOk at h
  simp only [hc] at h
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hv
  have := Array.all_eq_true.1 h i hi
  simpa [h1, h2, h3] using this

/-- A passing statement of only its part's sampled proof blocks: `stmtCodes_nil`'s checks, with its header its
template's reference's placed at the part's sampled slots (`sampledHeaderOk`), its population those slots
(`sampledOk`) and its session's sample every unit of it (`wholeSample`) where `stmtCodes_nil` has `headerOk`,
`populationOk` and `ownSample`. -/
theorem sampledCodes_nil {own : Own} {named : Bool} {k : Nat} {p : Part} {s : Stmt}
    (h : sampledCodes own named k p s = []) :
    verifiedOk own s = true ∧ tablesOk own s = true ∧
      sampledHeaderOk own named k s.template s.header (sampledSlots own.sample.size p own.drawn)
        (entries (getD s.registered "ports")) = true ∧
      sampledOk own.sample.size p own.drawn s.header = true ∧
      rootsOk named own.interior s.template s.header ((entries (getD s.registered "ports")).map (·.1)) = true ∧
      jeq s.draw (wholeSample (sampledSlots own.sample.size p own.drawn).size) = true ∧
      (entries (getD s.registered "ports")).all (fun (port, e) => readOk own named p port e) = true ∧
      linkedRead own p.template (entries (getD s.registered "ports")) = true ∧
      wiredRead own p.template (entries (getD s.registered "ports")) = true ∧ refsOk s = true ∧
      valuesOk own s = true := by
  simpa [sampledCodes, List.filterMap_eq_nil_iff, and_assoc] using h

/-- The checks a passing statement of either shape passes alike: `stmtCodes_nil`'s but its header's, its population's
and its session's sample's. -/
theorem codes_nil_shared {own : Own} {named : Bool} {k : Nat} {p : Part} {s : Stmt}
    (h : (if sampledOnly own.sample.size p own.drawn s.header then sampledCodes else stmtCodes) own named k p s = []) :
    verifiedOk own s = true ∧ tablesOk own s = true ∧
      rootsOk named own.interior s.template s.header ((entries (getD s.registered "ports")).map (·.1)) = true ∧
      (entries (getD s.registered "ports")).all (fun (port, e) => readOk own named p port e) = true ∧
      linkedRead own p.template (entries (getD s.registered "ports")) = true ∧
      wiredRead own p.template (entries (getD s.registered "ports")) = true ∧ refsOk s = true ∧
      valuesOk own s = true := by
  by_cases hs : sampledOnly own.sample.size p own.drawn s.header = true
  · rw [if_pos hs] at h
    obtain ⟨h1, h2, -, -, h5, -, h7, h8, h9, h10, h11⟩ := sampledCodes_nil h
    exact ⟨h1, h2, h5, h7, h8, h9, h10, h11⟩
  · rw [if_neg hs] at h
    obtain ⟨h1, h2, -, -, h5, -, h7, h8, h9, h10, h11⟩ := stmtCodes_nil h
    exact ⟨h1, h2, h5, h7, h8, h9, h10, h11⟩

/-- An entry of a list at a position it has is one of the list's. -/
theorem getD_mem_of_lt {α : Type} (l : List α) (d : α) {i : Nat} (h : i < l.length) : l.getD i d ∈ l := by
  induction l generalizing i with
  | nil => simp at h
  | cons a l ih =>
    cases i with
    | zero =>
      have h0 : (a :: l).getD 0 d = a := rfl
      rw [h0]
      simp
    | succ n =>
      have hm : l.getD n d ∈ l := ih (by simpa using h)
      have hn : (a :: l).getD (n + 1) d = l.getD n d := rfl
      rw [hn]
      exact List.mem_cons_of_mem _ hm

/-- With no statement of only its part's sampled proof blocks among `ss` (`anySampled`), none of them is one. -/
theorem anySampled_false {own : Own} {ps : Array Part} {ss : List Json} (h : anySampled own ps ss = false)
    {j : Json} (hj : j ∈ ss) {s : Stmt} {p : Part} (hs : Stmt.ofJson? j = some s)
    (hp : ps.find? (·.template == s.template) = some p) :
    sampledOnly own.sample.size p own.drawn s.header = false := by
  unfold anySampled at h
  have h' := List.any_eq_false.1 h j hj
  simpa [hs, hp] using h'

/-- **What a passing `checkAny` says**: what `check_ok` says of a passing `check`, each statement passing the checks
of its own shape at its position: `sampledCodes` where it is of only its part's sampled proof blocks (`sampledOnly`,
`sampledCodes_nil`), `stmtCodes` otherwise (`stmtCodes_nil`). -/
theorem checkAny_ok {allow : Bool} {own : Own} {ss : Array Json} (h : (checkAny allow own (some ss)).codes = #[]) :
    allow = true ∧ interiorOrderOk own = true ∧ ∃ ps, own.parts = some ps ∧ linksOk ps own = true ∧
      wiresOk ps own = true ∧
      (∀ i < ss.size, ∃ s p, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧
        ps.find? (·.template == s.template) = some p ∧ (templates (ss.toList.take i)).contains s.template = false ∧
        ps[i]?.map (·.template) = some s.template ∧
        (if sampledOnly own.sample.size p own.drawn s.header then sampledCodes else stmtCodes)
          own (byTemplate own.interior) (posOf ps s.template) p s = []) ∧
      (∀ p ∈ ps.toList, (templates ss.toList).contains p.template = true) ∧
      Record.sameDict (own.reported.getD #[]) (statementReads ss.toList) = true := by
  cases hp : own.parts with
  | none =>
    have hc : checkAny allow own (some ss) = check allow own (some ss) := by
      unfold checkAny
      simp only [hp]
    rw [hc] at h
    obtain ⟨-, -, ps, hp', -⟩ := check_ok h
    rw [hp] at hp'
    cases hp'
  | some ps =>
    by_cases hb : (designOk ps own && anySampled own ps ss.toList) = true
    · have hc : checkAny allow own (some ss) =
          (failuresSampled allow own ps ss.toList).foldl (fun (c : Check) x => c.fail x.1 x.2) {} := by
        unfold checkAny
        simp only [hp]
        exact if_pos hb
      rw [hc, fold_codes_empty] at h
      have hb' : designOk ps own = true ∧ anySampled own ps ss.toList = true := by simpa using hb
      obtain ⟨hl, hw⟩ : linksOk ps own = true ∧ wiresOk ps own = true := by simpa [designOk] using hb'.1
      simp only [failuresSampled] at h
      split at h
      · simp at h
      · rename_i hne
        simp only [h, List.isEmpty_nil, Bool.true_and, Bool.not_eq_true', Bool.not_eq_false] at hne
        simp only [List.append_eq_nil_iff, List.flatMap_eq_nil_iff, List.mem_range, and_assoc] at h
        obtain ⟨hint, hst, hord, hcov, hrep, -⟩ := h
        refine ⟨by simpa using hne, interiorFailures_nil hint, ps, by first | rfl | exact hp, hl, hw, ?_, ?_, ?_⟩
        · intro i hi
          have hi' : i < ss.toList.length := by simpa using hi
          have := hst i hi'
          simp only [sampledFailures] at this
          split at this
          · simp at this
          · rename_i s hs
            split at this
            · rename_i p hp'
              split at this
              · simp at this
              · rename_i hdup
                have hd : (templates (ss.toList.take i)).contains s.template = false := by simpa using hdup
                exact ⟨s, p, hs, hp', hd, orderFailures_nil hord hi' hs hp' hd, by simpa using this⟩
            · simp at this
        · intro p hp'
          have := List.filterMap_eq_nil_iff.1 hcov p hp'
          split at this
          · simp at this
          · rename_i hc
            simpa using hc
        · unfold reportedFailures at hrep
          split at hrep
          · assumption
          · simp at hrep
    · have hc : checkAny allow own (some ss) = check allow own (some ss) := by
        unfold checkAny
        simp only [hp]
        exact if_neg hb
      rw [hc] at h
      obtain ⟨ha, hio, ps', hp', hl, hw, hst, hcov, hrep⟩ := check_ok h
      have hpp : ps = ps' := Option.some.inj (hp.symm.trans hp')
      rw [hpp] at hb
      have hd : designOk ps' own = true := by simp [designOk, hl, hw]
      have hn : anySampled own ps' ss.toList = false := by simpa [hd] using hb
      refine ⟨ha, hio, ps', by first | exact hp' | exact hp.symm.trans hp', hl, hw, fun i hi => ?_, hcov, hrep⟩
      obtain ⟨s, p, hs, hf, hd', hi', hc'⟩ := hst i hi
      have hmem : ss.toList.getD i .null ∈ ss.toList := getD_mem_of_lt _ _ (by simpa using hi)
      have hso := anySampled_false hn hmem hs hf
      exact ⟨s, p, hs, hf, hd', hi', by simpa [hso] using hc'⟩

/-- **Every port a header shares rows at is a read the second stage holds**, of a statement of either shape: `check_ok_refs` of
a passing `checkAny`. -/
theorem checkAny_ok_refs {allow : Bool} {own : Own} {ss : Array Json}
    (h : (checkAny allow own (some ss)).codes = #[]) :
    ∃ ps, own.parts = some ps ∧ ∀ i < ss.size, ∃ s p, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧
      ps.find? (·.template == s.template) = some p ∧ ∀ port c, (port, c) ∈ entries (getD s.header "shared_rows") →
        ∃ e, (port, e) ∈ entries (getD s.registered "ports") ∧ readOk own (byTemplate own.interior) p port e = true := by
  obtain ⟨-, -, ps, hp, -, -, hst, -, -⟩ := checkAny_ok h
  refine ⟨ps, hp, fun i hi => ?_⟩
  obtain ⟨s, p, hs, hf, -, -, hc⟩ := hst i hi
  obtain ⟨-, -, -, hr, -, -, hrefs, -⟩ := codes_nil_shared hc
  refine ⟨s, p, hs, hf, fun port c hm => ?_⟩
  obtain ⟨e, he⟩ := refsOk_read hrefs hm
  exact ⟨e, he, Array.all_eq_true_iff_forall_mem.mp hr _ he⟩

/-- **Every wired port reads its writer unit's value**, of a statement of either shape: `check_ok_wired` of a passing
`checkAny`. A statement of only its part's sampled proof blocks is held at every slot of its part all the same: a
read's rows are placed for the part's `Q·n_t` slots (`placedOk`), the slots it holds among them. -/
theorem checkAny_ok_wired {allow : Bool} {own : Own} {ss : Array Json}
    (h : (checkAny allow own (some ss)).codes = #[]) :
    ∃ ps, own.parts = some ps ∧ ∀ i < ss.size, ∃ s p, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧
      ps.find? (·.template == s.template) = some p ∧
      wiredRead own s.template (entries (getD s.registered "ports")) = true ∧
      ∀ port e w, (port, e) ∈ entries (getD s.registered "ports") → linkOf own s.template port = none →
        wireOf own s.template port = some w →
        ∃ x pw r, getD e "value" = .str (portOf (byTemplate own.interior) w.writer w.wport) ∧
          (portOf (byTemplate own.interior) w.writer w.wport, x) ∈ Record.byPort own.interior ∧
          writerOf own w = some pw ∧ rowsOf own p.template port = some r ∧ pw.n = r.per ∧ sameAt e x = true ∧
          placedOk own.sample.size p e (own.sample.size * r.per) (wiredRow p r) = true := by
  obtain ⟨-, -, ps, hp, -, -, hst, -, -⟩ := checkAny_ok h
  refine ⟨ps, hp, fun i hi => ?_⟩
  obtain ⟨s, p, hs, hf, -, -, hc⟩ := hst i hi
  obtain ⟨-, -, -, hr, -, hwr, -, -⟩ := codes_nil_shared hc
  have hpt : p.template = s.template := by simpa using Array.find?_some hf
  rw [hpt] at hwr
  refine ⟨s, p, hs, hf, hwr, fun port e w he hl hw => ?_⟩
  rw [← hpt] at hl hw
  exact readOk_wired hl hw (Array.all_eq_true_iff_forall_mem.mp hr _ he)

/-- **Every statement reads registered what its template's reference statement reads**, of a statement of either
shape: `check_ok_values` of a passing `checkAny`. -/
theorem checkAny_ok_values {allow : Bool} {own : Own} {ss : Array Json}
    (h : (checkAny allow own (some ss)).codes = #[]) :
    ∀ i < ss.size, ∃ s r, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧ (s.template, r) ∈ own.refReads ∧
      jeq (blankReads own s.template s.registered) (blankReads own s.template r) = true := by
  obtain ⟨-, -, _ps, -, -, -, hst, -, -⟩ := checkAny_ok h
  intro i hi
  obtain ⟨s, p, hs, -, -, -, hc⟩ := hst i hi
  obtain ⟨-, -, -, -, -, -, -, hv⟩ := codes_nil_shared hc
  obtain ⟨r, hr, hj⟩ := valuesOk_ref hv
  exact ⟨s, r, hs, hr, hj⟩

/-- Each of part `p`'s sampled slots is one of the second sample's units and one of the part's slots. -/
theorem mem_sampledSlots {q : Nat} {p : Part} {us : Array Nat} {u : Nat} (h : u ∈ sampledSlots q p us) :
    u ∈ us ∧ p.holds q u = true := by
  simpa [sampledSlots] using h

/-- **A passing `second-population` of a statement of only its part's sampled proof blocks**: the header's
`units.indices` are the part's slots the second stage samples, in the second sample's order (`sampledSlots`), at
least one, and its `instances` their count. -/
theorem sampledOk_indices {q : Nat} {p : Part} {us : Array Nat} {h : Json} (hh : sampledOk q p us h = true) :
    ∃ is, getD (getD h "units") "indices" = .arr is ∧
      nat? (getD h "instances") = some (sampledSlots q p us).size ∧ 0 < (sampledSlots q p us).size ∧
      is.toList.map nat? = (sampledSlots q p us).toList.map some := by
  simp only [sampledOk] at hh
  split at hh
  · rename_i is m hi hm
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hh
    obtain ⟨⟨h0, rfl⟩, hl⟩ := hh
    exact ⟨is, hi, hm, h0, hl⟩
  · simp at hh

/-- **A passing `second-header` of a statement of only its part's sampled proof blocks** names the reference header
the auditor's params give template `t`, which header `h` is, placed in the run (`atRun`) and at the statement's slots
(`atSampled`), but for the run's fields. -/
theorem sampledHeaderOk_ref {own : Own} {named : Bool} {k : Nat} {t : String} {h : Json} {slots : Array Nat}
    {reads : Array (String × Json)} (hh : sampledHeaderOk own named k t h slots reads = true) :
    ∃ r r', (t, r) ∈ own.refs ∧ atRun own named k t r = some r' ∧
      jeq (blankRun h) (blankRun (atSampled r' slots reads)) = true := by
  unfold sampledHeaderOk at hh
  cases hf : refOf own t with
  | none => simp [hf] at hh
  | some r =>
    cases ha : atRun own named k t r with
    | none => simp [hf, ha] at hh
    | some r' =>
      simp only [hf, ha] at hh
      exact ⟨r, r', refOf_mem hf, ha, hh⟩

/-- **What a statement of only its part's sampled proof blocks holds**: in a passing `checkAny`, where the `i`-th
statement is one (`sampledOnly`), its `units.indices` are its part's slots the second stage samples, in the second
sample's order (`sampledSlots`), at least one, its `instances` their count, its session's sample every unit of it
(`wholeSample`), and its header its template's reference's placed in the run and at those slots. So instance `j` of its
session is the part's `j`-th sampled slot, and every instance is a checked one. -/
theorem checkAny_ok_sampled {allow : Bool} {own : Own} {ss : Array Json}
    (h : (checkAny allow own (some ss)).codes = #[]) :
    ∃ ps, own.parts = some ps ∧ ∀ i < ss.size, ∃ s p, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧
      ps.find? (·.template == s.template) = some p ∧
      (sampledOnly own.sample.size p own.drawn s.header = true →
        (∃ is, getD (getD s.header "units") "indices" = .arr is ∧
          nat? (getD s.header "instances") = some (sampledSlots own.sample.size p own.drawn).size ∧
          0 < (sampledSlots own.sample.size p own.drawn).size ∧
          is.toList.map nat? = (sampledSlots own.sample.size p own.drawn).toList.map some) ∧
        jeq s.draw (wholeSample (sampledSlots own.sample.size p own.drawn).size) = true ∧
        ∃ r r', (s.template, r) ∈ own.refs ∧
          atRun own (byTemplate own.interior) (posOf ps s.template) s.template r = some r' ∧
          jeq (blankRun s.header) (blankRun (atSampled r' (sampledSlots own.sample.size p own.drawn)
            (entries (getD s.registered "ports")))) = true) := by
  obtain ⟨-, -, ps, hp, -, -, hst, -, -⟩ := checkAny_ok h
  refine ⟨ps, hp, fun i hi => ?_⟩
  obtain ⟨s, p, hs, hf, -, -, hc⟩ := hst i hi
  refine ⟨s, p, hs, hf, fun hso => ?_⟩
  rw [if_pos hso] at hc
  obtain ⟨-, -, hhead, hpop, -, hdraw, -⟩ := sampledCodes_nil hc
  exact ⟨sampledOk_indices hpop, hdraw, sampledHeaderOk_ref hhead⟩

/-- Two lists alike under two maps are alike place by place: where the second has `b` at place `i`, the first has an
entry there that its map sends where the second's sends `b`. -/
theorem map_eq_at {α β γ : Type} (f : α → γ) (g : β → γ) :
    ∀ (l : List α) (m : List β), l.map f = m.map g → ∀ (i : Nat) (b : β), m[i]? = some b →
      ∃ a, l[i]? = some a ∧ f a = g b
  | [], [], _, _, _, hb => by simp at hb
  | [], _ :: _, h, _, _, _ => by simp at h
  | _ :: _, [], h, _, _, _ => by simp at h
  | a :: l, _ :: m, h, i, b, hb => by
    simp only [List.map_cons, List.cons.injEq] at h
    cases i with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hb
      rw [hb] at h
      exact ⟨a, by simp, h.1⟩
    | succ n =>
      simp only [List.getElem?_cons_succ] at hb
      obtain ⟨a', ha', hf⟩ := map_eq_at f g l m h.2 n b hb
      exact ⟨a', by simpa using ha', hf⟩

/-- An entry a list has at a place is one of the list's, and the place is below its length. -/
theorem mem_of_at {α : Type} : ∀ (l : List α) (i : Nat) (a : α), l[i]? = some a → a ∈ l ∧ i < l.length
  | [], _, _, h => by simp at h
  | b :: l, 0, a, h => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    rw [h]
    exact ⟨by simp, by simp⟩
  | _ :: l, n + 1, a, h => by
    simp only [List.getElem?_cons_succ] at h
    obtain ⟨hm, hn⟩ := mem_of_at l n a h
    refine ⟨List.mem_cons_of_mem _ hm, ?_⟩
    simp only [List.length_cons]
    omega

/-- An entry of a list is at one of its places. -/
theorem at_of_mem {α : Type} : ∀ (l : List α) (a : α), a ∈ l → ∃ i : Nat, l[i]? = some a
  | [], _, h => by simp at h
  | b :: l, a, h => by
    rcases List.mem_cons.1 h with hab | h'
    · exact ⟨0, by simp [hab]⟩
    · obtain ⟨i, hi⟩ := at_of_mem l a h'
      exact ⟨i + 1, by simpa using hi⟩

/-- **Instance `j` of a statement of only its part's sampled proof blocks is the part's `j`-th sampled slot**: where
the header's `units.indices` `is` are the part's sampled slots (`sampledOk_indices`), the slot `u` at place `j` of
`sampledSlots` is a unit of the second sample and a slot of the part, and `is` has its number at place `j`. -/
theorem sampled_at {q : Nat} {p : Part} {us : Array Nat} {is : Array Json}
    (hl : is.toList.map nat? = (sampledSlots q p us).toList.map some) {j u : Nat}
    (hj : (sampledSlots q p us).toList[j]? = some u) :
    u ∈ us ∧ p.holds q u = true ∧ ∃ x, is.toList[j]? = some x ∧ nat? x = some u := by
  obtain ⟨x, hx, hn⟩ := map_eq_at nat? some _ _ hl j u hj
  have hm : u ∈ sampledSlots q p us := by
    have hmem := (mem_of_at _ j u hj).1
    first
      | exact Array.mem_def.2 hmem
      | simpa using hmem
  obtain ⟨hd, hp⟩ := mem_sampledSlots hm
  exact ⟨hd, hp, x, hx, hn⟩

/-- **Every slot of the part the second stage samples is an instance of the statement**: where the header's
`units.indices` `is` are the part's sampled slots, a unit `u` of the second sample that is a slot of part `p` has its
number at a place `j` of `is`, below the count of the part's sampled slots, the statement's `instances`. -/
theorem sampled_slot {q : Nat} {p : Part} {us : Array Nat} {is : Array Json}
    (hl : is.toList.map nat? = (sampledSlots q p us).toList.map some) {u : Nat} (hu : u ∈ us)
    (hp : p.holds q u = true) :
    ∃ (j : Nat) (x : Json), j < (sampledSlots q p us).size ∧ is.toList[j]? = some x ∧ nat? x = some u := by
  have hm : u ∈ sampledSlots q p us := by simpa [sampledSlots] using And.intro hu hp
  have hmem : u ∈ (sampledSlots q p us).toList := by
    first
      | exact Array.mem_def.1 hm
      | simpa using hm
  obtain ⟨j, hj⟩ := at_of_mem _ u hmem
  obtain ⟨-, -, x, hx, hn⟩ := sampled_at hl hj
  have hlt := (mem_of_at _ j u hj).2
  exact ⟨j, x, by simpa using hlt, hx, hn⟩

/-- **The slots of a statement of only its part's sampled proof blocks, each way**: in a passing `checkAny`, where
the `i`-th statement is one (`sampledOnly`), its `units.indices` hold at each place `j` the number of the `j`-th slot
of its part that the second stage samples, a unit of the second sample and a slot of the part (`sampled_at`); and
every unit of the second sample that is a slot of its part has its number at a place below the statement's
`instances` (`sampled_slot`). So the statement's instances and the part's sampled slots are one list, in the second
sample's order: the map from a sampled slot to its instance is its place among the part's sampled slots, where a
statement of every slot of the part has `u - lo`. -/
theorem checkAny_ok_slots {allow : Bool} {own : Own} {ss : Array Json}
    (h : (checkAny allow own (some ss)).codes = #[]) :
    ∃ ps, own.parts = some ps ∧ ∀ i < ss.size, ∃ s p, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧
      ps.find? (·.template == s.template) = some p ∧
      (sampledOnly own.sample.size p own.drawn s.header = true →
        ∃ is, getD (getD s.header "units") "indices" = .arr is ∧
          nat? (getD s.header "instances") = some (sampledSlots own.sample.size p own.drawn).size ∧
          (∀ (j u : Nat), (sampledSlots own.sample.size p own.drawn).toList[j]? = some u →
            u ∈ own.drawn ∧ p.holds own.sample.size u = true ∧ ∃ x, is.toList[j]? = some x ∧ nat? x = some u) ∧
          (∀ u, u ∈ own.drawn → p.holds own.sample.size u = true →
            ∃ (j : Nat) (x : Json), j < (sampledSlots own.sample.size p own.drawn).size ∧ is.toList[j]? = some x ∧
              nat? x = some u)) := by
  obtain ⟨ps, hp, hst⟩ := checkAny_ok_sampled h
  refine ⟨ps, hp, fun i hi => ?_⟩
  obtain ⟨s, p, hs, hf, hso⟩ := hst i hi
  refine ⟨s, p, hs, hf, fun hso' => ?_⟩
  obtain ⟨⟨is, hix, hm, -, hl⟩, -, -⟩ := hso hso'
  exact ⟨is, hix, hm, fun _ _ hj => sampled_at hl hj, fun _ hu hpu => sampled_slot hl hu hpu⟩

end Spot.SecondStage
