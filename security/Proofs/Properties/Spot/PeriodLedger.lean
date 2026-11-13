module

public import Properties.Spot.PeriodLedger
public import Proofs.Properties.Spot.Period
public import Proofs.Properties.Challenger.Window
import Proofs.Properties.Nci.RecordJson

/-!
# Proofs that the ledger a period's check reads is the device's own (`Properties.Spot.PeriodLedger`)

`read` passes the ledger to `Challenger.checked`, which reads every line as an entry (`checked_lines`) and runs
`verifyLedger` on them: entry `i` is numbered `i` (`verifyLedger_seq`) and its signature verifies over its message
under the key (`Window.verifyLedger_sig`). A line whose signature verifies says what a line of the device's own
ledger says, or the finder outputs a break (`Window.line_of_device`); so each presented line does (`line_at`,
`ledgerLinesOfDevice`).

Two entries whose texts without their `sig` agree have one number, the number being a field of that text
(`seq_of_text`). So where the device numbers its lines, its line that says what presented line `i` says is its
line `i` (`ledgerOfDevice`).
-/

@[expose] public section

set_option mvcgen.warning false

namespace Properties.Spot.Period

open Lean (Json JsonNumber)
open Std.Do
open _root_.Spot _root_.Spot.Challenger Properties.Challenger
open FlockDeploy (JsonEquiv)
open Properties.Pouw.LiveSalt.Ledger (bind_ok of_triple of_triple_opt throw_spec ok_spec none_spec failure_spec
  toOption_spec ledgerTag_spec nat_spec some_pure_spec forIn_array_zipIdx_ok wp_ite restart join_spec
  ledgerMessage_spec chain)
open Properties.Challenger.Window (SigInv verifyLedger_sig sig_lms line_of_device ledgerEntry_fields
  entry_fields_equiv)

/-! ## The lines `checked` reads -/

theorem step_line {b : Array LedgerEntry} {pre : List (Json × Nat)} {cur : Json × Nat} {e : LedgerEntry}
    (h : pre.map (fun x => ledgerEntry? x.1) = b.toList.map some) (he : ledgerEntry? cur.1 = some e) :
    (pre ++ [cur]).map (fun x => ledgerEntry? x.1) = (b.push e).toList.map some := by
  simp [h, he]

theorem lines_arr {ls : Array Json} {es : Array LedgerEntry}
    (h : ls.zipIdx.toList.map (fun x => ledgerEntry? x.1) = es.toList.map some) :
    ls.toList.map ledgerEntry? = es.toList.map some := by
  have hm : ∀ l : List (Json × Nat), l.map (fun x => ledgerEntry? x.1) = (l.map Prod.fst).map ledgerEntry? := by
    intro l
    first
      | exact List.map_map.symm
      | simp [Function.comp_def]
  rw [← h, Array.toList_zipIdx, hm, List.zipIdx_map_fst]

-- the closers must not unfold `verifyLedger` or the key's hash while they search the context
set_option maxHeartbeats 1000000 in
/-- `checked` reads a list of lines, each an entry, and runs `verifyLedger` on the entries in the lines' order. -/
theorem checked_lines {ledger : Json} {key : ChallengerKey} {r : Array Win × Array MonitoringPeriod}
    (h : checked ledger key = .ok r) :
    ∃ (lines : Array Json) (es : Array LedgerEntry), ledger = .arr lines ∧
      lines.toList.map ledgerEntry? = es.toList.map some ∧ verifyLedger es key = .ok r := by
  refine of_triple (Q := fun r => ∃ (lines : Array Json) (es : Array LedgerEntry), ledger = .arr lines ∧
      lines.toList.map ledgerEntry? = es.toList.map some ∧ verifyLedger es key = .ok r) ?_ h
  clear h
  mvcgen -trivial [checked, -Spec.throw_Except, throw_spec] invariants
  · ⇓? ⟨cur, es⟩ => ⌜cur.prefix.map (fun x => ledgerEntry? x.1) = es.toList.map some⌝
  all_goals first
    | exact True.intro
    | (apply step_line <;> with_reducible assumption)
    | exact ⟨_, _, rfl, lines_arr ‹_›, ‹_›⟩
    | exact (rfl : ([] : List (Option LedgerEntry)) = [])
    | fail "checked_lines: a goal mvcgen left is not closed"

/-- The line at a place of a ledger whose lines are all entries is the entry at that place. -/
theorem line_entry {lines : Array Json} {es : Array LedgerEntry}
    (h : lines.toList.map ledgerEntry? = es.toList.map some) {i : Nat} {line : Json} (hi : lines[i]? = some line) :
    ∃ e, ledgerEntry? line = some e ∧ es.toList[i]? = some e := by
  have h1 : (lines.toList.map ledgerEntry?)[i]? = (es.toList.map some)[i]? := by rw [h]
  rw [List.getElem?_map, List.getElem?_map, Array.getElem?_toList (xs := lines), hi] at h1
  cases he : es.toList[i]? with
  | none =>
    rw [he] at h1
    cases h1
  | some e =>
    rw [he] at h1
    exact ⟨e, Option.some.inj h1, rfl⟩

/-! ## Each entry's number -/

/-- Entry `i` of `pre` is numbered `i`. -/
def SeqInv (pre : List LedgerEntry) : Prop :=
  ∀ i e, pre[i]? = some e → e.seq = i

theorem seq_step {pre : List LedgerEntry} {e : LedgerEntry} (hI : SeqInv pre) (h : e.seq = pre.length) :
    SeqInv (pre ++ [e]) := by
  intro i x hx
  rcases getElem?_snoc hx with hx | ⟨hi, hxe⟩
  · exact hI i x hx
  · rw [hi, ← hxe]
    exact h

-- `mvcgen` splits `verifyLedger`'s loop body into one goal per path through its checks
set_option maxHeartbeats 4000000 in
/-- **`verifyLedger` checked every entry's number**: entry `i` is numbered `i`. -/
theorem verifyLedger_seq {es : Array LedgerEntry} {key : ChallengerKey} {r : Array Win × Array MonitoringPeriod}
    (h : verifyLedger es key = .ok r) : SeqInv es.toList := by
  unfold verifyLedger at h
  rw [bind_ok] at h
  obtain ⟨s, hs, -⟩ := h
  refine forIn_array_zipIdx_ok es _ _ (fun pre _ => SeqInv pre) (fun i e hi => by simp at hi) ?_ hs
  intro pre e s t hI hb
  obtain ⟨wins, leaves, records, periods, current, reserves, prev, last, v2, twoStep⟩ := s
  refine of_triple (Q := fun t => ∃ s', t = .yield s' ∧ SeqInv (pre ++ [e])) ?_ hb
  clear hb hs
  mvcgen [-Spec.throw_Except, throw_spec, ledgerMessage_spec, join_spec]
  all_goals (try (refine wp_ite (fun _ => restart ?_) (fun _ => restart ?_) <;>
    mvcgen [-Spec.throw_Except, throw_spec, ledgerMessage_spec, join_spec]))
  all_goals refine ⟨_, rfl, seq_step hI (chain ‹_›).1⟩

-- `mvcgen` splits `Core.Record.ofJson` into one goal per field it parses
set_option maxHeartbeats 4000000 in
/-- A record entry's number is its JSON's `seq`, a number's mantissa. -/
theorem recordOfJson_seq {j : Json} {r : Core.Record} (h : Core.Record.ofJson j = .ok r) :
    ∃ n : JsonNumber, getD j "seq" = .num n ∧ r.seq = n.mantissa.toNat := by
  refine of_triple (Q := fun r => ∃ n : JsonNumber, getD j "seq" = .num n ∧ r.seq = n.mantissa.toNat) ?_ h
  clear h
  mvcgen [Core.Record.ofJson, -Spec.throw_Except, throw_spec]
  all_goals first
    | exact ⟨_, by assumption, rfl⟩
    | fail "recordOfJson_seq: a goal mvcgen left is not closed"

-- `mvcgen` splits `ledgerEntry?` into one goal per field it parses
set_option maxHeartbeats 1000000 in
/-- A ledger entry's record decodes from its entry, with its number. -/
theorem ledgerEntry_seq {line : Json} {e : LedgerEntry} (h : ledgerEntry? line = some e) :
    ∃ r, (Core.Record.ofJson e.entry).toOption = some r ∧ r.seq = e.seq := by
  refine of_triple_opt (Q := fun e => ∃ r, (Core.Record.ofJson e.entry).toOption = some r ∧ r.seq = e.seq) ?_ h
  clear h
  mvcgen [ledgerEntry?, guard, -Spec.throw_Option, none_spec, failure_spec, toOption_spec, ledgerTag_spec, nat_spec,
    some_pure_spec]
  all_goals exact ⟨_, by assumption, rfl⟩

theorem equiv_num {x : Json} {n : JsonNumber} (h : JsonEquiv x (.num n)) : x = .num n := by
  cases x with
  | num c => cases h; rfl
  | arr a => rw [JsonEquiv, Properties.Challenger.CanonRead.jsonForm_arr] at h; cases h
  | obj m => rw [JsonEquiv, FlockDeploy.jsonForm_obj] at h; cases h
  | _ => cases h

/-- **Two entries whose texts without their `sig` agree have one number.** -/
theorem seq_of_text {line line' : Json} {e e' : LedgerEntry} (he : ledgerEntry? line = some e)
    (he' : ledgerEntry? line' = some e') {s : String}
    (hs : Flock.canonUtf8 (Core.Record.unsigned e.entry) = .ok s)
    (hs' : Flock.canonUtf8 (Core.Record.unsigned e'.entry) = .ok s) : e.seq = e'.seq := by
  obtain ⟨sO, objO, -⟩ := ledgerEntry_fields he
  obtain ⟨sE, objE, -⟩ := ledgerEntry_fields he'
  have hq : JsonEquiv (getD e.entry "seq") (getD e'.entry "seq") :=
    entry_fields_equiv sO sE objO objE hs hs' (by decide)
  obtain ⟨r, hr, hrs⟩ := ledgerEntry_seq he
  obtain ⟨r', hr', hrs'⟩ := ledgerEntry_seq he'
  obtain ⟨n, hn, hrn⟩ := recordOfJson_seq (Properties.Challenger.Session.toOption_some hr)
  obtain ⟨n', hn', hrn'⟩ := recordOfJson_seq (Properties.Challenger.Session.toOption_some hr')
  rw [hn, hn'] at hq
  have hnn : n = n' := Json.num.inj (equiv_num hq)
  rw [← hrs, ← hrs', hrn, hrn', hnn]

/-! ## Each line is the device's -/

theorem getD_ledger (ledger : Json) : getD (Json.mkObj [("ledger", ledger)]) "ledger" = ledger := by
  have h : (Json.mkObj [("ledger", ledger)]).getObjVal? "ledger" = .ok ledger :=
    (NciCheck.Record.SecurityProofs.Json.getObjVal?_mkObj (by simp) "ledger" ledger).2 (by simp)
  simp only [getD, field?, h, Except.toOption, Option.getD_some]

theorem unsignedText_eq {line : Json} {e : LedgerEntry} {s : String} (he : ledgerEntry? line = some e)
    (hs : Flock.canonUtf8 (Core.Record.unsigned e.entry) = .ok s) : unsignedText line = some s := by
  first
    | simp [unsignedText, he, hs, Except.toOption]
    | (unfold unsignedText
       rw [he]
       show (Flock.canonUtf8 (Core.Record.unsigned e.entry)).toOption = some s
       rw [hs]
       rfl)

/-- What `read` read of the ledger where it returns: a list of lines that reaches the monitoring period's end, each
line an entry, entry `i` numbered `i` and signed under the key. -/
theorem read_lines {p : _root_.Spot.Period.Params} {ledger : Json} {r : _root_.Spot.Period.Read}
    (h : _root_.Spot.Period.read p ledger = .ok r) :
    ∃ (lines : Array Json) (es : Array LedgerEntry), ledger = .arr lines ∧ Reaches p ledger ∧
      lines.toList.map ledgerEntry? = es.toList.map some ∧ SeqInv es.toList ∧ SigInv p.key es.toList := by
  obtain ⟨hck, stop, hstop, -⟩ := read_ok h
  obtain ⟨lines, es, hl, hlines, hvl⟩ := checked_lines hck
  exact ⟨lines, es, hl, by simp [Reaches, hstop], hlines, verifyLedger_seq hvl, verifyLedger_sig hvl⟩

/-- **A presented line is an entry under its place's number that says what a line of the device's own ledger
says**, where the finder outputs no break. -/
theorem line_at {dev : History} {key : ChallengerKey} {label : String} {lines : Array Json}
    {es : Array LedgerEntry} (hlines : lines.toList.map ledgerEntry? = es.toList.map some)
    (hseq : SeqInv es.toList) (hsig : SigInv key es.toList)
    (hb : ¬ Break dev key label (Json.mkObj [("ledger", Json.arr lines)])) {i : Nat} {line : Json}
    (hi : lines[i]? = some line) :
    ∃ (e : LedgerEntry) (s : String) (own : Json) (e' : LedgerEntry), ledgerEntry? line = some e ∧ e.seq = i ∧
      Flock.canonUtf8 (Core.Record.unsigned e.entry) = .ok s ∧ own ∈ dev.ledger key.pub ∧
      ledgerEntry? own = some e' ∧ Flock.canonUtf8 (Core.Record.unsigned e'.entry) = .ok s := by
  obtain ⟨e, hle, hei⟩ := line_entry hlines hi
  obtain ⟨m, hm, hv⟩ := sig_lms hsig (List.mem_of_getElem? hei)
  have hl : line ∈ lines.toList := List.mem_of_getElem? (by rw [Array.getElem?_toList]; exact hi)
  rcases line_of_device (dev := dev) (label := label) (getD_ledger (Json.arr lines)) hl hle hm hv with
    ⟨own, hown, e', s, he', hs, hs'⟩ | b
  · exact ⟨e, s, own, e', hle, hseq i e hei, hs, hown, he', hs'⟩
  · exact absurd b hb

/-- **Each line of the ledger a period's check reads is one the device wrote**
(`Properties.Spot.Period.LedgerLinesOfDevice`). -/
theorem ledgerLinesOfDevice : LedgerLinesOfDevice := by
  intro dev p ledger r h
  obtain ⟨lines, es, hl, hreach, hlines, hseq, hsig⟩ := read_lines h
  subst hl
  by_cases hb : Break dev p.key p.label (Json.mkObj [("ledger", Json.arr lines)])
  · exact .inr hb
  · refine .inl ⟨lines, rfl, hreach, fun i line hi => ?_⟩
    obtain ⟨e, s, own, e', hle, hei, hs, hown, he', hs'⟩ := line_at (dev := dev) hlines hseq hsig hb hi
    exact ⟨⟨e, hle, hei⟩, by simp [unsignedText_eq hle hs], own, hown,
      (unsignedText_eq he' hs').trans (unsignedText_eq hle hs).symm⟩

/-- **The ledger a period's check reads is the device's own, whole up to the period's end**
(`Properties.Spot.Period.LedgerOfDevice`). -/
theorem ledgerOfDevice : LedgerOfDevice := by
  intro dev p ledger r h hnum
  obtain ⟨lines, es, hl, hreach, hlines, hseq, hsig⟩ := read_lines h
  subst hl
  by_cases hb : Break dev p.key p.label (Json.mkObj [("ledger", Json.arr lines)])
  · exact .inr hb
  · have hat : ∀ (i : Nat) (line : Json), lines[i]? = some line →
        ∃ own s, (dev.ledger p.key.pub)[i]? = some own ∧ unsignedText line = some s ∧
          unsignedText own = some s := by
      intro i line hi
      obtain ⟨e, s, own, e', hle, hei, hs, hown, he', hs'⟩ := line_at (dev := dev) hlines hseq hsig hb hi
      obtain ⟨j, hj⟩ := List.getElem?_of_mem hown
      obtain ⟨e'', he'', hj'⟩ := hnum j own hj
      have hee : e'' = e' := Option.some.inj (he''.symm.trans he')
      have hij : i = j := by
        rw [← hei, ← hj', hee]
        exact seq_of_text hle he' hs hs'
      exact ⟨own, s, by rw [hij]; exact hj, unsignedText_eq hle hs, unsignedText_eq he' hs'⟩
    refine .inl ⟨lines, rfl, hreach, ?_, fun line hl => ?_, ?_⟩
    · refine Nat.le_of_not_lt fun hlt => ?_
      obtain ⟨own, s, hown, -, -⟩ := hat _ _ (Array.getElem?_eq_getElem hlt)
      obtain ⟨hlt', -⟩ := List.getElem?_eq_some_iff.1 hown
      exact Nat.lt_irrefl _ hlt'
    · obtain ⟨i, hi⟩ := List.getElem?_of_mem hl
      rw [Array.getElem?_toList] at hi
      obtain ⟨own, s, -, hs, -⟩ := hat i line hi
      simp [hs]
    · refine List.ext_getElem? fun i => ?_
      rw [List.getElem?_map, List.getElem?_map, Array.getElem?_toList]
      by_cases hi : i < lines.size
      · obtain ⟨own, s, hown, hs, hs'⟩ := hat i _ (Array.getElem?_eq_getElem hi)
        rw [Array.getElem?_eq_getElem hi, List.getElem?_take_of_lt hi, hown, Option.map_some, Option.map_some, hs,
          hs']
      · rw [Array.getElem?_eq_none (Nat.le_of_not_lt hi), List.getElem?_take_eq_none (Nat.le_of_not_lt hi)]

end Properties.Spot.Period
