module

public import Proofs.Properties.Challenger.Window
import Proofs.Properties.Nci.RecordJson

/-!
# Each line of a ledger `checked` accepts is numbered by its place and is the device's own, or a break

`Challenger.checked` reads every line of the ledger as an entry (`checked_lines`) and runs `verifyLedger` on the
entries: entry `i` is numbered `i` (`verifyLedger_seq`) and its signature verifies over its message under the key
(`Window.verifyLedger_sig`). A line whose signature verifies says what a line of the device's own ledger says, or the
finder outputs a break (`Window.line_of_device`); so the line at each place does (`line_at`).

Two entries whose texts without their `sig` agree have one number, the number being a field of that text
(`seq_of_text`).
-/

@[expose] public section

set_option mvcgen.warning false

namespace Properties.Challenger.Lines

open Lean (Json JsonNumber)
open Std.Do
open Spot Spot.Challenger
open FlockDeploy (JsonEquiv)
open Properties.Pouw.LiveSalt.Ledger (bind_ok of_triple of_triple_opt throw_spec none_spec failure_spec
  toOption_spec ledgerTag_spec nat_spec some_pure_spec forIn_array_zipIdx_ok wp_ite restart join_spec
  ledgerMessage_spec chain)
open Properties.Challenger.Window (SigInv sig_lms line_of_device ledgerEntry_fields entry_fields_equiv)

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
  ∀ (i : Nat) (e : LedgerEntry), pre[i]? = some e → e.seq = i

theorem seq_step {pre : List LedgerEntry} {e : LedgerEntry} (hI : SeqInv pre) (h : e.seq = pre.length) :
    SeqInv (pre ++ [e]) := by
  intro i x hx
  by_cases hi : i < pre.length
  · rw [List.getElem?_append_left hi] at hx
    exact hI i x hx
  · rw [List.getElem?_append_right (Nat.le_of_not_lt hi), List.getElem?_singleton] at hx
    by_cases h0 : i - pre.length = 0
    · rw [ite_eq_left h0] at hx
      rw [← Option.some.inj hx, h]
      omega
    · rw [ite_eq_right h0] at hx
      cases hx

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
  all_goals exact ⟨_, by assumption, rfl⟩

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
  obtain ⟨n, hn, hrn⟩ := recordOfJson_seq (Properties.Challenger.ChallengePhase.toOption_some hr)
  obtain ⟨n', hn', hrn'⟩ := recordOfJson_seq (Properties.Challenger.ChallengePhase.toOption_some hr')
  rw [hn, hn'] at hq
  have hnn : n = n' := Json.num.inj (equiv_num hq)
  rw [← hrs, ← hrs', hrn, hrn', hnn]

/-! ## Each line is the device's -/

theorem getD_ledger (ledger : Json) : getD (Json.mkObj [("ledger", ledger)]) "ledger" = ledger := by
  have h : (Json.mkObj [("ledger", ledger)]).getObjVal? "ledger" = .ok ledger :=
    (NciCheck.Record.SecurityProofs.Json.getObjVal?_mkObj (by simp) "ledger" ledger).2 (by simp)
  simp only [getD, field?, h, Except.toOption, Option.getD_some]

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

end Properties.Challenger.Lines
