module

public import Proofs.Properties.Spot.MonitoringPeriodLedger
import Proofs.Flock.Bytes

/-!
# Proof that every line of the ledger a period's check reads but the last is the device's own, signature included

`Properties.Spot.MonitoringPeriod.LedgerOfDeviceSigned`, from the reader's chain.

* **The reader chains the lines.** `verifyLedger` holds each entry's `prev` to the SHA-512 of the whole text of the
  entry before it, in hex (`PrevInv`, `verifyLedger_prev`).
* **A line's `prev` is in what the device signs.** Two entries whose texts without their `sig` agree name one `prev`
  (`prev_of_text`), and a line is its entry's text (`ledgerEntry_text`).
* **So the texts at each place but the last have one digest.** The presented line `i + 1` and the device's own say
  the same (`own_line_at`), so they name one `prev`: the presented line `i`'s digest by the reader's chain, the
  device's own line `i`'s by the device's (`Chained`). Hex is injective (`hexOf_inj`).
* **Two texts with one digest are one, or the finder outputs them.** The finder at the lines' whole texts hashes
  both (`mem_lineTexts`), so two different ones are its collision (`lineBreak_of_collision`), and a break of the
  fixed finder is a break of it (`lineBreak_of_break`).
-/

@[expose] public section

set_option mvcgen.warning false

namespace Properties.Spot.MonitoringPeriod

open Lean (Json)
open Std.Do
open _root_.Spot _root_.Spot.Challenger Properties.Challenger
open FlockDeploy (JsonEquiv)
open Properties.Pouw.LiveSalt.Ledger (bind_ok of_triple of_triple_opt throw_spec none_spec failure_spec
  toOption_spec ledgerTag_spec nat_spec some_pure_spec forIn_array_zipIdx_ok wp_ite restart join_spec
  ledgerMessage_spec)
open Properties.Challenger.Window (verifyLedger_sig ledgerEntry_fields entry_fields_equiv)
open Properties.Challenger.Lines (checked_lines verifyLedger_seq line_entry)
open Properties.Challenger.CanonRead (equiv_str)

/-! ## The reader's chain -/

/-- Each entry of `pre` after its first names as `prev` the SHA-512 of the whole text of the entry before it, in
hex, and `prev` is that of its last entry. -/
def PrevInv (pre : List LedgerEntry) (prev : String) : Prop :=
  (∀ (i : Nat) (e e' : LedgerEntry), pre[i]? = some e → pre[i + 1]? = some e' →
      e'.prev = Flock.hexOf (Flock.Sha512.hash e.text.toUTF8)) ∧
    ∀ e, pre.getLast? = some e → prev = Flock.hexOf (Flock.Sha512.hash e.text.toUTF8)

theorem prev_nil (prev : String) : PrevInv [] prev :=
  ⟨fun i e e' h => by simp at h, fun e h => by simp at h⟩

theorem prev_step {pre : List LedgerEntry} {e : LedgerEntry} {prev : String} (hI : PrevInv pre prev)
    (h : e.prev = prev) : PrevInv (pre ++ [e]) (Flock.hexOf (Flock.Sha512.hash e.text.toUTF8)) := by
  refine ⟨fun i x x' hx hx' => ?_, fun x hx => ?_⟩
  · by_cases hi : i + 1 < pre.length
    · rw [List.getElem?_append_left (Nat.lt_of_succ_lt hi)] at hx
      rw [List.getElem?_append_left hi] at hx'
      exact hI.1 i x x' hx hx'
    · have hlen : i + 1 < (pre ++ [e]).length := (List.getElem?_eq_some_iff.1 hx').1
      have hi' : i + 1 = pre.length := by
        rw [List.length_append, List.length_singleton] at hlen
        omega
      rw [List.getElem?_append_right (Nat.le_of_eq hi'.symm), hi', Nat.sub_self, List.getElem?_cons_zero] at hx'
      rw [List.getElem?_append_left (by omega)] at hx
      have hlast : pre.getLast? = some x := by
        rw [List.getLast?_eq_getElem?, ← hi', Nat.add_sub_cancel]
        exact hx
      rw [← Option.some.inj hx', h]
      exact hI.2 x hlast
  · rw [List.getLast?_append, List.getLast?_singleton] at hx
    rw [← Option.some.inj hx]

theorem chain_prev {v2 : Bool} {e : LedgerEntry} {n : Nat} {prev : String}
    (h : (!(v2 && e.schema == LEDGER_SCHEMA_V1) && e.seq == n && e.prev == prev) = true) : e.prev = prev := by
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  exact h.2

-- `mvcgen` splits `verifyLedger`'s loop body into one goal per path through its checks
set_option maxHeartbeats 4000000 in
/-- **`verifyLedger` chained every entry to the one before it**: entry `i + 1` names as `prev` the SHA-512 of entry
`i`'s whole text. -/
theorem verifyLedger_prev {es : Array LedgerEntry} {key : ChallengerKey} {r : Array Win × Array MonitoringPeriod}
    (h : verifyLedger es key = .ok r) : ∃ prev, PrevInv es.toList prev := by
  unfold verifyLedger at h
  rw [bind_ok] at h
  obtain ⟨s, hs, -⟩ := h
  have hI := forIn_array_zipIdx_ok es _ _ (fun pre s => PrevInv pre s.2.2.2.2.2.2.1) (prev_nil _) ?step hs
  · exact ⟨_, hI⟩
  · intro pre e s t hI hb
    obtain ⟨wins, leaves, records, periods, current, reserves, prev, last, v2, twoStep⟩ := s
    refine of_triple (Q := fun t => ∃ s', t = .yield s' ∧ PrevInv (pre ++ [e]) s'.2.2.2.2.2.2.1) ?_ hb
    clear hb hs
    mvcgen [-Spec.throw_Except, throw_spec, ledgerMessage_spec, join_spec]
    all_goals (try (refine wp_ite (fun _ => restart ?_) (fun _ => restart ?_) <;>
      mvcgen [-Spec.throw_Except, throw_spec, ledgerMessage_spec, join_spec]))
    all_goals exact ⟨_, rfl, prev_step hI (chain_prev ‹_›)⟩

/-! ## A line's `prev` and its text -/

-- `mvcgen` splits `Core.Record.ofJson` into one goal per field it parses
set_option maxHeartbeats 4000000 in
/-- A record entry's `prev` is its JSON's. -/
theorem recordOfJson_prev {j : Json} {r : Core.Record} (h : Core.Record.ofJson j = .ok r) :
    getD j "prev" = .str r.prev := by
  refine of_triple (Q := fun r => getD j "prev" = .str r.prev) ?_ h
  clear h
  mvcgen [Core.Record.ofJson, -Spec.throw_Except, throw_spec]

-- `mvcgen` splits `ledgerEntry?` into one goal per field it parses
set_option maxHeartbeats 1000000 in
/-- A ledger entry's record decodes from its entry, with its `prev`. -/
theorem ledgerEntry_prev {line : Json} {e : LedgerEntry} (h : ledgerEntry? line = some e) :
    ∃ r, (Core.Record.ofJson e.entry).toOption = some r ∧ r.prev = e.prev := by
  refine of_triple_opt (Q := fun e => ∃ r, (Core.Record.ofJson e.entry).toOption = some r ∧ r.prev = e.prev) ?_ h
  clear h
  mvcgen [ledgerEntry?, guard, -Spec.throw_Option, none_spec, failure_spec, toOption_spec, ledgerTag_spec, nat_spec,
    some_pure_spec]
  all_goals exact ⟨_, by assumption, rfl⟩

-- `mvcgen` splits `ledgerEntry?` into one goal per field it parses
set_option maxHeartbeats 1000000 in
/-- **A ledger line is its entry's text**, signature included. -/
theorem ledgerEntry_text {line : Json} {e : LedgerEntry} (h : ledgerEntry? line = some e) : line = .str e.text := by
  refine of_triple_opt (Q := fun e => line = .str e.text) ?_ h
  clear h
  mvcgen [ledgerEntry?, guard, -Spec.throw_Option, none_spec, failure_spec, toOption_spec, ledgerTag_spec, nat_spec,
    some_pure_spec]

/-- **Two entries whose texts without their `sig` agree name one `prev`.** -/
theorem prev_of_text {line line' : Json} {e e' : LedgerEntry} (he : ledgerEntry? line = some e)
    (he' : ledgerEntry? line' = some e') {s : String}
    (hs : Flock.canonUtf8 (Core.Record.unsigned e.entry) = .ok s)
    (hs' : Flock.canonUtf8 (Core.Record.unsigned e'.entry) = .ok s) : e.prev = e'.prev := by
  obtain ⟨sO, objO, -⟩ := ledgerEntry_fields he
  obtain ⟨sE, objE, -⟩ := ledgerEntry_fields he'
  have hq : JsonEquiv (getD e.entry "prev") (getD e'.entry "prev") :=
    entry_fields_equiv sO sE objO objE hs hs' (by decide)
  obtain ⟨r, hr, hrs⟩ := ledgerEntry_prev he
  obtain ⟨r', hr', hrs'⟩ := ledgerEntry_prev he'
  rw [recordOfJson_prev (Properties.Challenger.ChallengePhase.toOption_some hr),
    recordOfJson_prev (Properties.Challenger.ChallengePhase.toOption_some hr')] at hq
  rw [← hrs, ← hrs']
  exact Json.str.inj (equiv_str hq)

/-! ## The finder at the lines' whole texts -/

/-- The whole text of a line of `lines` that is an entry is among `lineTexts lines`. -/
theorem mem_lineTexts {lines : List Json} {line : Json} {e : LedgerEntry} (hl : line ∈ lines)
    (he : ledgerEntry? line = some e) : e.text.toUTF8 ∈ lineTexts lines :=
  List.mem_filterMap.2 ⟨line, hl, by simp [he]⟩

/-- **A break of the fixed finder on the ledger is a break at the lines' whole texts.** -/
theorem lineBreak_of_break {dev : History} {key : ChallengerKey} {label : String} {lines : Array Json}
    (b : Break dev key label (Json.mkObj [("ledger", Json.arr lines)])) : LineBreak dev key label lines := by
  obtain ⟨w, hw, hv⟩ := b
  exact ⟨w, by simp only [findLineBreak, hw], hv⟩

/-- **Two different whole texts with one SHA-512 digest are a break**, among the presented lines' and the device's
own. -/
theorem lineBreak_of_collision {dev : History} {key : ChallengerKey} {label : String} {lines : Array Json}
    {x y : ByteArray} (hx : x ∈ lineTexts lines.toList ++ lineTexts (dev.ledger key.pub))
    (hy : y ∈ lineTexts lines.toList ++ lineTexts (dev.ledger key.pub)) (hne : x ≠ y)
    (he : Flock.Sha512.hash x = Flock.Sha512.hash y) : LineBreak dev key label lines := by
  cases hf : findBreak dev key label (Json.mkObj [("ledger", Json.arr lines)]) with
  | some w => exact lineBreak_of_break ⟨w, hf, findBreak_valid hf⟩
  | none =>
    cases hc : collisionIn (lineTexts lines.toList ++ lineTexts (dev.ledger key.pub)) with
    | none => exact absurd hc (collisionIn_ne_none hx hy hne he)
    | some c =>
      obtain ⟨a, b⟩ := c
      exact ⟨.collision a b, by simp only [findLineBreak, hf, hc, Option.map_some], collisionIn_valid hc⟩

/-! ## The statement -/

/-- **Every line of the ledger a period's check reads but the last is the device's own line, signature included**
(`Properties.Spot.MonitoringPeriod.LedgerOfDeviceSigned`). -/
theorem ledgerOfDeviceSigned : LedgerOfDeviceSigned := by
  intro dev p ledger r h hnum hch
  obtain ⟨hck, -⟩ := read_ok h
  obtain ⟨lines, es, hl, hlines, hvl⟩ := checked_lines hck
  subst hl
  refine ⟨lines, rfl, ?_⟩
  by_cases hb : LineBreak dev p.key p.label lines
  · exact .inr hb
  · have hnb : ¬ Break dev p.key p.label (Json.mkObj [("ledger", Json.arr lines)]) :=
      fun b => hb (lineBreak_of_break b)
    obtain ⟨prev, hprev, -⟩ := verifyLedger_prev hvl
    have hseq := verifyLedger_seq hvl
    have hsig := verifyLedger_sig hvl
    refine .inl (List.ext_getElem? fun i => ?_)
    by_cases hi : i + 1 < lines.size
    · have hi0 : i < lines.size := by omega
      have hli := Array.getElem?_eq_getElem hi0
      have hlj := Array.getElem?_eq_getElem hi
      obtain ⟨e, s, own, e', hle, hs, hown, he', hs'⟩ := own_line_at hlines hseq hsig hnum hnb hli
      obtain ⟨f, t, next, f', hlf, ht, hnext, hf', ht'⟩ := own_line_at hlines hseq hsig hnum hnb hlj
      obtain ⟨e0, he0, hes⟩ := line_entry hlines hli
      obtain ⟨f0, hf0, hfs⟩ := line_entry hlines hlj
      rw [Option.some.inj (he0.symm.trans hle)] at hes
      rw [Option.some.inj (hf0.symm.trans hlf)] at hfs
      have h1 : f.prev = Flock.hexOf (Flock.Sha512.hash e.text.toUTF8) := hprev i e f hes hfs
      have h2 : f'.prev = Flock.hexOf (Flock.Sha512.hash e'.text.toUTF8) := hch i own next e' f' hown hnext he' hf'
      have h3 : f.prev = f'.prev := prev_of_text hlf hf' ht ht'
      have hh : Flock.Sha512.hash e.text.toUTF8 = Flock.Sha512.hash e'.text.toUTF8 :=
        Properties.NetworkRecords.Bytes.hexOf_inj (h1.symm.trans (h3.trans h2))
      have htext : e.text.toUTF8 = e'.text.toUTF8 := by
        refine Classical.byContradiction fun hne => hb ?_
        have hl : lines[i] ∈ lines.toList := List.mem_of_getElem? (by rw [Array.getElem?_toList]; exact hli)
        exact lineBreak_of_collision (List.mem_append_left _ (mem_lineTexts hl hle))
          (List.mem_append_right _ (mem_lineTexts (List.mem_of_getElem? hown) he')) hne hh
      have hstr : e.text = e'.text := String.toByteArray_inj.1
        (String.toUTF8_eq_toByteArray.symm.trans (htext.trans String.toUTF8_eq_toByteArray))
      rw [List.getElem?_take_of_lt (by omega), List.getElem?_take_of_lt (by omega), Array.getElem?_toList, hli, hown]
      exact congrArg some ((ledgerEntry_text hle).trans ((congrArg Json.str hstr).trans (ledgerEntry_text he').symm))
    · rw [List.getElem?_take_eq_none (by omega), List.getElem?_take_eq_none (by omega)]

end Properties.Spot.MonitoringPeriod
