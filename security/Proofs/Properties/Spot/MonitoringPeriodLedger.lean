module

public import Properties.Spot.MonitoringPeriodLedger
public import Proofs.Properties.Spot.MonitoringPeriod
public import Proofs.Properties.Challenger.Lines

/-!
# Proofs that the ledger a period's check reads is the device's own (`Properties.Spot.MonitoringPeriodLedger`)

`read` passes the ledger to `Challenger.checked`, so each presented line is an entry numbered by its place that
says what a line of the device's own ledger says, or the finder outputs a break (`read_lines`, `Lines.line_at`,
`ledgerLinesOfDevice`).

Two entries whose texts without their `sig` agree have one number (`Lines.seq_of_text`). So where each entry of the
device's ledger is the line at its own number, its line that says what presented line `i` says is its line `i`
(`own_line_at`, `ledgerOfDevice`).
-/

@[expose] public section

namespace Properties.Spot.MonitoringPeriod

open Lean (Json)
open _root_.Spot _root_.Spot.Challenger Properties.Challenger
open Properties.Challenger.Window (SigInv verifyLedger_sig)
open Properties.Challenger.Lines (SeqInv checked_lines verifyLedger_seq seq_of_text line_at)

theorem unsignedText_eq {line : Json} {e : LedgerEntry} {s : String} (he : ledgerEntry? line = some e)
    (hs : Flock.canonUtf8 (Core.Record.unsigned e.entry) = .ok s) : unsignedText line = some s := by
  simp [unsignedText, he, hs, Except.toOption]

/-- What `read` read of the ledger where it returns: a list of lines that reaches the monitoring period's end, each
line an entry, entry `i` numbered `i` and signed under the key. -/
theorem read_lines {p : _root_.Spot.MonitoringPeriod.Params} {ledger : Json} {r : _root_.Spot.MonitoringPeriod.Read}
    (h : _root_.Spot.MonitoringPeriod.read p ledger = .ok r) :
    ∃ (lines : Array Json) (es : Array LedgerEntry), ledger = .arr lines ∧ Reaches p ledger ∧
      lines.toList.map ledgerEntry? = es.toList.map some ∧ SeqInv es.toList ∧ SigInv p.key es.toList := by
  obtain ⟨hck, stop, hstop, -⟩ := read_ok h
  obtain ⟨lines, es, hl, hlines, hvl⟩ := checked_lines hck
  exact ⟨lines, es, hl, by simp [Reaches, hstop], hlines, verifyLedger_seq hvl, verifyLedger_sig hvl⟩

/-- **Each line of the ledger a period's check reads is one the device wrote**
(`Properties.Spot.MonitoringPeriod.LedgerLinesOfDevice`). -/
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

/-- **Where the device numbers its ledger's lines, the presented line at place `i` and the device's own line `i` are
entries that say the same**, their signatures aside, where the finder outputs no break. -/
theorem own_line_at {dev : History} {key : ChallengerKey} {label : String} {lines : Array Json}
    {es : Array LedgerEntry} (hlines : lines.toList.map ledgerEntry? = es.toList.map some)
    (hseq : SeqInv es.toList) (hsig : SigInv key es.toList) (hnum : Numbered dev key.pub)
    (hb : ¬ Break dev key label (Json.mkObj [("ledger", Json.arr lines)])) {i : Nat} {line : Json}
    (hi : lines[i]? = some line) :
    ∃ (e : LedgerEntry) (s : String) (own : Json) (e' : LedgerEntry), ledgerEntry? line = some e ∧
      Flock.canonUtf8 (Core.Record.unsigned e.entry) = .ok s ∧ (dev.ledger key.pub)[i]? = some own ∧
      ledgerEntry? own = some e' ∧ Flock.canonUtf8 (Core.Record.unsigned e'.entry) = .ok s := by
  obtain ⟨e, s, own, e', hle, hei, hs, hown, he', hs'⟩ := line_at (dev := dev) hlines hseq hsig hb hi
  have hj := hnum own hown e' he'
  have hij : i = e'.seq := by
    rw [← hei]
    exact seq_of_text hle he' hs hs'
  exact ⟨e, s, own, e', hle, hs, by rw [hij]; exact hj, he', hs'⟩

/-- **The ledger a period's check reads is the device's own, whole up to the period's end**
(`Properties.Spot.MonitoringPeriod.LedgerOfDevice`). -/
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
      obtain ⟨e, s, own, e', hle, hs, hown, he', hs'⟩ := own_line_at hlines hseq hsig hnum hb hi
      exact ⟨own, s, hown, unsignedText_eq hle hs, unsignedText_eq he' hs'⟩
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

end Properties.Spot.MonitoringPeriod
