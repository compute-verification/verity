module

public import Proofs.Flock.SpotFlock.RunsAs.Top
public import Proofs.Flock.SpotFlock.Jeq

/-!
# The reads an interior record's checks leave out are reads the second stage's statement check held

Under a two-stage rule the run of record hands the interior record's checks (`Spot.Record.checks`) the reads the
verdicts of record report for the registration's own values, less the reads of the registration's own ports at their
own roots (`Spot.Record.lessOwnReads`). This file says what stands behind each read so left out, at an accepted
two-stage run (`AcceptedTwo`):

* a read left out names one of the record's ports at that port's root (`lessOwnReads_out`);
* a read the verdicts report is the value and root of a registered read one of the session statements states, which
  the second stage's statement check passed through `SecondStage.readOk` (`reported_stated`: the check holds the
  reported reads to the statements', `SecondStage.check_ok`, and every read of a statement to `readOk`,
  `SecondStage.stmtCodes_nil`), or its root prints as null, which is how a value reported under two roots is reported;
* so each read the verdicts report for the record is `HeldRead` or has such a root (`accepted_ownReads`), and each
  read left out is `HeldRead` unless the record's port of its name has a root that prints as null
  (`accepted_leftOut`);
* and `readOk` holds a read of one of the interior record's ports to an entry of a record, arm by arm (`readOk_port`).

**The link-named case.** At a port a link names, `readOk` holds the read to the interior registration's registered
value of the link's value's name where it registers one, else to the boundary record's port of that name, and never
to the interior record's port. Such a read is left out of the interior record's registered reads only where the
interior record has a port of that name at the root the read is held to. Where the interior record's port of that
name has another root, the read stays among the reads opened and the record's `registered` check fails.
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

/-- `Spot.jeq` is transitive. -/
theorem jeq_trans {a b c : Json} (h1 : Spot.jeq a b = true) (h2 : Spot.jeq b c = true) : Spot.jeq a c = true := by
  unfold Spot.jeq at h1 h2 ⊢
  cases ha : Flock.canon a <;> cases hb : Flock.canon b <;> cases hc : Flock.canon c <;>
    rw [ha, hb] at h1 <;> rw [hb, hc] at h2 <;> simp_all

/-- A fold keeps what each of its steps keeps. -/
theorem foldl_holds {α β : Type} (P : β → Prop) (f : β → α → β) :
    ∀ (l : List α) (b : β), P b → (∀ b a, a ∈ l → P b → P (f b a)) → P (l.foldl f b)
  | [], _, hb, _ => hb
  | a :: l, b, hb, hf => by
    rw [List.foldl_cons]
    exact foldl_holds P f l (f b a) (hf b a (by simp) hb) fun b' a' ha' => hf b' a' (by simp [ha'])

/-- **A read `Record.lessOwnReads` leaves out names one of the record's ports at that port's root.** -/
theorem lessOwnReads_out {reg : Json} {opened : Array (String × Json)} {v : String} {r : Json}
    (hv : (v, r) ∈ opened) (hno : (v, r) ∉ (Spot.Record.lessOwnReads reg (some opened)).getD #[]) :
    ∃ x, (v, x) ∈ Spot.Record.byPort reg ∧ Spot.jeq (Spot.getD x "root") r = true := by
  by_contra hne
  apply hno
  unfold Spot.Record.lessOwnReads
  simp only [Option.map_some, Option.getD_some, Array.mem_filter]
  refine ⟨hv, ?_⟩
  rw [Bool.not_eq_true']
  by_contra hany
  rw [Bool.not_eq_false] at hany
  obtain ⟨px, hpx, hpred⟩ := Array.any_eq_true'.1 hany
  obtain ⟨p, x⟩ := px
  simp only [Bool.and_eq_true, beq_iff_eq] at hpred
  exact hne ⟨x, hpred.1 ▸ hpx, hpred.2⟩

/-- Each entry of `acc` has a null root, or is the value and root of a registered read one of the statements `l`
states. -/
def ReadsOf (l : List Json) (acc : Array (String × Json)) : Prop :=
  ∀ k x, (k, x) ∈ acc → x = .null ∨ ∃ j ∈ l, ∃ s, Spot.SecondStage.Stmt.ofJson? j = some s ∧
    ∃ pe ∈ Spot.entries (Spot.getD s.registered "ports"),
      Spot.getD pe.2 "value" = .str k ∧ Spot.getD pe.2 "root" = x

/-- **An entry of the statements' reads is one statement's read**: `SecondStage.statementReads` maps a value to the
root one of the statements reads it at, or to null where two of them read it at two roots. -/
theorem statementReads_reads (l : List Json) : ReadsOf l (Spot.SecondStage.statementReads l) := by
  unfold Spot.SecondStage.statementReads
  refine foldl_holds (ReadsOf l) _ l #[] (by intro k x hx; simp at hx) ?_
  intro acc j hj hacc
  try dsimp only
  cases hs : Spot.SecondStage.Stmt.ofJson? j with
  | none => exact hacc
  | some s =>
    dsimp only
    rw [← Array.foldl_toList]
    refine foldl_holds (ReadsOf l) _ _ acc hacc ?_
    intro b pe hpe hb
    have hpe' : pe ∈ Spot.entries (Spot.getD s.registered "ports") := Array.mem_toList_iff.1 hpe
    obtain ⟨port, e⟩ := pe
    dsimp only
    cases hval : Spot.getD e "value" with
    | str v =>
      dsimp only
      split
      · rename_i k₀ r hfind
        intro k x hkx
        obtain ⟨⟨k', x'⟩, hmem, heq⟩ := Array.mem_map.1 hkx
        dsimp only at heq
        split at heq
        · rename_i hkv
          have hk'v : k' = v := by simpa using hkv
          obtain ⟨rfl, hx⟩ := Prod.mk.inj heq
          split at hx
          · subst hx
            have hk₀ : k₀ = v := by simpa using Array.find?_some hfind
            have hin : (v, r) ∈ b := hk₀ ▸ Array.mem_of_find?_eq_some hfind
            rw [hk'v]
            exact hb v r hin
          · exact Or.inl hx.symm
        · obtain ⟨rfl, rfl⟩ := Prod.mk.inj heq
          exact hb _ _ hmem
      · intro k x hkx
        rcases Array.mem_push.1 hkx with hkx | hkx
        · exact hb k x hkx
        · obtain ⟨rfl, rfl⟩ := Prod.mk.inj hkx
          exact Or.inr ⟨j, hj, s, hs, (port, e), hpe', hval, rfl⟩
    | _ => exact hb

/-- **A read the verifiers report is a statement's, and `readOk` passed it**: under a passing check, each
`(v, r)` the verifiers report either has a root that prints as null (a value reported under two roots is
reported so), or is the value and root of a registered read `e`, at a port `port`, of one of the statements, which
`SecondStage.readOk` passed against its template's part. -/
theorem reported_stated {allow : Bool} {own : Spot.SecondStage.Own} {ss : Array Json}
    (h : (Spot.SecondStage.check allow own (some ss)).codes = #[]) {v : String} {r : Json}
    (hv : (v, r) ∈ own.reported.getD #[]) :
    Spot.jeq r .null = true ∨
      ∃ ps, own.parts = some ps ∧ ∃ k < ss.size, ∃ s p port e,
        Spot.SecondStage.Stmt.ofJson? (ss.toList.getD k .null) = some s ∧
        ps.find? (·.template == s.template) = some p ∧
        (port, e) ∈ Spot.entries (Spot.getD s.registered "ports") ∧ Spot.getD e "value" = .str v ∧
        Spot.jeq r (Spot.getD e "root") = true ∧
        Spot.SecondStage.readOk own (Spot.SecondStage.byTemplate own.interior) p port e = true := by
  obtain ⟨-, -, ps, hp, -, -, hst, -, hrep⟩ := Spot.SecondStage.check_ok h
  unfold Spot.Record.sameDict at hrep
  simp only [Bool.and_eq_true] at hrep
  have hall := Array.all_eq_true_iff_forall_mem.1 hrep.2 (v, r) hv
  try dsimp only at hall
  obtain ⟨⟨k', r'⟩, hmem, hpred⟩ := Array.any_eq_true'.1 hall
  simp only [Bool.and_eq_true, beq_iff_eq] at hpred
  obtain ⟨hvk, hjr⟩ := hpred
  subst hvk
  rcases statementReads_reads ss.toList _ _ hmem with hnull | ⟨j, hj, s, hs, ⟨port, e⟩, hpe, hval, hroot⟩
  · subst hnull
    exact Or.inl hjr
  · right
    obtain ⟨k, hk, hkj⟩ := List.mem_iff_getElem.1 hj
    have hk' : k < ss.size := by simpa using hk
    have hget : ss.toList.getD k .null = j := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hk, Option.getD_some, hkj]
    obtain ⟨s', p, hs', hf, -, -, hc⟩ := hst k hk'
    have hss' : s' = s := by
      rw [hget, hs] at hs'
      exact (Option.some.inj hs').symm
    rw [hss'] at hf hc
    obtain ⟨-, -, -, -, -, -, hr, -, -, -, -⟩ := Spot.SecondStage.stmtCodes_nil hc
    have hok := Array.all_eq_true_iff_forall_mem.1 hr (port, e) hpe
    have hjr' : Spot.jeq r (Spot.getD e "root") = true := by
      have hroot' : Spot.getD e "root" = r' := hroot
      rw [hroot']
      exact hjr
    exact ⟨ps, hp, k, hk', s, p, port, e, by rw [hget]; exact hs, hf, hpe, hval, hjr', hok⟩

/-- **What `readOk` holds a read of one of the interior record's ports to**, arm by arm. At a port no link names (a
wire's reader, or the template's own port) the read is held to the interior record's entry of that port: its root,
domain and leaf count (`SecondStage.sameAt`). **The link-named case**: at a port a link names, the read is of the
link's value, held to the interior registration's registered value of that name where it registers one, else to the
boundary record's port of that name. Such a read is left out of the record's registered reads only where the interior
record has a port of the link's value's name at the root the read is held to. -/
theorem readOk_port {own : Spot.SecondStage.Own} {named : Bool} {p : Spot.SecondStage.Part} {port : String}
    {e : Json} {v : String} (hv : Spot.getD e "value" = .str v)
    (hp : ((Spot.Record.byPort own.interior).find? (·.1 == v)).isSome = true)
    (hok : Spot.SecondStage.readOk own named p port e = true) :
    (Spot.SecondStage.linkOf own p.template port = none ∧
        ∃ kx, (Spot.Record.byPort own.interior).find? (·.1 == v) = some kx ∧
          Spot.SecondStage.sameAt e kx.2 = true) ∨
      ∃ l, Spot.SecondStage.linkOf own p.template port = some l ∧ v = l.value ∧
        ((∃ y, Spot.SecondStage.registeredEntry own.interior v = some y ∧ Spot.SecondStage.sameAt e y = true) ∨
          (Spot.SecondStage.registeredEntry own.interior v = none ∧
            ∃ kb, (Spot.Record.byPort own.boundary).find? (·.1 == v) = some kb ∧
              Spot.SecondStage.sameAt e kb.2 = true)) := by
  unfold Spot.SecondStage.readOk at hok
  rw [hv] at hok
  cases hl : Spot.SecondStage.linkOf own p.template port with
  | some l =>
    right
    cases hw : Spot.SecondStage.wireOf own p.template port <;> rw [hl, hw] at hok <;>
      simp only [Bool.and_eq_true, beq_iff_eq] at hok <;> obtain ⟨hvl, hok⟩ := hok <;>
      refine ⟨l, rfl, hvl, ?_⟩ <;>
      (cases hre : Spot.SecondStage.registeredEntry own.interior v with
        | some y =>
          left
          rw [hre] at hok
          cases hrw : Spot.SecondStage.rowsOf own p.template port with
          | none => rw [hrw] at hok; simp at hok
          | some rs =>
            rw [hrw] at hok
            simp only [Bool.and_eq_true] at hok
            exact ⟨y, rfl, hok.1⟩
        | none =>
          right
          rw [hre] at hok
          refine ⟨rfl, ?_⟩
          cases hfb : (Spot.Record.byPort own.boundary).find? (·.1 == v) with
          | none => rw [hfb] at hok; simp at hok
          | some kb =>
            obtain ⟨k, b⟩ := kb
            rw [hfb] at hok
            cases hm : Spot.nat? (Spot.getD own.boundary "population") with
            | none => rw [hm] at hok; simp at hok
            | some m =>
              cases hrw : Spot.SecondStage.rowsOf own p.template port with
              | none => rw [hm, hrw] at hok; simp at hok
              | some rs =>
                rw [hm, hrw] at hok
                simp only [Bool.and_eq_true] at hok
                exact ⟨(k, b), rfl, hok.1⟩)
  | none =>
    left
    refine ⟨rfl, ?_⟩
    rw [hl] at hok
    cases hw : Spot.SecondStage.wireOf own p.template port with
    | some w =>
      rw [hw] at hok
      simp only [Bool.and_eq_true, beq_iff_eq] at hok
      obtain ⟨-, hok⟩ := hok
      cases hf : (Spot.Record.byPort own.interior).find? (·.1 == v) with
      | none => rw [hf] at hok; simp at hok
      | some kx =>
        obtain ⟨k, x⟩ := kx
        rw [hf] at hok
        cases hpw : Spot.SecondStage.writerOf own w with
        | none => rw [hpw] at hok; simp at hok
        | some pw =>
          cases hrw : Spot.SecondStage.rowsOf own p.template port with
          | none => rw [hpw, hrw] at hok; simp at hok
          | some rs =>
            rw [hpw, hrw] at hok
            simp only [Bool.and_eq_true] at hok
            exact ⟨(k, x), rfl, hok.1.1⟩
    | none =>
      rw [hw] at hok
      cases hf : (Spot.Record.byPort own.interior).find? (·.1 == v) with
      | none =>
        rw [hf] at hp
        simp at hp
      | some kx =>
        obtain ⟨k, x⟩ := kx
        simp only [hf, Bool.and_eq_true, beq_iff_eq] at hok
        exact ⟨(k, x), rfl, hok.1.2⟩

/-- **The read `(v, r)` is one the second stage's statement check held** at the run's input `i` and interior `it`:
the check with the form allowed reports no code at an `Own` whose interior record is `it`'s and whose boundary record
is `i`'s, and one of its statements states a registered read `e` of the value `v` at the root `r`, at a port `port`
of its template's part `p`, which `SecondStage.readOk` passed. -/
def HeldRead (i : Spot.Run.Input) (it : Spot.Run.Interior) (v : String) (r : Json) : Prop :=
  ∃ own ss ps, it.statements = some ss ∧ (Spot.SecondStage.check true own (some ss)).codes = #[] ∧
    own.interior = it.record ∧ own.boundary = i.record ∧ own.parts = some ps ∧
    ∃ k < ss.size, ∃ s p port e, Spot.SecondStage.Stmt.ofJson? (ss.toList.getD k .null) = some s ∧
      ps.find? (·.template == s.template) = some p ∧
      (port, e) ∈ Spot.entries (Spot.getD s.registered "ports") ∧ Spot.getD e "value" = .str v ∧
      Spot.jeq r (Spot.getD e "root") = true ∧
      Spot.SecondStage.readOk own (Spot.SecondStage.byTemplate it.record) p port e = true

/-- **Each read the verdicts report for the interior record is one the second stage's statement check held**: at an
accepted two-stage run, each of the reads the interior record's checks start from (the verdicts' own for the record's
ports and registered values, `Record.ownRegistered` of `Record.sessionRegistered`) is `HeldRead`, or its root prints
as null (a value the verdicts report under two roots is reported so). -/
theorem accepted_ownReads {i : Spot.Run.Input} {it : Spot.Run.Interior} {j : Json} (ha : AcceptedTwo i it j)
    {opened : Array (String × Json)}
    (ho : Spot.Record.ownRegistered it.record (Spot.Record.sessionRegistered i.verdicts) = some opened)
    {v : String} {r : Json} (hv : (v, r) ∈ opened) :
    Spot.jeq r .null = true ∨ HeldRead i it v r := by
  obtain ⟨-, workTable, resolved, draw, rest, read, -, -, -, -, grain, replay, k, ns, ids, ks, draw₂, read₂, strata,
    -, -, -, -, -, -, -, hc⟩ := accepted_steps ha
  obtain ⟨ss, hss⟩ := statements_some hc
  rw [hss] at hc
  have hrep : (v, r) ∈ (Spot.Record.sessionRegistered i.verdicts).getD #[] := by
    unfold Spot.Record.ownRegistered at ho
    cases hsr : Spot.Record.sessionRegistered i.verdicts with
    | none =>
      rw [hsr] at ho
      simp at ho
    | some reps =>
      rw [hsr] at ho
      simp only [Option.map_some, Option.some.injEq] at ho
      subst ho
      simp only [Option.getD_some]
      exact (Array.mem_filter.1 hv).1
  rcases reported_stated (own := runOwn i it ((Spot.Draw.proved draw).map Int.toNat) strata draw₂) hc hrep with
    hnull | ⟨ps, hp, k', hk', s, p, port, e, hs, hf, hpe, hval, hjr, hok⟩
  · exact Or.inl hnull
  · refine Or.inr ?_
    unfold HeldRead
    exact ⟨runOwn i it ((Spot.Draw.proved draw).map Int.toNat) strata draw₂, ss, ps, hss, hc, rfl, rfl, hp, k', hk',
      s, p, port, e, hs, hf, hpe, hval, hjr, hok⟩

/-- **The fact `Record.checks` rests on at the reads a two-stage run hands it**: the run of record and the form's
last step both call `Spot.Record.checks` for the interior record at
`Record.lessOwnReads rec (Record.ownRegistered rec (Record.sessionRegistered verdicts))`. At an accepted two-stage
run, each read `Record.lessOwnReads` takes out of the verdicts' own names one of the interior record's ports at that
port's root, and is `HeldRead`: the second stage's statement check held it through `SecondStage.readOk`
(`readOk_port` says to what, arm by arm). The other disjunct is a port of the interior record whose root prints as
null. -/
theorem accepted_leftOut {i : Spot.Run.Input} {it : Spot.Run.Interior} {j : Json} (ha : AcceptedTwo i it j)
    {opened : Array (String × Json)}
    (ho : Spot.Record.ownRegistered it.record (Spot.Record.sessionRegistered i.verdicts) = some opened)
    {v : String} {r : Json} (hv : (v, r) ∈ opened)
    (hno : (v, r) ∉ (Spot.Record.lessOwnReads it.record (some opened)).getD #[]) :
    ∃ x, (v, x) ∈ Spot.Record.byPort it.record ∧ Spot.jeq (Spot.getD x "root") r = true ∧
      (Spot.jeq (Spot.getD x "root") .null = true ∨ HeldRead i it v r) := by
  obtain ⟨x, hx, hxr⟩ := lessOwnReads_out hv hno
  refine ⟨x, hx, hxr, ?_⟩
  rcases accepted_ownReads ha ho hv with hn | hh
  · exact Or.inl (jeq_trans hxr hn)
  · exact Or.inr hh

end SpotFlock.RunsAs
