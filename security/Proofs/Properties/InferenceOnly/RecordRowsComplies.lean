module

public import Properties.InferenceOnly.RecordRows
public import Properties.InferenceOnly.RowsHonest
public import Proofs.Properties.InferenceOnly.RecordComplies
public import Proofs.Properties.InferenceOnly.RecordBlockGates
public import Proofs.Properties.InferenceOnly.RecordTemplates
public import Proofs.Properties.InferenceOnly.RecordJson
public import Proofs.Properties.InferenceOnly.RecordUnits
public import Std.Data.TreeMap.Raw.Lemmas
import all Lean.Data.Json.Basic

/-!
# Accepting a record in rows gives the rate rule in rows, proved

`RowsComplies` (`Properties.InferenceOnly.RecordRows`). An acceptance of `InferenceOnly.RowsProtocol.checkBlobs` is the
record check reading the case, the rows table, a block's rows on the summary (`RecordRows.ofSummary`) and the rule in
rows giving no failure (`checkBlobs_accept`). The rule bounds the rows by `Γ` and the summary's fresh bits by `α` times
them (`form_rates` on the pair), and every block reads at most the summary's fresh bits (`form_counts`).

The one step of its own is that a block with a gate has the summary's rows (`rowsAt_summary`). Such a block is a slot
below the number of slots (`slot_lt`), where each template has its units over the slots (`tmpl_slot_units`), the
summary's units of it. The table is read by template id and the summary's units by Definition, and a template's id is
its Definition's key (`tmpl_key`), so the two counts agree term by term (`blockRows_congr`).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.Record.SecurityProofs

open Lean (Json)
open Flock.Program (Prog)
open Spot.RecordText (Tmpl templatesOf gridsOf slotIn)
open InferenceOnlyVerifier.Record.SecurityProofs (form form_owns form_laidOut form_rates read_front finish_text
  front_text front_inst ranges_spec tmplOf_spec)
open InferenceOnlyVerifier.Record.SecurityProofs.Json (mapM_ok)

/-! ## A block's rows, template by template -/

private theorem fold_none {κ : Type} (g : Option ℕ → κ × ℕ → Option ℕ) (hg : ∀ p, g none p = none) :
    ∀ l : List (κ × ℕ), l.foldl g none = none
  | [] => rfl
  | p :: l => by rw [List.foldl_cons, hg p]; exact fold_none g hg l

private theorem fold_congr {α κ κ' : Type} (m : κ → Option ℕ) (m' : κ' → Option ℕ) (f : α → κ) (f' : α → κ')
    (n : α → ℕ) (g : Option ℕ → κ × ℕ → Option ℕ) (g' : Option ℕ → κ' × ℕ → Option ℕ)
    (hg : ∀ t k a, g (some a) (t, k) = (m t).map fun b => a + k * b) (hg0 : ∀ p, g none p = none)
    (hg' : ∀ t k a, g' (some a) (t, k) = (m' t).map fun b => a + k * b) :
    ∀ (l : List α) (a r : ℕ), (∀ x ∈ l, ∀ b, m (f x) = some b → m' (f' x) = some b) →
      (l.map fun x => (f x, n x)).foldl g (some a) = some r →
      (l.map fun x => (f' x, n x)).foldl g' (some a) = some r
  | [], _, _, _, h => h
  | x :: l, a, r, hm, h => by
    rw [List.map_cons, List.foldl_cons, hg] at h
    rw [List.map_cons, List.foldl_cons, hg']
    cases hx : m (f x) with
    | none =>
      rw [hx, Option.map_none, fold_none g hg0] at h
      cases h
    | some b =>
      rw [hx, Option.map_some] at h
      rw [hm x List.mem_cons_self b hx, Option.map_some]
      exact fold_congr m m' f f' n g g' hg hg0 hg' l _ r (fun y hy => hm y (List.mem_cons_of_mem x hy)) h

/-- **Two counts in rows that agree term by term agree**: where every entry the first reads gives the second's entry
the same rows, a block's rows by the first are its rows by the second. -/
theorem blockRows_congr {α κ κ' : Type} (m : κ → Option ℕ) (m' : κ' → Option ℕ) (f : α → κ) (f' : α → κ')
    (n : α → ℕ) (l : List α) (hm : ∀ x ∈ l, ∀ b, m (f x) = some b → m' (f' x) = some b) {r : ℕ}
    (h : RecordRows.blockRows m (l.map fun x => (f x, n x)) = some r) :
    RecordRows.blockRows m' (l.map fun x => (f' x, n x)) = some r := by
  unfold RecordRows.blockRows at h ⊢
  exact fold_congr m m' f f' n _ _ (fun _ _ _ => rfl) (fun ⟨_, _⟩ => rfl) (fun _ _ _ => rfl) l 0 r hm h

/-- **A template's id is its Definition's key**: the template query names a root node's template by its callee's key,
and the template's Definition is that callee. -/
theorem tmpl_key {P : Prog} {doc : Json} {ts : List Tmpl} (hP : ReadProg P) (hts : templatesOf P doc = .ok ts)
    {t : Tmpl} (ht : t ∈ ts) : (P.defs[t.defn]!).key = t.tid := by
  obtain ⟨tids, rs, hev, -, hmapM⟩ := templatesOf_spec hts
  obtain ⟨r, hr, htm⟩ := (mapM_ok (Spot.RecordText.tmplOf P) _ _ hmapM t).1 ht
  obtain ⟨hlt, htmpl, -⟩ := (ranges_spec (read_bodies hP).1 hP.2.2.2.2.2.2 (readProg_inputs hP) hev).2 r
    (by simpa using hr)
  obtain ⟨hd, htid, -⟩ := tmplOf_spec htm
  rw [hd, htid, htmpl, Array.getElem?_eq_getElem hlt, getElem!_pos (P.defs[P.root]!).body.nodes _ hlt]
  rfl

/-- **The rows of one unit of a template's Definition are the table's entry at its id**, where the Definition is
one of the Program's. -/
theorem rowsOfDefn_tmpl {P : Prog} {doc : Json} {ts : List Tmpl} (hP : ReadProg P) (hts : templatesOf P doc = .ok ts)
    (m : String → Option ℕ) {t : Tmpl} (ht : t ∈ ts) {b : ℕ} (h : RecordRows.rowsOfDefn P m t.defn = some b) :
    m t.tid = some b := by
  unfold RecordRows.rowsOfDefn at h
  cases hx : P.defs[t.defn]? with
  | none => rw [hx] at h; cases h
  | some x =>
    rw [hx, Option.bind_some] at h
    obtain ⟨hlt, hx'⟩ := Array.getElem?_eq_some_iff.1 hx
    have hkey := tmpl_key hP hts ht
    rw [getElem!_pos P.defs _ hlt, hx'] at hkey
    rw [← hkey]
    exact h

/-! ## The case the check reads -/

private theorem getObjVal?_mkObj {l : List (String × Json)} (hd : l.Pairwise (fun a b => ¬ compare a.1 b.1 = .eq))
    {k : String} {v : Json} (hm : (k, v) ∈ l) : (Json.mkObj l).getObjVal? k = .ok v := by
  simp only [Json.mkObj, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?]
  rw [Std.TreeMap.Raw.getElem?_ofList_of_mem (Std.ReflCmp.compare_self) hd hm]
  rfl

/-- The instance the rows check reads holds the params' Program. -/
theorem rows_instance_program (p : InferenceOnly.RowsProtocol.Params) (e : InferenceOnly.RowsProtocol.Evidence) :
    (InferenceOnly.RowsProtocol.instanceOf p e).getObjVal? "program" = .ok (.str p.program) := by
  obtain ⟨record, _ | o⟩ := e <;> exact getObjVal?_mkObj (by simp) (by simp [InferenceOnly.RowsProtocol.instanceOf])

/-- The instance the rows check reads holds the evidence's record. -/
theorem rows_instance_record (p : InferenceOnly.RowsProtocol.Params) (e : InferenceOnly.RowsProtocol.Evidence) :
    (InferenceOnly.RowsProtocol.instanceOf p e).getObjVal? "record" = .ok (.str e.record) := by
  obtain ⟨record, _ | o⟩ := e <;> exact getObjVal?_mkObj (by simp) (by simp [InferenceOnly.RowsProtocol.instanceOf])

/-- **What the rows check's acceptance is**: the record check reads the case, the files give the rows table, the
summary's units give a block's rows, and the rule in rows gives no failure on them and the summary's fresh bits. -/
theorem checkBlobs_accept {p : InferenceOnly.RowsProtocol.Params} {c : InferenceOnlyVerifier.Protocol.Commitment}
    {e : InferenceOnly.RowsProtocol.Evidence} {bs : Core.Blobs} {pub : Unit} {prof : Json}
    (h : InferenceOnly.RowsProtocol.checkBlobs p c e bs = .accept pub prof) :
    ∃ (r : InferenceOnlyVerifier.Record.Case) (table : Std.HashMap String ℕ) (rows : ℕ),
      InferenceOnlyVerifier.Record.read (InferenceOnly.RowsProtocol.instanceOf p e)
        (InferenceOnly.RowsProtocol.trustedOf p c) = .ok r ∧
      InferenceOnly.RowsProtocol.tableOf p.circuits bs = .ok table ∧
      RecordRows.ofSummary r.f.P (table[·]?) (InferenceOnlyVerifier.Record.summarize r) = some rows ∧
      RecordRows.rate p.gamma p.num p.den rows (InferenceOnlyVerifier.Record.summarize r).fresh = none := by
  unfold InferenceOnly.RowsProtocol.checkBlobs at h
  dsimp only at h
  split at h
  · cases h
  rename_i r hr
  split at h
  · cases h
  split at h
  · cases h
  rename_i table ht
  split at h
  · cases h
  rename_i rows hrows
  split at h
  · cases h
  rename_i hrate
  exact ⟨r, table, rows, hr, ht, hrows, hrate⟩

/-! ## A block with a gate has the summary's rows -/

/-- **A block with a gate is a slot below the number of slots**: its gate lies in a unit, whose slot is. -/
theorem slot_lt {j tr : Json} {r : InferenceOnlyVerifier.Record.Case}
    (hread : InferenceOnlyVerifier.Record.read j tr = .ok r) {d : Verity.RegisteredCircuit} (hd : Served form r d)
    {u : Fin d.n} (hu : d.𝓘.gates u ≠ ∅) : u.val < r.S := by
  obtain ⟨P, doc, W, Sw, rs, at_, hdec, -, -, -, -, -, -, -, hrs, -, -, -, -, -, -, -, hDiv⟩ := hd
  have hPc : r.f.P = P := Except.ok.inj ((decode_of_read hread).symm.trans hdec)
  subst hPc
  obtain ⟨f, pin, hf, hc⟩ := read_front hread
  obtain ⟨rfl, hgr, -⟩ := finish_text hc
  have hts := front_text hf
  have hP := readProg_of_read hread
  have hnd := (tmpls_named hts).1
  have hDiv' : Divides r.f.P rs (slotIn r.S r.grids) at_ d.𝓘 := hDiv
  obtain ⟨x, hx⟩ := Finset.nonempty_iff_ne_empty.2 hu
  have hxu : d.𝓘.unit x = some u := by
    simpa only [Verity.Partition.gates, Finset.mem_filter, Finset.mem_univ, true_and] using hx
  have hxi : d.D.isInput x = false := by
    cases hb : d.D.isInput x with
    | false => rfl
    | true =>
      have hnone := (d.𝓘.input_iff x).2 hb
      rw [hxu] at hnone
      exact absurd hnone (by simp)
  obtain ⟨tid, v, sc, s, hunit, -, -, hs, hmapu⟩ := hDiv' x hxi
  obtain ⟨t, ht, htid, hlt, -, -⟩ := unit_tmpl hP hts hrs hunit
  rw [← htid] at hs
  obtain ⟨s', hs', hlt'⟩ := tmpl_slot_lt hgr hnd ht hlt
  rw [hxu] at hmapu
  have h1' : some u.val = some s := hmapu
  have h1 : u.val = s := Option.some.inj h1'
  have h2 : s = s' := Option.some.inj (hs.symm.trans hs')
  omega

/-- **A slot below the number of slots has the summary's rows**: on a record the check reads, block `u`'s rows on the
record's text are the summary's, by the table's entries. -/
theorem rowsAt_summary {j tr : Json} {r : InferenceOnlyVerifier.Record.Case}
    (hread : InferenceOnlyVerifier.Record.read j tr = .ok r) (m : String → Option ℕ) {u rows : ℕ} (hu : u < r.S)
    (h : RecordRows.ofSummary r.f.P m (InferenceOnlyVerifier.Record.summarize r) = some rows) :
    rowsAt r.f.P r.f.doc m u = some rows := by
  obtain ⟨f, pin, hf, hc⟩ := read_front hread
  obtain ⟨rfl, hgr, -⟩ := finish_text hc
  have hts := front_text hf
  have hP := readProg_of_read hread
  have hnd := (tmpls_named hts).1
  have hslot : ∀ tid v, Spot.RecordText.slot r.f.P r.f.doc tid v = slotIn r.S r.grids tid v := by
    intro tid v
    unfold Spot.RecordText.slot
    simp only [hts, hgr]
  have hcount : ∀ t ∈ r.f.ts,
      (t.tid, ((List.range t.units).filter fun v => Spot.RecordText.slot r.f.P r.f.doc t.tid v == some u).length) =
        (t.tid, t.units / r.S) := by
    intro t ht
    simp only [hslot]
    rw [tmpl_slot_units hgr hnd ht hu]
  unfold rowsAt
  rw [hts]
  dsimp only
  rw [List.map_congr_left hcount]
  have h' : RecordRows.blockRows (RecordRows.rowsOfDefn r.f.P m)
      (r.f.ts.map fun t => (t.defn, t.units / r.S)) = some rows := h
  exact blockRows_congr (RecordRows.rowsOfDefn r.f.P m) m (·.defn) (·.tid) (fun t => t.units / r.S) r.f.ts
    (fun t ht b hb => rowsOfDefn_tmpl hP hts m ht hb) h'

/-! ## The guarantee -/

/-- **Accepting a record in rows gives the rate rule in rows on its circuit.** -/
theorem RowsComplies : Properties.Record.RowsComplies := by
  intro p c e bs pub prof h
  unfold Properties.Remote.RowsHeld
  obtain ⟨r, table, rows, hread, htable, hrows, hrate⟩ := checkBlobs_accept h
  have hR := form_rates p.gamma p.num p.den ⟨rows, (InferenceOnlyVerifier.Record.summarize r).fresh⟩ hrate
  refine ⟨r, table, hread, htable, form_laidOut _ _ r hread, fun d hd u hu => ?_⟩
  have hfresh := (form_counts _ _ r hread d hd u).1
  have hown := form_owns _ _ r hread
  have hprog : r.f.i.program = p.program :=
    Json.str.inj (Except.ok.inj (hown.1.symm.trans (rows_instance_program p e)))
  have hrec : r.f.i.record = e.record :=
    Json.str.inj (Except.ok.inj (hown.2.1.symm.trans (rows_instance_record p e)))
  obtain ⟨f, pin, hf, hc⟩ := read_front hread
  obtain ⟨rfl, -, -⟩ := finish_text hc
  refine ⟨r.f.P, r.f.doc, rows, ?_, ?_, rowsAt_summary hread _ (slot_lt hread hd hu) hrows, hR.1, ?_⟩
  · rw [← hprog]
    exact decode_of_read hread
  · rw [← hrec]
    exact (front_inst hf).2
  · exact Nat.le_trans (Nat.mul_le_mul_right _ hfresh) hR.2

/-- A case whose `α` has denominator 0. -/
def zeroDen : InferenceOnly.RowsProtocol.Params := ⟨0, 0, 0, #[], #[], #[], ""⟩

theorem zeroDen_parse :
    InferenceOnlyVerifier.Record.parse (InferenceOnly.RowsProtocol.instanceOf zeroDen ⟨"", none⟩) = .error "params" := by
  have hg : (InferenceOnly.RowsProtocol.instanceOf zeroDen ⟨"", none⟩).getObjVal? "gamma" = .ok (Lean.toJson (0 : ℕ)) :=
    getObjVal?_mkObj (by simp) (by simp [InferenceOnly.RowsProtocol.instanceOf, zeroDen])
  have ha : (InferenceOnly.RowsProtocol.instanceOf zeroDen ⟨"", none⟩).getObjVal? "alpha" =
      .ok (Json.arr #[Lean.toJson (0 : ℕ), Lean.toJson (0 : ℕ)]) :=
    getObjVal?_mkObj (by simp) (by simp [InferenceOnly.RowsProtocol.instanceOf, zeroDen])
  have hp : (InferenceOnly.RowsProtocol.instanceOf zeroDen ⟨"", none⟩).getObjVal? "pins" = .ok (Json.arr #[]) :=
    getObjVal?_mkObj (by simp) (by simp [zeroDen])
  unfold InferenceOnlyVerifier.Record.parse InferenceOnlyVerifier.field
  rw [rows_instance_program, rows_instance_record, hg, ha, hp]
  rfl

/-- **`RowsComplies`' honest witness**: the compiled rows check accepts the honest case (`RowsHonestAccepted`), and
there its conclusion holds. -/
theorem RowsComplies_honest (h : RowsHonestAccepted) :
    ∃ (p : InferenceOnly.RowsProtocol.Params) (c : InferenceOnlyVerifier.Protocol.Commitment)
      (e : InferenceOnly.RowsProtocol.Evidence) (bs : Core.Blobs) (pub : Unit) (prof : Json),
      (Core.canonicalJson RowsHonest.params.toUTF8 >>= InferenceOnly.RowsProtocol.paramsOf) = .ok p ∧
        InferenceOnlyVerifier.Protocol.commitmentOf RowsHonest.commitment.toUTF8 = .ok c ∧
        InferenceOnly.RowsProtocol.evidenceOf RowsHonest.evidence.toUTF8 = .ok e ∧
        InferenceOnly.RowsProtocol.checkBlobs p c e bs = .accept pub prof ∧ _root_.Properties.Remote.RowsHeld p c e bs := by
  obtain ⟨p, c, e, bs, pub, prof, hp, hc, he, ha⟩ := h
  exact ⟨p, c, e, bs, pub, prof, hp, hc, he, ha, RowsComplies p c e bs pub prof ha⟩

/-- **`RowsComplies` fails without `InferenceOnly.RowsProtocol.checkBlobs`.** At a case whose `α` has denominator 0,
which a verifier that accepted every case would accept, the record check reads no record, so `RowsHeld` fails. -/
theorem RowsComplies_nontrivial :
    ¬ _root_.Properties.Remote.RowsHeld zeroDen ⟨.null, "", .empty⟩ ⟨"", none⟩ Core.Blobs.empty := by
  rintro ⟨r, _, h, -⟩
  unfold InferenceOnlyVerifier.Record.read InferenceOnlyVerifier.Record.front at h
  rw [zeroDen_parse] at h
  cases h

end InferenceOnlyVerifier.Properties.Record.SecurityProofs
