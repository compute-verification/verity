module

public import Proofs.Properties.Nci.RecordForm
public import Proofs.Properties.Nci.RecordRoles
public import Proofs.Properties.Nci.RecordTemplates

/-!
# Every row the record check holds is registered, for the record check's own form

`form_provenance` (`Properties.Record.Provenance`): of a record the record check reads, the trusted input holds a
chained log and names the run, one of the log's records, by its digest; the run names the Program and the record's
text by their SHA-512s; and the reason the check gives for each row it holds is good (`Properties.Record.Good`).

The log and the run are the trusted input's own (`front_trusted`, `trusted_run`). A reason's parameter is θ
(`heldAt_theta`, `case_theta`, `isTheta_root`), and the log is not empty since it holds the run. For a row held as an
opening's port (`heldAt_opened`, with `read_pin` and `pinOf_some`): the pin is one of the instance's and is the
reader's closure digest in the Program's descriptor; the reader's template is one of the text's, whose id is its
Definition's key in the Program's index (`tmpl_index`); that Definition has the layout, by which the port is an
opened row and the key is none (`layoutOf_key`); and the same unit's root and domain id rows are held as table reads
of the same table.

`tmpl_index`: a template of the text is the text's reading of a range of the root (`templatesOf_spec`), whose id is
the key of the Definition its root node calls (`ranges_spec`, `tmplOf_spec`); the callee is a Definition before the
root, and a decoded Program's index sends each Definition's key to its own place (`DecodeKeys.decode_keys`).

`form_layoutReads` (`Properties.Record.LayoutReads`): the form's layout reads a Definition's parameters and return
type alone (`layoutOf_reads`).

`form_pinnedOpens` (`Properties.Record.PinnedOpens`): the reader of an opening the check holds under a pin is the pinned
Definition, for the form, from the closure digest's pin (`Properties.Record.DigestPins`, by `pinnedOpens_of`).
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Lean (Json)
open Flock.Program (Prog)
open NciCheck.RecordText (Tmpl templatesOf)
open NciCheck.Record (layoutOf)
open NciCheck.Record.SecurityProofs (form front_text tmplOf_spec ranges_spec)

/-- **A template's id is its Definition's key in the Program's index**, of the templates the record's text gives of a
Program the record check reads; and its Definition is one of the Program's. -/
theorem tmpl_index {P : Prog} {doc : Json} {ts : List Tmpl} (hP : ReadProg P) (hts : templatesOf P doc = .ok ts)
    {t : Tmpl} (ht : t ∈ ts) : P.index[t.tid]? = some t.defn ∧ t.defn < P.defs.size := by
  obtain ⟨tids, rs, hev, -, hmapM⟩ := templatesOf_spec hts
  obtain ⟨r, hr, htm⟩ := (NciCheck.Record.SecurityProofs.Json.mapM_ok _ _ _ hmapM t).1 ht
  obtain ⟨hnode, htpl, -⟩ := (ranges_spec (read_bodies hP).1 hP.2.2.2.2.2.2 (readProg_inputs hP) hev).2 r hr
  obtain ⟨hdefn, htid, -⟩ := tmplOf_spec htm
  obtain ⟨⟨bytes, hdec⟩, -⟩ := hP
  obtain ⟨hroot, -⟩ := SecurityProofs.BodiesOk.decode_root bytes P hdec
  obtain ⟨st, hPd, hinv, -⟩ := SecurityProofs.BodiesOk.decode_spec bytes P hdec
  have hok : ∀ i < P.defs.size, SecurityProofs.BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hPd]; exact hinv.1
  have hfn : ((P.defs[P.root]!).body.nodes[r.node]!).fn < P.root := ((hok P.root hroot).2.1 r.node hnode).1
  have hd : t.defn = ((P.defs[P.root]!).body.nodes[r.node]!).fn := by
    rw [hdefn, Array.getElem?_eq_getElem hnode, getElem!_pos (P.defs[P.root]!).body.nodes _ hnode]
    rfl
  have hlt : t.defn < P.defs.size := by rw [hd]; exact Nat.lt_trans hfn hroot
  refine ⟨?_, hlt⟩
  rw [htid, htpl, ← hd]
  exact (SecurityProofs.DecodeKeys.decode_keys bytes P hdec).2 t.defn hlt

/-- **The layout reads the signature alone**, for the record check's own form. -/
theorem form_layoutReads : LayoutReads form := by
  intro d d' hp hr
  show (layoutOf d).map _ = (layoutOf d').map _
  rw [layoutOf_reads hp hr]

/-- **Every row the check holds is registered**, for the record check's own form. -/
theorem form_provenance : Provenance form := by
  intro j tr c P pj hread hdec hparse
  have hread' : NciCheck.Record.read j tr = .ok c := hread
  obtain ⟨desc, hdesc, hf, hc⟩ := read_pin hread'
  have hpj : desc = pj := hdesc pj hparse
  subst hpj
  have hP : c.f.P = P := Except.ok.inj ((decode_of_read hread').symm.trans hdec)
  subst hP
  obtain ⟨hT, hprog, hcirc⟩ := front_trusted hf
  obtain ⟨hlog, hch, hrun, rdig, hdig, hname⟩ := trusted_run hT
  have htheta : ∀ p, c.isTheta p = true → IsTheta c.f.T.log c.f.T.log[c.f.T.run]! p := by
    intro p hp
    obtain ⟨leaves, hl⟩ := case_theta hc hp
    exact ⟨Nat.zero_lt_of_lt hrun, isTheta_root hl⟩
  refine ⟨c.f.T.log, c.f.T.log[c.f.T.run]!, ⟨hlog, hch, c.f.T.run, hrun, rfl, rdig, hdig, hname⟩,
    ⟨hprog, hcirc⟩, ?_⟩
  intro r h hheld
  have hheld' : c.heldAt r = some h := hheld
  have hth := htheta _ (heldAt_theta hheld')
  cases h with
  | read p => exact hth
  | table name p root => exact hth
  | opened name p pin =>
    obtain ⟨hpin, t, l, ht, hl, hport, hroot, hdom⟩ := heldAt_opened hc hheld'
    obtain ⟨hdigest, hmem⟩ := pinOf_some hpin
    have htid : t.tid = r.tid := by simpa using List.find?_some ht
    obtain ⟨hidx, hlt⟩ := tmpl_index (readProg_of_read hread') (front_text hf) (List.mem_of_find?_eq_some ht)
    refine ⟨hth, hmem, hdigest, t.defn, c.f.P.defs[t.defn]!, ⟨l.key, l.root, l.domain, l.ports, l.verdict⟩, ?_, ?_,
      ?_, hport, layoutOf_key hl, hroot, hdom⟩
    · rw [← htid]
      exact hidx
    · rw [Array.getElem?_eq_getElem hlt, getElem!_pos c.f.P.defs _ hlt]
    · show (layoutOf _).map _ = _
      rw [hl]
      rfl

/-- **The reader of a held opening is the pinned Definition**, for the record check's own form, from the closure
digest's pin. -/
theorem form_pinnedOpens (hpins : DigestPins) : PinnedOpens form :=
  pinnedOpens_of hpins form_provenance form_layoutReads

end NciCheck.Properties.Record.SecurityProofs
