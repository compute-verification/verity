module

public import Proofs.Properties.InferenceOnly.RecordForm
public import Proofs.Properties.InferenceOnly.RecordRoles
public import Proofs.Properties.InferenceOnly.RecordTemplates

/-! Every row the record check holds is registered, for the record check's own form (`form_provenance`). -/

@[expose] public section

namespace InferenceOnlyCheck.Properties.Record.SecurityProofs

open Lean (Json)
open Flock.Program (Prog)
open Spot.RecordText (Tmpl templatesOf)
open InferenceOnlyCheck.Record (layoutOf)
open InferenceOnlyCheck.Record.SecurityProofs (form front_text tmplOf_spec ranges_spec)

theorem tmpl_index {P : Prog} {doc : Json} {ts : List Tmpl} (hP : ReadProg P) (hts : templatesOf P doc = .ok ts)
    {t : Tmpl} (ht : t ∈ ts) : P.index[t.tid]? = some t.defn ∧ t.defn < P.defs.size := by
  obtain ⟨tids, rs, hev, -, hmapM⟩ := templatesOf_spec hts
  obtain ⟨r, hr, htm⟩ := (InferenceOnlyCheck.Record.SecurityProofs.Json.mapM_ok _ _ _ hmapM t).1 ht
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

theorem form_layoutReads : LayoutReads form := by
  intro d d' hp hr
  show (layoutOf d).map _ = (layoutOf d').map _
  rw [layoutOf_reads hp hr]

theorem form_provenance : Provenance form := by
  intro j tr c P pj hread hdec hparse
  have hread' : InferenceOnlyCheck.Record.read j tr = .ok c := hread
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

theorem form_pinnedOpens (hpins : DigestPins) : PinnedOpens form :=
  pinnedOpens_of hpins form_provenance form_layoutReads

end InferenceOnlyCheck.Properties.Record.SecurityProofs
