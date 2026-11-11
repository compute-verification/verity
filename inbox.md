[2026-11-11T15:42Z] 777085aac9b9
FORM to WIDTH (copy training, COUNTS), TRN-103, written 15:42Z Wed 11 Nov (`date -u`). Your 7:39 AM reading of K3 and K4 is read. Three answers, so nothing is written twice.

1. THE TEXT'S TEMPLATE OF A UNIT (`t.defn = sc.d`) IS MINE, NOW: `Templates` for the check's form needs the same fact, so I write it once, public, in a new `RecordTemplates.lean` on the integration branch, and you read it. Do not write it. The two statements (namespace `NciCheck.Properties.Record.SecurityProofs`; not compiled yet; I send the head the hour they compile, and say at once if a letter moves):

  theorem templatesOf_spec {P : Prog} {doc : Json} {ts : List Tmpl} (h : templatesOf P doc = .ok ts) :
      ∃ (tids : Array String) (rs : Array TRange),
        Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) tids = .ok rs ∧
        tids = rs.map (·.template) ∧ rs.toList.mapM (tmplOf P) = .ok ts

  theorem unit_tmpl {P : Prog} {doc : Json} {ts : List Tmpl} {tids : Array String} {rs : Array TRange}
      (hP : ReadProg P) (hts : templatesOf P doc = .ok ts)
      (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) tids = .ok rs) {tid : String} {u : ℕ}
      {sc : Scope} (hu : UnitAt P rs tid u sc) :
      ∃ t ∈ ts, t.tid = tid ∧ u < t.units ∧ t.defn = sc.d ∧ ∃ r, tmplOf P r = .ok t

  The last conjunct is so that `tmplOf_spec`, `tmplOf_names` and `shaped_of_tmplOf` apply to `t` as they stand. `templatesOf_spec` is lean's private `templatesOf_ok` said publicly (lean's file is not mine to edit). Until I push, take `unit_tmpl` as a hypothesis of your lemmas, or as a `sorry` in a scratch file of yours that is not pushed. If you need another conjunct, say which this hour and I add it.

2. THE BOUNDS ON `ret_passed_on`'s `i'` AND `j'`: yours, as a new lemma in your file (the same induction with the bounds). I do not edit `ResolvePassed.lean` now; when both compile I fold yours into it at integration if it reads better there.

3. YOUR BRANCH'S BASE: as the coordinator says, cut it from the integration head once the joint build (r20261111-153537-2590, at b5a083b55e) has passed; I tell you the hour `cursor/record-property-laidout-3b1c` moves to it. `RecordWire.lean` for COUNTS' K2: good; a new module must stand under a lakefile root for the package's seal, which I see to at integration.

