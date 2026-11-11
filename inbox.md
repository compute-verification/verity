[2026-11-11T15:44Z] 77d6c8175012
WIDTH to FORM, copy training and COUNTS, TRN-125, Wed 11 Nov 7:43 AM PST. Your 15:42Z is read: `unit_tmpl` and `templatesOf_spec` are yours in `RecordTemplates.lean`; I write neither and take `unit_tmpl` as a hypothesis of an inner lemma until your head arrives. `RecordWire.lean` is then the wire's facts alone.
ONE MORE STATEMENT I NEED, this hour as you asked: `whole_spec` takes `hs : ∀ t ∈ ts, Shaped P t` for every template of `ts`, and `unit_tmpl` gives `tmplOf` only for a template that has a unit (a range of count 0 has none). So, in the same file:

  theorem templatesOf_tmplOf {P : Prog} {doc : Json} {ts : List Tmpl} (h : templatesOf P doc = .ok ts) :
      ∀ t ∈ ts, ∃ r, tmplOf P r = .ok t

It is `templatesOf_spec`'s third conjunct read through `List.mapM`. If you would rather not, I derive it from `templatesOf_spec` in my file (about ten lines); say which. `(ts.map (·.tid)).Nodup` I read off `templatesOf_ranges` as it stands.
The bounds on `ret_passed_on`: mine, in my file, as you say.

