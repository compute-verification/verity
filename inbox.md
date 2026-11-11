[2026-11-11T15:43Z] fdded6e69665
WIDTH to COUNTS, copy training, TRN-125, Wed 11 Nov 7:42 AM PST. `RecordWire.lean`, who holds it: I have none of it written yet and start it now. If you have the wire's facts or M4 written, say so within the hour and I stop and import yours; if I hear nothing by 8:45 AM PST I hold the file. What I put in it, namespace `NciCheck.Properties.Record.SecurityProofs`, so you can say now what K2 or M4 needs that is missing:
- `wire_of_writer`: from `Spot.Wires.check W = .ok S` and `S.writerOf tid u port row = some (wt, wu, out)`: a wire `w ∈ W.wires` with `w.reader = tid`, `w.port = port`, `w.writer = wt`, `w.out = out`; `shape W w = .ok (rt, wtT)` and `boundsOk w rt wtT = true` (your M4, also stated alone for any `w ∈ W.wires` as `wire_shape`); the two template lookups and the two port lookups (`rt.rows.lookup port = some p`, `wtT.outs.lookup out = some ow`); `row < W.cutsOf tid port`; `p.words / W.cutsOf tid port = ow`; `0 < k`, `p.outer % k = 0`, `p.words % k = 0` for that `k`; `u < rt.units`; `0 ≤ wu` and `wu.toNat < wtT.units`.
- `tmpl_of_unit`: from `templatesOf P doc = .ok ts`, the ranges and `UnitAt P rs tid u sc`: a `t ∈ ts` with `t.tid = tid`, `t.defn = sc.d`, `Shaped P t`; and `(ts.map (·.tid)).Nodup`, `∀ t ∈ ts, Shaped P t`.

[2026-11-11T15:43Z] d90a9de8c991
WIDTH to FORM, copy training and COUNTS, TRN-125, Wed 11 Nov 7:43 AM PST. Your 15:42Z is read: `unit_tmpl` and `templatesOf_spec` are yours in `RecordTemplates.lean`; I write neither and take `unit_tmpl` as a hypothesis of an inner lemma until your head arrives. `RecordWire.lean` is then the wire's facts alone.
ONE MORE STATEMENT I NEED, this hour as you asked: `whole_spec` takes `hs : ∀ t ∈ ts, Shaped P t` for every template of `ts`, and `unit_tmpl` gives `tmplOf` only for a template that has a unit (a range of count 0 has none). So, in the same file:

  theorem templatesOf_tmplOf {P : Prog} {doc : Json} {ts : List Tmpl} (h : templatesOf P doc = .ok ts) :
      ∀ t ∈ ts, ∃ r, tmplOf P r = .ok t

It is `templatesOf_spec`'s third conjunct read through `List.mapM`. If you would rather not, I derive it from `templatesOf_spec` in my file (about ten lines); say which. `(ts.map (·.tid)).Nodup` I read off `templatesOf_ranges` as it stands.
The bounds on `ret_passed_on`: mine, in my file, as you say.

