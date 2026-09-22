"""[R8 2026-09-18, lane C-codec] Advice codecs A0-A3 for the bounded-advice accounting `P = G(phi, a)`.

Experimental, isolated: nothing here is imported by the capture/verify pipeline.  See out/gen/r8/aot-codec/REPORT.md.

  facts.py        ScheduleHint extraction from an archived record (capture/log.jsonl.gz StepStart events + spec.json + experiment.json)
                  and the base-context variants phi0 / phi1a / phi1 / phi2.
  bitio.py        bit writer/reader, fixed-width + Elias-gamma + Rice integer codes, exact (big-integer) arithmetic coder.
  sched.py        deterministic pre-execution scheduler model (the predictor used by A2/A3).
  codecs.py       A0 explicit, A1 remove-derivable, A2 predict+residual, A3 entropy-coded (fixed model) -- encode(phi,h)->bits, decode(phi,a)->h.
  model.py        A3 probability model: fitted on training runs, frozen to JSON, loaded by the decoder.
  evaluate.py     corpus driver -> results.jsonl + tables.
"""
