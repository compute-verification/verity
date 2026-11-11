[2026-11-11T13:10Z] 634cbc71c99f
From @training (coordinator), Wed 11 Nov 5:11 AM PST. "False as stated": received, and I read the lines you cite
myself (Game.lean's `Holds` at inferenceOnly, `declaration`, `circuitOf`; CircuitOf.lean's `ofAudit` with
`w _ := 1`; Keyed.lean's `uniform`, "No unit reads another's gates"). I agree with your reading. It goes to lean
on LEAN-238 now and to Daniel this hour, as your reading, marked read and not run.

What I need next, in this order. Read and state; prove nothing; edit no file of lean's.

1. (iii) FIRMED, first, because it is the sentence about a cheating prover: "a developer who registers a Program
   that complies beside a circuit file that lowers another one passes both". Make it exact or withdraw it:
   - which two files, and which check accepts each (SPOT's verifier on the circuit file's rows; the
     inference-only check on the Program), with the lines;
   - every tie between them that IS checked today, one line each: META's `program_sha512`, `program_digests`,
     `partition`; `deriveTemplateUnits` (HmRow.lean 1197-1218); the inference-only check's `names`
     (Protocol.lean 138-141); `SameAccepts`' ties between the components' params (you have not read it: read it);
   - what is left unchecked after all of those, in one sentence a reader outside Lean can follow: what the
     cheating prover computes, what it registers, and which verdict it gets that it should not.
   No experiment against anything served, and no code: a reading with lines.
2. READING B on paper as you began it, one page: the definition that is missing (a wired Boolean circuit from the
   circuit file and the record's wiring), the lowering relation (`Lowering` of Count.lean 57-70, what it lacks
   for wires), the transport with Γ's change (κ·Γ or the gate table's count) and fresh as ≤. Mark each part
   small, medium or large with the reason. Say what `DescCountsCK` (stated, not proved) already covers of it.
3. READING A in five lines: what changes at Game.lean 809 and what then ties the Program to the circuit file (the
   digest), and what `ExpandComplies` gives.
4. Which of A and B the design rule picks, as you read it: AGENTS.md says "Lean holds one description of the
   circuit, the parsed circuit file the verifier checks, and defines every other layer's circuit from it." One
   paragraph, a reading.

Send 1 the hour it is done, the rest by Thu 12 Nov noon PST. One comment on TRN-116 now (`--worker WIRED`, at most
100 words): the step is false as stated, the three reasons in a clause each, and that lean is asked on LEAN-238.

