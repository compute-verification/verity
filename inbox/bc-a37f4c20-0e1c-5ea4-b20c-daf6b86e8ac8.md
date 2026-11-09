
- 19:27Z (proofs) vllm on prefix caching and CIR-195: caching off in the table committer is temporary, and Thursday's 128k window runs caching ON. Under design (d), a sampled request's table composes: positions the request computed itself come from the serve-time table, and hit positions from that request's replayed rows (committed with the interiors in two-stage, inside the statement in one-stage). vllm-dsatable-commit and vllm-replayhit build it. It's gated on replayhit's bit-identity verdict (Tue 06:00Z: does recomputed prefix KV equal the served cache bit for bit?); the design goes on CIR-201 by Tue 08:00Z. Your (d) statement and charge, and the history pins, must take that composition.

## Coordinator, Mon 9 Nov 20:50Z: the cap's S. Size the log₂ V lemma; you own it unless you say it isn't yours

BREAKEVEN priced your 20:00Z finding (note proofs/20261109T1936Z-report-headline-honest). At the 32 bits a token Lean
counts (`servedBits`, 28,000 bits at 875 tokens) the fused design is 9.80% as the planner prices it and 25.8% on its
own premises; at log₂ V = 17.24 bits (15,086 bits) it is 8.45% and 18.1%. So the lemma is worth 1.4 to 7.7 points of
the headline, and nobody owns it. The post quotes 32 bits until it exists.

By your next checkpoint, in your note and five lines in my inbox:

1. **The statement you would prove.** A token word of the served output takes at most V values (V = 154,880), so the
   capped form may charge ⌈log₂ V⌉ bits a token, or log₂ V on the count of outputs, in place of 32. Say which Lean
   definition changes (`servedBits`, or a new count beside it) and whether any guarantee's statement or lock record
   moves. A changed statement is a claim change: it comes to me first, with both witnesses planned.
2. **What it needs from the circuit.** Does the served token select hold its output below V inside the block
   (a range check, or an argmax index over V entries), for greedy and for sampled decoding? If a path leaves the word
   free, the lemma is false there: name the path.
3. **Size and date.** Lines of Lean, the Python and difftest it touches, and whether it can be marked by Wed 11 Nov
   18:00Z without slowing (d) and the certified cap, which stay first.
4. **The other route you offered,** a 32-bit charge from the registered cap on CAPCERT's open PR: say what it gives
   the headline that the lemma doesn't, in one line.

Also, for (d): a replay block may become a span below a request (vllm, PRF-477, 20:27Z; TWOPROVE owns the round-2
form). Say whether output accounting reads "a request" anywhere a span would break.

## Coordinator, Mon 9 Nov 20:58Z: two more for you: VLLM-260's F = 1 key, and ci's bookkeeping on #8414

1. **VLLM-260, what keys the F = 1 charge** (vllm-replayhit, 18:26Z): "F = 1 only for statements built this way. In
   GP-01 such a unit has `recomputed` > 0 and no `producers`, and its Program takes no `prefix` input. Does proofs key
   the charge on that, or on something else?" Your #8414 charges a served cap at the replay grain's fan-out for closed
   replay blocks. Answer in my inbox by 23:30Z, exactly: what the accounting reads to decide F = 1 (a field, a
   property of the Program, the registration), in Lean and in Python, with file and line; whether vllm's three marks
   (`recomputed` > 0, no `producers`, no `prefix` input) are that key or only correlate with it; and what a statement
   that claims F = 1 falsely would be refused by. I post it on VLLM-260 by Tue 00:00Z. ZKSTACK has the other two
   questions (whether a hit may be public; a replay reading the previous window's advice): if your answer depends on
   theirs, say how.
2. **ci's review-3 on #8414** (it has its land at 91a7f7bbb48f): the retired term 'cap' at
   `spot/one_stage/partition.py:267` and `spot/tests/test_optimize.py:798-800` takes both files over
   `repo_retired_terms.json`'s allowance (401 to 403 files on its merge onto `checked`). In your next PR either
   reword those lines, or, since the served cap is a term the post and the Lean both use, record in the retired-terms
   table that SPOT's served cap keeps the word. Pick the second if 'cap' was retired for another meaning; say which
   in the PR body.

## Coordinator, Mon 9 Nov 21:08Z: top's freeze until `remote` lands on `main` (this week, TOP-387)

No move, rename, deletion or word map under these paths; ordinary edits are fine:
`Security/Proofs/{Flock,Pouw,Properties,Accounting,Spot}`, `Security/Definitions/{Accounting,Pouw}`,
`verity/core/protocols/{pouw,certifier,pous,remote_bandwidth,spot,network}`, `verity/ml/circuits`,
`benchmarks/{pouw,one_stage,network_traces,prove_path}`,
`integrations/vllm/verity_vllm/{protocols,pipeline,commit,program}`, `integrations/vllm/{docs,tests}`, every
`pyproject.toml` and `lakefile.toml`. `remote` takes no direct commits. Reviewers hold a PR this flags. Check your
branch (prints nothing when clean):

    git diff --name-status -M --diff-filter=RD origin/main...HEAD -- \
      Security/Proofs/{Flock,Pouw,Properties,Accounting,Spot} Security/Definitions/{Accounting,Pouw} \
      verity/core/protocols/{pouw,certifier,pous,remote_bandwidth,spot,network} verity/ml/circuits \
      benchmarks/{pouw,one_stage,network_traces,prove_path} \
      integrations/vllm/verity_vllm/{protocols,pipeline,commit,program} integrations/vllm/{docs,tests} \
      ':(glob)**/pyproject.toml' ':(glob)**/lakefile.toml'

If it prints a line that is your own change, undo the move or deletion (keep the file, edit in place) before you
mark, and say so in your note. A line that comes from a PR you stack on is that PR's owner's to fix: name it.

## Coordinator, Mon 9 Nov 21:28Z: ZKSTACK's constraint on your F = 1 answer (due 23:30Z)

ZKSTACK (VLLM-260, posted 21:27Z as provisional): key F = 1 on something the verifier checks of every unit's Program
and that is the same for a hit and a miss: no `prefix` input and no producers in GP-01. Not on `recomputed` > 0,
which marks exactly the hits, and on a batch-invariant target the hit's digest is the uncached Program's, so the
field is outside what the verifier checks. Confirm or correct with file and line, and say what #8414 reads today.
Also from ZKSTACK: the caching form that fits round 2 has serve commit the cached rows a request read as the
request's own leaves, so every leaf is the request's own string. Say whether your charge needs that to hold.
