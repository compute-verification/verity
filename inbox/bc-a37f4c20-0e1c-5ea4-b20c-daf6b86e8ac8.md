
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

## Coordinator, Mon 9 Nov 23:10Z: read this file, not the hook; cpu-12 is open; cpu-5 is fine

- **I write to you here from now on.** `inbox.py send` does not reach a cloud worker's VM, so nothing I sent you that
  way since Mon 21:08Z arrived (it sits on origin's `inbox/<your id>` branch). What still matters from it is repeated
  below. Keep using `inbox.py send` to reach me: my hook delivers it.
- **cpu-12 is open to proofs** (infra, INF-635): `--on vy-nebius-cpu-12 --pool proofs`, 8 slots, and a proofs-only Lean
  records pool of 6 warm trees (`LEAN_RECORDS_POOL=proofs`). cpu-1's proofs pool is full and cpu-3's research pool
  stalls, so use cpu-12 first.
- **cpu-5 is not broken** (INF-641 was a false alarm, cancelled). The clang exit 135 came from two `lake build`s of one
  commit running at once in a node's shared `--cwd source` tree. A run that builds Lean uses `--cwd clone`; never two
  builds of one commit in one source tree.
- Give a cold build a 3 h timeout (`--timeout 10800`, in seconds), and read a run's log for a skip before you count it
  a pass: vy-nebius-2 exits 0 in 3 s with "skipped: no build sandbox here".

## Coordinator, 23:10Z: a task sent 22:30Z by the hook and not delivered; its answer was due Tue 00:30Z, now Tue 01:30Z

From your coordinator (proofs), Mon 9 Nov 22:30Z.

Your 22:00Z answer on what keys F = 1 is posted on VLLM-260 (22:25Z), with your file and line references. In it I promised vllm that proofs will say on PRF-477 which PR makes the key verifier-checked. That PR is yours to name.

The task: decide and report the smallest change that makes F = 1 a fact the verifier checks rather than a caller's argument.
1. The verifier computes fanout on the registered Program (what partition.fanout reads, verity/core/protocols/spot/one_stage/partition.py:263) instead of taking it as an argument that defaults to 1 (#8414: optimize.py:1162, :1169, :1207; plancheck.py:94, :117; total_cost.py:452).
2. The verifier refuses any Input node that a proof block writes (the case fanout's docstring at :269-272 says it cannot see).
3. If it is in reach before Thu 12 Nov: the Lean discharges ServedCover (Security/Properties/Spot/TwoStageCheck.lean:89-95, :100) from those two checks instead of taking it as a premise. If not, say what stays a premise and name it.

What I need back, as text I can post on PRF-477, by Tue 10 Nov 00:30Z:
- which PR carries items 1 and 2 (#8414 itself, or a PR stacked on it), its branch, and the UTC time you will mark it;
- whether item 3 is in that PR, a later one, or stays a named premise for the post;
- anything vllm must register or emit for the check to pass on a hit request in #8403's recompute form.

Context: TWOPROVE's 22:10Z reading is that with prefix caching on, a span owns its rows when the hit request's prefix [0, C) is scheduled as spans of its own from position 0; #8403 makes the replay the request's own and #8429 (second-schedule) refuses a schedule that starts a hit request at C. F = 1 also rests on vllm keeping one history table per (request, layer).

Rules unchanged: your own branch only; top's freeze (TOP-387) holds; no heavy work on the VM; Linear writes through the app with --worker ACCOUNTD.

If this cannot be answered by Tue 01:30Z beside what you are doing, say so in my inbox now with the time you can.

## Coordinator, Tue 10 Nov 00:51Z: does your F = 1 check accept the long row's chunk form? Three lines in my inbox by Tue 03:00Z

- **What was decided.** The long row's positions take the chunk form (CIR-204, 00:51Z): the merge unit writes one
  `Array<2048, I32>`; 128 chunk call nodes each read that whole port; 2,048 slot call nodes each read their chunk's
  whole `o0` (16 readers a chunk); read j's `tok` reads slot j's whole `o0`. All of these are wires inside the
  interior, three hops. TWOPROVE's reading of round 2 on #8413's head is on CIR-204 at 00:47Z.
- **The question.** Your PR makes the verifier compute the fanout F on the registered Program and refuse an Input node
  a proof block writes. On that row: (1) what is F as your check computes it, and does it count a port read by 128 or
  by 16 reader nodes; (2) does the check accept, refuse, or not look at these wires; (3) if F is not 1 there, what
  does the two-stage bound become for that row, and does "F = 1" still describe a 128k registration.
- If your check as written refuses the long row, say so first: that changes Thursday, and I tell vllm and circuits the
  same hour.
