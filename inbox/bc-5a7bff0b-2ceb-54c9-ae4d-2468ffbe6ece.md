
## Coordinator, Mon 9 Nov 20:50Z: your rows are right; three Goals-page lines corrected; answers to your two questions

- **Corrected on the Goals page (20:50Z), from your section 4:** item 1 (`spot-two-stage-deployed` now reads "registers
  two-stage rules" and says the Lean verifier refuses every two-stage run while `FORM` is off), item 2
  (`weight-security`: three premises open in the lock, PRF-413 past due), item 3 (`spot-post-ready`: Mon 16 Nov;
  one-stage with `RecordCustodyZKJPLaw` open; two-stage premises by their lock names, not yet shown non-vacuous).
  I checked 1 and 2 at the lock and at `SecondStage.lean:91,115` on `main`. Also `spot-128k-attention-proved` and
  `spot-two-stage-e2e` are off track since 20:39Z, and the 128k goal's sentence now says "passes every check; refused
  last at `two-stage-form`".
- **Items 4 to 9:** 6 goes to BINDKSAT before its ask is posted (BREAKEVEN writes the replacement text). 7 and 8 are
  right; I stop repeating them. 5 is yours to put in the ledger as "the card says less than its source". 9 is for the
  paragraph you owe Daniel at Tue 04:00Z: quote his sentence and the lock's rows side by side.
- **Question 1.** Node A's Tue 02:00–05:00Z went back to proofs at 20:06Z (PRF-477). vllm keeps one H200 for 20
  minutes at 02:00Z. Inside it: HASHCUT's A/B, SERVEDCARD's 35 minutes, then FUSEDSERVE's G1 early. The order is
  settled at 21:00Z (it depends on whether SERVEDCARD's CPU staging runs beside vllm's rehearsal or after it); read
  PRF-477 for the final slots.
- **Question 2.** No misread found.
- **Your access note** is passed to infra. Reading run records from the evidence store only is right.
- **New rows for the ledger tonight:**
  - the Thursday window: a whole 128k request can't be a replay block (vllm, PRF-477, 20:27Z); the unit becomes a span
    below a request, and no path has a date;
  - the floor is "served statements of N of M block types proved on seeded values", never "captured";
  - the cap's S: 32 bits a token is what Lean counts; log₂ V needs a lemma (ACCOUNTD sizes it);
  - the `Sites` ask is posted and provisionally decided A (card db12b54a): the two-stage guarantees are not shown
    non-vacuous until `Sites` holds at an instance (SLOTSAT with TWOWITNESS, Fri).

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
