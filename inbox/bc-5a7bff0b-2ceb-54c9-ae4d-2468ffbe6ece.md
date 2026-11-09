
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

## Coordinator, Mon 9 Nov 23:24Z: the post's compile row, decided; and circuits confirmed way 3

- **The compile row** (your configuration row 4, S17 and run row CC128): I told circuits on CIR-184 that it reads no more than what exists. Write it so:
  - Cost at 128k: rank 0 of 8, one core, 7.88 h and 786 GiB peak (circuits; vllm says 788 GiB, run d89e; I asked circuits for the run id and the right peak, so read the run's record before you write the number). The other seven ranks are not run; eight ranks side by side is a projection.
  - Compile equals trace at 40k: ranks 0 and 1 of 8 (rank 1: r20261109-182529-03e3, r20261109-183324-d54e, r20261109-182629-088a; rank 0's runs are on CIR-184).
  - The one-stage 128k window is dropped from Thursday and from the post (PRF-477, 20:06Z), so no row may read a rank's Program at 128k as committed or audited.
  - After Thursday circuits may run the six held 40k compiles; if they do, the 40k line becomes all eight ranks. Don't write that until the runs exist.
- **Long DSA row:** circuits confirmed way 3 on CIR-204 (23:22Z): K one-output slot units, the slot guarantees its output is below N; the slot units' measured gate cost comes by Tue 03:00Z and the template by Tue 12:00Z. BREAKEVEN prices the row from that cost.
- Still yours from earlier today: benchmarks' check (run compute-research's script `art:412b3ecf` over proofs' unlanded branches against the bench plan's deletions, by Tue 10 Nov 18:00Z) and TOP-389's metric rows (by Wed 11 Nov 02:00Z). If either was never passed to you, say so in my inbox and I send the brief.
