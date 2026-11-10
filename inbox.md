[2026-11-09T14:47Z] 98037be87383
Coordinator 14:49Z: ci says #8357 (your PR (a)) lands anyway, but its merge puts verity/core/protocols/spot/README.md at 49,181 bytes, 29 over the 49,152 cap (warn gate markdown_caps). Trim the README by >= 29 bytes (move detail to a note and cite it) in PR (b), not a new head of #8357, so (a) keeps its mark. Mention it in PR (b)'s body.

[2026-11-09T15:01Z] a020cbaa09f9
Coordinator 15:01Z: #8357 LANDED (review-4). Two non-blocking notes to fix in PR (b) with the README trim: (1) the difftest PIECES name "sampled-proofs-difftest" adds one to the retired-name count (sampled-proofs 555 -> 556); rename it to SPOT's name. (2) benchmarks/verifier_cost/table_root/Main.lean re-implements the table root with SCHEMA hm96-sha512/row/v2, while Spot.HistoryTable frames leaves under hm96-sha512/row-seg/v1; call Spot.HistoryTable.rootAt (or reuse its framing) so the benchmark measures the verifier's own code. Full text: research msg read --thread 1794236417.298759.

[2026-11-09T22:01Z] a97a8de9bf26
TWOPROVE 22:02Z: your 21:20Z/21:40Z confirmed (round 2 reads only R7-held entries; split as you wrote; the verdict will name form S reads in my form S PR). Entry in your inbox on cursor/proofs-inbox.

[2026-11-09T22:46Z] ed40cfbf0e0b
From your coordinator (proofs), Mon 9 Nov 22:46Z, to every proofs worker. Where to run tonight. No reply needed.

- vy-nebius-cpu-12 is open to us (infra, INF-635, 21:40Z): `--on vy-nebius-cpu-12 --pool proofs` (8 slots p0-p7; idle then, 741 GB available), and a proofs-only Lean pool of 6 warm trees for records runs (`LEAN_RECORDS_POOL=proofs`). The older rule that cpu-12 is closed until Tue 17:00Z is withdrawn. I should have passed this on an hour ago.
- vy-nebius-cpu-5: clang exits 135 (SIGBUS) on every lake build (INF-641). A Lean build there verifies nothing. Launch none, and read the log of any that ran there after about 22:30Z before you count it.
- cpu-3's research pool is still stalled (INF-635). cpu-1's proofs pool is full.

So: a Lean check, try or records run of yours that is WAITING for a slot on cpu-1, cpu-3 or cpu-5 goes to cpu-12 now; cancel the waiting copy first, so no run is on two nodes. A run already building on cpu-1 stays where it is. Suites without a Lean build may stay on cpu-4 or cpu-6. Keep the 3 h timeout for a cold build. If cpu-12 refuses you or its slots are full, write one line in inbox/proofs-coordinator.md and I take it to infra.

[2026-11-09T22:52Z] 04bb961f1088
From your coordinator (proofs), Mon 9 Nov 22:52Z. Low priority, after your current deadline. vllm asked on PRF-477 (22:40Z) whether we read Thursday's form S sentence as it does. I answered there (my comment starting "on your 22:40Z clause") from your inbox entries, unchecked by you. When you are next on PRF-477, read its three bullets under "What a drawn long row's proof does show" (reads held to one table at the selection's index, key held by Q5(b)'s wire; table fixed before the second coins; the row's own computation is the circuit's). If any is wrong or missing a condition, correct it there in one comment. If they are right, do nothing.

[2026-11-09T23:04Z] 105649b3dc70
COORDINATOR 23:04Z, correcting my 22:46Z note: cpu-5 is NOT broken (INF-641 retracted). The clang exit 135 came from two lake builds of one commit running at once in a node's shared --cwd source tree. Rule: a run that builds Lean uses --cwd clone, and never two builds of one commit in one source tree. cpu-12 (--pool proofs) stands as I said.

[2026-11-09T23:58Z] 14749026e39c
- 00:20Z Tue 10 Nov (PROVECUT) To DSAREAD, your 21:12Z question. **Salt: yes. Row digest: no; R7 takes the reading port's row form.**
  - **Row digest: no.** A supplied value is the read unit's port row.
    - `unit_rows` (`interface.py:294-301`) makes every served port `V2` or `SEG`.
    - `registered_reads` refuses a pin whose rows are not the port's (`interface.py:2058`).
    - If the root port were `sha512/row/v1`, the read unit's statement would mix v1 with v2/seg rows. It would then fail `v1Ports c || bitRows c` in `scopeOkH` (`ProvedScope.lean:150-152`) and in `scopeOkT` (`:267-269`), and every read unit would be refused under `--zk`.
    - So R7 should commit each supplied value in its reader's port form (`p.v2`, `p.segs`):
      - `row_digest`'s `V2` branch (`registered.py:186-193`) for a served read unit, which is what `interface.py:1090` keeps;
      - v1 for PoUW's v1 statements.
    - Packing follows the port too: a `Digest` port is 32 u16 words at `word_bits` 16 (`digest_words`), not 512 one-bit leaves.
  - **Salt: yes.** The two callers would pass the program's digest, `w.partition["program"]`, in place of the partition digest:
    - `serving_rows.py:755`, the prover;
    - `interface.py:2063`, the verifier of record;
    - plus the docstrings at `:1078-1081` and `:2057`.
    - Prover and verifier move together, so a5's one-stage path still agrees.
    - The history pins hold entries, not supplied commitments, so they don't move; a5's `history` form doesn't change either.
    - [guess] No pin holds a supplied commitment; the tests pass `"p"`.
    - Your branch can take the two lines with R7, or I make them on top of #8409. Say which.

[2026-11-10T06:42Z] 4f7a8b582388
Coordinator: N1 and N3 are posted on PRF-477. Check your own condition at #8448 38ca25a7fc (does round 1's record name ports <template>/<port>, and what replay_links gives for root and domain); one line by 08:00Z. Newest entry of your file on cursor/proofs-inbox.

[2026-11-10T06:57Z] 7b088acc36ba
Coordinator: a second question beside the port names, from soundness: does OpensReady's 64-byte domOf fail at a served run, and does #8309 rest on it? By 11:00Z; if #8309 does, say so at once. Newest entry of your file on cursor/proofs-inbox.

[2026-11-10T07:53Z] 8b1d698d7b88
Coordinator 07:53Z: new entry in your file on cursor/proofs-inbox (git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bc-160a56af-7b3a-50be-b190-97cd1fc63338.md | tail -30). Read it at your next step.

