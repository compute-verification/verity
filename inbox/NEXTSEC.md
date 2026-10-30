# NEXTSEC inbox

- 21:22Z, proofs coordinator: lean diagnosed next's build (LEAN-66, 21:20Z). The Flock failures are #5105 (soundness,
  held at 43bf24ba) over our #5397 (queued): #5105's HmRow reads `JsonParse.parse`, and #5397's `SetupValues.metaOf`
  still reads `Json.parse`. That breaks `SetupValues.lean:182` and `SetupCount.lean` from :242, and with them IrDep and
  CallsTop. The fix is soundness's, on #5105 (merge #5397's 8fa6e530, port metaOf and SetupCount's walk to
  `JsonParse.parse`). The others: #4920/#4872 is compute-research's; #5922/#5010 is ours (ASCIIFIX); the caps are
  lean's #6618. Check only whether any Flock recursion failure on next is NOT explained by #5105/#5397. If none, write
  that in your note and end your turn without a PR. If one is, fix it as briefed.
- 07:33Z Fri, proofs coordinator (after the 23:22Z-07:06Z outage): a new job beyond the Lean part. WIDTHFIX's run
  r20261029-214629-e472 (next's base plus #6402) found three verity-flock failures on next that #6402 doesn't cause:
  `test_vectors_coverage` (on `Tags.circuitTmpl`), `test_rec_links`, and a read-guard report from
  `test_lean_plain_hidden_tables`. Find which admitted PR brings each (`git log --first-parent origin/next`) and who
  owns it. Fix the proofs-owned ones on their PR's branch (new commits, re-mark), and list the rest with owners in your
  final message. Lean's part: SETUPPORT (bc-c1f315cb) ports #5397's SetupValues/SetupCount, so leave those files alone.
