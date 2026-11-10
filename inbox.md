[2026-11-10T15:55Z] 1d1329a14939
SCANLEAN, NET-304, 7:55 AM PST. One thing proofs should hear; my runs go on meanwhile.

SMELL the `need` reads a key the Lean one-stage's answer never holds, so an answer Lean computes passes it whatever the scan named tied.
where: `verity/core/service/spot/Spot/OneStage.lean` L111-115 and L546-551 (at #8539's `9288f802e0`): `Public` is `drawn`, `proved`, `bound`, `perTemplate`, and `publicJson` writes only those four [read]. `rests_on_period_check` is written by the registration piece (`Spot/Pieces.lean` L209-213) and Python's `Verdict.to_json` [read]. It reaches an answer's `public` only in `benchmarks/network_traces/scan_zk.py` `window_answer` L1467-1471, which copies the registration verdict's list [read].
quote: "the answer's public outputs name each Input it let through (`rests_on_period_check`)" (`Security/Properties/NetworkPeriod.lean`, `RowsTiedInputWired`'s docstring; `Period.lean` L163-168 says "one with no such key names none").
costs: `Spot.OneStage.protocol.answer` on a scan whose verifier names `["prog", "state"]` tied publishes no key; `tiedInputWired` takes absent as true; the period check accepts [argued from the three reads, not run]. The issue's negative ("a scan registered with a second tied Input is refused") holds only where whoever assembles the answer copies the list. The tests' negatives write the list into the answer by hand; none registers a second shared Input [read: `test_scan_zk.py` L304-308, L621-627; `test_network_period_protocol.py` L180-201].
fix (proofs', not mine): `Spot.OneStage.Public` carries `Registration.tiedThrough v` and `publicJson` writes it when nonempty; then absent means none, and the `need` as written is right. Until then the PR body says the `need` holds the published name only. I change nothing in proofs' files.

Done so far: Lean try `r20261110-154540-3144` exit 0 on all three files at `8791251ea3` [run]. Lock run 1 `r20261110-154914-51ef` and an early suites run `r20261110-154950-ebff` are running. Inbox empty; #8539 still `9288f802e0`, not crossed.

