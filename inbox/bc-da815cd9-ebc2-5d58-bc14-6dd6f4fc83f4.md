## 01:50Z from soundness: more Lean capacity (infra 00:38Z); prefer cpu-3

- cpu-3 has 4 warm build trees from ~00:45Z and is the quietest node: **`verity lean try FILE --on vy-nebius-cpu-3 --as soundness`**
  is the shortest path now.
- cpu-1's research pool went from 13 to 21 slots, so the default (no `--on`) drains faster too.
- cpu-5 is usable again (85% inodes, 4 warm trees).
If a try of yours has waited over 20 min in a line, cancel it and relaunch on cpu-3. Never run two copies of one try.

## 02:35Z from soundness: rulings on your two points (top 01:03Z, applying standing rulings)

1. **Coin root: the device withholds it.** The challenger releases its hm96 coin-commitment root only **after** the prover's
   commitment is in. No hiding premise goes into soundness (standing ruling: no assumed hash or signature advantages). You own
   sampled proofs' challenger side (ONE-4): change the release order in the same PR (#4888 or a follow-up), and check nothing
   in sampled proofs reads the early root. memory-accounting checks the PoUS memory challenger and the remote bandwidth audit's
   window coins.
2. **LMS: two break terms, tied to the run.** The event is "a SHA-512 collision among the run's hash inputs ∨ a verifying LMS
   signature on a message missing from the device's log", and the bound is `≤ δ + f(Pr[B exhibits a collision]) (+ LMS forgery)`.
   `ecr/sha-512` and `euf-cma/lms-sha512-n64` are already in `verity.core.claims`, applied outside the theorem to the fixed
   finders and the forgery finder. Keep the forgery break as an event of the device's log, not as an advantage bound.

## soundness 01:20Z: coin-root release order (Daniel's ruling, 6:15 PM PDT)
Daniel approved: the challenger releases its hm96 coin-commitment root only after the prover's commitment is in, and soundness takes no hiding premise on the root. You own this change for sampled proofs' challenger: put it in #4888 (device withholds the root until the prover's commitment; drop any hiding premise), with _honest/_nontrivial. Date: #4888 marked by Tue 12:00Z. Say on PRF/ONE-4 if that date slips. PoUS and remote bandwidth are memory-accounting's (Wed 10:00Z).

## soundness 03:05Z: two things for #4888
1. **The Run path's `coinsReason` must require withheld too**, in #4888 itself: the rule is that every path of the verifier of record refuses a form the proof doesn't cover, so one reader that still takes an unwithheld open is a hole in protocol soundness. Add it with a difftest case, same as `drawCoins`.
2. **Landing order:** #4888 must not land before INF-280 (the auditor gate), or every gate audit is refused. I asked infra for INF-280 marked by Tue 10:00Z with an end-to-end gate test against your head. Put "lands after INF-280" in #4888's body; still mark it by Tue 12:00Z. If INF-280 is late, add a `hold` reason rather than unmarking.
Also: don't cancel a run that already holds a check slot (briefs/common.md); your try on n1 is in the try lane, leave it there.

## soundness 08:36Z: tie F3's `key θ` to the key the compiled components verify under
ONETHM made `Vocab.forgery θ o m sig := LmsVerifies (key θ) m sig ∧ ¬ o.Signed m` (F3 @ 83255f8c57), with `key` and `sign` new `Vocab` fields. Nothing yet ties `key θ` to the LMS key the compiled components (your `drawCoins`/`coinsReason`, and the new per-round check) actually verify under. Define that decoding with your round-coin work: `key θ` := the registered device key in θ's params as the compiled check reads it, and state the lemma that the compiled check's LMS verify is `LmsVerifies (key θ)`. Post it on PRF-200/PRF-193.

## soundness 09:19Z: cpu-4 is out of inodes/space
Runs there die at clone or with 'no space left on device'. Move any queued cpu-4 run that holds no check slot (check the cancel output) to cpu-3 (`--pool research`) or cpu-1; new runs don't go to cpu-4 until infra says it's clear (briefs/common.md).

## soundness 09:45Z: two of your runs are on cpu-4 (out of inodes/space)
r20261027-062057-357b (Lean build) and r20261027-062557-2288 (Proofs lemma try) are queued on cpu-4, where runs die at clone or with 'no space left on device'. If they hold no check slot (check the cancel output), cancel and resubmit on cpu-3 (`--pool research`) or cpu-1. Also: hold #5089 on INF-291 as well as INF-280 (honest v2 serving is refused until the live coin server takes coins from the window's withheld session); I've asked infra for INF-291's date.

## soundness 10:15Z: no lane markers in anything you land (cleanup window; top lands a test that refuses them)
In docstrings, comments and Markdown of your PR: no Linear ids (PRF-, LEAN-, NET-, MEM-, CMP-, INF-, TOP-, ASK- …), no `note:` or `art:` ids, no PR numbers ("#4613", "PR #…"), no Notion/Slack links, no "Daniel"/handles/agent ids/dates. Keep the reason, drop the id, date and name. Run before marking:
`rg -n -e '\b(TOP|CMP|LEAN|INF|TRN|PRF|NET|MEM|VLLM|LINTOOLS|ASK)-[0-9]+\b|note:[0-9]{8}T|\bart:[0-9a-f]{6}|notion\.(so|com)|slack\.com|\(#[0-9]{3,5}\)|PR #[0-9]+|\bDaniel\b|@(top|ci|infra|proofs|lean|comms)\b' <your changed files>`
Fold the fix into your next commit (no extra records run just for this unless the file is in a lock's record).

## 20:40Z
Run r20261027-182146-a4ec, the full check at #5330's 3f043281ab, failed in lean-suites. Every other step passed.
- verity/ml/flock: about 14 tests in test_lean_witness.py fail: test_f1_the_session_is_accepted_as_the_verifiers_statement[L0..L3, alg/p0, alg/p1, cap, cap/L1..L3], test_f1_verify_refuses_the_session_only_for_its_undrawn_inner_session[cap, L3] and test_the_recorded_level_was_proved_before_the_os_masks. It looks like your new v2 refusal turns away F1's recursive sessions. Check that first and fix it.
- benchmarks/network_traces: test_scan_zk.py's rows_forms case fails. Fix it if #5330 caused it.
- integrations/vllm: test_attention_boundaries[fa3_check_inf] and test_lint::test_query_package_exists fail. These are probably red on main too; confirm, and leave them if so.
Rerun only those suites, at `--cwd clone`, on one node. Mark #5330 once they're green apart from failures main shares, by Wed 03:00Z.
