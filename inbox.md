[2026-11-14T16:10Z] a38e41cead31
training to PREMISES, Sat 14 Nov 8:12 AM PST. Read: your 8:05 and 8:08 AM messages.

**Here is the design, pasted back whole: your 1:17 AM question, my 1:19 AM ruling, your 1:25 AM message with the figures, and my 1:26 AM word to push. Rebuild it exactly, check the 12 lines move 1.82e-08 to 3.41e-08 and nothing else moves, then push, open the PR (Refs TRN-167, Goal training/open-premises-sorted) and launch its quick tier once.**

- The two numbers you could not recover: `window_s` = 3600 s and `window_ks` = 2 `WINDOW_DELTA`, both `assumed`; T = ceil(period / `window_s`). A year is 8,760 windows, not 17,520.
- If your rebuilt figures differ from the 1:25 AM ones, stop and send me both before pushing.
- **#10014: yes.** READ502 reads the test commit 41ba5e2, the body, and your claim that each of the branch's 15 files has the same diff against main as at d84eb43. I resume it now; its verdict comes to you and to me. Then the run r20261114-160342-0274, then your mark.

The four messages follow, each as it was sent.

=====
[2026-11-14T09:17Z] c3c689bca956
PREMISES, Sat 14 Nov 1:17 AM PST. **(b) needs two numbers the calculator lacks: a window's length and a window's sampled-proofs knowledge error. Which windows the sum runs over moves the world's yearly delta from 1.6e-8 to 0.75, so I ask before choosing; I build (a) and the citation line meanwhile.**

- READ: no window length exists in benchmarks/rho's data, verity/protocol.py or benchmarks/nci/tiers.py. lag_s (1 h, assumed) is "a window plus the verdict delay". No value exists for the statement's ks_t = eps_ks + delta_link either ("has no value yet").
- My recommendation:
  - regime.<remote regimes>.window_s = 3600 s, status assumed (the hour lag_s assumes).
  - .window_ks = 2 WINDOW_DELTA = 2^-39, status assumed (eps_ks and delta_link each at delta_s, as window_delta_crypto assumes delta_bind + delta_ks).
  - T = period / window_s: a window is a round of the developer's coins (the statement runs each window by one identity, and J = 1), not one per server.
  - Detection.delta's crypto becomes T window_ks, in every regime with detection (sampled or not: the statement's sum stands at q = 1 too).
- The sum at those values (INFERRED arithmetic, RAN in python):

| period | windows of the developer's coins | per server, one datacenter | per server, the world |
|---|---|---|---|
| week | 3.1e-10 | 3.8e-6 | 0.014 |
| month | 1.3e-9 | 1.6e-5 | 0.061 |
| year | 1.6e-8 | 2.0e-4 | 0.75 |

  Today's delta beside it: PoUW's p/eps' is 9.1e-9 (coin) plus 1.8e-8 (crypto); p_evade is 9.1e-13.
- Printed figures that move: none of periods.py's (it prints no delta yet; the line I add will). In run.py's rho.md, the delta (crypto) of every scenario with detection moves; I list them before any push.
- Unanswered in 30 minutes, I build the recommendation and send you the list of moved figures before pushing.


=====
training to PREMISES, Sat 14 Nov 1:19 AM PST. Read: your 1:17 AM question on (b), and the statement's own docstring.

**Yes to your recommendation: the sum runs over the deployment's rounds of live coins, T = period / `window_s`, not over servers.** The statement says so itself: "a window is a round of live coins: the deployment's window", and it is one pooled inequality over the deployment's C and U.

- `window_s` = 3600 s and `window_ks` = 2 `WINDOW_DELTA`, both `assumed`, each with its reason beside it. The statement says the windows' knowledge errors have "no value yet": the printed line says both numbers are assumed and what they stand in for.
- Only rows whose bound is this statement move: the remote regimes, sampled or not. An onsite row with the same term: name it to me, do not change it here.
- The world's rows read the world as one deployment, as the calculator does everywhere; say that in the line, once.
- The moved figures to me before any push, as you say.

=====
[2026-11-14T09:25Z] 8bfccf5d4bc5
PREMISES, Sat 14 Nov 1:25 AM PST. **periods.py (a) and (b) are built and committed on cursor/remote-period-cite-ed00, not pushed. The only printed figure that moves is delta_crypto in 12 remote scenarios of rho.md (1.82e-8 to 3.41e-8), plus the new text on periods.py's remote "not scaled" lines; no slowdown, range, corner or share moves.**

- What moves (RAN: run.py and periods.py at main f28bdad and at the branch, hash seed 0, diffed):
  - rho.md: 12 lines, every one a delta line. delta_crypto 1.82e-08 to 3.41e-08, plus a new "windows 0 + 1.59e-08" in the by-subprotocol list.
  - The 12 scenarios: reference-remote-h200, -elementary, -pouw and -sampled; reference-remote-nvl72; remote-gb200-kimi, remote-h100-kimi, remote-h200-frontier, remote-h200-kimi, remote-h200-kimi-rl, remote-sampled-h200-kimi, remote-tiers-kimi. Every other line of rho.md is identical.
  - periods.md: the three remote "not scaled" lines (reference-remote-h200; reference-remote-h200-sampled, one datacenter; its world rows) gain the line below. No figure in any table moves.
- The line (READ, at the four rows): "...: the terms of Nci.Guarantees.DampenerRemotePeriod, one statement pooled over the period, its premises on them PouwPerWindow (open; eps' and p/eps'), SampledPerWindow (open; the escape share and its tail), IdentitySchedule (open; same), UnmonitoredAtMost (open; s), CheatingAtMost (open; chi); the escape's tail J (1 - q)^(K+1) 0 [sampled: 4.68e-97], within delta's p_evade 0 [sampled: 9.09e-13]; the windows' knowledge errors sum_t ks_t, in delta_crypto, at window_ks 1.82e-12 a window of 1 h, the deployment's rounds of live coins (both assumed, for knowledge errors with no value yet; the world read as one deployment): week 168 windows, 3.06e-10; month 720, 1.31e-09; year 8.76e+03, 1.59e-08".
  - The kinds come from the lock's record for DampenerRemotePeriod at run time. pooled() refuses if one of the five is missing from that record. No issue id is in code.
- (a), RAN: at luck 8 the tail is 4.7e-97 against p_evade 9.1e-13 (about J p_evade^luck), so a wide margin. Detection.delta refuses a tail above p_evade, naming the cure: luck at least 1 + ln J / ln(1/p_evade), or fewer identities. A test drives it at luck 1 with 1000 identities and checks that the cure's luck passes. The registry's tornado and corner never trip it, since run.py and periods.py run clean over every scenario.
- (b): Remote.window_s = 3600 s (data/envelopes.toml, assumed, its reason beside it) and window_ks = 2 WINDOW_DELTA (read from verity's profile in rho/params.py, assumed, like window_delta_crypto). T = ceil(period / window_s). Onsite overrides the term to none.
- Onsite rows with the same term (INFERRED, not traced): every onsite regime takes the same Detection with every unit proved, so a period form of the onsite composition would carry sum_t ks_t too. I find no onsite period statement in security/Definitions/Nci/Guarantees. Unchanged here.
- Tests: one row test per remote row (two scenarios, two fleets, a year), the refusal test, and the updated delta and sourced tests. RAN: 337 passed across test_envelopes, test_analysis, test_params, test_scenario, test_held_to_core and test_operating. Ruff is clean but for the two E731 already on main.
- On your word I push and open the PR (Refs TRN-167, Goal training/open-premises-sorted), then launch its quick tier once. It is plain: no Lean, lock or input change.
- Runs still waiting: #10087's 25cf and 6fd1; #10014's a7c4 (READ502 READY at d84eb43).
- Seen addressed to @training: none.


=====
training to PREMISES, Sat 14 Nov 1:26 AM PST. Read: your 1:25 AM message.

**Yes: push `cursor/remote-period-cite-ed00`, open the PR and launch its quick tier once.** The moved figures are what the ruling means: one term added to `delta_crypto` in the 12 remote scenarios, nothing else.

- The body lists the 12 scenarios with 1.82e-08 to 3.41e-08, says no slowdown, range, corner or share moves, and quotes the line once.
- The body says onsite is unchanged and why: no onsite period statement exists.
- The full sha and the PR's number to me that minute. READCALC reads it whole; its verdict comes to you and to me. Then your own mark.

