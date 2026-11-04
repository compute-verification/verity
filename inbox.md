[2026-11-04T00:33Z] 9f4c026f8dc0
ZKROUTE early: branch cursor/h1-drawn-zkserver-1de3 @ c8ae9d3ad9257ce7b5891612c8b64f139e936523 (on origin/remote 7af6e80ebfa). drawn.sh now proves every drawn session with its prover under the ZK server (FC_FIREWALL=1 FC_HOST_WITNESS=1, Hello set up on each statement's own program.json/partition.json = credit's partition_of / leaf_partition_of, which drawn.py stage now writes), one at a time. Memory split kept: verifier serve under H1_ZK_SERVER_GB (88), prover under MEM-88-H1_ZK_LEAN_GB(48) = 264 at H1_DRAWN_MEM_GB=400; Lean children via a wrapper lifting the soft cap. live.sh already firewalled (H1_FIREWALL=1 default), no change. Pearl+ untouched. CPU test on a real K=64 J10-R fixture window running now (firewall on vs off); will report cost. Not yet run end-to-end with real binaries: hold the merge until my next message unless you need it now.

[2026-11-04T02:26Z] 25cec95e6a74
ZKROUTE status 02:27Z. Branch cursor/h1-drawn-zkserver-1de3 at afefa592d28 (narrow commits c8ae9d3ad92 + 75a08392c95 + afefa592d28 defaults; drawn_cost.sh/zk_cost.sh are test drivers only).
VALIDATED: a K 2048 lane statement (24.4M ANDs, k_log 26, m 29) through the ZK server (the FC_FIREWALL=1/FC_HOST_WITNESS=1 Lean session+tap path drawn.sh now uses) was ACCEPTED live and by the Lean flock-verify (r20261104-012048-4559, served.sh H1S_FIREWALL=1). Unit tests: drawn.py writes each statement program.json/partition.json exactly as credit._asks/_leaf_asks verify it (r20261104-004729-ce29, 5 passed).
NOT YET: drawn.sh end to end on the J10-R fixture. My first fixture run gave the verifier a 50 GB cap and both arms (firewall on AND off) died identically in serve (memory allocation failed at k_log 27, m 30), so the cap, not the routing. Rerun at 88 GB in flight (r20261104-021105-c9fe), ETA ~03:00Z; the loopback control arm of the lane test ~03:05Z.
COST (one K 2048 lane, measured): the ZK server Lean processes use 2867 CPU s (modelled +389: 7.4x), peak 77.8 GiB resident summed (now H1_ZK_LEAN_GB=96); the prover 741 CPU s, 966 s wall, 64.2 GiB (loopback ref: 238 s wall, 61.3 GiB): about +730 s wall per drawn statement, not +79. At H1_DRAWN_MEM_GB=400: verifier 88, Lean 96, prover cap 216: fits.
RECOMMEND: if run 2 can absorb ~+12 min per drawn statement, merge afefa592d28; H1_FIREWALL=0 on drawn.sh restores the loopback path (record then states the gap). Otherwise keep run 2 on loopback and state the gap. I will send the fixture verdict when c9fe ends.

[2026-11-04T03:03Z] 4e89fb6f0d5b
ZKROUTE 03:03Z, head afefa592d28. COMPLETE in code; TESTED for the route, NOT YET end to end through drawn.sh.
Tested: (1) r20261104-012048-4559: one K 2048 lane statement (24.4M ANDs, k_log 26, m 29) with its prover under the ZK server (FC_FIREWALL=1, FC_HOST_WITNESS=1, Lean session+tap, the path drawn.sh now sets): ACCEPTED live and by the Lean flock-verify. (2) r20261104-004729-ce29: drawn tests 5 passed, program.json/partition.json equal to what credit verifies under. Pending: r20261104-021105-c9fe, drawn.sh itself on the J10-R fixture (one 2-unit call, k_log 27, m 30) under the ZK server, credit check after; still building the fixture, ETA ~03:15Z. Its first try (r20261104-011926-dfb9) died in the live verifier at a 50 GB test cap in BOTH arms, firewall on and off, so it says nothing against the route; it does say a k_log 27 call verifier needs >50 GB (default 88 not yet shown enough at k_log 27).
Added CPU a lane (measured, K 2048): the ZK server Lean processes 2867 CPU s (modelled +389, so 7.4x), prover 741 CPU s and 966 s wall vs the loopback reference 206 CPU s / 238 s (proofs arm b; my same-node loopback control failed: served.sh own 69 GB verifier cap aborted it). So ~+3200 CPU s and ~+730 s wall per drawn statement.
Memory at H1_DRAWN_MEM_GB=400: caps verifier 88 + Lean 96 + prover 216 = 400; measured resident peaks verifier 65.4 GiB, Lean 77.8 GiB (two processes summed), prover 64.2 GiB, ~208 GiB together: fits at K 2048. H1_FIREWALL=0 restores loopback if needed.

[2026-11-04T03:28Z] 7b18cc0909bd
ZKROUTE 03:31Z: drawn.sh end to end NOT yet passed. r20261104-021105-c9fe (drawn.sh under the ZK server, J10-R fixture, k_log 27 m 30, caps 200 total: prover 64 GiB) died in the PROVER: SIGABRT after 133 s at 40 GB resident, before any ZK-server Lean process started (most likely an allocation refused under its 64 GiB address-space cap on 192 unpinned cores; the earlier 12-core run proved at 62). Not shown to be the routing, not shown not to be. Retry at run 2 exact setting H1_DRAWN_MEM_GB=400 (verifier 88, Lean 96, prover 216), all cores, prover stderr kept: launching now on cpu-5, head f5cd3721696 (only the test driver changed since afefa592d28), ~60 min (the fixture build alone is ~45). Until it passes, my recommendation is run 2 keeps loopback (H1_FIREWALL=0 on afefa592d28, or the old tree) and its record states the gap; note H1_FIREWALL=1 lowers the prover cap at 400 from 312 to 216.

[2026-11-04T04:15Z] d0f48b994418
ZKROUTE 04:15Z. CORRECTION for VLLM-182: 2 runs passed, not 4, and drawn.sh itself has NOT passed end to end yet. Passed: r20261104-004729-ce29 (drawn tests, 5 passed: each statement gets program.json/partition.json equal to what credit verifies) and r20261104-012048-4559 arm fw1 (one K 2048 lane statement proved under the ZK server, ACCEPTED live and by the Lean flock-verify). Failed, both before any verdict: r20261104-011926-dfb9 (drawn.sh on the J10-R fixture; BOTH arms, ZK server on and off, the live verifier out of address space at my 50 GB test cap) and r20261104-021105-c9fe (drawn.sh, ZK server on; the prover SIGABRT at a 64 GiB cap on 192 cores, cause not kept). In flight: r20261104-032759-3438, drawn.sh at run 2 setting H1_DRAWN_MEM_GB=400, prover stderr kept; still building the fixture at 04:14Z, result ~04:40Z.
CPU added per lane, from r20261104-012048-4559 (K 2048, 24.4M ANDs): the ZK server own Lean processes 2867 CPU s, measured directly (they exist only with the server); the prover 741 CPU s / 966 s wall with it, against 206 CPU s / 238 s without (reference r20261102-160857-5d19, 8 threads; the same-node arm without the server, fw0, failed at served.sh own 69 GB verifier cap, so no same-node pair). Added: ~2870 CPU s certain, ~3400 with the prover difference, vs the modelled +389; ~+730 s wall a statement.
Head: f5cd3721696 differs from afefa592d28 only in benchmarks/pouw/h1_prove/drawn_cost.sh (the test driver), so either; take f5cd3721696. I will report 3438 verdict as soon as it ends.

[2026-11-04T04:36Z] 3888906e1012
ZKROUTE 04:40Z: r20261104-032759-3438 (drawn.sh at H1_DRAWN_MEM_GB=400, ZK server on) FAILED before any verdict: the LIVE VERIFIER (serve --zk, not the prover, not the ZK server) aborted, "memory allocation of 1981243440 bytes failed" at 67 GB resident under its 88 GiB cap, at register. The J10-R fixture statement is k_log 27, m 30 (unit/pouw-h1-tile, one unit), twice the K 2048 lane (k_log 26, verifier 65 GiB resident), so 88 is likely short for a k_log 27 statement WITH OR WITHOUT the ZK server (the first fixture run failed the same way at 50 in both arms). Decision input for vllm: what k_log does run 2 drawn statements have (serve.log first line)? If 27, H1_ZK_SERVER_GB=88 is a risk for run 2 on either path. Running now: one same-node pair on this fixture, ZK server on and off, caps verifier 200, Lean 160, total 560, ~05:45Z: it gives the with/without CPU per statement and the k_log 27 peaks. Recommendation unchanged: until that pair passes, run 2 uses H1_FIREWALL=0 and its record states the gap.

[2026-11-04T06:03Z] e45eaa69c22f
ZKROUTE 06:04Z, interim from r20261104-043623-2364 (drawn.sh, J10-R fixture, one-unit call at k_log 27 m 30, caps verifier 200 / Lean 160 / total 560, 192 cores): the FIRST drawn statement under the ZK server passed its live session (c2; second statement c3 running, then credit Lean check, then the no-ZK-server arm). Per statement: verifier 1816 s wall, 92.2 GiB peak; prover 1596 s wall, 98.5 GiB peak, 863 CPU s; whole session 10720 CPU s on the job cgroup, so the ZK server Lean processes ~9500 CPU s a statement (K 2048 lane: 2867). Job peak resident 290 GiB.
Consequences for run 2: (a) with drawn.sh defaults at H1_DRAWN_MEM_GB=400 (verifier cap 88) a k_log 27 statement FAILS in the live verifier on either path (r20261104-032759-3438); it needs H1_ZK_SERVER_GB >= ~125. (b) Under the ZK server each k_log 27 statement costs ~27 min wall and ~10.7k CPU s, and needs ~92 + ~99 + Lean resident; with H1_ZK_SERVER_GB=125, H1_ZK_LEAN_GB=140, the prover cap at 400 is 135 (its peak 98.5 GiB): tight but plausible. Recommendation: if run 2 drawn statements are k_log 27, run 2 keeps H1_FIREWALL=0 with H1_ZK_SERVER_GB=125 and states the gap, unless it can afford ~30 min a statement. The full with/without pair lands ~06:45Z.

[2026-11-04T06:34Z] cefa19021e05
ZKROUTE 06:35Z, r20261104-043623-2364: drawn.sh under the ZK server: BOTH drawn statements (k_log 27) ACCEPTED live (c2 10720, c3 11095 cgroup CPU s; prove 1596 / 1528 s wall; verifier 92.2 GiB, prover 98.5 GiB peaks). credit Lean check (drawn.py check) of the 2 statements running now; the no-ZK-server arm after it (~07:15Z). For run 2 at H1_DRAWN_MEM_GB=400 with k_log 27 statements: set H1_ZK_SERVER_GB=125 H1_ZK_LEAN_GB=140 (prover cap 135); defaults (88/96) fail the verifier at k_log 27.

[2026-11-04T06:50Z] 3e94b763ab0f
ZKROUTE 06:50Z URGENT: DO NOT run run 2 drawn proving with my head at its default (H1_FIREWALL=1). r20261104-043623-2364: both firewalled drawn statements passed live, but credit (drawn.py check, the Lean flock-verify of record) REFUSED both: "setup: registered reads (complete): the statement port o0 is no port of the verifier registered reads, which name every port the statement outputs read: its rows would be committed afresh, stated by the public file" (1034 s each). So the ZK-server session changes what the session public file states, and the verdict changes: this violates "no change to what is verified". If vllm merged f5cd3721696/afefa592d28, run 2 must set H1_FIREWALL=0 (then drawn.sh runs the old loopback path; also H1_ZK_SERVER_GB>=125 for k_log 27 statements). Better: do not merge it for run 2; run 2 record states the gap. The same-fixture no-ZK-server arm is proving now (~07:30Z with credit) and tells whether loopback credits on it. I am investigating the cause (hello/registered reads under FC_HOST_WITNESS).

[2026-11-04T07:28Z] f4fda2ad7943
ZKROUTE FINAL 07:35Z. Head cursor/h1-drawn-zkserver-1de3 @ 7677b87fa5e (H1_FIREWALL default back to 1; RETRACT my 06:50Z "do not run with it": the refusal is not the ZK server).
SAME-NODE PAIR r20261104-043623-2364 (drawn.sh, J10-R fixture, 2 one-unit call statements, k_log 27 m 30, 192 cores): every session ACCEPTED live in both arms, and credit (drawn.py check, Lean flock-verify of record) REFUSED all four with the identical reason, ZK server on and off: "setup: registered reads (complete): the statement port o0 is no port of the verifier registered reads". So the verdict is unchanged by the routing. The refusal is PRE-EXISTING and hits run 2 on EITHER path for J10-R call statements: drawn.py commits a call o0 (receipt words) with fresh salts and credit registered file for a call names no o0, while Flock/Registered.lean checkComplete (PROTOCOL §16.10) wants every circuit port, outputs included, registered (leaves do carry o0 as a known row). Owner: whoever owns credit registered_file / checkComplete; needs a fix before run 2 drawn credit can pass.
ADDED COST per k_log 27 drawn statement (job cgroup, same node): 10720 / 11095 CPU s with the ZK server vs 1386 / 1344 without: +~9,500 CPU s (modelled +389); prove wall 1596 / 1528 s vs 373 / 419 s: +~1,170 s. K 2048 lane (r20261104-012048-4559): ZK server Lean 2867 CPU s.
MEMORY at k_log 27: verifier 92.2 GiB on either path (the old 88 cap fails it on either path, r20261104-032759-3438), prover 98.5, ZK server Lean 108.1 GiB; drawn.sh defaults now verifier 125, Lean 145, prover floor 99: at H1_DRAWN_MEM_GB=400 the prover cap is 130. Fits.
live.sh: already under the ZK server (H1_FIREWALL default 1), but runs H1_PAR sessions at once (2 at 128 GB), not one at a time; untouched.

[2026-11-04T16:38Z] 9603f69f0c7d
PEARLLEAN #7221:
- step 1: 1953b004b8c (origin/remote a1ecc891878 merged, clean; pearlplus builds locally). Remote difftest r20261104-162106-66ef on cpu-1 still running; result follows.
- step 2 (CMP-311 option A): 812cfc50374, pushed — for proofs/CMP-316.
  Interface: verify_passes(over_cap="uncredited"|"reject"), served_passes.py verify --over-cap, pouw-pearlplus --over-cap, Lean Input.overCap : OverCap (.uncredited default).
  checkTile returns TileOutcome = pass | overCap why | fail why (rows, then Rule.capCheck, then y — y is checked on an over-bound tile too).
  creditAtAssumption passes t X_O δ with X_O = overCapDraws pl verdicts (draws, with multiplicity, on overCap tiles). ε = epsilon t X_O δ: least ε on the 1e-9 grid with P[Bin(t,ε) ≤ X_O] ≤ δ, +1e-9 (Clopper–Pearson; X_O=0 is today's, identical on ca34).
  Output: top-level over_cap; a tile over the bound keeps its reason, ok=true under uncredited; work gains over_cap_draws.
  Properties' AcceptMeans/TileAcceptMeans/WorkIsTheCredit/PlanMeans follow the new shapes, still unstated; they typecheck.
  Python -k (give-up ⇔ volunteer-needed, ε) passes locally; the difftest of both modes runs remotely next.

[2026-11-04T16:40Z] b2683afd4477
PEARLLEAN #7221: CMP-326 done — benchmarks/pouw/pearl_c_sm90 added to pearlplus_lean's [tool.verity.tests] inputs at e53a7b82409, pushed (on top of step 2's 812cfc50374).

[2026-11-04T16:46Z] 6817e8835027
PEARLLEAN: step 3 (CMP-317) pushed at eb37083edee on #7221: checkTile refuses k > K_MAX with the pass load domain reason ("DomainError: ..."), Words.ofInt untouched; difftest test_a_layer_past_k_max (k = 2^16+1). Steps 2 (812cfc50374) and CMP-326 (e53a7b82409) were pushed earlier. Step 1 difftest r20261104-162106-66ef on 1953b004b8c still running. Doing step 4 now.

[2026-11-04T16:52Z] 377df1700242
PEARLCAP (CMP-325): the forming term is pushed on cursor/pearlplus-forming-1de3 at 00eb2d1aa54, merged with #7221 at 324ca639256 (option A: capCheck charges the forming elements). Honest runs are unchanged: ca34 accepts with answer and all 459 verdicts identical to f592's (63,313,897,699 of 64,116,490,240 MACs; 0 forming elements), and run 1's replay accepts with its answer identical (7,812,084,949,428 of 7,911,115,436,032; 11 forming elements on 7 tiles, no ratio moved). Credit removed: 0 on both. core pouw suite passed; the Lean check and the benchmarks/pearlplus_lean/one_stage suites are rerunning on 00eb2d1aa54. Starting CMP-327 (band rule) next.

[2026-11-04T16:52Z] a023999d7b8d
PEARLLEAN: step 4 pushed at 324ca639256 on #7221.
- OpeningBinds is #7248's text exactly (segHashed, rowsHashed with FV3.uint for the leaf frame's position, `import Properties.Pouw.Commit`), and the layer entry has Properties.Pouw.Commit and #7248's why. It typechecks locally against the built pouw_pearlplus.
- The nine frozen reasons each state their property, with no lane names or note ids. All stay `unstated` and out of the guarantees list. OpeningBinds' reason says its proof comes in Security/Proofs (Proofs.Properties.Pouw.PearlPlus), not that it is proved here.
- No run id in a docstring any more: Unscale.lean, Rule.lean's H200_Y, the Properties module docstring, and the difftest's module and recorded-test docstrings. The difftest docstring now names over_cap and K_MAX.
Runs on 324ca639256: difftest plus the over-cap and K_MAX Python tests in r20261104-165203-5726, and the Lean check of Security/Properties in r20261104-165236-bb28. Step 1's r20261104-162106-66ef (on 1953b004b8c) is still running.
Friction: this VM's disk sits at the 1.5 GB research-run floor, filled by others' data (eviction finds nothing evictable). I freed 30 MB of my own scratch to launch; the next launch here may be refused again.

[2026-11-04T17:32Z] ab1b96779069
PEARLLEAN: CMP-328 pushed at dd9f749967e on #7221, on top of step 4 (324ca639256).
- Both sides add two refusals beside the one-scheme check:
  - "ValueError: the passes are not all of one salt";
  - unless the input is a test instance, "ValueError: pass i's tile is h x w, not 64 x 64: a test instance is read only as one, crediting nothing".
- In Python they live in verify_run.verify_passes. In Lean they live in Check.plan (after load), so Load.lean and Coins.lean are untouched.
- passes_distinct stays per salt.
- Test-instance path: verify_passes(test_instance=True), `served_passes.py verify --test-instance` (passed on to the Lean by `lean`), and `pouw-pearlplus --test-instance`. A test instance's work has credited = 0; every other field (macs, credited_tiles, rho, epsilon, over_cap_draws) is unchanged.
- PlanMeans now states one salt and, when testInstance = false, tileM = tileN = TILE (64). WorkIsTheCredit goes through testCredit. Both stay unstated, and their lock reasons are updated.
- Tests:
  - test_pearl_c_vllm: every verify_passes call on proved_pass passes test_instance=True, and served_verify passes --test-instance. The credit assertions now expect credited 0. The new test_verify_passes_reads_a_test_instance_only_as_one_and_a_runs_passes_of_one_salt covers both refusals; proved_pass gained a seed, and a different seed gives other weights and so another salt under one coin.
  - The difftest has a new case, test_a_served_runs_tile_and_salt.
  - Locally: the new Python tests plus the 10 others whose assertions changed pass (14 in all), and 6 difftest cases pass (Lean = Python).
Step 1 result (r20261104-162106-66ef, merge head 1953b004b8c): 8 of 9 difftest cases passed, including the recorded H200 served run (ACCEPT, Lean = Python). The control case hit the run's 3600 s timeout before it finished, so it was not a failure. I've raised the timeout to 7200 s.
Runs on dd9f749967e: difftest with the recorded H200 data in r20261104-173133-ee38, and test_pearl_c_vllm plus glm53 in r20261104-173224-b67e. The Security/Properties Lean check is still running on 324ca639256 (r20261104-165236-bb28). I cancelled r20261104-165203-5726: it was superseded and had the same short timeout.
Coverage change: a test instance credits 0, so the CPU difftest no longer compares `credited` at X_O > 0 between Lean and Python. It still compares ε, ρ, MACs, tiles and X_O. The recorded H200 run still compares credited, at X_O = 0.

[2026-11-04T17:52Z] 5f9d16633f03
PEARLCAP, 17:53Z. CMP-325 forming term: pushed at 00eb2d1aa54 on cursor/pearlplus-forming-1de3 (merged #7221 at 324ca639256). Its suites (r20261104-165229-de11): core pouw 2072 passed; 3 failures that are not the forming term: pearlplus_lean fails the read guard (its tests read benchmarks/pouw/run_result.py, settings.py and tests/test_pearl_c_sm90_prefill.py, which its pyproject does not declare: #7221 / PEARLLEAN), test_pearl_c_vllm_lean::test_an_agreeing_lean_verdict... (the answer tie field, #7221 code), and one_stage test_served_bandwidth (ConnectionError between the parties). CMP-327 band rule: done in Python and Lean (BAND = 64 at absolute indices, as auditTiling 64 64), with tests (a voluntary row zeroes its band, a padding row zeroes its block) and a Lean difftest (an excluded weight row zeroes its block). #7221 is merged at dd9f749967e, so the head is 72777b00df5, not pushed yet; suites, the Lean check and the ca34 and run 1 credit-loss replays are running. Proofs Cap.lean (d03d8373c42) has no band-rule statement yet to match. Heads-up: this VM has 1.3 GB of disk free and falling, so research run refuses launches under its 1.5 GB floor.

[2026-11-04T18:11Z] 640ef18003eb
PEARLLEAN #7221 interim, head dd9f749967e (CMP-328). The difftest r20261104-175718-dbce passed every CPU case, Lean = Python: 9 passed, among them the new test_a_served_runs_tile_and_salt. Its 2 recorded H200 cases skipped because vy-nebius-cpu-5 refused the store fetch with "no manifest locally (no remote configured)" (friction: --on auto keeps placing there). I have relaunched those cases pinned to vy-nebius-cpu-1 through its research pool slot, as r20261104-181111-b93e (the control case takes about 40 min). Earlier, b67e passed test_pearl_c_vllm and glm53 on dd9f. The Lean check bb28 on 324ca is still running. The final report follows when both end.

[2026-11-04T18:18Z] 747abd6c7a64
PEARLCAP, CMP-327 band rule, credit loss on run 1 (r20261103-225039-run1). As served: its V-EX gave up 2754 A + 1141 B rows (2736 A + 1135 B in prefill pass 3, for 3927 of 777,216 tiles over a cap). Under the band rule each give-up zeroes its 64-row band / 64-col block: credit 7,812,084,949,428 -> 4,667,340,670,145, a 40.25% loss (r20261104-171856-1a35, verifier 22bfa6c, ACCEPT). If the prover instead gives up nothing and option A credits the 3927 over-cap tiles 0: no-give-up credit 7,902,836,811,341 -> 7,870,512,158,675, a 0.409% loss (MACs 8,003,013,050,368 -> 7,970,279,260,160; r20261104-180344-acdc; at X_O=0 -- the ~2 expected over-cap draws at t=459 widen eps a little). Recommendation: under the band rule the honest prover should not volunteer; option A is the remedy (and is above run 1 as served). ca34 replay still running. Benchmark tests fixed for the band rule at 790fbaf5b57 (volunteer_words now one A row a band); verifying remotely before push.

[2026-11-04T18:23Z] ae8a97782c49
PEARLCAP, CMP-327 ca34 (r20261103-140949-ca34, f592 passes): as served (23 A + 8 B rows given up for its 31 over-cap tiles) the band rule costs 37.17%: 63,313,897,699 -> 39,777,003,483 (r20261104-172816-ec72, ACCEPT, verifier 22bfa6c). With nothing given up and option A crediting the 31 over-cap tiles 0: 63,899,730,031 -> 63,642,936,785, a 0.402% loss (r20261104-182257-5688). So both runs are under 0.5% (run 1 0.409%, ca34 0.402%) iff the prover stops volunteering. Same recommendation: drop V-EX give-ups under the band rule; option A is the remedy.

[2026-11-04T19:20Z] 1d67f0cf3749
PEARLCAP: pushed cursor/pearlplus-forming-1de3 at 5f285271917, the forming term (CMP-325) and band rule (CMP-327) together, stacked on #7221 dd9f749967e (its head). No passing forming-only sha: the Lean check at 00eb2d1 found my forming edit to Properties.Pouw.PearlPlus TileKeepsTheCap did not elaborate (mapM pattern lambda); fixed in 5f28527. Python: pouw benchmarks at 790fbaf pass but for #7221 test_pearl_c_vllm_lean:92 (tie field); pearlplus_lean fails only #7221 read guard (run_result.py, settings.py, test_pearl_c_sm90_prefill.py undeclared in its pyproject inputs) -- both PEARLLEAN-side. Lean check at 5f28527 running: Security/Properties r20261104-191856-abfd, pearlplus r20261104-191946-cd11; I report when they land.

[2026-11-04T19:21Z] 52e8209160d9
PEARLLEAN: #7221 is now at 99de0d9535e (pushed). I merged PEARLWTIE's cursor/pearlplus-lean-live-tie-1de3 as a fast-forward, which brings #7277's 50a763e8 with its PouwCommit.fieldIs compares unchanged (Load.lean 147/150/189/226 and Coins.lean). That covers steps 1, 2 and 3 of CMP-331. Step 4, PEARLCAP's forming term and band rule, is still to come, so this head isn't final yet.

Runs:
- The difftest at dd9f749967e on vy-nebius-cpu-1, r20261104-181111-b93e: 11 of 11 passed, including both recorded H200 cases (run and control, 46 min). Lean = Python on every case, the new test_a_served_runs_tile_and_salt among them. The run's rc of 1 is my job script's cleanup trap failing (rm -rf on the fetched read-only tree), not pytest.
- The same difftest on cpu-5, r20261104-175718-dbce: 9 CPU cases passed and the 2 recorded cases were skipped, because cpu-5's store has no remote configured.
- At 99de0d9535e, I built pouw_pearlplus locally (197 jobs, clean), and Properties.Pouw.PearlPlus elaborates on it with a probe: PlanMeans and WorkIsTheCredit carry CMP-328's one-salt and tile conjuncts and testCredit, and OpeningBinds uses only propext, Classical.choice and Quot.sound. Note for proofs: plan now goes through PEARLWTIE's afterTie, so GSJOINT's derivations of tile64 and index_inj from plan unfold through it.
- The Security/Properties Lean check at 324ca639256, r20261104-165236-bb28: the build passed (9588 jobs) and the kernel replay accepted with only the three allowed axioms. It reported 76 lock failures, and none names Pearl+: 71 are record drift in guarantees this branch doesn't touch (Flock, RemoteAudit, NCI, CoreProtocol, SampledProofs), 3 are missing witnesses on Lead.Commit, and 2 are VStar reads. All of #7221's Lean is new files, and no locked guarantee reads them. The run was then cancelled at its 5400 s timeout, 4296 s of it spent waiting on the Proofs side.

Next: I'm rerunning the recorded H200 difftest at 99de0d9535e on cpu-1, since live coins changed load, and I'll merge PEARLCAP's commits when you send their sha.

Friction:
- --on auto puts jobs on cpu-5, whose `research data fetch` fails with "no manifest locally (no remote configured)".
- This VM's ~/.research/runs disk sat at the 1.5 GB floor until about 18:30Z.
- RESEARCH_NOTES_TOKEN isn't set here, so my draft note (lanes/compute-accounting/20261103T2345Z-draft-pearlplus-lean-check.md) isn't synced.

[2026-11-04T19:31Z] 3ddc747dfcae
PEARLLEAN: #7221 is now at 3464b8ec887 (pushed). It's one commit on 99de0d9535e that closes a gap in my CMP-328 tile refusal. Please pass it on to proofs, since it adds a conjunct to PlanMeans.

The gap: a manifest's `instance` sets both k_min and tile, and `scheme_of` and `pearl_c` say a smaller bound of either makes a test instance (Pearl's domain is k ≥ 1024). The refusal checked only the tile, so a pass at 64x64 tiles with instance k_min 32 was credited as of record. That is the toy you suggested for the e2e (k = 128).

The fix, in Python and Lean together:
- Of record, every pass's instance must be the served one: k_min = pearl_c.K_MIN (1024) at 64x64 tiles. Anything else is a test instance's, read only under test_instance / --test-instance, crediting nothing.
- The refusal now reads "ValueError: pass j's instance is k_min K at M x N tiles, not the served 1024 at 64 x 64: a test instance is read only as one, crediting nothing".
- PlanMeans' conjunct is now `testInstance = false → ∀ p ∈ passes, p.scheme.kMin = K_MIN ∧ tileM = TILE ∧ tileN = TILE`, and its lock reason says the same.
- The honest served H200 manifests carry no `instance` (I checked art:4e14's prefill, decode and control), so they meet it.

Tests: proved_pass takes `tile` (at 64x64 its two weights have 64 rows). Both CMP-328 tests now hold the refusal at 16x16 and at 64x64 with k_min 32, and accept the 64x64 pass as a test instance's crediting 0. Locally: the pouw_pearlplus build is clean, Properties.Pouw.PearlPlus typechecks, and the Python test (18 s) and the difftest test_a_served_runs_tile_and_salt (17 s) pass. Ruff counts are unchanged.

Running at 3464b8ec887:
- r20261104-192941-f695, the pearlplus_lean difftest with the recorded H200 cases, on cpu-1. I cancelled the same run at 99de (8be2) for it.
- r20261104-193038-14a4, test_pearl_c_vllm, glm53 and one_stage's test_served_pearlplus_lean.

On the optional 64x64 toy helper: a toy of record now needs k ≥ 1024 as well as 64x64 tiles. That's roughly 8x the toy's Lean replay, plus a re-proved fixture. I'd leave it out of #7221, since you want the head frozen. If you want it, it can go in a stacked PR.

Still waiting on PEARLCAP's sha for step 4. The head isn't final until those commits are in.

[2026-11-04T19:51Z] f8a96d01373d
PEARLCAP, for PEARLLEAN (#7221 at dd9f749967e, seen on my stacked runs; none is from my change): (1) Lean check of pearlplus FAILs only on "orphan: PouwPearlPlus/Main.lean is not imported by the roots nor under exempt" (lean-audit.json exempt is {}; lakefile root PouwPearlPlus.Main) -- r20261104-191946-cd11; (2) benchmarks/one_stage tests/test_served_pearlplus_lean.py:71 verify rejects "pass 0 tile is 16 x 16, not 64 x 64" -- needs --test-instance after CMP-328 (r20261104-175205-a21e); (3) still: pearlplus_lean read guard (run_result.py, settings.py, tests/test_pearl_c_sm90_prefill.py undeclared) and benchmarks test_pearl_c_vllm_lean.py:92. Core pouw passes 2074 on my branch.

[2026-11-04T20:36Z] 5aa1167ff35c
PEARLCAP to proofs (GSJOINT, CMP-327): confirmed, the band rule is only in UnitRecord.credited (pearl_c_work.py, BAND = 64, absolute i//64, j//64) and its Lean mirror Records.credited (PouwPearlPlus/Records.lean); nothing else changed its rule. clear_work.tile_weight/tickets/draw (Python) and Run.ticketWeight/tickets/draw (Lean) read credited, as do the cap and credit. Records.credited is your Cap.lean bandCredited pointwise (the per-row voluntary/excluded filters are implied by the band/block ones), so it can replace it. Counterexample added as test_the_band_rule_holds_in_the_draw_a_tile_of_a_voluntary_rows_band_has_no_ticket (64x128x64, realRows 64, voluntaryA [5]: credited ([], 0..63), ticket 0, 200 draws never land on it) at a3f8a7d1f2b. Pushing now on #7221 3464b8ec887 (merged at f8b7bc0c4cc).

[2026-11-04T20:55Z] bbbec250b065
PEARLLEAN: runs for #7221 at head 3464b8ec887 (pushed, clean). Both pass.

- The pearlplus_lean difftest, r20261104-192941-f695 on vy-nebius-cpu-1: 16 of 16 passed (1 h 12 min). Lean = Python on all of them:
  - CMP-328's two refusals, including the k_min addition (test_a_served_runs_tile_and_salt);
  - PEARLWTIE's live epoch, challenger draw, of-record and W-tie cases;
  - both recorded H200 cases, the run read as of record and accepted, and the control rejected, as recorded.

  The run's rc of 1 isn't pytest's, whose summary reads "16 passed". It persists with the cleanup trap fixed, so it comes from elsewhere in my job wrapper (likely the bash -l login shell's exit). It isn't from the code.
- r20261104-193038-14a4 on cpu-5: test_pearl_c_vllm, test_pearl_c_sm90_glm53 and one_stage's test_served_pearlplus_lean gave 188 passed and 5 skipped (GPU-only screens), rc 0. The one-stage test ran the real Lean executable (37 s) and passed.

Steps 1–3 of CMP-331 are in #7221, plus the k_min commit. I'm waiting for PEARLCAP's sha (step 4) and will merge it, push and send the head marked final. #7221 stays unmarked until the Props' lock status is settled with proofs.

[2026-11-04T21:36Z] 2b52ae21e492
PEARLLEAN: item 1 is done. #7221 is at 6da3df271cb (pushed), one commit on 3464b8ec887. WorkIsTheCredit is now (∀ i opener, (check i opener).accept = true → ∃ pl, plan i = .ok pl ∧ work = some (testCredit … (creditAtAssumption …))) ∧ (∀ i opener, accept = false → work = none). The meaning is the same; only the conjunction moved to the top level. Checked locally on the built tree with lake env lean: Properties/Pouw/PearlPlus.lean elaborates, and a probe showed WorkIsTheCredit equals that explicit two-conjunct form by rfl (a planted 1 = 2 in the same probe was caught). Note for proofs: #7308 (5ca20b4b8a) is on an older base, without the tie, testCredit or K_MIN, so its proof of workIsTheCredit will need re-heading onto this statement. Item 2: origin/cursor/pearlplus-forming-1de3 is still at f8b7bc0c4cc, without the partially flagged row case, so I'm stopping here as you asked; resume me for the merge.

[2026-11-04T21:58Z] 2580f877c2be
PEARLCAP: pushed df3b87fcfa1 (on #7221 6da3df271cb). New: the partially-flagged-row difftest (46983091b29), test_the_forming_term_reads_a_block_every_credited_word_flags_and_not_one_some_flag: zero key, row 11 flags blocks 0-3 all in one column; at 16x16 some of its 16 credited words flag each block (not inert, debit 160 with or without E), at 16x1 its one credited word flags all four (inert, tile debit 672; 160 without E); Lean == Python under each over_cap rule (prototyped on the Python side; the Lean run is in py12). Earlier fails on cpu-5 were STORAGE_EXHAUSTED / no space (lean10 c96a, py11 a31a). Reruns pinned to vy-nebius-cpu-1: suites py12 r20261104-215704-2d43, pearlplus Lean r20261104-215734-3a6f; Security/Properties at f8b7bc0 r20261104-211014-f1df (cpu-1). Properties at 5f28527 built clean (my TileKeepsTheCap fix elaborates); its audit flags only non-PearlPlus lock records. I will name the final sha once these pass.

[2026-11-04T23:27Z] 4f470daed737
PEARLCAP final report. Branch cursor/pearlplus-forming-1de3, head df3b87fcfa1 (pushed; nothing more will be pushed).

ONE FIX FOR PEARLLEAN TO FOLD INTO #7221 FINAL HEAD: py13 (r20261104-222847-eff0, cpu-3, df3b87) fails benchmarks/pouw/pearlplus_lean::test_a_salt_dead_element_over_the_row_cap (16 passed, 1 failed). Cause: my test (fb59139) predates the CMP-328 test-instance rule; its direct call at tests/test_pearlplus_lean.py:210 lacks test_instance=True, so the 16x16 pass is rejected for its tile. Fix, one line: `without = answer(VR.verify_passes([path], 16, 1, "test", T.published(path), SOURCE, test_instance=True))`. Every other direct verify_passes call in that file already passes it. py13 is still running benchmarks/pouw and one_stage.

Shas: 00eb2d1aa54 forming term; b68acb3526c Python band rule; 22bfa6c2755 Lean credited and difftest; 4032672836b C4 tests; 72777b00df5 merge #7221@dd9f749; 790fbaf5b57 benchmark band-rule fixes; 5f285271917 TileKeepsTheCap fix; a3f8a7d1f2b proofs counterexample test (64xKx64, realRows 64, voluntaryA [5], in core test_pouw_pearl_c_work); f8b7bc0c4cc merge #7221@3464b8ec887; 46983091b29 partial-flag difftest; df3b87fcfa1 merge #7221@6da3df271cb.

Honest runs credit:
- Forming term: removes nothing on either run. Run 1 (r20261103-225039-run1): 11 forming elements over 7 tiles, credit unchanged (r20261104-145940-0690). ca34 (f592 passes): 0 elements, unchanged (r20261104-155225-53c9).
- Band rule as served (prover still volunteers): Run 1 7,812,084,949,428 -> 4,667,340,670,145 (-40.25%, r20261104-171856-1a35); ca34 63,313,897,699 -> 39,777,003,483 (-37.17%, r20261104-172816-ec72). Each V-EX give-up now zeroes a whole 64-band or block.
- Band rule, prover gives up nothing, over-cap tiles credited 0 (option A): Run 1 7,902,836,811,341 -> 7,870,512,158,675 (-0.409%, r20261104-180344-acdc); ca34 63,899,730,031 -> 63,642,936,785 (-0.402%, r20261104-182257-5688). Under the 0.5% expected.
Recommendation: under the band rule the honest prover should stop volunteering and take option A; volunteering costs ~38-40%.

Verification: core pouw 2075 passed at df3b87 (r20261104-215704-2d43, cpu-1). Benchmarks at 790fbaf: only #7221 failures (r20261104-181948-75e6). pearlplus Lean audit at df3b87 (r20261104-215734-3a6f): fails only on #7221 items (orphan PouwPearlPlus/Main.lean, unnamed instance at Coins.lean:51). Security/Properties built and audited at 5f28527 (r20261104-191856-abfd) and f8b7bc0 (r20261104-211014-f1df), failing only on non-PearlPlus lock records. Earlier #7221-side items: pearlplus_lean read guard, test_pearl_c_vllm_lean.py:92, one_stage test_served_pearlplus_lean needs --test-instance.

Cap.lean: band rule = bandCredited, forming test = formingCapOK (lean CKFORM3b). Open bridge assumptions: semH3 alpha/beta/x vs the FP32 s5 words; atomMixP hopper = FormChain.atomMix. Gap: the prover words-only remedy (volunteer_words) does not see the forming term, so a salt-dead element can still leave a tile over cap; option A covers it.

[2026-11-04T23:48Z] 100086a0c5c0
PEARLCAP: py13 (r20261104-222847-eff0, cpu-3) finished at df3b87fcfa1. verity-pouw-benchmarks 1358 passed, benchmarks/one_stage 656 passed, benchmarks/pouw/pearlplus_lean 16 passed 1 failed: only test_a_salt_dead_element_over_the_row_cap, the missing test_instance=True at test_pearlplus_lean.py:210 from my report. With that one line in #7221 final head, every suite I touch passes. Removing my worktree now; done.

