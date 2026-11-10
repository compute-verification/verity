# proofs' Goals page (CON-12, TOP-394): a row for each open goal. Draft by HOUSEKEEP, Tue 10 Nov 06:56Z

**Nothing here is written to the page.** `proofs-goals-fix.sh`, beside this file, holds one `research goals set` line a
goal (24 lines) and has not been run. The page was read at Tue 10 Nov 06:56Z (`research goals show --json`), after
top's re-cut (TOP-394) reached it: 287 goals, 25 open with owner `proofs`. After the script, counting its 24
goals: 24 open, 10 on track, 14 off track (8 with a link, 6 with a blocker text).

**To run it:** `ASK_128K=e9c0fbf9-c4f2-4858-97bb-6c1f135efb44 bash proofs-goals-fix.sh` (ASK-264's card). Without `ASK_128K` it stops before it writes
anything (ASK 4). `DRY=1 bash proofs-goals-fix.sh` prints the 24 commands. The texts are in `desc/` and
`blockers/`, one file a goal.

**Changed since the first draft (Tue 01:51Z):**

- **Top's re-cut reached the page** (TOP-394) between my reads of Tue 02:27Z and 04:58Z: 233 goals became 279. What it changes
  here: `spot-proof-server-steady-state` is kernels' goal now, so its line is gone (24 lines, from 25); top made
  two goals for proofs, rows 25 and 26, which get no line (ASK 16); you shortened four titles and fixed
  `spot-figures`' links yourself, so the script leaves those; top linked vllm's new goal under
  `spot-two-stage-e2e` and two of circuits' under `spot-128k-attention-proved`, and the script keeps them.
- **Your answers of Tue 02:18Z.** Status: `spot-verifier-cost` to on-track (ASK 9). Dates: `terminology-clean` to
  Fri 20 Nov (ASK 10), `zk-leaks-closed` to Mon 30 Nov (ASK 12). One goal added, row 24:
  `proofs/spot-two-stage-not-vacuous`, Fri 13 Nov 12 PM PST, in `spot-post-ready`'s links in place of the done
  `proofs/spot-architecture-matches` (ASK 14). Texts: `spot-numbers-lean-checked` drops the caption and
  `spot-figures` takes it, with its holder (ASKs 2, 11); `spot-advice-certified` names vllm and what it waits on
  (ASK 5); `spot-two-stage-e2e` carries your two dated steps (ASK 8); `spot-128k-attention-proved` says "an ask to
  Daniel" and its line waits on the card (ASK 4; since Tue 05:17Z the card is ASK-264's, and the text links it).
- **Two issues filed Tue 02:48Z that name these texts.** PRF-495 item 3 ("goes into HOUSEKEEP's Goals script"):
  `spot-two-stage-e2e` says that on main serving commits only the ids in and out, with the schedule's read and the
  history tables in review. PRF-494 (yours, Thu): `spot-two-stage-e2e` states the refusal at the last step, and
  `spot-two-stage-sound` and `spot-post-ready` name the three premises open in the lock on `main`
  (`CompiledTwo.Scope`, `SpotFlock.Embeds`, `SpotFlock.FrameIs`) [read 03:13Z]. None of my texts keeps the page's
  "proved for the Lean verifier with `Evaluates` and `Reveals` open". The wording that follows the page to Daniel
  stays yours.
- **Runs that reported.** Row 5: the first serve with the tables failed at Tue 04:06Z (a table declared twice,
  vllm's code on `main`) and its retry passed end to end at 06:06Z (FUSEDSERVE, 06:11Z). The row stays off track
  until G2 returns (your note headed 06:38Z), with your four reasons as its blocker text, and the description says
  what the retry showed and what is left. Rows 11 and 18 no longer say the design is "not served yet": they say its
  first serve passed and its price is still modelled.
- **The two asks (your note headed 06:38Z).** Row 6 links ASK-264's card in its text and no longer says "posted Mon
  11 PM"; the run command carries the card's id. ASK-268 made three texts of mine false, the ones saying the post
  claims no zero knowledge: rows 18, 20 and 21 are reworded (ASK 17). Row 8: MOEBIND2 answered that its open
  premise is not about the read's gates, and its PR is open. Row 26's cell names the sampling plan's label.
  Row 10 states the first timed run's result (SERVEDCARD, 02:08Z). Row 12 carries vllm's step of 02:46Z.
  Row 8 carries MOEBIND2's date of 05:14Z (ready to merge by Tue 08:00Z) and the premise it leaves open.
- **Still open:** row 5 again after G2 ends (Tue 08:52Z), ASK 4 (whether the 128k goal waits on a card the lane
  decided), ASK 17 (the reworded zero-knowledge sentences), ASK 16 (who holds P1), and one more re-diff after top's last objections (Tue 07:30Z).

**Rules I held to.** A status changes only where a source from the last 24 h supports it: on track needs a named
holder who is running and a dated plan that ends by the date; off track names what it waits on. A date changes only
to one a source gives. Where the sources do not say, the cell is an ASK and the script leaves the field as it is.
"Live" means a running worker holds it this week; "parked" means nobody at proofs does.

**Sources**, by the short names the table uses:

- IDX:n: line n of the swarm index, `20261106T0639Z-report-spot-swarm.md` (resume queue "updated Tue 00:38Z"; the
  LANDING paragraph, line 91, "read 01:00Z Tue"; line 204 is the UNSTAFFED row of the Mon queue). IDX:95 is the
  index as updated Tue 04:57Z.
- GA:n: line n of `20261109T1855Z-report-goal-audit.md` (Mon 18:55Z). The page was rewritten after it, Mon 19:06Z to 20:50Z.
- HH:n: line n of `20261109T1936Z-report-headline-honest.md` (BREAKEVEN, Mon 19:36Z). SC: `20261110T0002Z-report-smell-check-post.md`.
- N0104, N0114, N0147, N0218, N0638: your notes to me in `inbox/HOUSEKEEP.md`, headed Tue 01:04Z, 01:14Z, 01:47Z,
  02:18Z and 06:38Z.
- ASK-264 and ASK-268: their pages in the Asks for Daniel database, read Tue 06:53Z, and ASK-264's card by
  `research msg ask-daniel status` (kind default, decided, choice a, source lane).
- CON12: a comment on CON-12, with its lead and time. An issue key with a name and a time is that issue's comment.
- TOP-394: top's issue for the eleven goals; proofs' comments there are yours, Mon 7:33 PM to 8:45 PM PST.
- [gh 01:26Z]: `gh pr view`, read by me then. Times are UTC unless marked PST. The texts for the page use Pacific
  times with no zone, as the site's rules ask.

## The table

| # | Goal | This week | Status: now, proposed | Date we now believe | The one thing it waits on | Worker | Source |
|---|---|---|---|---|---|---|---|
| 1 | `slice-clean` | parked | off-track (kept) | **Fri 13 Nov 6 PM PST** (from Tue 10 Nov 6 PM PST). CHECK: it is PRF-387's date, set when the post was Wed 11 Nov | nothing linked (its two links are done goals; dropped). Blocker text: no proofs worker, top holds it. `top/slices-clean` (Fri 20 Nov) links it | none at proofs; top (the page's own text) | IDX:204 "slice-clean (move to top)"; GA:59 "owner says proofs, top does it ... move to top"; PRF-387 coordinator Sun 00:20Z "moved to Fri 13 Nov ... unstaffed until Thu"; #7850 open, untouched since Sat 7 Nov [gh 01:26Z] |
| 2 | `spot-bind-term` | live | off-track, **on-track** | **Fri 13 Nov** (from Wed 11 Nov 6 PM PST). The source gives the day; 6 PM PST is the project's hour | `ci/fine-grained-check` (main moving), after BINDKSAT's one PR, marked Thu 12 Nov 12:00Z | BINDKSAT; ZKSTACK (two witnesses) | PRF-466, coordinator Tue 00:33Z: "pieces elaborate by Tue 10 Nov 04:00Z; witnesses by Wed 11 Nov 12:00Z; lock entries by Wed 18:00Z; one PR marked Thu 12 Nov 12:00Z; on `main` Fri 13 Nov"; IDX:104, IDX:105 |
| 3 | `spot-checks-closed` | live | off-track, **on-track** | Wed 11 Nov 6 PM PST (unchanged) | nothing outside proofs: the coordinator's own ruling on items 5, 6 and 7, Wed 11 Nov 18:00Z | the coordinator (no worker: GA:43, IDX:204) | N0114: "PRF-461: Due 2026-11-11 (its goal's date) ... I settle 5, 6 and 7 by Wed 11 Nov 18:00Z"; GA:82-97 (six of eight settled; item 1's run a1ee) |
| 4 | `spot-numbers-lean-checked` | live (work done, waiting to merge) | on-track (kept) | Wed 11 Nov 6 PM PST (unchanged) | `ci/fine-grained-check`: #8346 (on `checked`) and #8360 (on `next`) reach `main` | FIG1LEAN (done, idle) | N0218 item 2 ("`write_form` is captioned, not stated in Lean, and the caption is a piece of `spot-figures` ... done when #8346 and #8360 are on `main`"); GA:53 "keep; Wed 11 Nov"; IDX:91 (#8360 on `next`); both PRs open [gh 01:26Z] |
| 5 | `spot-fused-cut-served` | live | off-track (kept) | Thu 12 Nov 12 PM PST (unchanged) | FUSEDSERVE's G2 (`r20261110-062528-1eb6`, launched; it waits for Tue 08:10Z and holds one card until 08:52Z at the latest): the four served fused statements at 1, then 4, then 2 provers, 32 timed sessions each; then the fit, Wed 11 Nov 18:00Z. The retry `r20261110-042659-14a1` passed (exit 0, Tue 06:06Z, at `dc0a6e8fe6`); `ks-form` said ok on all six in `r20261110-055846-47b2`. #8454 gets a new head with the timing tool's fix, and its land ask is on hold until that head is marked. The done link `circuits/glm-production-compile` is dropped | FUSEDSERVE | FUSEDSERVE's entry headed Tue 06:11Z in your inbox ("the retry passed end to end"; "The four fused members proved under `--zk`, warm: 16 sessions by one prover and 64 by four at once, all 80 accepted by the Rust verifier in loopback"; "Lean's `ks-form` says ok on all six"; "The two scale reads were not proved"; "The Commit's own verdict is PENDING, since its CPU replay did not run"; "measured, but too thin to replace FUSED_H200 today"; "no text says the fused cut is served"), its entry headed 06:31Z ("G2 is launched ... 32 timed sessions per K, 2 warm per prover, verify-ahead 2") and its entries headed 05:50Z and 05:53Z; N0218 item 3; HH:179 "the fit Wed 11 Nov 18:00Z; goal ... Thu 12 Nov 12 PM PST" |
| 6 | `spot-128k-attention-proved` | live | off-track (kept) | Thu 12 Nov 6 PM PST (fixed, N0104) | `circuits/glm-sparse-attention-zk` (CIR-210's `_v2` PR, marked Tue 20:00Z). Also linked: `circuits/glm-long-dsa-row`, `vllm/served-128k-window-committed`, and the two top added Tue 03:59Z, `circuits/glm-served-calls-fit-block` and `circuits/glm-top-p-zk` (kept, though circuits' own text says Thursday's request needs no top-p), and ASK-264's card `e9c0fbf9-c4f2-4858-97bb-6c1f135efb44` (the script's `ASK_128K`) | TWOPROVE, DSAREAD, ACCOUNTD, PROVECUT | N0104 (both fixed points and fact 2); N0218 item 4 ("no card exists yet. ASKPAGE drafts the page by 04:30Z and I post page and card by 07:00Z"); CON12 circuits 01:07Z and vllm 01:11Z (the three links); IDX:95, 98, 100, 101; CIR-210 circuits Tue 00:21Z; SC section 1 |
| 7 | `spot-advice-certified` | live (vllm's) | off-track (kept) | Thu 12 Nov 6 PM PST (unchanged; N0218 item 5) | vllm's replay of a request's cache hit (VLLM-260's verdict, Tue 06:00Z). vllm has no goal on the page for it [page 02:27Z], so the blocker text says it and nothing is linked | vllm (VLLM-242) | N0218 item 5; VLLM-242 vllm Tue 00:29Z ("It waits on VLLM-260 (each sampled replay re-proves its request's hit), whose bit-identity verdict is due Tue 10 Nov 06:00Z. Next: restate this issue on that verdict, Tue 10 Nov 10 AM PST"); GA:51 |
| 8 | `spot-moe-binding` | live | off-track (kept) | Thu 12 Nov 6 PM PST (unchanged; N0114 confirms it for PRF-458) | MOEBIND2's records rerun `r20261110-062950-413d` (for the reworded lock reason; its guess: ends about Tue 06:55Z), then the quick tier marks #8565 (head `38e2f7d067`): "marked by Tue 10 Nov 08:00Z". Then main moving. `ExpertReadAccepted` stays `open` in the lock ("to be proved by PRF-458"), and so does `Hash.RowLeaf.accepts` (kernels', LEAN-186, due 30 Nov), already on `main` | MOEBIND2 | MOEBIND2's entry in your inbox headed Tue 06:34Z ("No: `ExpertReadAccepted` is about the template check's code, not the read's gates"; "`TableReadBinding` quantifies over every circuit whose template `check` accepts, and it assumes nothing of the gates"; "So the goal's first sentence holds on Thursday as worded"; "`r20261110-051140-4130` (cpu-5): passed"; "PR: #8565, head `38e2f7d067`") and its entries headed 05:09Z and 05:14Z; N0114; N0218 item 6; IDX:112. I did not read the Lean statement [not checked] |
| 9 | `spot-one-stage-premises` | live (proofs' part done) | on-track (kept) | Thu 12 Nov 6 PM PST (unchanged) | `ci/fine-grained-check`: #8411 (admitted, `1562e328da`) reaches `main`; lean's LEAN-235 line, marked by Tue 10 Nov 12 PM PST | TRUST6 (done); lean for LEAN-235 | LEAN-186: lean Mon 22:46Z, proofs 22:55Z, lean 23:03Z ("LEAN-235 ... marked by Tue 10 Nov 12 PM PST"; "provisional: a default, not yet a ruling"); IDX:91 (#8411 admitted); SC section 8 |
| 10 | `spot-overhead-measured` | live | off-track (kept) | Thu 12 Nov 6 PM PST (unchanged) | `circuits/glm-served-calls-fit-block` (the link circuits asked for). Also true, not linked: `circuits/glm-sparse-attention-zk` (the refused 10%) | SERVEDCARD; HASHCUT | SERVEDCARD's note, checkpoint Tue 02:08Z ("Window 1 ... proved 65 of 74 staged statements, all 34 types they cover, with every one of the 260 `serve --zk` sessions accepted by Flock's loopback verifier. There was no Lean verdict (NO_PROGRAM)") and 01:17Z ("The form-refused share is 9.8-10.8% of the card's cost"); HH:180; CON12 circuits 01:07Z |
| 11 | `spot-ten-percent` | live | off-track (kept) | Thu 12 Nov 6 PM PST (unchanged) | `proofs/spot-fused-cut-served` (no figure under 10% is served until it is). Links unchanged: its four open pieces and three done ones; one of the four, `proofs/spot-proof-server-steady-state`, is kernels' since Tue 04:06Z | BREAKEVEN (lead since Mon 19:15Z; idle until vllm's hashing rate, Tue 02:45Z); HASHCUT | HH:8 (BREAKEVEN "lead of proofs/spot-ten-percent from 19:15Z"), HH:41, HH:179-180; LEDGER's note line 104 (H7: 9.80%, 97.05%, 25.76%, each at 32 bits a token, modelled, world year); IDX:110; SC section 6 |
| 12 | `spot-two-stage-e2e` | live | off-track (kept) | Thu 12 Nov 6 PM PST (unchanged; N0218 item 8) | `vllm/two-stage-schedule-committed` (vllm's new goal, tier 3, due Tue 10 Nov 2 PM PST, linked by top Tue 03:59Z; kept: the line passes no `--wait`). Also, not a goal: the verifier's link read, marked by Wed 11 Nov 06:00Z | TWOPROVE, TWOVERIFY, TWOOPENS, ACCOUNTD; vllm-twostage | N0218 item 8; PRF-495 item 3 (its description: "Item 3 goes into HOUSEKEEP's Goals script"; `PORTS = (INGRESS, EGRESS)` at `boundary_rows.py:68` on `main` [read 03:12Z]; #8429 and #8434 open [gh 03:12Z]); PRF-494 (the goal states the verdict; `FORM : Bool := false`, `SecondStage.lean:115` on `main` [read 03:12Z]); VLLM-250: vllm-twostage Mon 23:19Z ("the advice step is built and passes on the endpoint fixture. The test on real data runs after 02:10Z"), proofs Tue 02:19Z (asks vllm for a goal id), vllm-twostage Tue 02:46Z ("the advice step passes on last night's served window ... Tuesday's 16:00Z serve runs the step itself"); the page's blocker (coordinator, Mon 20:43Z); LEDGER's note, checkpoints Mon 22:09Z and 23:24Z; IDX:98, 111 |
| 13 | `spot-verifier-cost` | live | off-track, **on-track** | Thu 12 Nov 6 PM PST (unchanged; N0218 item 9: PRF-464's Fri is one later PR) | `ci/fine-grained-check`: #8425 (marked, `8d40023923`) and #8453 (admitted, `0054c395fe`) reach `main`. **Kept beside it:** `circuits/verifier-templates-not-unrolled`, the link circuits asked for (N0147). N0218 says it waits "only" on the queue: delete that one `--wait` if you mean circuits' link goes | VERIFYFAST-B | N0218 item 9; VERIFYFAST-B's checkpoints Tue 02:03Z ("#8425 is marked at 8d40023923 and #8453 is admitted") and 00:55Z ("1,899.6 core-s flat and 257.7 templated"); CON12 circuits 01:07Z |
| 14 | `terminology-clean` | parked | on-track, **off-track** | **Fri 20 Nov 6 PM PST** (from Thu 12 Nov 6 PM PST; N0218 item 10) | the SPOT post being out, Mon 16 Nov (blocker text; its one link is a done goal, dropped). `top/slices-clean` (Fri 20 Nov) links it, which agrees with this date | none until the post is out (RENAME idle: GA:63) | N0218 item 10 ("after the post. Date Fri 20 Nov 6 PM PST, off-track, blocker text 'nobody until the post is out (Mon 16 Nov)'. Owner stays proofs"); GA:63 ("539 'sampled proofs', 247 'C-Flock' on main"); PRF-361 LINEARHYG Mon 12:32Z (#7153 and #8174 reached main Mon 07:33Z) |
| 15 | `spot-figures` | live from Wed 11 Nov | off-track (kept) | Fri 13 Nov 12 PM PST (unchanged; N0114: "closes with the figures (Fri 13 Nov)") | `proofs/spot-overhead-measured`: its final card run, verdict Wed 11 Nov 18:00Z; and `ci/fine-grained-check`. Both links are yours of Tue 04:17Z, and the line passes no `--wait`, so they stay | BREAKEVEN (N0218 item 11) | N0218 item 11 ("BREAKEVEN renders them from the labelled rows and writes `write_form`'s caption, after the final card run (verdict Wed 11 Nov 18:00Z). Date unchanged"); CON12 top Mon 5:10 PM PST (the dead link); N0114 on PRF-456; GA:48 |
| 16 | `spot-claims-checked` | live | off-track, **on-track** | Fri 13 Nov 6 PM PST (unchanged) | nothing outside proofs: LEDGER's passes Tue 03:00Z, 06:00Z, 12:00Z, 20:00Z, then daily through Fri | LEDGER | IDX:107; IDX:193 ("RESUME DAILY through Fri | LEDGER (... goal spot-claims-checked)"); LEDGER's note `20261109T2010Z-report-claims-ledger` (checkpoint Mon 20:54Z: 19 secondary claims; H7 and H10 marked NOT) |
| 17 | `spot-output-accounting` | live | off-track (kept) | Fri 13 Nov 6 PM PST (unchanged) | `network/cap-certified` (NET-281: #8405's fix, then ACCOUNTD's #8462, target Wed 18:00Z) | ACCOUNTD; ZKSTACK (item 5, #8311) | ACCOUNTD's checkpoint Tue 00:12Z (#8462 "waits on the fix for #8405", "target Wed 18:00Z"; the F = 1 PR "Mark by Thu 12 Nov 12:00Z"); NET-281 network-accounting Mon 23:35Z; PRF-462 LINEARHYG Mon 00:36Z; GA:309-321; #7898, #7933, #7934, #7908 merged [gh 01:26Z] |
| 18 | `spot-post-ready` | live | off-track (kept) | Fri 13 Nov 6 PM PST (fixed, N0104) | its pieces: 14 open of proofs' (the new goal of row 24 among them), `circuits/glm-compile-cost-measured` (in place of the done `circuits/glm-128k-indexer`), 1 done (`proofs/spot-glm-replay`). 16 links, the cap; the done `proofs/spot-architecture-matches` leaves to make room. Tier 1 was taken off by top (Tue 03:59Z): it sits under `accounting/network-glm53` | the coordinator | N0104 (fixed point; "every goal the post reads is a piece of it"); CON12 circuits 01:07Z; CON12 proofs Mon 14:19Z (tier 1, set, with its reason); GA:64 |
| 19 | `weight-security-bridge` | live | on-track (kept) | Fri 13 Nov 6 PM PST (unchanged) | `ci/fine-grained-check`: #8311 (`317bdbc1`, admitted) and #8400 (`9d9fe580`, marked; carries #8224's commits) reach `main` | ZKSTACK, CUTLEMMA | IDX:91, IDX:99 (#8400 marked at `9d9fe58033`; "No hold for #8224"), IDX:105; GA:58 ("LeaksThrough open to 17 Nov ... Fri 13 Nov; state the premises") |
| 20 | `zero-knowledge` | parked | off-track (kept) | **Mon 30 Nov** (from Fri 20 Nov; N0114) | the SPOT post being out, Mon 16 Nov: nobody is on it until then. Links unchanged (`proofs/zk-leaks-closed` and six done goals), so the page shows the links and hides the blocker text | none until the post is out | N0114 "PRF-366 ... Nobody is on it until the post is out (Mon 16 Nov) ... 30 Nov is the date I believe ... The goal `proofs/zero-knowledge` gets the same date"; N0104; GA:65 ("keep parked; rewrite") |
| 21 | `zk-leaks-closed` | parked | off-track (kept) | **Mon 30 Nov 6 PM PST** (from Fri 20 Nov; N0218 item 12: with its parent) | the SPOT post being out, Mon 16 Nov (blocker text; it has no links, so the page shows it) | none until the post is out | N0218 item 12; N0104 ("have nobody on them until the post is out: say so as their blocker"); GA:66; the page's own text for what is on main |
| 22 | `sampling-generic` | live | on-track (kept) | Mon 30 Nov 6 PM PST (unchanged, N0104) | its pieces (unchanged): `proofs/spot-two-stage-sound`, `proofs/spot-one-stage-premises`, `proofs/spot-moe-binding`. Tier 1 was taken off by top (Tue 03:59Z): it sits under `accounting/network-glm53` | the two-stage Lean workers (see `spot-two-stage-sound`) | N0104 (the project for `spot-two-stage-sound`); CON12 proofs Mon 14:19Z and soundness Tue 01:00Z (its link to `proofs/sampled-audits-sound` is dropped, so the text no longer names it); GA:67 ("keep; rewrite") |
| 23 | `spot-two-stage-sound` | live | on-track (kept) | Mon 30 Nov 6 PM PST (unchanged) | nothing outside proofs. Next dated steps: TWOWITNESS's statement (A) Tue 18:00Z, witness (D) Fri 12:00Z; #8309's mark after records 7632 | TWOOPENS, SLOTSAT, TWOWITNESS | IDX:96 (#8309 at `70cb152cb5`, mark after records 7632), IDX:111 ("SLOTSAT: #8363 after #8309; Friday's non-vacuity ... TWOWITNESS: statement (A) frozen Tue 18:00Z, witness (D) Fri 12:00Z"); PRF-419 LINEARHYG Mon 12:32Z ("`FORM := false` refuses every two-stage run ... due 30 Nov"); GA:68 |
| 24 | `spot-two-stage-not-vacuous` | live | not on the page, **on-track** | **Fri 13 Nov 12 PM PST** (new; N0218 item 14) | nothing outside proofs. Dated steps: TWOWITNESS's statement (A) frozen Tue 18:00Z, witness (D) Fri 12:00Z | SLOTSAT with TWOWITNESS | N0218 item 14 ("add the goal for Friday's non-vacuity ... due Fri 13 Nov 12 PM PST, holder SLOTSAT with TWOWITNESS; it takes the place of one done piece in `spot-post-ready`'s links"); IDX:111; GA:68 |
| 25 | `accounting/network-glm53` (no line) | live (goal 6 of top's eleven) | off-track (top's, kept) | Fri 13 Nov 6 PM PST (top's; proofs on TOP-394: "prints Thu 12 Nov; on `main` Fri 13 Nov. The 300 GB has no date") | 12 links: P1 (`proofs/network-number-printed`), `spot-128k-attention-proved`, `spot-ten-percent`, `spot-fused-cut-served`, `spot-output-accounting`, `spot-two-stage-e2e`, `spot-post-ready`, `sampling-generic`, three of network's and `circuits/glm-compile-cost-measured`. The blocker text top wrote is hidden by the links | the coordinator (owner of 6); BREAKEVEN for the figure | the page (made by top Tue 04:12Z); TOP-394: proofs Mon 7:33 PM PST ("Owner of 6"), 7:57 PM (8,456 GB), 8:08 PM (the path stops at 427 GB); top's tree. **ASK 16** |
| 26 | `proofs/network-number-printed` (no line) | **ASK 16**: no holder named in what I read | on-track (top's, kept) | Thu 12 Nov 6 PM PST (top's; proofs on TOP-394: P1, "Thu 12 Nov") | nothing linked. Its test: benchmarks' one command prints the world-year GB at 10% with network's term beside the compute cut | **ASK 16** (IDX:95 lists P1 under you, with no worker) | the page (made by top Tue 04:01Z); TOP-394: proofs Mon 7:33 PM PST (P1) and 7:51 PM ("The numbers as JSON ...: P1 prints that way"); IDX:95. The sampling plan the command reads is labelled: `art:2ea6f506e79536372ddf0b9c0741dcb68888b3c74c60efd7739450023bcb3c65` carries `run_of_record=prove-glm53-spot-sampling`, by proofs, Tue 06:27Z [run: `research data labels`, 06:54Z]; BREAKEVEN's entry headed 06:15Z says it is modelled |

## The ASKs

**Still open:**

- **ASK 4, the card exists; the run command now carries its id:** ASK-264, card `e9c0fbf9-c4f2-4858-97bb-6c1f135efb44`, posted Tue 05:17Z, kind default, decided by the lane for A, which applies Wed 11 Nov 06:00Z [read: `research msg ask-daniel status`, and the page, 06:53Z]. The text links it and no longer says "posted Mon 11 PM". The script still stops without `ASK_128K`. Yours: whether a goal should wait on a card the lane has decided (`goals.py` says "an open ask"); `ASK_128K=none` drops the link and keeps the text's. (`spot-128k-attention-proved`)
- **ASK 17, a text of mine that ASK-268 made false:** three texts said the post claims no zero knowledge. ASK-268 (card `368336ce-485a-4c79-8c97-337a9a039937`, posted Tue 06:31Z, decided by the lane for A, applies Wed 11 Nov 02:00Z [read on its page, 06:53Z]) asks what the post says about zero knowledge, and its option A prints a sentence with its limits. So `spot-post-ready` now says that this is an ask to Daniel and links the card, and `zero-knowledge` and `zk-leaks-closed` say the post's system has no recursion layer and no zero-knowledge server (AGENTS.md: SPOT with Flock on its own "has none of that"). Say if the first sentence of `spot-post-ready` should read otherwise. (`spot-post-ready`)

**Answered (your notes headed 02:18Z and 06:38Z, and the workers' entries named), and applied:**

- **ASK 1, answered:** leave the owner; you ask top in your CON-12 reply and add `--owner top` on a yes. No change to the line. (`slice-clean`)
- **ASK 2, answered:** `write_form` is captioned, not stated in Lean, and the caption belongs to `spot-figures`. This goal's text no longer names it. (`spot-numbers-lean-checked`)
- **ASK 3, answered in N0638:** off-track until G2 returns, and the description stays. The blocker text holds your four reasons in the page's form. Your own wording passes the site's check and trips three of the skill's rules (a PR number, UTC times, "marked"); to use it as written, replace the file `blockers/spot-fused-cut-served.md` with: "G2 (Tue 08:10Z to 08:52Z) has to return a usable rate at 1, 2 and 4 provers; the two scale reads are not proved; the Commit's verdict is PENDING, its CPU replay did not run; #8454's new head is not marked". After G2 I apply your rule: on-track only if its rows are not void and FUSEDSERVE dates the scale reads and the Commit's replay before Thu 12 Nov 12 PM PST. (`spot-fused-cut-served`)
- **ASK 5, answered:** vllm holds it under VLLM-242; the date stays; the blocker text is yours, in plain words. vllm has no goal on the page for the replay of a cache hit, so no link. (`spot-advice-certified`)
- **ASK 6, answered by MOEBIND2 (06:34Z):** `ExpertReadAccepted` is about the template check's code, not the read's gates, so the goal's first sentence stands. The text now says what the open premise is. The date in the texts is MOEBIND2's "marked by Tue 10 Nov 08:00Z". (`spot-moe-binding`)
- **ASK 8, answered and closed:** the date stays, the text carries your two dated steps, and vllm's goal `vllm/two-stage-schedule-committed` is on the page and linked (top, Tue 03:59Z). The line keeps that link, drops the blocker text (the page would hide it) and the description links the goal. (`spot-two-stage-e2e`)
- **ASK 9, answered:** Thu stays; on-track, linked to `ci/fine-grained-check` (my rule allows it: VERIFYFAST-B runs, both changes are in the queue since Tue 02:03Z). I kept circuits' link beside it; see the row. (`spot-verifier-cost`)
- **ASK 10, answered:** after the post: Fri 20 Nov 6 PM PST, off-track, "nobody until the post is out", owner unchanged. (`terminology-clean`)
- **ASK 11, answered:** BREAKEVEN renders them and writes the caption after the final card run. I left the status off-track, because it waits on `spot-overhead-measured`, which is off track; say if a holder and a dated plan make it on-track for you. (`spot-figures`)
- **ASK 12, answered:** Mon 30 Nov, with its parent. (`zk-leaks-closed`)
- **ASK 13, answered and done:** PRF-363 is Done since Tue 02:22Z, with your comment's facts checked on `gh` (five PRs merged, two closed unmerged, #7228 open). No change to the line. (`sampling-generic`)
- **ASK 14, answered:** the goal for Friday's non-vacuity is row 24 and the script's first line. It takes the place of the done `proofs/spot-architecture-matches` in `spot-post-ready`'s links, which stay at 16. (`spot-two-stage-sound`)
- **ASK 7, moot:** `spot-proof-server-steady-state` is kernels' goal since Tue 04:06Z (5b; your TOP-394 comment
  handed it over). Its line is gone from the script. Its new text has one thing for proofs: by **Fri 13 Nov 10 AM
  PST**, "@proofs has confirmed that the blocks measured are the blocks proved". No worker of ours is named for that.
- **ASK 15, yours for Wed (PRF-384, PRF-454, PRF-372).** None of the three sits under one of these goals, so none
  has a row: PRF-384 and PRF-372 carry labels of goals other leads own (`proofs/soundness-nothing-open`, soundness;
  `demos/onsite-demo`), and PRF-454's goal, `proofs/spot-architecture-matches`, is done. PRF-384: the 7 Nov plan's
  Mon and Tue pieces show no commit. PRF-454: no worker or date for the layering moves; since Tue 02:22Z it also
  carries lean's PRF-452. PRF-372: its 20 Nov is before the onsite goal's 24 Nov.
- **ASK 16, new: the two goals top made for proofs** (rows 25 and 26). They have no line: their texts are top's,
  written Tue 04:01Z and 04:12Z. Two questions. Who holds P1, `proofs/network-number-printed`? It is on track on
  the page, and I found no worker named for it. And do you want either text rewritten to the skill's rules (they
  name TOP-394, PRF-456 and NET-295, and write "PST"; the site accepts them)?

## Where each goal sits in top's tree (TOP-394), as the page has it

The page's tier 1 is now top's eleven. Top took tier 1 off `spot-post-ready` and `sampling-generic` (Tue 03:59Z).
Three goals keep a set tier 2 (`spot-claims-checked`, `spot-figures`, `spot-overhead-measured`). The script sets
no tier. After the script, every goal of proofs' is linked from a goal above it:

- **Goal 6, `accounting/network-glm53`** (owner proofs) links `network-number-printed` (P1),
  `spot-128k-attention-proved` (P2), `spot-ten-percent` (P3), `spot-fused-cut-served` (P4),
  `spot-output-accounting` (P5), `spot-two-stage-e2e` (P6), `spot-post-ready` (P7) and `sampling-generic`.
- **`spot-post-ready`** links 14 open pieces of proofs', among them the new `spot-two-stage-not-vacuous`.
  **`sampling-generic`** links `spot-two-stage-sound`. **`spot-ten-percent`** and `spot-figures` link
  `spot-overhead-measured`; `spot-ten-percent` also links `spot-verifier-cost`.
- **Goal 5, `benchmarks/proof-system-performance`** links `spot-verifier-cost`.
- **Goal 3**, through `top/slices-clean` (Fri 20 Nov): `slice-clean` and `terminology-clean`.
- **Parked**, through `top/parked` (Mon 30 Nov): `zero-knowledge`, which links `zk-leaks-closed`.

Your TOP-394 comment lists `spot-two-stage-not-vacuous` among proofs' open goals; it is on the page only once the
script's first line runs.

**Off track means a date or a blocker.** After the script 14 are off track: 8 link the goal they wait on, and 6
carry a blocker text because what they wait on is not a goal on the page.

**Titles.** You shortened the four top named (Tue 04:17Z), so the script leaves them. One title change is left:

- `spot-checks-closed`: "Guarantee checks closed or scoped (PRF-461)" (6 words) becomes "Guarantee checks closed or scoped" (5).

   Its title carried a ticket id.

## What the script does that the table does not show

- **The gap for ASK 4.** The 128k goal's line carries `--wait ask:$ASK_128K`. The script stops with exit 2, before
  any write, when `ASK_128K` is empty; `ASK_128K=none` drops the link; under `DRY=1` it prints `ask:THE-ASK-CARD-ID`.
- **The new goal goes first**, so that `spot-post-ready`'s line, which links it, finds it on the page.
- **`--wait` only where the list changes.** `--wait` replaces a goal's whole list, so a line passes it only when
  its list differs from the page's as read at Tue 10 Nov 06:56Z. 11 lines pass it. Links other leads added tonight
  are kept: top's two under `spot-128k-attention-proved` (`circuits/glm-served-calls-fit-block`,
  `circuits/glm-top-p-zk`), top's `vllm/two-stage-schedule-committed` under `spot-two-stage-e2e`, and your two
  under `spot-figures`. **A link added to one of those 11 goals after Tue 10 Nov 06:56Z would be dropped**: I
  re-diff before you run.
- **Links other leads asked for (CON12, Tue 01:07Z and 01:11Z; N0147):** `circuits/glm-long-dsa-row`,
  `circuits/glm-sparse-attention-zk` and `vllm/served-128k-window-committed` under `spot-128k-attention-proved`
  (on the page already); `circuits/glm-served-calls-fit-block` under `spot-overhead-measured`;
  `circuits/verifier-templates-not-unrolled` under `spot-verifier-cost`; `circuits/glm-compile-cost-measured`
  under `spot-post-ready` in place of the done `circuits/glm-128k-indexer`; `spot-fused-cut-served` drops the done
  `circuits/glm-production-compile`. No goal passes 16 links: `spot-post-ready` is at 16 exactly.
  **Every goal id the script names resolves** on the page as read at Tue 10 Nov 06:56Z, the new goal aside [run]:

  - `ci/fine-grained-check`: ci, on-track, Tue 10 Nov 6 PM PST, "Main moves on the verified queue's check"
  - `vllm/served-128k-window-committed`: vllm, on-track, Thu 12 Nov 8 AM PST, "A served 128k GLM-5.3 window committed for audit"
  - `circuits/glm-long-dsa-row`: circuits, on-track, Wed 11 Nov 10 AM PST, "GLM-5.3's long DSA rows in provable units"
  - `circuits/glm-served-calls-fit-block`: circuits, on-track, Wed 11 Nov 10 AM PST, "GLM-5.3's served calls each fit a Flock block"
  - `circuits/glm-sparse-attention-zk`: circuits, on-track, Wed 11 Nov 10 AM PST, "GLM-5.3's sparse attention accepted under zero knowledge"
  - `circuits/glm-top-p-zk`: circuits, on-track, Fri 13 Nov 10 AM PST, "GLM-5.3's top-p sampling accepted under zero knowledge"
  - `circuits/verifier-templates-not-unrolled`: circuits, on-track, Wed 11 Nov 10 AM PST, "Templates the verifier reads without unrolling"
  - `circuits/glm-compile-cost-measured`: circuits, on-track, Fri 13 Nov 12 PM PST, "GLM-5.3's compile cost measured for the post"
  - `proofs/spot-glm-replay`: proofs, done, Mon 09 Nov 6 AM PST, "The headline model replays deterministically"
  - and 13 open goals of proofs' own, each among the 24; `proofs/spot-two-stage-not-vacuous` is the one the script makes.

- **Waiting on `main`:** the five on-track goals whose work is written and waits to merge link
  `ci/fine-grained-check` ("Main moves on the verified queue's check", ci, due Tue 10 Nov 6 PM PST, on track), the
  narrowest goal for it and due before each; circuits' goals link the same one since top's re-cut. If CI-496's
  answer (Tue 18:00Z) says the queue cannot carry them by their dates, those five go off track with the same link.
- **Blocker texts.** The page shows a goal's blocker text only when the goal has no link
  (`goals-view.tsx:615` in compute-verification/infra: `m.waitingOn?.length ? undefined : m.blockers`). So a goal
  with links gets `--no-blockers` (its stale text was hidden anyway) and says where it stands in its description.
  `zero-knowledge` keeps its seven links, so "nobody until the post is out" shows in its description and through
  `zk-leaks-closed`, whose blocker text the page does show.
- **Tags.** Four goals had none: `spot-overhead-measured`, `spot-figures` and `spot-claims-checked` take `proofs`;
  `slice-clean` takes `cleanup` and `proofs`, as lean's, vllm's, circuits' and network's slice goals take theirs.
  The new goal takes `proofs`. Drop the flags if you'd rather not.
- **Words.** The texts name no worker, PR, issue, run or branch: "a @proofs worker", "in review", "waits to
  merge". The Worker column here has the names. If you want names in the texts, say so.

## What the script does not do

- It sets no priority and no tier, moves no owner, marks nothing done and removes nothing. The priorities changed
  tonight (1 on the goals under the eleven, 3 on `zero-knowledge` and `zk-leaks-closed`); `research goals set`
  sends no priority (`STORED_ONLY` in `goals.py`), so they stay.
- It adds one goal and no other. The skill's rule that an open goal due within a week waits on a tier-3 step due
  within a day is not applied: that would be about 20 new goals, and top's list does not ask for it.
- It does not touch the goals with a `proofs/` id that other leads own (`proofs/sampled-audits-sound`,
  `proofs/soundness-nothing-open`, `proofs/proof-check-fast`, and now `proofs/spot-proof-server-steady-state`),
  or the two goals top made for proofs tonight (rows 25 and 26).

## Checks I ran, and what I could not check

- **The site's check, ported.** The site checks a goal's whole text on every write (`parseGoal`,
  `website/apps/docs/lib/goals/format.ts` and `plain.ts` in compute-verification/infra, read at 01:24Z): the four
  words Daniel doesn't know ("served window" among them), a Greek letter or a superscript outside math, a word with
  an underscore outside code, a list with no lead-in line, an off-track goal with neither a link nor a blocker, more
  than 16 links. A Python port of those rules passes all 24 goals as the script leaves them [run].
- **The skill's rules the site does not check:** no PR number, issue key, run id, agent id, branch, file path,
  worker name, time zone or pipeline word ("marked", "landed", "admitted") in any text; titles of at most eight
  words; two to four sentences a description. All 24 pass [run].
- **The commands.** Each of the script's 24 lines went through `research goals`' own argument parser and edit
  (`research.goals.main`) against a stand-in route that answers from the saved page and sends nothing: 24
  writes built, the new goal's among them, none refused by the command [run]. So a flag that cannot be combined, or
  a missing file, would have shown. The guard was tried once without `ASK_128K` and without `DRY`: exit 2, and the
  page's goals read the same before and after [run].
- **Not checked:** a real write (I ran no `research goals set`); that the deployed site runs the `plain.ts` I read,
  and what it asks of an `ask:` id; how the page draws 16 links; whether top agrees to own `slice-clean`; every
  figure in a text (the 14,903 steps, 97.05%, 9.80%, 25.76%, 1,899.6 and 257.7 core-seconds, 539 and 247 uses,
  22.7 GB and 69.9 MB, 65 of 74 and 260 sessions): each is quoted from the note or the page text the Source column
  names, and I opened none of the runs.
