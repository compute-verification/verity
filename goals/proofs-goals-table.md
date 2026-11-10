# proofs' Goals page (CON-12): the 24 open goals, a row each. Draft by HOUSEKEEP, Tue 10 Nov 01:51Z

**Nothing here is written to the page.** `proofs-goals-fix.sh`, beside this file, holds one `research goals set` line a
goal and has not been run. The page was read at Tue 10 Nov 01:49Z (`research goals show --json`): 228 goals, 24 open with
owner `proofs`, 18 off track. After the script: 24 open, 16 off track, each with a link or a blocker text; 8 on track.

**To check in ten minutes:** read the ASKs below (each says what the script does meanwhile), then the cells in bold
in the "Status" and "Date" columns: four statuses and three dates change. `DRY=1 bash proofs-goals-fix.sh` prints the
commands. The texts are in `desc/` and `blockers/`, one file a goal.

**Rules I held to.** A status changes only where a source from the last 24 h supports it: on track needs a named
holder who is running and a dated plan that ends by the date; off track names what it waits on. A date changes only
to one a source gives. Where the sources do not say, the cell is an ASK and the script leaves the field as it is.
"Live" means a running worker holds it this week; "parked" means nobody at proofs does.

**Sources**, by the short names the table uses:

- IDX:n: line n of the swarm index, `20261106T0639Z-report-spot-swarm.md` (resume queue "updated Tue 00:38Z"; the
  LANDING paragraph, line 91, "read 01:00Z Tue"; line 204 is the UNSTAFFED row of the Mon queue).
- GA:n: line n of `20261109T1855Z-report-goal-audit.md` (Mon 18:55Z). The page was rewritten after it, Mon 19:06Z to 20:50Z.
- HH:n: line n of `20261109T1936Z-report-headline-honest.md` (BREAKEVEN, Mon 19:36Z). SC: `20261110T0002Z-report-smell-check-post.md`.
- N0104, N0114, N0147: your notes to me in `inbox/HOUSEKEEP.md`, Tue 01:04Z, 01:14Z and 01:47Z.
- CON12: a comment on CON-12, with its lead and time. An issue key with a name and a time is that issue's comment.
- [gh 01:26Z]: `gh pr view`, read by me then. Times are UTC unless marked PST. The texts for the page use Pacific
  times with no zone, as the site's rules ask.

## The table

| # | Goal | This week | Status: now, proposed | Date we now believe | The one thing it waits on | Worker | Source |
|---|---|---|---|---|---|---|---|
| 1 | `slice-clean` | parked | off-track (kept) | **Fri 13 Nov 6 PM PST** (from Tue 10 Nov 6 PM PST). CHECK: it is PRF-387's date, set when the post was Wed 11 Nov | nothing linked (its two links are done goals; dropped). Blocker text: no proofs worker, top holds it | none at proofs; top (the page's own text) | IDX:204 "slice-clean (move to top)"; GA:59 "owner says proofs, top does it ... move to top"; PRF-387 coordinator Sun 00:20Z "moved to Fri 13 Nov ... unstaffed until Thu"; #7850 open, untouched since Sat 7 Nov [gh 01:26Z] |
| 2 | `spot-bind-term` | live | off-track, **on-track** | **Fri 13 Nov** (from Wed 11 Nov 6 PM PST). The source gives the day; 6 PM PST is the project's hour | `ci/fine-grained-check` (main moving), after BINDKSAT's one PR, marked Thu 12 Nov 12:00Z | BINDKSAT; ZKSTACK (two witnesses) | PRF-466, coordinator Tue 00:33Z: "pieces elaborate by Tue 10 Nov 04:00Z; witnesses by Wed 11 Nov 12:00Z; lock entries by Wed 18:00Z; one PR marked Thu 12 Nov 12:00Z; on `main` Fri 13 Nov"; IDX:104, IDX:105 |
| 3 | `spot-checks-closed` | live | off-track, **on-track** | Wed 11 Nov 6 PM PST (unchanged) | nothing outside proofs: the coordinator's own ruling on items 5, 6 and 7, Wed 11 Nov 18:00Z | the coordinator (no worker: GA:43, IDX:204) | N0114: "PRF-461: Due 2026-11-11 (its goal's date) ... I settle 5, 6 and 7 by Wed 11 Nov 18:00Z"; GA:82-97 (six of eight settled; item 1's run a1ee) |
| 4 | `spot-numbers-lean-checked` | live (work done, waiting to merge) | on-track (kept) | Wed 11 Nov 6 PM PST (unchanged) | `ci/fine-grained-check`: #8346 (on `checked`) and #8360 (on `next`) reach `main` | FIG1LEAN (done, idle); nobody on `write_form` | GA:53 "keep; Wed 11 Nov", GA:292-297 ("`write_form` has no owner"); IDX:91 (#8360 on `next`); both PRs open [gh 01:26Z] |
| 5 | `spot-fused-cut-served` | live | off-track (kept) | Thu 12 Nov 12 PM PST (unchanged) | FUSEDSERVE's first Commit with the tables: G1 Tue 03:20Z, verdict 04:15Z; the fit Wed 11 Nov 18:00Z. The done link `circuits/glm-production-compile` is dropped | FUSEDSERVE | IDX:103; FUSEDSERVE's checkpoints Tue 01:05Z and 01:15Z (#8454 marked at `283188b306`, no review yet; node A's disk 94% full); HH:179 "the fit Wed 11 Nov 18:00Z; goal ... Thu 12 Nov 12 PM PST"; CON12 circuits 01:07Z |
| 6 | `spot-128k-attention-proved` | live | off-track (kept) | Thu 12 Nov 6 PM PST (fixed, N0104) | `circuits/glm-sparse-attention-zk` (CIR-210's `_v2` PR, marked Tue 20:00Z); also linked: `circuits/glm-long-dsa-row`, `vllm/served-128k-window-committed` | TWOPROVE, DSAREAD, ACCOUNTD, PROVECUT | N0104 (both fixed points and fact 2); CON12 circuits 01:07Z and vllm 01:11Z (the three links); IDX:95, 98, 100, 101; CIR-210 circuits Tue 00:21Z; SC section 1 |
| 7 | `spot-advice-certified` | parked | off-track (kept) | **ASK 5** (page: Thu 12 Nov 6 PM PST; nobody holds it, so no source gives a date) | nothing linked. Blocker text: shown only with caching off; no worker | none (IDX:204) | IDX:204 "spot-advice-certified with caching on" under UNSTAFFED; GA:51 "0 decisions over 3,000, with prefix caching OFF ... re-scope to caching on; Thu 12 Nov"; the page's blocker (coordinator, Mon 19:07Z) |
| 8 | `spot-moe-binding` | live | off-track (kept) | Thu 12 Nov 6 PM PST (unchanged; N0114 confirms it for PRF-458) | MOEBIND2's fix of its failed records run (963a, Tue 00:51Z); no date. Then main moving | MOEBIND2 | N0114 "PRF-458: Due 2026-11-12 (the goal `proofs/spot-moe-binding`). MOEBIND2 holds it; its records run ... failed the audit Tue 00:51Z and the fix is under way"; IDX:112 |
| 9 | `spot-one-stage-premises` | live (proofs' part done) | on-track (kept) | Thu 12 Nov 6 PM PST (unchanged) | `ci/fine-grained-check`: #8411 (admitted, `1562e328da`) reaches `main`; lean's LEAN-235 line, marked by Tue 10 Nov 12 PM PST | TRUST6 (done); lean for LEAN-235 | LEAN-186: lean Mon 22:46Z, proofs 22:55Z, lean 23:03Z ("LEAN-235 ... marked by Tue 10 Nov 12 PM PST"; "provisional: a default, not yet a ruling"); IDX:91 (#8411 admitted); SC section 8 |
| 10 | `spot-overhead-measured` | live | off-track (kept) | Thu 12 Nov 6 PM PST (unchanged) | `circuits/glm-served-calls-fit-block` (the link circuits asked for). Also true, not linked: `circuits/glm-sparse-attention-zk` (the refused 10%) | SERVEDCARD; HASHCUT (A/B 02:20-03:20Z) | IDX:97 ("76 of 78 staged statements accepted by the ks-form mirror ... Window 1 runs no Lean verifier"; refused share "about 10%, at most 12%"); IDX:115 (Window 1 01:35-02:10Z); HH:180; CON12 circuits 01:07Z |
| 11 | `spot-proof-server-steady-state` | parked | off-track (kept) | Thu 12 Nov 6 PM PST (unchanged; N0114 gives PRF-468 2026-11-12) | `circuits/glm-served-calls-fit-block`: #8281's units on `main`, which the rerun measures | **ASK 7** (STATIONARY idle; no row in IDX) | N0114 "PRF-468: Due 2026-11-12. Left: the goal's rerun at steady state on #8281's units, and circuits' Mon 16:57Z request"; GA:45 ("fold into spot-overhead-measured"; STATIONARY idle); PRF-468 circuits Mon 16:57Z |
| 12 | `spot-ten-percent` | live | off-track (kept) | Thu 12 Nov 6 PM PST (unchanged) | `proofs/spot-fused-cut-served` (no figure under 10% is served until it is). Links unchanged: its four open pieces and three done ones | BREAKEVEN (lead since Mon 19:15Z; idle until vllm's hashing rate, Tue 02:45Z); HASHCUT | HH:8 (BREAKEVEN "lead of proofs/spot-ten-percent from 19:15Z"), HH:41, HH:179-180; LEDGER's note line 104 (H7: 9.80%, 97.05%, 25.76%, each at 32 bits a token, modelled, world year); IDX:110; SC section 6 |
| 13 | `spot-two-stage-e2e` | live | off-track (kept) | **ASK 8** (page: Thu 12 Nov 6 PM PST; no source dates the end-to-end run) | vllm's schedule commit before the coins (VLLM-250, no goal on the page); TWOPROVE's link after #8413. Blocker text says both | TWOPROVE, TWOVERIFY, TWOOPENS, ACCOUNTD; vllm-twostage | the page's blocker (coordinator, Mon 20:43Z); VLLM-250 vllm-twostage Mon 23:19Z ("the advice step is built and passes on the endpoint fixture. The test on real data runs after 02:10Z"); LEDGER's note, checkpoints Mon 22:09Z and 23:24Z; IDX:98, 111; GA:60 |
| 14 | `spot-verifier-cost` | live | off-track (kept) | Thu 12 Nov 6 PM PST (unchanged). **ASK 9**: PRF-464 says Fri 13 Nov for its PR | `circuits/verifier-templates-not-unrolled` (CIR-203; the link circuits asked for) | VERIFYFAST-B | VERIFYFAST-B's checkpoint Tue 00:55Z ("1,899.6 core-s flat and 257.7 templated"; Lean accepted all 35 verifies under `--zk`); IDX:102 (#8453, #8425 to be marked); PRF-464 LINEARHYG Mon 12:32Z ("PR E (VERIFYFAST-B), due Fri 13 Nov"); CON12 circuits 01:07Z |
| 15 | `terminology-clean` | parked | on-track, **off-track** | **ASK 10** (page: Thu 12 Nov 6 PM PST, on track; nobody holds it) | nothing linked (its one link is a done goal; dropped). Blocker text: no worker, rename not run | none (RENAME idle: GA:63; IDX:204) | GA:63 ("539 'sampled proofs', 247 'C-Flock' on main ... after the post, or to top"); IDX:204 "terminology-clean (after post / top)"; PRF-361 LINEARHYG Mon 12:32Z (#7153 and #8174 reached main Mon 07:33Z; #7716 open, not marked) |
| 16 | `spot-figures` | parked | off-track (kept) | Fri 13 Nov 12 PM PST (unchanged; N0114: "closes with the figures (Fri 13 Nov)") | `proofs/spot-ten-percent` (BREAKEVEN's labelled rows). `proofs/spot-draft-tonight` (no such goal) and the done `vllm/tick-tradeoff-measured` are dropped | **ASK 11** (OPTMODES inactive: GA:48; IDX:204) | CON12 top Mon 5:10 PM PST (the dead link); N0114 on PRF-456; IDX:204 "card labels (spot-figures)"; GA:48; the page's blocker (coordinator, Mon 19:06Z) |
| 17 | `spot-claims-checked` | live | off-track, **on-track** | Fri 13 Nov 6 PM PST (unchanged) | nothing outside proofs: LEDGER's passes Tue 03:00Z, 06:00Z, 12:00Z, 20:00Z, then daily through Fri | LEDGER | IDX:107; IDX:193 ("RESUME DAILY through Fri | LEDGER (... goal spot-claims-checked)"); LEDGER's note `20261109T2010Z-report-claims-ledger` (checkpoint Mon 20:54Z: 19 secondary claims; H7 and H10 marked NOT) |
| 18 | `spot-output-accounting` | live | off-track (kept) | Fri 13 Nov 6 PM PST (unchanged) | `network/cap-certified` (NET-281: #8405's fix, then ACCOUNTD's #8462, target Wed 18:00Z) | ACCOUNTD; ZKSTACK (item 5, #8311) | ACCOUNTD's checkpoint Tue 00:12Z (#8462 "waits on the fix for #8405", "target Wed 18:00Z"; the F = 1 PR "Mark by Thu 12 Nov 12:00Z"); NET-281 network-accounting Mon 23:35Z; PRF-462 LINEARHYG Mon 00:36Z; GA:309-321; #7898, #7933, #7934, #7908 merged [gh 01:26Z] |
| 19 | `spot-post-ready` | live | off-track (kept) | Fri 13 Nov 6 PM PST (fixed, N0104) | its pieces: 13 open of proofs', `circuits/glm-compile-cost-measured` (in place of the done `circuits/glm-128k-indexer`), 2 done. 16 links, the cap | the coordinator | N0104 (fixed point; "every goal the post reads is a piece of it"); CON12 circuits 01:07Z; CON12 proofs Mon 14:19Z (tier 1, set, with its reason); GA:64 |
| 20 | `weight-security-bridge` | live | on-track (kept) | Fri 13 Nov 6 PM PST (unchanged) | `ci/fine-grained-check`: #8311 (`317bdbc1`, admitted) and #8400 (`9d9fe580`, marked; carries #8224's commits) reach `main` | ZKSTACK, CUTLEMMA | IDX:91, IDX:99 (#8400 marked at `9d9fe58033`; "No hold for #8224"), IDX:105; GA:58 ("LeaksThrough open to 17 Nov ... Fri 13 Nov; state the premises") |
| 21 | `zero-knowledge` | parked | off-track (kept) | **Mon 30 Nov** (from Fri 20 Nov; N0114) | the SPOT post being out, Mon 16 Nov: nobody is on it until then. Links unchanged (`proofs/zk-leaks-closed` and six done goals), so the page shows the links and hides the blocker text | none until the post is out | N0114 "PRF-366 ... Nobody is on it until the post is out (Mon 16 Nov) ... 30 Nov is the date I believe ... The goal `proofs/zero-knowledge` gets the same date"; N0104; GA:65 ("keep parked; rewrite") |
| 22 | `zk-leaks-closed` | parked | off-track (kept) | Fri 20 Nov 6 PM PST (unchanged; N0104). **ASK 12** | the SPOT post being out, Mon 16 Nov (blocker text; it has no links, so the page shows it) | none until the post is out | N0104 ("have nobody on them until the post is out: say so as their blocker"); GA:66; the page's own text for what is on main |
| 23 | `sampling-generic` | live | on-track (kept) | Mon 30 Nov 6 PM PST (unchanged, N0104) | its pieces (unchanged): `proofs/spot-two-stage-sound`, `proofs/spot-one-stage-premises`, `proofs/spot-moe-binding` | the two-stage Lean workers (see `spot-two-stage-sound`) | N0104 (the project for `spot-two-stage-sound`); CON12 proofs Mon 14:19Z and soundness Tue 01:00Z (its link to `proofs/sampled-audits-sound` is dropped, so the text no longer names it); GA:67 ("keep; rewrite") |
| 24 | `spot-two-stage-sound` | live | on-track (kept) | Mon 30 Nov 6 PM PST (unchanged) | nothing outside proofs. Next dated steps: TWOWITNESS's statement (A) Tue 18:00Z, witness (D) Fri 12:00Z; #8309's mark after records 7632 | TWOOPENS, SLOTSAT, TWOWITNESS | IDX:96 (#8309 at `70cb152cb5`, mark after records 7632), IDX:111 ("SLOTSAT: #8363 after #8309; Friday's non-vacuity ... TWOWITNESS: statement (A) frozen Tue 18:00Z, witness (D) Fri 12:00Z"); PRF-419 LINEARHYG Mon 12:32Z ("`FORM := false` refuses every two-stage run ... due 30 Nov"); GA:68 |

## The ASKs

- **ASK 1.** GA:59 and IDX:204 both say "move to top". Has top agreed to own it? If yes, add `--owner top`. The script leaves the owner. (`slice-clean`)
- **ASK 2.** Who states `write_form` in Lean or captions it, and by when? GA:296 proposes the figures worker; there is none (ASK 11). (`spot-numbers-lean-checked`)
- **ASK 3.** Status after G1's verdict (Tue 04:15Z): the script keeps off-track with the blocker text above. If G1 passed, change the line to `--status on-track --no-blockers` and cut the booking from the text. I re-read FUSEDSERVE's note after 04:20Z and update this row. (`spot-fused-cut-served`)
- **ASK 4.** IDX:114 says Daniel was asked at 00:03Z what Thursday's result is called while `FORM` is false. vllm (PRF-477, 01:10Z) finds no page and no card for it. If a card exists, add `--wait ask:<card id>`; if not, the text's "Daniel's call" names an ask not yet filed. (`spot-128k-attention-proved`)
- **ASK 5.** Who runs it with caching on, and by when? GA:51 names vllm and network as the owners of the run. The script leaves the date. (`spot-advice-certified`)
- **ASK 6.** MOEBIND2 has no dated step in the index. What date does it give for a change in review? Until then the script keeps off-track with the blocker text. (`spot-moe-binding`)
- **ASK 7.** Who holds the rerun? And GA:45 proposes folding this goal into `spot-overhead-measured`, while N0114 keeps PRF-468 with work left: keep the goal (the script does) or `research goals remove` it? (`spot-proof-server-steady-state`)
- **ASK 8.** What date do TWOPROVE and vllm give for one run through both rounds? And should vllm put its two-stage serving side (VLLM-250) on the page, so this goal links it? (`spot-two-stage-e2e`)
- **ASK 9.** Thu 12 Nov 6 PM PST (the page, and `spot-ten-percent` needs it then) or Fri 13 Nov (PRF-464)? The script leaves Thu. (`spot-verifier-cost`)
- **ASK 10.** After the post, or handed to top (`top/repo-cleaned-up`, Fri 13 Nov, waits on it)? The script sets off-track with the blocker text and leaves the date and the owner. (`terminology-clean`)
- **ASK 11.** Who renders the figures (and takes `write_form`, ASK 2)? (`spot-figures`)
- **ASK 12.** Does it follow `proofs/zero-knowledge` to Mon 30 Nov? N0104 says Fri 20 Nov for both, N0114 moves only the parent. The script leaves Fri 20 Nov. (`zk-leaks-closed`)
- **ASK 13 (PRF-363, yours for Wed).** Its issue under this goal: only #7228, a skill PR untouched since 4 Nov, is still attached and it has no done-condition. Close it, or say what is left. (`sampling-generic`)
- **ASK 14.** GA:68 proposes a goal of its own for Friday's non-vacuity, a piece of `spot-post-ready` (whose old blocker names it). The script has that goal as a commented line; adding it puts `spot-post-ready` at 17 links, one over the cap, so it would replace one done piece. (`spot-two-stage-sound`)
- **ASK 15 (PRF-384, PRF-454, PRF-372, yours for Wed).** None of the three sits under one of these 24 goals, so none
  has a row: PRF-384 and PRF-372 carry labels of goals other leads own (`proofs/soundness-nothing-open`, soundness;
  `demos/onsite-demo`), and PRF-454's goal, `proofs/spot-architecture-matches`, is done. PRF-384: the 7 Nov plan's
  Mon and Tue pieces show no commit. PRF-454: no worker or date for the layering moves. PRF-372: its 20 Nov is
  before the onsite goal's 24 Nov.

## top's four lines, and where each is fixed

1. *The page says what we are working on now.* Every row's text is rewritten to today (two to four sentences: what
   will be true, where it stands with its date, who holds it). Nothing is marked done or removed: no source says
   one of the 24 is done or dropped. The one candidate for removal is ASK 7 (`spot-proof-server-steady-state`).
2. *Every goal sits under a project.* After the script: `spot-post-ready` (tier 1, set) links 13 open pieces of
   proofs', one of circuits' and 2 done; `spot-fused-cut-served`, `spot-overhead-measured`,
   `spot-proof-server-steady-state` and `spot-verifier-cost` sit under its piece `spot-ten-percent`;
   `sampling-generic` (tier 1, set) links `spot-two-stage-sound` and two pieces it shares with the post;
   `zero-knowledge` (tier 1, derived: nothing waits on it) links `zk-leaks-closed`; `slice-clean` and
   `terminology-clean` sit under top's `top/repo-cleaned-up`. No goal of the 24 is tier 1 for want of a link.
3. *Off track means a date or a blocker.* 16 stay or become off track. 9 link the goal they wait on; 7 carry a
   blocker text because what they wait on is not a goal on the page (no worker, or proofs' own work).
4. *Our lines.* `spot-figures` no longer links `proofs/spot-draft-tonight`. The four not written in 48 h
   (`terminology-clean`, `zero-knowledge`, `zk-leaks-closed`, `spot-two-stage-sound`) are rewritten. Titles:

- `spot-checks-closed`: "Guarantee checks closed or scoped (PRF-461)" (6 words) becomes "Guarantee checks closed or scoped" (5).
- `spot-overhead-measured`: "SPOT's proof overhead measured on the headline model's blocks" (9 words) becomes "Proof overhead measured on GLM-5.3's served blocks" (7).
- `spot-verifier-cost`: "Verifier cost in the cost table, driven to O(description)" (9 words) becomes "Verifier cost tracked and driven down" (6).
- `terminology-clean`: "Retired terms gone from the tree, and new uses refused" (10 words) becomes "Retired terms gone, and new uses refused" (7).
- `spot-claims-checked`: "Every claim in the SPOT post traced to evidence" (9 words) becomes "Every claim in the post traced to evidence" (8).

   The first was not on top's list: its title carried a ticket id. No priority is set.

## What the script does that the table does not show

- **Links other leads asked for (CON12, Tue 01:07Z and 01:11Z; your note of 01:47Z), all seven in the script:**
  `circuits/glm-long-dsa-row`, `circuits/glm-sparse-attention-zk` and `vllm/served-128k-window-committed` under
  `spot-128k-attention-proved`; `circuits/glm-served-calls-fit-block` under `spot-overhead-measured`;
  `circuits/verifier-templates-not-unrolled` under `spot-verifier-cost`; `circuits/glm-compile-cost-measured`
  under `spot-post-ready` in place of the done `circuits/glm-128k-indexer`; `spot-fused-cut-served` drops the done
  `circuits/glm-production-compile`. No goal passes 16 links: `spot-post-ready` is at 16 exactly, so nothing was
  dropped that you'd miss, and nothing more fits. **Every goal id the script names resolves** on the page as read
  at Tue 10 Nov 01:49Z [run]:

  - `ci/fine-grained-check`: ci, on-track, Tue 10 Nov 6 PM PST, "Main moves on the verified queue's check"
  - `circuits/glm-sparse-attention-zk`: circuits, on-track, Wed 11 Nov 10 AM PST, "GLM-5.3's sparse attention accepted under zero knowledge"
  - `circuits/glm-long-dsa-row`: circuits, on-track, Wed 11 Nov 10 AM PST, "GLM-5.3's long DSA rows in provable units"
  - `vllm/served-128k-window-committed`: vllm, on-track, Thu 12 Nov 8 AM PST, "A served 128k GLM-5.3 window committed for audit"
  - `circuits/glm-served-calls-fit-block`: circuits, on-track, Wed 11 Nov 10 AM PST, "GLM-5.3's served calls each fit a Flock block"
  - `circuits/verifier-templates-not-unrolled`: circuits, on-track, Wed 11 Nov 10 AM PST, "Templates the verifier reads without unrolling"
  - `network/cap-certified`: network-accounting, on-track, Tue 10 Nov 6 PM PST, "The response cap is certified and published"
  - `circuits/glm-compile-cost-measured`: circuits, on-track, Fri 13 Nov 12 PM PST, "GLM-5.3's compile cost measured for the post"
  - `proofs/spot-glm-replay`: proofs, done, Mon 09 Nov 6 AM PST, "The headline model replays deterministically"
  - `proofs/spot-architecture-matches`: proofs, done, Sun 08 Nov 6 AM PST, "Code shows the three-role architecture"
  - and 14 open goals of proofs' own, each among the 24.

- **Waiting on `main`:** the four on-track goals whose work is written and waits to merge link
  `ci/fine-grained-check` ("Main moves on the verified queue's check", ci, due Tue 10 Nov 6 PM PST, on track), the
  narrowest goal for it and due before each. If CI-496's answer (Tue 18:00Z) says the queue cannot carry them by
  their dates, those four go off track with the same link.
- **Blocker texts.** The page shows a goal's blocker text only when the goal has no link
  (`goals-view.tsx:615` in compute-verification/infra: `m.waitingOn?.length ? undefined : m.blockers`). So a goal
  with links gets `--no-blockers` (its stale text was hidden anyway) and says where it stands in its description.
  `zero-knowledge` keeps its seven links, so "nobody until the post is out" shows in its description and through
  `zk-leaks-closed`, whose blocker text the page does show.
- **Tags.** Four goals had none: `spot-overhead-measured`, `spot-figures` and `spot-claims-checked` take `proofs`;
  `slice-clean` takes `cleanup` and `proofs`, as lean's, vllm's, circuits' and network's slice goals take theirs. Drop the flags if you'd rather not.
- **Words.** The texts name no worker, PR, issue, run or branch: "a @proofs worker", "in review", "waits to
  merge". The Worker column here has the names. If you want names in the texts, say so.

## What the script does not do

- It sets no priority and no tier, moves no owner, marks nothing done and removes nothing.
- It adds no goal. The skill's rule that an open goal due within a week waits on a tier-3 step due within a day is
  not applied: that would be about 20 new goals, and top's list does not ask for it. ASK 14 is the one candidate.
- It does not touch the goals with a `proofs/` id that other leads own (`proofs/sampled-audits-sound`,
  `proofs/soundness-nothing-open`, `proofs/proof-check-fast`).

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
  (`research.goals.main`) against a stand-in route that answers from the saved page and sends nothing: 24 writes
  built, none refused by the command [run]. So a flag that cannot be combined, or a missing file, would have shown.
- **Not checked:** a real write (I ran no `research goals set`); that the deployed site runs the `plain.ts` I read;
  how the page draws 16 links; whether top agrees to own `slice-clean`; every "[relayed]" figure in a text
  (the 14,903 steps, 97.05%, 9.80%, 25.76%, 1,899.6 and 257.7 core-seconds, 539 and 247 uses, 22.7 GB and 69.9 MB,
  76 of 78): each is quoted from the note or the page text the Source column names, and I opened none of the runs.

