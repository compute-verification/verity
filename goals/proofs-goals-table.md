# proofs' Goals page (CON-12): the 24 open goals and one new, a row each. Draft by HOUSEKEEP, Tue 10 Nov 03:14Z

**Nothing here is written to the page.** `proofs-goals-fix.sh`, beside this file, holds one `research goals set` line a
goal (25 lines) and has not been run. The page was read at Tue 10 Nov 02:27Z (`research goals show --json`): 233
goals, 24 open with owner `proofs`, 18 off track. After the script: 25 open, 10 on track, 15 off track (8 with a link, 7 with a blocker text).

**To run it:** `ASK_128K=<the ask card's id> bash proofs-goals-fix.sh`. Without `ASK_128K` it stops before it writes
anything (ASK 4). `DRY=1 bash proofs-goals-fix.sh` prints the 25 commands. The texts are in `desc/` and
`blockers/`, one file a goal.

**Changed since the first draft (01:51Z), all from your note headed 02:18Z:**

- Status: `spot-verifier-cost` to on-track (ASK 9). Dates: `terminology-clean` to Fri 20 Nov (ASK 10),
  `zk-leaks-closed` to Mon 30 Nov (ASK 12).
- Links: `spot-proof-server-steady-state` now waits on `proofs/spot-fused-cut-served` (ASK 7);
  `spot-verifier-cost` gains `ci/fine-grained-check`; `spot-128k-attention-proved` gains the ask card (ASK 4).
- One goal added, row 25: `proofs/spot-two-stage-not-vacuous`, Fri 13 Nov 12 PM PST, in `spot-post-ready`'s links
  in place of the done `proofs/spot-architecture-matches` (ASK 14).
- Texts: `spot-numbers-lean-checked` drops the caption and `spot-figures` takes it, with its holder (ASKs 2, 11);
  `spot-advice-certified` names vllm and what it waits on (ASK 5); `spot-two-stage-e2e` carries your two dated
  steps (ASK 8); `spot-128k-attention-proved` says "an ask to Daniel, posted Mon 11 PM".
- Two texts brought up to what the notes say now, not asked by you: `spot-overhead-measured` states the first
  timed run's result (SERVEDCARD, 02:08Z) in place of its booking, and `spot-fused-cut-served` says its serving
  change waits to merge (FUSEDSERVE, 02:18Z).
- From two issues filed at 02:48Z, which name these texts: PRF-495 item 3 ("goes into HOUSEKEEP's Goals script"):
  `spot-two-stage-e2e` now says that on main serving commits only the ids in and out, with the schedule's read and
  the history tables in review. PRF-494 (yours, Thu; "the three goals state the verdict a two-stage run gets ...
  and name the lock's open premises"): `spot-two-stage-e2e` states the refusal at the last step, and
  `spot-two-stage-sound` and `spot-post-ready` name the three premises open in the lock on `main`
  (`CompiledTwo.Scope`, `SpotFlock.Embeds`, `SpotFlock.FrameIs`) [read 03:20Z]. None of my texts keeps the page's
  "proved for the Lean verifier with `Evaluates` and `Reveals` open". The wording that follows the page to Daniel
  stays yours.
- `spot-two-stage-e2e` also carries vllm's step of 02:46Z: its Tue 16:00Z serve is the first to commit the schedule.
- Four things still open: rows 5 (G1's verdict, 04:20Z), 6 (MOEBIND2's date, yours), 7 (the holder, yours by
  05:00Z) and the card's id.

**Rules I held to.** A status changes only where a source from the last 24 h supports it: on track needs a named
holder who is running and a dated plan that ends by the date; off track names what it waits on. A date changes only
to one a source gives. Where the sources do not say, the cell is an ASK and the script leaves the field as it is.
"Live" means a running worker holds it this week; "parked" means nobody at proofs does.

**Sources**, by the short names the table uses:

- IDX:n: line n of the swarm index, `20261106T0639Z-report-spot-swarm.md` (resume queue "updated Tue 00:38Z"; the
  LANDING paragraph, line 91, "read 01:00Z Tue"; line 204 is the UNSTAFFED row of the Mon queue).
- GA:n: line n of `20261109T1855Z-report-goal-audit.md` (Mon 18:55Z). The page was rewritten after it, Mon 19:06Z to 20:50Z.
- HH:n: line n of `20261109T1936Z-report-headline-honest.md` (BREAKEVEN, Mon 19:36Z). SC: `20261110T0002Z-report-smell-check-post.md`.
- N0104, N0114, N0147, N0218: your notes to me in `inbox/HOUSEKEEP.md`, headed Tue 01:04Z, 01:14Z, 01:47Z and 02:18Z.
- CON12: a comment on CON-12, with its lead and time. An issue key with a name and a time is that issue's comment.
- [gh 01:26Z]: `gh pr view`, read by me then. Times are UTC unless marked PST. The texts for the page use Pacific
  times with no zone, as the site's rules ask.

## The table

| # | Goal | This week | Status: now, proposed | Date we now believe | The one thing it waits on | Worker | Source |
|---|---|---|---|---|---|---|---|
| 1 | `slice-clean` | parked | off-track (kept) | **Fri 13 Nov 6 PM PST** (from Tue 10 Nov 6 PM PST). CHECK: it is PRF-387's date, set when the post was Wed 11 Nov | nothing linked (its two links are done goals; dropped). Blocker text: no proofs worker, top holds it | none at proofs; top (the page's own text) | IDX:204 "slice-clean (move to top)"; GA:59 "owner says proofs, top does it ... move to top"; PRF-387 coordinator Sun 00:20Z "moved to Fri 13 Nov ... unstaffed until Thu"; #7850 open, untouched since Sat 7 Nov [gh 01:26Z] |
| 2 | `spot-bind-term` | live | off-track, **on-track** | **Fri 13 Nov** (from Wed 11 Nov 6 PM PST). The source gives the day; 6 PM PST is the project's hour | `ci/fine-grained-check` (main moving), after BINDKSAT's one PR, marked Thu 12 Nov 12:00Z | BINDKSAT; ZKSTACK (two witnesses) | PRF-466, coordinator Tue 00:33Z: "pieces elaborate by Tue 10 Nov 04:00Z; witnesses by Wed 11 Nov 12:00Z; lock entries by Wed 18:00Z; one PR marked Thu 12 Nov 12:00Z; on `main` Fri 13 Nov"; IDX:104, IDX:105 |
| 3 | `spot-checks-closed` | live | off-track, **on-track** | Wed 11 Nov 6 PM PST (unchanged) | nothing outside proofs: the coordinator's own ruling on items 5, 6 and 7, Wed 11 Nov 18:00Z | the coordinator (no worker: GA:43, IDX:204) | N0114: "PRF-461: Due 2026-11-11 (its goal's date) ... I settle 5, 6 and 7 by Wed 11 Nov 18:00Z"; GA:82-97 (six of eight settled; item 1's run a1ee) |
| 4 | `spot-numbers-lean-checked` | live (work done, waiting to merge) | on-track (kept) | Wed 11 Nov 6 PM PST (unchanged) | `ci/fine-grained-check`: #8346 (on `checked`) and #8360 (on `next`) reach `main` | FIG1LEAN (done, idle) | N0218 item 2 ("`write_form` is captioned, not stated in Lean, and the caption is a piece of `spot-figures` ... done when #8346 and #8360 are on `main`"); GA:53 "keep; Wed 11 Nov"; IDX:91 (#8360 on `next`); both PRs open [gh 01:26Z] |
| 5 | `spot-fused-cut-served` | live | off-track (kept) | Thu 12 Nov 12 PM PST (unchanged) | FUSEDSERVE's first Commit with the tables: G1 (now `r20261110-021137-b418`) Tue 03:20Z, verdict line by 04:20Z; the fit Wed 11 Nov 18:00Z. The done link `circuits/glm-production-compile` is dropped | FUSEDSERVE | IDX:103; FUSEDSERVE's checkpoint Tue 02:18Z (G1 relaunched with parallel staging; #8454 "marked at 283188b306 and admitted; it waits for a land verdict"; `/` has 119 GB free); N0218 item 3; HH:179 "the fit Wed 11 Nov 18:00Z; goal ... Thu 12 Nov 12 PM PST"; CON12 circuits 01:07Z |
| 6 | `spot-128k-attention-proved` | live | off-track (kept) | Thu 12 Nov 6 PM PST (fixed, N0104) | `circuits/glm-sparse-attention-zk` (CIR-210's `_v2` PR, marked Tue 20:00Z); also linked: `circuits/glm-long-dsa-row`, `vllm/served-128k-window-committed`, and the ask card once it exists (the script's `ASK_128K`) | TWOPROVE, DSAREAD, ACCOUNTD, PROVECUT | N0104 (both fixed points and fact 2); N0218 item 4 ("no card exists yet. ASKPAGE drafts the page by 04:30Z and I post page and card by 07:00Z"); CON12 circuits 01:07Z and vllm 01:11Z (the three links); IDX:95, 98, 100, 101; CIR-210 circuits Tue 00:21Z; SC section 1 |
| 7 | `spot-advice-certified` | live (vllm's) | off-track (kept) | Thu 12 Nov 6 PM PST (unchanged; N0218 item 5) | vllm's replay of a request's cache hit (VLLM-260's verdict, Tue 06:00Z). vllm has no goal on the page for it [page 02:27Z], so the blocker text says it and nothing is linked | vllm (VLLM-242) | N0218 item 5; VLLM-242 vllm Tue 00:29Z ("It waits on VLLM-260 (each sampled replay re-proves its request's hit), whose bit-identity verdict is due Tue 10 Nov 06:00Z. Next: restate this issue on that verdict, Tue 10 Nov 10 AM PST"); GA:51 |
| 8 | `spot-moe-binding` | live | off-track (kept) | Thu 12 Nov 6 PM PST (unchanged; N0114 confirms it for PRF-458) | MOEBIND2's fix of its failed records run (963a, Tue 00:51Z); no date. Then main moving | MOEBIND2 | N0114 "PRF-458: Due 2026-11-12 ... MOEBIND2 holds it"; MOEBIND2's checkpoint Tue 01:49Z (records run `963a` "FAILED the Security audit": `TableReadBinding_nontrivial` takes a closed Prop applied; two witnesses read the proofs tree; next: move the statements onto the definitions tree); N0218 item 6; IDX:112 |
| 9 | `spot-one-stage-premises` | live (proofs' part done) | on-track (kept) | Thu 12 Nov 6 PM PST (unchanged) | `ci/fine-grained-check`: #8411 (admitted, `1562e328da`) reaches `main`; lean's LEAN-235 line, marked by Tue 10 Nov 12 PM PST | TRUST6 (done); lean for LEAN-235 | LEAN-186: lean Mon 22:46Z, proofs 22:55Z, lean 23:03Z ("LEAN-235 ... marked by Tue 10 Nov 12 PM PST"; "provisional: a default, not yet a ruling"); IDX:91 (#8411 admitted); SC section 8 |
| 10 | `spot-overhead-measured` | live | off-track (kept) | Thu 12 Nov 6 PM PST (unchanged) | `circuits/glm-served-calls-fit-block` (the link circuits asked for). Also true, not linked: `circuits/glm-sparse-attention-zk` (the refused 10%) | SERVEDCARD; HASHCUT | SERVEDCARD's note, checkpoint Tue 02:08Z ("Window 1 ... proved 65 of 74 staged statements, all 34 types they cover, with every one of the 260 `serve --zk` sessions accepted by Flock's loopback verifier. There was no Lean verdict (NO_PROGRAM)") and 01:17Z ("The form-refused share is 9.8-10.8% of the card's cost"); HH:180; CON12 circuits 01:07Z |
| 11 | `spot-proof-server-steady-state` | parked until the fused cut is served | off-track (kept) | Thu 12 Nov 6 PM PST (unchanged; N0114 gives PRF-468 2026-11-12) | `proofs/spot-fused-cut-served`: the rerun is on served fused statements (N0218 item 7; before, I had circuits' units) | **ASK 7, open**: you name FUSEDSERVE or STATIONARY by 05:00Z | N0218 item 7 ("keep the goal (the post's 'provers per GPU' switch reads it). It waits on `proofs/spot-fused-cut-served`: the rerun is on served fused statements"); N0114 on PRF-468; PRF-468 circuits Mon 16:57Z |
| 12 | `spot-ten-percent` | live | off-track (kept) | Thu 12 Nov 6 PM PST (unchanged) | `proofs/spot-fused-cut-served` (no figure under 10% is served until it is). Links unchanged: its four open pieces and three done ones | BREAKEVEN (lead since Mon 19:15Z; idle until vllm's hashing rate, Tue 02:45Z); HASHCUT | HH:8 (BREAKEVEN "lead of proofs/spot-ten-percent from 19:15Z"), HH:41, HH:179-180; LEDGER's note line 104 (H7: 9.80%, 97.05%, 25.76%, each at 32 bits a token, modelled, world year); IDX:110; SC section 6 |
| 13 | `spot-two-stage-e2e` | live | off-track (kept) | Thu 12 Nov 6 PM PST (unchanged; N0218 item 8) | vllm's schedule commit before the coins (VLLM-250: its step passed on Mon's served window, and "Tuesday's 16:00Z serve runs the step itself"; no goal of vllm's on the page [page 02:27Z], asked by you on VLLM-250 at 02:19Z); the verifier's link read, marked by Wed 11 Nov 06:00Z. Blocker text says both | TWOPROVE, TWOVERIFY, TWOOPENS, ACCOUNTD; vllm-twostage | N0218 item 8; PRF-495 item 3 (its description: "Item 3 goes into HOUSEKEEP's Goals script"; `PORTS = (INGRESS, EGRESS)` at `boundary_rows.py:68` on `main` [read 03:12Z]; #8429 and #8434 open [gh 03:12Z]); PRF-494 (the goal states the verdict; `FORM : Bool := false`, `SecondStage.lean:115` on `main` [read 03:12Z]); VLLM-250: vllm-twostage Mon 23:19Z ("the advice step is built and passes on the endpoint fixture. The test on real data runs after 02:10Z"), proofs Tue 02:19Z (asks vllm for a goal id), vllm-twostage Tue 02:46Z ("the advice step passes on last night's served window ... Tuesday's 16:00Z serve runs the step itself"); the page's blocker (coordinator, Mon 20:43Z); LEDGER's note, checkpoints Mon 22:09Z and 23:24Z; IDX:98, 111 |
| 14 | `spot-verifier-cost` | live | off-track, **on-track** | Thu 12 Nov 6 PM PST (unchanged; N0218 item 9: PRF-464's Fri is one later PR) | `ci/fine-grained-check`: #8425 (marked, `8d40023923`) and #8453 (admitted, `0054c395fe`) reach `main`. **Kept beside it:** `circuits/verifier-templates-not-unrolled`, the link circuits asked for (N0147). N0218 says it waits "only" on the queue: delete that one `--wait` if you mean circuits' link goes | VERIFYFAST-B | N0218 item 9; VERIFYFAST-B's checkpoints Tue 02:03Z ("#8425 is marked at 8d40023923 and #8453 is admitted") and 00:55Z ("1,899.6 core-s flat and 257.7 templated"); CON12 circuits 01:07Z |
| 15 | `terminology-clean` | parked | on-track, **off-track** | **Fri 20 Nov 6 PM PST** (from Thu 12 Nov 6 PM PST; N0218 item 10) | the SPOT post being out, Mon 16 Nov (blocker text; its one link is a done goal, dropped). `top/repo-cleaned-up` (Fri 13 Nov) still links it; you tell top | none until the post is out (RENAME idle: GA:63) | N0218 item 10 ("after the post. Date Fri 20 Nov 6 PM PST, off-track, blocker text 'nobody until the post is out (Mon 16 Nov)'. Owner stays proofs"); GA:63 ("539 'sampled proofs', 247 'C-Flock' on main"); PRF-361 LINEARHYG Mon 12:32Z (#7153 and #8174 reached main Mon 07:33Z) |
| 16 | `spot-figures` | live from Wed 11 Nov | off-track (kept) | Fri 13 Nov 12 PM PST (unchanged; N0114: "closes with the figures (Fri 13 Nov)") | `proofs/spot-ten-percent`: its final card run, verdict Wed 11 Nov 18:00Z. `proofs/spot-draft-tonight` (no such goal) and the done `vllm/tick-tradeoff-measured` are dropped | BREAKEVEN (N0218 item 11) | N0218 item 11 ("BREAKEVEN renders them from the labelled rows and writes `write_form`'s caption, after the final card run (verdict Wed 11 Nov 18:00Z). Date unchanged"); CON12 top Mon 5:10 PM PST (the dead link); N0114 on PRF-456; GA:48 |
| 17 | `spot-claims-checked` | live | off-track, **on-track** | Fri 13 Nov 6 PM PST (unchanged) | nothing outside proofs: LEDGER's passes Tue 03:00Z, 06:00Z, 12:00Z, 20:00Z, then daily through Fri | LEDGER | IDX:107; IDX:193 ("RESUME DAILY through Fri | LEDGER (... goal spot-claims-checked)"); LEDGER's note `20261109T2010Z-report-claims-ledger` (checkpoint Mon 20:54Z: 19 secondary claims; H7 and H10 marked NOT) |
| 18 | `spot-output-accounting` | live | off-track (kept) | Fri 13 Nov 6 PM PST (unchanged) | `network/cap-certified` (NET-281: #8405's fix, then ACCOUNTD's #8462, target Wed 18:00Z) | ACCOUNTD; ZKSTACK (item 5, #8311) | ACCOUNTD's checkpoint Tue 00:12Z (#8462 "waits on the fix for #8405", "target Wed 18:00Z"; the F = 1 PR "Mark by Thu 12 Nov 12:00Z"); NET-281 network-accounting Mon 23:35Z; PRF-462 LINEARHYG Mon 00:36Z; GA:309-321; #7898, #7933, #7934, #7908 merged [gh 01:26Z] |
| 19 | `spot-post-ready` | live | off-track (kept) | Fri 13 Nov 6 PM PST (fixed, N0104) | its pieces: 14 open of proofs' (the new goal of row 25 among them), `circuits/glm-compile-cost-measured` (in place of the done `circuits/glm-128k-indexer`), 1 done (`proofs/spot-glm-replay`). 16 links, the cap; the done `proofs/spot-architecture-matches` leaves to make room | the coordinator | N0104 (fixed point; "every goal the post reads is a piece of it"); CON12 circuits 01:07Z; CON12 proofs Mon 14:19Z (tier 1, set, with its reason); GA:64 |
| 20 | `weight-security-bridge` | live | on-track (kept) | Fri 13 Nov 6 PM PST (unchanged) | `ci/fine-grained-check`: #8311 (`317bdbc1`, admitted) and #8400 (`9d9fe580`, marked; carries #8224's commits) reach `main` | ZKSTACK, CUTLEMMA | IDX:91, IDX:99 (#8400 marked at `9d9fe58033`; "No hold for #8224"), IDX:105; GA:58 ("LeaksThrough open to 17 Nov ... Fri 13 Nov; state the premises") |
| 21 | `zero-knowledge` | parked | off-track (kept) | **Mon 30 Nov** (from Fri 20 Nov; N0114) | the SPOT post being out, Mon 16 Nov: nobody is on it until then. Links unchanged (`proofs/zk-leaks-closed` and six done goals), so the page shows the links and hides the blocker text | none until the post is out | N0114 "PRF-366 ... Nobody is on it until the post is out (Mon 16 Nov) ... 30 Nov is the date I believe ... The goal `proofs/zero-knowledge` gets the same date"; N0104; GA:65 ("keep parked; rewrite") |
| 22 | `zk-leaks-closed` | parked | off-track (kept) | **Mon 30 Nov 6 PM PST** (from Fri 20 Nov; N0218 item 12: with its parent) | the SPOT post being out, Mon 16 Nov (blocker text; it has no links, so the page shows it) | none until the post is out | N0218 item 12; N0104 ("have nobody on them until the post is out: say so as their blocker"); GA:66; the page's own text for what is on main |
| 23 | `sampling-generic` | live | on-track (kept) | Mon 30 Nov 6 PM PST (unchanged, N0104) | its pieces (unchanged): `proofs/spot-two-stage-sound`, `proofs/spot-one-stage-premises`, `proofs/spot-moe-binding` | the two-stage Lean workers (see `spot-two-stage-sound`) | N0104 (the project for `spot-two-stage-sound`); CON12 proofs Mon 14:19Z and soundness Tue 01:00Z (its link to `proofs/sampled-audits-sound` is dropped, so the text no longer names it); GA:67 ("keep; rewrite") |
| 24 | `spot-two-stage-sound` | live | on-track (kept) | Mon 30 Nov 6 PM PST (unchanged) | nothing outside proofs. Next dated steps: TWOWITNESS's statement (A) Tue 18:00Z, witness (D) Fri 12:00Z; #8309's mark after records 7632 | TWOOPENS, SLOTSAT, TWOWITNESS | IDX:96 (#8309 at `70cb152cb5`, mark after records 7632), IDX:111 ("SLOTSAT: #8363 after #8309; Friday's non-vacuity ... TWOWITNESS: statement (A) frozen Tue 18:00Z, witness (D) Fri 12:00Z"); PRF-419 LINEARHYG Mon 12:32Z ("`FORM := false` refuses every two-stage run ... due 30 Nov"); GA:68 |
| 25 | `spot-two-stage-not-vacuous` | live | not on the page, **on-track** | **Fri 13 Nov 12 PM PST** (new; N0218 item 14) | nothing outside proofs. Dated steps: TWOWITNESS's statement (A) frozen Tue 18:00Z, witness (D) Fri 12:00Z | SLOTSAT with TWOWITNESS | N0218 item 14 ("add the goal for Friday's non-vacuity ... due Fri 13 Nov 12 PM PST, holder SLOTSAT with TWOWITNESS; it takes the place of one done piece in `spot-post-ready`'s links"); IDX:111; GA:68 |

## The ASKs

**Still open:**

- **ASK 3, open until 04:20Z:** the line keeps off-track with its blocker text until G1's verdict. I re-read FUSEDSERVE's note then and change this row, its texts and the line. (`spot-fused-cut-served`)
- **ASK 4, a gap you fill:** the script stops, before it writes anything, until `ASK_128K` holds the card's id: `ASK_128K=<id> bash proofs-goals-fix.sh` (or `ASK_128K=none` to run without the link). The text already says "an ask to Daniel, posted Mon 11 PM", so it is true only once the card is up. (`spot-128k-attention-proved`)
- **ASK 6, open, yours:** you write MOEBIND2's date here on its return. Until then the line keeps my text. (`spot-moe-binding`)
- **ASK 7, half answered:** keep the goal, linked to `proofs/spot-fused-cut-served`. Its last sentence says nobody holds the rerun until you name the holder (by 05:00Z); I change it then. (`spot-proof-server-steady-state`)
- **ASK 8, answered, one part open:** the date stays, and the text carries your two dated steps. If vllm puts its serving side on the page before you run, add `--wait goal:<its id>` to the line and delete `--blockers @blockers/spot-two-stage-e2e.md` in favour of `--no-blockers`; I look again before 08:00Z. (`spot-two-stage-e2e`)

**Answered in your note headed 02:18Z, and applied:**

- **ASK 1, answered:** leave the owner; you ask top in your CON-12 reply and add `--owner top` on a yes. No change to the line. (`slice-clean`)
- **ASK 2, answered:** `write_form` is captioned, not stated in Lean, and the caption belongs to `spot-figures`. This goal's text no longer names it. (`spot-numbers-lean-checked`)
- **ASK 5, answered:** vllm holds it under VLLM-242; the date stays; the blocker text is yours, in plain words. vllm has no goal on the page for the replay of a cache hit, so no link. (`spot-advice-certified`)
- **ASK 9, answered:** Thu stays; on-track, linked to `ci/fine-grained-check` (my rule allows it: VERIFYFAST-B runs, both changes are in the queue since Tue 02:03Z). I kept circuits' link beside it; see the row. (`spot-verifier-cost`)
- **ASK 10, answered:** after the post: Fri 20 Nov 6 PM PST, off-track, "nobody until the post is out", owner unchanged. (`terminology-clean`)
- **ASK 11, answered:** BREAKEVEN renders them and writes the caption after the final card run. I left the status off-track, because it waits on `spot-ten-percent`, which is off track; say if a holder and a dated plan make it on-track for you. (`spot-figures`)
- **ASK 12, answered:** Mon 30 Nov, with its parent. (`zk-leaks-closed`)
- **ASK 13, answered and done:** PRF-363 is Done since Tue 02:22Z, with your comment's facts checked on `gh` (five PRs merged, two closed unmerged, #7228 open). No change to the line. (`sampling-generic`)
- **ASK 14, answered:** the goal for Friday's non-vacuity is row 25 and the script's first line. It takes the place of the done `proofs/spot-architecture-matches` in `spot-post-ready`'s links, which stay at 16. (`spot-two-stage-sound`)
- **ASK 15, yours for Wed (PRF-384, PRF-454, PRF-372).** None of the three sits under one of these goals, so none
  has a row: PRF-384 and PRF-372 carry labels of goals other leads own (`proofs/soundness-nothing-open`, soundness;
  `demos/onsite-demo`), and PRF-454's goal, `proofs/spot-architecture-matches`, is done. PRF-384: the 7 Nov plan's
  Mon and Tue pieces show no commit. PRF-454: no worker or date for the layering moves; since Tue 02:22Z it also
  carries lean's PRF-452. PRF-372: its 20 Nov is before the onsite goal's 24 Nov.

## top's four lines, and where each is fixed

1. *The page says what we are working on now.* Every row's text is rewritten to today (two to four sentences: what
   will be true, where it stands with its date, who holds it). Nothing is marked done or removed: no source says
   one of the 24 is done or dropped. One goal is added, for Friday's non-vacuity (row 25).
2. *Every goal sits under a project.* After the script: `spot-post-ready` (tier 1, set) links 14 open pieces of
   proofs', one of circuits' and 1 done; `spot-fused-cut-served`, `spot-overhead-measured`,
   `spot-proof-server-steady-state` and `spot-verifier-cost` sit under its piece `spot-ten-percent`;
   `sampling-generic` (tier 1, set) links `spot-two-stage-sound` and two pieces it shares with the post;
   `zero-knowledge` (tier 1, derived: nothing waits on it) links `zk-leaks-closed`; `slice-clean` and
   `terminology-clean` sit under top's `top/repo-cleaned-up`. No goal of the 25 is tier 1 for want of a link.
3. *Off track means a date or a blocker.* 15 stay or become off track: 8 link the goal they wait on, and 7 carry a
   blocker text because what they wait on is not a goal on the page (no worker, another lead's work with no goal,
   or proofs' own work).
4. *Our lines.* `spot-figures` no longer links `proofs/spot-draft-tonight`. The four not written in 48 h
   (`terminology-clean`, `zero-knowledge`, `zk-leaks-closed`, `spot-two-stage-sound`) are rewritten. Titles:

- `spot-checks-closed`: "Guarantee checks closed or scoped (PRF-461)" (6 words) becomes "Guarantee checks closed or scoped" (5).
- `spot-overhead-measured`: "SPOT's proof overhead measured on the headline model's blocks" (9 words) becomes "Proof overhead measured on GLM-5.3's served blocks" (7).
- `spot-verifier-cost`: "Verifier cost in the cost table, driven to O(description)" (9 words) becomes "Verifier cost tracked and driven down" (6).
- `terminology-clean`: "Retired terms gone from the tree, and new uses refused" (10 words) becomes "Retired terms gone, and new uses refused" (7).
- `spot-claims-checked`: "Every claim in the SPOT post traced to evidence" (9 words) becomes "Every claim in the post traced to evidence" (8).

   The first was not on top's list: its title carried a ticket id. No priority is set.

## What the script does that the table does not show

- **The gap for ASK 4.** The 128k goal's line carries `--wait ask:$ASK_128K`. The script stops with exit 2, before
  any write, when `ASK_128K` is empty; `ASK_128K=none` drops the link; under `DRY=1` it prints `ask:THE-ASK-CARD-ID`.
- **The new goal goes first**, so that `spot-post-ready`'s line, which links it, finds it on the page.
- **Links other leads asked for (CON12, Tue 01:07Z and 01:11Z; N0147), all seven in the script:**
  `circuits/glm-long-dsa-row`, `circuits/glm-sparse-attention-zk` and `vllm/served-128k-window-committed` under
  `spot-128k-attention-proved`; `circuits/glm-served-calls-fit-block` under `spot-overhead-measured`;
  `circuits/verifier-templates-not-unrolled` under `spot-verifier-cost`; `circuits/glm-compile-cost-measured`
  under `spot-post-ready` in place of the done `circuits/glm-128k-indexer`; `spot-fused-cut-served` drops the done
  `circuits/glm-production-compile`. No goal passes 16 links: `spot-post-ready` is at 16 exactly.
  **Every goal id the script names resolves** on the page as read at Tue 10 Nov 02:27Z, the new goal aside [run]:

  - `ci/fine-grained-check`: ci, on-track, Tue 10 Nov 6 PM PST, "Main moves on the verified queue's check"
  - `circuits/glm-sparse-attention-zk`: circuits, on-track, Wed 11 Nov 10 AM PST, "GLM-5.3's sparse attention accepted under zero knowledge"
  - `circuits/glm-long-dsa-row`: circuits, on-track, Wed 11 Nov 10 AM PST, "GLM-5.3's long DSA rows in provable units"
  - `vllm/served-128k-window-committed`: vllm, on-track, Thu 12 Nov 8 AM PST, "A served 128k GLM-5.3 window committed for audit"
  - `circuits/glm-served-calls-fit-block`: circuits, on-track, Wed 11 Nov 10 AM PST, "GLM-5.3's served calls each fit a Flock block"
  - `circuits/verifier-templates-not-unrolled`: circuits, on-track, Wed 11 Nov 10 AM PST, "Templates the verifier reads without unrolling"
  - `network/cap-certified`: network-accounting, on-track, Tue 10 Nov 6 PM PST, "The response cap is certified and published"
  - `circuits/glm-compile-cost-measured`: circuits, on-track, Fri 13 Nov 12 PM PST, "GLM-5.3's compile cost measured for the post"
  - `proofs/spot-glm-replay`: proofs, done, Mon 09 Nov 6 AM PST, "The headline model replays deterministically"
  - and 15 open goals of proofs' own, each among the 24; `proofs/spot-two-stage-not-vacuous` is the one the script makes.

- **Waiting on `main`:** the five on-track goals whose work is written and waits to merge link
  `ci/fine-grained-check` ("Main moves on the verified queue's check", ci, due Tue 10 Nov 6 PM PST, on track), the
  narrowest goal for it and due before each. If CI-496's answer (Tue 18:00Z) says the queue cannot carry them by
  their dates, those five go off track with the same link.
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

- It sets no priority and no tier, moves no owner, marks nothing done and removes nothing.
- It adds one goal and no other. The skill's rule that an open goal due within a week waits on a tier-3 step due
  within a day is not applied: that would be about 20 new goals, and top's list does not ask for it.
- It does not touch the goals with a `proofs/` id that other leads own (`proofs/sampled-audits-sound`,
  `proofs/soundness-nothing-open`, `proofs/proof-check-fast`).

## Checks I ran, and what I could not check

- **The site's check, ported.** The site checks a goal's whole text on every write (`parseGoal`,
  `website/apps/docs/lib/goals/format.ts` and `plain.ts` in compute-verification/infra, read at 01:24Z): the four
  words Daniel doesn't know ("served window" among them), a Greek letter or a superscript outside math, a word with
  an underscore outside code, a list with no lead-in line, an off-track goal with neither a link nor a blocker, more
  than 16 links. A Python port of those rules passes all 25 goals as the script leaves them [run].
- **The skill's rules the site does not check:** no PR number, issue key, run id, agent id, branch, file path,
  worker name, time zone or pipeline word ("marked", "landed", "admitted") in any text; titles of at most eight
  words; two to four sentences a description. All 25 pass [run].
- **The commands.** Each of the script's 25 lines went through `research goals`' own argument parser and edit
  (`research.goals.main`) against a stand-in route that answers from the saved page and sends nothing: 25
  writes built, the new goal's among them, none refused by the command [run]. So a flag that cannot be combined, or
  a missing file, would have shown. The guard was tried once without `ASK_128K` and without `DRY`: exit 2, and the
  page's 24 goals read the same before and after [run].
- **Not checked:** a real write (I ran no `research goals set`); that the deployed site runs the `plain.ts` I read,
  and what it asks of an `ask:` id; how the page draws 16 links; whether top agrees to own `slice-clean`; every
  figure in a text (the 14,903 steps, 97.05%, 9.80%, 25.76%, 1,899.6 and 257.7 core-seconds, 539 and 247 uses,
  22.7 GB and 69.9 MB, 65 of 74 and 260 sessions): each is quoted from the note or the page text the Source column
  names, and I opened none of the runs.
