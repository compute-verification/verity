# Network accounting's open PRs and cleanup slice (TOP-223)

Read-only triage, Sun 8 Nov ~01:45Z, for top's cleanup (Sun 6 PM PST). Nothing was closed, marked or pushed.

- **Refs:** `origin/main` 0717ef602b04; `next` ff744ace042 (top's census); push 1b `cursor/onsite-push1b-061a` 9068336929e0
  (Monday's code, `env.sh`'s `REF`); the follow-on's pin `cursor/onsite-thu-pin-061a` 7ecf3f827b8 (`follow_launch.sh`'s
  `SERVE_REF`); push 2 `cursor/onsite-push2-061a` 4b8e1576e6e1; the old `next` ea3f9c5635.
- **Ours:** `research`'s `LANE_RE` (`tools/research/src/research/queue.py`) on the body, `Lane network-accounting`, else our
  branch names (`cursor/*-061a`, `cursor/net*`, `cursor/o6*`, `cursor/onsite*`, `cursor/od*`). That gives 83, as top's
  census does.
- **Carried:** a PR's head is an ancestor of push 1b, or its own commits (`log head ^main ^ea3f9c5635`) are in push 1b, push 2 or
  #7390 by subject. **Merges cleanly:** `git merge-tree --write-tree origin/main head`. A PR carrying the old `next`
  can't merge as it is: it carries thousands of commits main lacks, so it restacks whatever merge-tree says.
- **Live:** each issue's Linear status, read Sun 01:30Z. NET-85, NET-92, NET-145 and NET-163 are Canceled. NET-169, 170,
  175, 189, 195, 207, 208, 219 and 238 are In Progress or In Review.

## Counts

| Class | PRs |
| -- | -- |
| CLOSE | 73 |
| KEEP-MAIN | 10, which folds to 7 |
| KEEP-ONSITE | 0 |
| Total | 83 (36 carry the old `next`: CI-440's 34, plus the branch PRs #7301 and #7410) |

CLOSE, by reason:
- 52: carried by push 1b. Their heads are in it, and they land with the onsite merge (O6, Fri 20 Nov). 24 are from Oct 28–31
  without the old `next`; 28 are CI-440's.
- 2: in push 2, which goes to main after Monday (#7255, #7259).
- 2: folded into proofs' open #7390 (#7041 and #7130).
- 14: idle, their issue Canceled. NET-85: #4873, #6373, #6374, #6381, #6382, #6466. NET-92: #6367, #6964. NET-145:
  #6972–#6975. NET-163: #6944, #6947.
- 3 others:
  - #6362: idle since 2 Nov, and no open issue names it.
  - #6678: superseded. It retires the certifier's PROTOCOL.md, which proofs' #7898 now extends.
  - #7167: push 1b carries all but one line, a "warden" in `network_e2e/encodings/v1.py`. The retired-term pass takes
    that line.

**KEEP-ONSITE is 0.** Monday reads push 1b and the follow-on reads the Thursday pin. Neither is an open PR's head. No open
PR has a commit that push 1b or the pin needs, and #7410 is the only open PR based on an onsite branch.

**The KEEP-MAIN folds**, taking 10 PRs to 7:
- #7172 into #7165 (both NET-189, docs, and #7172 already lands after #7165).
- #7301 into #7326 (both NET-208, fixes 3 and 1).
- #7262 and #7410 into #7240. All three are onsite follow-ups that restack onto main after the onsite merge: NET-195's
  record, NET-219's test and NET-207's wires.

The other 4 stand alone:
- #7891 (NET-238) merges cleanly now.
- #7584 (LEAN-197) conflicts in one Lean file.
- #7109 (NET-169) and #7165 (NET-189) conflict, and restack on main.

**When closing:**
- Keep every branch. NET-236's restack (Tue 10 Nov) and the onsite merge read them.
- Closing the 52 forgoes the "merged" mark GitHub would give them when push 1b merges.
- #7130: the standing ruling was to close it when #7390 lands. Its uncarried commits are restack reruns and lock
  regenerations, plus one SessionLoop commit whose lemmas (`refK_of_none`, `schedOf_eq_none_iff`,
  `liveRoundEqOwn_ofLoopLink`) #7390 has, so closing it now loses nothing.

## Table

KEEP-MAIN first, then CLOSE: the 52 carried by push 1b, then the other 21.

| PR | Title | Head date | Base | Old `next` | Linear | Class | Reason |
| -- | -- | -- | -- | -- | -- | -- | -- |
| #7109 | NET-169: lock B <= b (UnrelLeBudget, UnreleasedLeBudget, StreamStopBudget) and … | 2026-11-04 | main | no | NET-169, PRF-310 | KEEP-MAIN | NET-169 (In Progress, due 11-13): 65 of its 66 commits are in neither main, push 1b nor push 2, and only 8 are in #7390; conflicts (Security lock, the ZK server's Record.lean and tests, `flock/live/PROTOCOL.md`), so restack on main |
| #7165 | Network docs: the bound in the Glossary's symbols, as PeriodLeakAccepted proves… | 2026-11-05 | main | no | NET-189, NET-42 | KEEP-MAIN | NET-189 (In Review, due 11-13): conflicts in docs only (ARCHITECTURE.md, README.md, the certifier's PROTOCOL.md and README.md), so restack on main |
| #7172 | Network accounting: the sync, jitter, wiring and map letters off the table's, a… | 2026-11-04 | main | no | NET-189 | KEEP-MAIN | NET-189: conflicts (ARCHITECTURE.md, README.md, `network_e2e/encodings`, `certifier_piece.py`); fold into #7165, the same issue, which it already follows |
| #7240 | onsite: each window-circuit wire inside one writer and one reader entry, and th… | 2026-11-04 | main | yes | CIR-157, NET-195, NET-202, NET-205, NET-207, NET-208, VLLM-146 | KEEP-MAIN | NET-207 (In Progress, due 11-13): 16 commits in neither main, push 1b nor push 2 (`window_write`, the wire-order tests; its tip 49b01aec881 never reached onsite); carries the old `next`, and builds on onsite's tree, so restack onto main after the onsite merge |
| #7262 | Onsite record: the period check's circuit files listed, public, by the SHA-512 … | 2026-11-04 | main | yes | NET-195 | KEEP-MAIN | NET-195 (In Progress): one commit in neither push 1b, push 2 nor the follow-on pin, 5a107b61078 (`onsite_record.py` lists a circuit file only where OUT holds it, through no link); `onsite_record.py` reaches main only with the onsite merge, so restack after it; fold into #7240 |
| #7301 | NET-208 fix 3: WrongBounds at the Bernoulli draw's published bound | 2026-11-04 | `cursor/windows-placed-061a` | yes | NET-208 | KEEP-MAIN | NET-208 fix 3: 3 commits in neither push 1b nor the follow-on pin; based on `windows-placed-061a`; fold into #7326, the same issue, and retarget to main |
| #7326 | NET-208 fix 1: WrongBounds at the parsed statement's design, at the code's deri… | 2026-11-04 | main | yes | NET-208, PRF-386 | KEEP-MAIN | NET-208 fix 1 (In Progress, due 11-10): 24 commits in neither main, push 1b nor push 2 (SameLaw, RetWalk, ParamsLaw and their locks); carries the old `next`, so restack (CI-440's restack, Tue 10 Nov) |
| #7410 | onsite: a5.window_record's refusals run in the suite, on a local challenger's e… | 2026-11-05 | `cursor/onsite-demo-061a` | yes | NET-219 | KEEP-MAIN | NET-219: one test commit, 0433096b4dc, in neither push 1b nor the follow-on pin; its base, onsite-demo, is superseded by push 1b, so retarget to main after the onsite merge; fold into #7240 |
| #7584 | Nci: honest and nontrivial witnesses for the ten ThreadedThrottled and Containe… | 2026-11-06 | main | no |  | KEEP-MAIN | LEAN-197 (its branch name), approved: conflicts in one file (`Security/Proofs/Nci/Witnesses.lean`), so restack on main |
| #7891 | corpus: serve the warden traces under --admit tick behind the network certifier | 2026-11-07 | main | no | NET-238 | KEEP-MAIN | NET-238 (In Progress): merges cleanly onto main 0717ef602b04; needs its suites, a ready mark and a queue link from a VM with credentials |
| #5781 | two demos: one HonestClock over both PoUW games | 2026-10-28 | main | no |  | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #5870 | two demos: CertifierDevice: closeWindow is closeSamplingBlock | 2026-10-28 | main | no | NET-120 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #5936 | NetTiming: the fixed advice budget's count, EncardAccSeqsLeAdviceBits | 2026-10-28 | main | no |  | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #5937 | Properties.CertifierDevice: CompleteRows over the handed buckets, the FIFO's re… | 2026-10-28 | main | no |  | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #5970 | two demos: known: Rate.Session reads the canonical play (Verity.Play) | 2026-10-28 | main | no |  | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #6035 | two demos: WindowCharge.Window becomes WindowCharge.Advice, in Security and the… | 2026-10-28 | main | no |  | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #6036 | two demos: CertifierDevice.Link becomes CertifierDevice.Arrivals | 2026-10-28 | main | no |  | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #6045 | NetTiming: the constant-rate Program over send ticks, with \|a\| bits per execu… | 2026-10-29 | main | no |  | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #6066 | NetTiming: \|a\| bits per execution block from any execution block's start | 2026-10-29 | main | no |  | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #6069 | verity/core/protocols/network: a README, the outline and API of network certifi… | 2026-10-28 | main | no | NET-127, NET-138 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #6158 | verity/core/devices/certifier: a README, the outline and API of the network cer… | 2026-10-28 | main | no | NET-128, NET-131 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #6166 | two demos: CertifierDevice.Certificate is NetworkCheck.Check.Certificate, and C… | 2026-10-29 | main | no |  | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #6269 | wipe_gate.sh: keep the gate's start reply in KEEP/RUN, never published | 2026-10-28 | main | no | NET-117 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #6383 | NET-92: the period check reads SelectEntry_v1 slot entries a lane a chunk, and … | 2026-10-30 | main | no | ASK-182, NET-92, TOP-137 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #6398 | NET-7: the deployed warden always paces, a lost link can't stall the pacer, and… | 2026-10-29 | main | no | NET-7, NET-92 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #6439 | certifier private blocks (NET-121), PR 2: the transfer heads narrow to the load | 2026-10-28 | main | no | NET-121 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #6474 | two demos: the network period check's refusals and their mirror name NetTiming.… | 2026-10-30 | main | no |  | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #6518 | NET-88: honest and nontrivial witnesses for the network certifier's counts and … | 2026-10-30 | main | no | NET-88, NET-92 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #6538 | NetTiming: PooledWrongUnitsLe becomes PooledIncorrectBlocksLe, incorrect proof … | 2026-10-29 | main | no | TOP-137 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #6559 | NET-92: the network certifier's period reads V3c's audit (scan_zk --v3c, rows_c… | 2026-10-31 | main | no | INF-348, NET-92, VLLM-166 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #6574 | friction: standing rulings: demos run fast; network accounting is consistency p… | 2026-10-29 | main | no |  | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #6576 | NET-92: the period check accepts a tied audit at any sample (the consistency ch… | 2026-10-31 | main | no | NET-92 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #6592 | live_challenger: the gate's scan rehearsal takes --scan-class (w63k4 at 5 egres… | 2026-10-30 | main | no | INF-348 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #6615 | NetworkCertifier: a later window's sync never changes an earlier window's rows,… | 2026-10-29 | main | no | ASK-190 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7154 | Network proof cost per execution phase: N2 at the certifier's grid, CompileSlic… | 2026-11-03 | main | yes | NET-188, PRF-310 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7155 | served_verdicts.sh: a remote run serves through the network certifier's front | 2026-11-03 | main | yes | NET-190 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7156 | Onsite demo's record of a served run: every verdict, the bound term by term and… | 2026-11-04 | main | yes | BENCH-37, NET-195, NET-197, NET-203, NET-204, NET-205, NET-206, NET-207, NET-208, NET-209, NET-212, PRF-195 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7171 | NET-195: onsite's verdict is Lean's AND of remote's and the network certifier's… | 2026-11-04 | main | yes | NET-195, NET-197, NET-200, NET-202, NET-204, NET-205, NET-207, NET-208, NET-209, PRF-372 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7176 | NET-194: the verifier computes nothing from the advice in the clear; a received… | 2026-11-04 | main | yes | NET-191, NET-192, NET-194, NET-203 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7177 | network_sim/onsite.py: the onsite network certifier path at each mock-datacente… | 2026-11-04 | main | yes | NET-198 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7178 | Network audit's advice: B declared as each day's one program before the first w… | 2026-11-04 | main | yes | NET-191, NET-194, NET-197 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7181 | NET-192: the egress tie as a Spec of the proof system (scan_service.EgressTie),… | 2026-11-04 | main | yes | NET-192, NET-194, NET-195, PRF-359 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7182 | NET-196: refuse or charge every attack on the served path (attacks.py: the tabl… | 2026-11-04 | main | yes | NET-196 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7196 | served_verdicts.sh: a remote-network run takes the split certifier, whose one c… | 2026-11-04 | main | yes | NET-194, NET-195, PRF-372 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7206 | network_sim/onsite.py: ℓ_a,j is the period check's scans[j].free, and every K i… | 2026-11-04 | main | yes | NET-198, NET-202 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7208 | NET-203: the verifier's window circuit is the class's, from what the run publis… | 2026-11-04 | main | yes | NET-203 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7215 | NET-203 M3: the scan's registration states each state port's leaf reads (stateT… | 2026-11-04 | main | yes | NET-203, NET-208 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7218 | onsite: the served network certifier scan proves at 2^-64, and the one-stage an… | 2026-11-04 | main | yes | NET-190, NET-206 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7219 | NET-203 lows: records_audit period refuses an advice binding (L1); sampled proo… | 2026-11-04 | main | yes | NET-203 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7227 | NET-204: PeriodLeakRegistered, the network certifier's period leak bound at the… | 2026-11-04 | main | yes | NET-204, NET-205, NET-211 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7235 | network: NetworkCheck.Period holds each phase's tied compute audit to the regis… | 2026-11-04 | main | yes | NET-202 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7236 | network: the period reads a tied audit's draw at its own delta_i = 2^-64, and e… | 2026-11-04 | main | yes | NET-195, NET-202, NET-206 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7252 | Onsite record: each guarantee the tight figure rests on names what its statemen… | 2026-11-04 | main | yes | NET-197, NET-205, NET-208 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7257 | served_verdicts.sh follow-on: the H1_TIE deferral's follow-on runs the split ne… | 2026-11-04 | main | yes | NET-212 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7270 | NET-204 follow-up: PeriodLeakRegistered reads its conditions from PeriodConditi… | 2026-11-04 | main | yes | NET-204 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7275 | NET-205: the sampled audit's guarantees over wired rows | 2026-11-04 | main | yes | NET-195, NET-205, NET-208, NET-209, NET-210, NET-211, NET-212 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7282 | NetTiming: the rows format's read-back from (r, D) alone (readFree_record) and … | 2026-11-04 | main | yes | NET-200, NET-204, NET-208 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7290 | onsite: each run's S and MAX_TOKENS label its record, and --beside sets Thursda… | 2026-11-04 | main | yes | NET-197 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7298 | onsite: schedules-covered is a property of the Lean audit over the network cert… | 2026-11-04 | main | yes | NET-197 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7306 | NET-200: PeriodLeakRegistered as a property of the period check's acceptance: a… | 2026-11-05 | main | yes | NET-200, NET-205, NET-209 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7311 | onsite record: sampled proofs' coins are the named condition live-coins until a… | 2026-11-05 | main | yes | NET-197, NET-209, NET-220, PRF-386 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #7330 | NET-209 item 12: the network statements take Claims.Sampling.KSBy at the wired … | 2026-11-04 | main | yes | NET-209, NET-212, PRF-384 | CLOSE | carried by push 1b (`cursor/onsite-push1b-061a` 9068336929e0 holds its head); lands with the onsite merge |
| #4873 | two demos: certifier/v0: the period check runs one-stage v2 on the proof of eac… | 2026-10-30 | main | no | CI-200, NET-75, NET-85, NET-89 | CLOSE | idle since 10-30; NET-85 Canceled, so no live goal; the base of #6373, #6374 and #6381, all closing |
| #6362 | two demos: known: Draw.Session and Sketch.Session read the canonical play (Veri… | 2026-11-02 | main | no | ASK-181 | CLOSE | idle since 11-02; the two-demos Draw/Sketch sessions (ASK-181), in neither main nor push 1b, and no open issue names them |
| #6367 | network chain: every drawing round takes its coins from a challenger session of… | 2026-11-02 | main | no | INF-348, NET-92 | CLOSE | idle since 11-02; NET-92 Canceled, so no live goal; the base of the NET-145 stack, all closing |
| #6373 | two demos: Properties.NetworkCertifier.PeriodAcceptMeans: an accepted network p… | 2026-11-02 | main | no | NET-85 | CLOSE | idle since 11-02; NET-85 Canceled, so no live goal |
| #6374 | two demos: window_le at the compiled verdict, the network leg's half (supersede… | 2026-11-02 | main | no | NET-85 | CLOSE | idle since 11-02; NET-85 Canceled, so no live goal |
| #6381 | two demos: window_le's record legs at the period's certificates, off the LMS or… | 2026-11-02 | main | no | NET-85 | CLOSE | idle since 11-02; NET-85 Canceled, so no live goal |
| #6382 | NET-85: RealizesCommOfPeriodLeakAccepted ties K to PeriodLeakAccepted's count, … | 2026-11-02 | main | no | ASK-187, NET-85, PRF-301 | CLOSE | idle since 11-02; NET-85 Canceled, so no live goal; draft; RealizesComm… is on main already |
| #6466 | Network certificates sign their role: record, clock, cells, transfer or out | 2026-11-02 | main | no | NET-85 | CLOSE | idle since 11-02; NET-85 Canceled, so no live goal |
| #6678 | Certifier: retire PROTOCOL.md and protocol/, with the README's links fixed (the… | 2026-11-02 | main | no | NET-115, NET-121, NET-129, NET-131, NET-132, NET-134, NET-135, NET-150, NET-151, NET-152, NET-153, NET-154, NET-155, NET-156, NET-157, NET-84, NET-90, NET-92, NET-98, TOP-122 | CLOSE | idle since 11-02; superseded: it retires the certifier's PROTOCOL.md, which proofs' #7898 (PRF-462) now extends |
| #6944 | a5 passes R7 a window's own registered values: their words, row format, 16-bit … | 2026-11-02 | main | no | NET-163, PRF-331, VLLM-125 | CLOSE | idle since 11-02; NET-163 Canceled, so no live goal |
| #6947 | scan_zk --v3c reads V3c's SCAN_STATE from its v2 registration's registered entr… | 2026-11-02 | main | no | NET-163, NET-164 | CLOSE | idle since 11-02; NET-163 Canceled, so no live goal |
| #6964 | Network period check: slot-entry pins and each audit's free inputs from its reg… | 2026-11-02 | main | no | NET-147, NET-2, NET-92 | CLOSE | idle since 11-02; NET-92 Canceled, so no live goal |
| #6972 | Network certifier: the PoUW lead's coin source, egress counted per forward | 2026-11-02 | main | no | NET-145 | CLOSE | idle since 11-02; NET-145 Canceled, so no live goal; stacked on #6367 |
| #6973 | Network certifier: every boundary link's proxy counts against the open forward;… | 2026-11-02 | main | no | NET-145, NET-158 | CLOSE | idle since 11-02; NET-145 Canceled, so no live goal; stacked on #6367 |
| #6974 | Certifier program: the calls' per-forward count in Lean, held to calls.Calls by… | 2026-11-02 | main | no | NET-145 | CLOSE | idle since 11-02; NET-145 Canceled, so no live goal; stacked on #6367 |
| #6975 | Certifier program: RefusesPastCallBits, proved of the calls' per-forward count,… | 2026-11-02 | main | no | NET-145 | CLOSE | idle since 11-02; NET-145 Canceled, so no live goal; stacked on #6367 |
| #7041 | SessionLoop.Phase: the algebraic phase's worker reads the answers after Hello's… | 2026-11-03 | main | no | NET-175, PRF-310 | CLOSE | folded into proofs' #7390 (open): its head is an ancestor of #7390's |
| #7130 | SessionLoop: LiveRoundEq at the worker's own histories, at every algebraic-phas… | 2026-11-07 | main | no | NET-170, NET-171, NET-172, NET-173, NET-175, PRF-310 | CLOSE | folded into proofs' #7390 (open): 258 of its 269 commits by subject; the other 11 are its own restack reruns and lock regenerations (standing ruling: close #7130 when #7390 lands; the branch stays) |
| #7167 | NET-193: the scan holds the network certifier path's egress counts at any servi… | 2026-11-04 | main | yes | NET-189, NET-190, NET-192, NET-193, NET-195, NET-197, NET-200 | CLOSE | push 1b carries all but 17513d9e19c, one line of `network_e2e/encodings/v1.py` that says 'warden' (main still does): the retired-term pass takes it on main |
| #7255 | NET-208: a tied audit's scan-state reads are held only where they are the scan'… | 2026-11-04 | main | yes | NET-206, NET-208 | CLOSE | every commit is in push 2 (`cursor/onsite-push2-061a` 4b8e1576e6e1), which goes to main after Monday |
| #7259 | NET-195: V3c's period hands on each template's C-Flock circuit file by its pin … | 2026-11-04 | main | yes | NET-195 | CLOSE | every commit is in push 2 (`cursor/onsite-push2-061a` 4b8e1576e6e1), which goes to main after Monday |

## Lane text in our slice: 17 allowlist entries, 7 sites

Top's 17 are the entries of `tests/lint/allowlists/repo_lane_text.json` (at `next`) whose path is network accounting's:
`network`, `certifier`, `NetTiming`, `one_stage`, `onsite`, `Network`, `Accounting/` and `remote_bandwidth`. The same
path filter reproduces top's 226 below exactly. The lint's own `sites()`, run over each file at `next` and at push 1b,
finds 7 sites in 5 files. The other 12 entries find none, and `python3 tests/lint/test_repo_lane_text.py --prune` lowers
them.

| File | Allowed | Sites at `next` | At push 1b | Push 1b changes it |
| -- | -- | -- | -- | -- |
| `verity/core/protocols/certifier/README.md` | 3 | 3: line 340 "(decided 24 Oct, option A)", 348 "8 Oct", 381 NET-42 | 3 | yes |
| `verity/core/protocols/remote_bandwidth/README.md` | 1 | 1: line 151, MEM-72 | 1 | yes |
| `Security/Definitions/Accounting/Guarantees/RemoteBinding.lean` | 1 | 1: line 59, a `note:` id | 1 | yes |
| `Security/Definitions/Accounting/Remote/Draw.lean` | 1 | 1: line 74, "27 Oct" | 1 | yes |
| `Security/Definitions/Accounting/Wipe.lean` | 1 | 1: line 22, "Daniel" as a source | 1 | |
| `Security/Definitions/Accounting/Guarantees/RateAudit.lean` | 1 | 0 | 1 (a lane codename) | yes |
| `Security/Definitions/Accounting/Bridge.lean` | 1 | 0 | 1 | |
| `Security/Definitions/Accounting/Device.lean` | 1 | 0 | 1 | |
| `Security/Definitions/Accounting/Joint.lean` | 1 | 0 | 1 | |
| `Security/Definitions/Accounting/Guarantees/JointLive.lean` | 3 | 0 | 0 | |
| `Security/Definitions/Accounting/JointLive.lean` | 2 | 0 | 0 | |
| `Security/Proofs/NetworkCertifier.lean` | 1 | 0 | 0 | yes |
| `verity/core/protocols/network/nci/lean/NciCheck.lean` | 1 | 0 | 0 | yes |
| `verity/core/protocols/network/nci/lean/NciCheck/Baseline.lean` | 2 | 0 | 0 | yes |
| `verity/core/protocols/network/nci/lean/NciCheck/Desc.lean` | 1 | 0 | 0 | yes |
| `verity/core/protocols/network/nci/lean/NciCheck/DescLiveness.lean` | 1 | 0 | 0 | yes |
| `verity/core/protocols/onsite/CoreOnsite.lean` | 3 | 0 | 0 | yes |

The onsite merge:
- Bridge, Device and Joint: push 1b doesn't change them, so main's clean copies win with no conflict.
- RateAudit.lean: push 1b changes it and still holds the site. Pruning its entry on main is right, but the onsite merge must
  keep main's clean line there, or the lint fails after the merge.
- The 4 files with sites that push 1b changes (the certifier README, the remote_bandwidth README, RemoteBinding.lean and
  Remote/Draw.lean): fixing them on main is safe for Monday, which runs from the branch, but conflicts at the onsite merge.
  Each is a few lines.

## Retired identifiers in our slice: 226 Lean names in 72 files

Top's 226 are the Lean declarations whose own names carry a word in `LEAN_WORDS` (`tests/test_retired_terms.py`:
draw→sample, law→sampling design, wrong→incorrect, dampener→slowdown, horizon→monitoring period, warden and tap→network
certifier), counted as `retired_names` counts them, in the files of the path filter above, at `next`. By word: draw 161,
horizon 18, tap 17, wrong 14, warden 10, dampener 4, law 2.

The split:
- **39 files (117 names) that push 1b doesn't change:** rename them on main now.
- **33 files (109 names) that push 1b changes:** renaming them on main is safe for Monday, which runs from the branch, but
  each conflicts at the onsite merge. Write those renames as a `tools/move/rename.py` map, and the onsite merge reruns the
  map on push 1b instead of resolving 33 files by hand. Monday's Lean audit builds push 1b's copies of these.

Separately, `tools/move/retired_ids_map.toml`'s identifiers occur 114 times in files of the same path filter at `next`, and its
`rename.py` run takes them. They are not part of the 226.

| File | Names | Words | Push 1b changes it |
| -- | -- | -- | -- |
| `Security/Definitions/Accounting/Compose.lean` | 2 | draw 2 | yes |
| `Security/Definitions/Accounting/Filler.lean` | 1 | wrong 1 |  |
| `Security/Definitions/Accounting/Gates/Codec.lean` | 4 | draw 4 |  |
| `Security/Definitions/Accounting/Guarantees/Actual/Dampener.lean` | 2 | dampener 2 |  |
| `Security/Definitions/Accounting/Guarantees/JointLive.lean` | 2 | horizon 2 |  |
| `Security/Definitions/Accounting/Guarantees/RateAuditHorizon.lean` | 4 | horizon 2, draw 2 |  |
| `Security/Definitions/Accounting/Guarantees/RemoteDraw.lean` | 5 | draw 5 | yes |
| `Security/Definitions/Accounting/ReadUnit.lean` | 1 | draw 1 |  |
| `Security/Definitions/Accounting/Remote/Draw.lean` | 2 | draw 1, wrong 1 | yes |
| `Security/Definitions/Accounting/Remote/Held.lean` | 1 | wrong 1 | yes |
| `Security/Definitions/Accounting/TS/Drawn.lean` | 2 | draw 2 |  |
| `Security/Definitions/Accounting/TS/Receipt.lean` | 1 | wrong 1 |  |
| `Security/Definitions/NetworkCertifier/Guarantees/Pooled.lean` | 1 | wrong 1 | yes |
| `Security/Definitions/NetworkCertifier/Tap.lean` | 2 | tap 2 | yes |
| `Security/Definitions/NetworkCertifier/TapRows.lean` | 4 | tap 4 | yes |
| `Security/Definitions/NetworkCertifier/Warden.lean` | 4 | warden 4 | yes |
| `Security/Proofs/Accounting/Actual/Dampener.lean` | 2 | dampener 2 |  |
| `Security/Proofs/Accounting/Compose.lean` | 2 | draw 2 | yes |
| `Security/Proofs/Accounting/Compose/ServiceBudget.lean` | 1 | draw 1 |  |
| `Security/Proofs/Accounting/JointLive.lean` | 2 | horizon 2 |  |
| `Security/Proofs/Accounting/Lemmas/TS.lean` | 3 | draw 3 |  |
| `Security/Proofs/Accounting/RateAudit/Deployed.lean` | 13 | draw 9, horizon 4 | yes |
| `Security/Proofs/Accounting/RateAudit/DeployedRelay.lean` | 1 | draw 1 | yes |
| `Security/Proofs/Accounting/RateAudit/DeployedSketch.lean` | 10 | draw 8, horizon 2 | yes |
| `Security/Proofs/Accounting/RateAudit/Horizon.lean` | 3 | horizon 3 |  |
| `Security/Proofs/Accounting/RateAudit/Session.lean` | 5 | draw 5 | yes |
| `Security/Proofs/Accounting/ReadUnit.lean` | 2 | draw 2 |  |
| `Security/Proofs/Accounting/RemoteHeld/Draw.lean` | 8 | draw 8 | yes |
| `Security/Proofs/Accounting/RemoteHeld/DrawMeets.lean` | 12 | draw 10, wrong 2 | yes |
| `Security/Proofs/Accounting/RemoteHeld/Line.lean` | 1 | wrong 1 | yes |
| `Security/Proofs/Accounting/Served.lean` | 1 | draw 1 |  |
| `Security/Proofs/Accounting/Stacking.lean` | 1 | draw 1 | yes |
| `Security/Proofs/Accounting/TS/CheckedSound.lean` | 1 | draw 1 |  |
| `Security/Proofs/Accounting/TS/Drawn.lean` | 3 | draw 3 |  |
| `Security/Proofs/Accounting/TS/FixedMeets.lean` | 2 | draw 2 |  |
| `Security/Proofs/Accounting/TS/Keyed.lean` | 10 | draw 10 |  |
| `Security/Proofs/Accounting/TS/Receipt.lean` | 1 | wrong 1 |  |
| `Security/Proofs/Accounting/TS/Session/Assumptions.lean` | 1 | law 1 | yes |
| `Security/Proofs/Accounting/TS/Session/Group/Premises.lean` | 1 | law 1 | yes |
| `Security/Proofs/NetworkCertifier.lean` | 1 | wrong 1 | yes |
| `Security/Proofs/NetworkCertifier/Pooled.lean` | 3 | draw 1, horizon 1, wrong 1 | yes |
| `Security/Proofs/NetworkCertifier/PooledLeak.lean` | 1 | horizon 1 | yes |
| `Security/Proofs/NetworkCertifier/RecordRows.lean` | 4 | warden 3, tap 1 | yes |
| `Security/Proofs/NetworkCertifier/Tap.lean` | 3 | tap 3 | yes |
| `Security/Proofs/NetworkCertifier/TapRows.lean` | 7 | tap 7 | yes |
| `Security/Proofs/NetworkCertifier/Warden.lean` | 2 | warden 2 | yes |
| `Security/Proofs/NetworkCertifier/Witnesses.lean` | 3 | wrong 2, draw 1 | yes |
| `Security/Proofs/Properties/Accounting/Remote/Coins.lean` | 3 | draw 3 |  |
| `Security/Proofs/Properties/Accounting/Remote/Draw.lean` | 1 | draw 1 |  |
| `Security/Proofs/Properties/Accounting/Remote/Frozen.lean` | 3 | draw 3 |  |
| `Security/Proofs/Properties/Accounting/Remote/Lifetime.lean` | 1 | draw 1 |  |
| `Security/Proofs/Properties/Accounting/Remote/Order.lean` | 3 | draw 3 |  |
| `Security/Proofs/Properties/Accounting/Remote/Pooled.lean` | 1 | wrong 1 |  |
| `Security/Proofs/Properties/Accounting/Remote/Released.lean` | 1 | draw 1 |  |
| `Security/Proofs/Properties/Accounting/Remote/Session.lean` | 2 | draw 2 |  |
| `Security/Proofs/Properties/Accounting/Remote/Shared.lean` | 12 | draw 12 |  |
| `Security/Proofs/Properties/Accounting/Remote/SharedLink.lean` | 18 | draw 18 |  |
| `Security/Proofs/Properties/Accounting/Remote/Window.lean` | 1 | draw 1 |  |
| `Security/Proofs/Properties/NetworkCertifier/Check/LinkWindow.lean` | 1 | warden 1 |  |
| `Security/Proofs/Properties/NetworkPeriod.lean` | 2 | draw 2 | yes |
| `Security/Properties/Accounting/Remote.lean` | 2 | draw 2 |  |
| `Security/Properties/Accounting/Remote/Coins.lean` | 4 | draw 4 |  |
| `Security/Properties/Accounting/Remote/Line/Record.lean` | 4 | draw 4 |  |
| `Security/Properties/Accounting/Remote/Shared.lean` | 9 | draw 9 |  |
| `Security/Properties/Accounting/Served.lean` | 1 | draw 1 | yes |
| `Security/Properties/NetworkCertifier/Device.lean` | 1 | horizon 1 |  |
| `Security/Properties/NetworkPeriod.lean` | 3 | draw 3 | yes |
| `verity/core/devices/certifier/lean/CertifierDevice/Driver.lean` | 1 | draw 1 | yes |
| `verity/core/protocols/remote_bandwidth/lean/RemoteAudit/Difftest.lean` | 1 | draw 1 | yes |
| `verity/core/protocols/remote_bandwidth/lean/RemoteAudit/Line.lean` | 1 | draw 1 |  |
| `verity/core/protocols/remote_bandwidth/lean/RemoteAudit/Registry.lean` | 1 | draw 1 |  |
| `verity/core/protocols/remote_bandwidth/lean/RemoteAudit/Window.lean` | 1 | draw 1 | yes |

## Reply for top's thread

network-accounting → top, TOP-223 (census at `next` ff744ace042). Our 83 open PRs: 73 CLOSE, 10 KEEP-MAIN, 0 KEEP-ONSITE.

Please take:
1. **Close** all 83 except #7109 #7165 #7172 #7240 #7262 #7301 #7326 #7410 #7584 #7891, keeping every branch. 54 are carried by push 1b or push 2, 2 by proofs' #7390, and 17 are idle, canceled or superseded.
2. **Lane text:** 7 sites remain, in the certifier README (3), the remote_bandwidth README, RemoteBinding.lean, Accounting/Remote/Draw.lean and Accounting/Wipe.lean. `--prune` our other 12 entries.
3. **Retired names:** 226 Lean names in 72 files. Rename the 39 files push 1b doesn't change now. Push 1b changes the other 33: renaming them is safe for Monday, but do it as a `rename.py` map, so the onsite merge reruns it.

We keep 10, folding to 7: live NET-169/189/195/207/208/219/238 and LEAN-197 work. Per-PR reasons: na-pr-triage.md.
