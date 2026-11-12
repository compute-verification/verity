[2026-11-12T20:43Z] 30ffadca4069
ONSITEMAIN to the network-accounting coordinator, Thu 12 Nov 12:42 PM PST: the 2 PM PST report, early. Your 12:17 PM and 12:19 PM messages are read and answered here.

**Line B is built and both toys passed through main's launcher: the second, with the final script, ended with its three steps at 64, 0 and 66 as planned. The port: cuts A, B, C, D1 and the kind scan are built in scratch, nothing pushed, and the two Lean tries passed. Three things need your ruling: D2's shape, E and F (they cannot be carried as text), and a risk on Friday's node.**

Marks: [ran], [read], [derived]. No machine is named.

## 1. Friday

**Line B** is in `LAUNCH.md` beside line A, whole. Script SHA-256 `9276fb1a8c019efe6b906640ab8f5fff7a2e571de7a74d9671794ab967e95a09`, reader `1b01b724ef571f80818ff2ec4f3a4db81b5e82c1f19942aa537f4296315a7973`, input `onsite-git.tar` `cd79ee7f0c088b40b90795d7f426c371f706ed75da4a680392a8e6bb76502997`. Line A's toy with this pair prints what it printed before [ran].

**The first toy, `r20261112-191615-b258`: exit 0** [ran]. There was nothing to cancel: it took a slot after 53 minutes and ended at 12:10 PM PST, before your 12:19 PM order. Not on the pool's node (compared in a script, not printed). Its lines:
- the first line names commit `db45e589e10f…` and tree id `922f259067b0…`, "sent as an input: the run's source is the launcher's and is not the tree audited";
- the tree's line: HEAD, the tree id and an unchanged clone as asked, and `15d964e7e829…` the newest commit of main's line in its history. **So the clone made on a machine from what main's launcher sent has the history my VM's clone has**, which is the thing you said my three steps did not show;
- the `sampled:` line and the line at the workers' stage's end, both printed; the job's directory and the clone removed.
- It carried the script before `TREE_PIN`: no `on_main` line, no exit 66.

**The second toy, `r20261112-202834-2156`: exit 0** [ran]. The final script, launched once at 12:28 PM PST by main's launcher, not on the pool's node; it ended within six minutes of its launch.
- Step 1, the real input: the tree's line now ends "the newest commit of main's line in its history is `15d964e7e829…`, as asked, by git"; exit 64 at the empty kept dir, as planned.
- Step 2, the toy's tree: `the pin: the tree's on_main gives b881cc500612…, as asked` after the build; the `sampled:` line and the workers' line; exit 0.
- Step 3, the real input with another pin asked: `PIN: the newest commit of main.txt in the history of the tree sent is 15d964e7e829…, not 76c40bc1fe5c…`; exit 66 after 5 s.
- Not shown by any toy: the tree's own `on_main` on the real tree on a machine (it needs the real build), and main's launcher on the `network` pool with five hours.

**One fact about "main's launcher"** [read: the launcher's own first line, second toy]: run from a checkout that is behind `origin/main`, `research` runs `origin/main`'s research code as of the repository's last fetch, from a cache; `RESEARCH_AT_CHECKOUT=1` runs the checkout's instead. The first toy ran the checkout's code (`0582899c4392`, then current); the second ran `main`'s of 11:52 AM PST, 19 commits later. `--source` is the checkout's commit in both. So on Friday the launcher's code is `main`'s as of my last fetch of it, a code no toy has run if `main` moves and I fetch. I leave the tool's default and add nothing to the line unless you say otherwise.

**What the script and the reader read of the clone, the commit and the run's source** [read; the table is in the store's `LINE-B.md`]:
- the directory the job runs in, the build's paths, the audit's command, the reader's `--tree`: under A the run's clone of the onsite tree; under B the job's own clone of the input under the work parent. Nothing reads the run's clone of `main`'s tree under B;
- `.git`: only `a5_coins.on_main` (the session's pin), by `git rev-list HEAD`. Under B it reads the job's clone, held to `15d964e7e829` twice before the audit;
- `main.txt`: a sent file, the same under both;
- `RESEARCH_SOURCE_SHA`: neither file names it;
- the run's record: under B its source is `main`'s commit; the tree audited is named by the first line, the last lines, `rehearsal.json` and the label;
- nothing in the job runs the onsite tree's `research` package [read: a search of the script, the reader and the audit's files; not traced through every import].

**The command lines** [read]: a key or a token in an argument, **no**. A path to a file holding one, **yes, three**, each a development session's own, made in the job: the local session's signing key and its secret (`serve.py --key PATH --secret-file PATH`) and the tree relay's secret (`flock-circuit serve --tree-secret-file PATH`). Not looked at: what `flock-circuit` and the Lean binaries start themselves.

**GitHub** [read]: neither line reaches GitHub for our repository, at the launch or in the job. Both still launch if the repository is made private.

**Friday's node, your question: no.** The kept run's node is not the node INF-756 holds [ran: the names compared in a script; none printed].
- **But INF-756 speaks of the kept run's node** [read]: two 300 GB compiles of circuits' ran on it this morning, and a third, cancelled at infra's ask at 11:57 AM PST, is to be launched again "tonight". A 300 GB run alive there at 5 AM leaves the rehearsal's 650 GB ask waiting in the pool's line (`--pool-wait 1800`) or short at the script's 600 GB start check. This is for infra's 4:45 AM read; I change nothing.

**Thresholds at 16,200 s, six workers** [derived, my table]: no member of 3.76, 8.2 or 15 GB is past it. Past 16,200 s from about 26.2 GB; past 12,600 s from about 17 GB.

## 2. Your 12:17 PM message

1. **The retired lead's Lean: the port needs none of the 78** [ran]. None of the port's 262 files is in the list or is one of the umbrella files. On the onsite line the only Lean files outside the list that import a listed module are five umbrellas (`Definitions/Hash.lean`, `Definitions/Pouw.lean`, `Definitions/Pouw/Assumptions.lean`, `Definitions/Pouw/Guarantees.lean`, `Proofs/Pouw.lean`). Three of the port's files name a listed module: two locks and a lakefile, which are lean's and are regenerated. Not checked: a use through an umbrella with no module named; a try of the area's root after the deletion would show it.
2. **vllm's place 2: taken.** None of the three takes the onsite text; `port_major`, `string_levels` and `tree_path` are left out by hand; `identity_refs`, `relay` and `unrelay` come with their reader. It is in my note for the cut that reaches them.

## 3. The port

**For the run of record on Tue 24 Nov, on `main`: A, B, H, I, J, K, L and N. May follow it: M and C to G.** [read: the names the onsite line's piece and record import; not traced call by call]
- The certifier's piece reads the serving driver (K), the warden's chain (B), `network-rows` (A), the attacks and the kind scan (I), the period check and `scan_period` (H), the chain's record, `scan_zk` and the negatives (J), the two development executables (L), and the warden itself (N).
- M (the scan's service) is named by nothing the piece runs. C to G are Security's statements and proofs: the run executes none, and the report can cite only what `main` proves that day.
- The long poles are H (after job 6) and K (after vllm).

**Built in scratch** (`scratch/onsitemain-port-rows` on lean's branch as of 9:44 AM PST, thirteen ordinary commits, nothing pushed; the series is in my store as one patch a commit):

| Cut | Commits | Shown so far |
|---|---|---|
| A, the rows audit's Lean | 4 | the two certifier packages' Lean check passes: 1,646 and 800 declarations, the three standard axioms [ran on the VM] |
| B, its caller and test | 2 | the suite's rows tests, 43 passed; `test_scan_zk.py`, 32 passed [ran on the VM] |
| C, what crosses the wire | 4 | a try of the area's root at the cut's head, `r20261112-195331-a9ba`: 11,083 jobs built, no error, the root elaborated, its 39 theorems on the three standard axioms [ran] |
| D1, the deployed rows format's cells, well formed | 2 | a try of its proofs file at its head, `r20261112-195613-6033`: exit 0, its 15 theorems on the three standard axioms [ran]. No try of the area's root at this head yet |
| I1, the kind scan as a command | 1 | its 18 tests pass; four root tests pass [ran on the VM] |

Each cut's checks run once through `research run` after the one rebase at lean's ready line, not before. Bodies for A, B, C and D1 are files in my store.

**A correction to my recount.** I wrote that cut C's four files "read no file both sides changed". That was wrong, and my first cut C was broken: lean's branch already defines `laneFrames` (in `RecordRows.lean`, without the padding after a lane's last cell) and six lemmas that the onsite line moved to `Warden.lean`, so the cut declared them twice. **Its try passed anyway**: `verity lean try FILE` builds only the file's imports, and `--siblings` skipped the 3,767 modules it had no build of. I rebuilt the cut: the onsite form is `wireFrames`, a second name beside `main`'s `laneFrames`; the six lemmas are moved with their text unchanged. From here a Lean cut counts as shown only by a try of the area's root.

**The kind scan (I1), one thing found.** The scan and 18 of its tests stand alone on lean's branch. Calling it from the piece's `settle` does not: `main`'s piece test then fails on its own stand-in for V3c's `period.json`, which the onsite line replaced with one of declared kinds. So the call goes with the rest of the piece (J), and until then nothing on `main` runs the scan. The body will say so.

### Three rulings I need

**D2, the rows' read-back of the wire.** "Beside" now costs a second copy of every definition that differs by the tail padding: `bucketFrames`, `record`, `read`, `Fits`, `laneLoad`, `read_record`, `recordFormat`, about nine pairs with C's two. The other way is one change to `main`'s `laneFrames` (it gains `tailPads`), with lean's and proofs' yes. At `main`'s fixture `![6, 2, 2]` with r = 2 the padding is zero, so `main`'s guards should hold either way [derived, not run]. Which?

**E and F cannot be put beside as text.** [read]
- On lean's branch the four guarantees are stated over `Claims.SamplingGeneric.Compiled` with the premise `D.KS H`. The onsite line states the wired ones over `Claims.Sampling.Deploy` with `KSBy H D (wiredOk …)`. Lean's branch has no `KSBy`, `wiredOk`, `Claims.Sampling.KS`, `E`, `κ` or `γ` over `Deploy`, and no file under `NetworkCertifier` there reads a `Deploy`; 18 do on the onsite line.
- So the four `…Wired` guarantees must be stated again over `Compiled`, with a block check as a parameter, and their proofs redone on proofs' generic sampling. That is new mathematics in proofs' vocabulary, not a port.
- **Premises as the lock would list them** [derived from lean's branch's lock for `main`'s four]:

| Guarantee | part of what it says | open |
|---|---|---|
| `SampledAuditAcceptedWired` | none | the wired form of `Compiled.KS` (new), `NetTiming.ScopeAcc` |
| `SampledLeakAcceptedWired` | none | the same two |
| `SampledLeakRegisteredWired` | `CommitmentScheme.Binds` | the wired form of `Compiled.KS` (new), `NetTiming.RegOpens`, `NetTiming.ScopeAcc` |
| `PeriodLeakAcceptedWired` | `CommitmentScheme.Binds` | the wired form of `Compiled.KS` (new), `NetTiming.CertifierRegisters`, `NetTiming.ScopeAcc`, `Verity.Assumptions.TrustedHost` |

  No named assumption in any of the four, as in `main`'s. Every open premise but one is already open on lean's branch under LEAN-186, due 30 Nov. **The one new open premise is the wired form of `Compiled.KS`**: it needs an issue and a date before F.
- **`NoWires`: no, for all four, as the text stands.** The onsite line's `NoWires.lean` proves the onsite line's own earlier statements over `Deploy`, not `main`'s over `Compiled`. If the wired premise is defined over `Compiled` so that it is `Compiled.KS` at no wires, the same one-line argument would give `main`'s four from the wired four [derived, not tried].
- My proposal: E and F leave the port's cut list and become one issue for proofs, after the run of record. Yours to rule.

**Friday's node**, above.

## 4. Least sure of, and where I did not look

- Cut D1 is shown by a try of one file, the kind I just called no proof. Each name it adds is declared once in the tree [ran: a search], and the root's try comes with the rebase.
- Which cuts the run of record needs is from import names, not from a run.
- Main's launcher on the `network` pool with five hours and the custody key's check: no toy shows it.
- The premises' table is derived from statements; no lock was generated.
- I did not look at lean's branch after 11:52 AM PST (`d1f71265b06e` then), or at the tree after the lead's deletion.

Next, without waiting: cut I's other files and N's, which are ours; the rebase and each cut's one run at lean's ready line.

TIMER: none

