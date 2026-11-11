"""`research queue`: the merge queue.  `next` is `checked` (drive.toml's `land_into`: `Queue.into`) plus every
admitted PR, one merge commit each, in the order they were admitted.  Drive (`research.drive`) fast-forwards `checked` to the top
of the longest prefix of `next` whose every member has a reviewer's land verdict at its head, merges onto `checked` each member
whose merge changes no Lean, lock or soundness-path file (drive.toml's `lowrisk_lane`) on its ready mark alone, and `main` to a commit of `checked`
only once that commit's after-landing run, the full check, passed every step on the merge gate (not circuit-check, policy.toml's
`off_gate`), or failed it only where an unexpired allowance of that policy covers: `main` holds only tested commits.

    research queue status [--json]                 each open PR as the verified queue admits it onto `main` (no `next`): what
                                                   it waits for, why it was refused, or that its head is on `main`;
                                                   `stale` when it is unmarked at its head and idle drive.toml's `stale_days`
                                                   (`research.stale`), which drive closes at `stale_close_days`; then
                                                   every open PR whose body names no Linear issue (`named_issue`: a key
                                                   after a Linear magic word, a bare key of a team main's Slack registry
                                                   lists, or an `Issue: none: REASON` line); --json gives each its `issue`
    research queue sync [--dry-run]                admit and drop PRs, merge them onto `next`, push it
    research queue eject PR --reason TEXT [--by HANDLE]
                                                   keep the PR's current head out of `next`, and a new head whose own change is
                                                   the same (`StoreLabels`); a ready mark re-admits it;
                                                   posted as HANDLE to `CI` and the owner in the merge thread (exit 1 if it couldn't be)
    research queue ready PR (--on MACHINE | --local | --mark) [--by LANE] [--no-preflight] [--sha SHA] [--json]
                                                   the owner's word that the PR is ready at its current head (a new push needs it
                                                   again unless its own change is the marked head's;
                                                   --sha: only at SHA, refused when GitHub still has another head after a few reads),
                                                   once the quick tier passes there: as a run on MACHINE in a check slot
                                                   (`RemoteQuickTier`; `auto` or M1,M2: the one with a free slot), or here
                                                   (`QuickTier`); exit 3 when it never started (no slot within its wait), 79 when
                                                   --on's run is still going after 20 min (research.waitcap: `judge RUN` marks it
                                                   once it ends); or, with
                                                   --mark, on the owner's word alone (`MarkOnly`).  First it refuses a PR
                                                   whose head is on another repository or whose author isn't the
                                                   organization's or one of `AUTHORS` (`PR.outside`), as admission does; it
                                                   warns on stderr, and goes on, when the body has neither a `Goal <id>` line
                                                   naming a live goal of the Goals page (on it, and not done) nor an `Issue:
                                                   none: REASON` line (`goal_gate`; the goals as this machine last read them
                                                   when the page can't be read, said in one line), and (the records gate,
                                                   `RecordsGate`: the lock run after its landing regenerates the records, and
                                                   a failure there blames it) when the head changes a Lean lock against the
                                                   landing branch that no records run or Lean check of any commit passed on
                                                   the head's Lean inputs (their key, `lean_changed.py --inputs-key`: the
                                                   files of the package and of the packages it reads, their toolchains and
                                                   the Lean checker), naming the run to start; then,
                                                   in about a minute, the preflight (`Preflight`) refuses, with
                                                   the fix, a head that conflicts with a
                                                   fresh `next` outside the pins (naming the paths), contains an
                                                   open PR that isn't admitted (naming it), fails a landing gate of drive.toml at `main` (with
                                                   the gate's fix; a warn gate's failure is printed as a warning and refuses
                                                   nothing), or, merged onto `next` in a temporary worktree, fails `pins.py
                                                   --stale`, tests/test_module_strings.py, test_pins.py, test_doc_paths.py,
                                                   test_repository.py, tools/research/tests/test_pythonpath.py, or `pytest
                                                   --collect-only` in a suite
                                                   whose inputs its change touches, or breaks the build of a Lean module that
                                                   is or imports what it changes (on a warm Lean tree: here on a node, else
                                                   on the first node of PREFLIGHT_LEAN_ON that starts it, whatever the tier;
                                                   refused when none can build them),
                                                   naming the PR on `next` that owns it, or
                                                   drops from a Lean lock it changes a shape.inline premise or shape.frozen
                                                   key `next`'s statements still need (naming each; the sync leaves such a
                                                   head off `next` too); a check it couldn't run it names under "checks
                                                   skipped";
                                                   --no-preflight skips it (the lander's)
    research queue drive [--on MACHINE] [--config PATH [--config-from-tree]] [--marked-after TS]
                         [--[no-]regenerate-locks] [--[no-]after-landing-no-cache] [--priority-file PATH] [--dry-run]
                                                   one tick of the landing automation (`research.drive`): sync and land the
                                                   reviewed prefix of `next` on `checked`, a Lean change as any other, commit
                                                   the lock records a lock run regenerates after a Lean change, then check
                                                   `checked` (--after-landing-no-cache: reusing no pass but circuit-check's)
                                                   and move `main` to what passed every gated step; tell `CI` the rest.
                                                   Its flags (`DriveFlags`) are tools/verity/check/drive.toml at `main` in the
                                                   queue's clone, fresh each tick, so a change to them is a reviewed commit
                                                   (`drive_flags`); a flag given here overrides its key only where drive.toml's
                                                   `command_line` lets it (`--on`), else the tick refuses; --config
                                                   PATH (at `main`, or with --config-from-tree in this file system) is refused,
                                                   naming each key, unless every key it sets has main's value.  Before it syncs,
                                                   the tick refuses, with the fix, a name its flags use that doesn't exist
                                                   (`drive_names`: the landing branch and drive's push to it, each machine of
                                                   `on`, each landing and warn gate at `main`); the effective flags print once
                                                   a tick.
                                                   --priority-file PATH: drive's priority list, the landing order: PR numbers
                                                   separated by spaces or commas, first to land first, each in its first place
                                                   (`priority_list`), read again at each sync (`Queue.priority`): no file is an
                                                   empty list, and one it can't read is logged once and taken as empty; a
                                                   --first is refused, naming this file
    research queue review [N (--land [--ref NOTE] | --hold REASON) --by LANE] [--sha SHA] [--json]
                                                   without N: the open PRs onto `main` marked at their head and not on `main`
                                                   that no land verdict there lands, oldest mark first, those the verified
                                                   queue waits on a reviewer for (number, head, owner, mark time, files, whether
                                                   it is a Lean change: `lean_change`); with N: a reviewer's verdict, a `reviewed` label on
                                                   `pr:N@HEAD` (its head on `next`, or --sha, a full or short SHA), which drive's
                                                   review landing counts unless LANE is the PR's owner, its handle or its ready
                                                   mark's author, refused here; without --sha, refused when GitHub has another
                                                   head than `next` (pushed since drive's last sync)
    research queue delegate AGENT --by LANE [--days N]
                                                   as LANE's holder, let the Cursor agent AGENT (a worker) sign ready marks and
                                                   land verdicts as LANE for N days (default 7, at most 14): see Signed verdicts
    research queue baseline SHA --by ci [--steps STEP,...]
                                                   fast-forward `main` now to SHA, a commit of `checked` known green on
                                                   every step on the merge gate, beside drive's tick (no lock, no state written);
                                                   drive's next tick takes it as verified, and the review landing lands on it
                                                   (on the control pod: `/data/infra/drive.sh queue baseline SHA --by ci`)
    research queue promote                          fast-forward `main` to the tip of drive's after-landing run once every shard
                                                   of it ended passing every step on the merge gate, beside drive's tick, as
                                                   `baseline` does; nothing to do before (on the control pod, from a loop:
                                                   `/data/infra/drive.sh queue promote`)
    research queue accept-checked SHA --by ci       record `checked`'s tip SHA, a commit drive didn't write, as one drive takes,
                                                   so its ticks, held on it (`Driver._unwritten`), go on from it
                                                   (on the control pod: `/data/infra/drive.sh queue accept-checked SHA --by ci`)
    research queue drive-shard-add TIP K RUN_ID     count RUN_ID, a run replacing shard K's twin, as one of that shard's in the
                                                   after-landing run of `checked` at TIP (`drive_shard_add`), refused unless
                                                   RUN_ID's own record is a check.py run of TIP's commit with --shard K/N
                                                   (an auto plan's: auto:K/N, with that plan among its inputs)
                                                   (on the control pod: `/data/infra/drive.sh queue drive-shard-add TIP K RUN_ID`)
    research queue judge RUN [--on MACHINE] [--by LANE]
                                                   the mark `ready --on` writes, from its quick-tier run RUN that already ended
                                                   (its caller gone), for the open PR at the head the tier ran at, or at its tree
    research queue open --head BRANCH --title T --body-file F --issue KEY|none:REASON [--sha SHA] [--base main] [--draft]
                                                   open a PR for any pushed branch through gh (the repo's broker wrapper when the
                                                   checkout has one), for a lead whose PR tool opens PRs only from its own run
                                                   branch, its body F with a last line naming its Linear issue: `Refs KEY`
                                                   (Linear's non-closing magic word: the merge leaves the issue open), or
                                                   `Issue: none: REASON` (`with_issue`); KEY's issue carries the label of the
                                                   goal it serves, a live goal's id on the Goals page, written as a `Goal <id>`
                                                   line after `Refs KEY` unless F has `Goal` lines of its own (`with_goals`),
                                                   warning on stderr, and opening it all the same, when KEY's labels name no
                                                   live goal or can't be read, or a `Goal` line of F names another;
                                                   refuses a KEY that isn't one (like ENG-123) or that Linear has no issue
                                                   for, an empty REASON, and when BRANCH
                                                   isn't on the remote, isn't at --sha, or already has an open PR; prints the
                                                   PR's URL (`open_pr`), then links the PR to KEY through the Linear app
                                                   (`link_issue`: Linear's GitHub integration is off for this repository, so
                                                   the line alone links nothing); a link that fails leaves the PR open, printing
                                                   why and the `link` command that retries it
    research queue link PR --issue KEY             link the open PR to Linear issue KEY through the Linear app, as `open` does
    research queue edit PR --body-file F [--title T] [--dry-run]
                                                   replace the PR's body with F outright (and its title with T), through gh as
                                                   `open` does, keeping only the Cursor PR tool's markers and footer (`edit_pr`);
                                                   refuses a body that drops or changes the PR's `Lane NAME` owner; --dry-run
                                                   prints the body it would write
    research queue stats --log FILE [--hours H] [--until TS] [--json]
                                                   where landed PRs' time went, from drive's log (`research.queue_stats`)
    research queue checked-heads [--tip REV] [--json] [--out FILE]
                                                   the commits on REV's first-parent line (default origin/main) whose own `check`
                                                   passed and landed them, oldest first, one a line: each train's head, by drive's
                                                   `landed` labels in the store index (`merge.checked_heads`), less a head whose
                                                   train a later `git revert` names. --out FILE writes them newest first, an
                                                   optional source for the verifier's --main file (`merge.main_line`), for a
                                                   deployment that wants credit's line to be the checked train heads rather than
                                                   the raw `git rev-list --first-parent origin/main`, the default
    research queue shadow-report [--tips N]
                                                   with drive.toml's `executor = "shadow"`: the newest N `checked` tips whose
                                                   after-landing run drive compared with the executor's verdicts, each one's
                                                   agreement and every disagreement (parts missing in the executor and tasks
                                                   only it ran among them), and whether the newest 3 agree fully, none
                                                   incomplete: the gate for `executor = "on"` (`research.executor`)
    common flags: [--shadow] [--workdir DIR] [--rules FILE] [--repo OWNER/NAME] [--remote URL]

Admission.  A PR is admitted while it is open against `main`, has an owner, carries no hold label, and is marked ready at its head;
no change waits on a per-head grant.  A PR whose head is on another repository than the queue's (GitHub's `head.repo.full_name`:
a fork's, or a deleted one's) or whose author is neither an owner or member of the organization (`author_association`; a
collaborator's isn't enough) nor one of `AUTHORS` (the team's Apps, the repository's admin) is refused, at its mark and at
admission, naming why and the fix: a maintainer opens it from a branch of this repository (`PR.outside`).  So is a PR whose head
carries merges of an old `next`, rebuilt since (`old_next`: `Queue: #N` merges the landing branch doesn't hold, of PRs neither open
nor landed, which would land with it; a demo branch that merged `next` carries hundreds), naming the fix: retarget it to the demo
branch it builds on, or restack its own commits onto `main` (`old_next_refusal`; at its mark, the preflight's first refusal:
`Preflight._old_next`).  A PR based on another branch is never admitted: `ready` marks it and says so,
and `status` lists it, once marked, as waiting to be retargeted to `main` (a PR containing another waits for it there).  Ready is a store label `ready = true` on `pr:{number}@{head sha}` that
`research queue ready PR` writes, so a new push needs a new one unless its own change is a marked head's (below; `ready` marks a head whose tree is a
marked head's, as a merge of `main` that changes no file leaves it, without a tier), and only after the quick tier passes at that
head: `suites.py --quick` on the suites the PR changes, from a clean checkout, either as a `research run` on a machine (`--on`,
`RemoteQuickTier`: in one of its check slots, so several PRs' tiers run at once and the caller only waits) or here (`--local`,
`QuickTier`); a failure refuses and names the tests, and a remote run whose result can't be read, or that reports another head, is a
failure.  `--mark` (`MarkOnly`) writes the mark with no tier, and the after-landing run is the head's first full check.  Draft state is ignored: agents open PRs as drafts, and
the ready label is what says one is done.  The hold label keeps a PR out whatever else holds.  The owner is the lane its body names
on a `Lane {name}` line, else the Cursor agent id its body carries.  The rules file (`ci/queue.toml` on
`main`, `tools/verity/check/queue.toml` until the repository move, or `--rules`) names the hold label.  Ready marks are read after
pulling the target's labels from the store's remote; when that pull can't happen (no remote here, one
that's unusable, a failed pull) the command fails with `STALE`, never answering from this machine's labels alone.  A head with no
ready mark (or land verdict) of its own that counts takes those of each other head of its PR whose own change is the same: the
whitespace-kept patch-id (`git patch-id --verbatim`) of its diff from its merge base with the landing branch (`Queue.own_change`,
`StoreLabels`), and the log names the head they came from.  So a merge of `main`, or a restack that leaves the diff as it was,
costs no new mark and no new review, a withdrawn mark holds there too, and a head whose own diff differs, if only in whitespace
(an indentation, the spaces in a string), takes none.

Preflight.  What breaks or cancels a train is cheap to catch at mark time: a conflict with a PR already on `next`, a stale Lake
pin, a module string naming a module only an unlanded PR adds, a workspace member without its REAL_ROOTS entry, a test that reads
a row another queued PR refits away (an error at collection), one lane's unlanded branch carried in two PRs, a merge of `next`
that carries a PR's head the PR no longer lands (closed, rebuilt without it, or withdrawn: `stale_carried`, after `old_next` for
one closed unlanded; the sync leaves such a head off `next` too), and a failure of
`tests/test_repository.py` (a report-genre name), which the preflight runs too, as it runs the landing gates the sync would (on
the head; the sync, on its merge onto `next`); a warn gate's failure (a Markdown file over its cap) it prints as a warning.  A
Lean change can pass its own Lean check and break a module
another PR on `next` owns (a changed signature, dropped premises, a dropped import), which drive's lock run finds only after it
lands, blaming it; the preflight builds those modules (`LakeBuild`), from a machine without a warm Lean tree on the first node of
`PREFLIGHT_LEAN_ON` that starts the build, whatever the tier, and refuses a head whose modules no node can build.  A head whose lock, merged onto `next` by
`tools/verity/lean/merge.py`, drops a `shape.inline` premise or a `shape.frozen` key that `next`'s unchanged statements still need
(a lock regenerated on an older tree) fails that lock run too; the preflight names each such entry (`lock_drops`), and the sync
leaves such a head off `next` with the same refusal.  So `ready` runs `Preflight` before the mark and before any quick tier: a refusal names what it found and the fix, a
failure `next` has too refuses nothing, and what keeps a check from running is said under "checks skipped" and refuses nothing
(the after-landing run checks it).  A conflict only in pins, or in Lean one side deletes that the other only ported, is merged
as the sync merges it (`merged_tree`), the merged tree's
Lean is ported as the sync ports it (`ported`), and the checks run on that tree.  The merge runs the merge drivers of the commit it
merges onto, as the sync's clone defines them (`base_drivers`): a branch's older checkout would run its own `tools/verity/lean/merge.py`, and one from before `--caps` reports a conflict in
`tests/test_lean_packages.py` the sync never sees.  `--no-preflight` skips it, for the lander.  Drive's sync reads none of it:
the mark admits the PR, and the sync's merge onto `next` is the only check after it (see `next`).

Records gate.  The lock run after a landing regenerates the records of the locks it changed, and what it can't regenerate fails
it, a red whose culprit is fixed forward or reverted; so the gate refuses nothing.  `ready` (and `judge`) warn, naming each package and the run that
would record it, when a head's change against the landing branch (drive.toml's `land_into` at `main`: `checked`, else `main`)
touches a `lean-audit.json` and, for that package, the store holds no records run or Lean check of any commit whose result lists
the package's Lean inputs key at the head among its `passes` (`lean_passed`: the key `lean_changed.py --inputs-key` prints, of the
files of the package and of the packages it reads, their toolchains and the checker), nor vouches for it through a head that only
merged the landing branch or only changed other packages' locks (`RecordsGate`), and go on.  It reads the store's remote and warns
when it can't.

`next`.  Its first-parent chain above `checked` (`Queue.base`) is the queue, and each merge commit's trailers name its PR, head and owner: git holds
the queue's state.  A sync keeps the longest prefix of that chain which still matches the admitted PRs (same PR, same head, same
order) and merges the rest on top, each by `git merge-tree` and `git commit-tree` (`Queue._merge`: no worktree, so a candidate
that conflicts costs one call).  A merge is recorded under what it reads (`Queue._merge_key`): the parent's tree, not its
commit, the head and their merge bases, with the merge's code, git and drivers; the same head merged again onto the same tree,
above a moved or dropped member or on a `checked` moved to the same tree, takes its record's tree, conflict, porting, rebuilt
pins and lock drops (the state's `merges`), and only `git commit-tree` runs; drive's low-risk lane merges onto `checked` through
the same records (`Queue._recorded`).  A merge whose only conflicts are in pins (`tools/verity/check/pins.py`'s `PINS`, generated
sha256 listings of Lean inputs) is kept, each conflicted pin regenerated from the merged tree (`repinned`): two PRs that each
change a pinned input both change its pin.  Two that each change what a JSON pin holds besides its digests (each adds vectors)
conflict in it, since keeping either side's text drops the other's.
So a new push moves its PR to the tip, a dropped PR takes the commits after it
along (they're merged again), and a `checked` that drive moved outside the sync (a landing, a revert, a merge of `main`) rebuilds
`next` on it; one drive didn't write holds drive's tick, which syncs nothing on it until `checked` is reset to the last tip
drive wrote or it is accepted (`Driver._unwritten`, `accept-checked`).  Only drive syncs (`Queue._drive_only`): `eject` only
withdraws the mark, for drive's next tick, and a hand `sync` on the control pod syncs as drive's tick does.  A
PR whose head contains another open PR's head, or by ancestry an older one (a head the queue saw it at, or its branch merged in:
`Queue.entries`; never by PR number, never for commits the two merely share, and never for commits admitted members below it on
`next` already bring), goes
in after it and waits while that one waits, unless it carries that one's change: the same owner, or that one's owner withdrew its
mark (`Queue._carried`); drive tells the owner of the one it waits for (unmarked, ejected, held or waiting itself) and those of
every marked PR left out for it in one post, with the fix (`Driver.tell_out`).  Drive's priority list (`--priority-file`,
read again at each sync: `Queue.priority`) is the landing order: its ready PRs go at the bottom of `next`, in its order, first
to land first, each after the PRs it contains, so an edit of the file reorders `next` from the first PR whose place it
changed.  Members on `next` never move for what a sync adds, merged again or not: those above a moved or dropped
member are merged again in their order, and only a PR of the priority list placed again, or a `next` rebuilt on a `main` that
moved off it, sorts what is above it.  Nor do they
move but for drive's lift (`Queue.lift`, `Driver._lift`): each sync, the members with a land verdict go below those waiting
for one (drive.toml's `verdict_first`), each after the PRs it contains, and land at once; and for drive's park (`Queue.park`,
`Driver._park`): each sync, a
member held by a verdict at its head goes to the top of `next` with every PR that contains it, above what the sync adds, so
one hold holds only itself and what contains it and the members that were above it land; it stays there until a land verdict
(then it lands in its place) or a new head.  A PR that doesn't merge cleanly is left out of that sync, and its owner is told
which files conflict and which queued PRs touched them; while its head stays, later syncs try it, and the PRs that contain it,
above everything else, whatever the priority list would say, so a sync with nothing new keeps `next`.  So is a PR
a landing gate refuses (drive.toml's `landing_gates` at
`main`, `gate_refusal`) on its merge onto `next`,
with the gate's fix, and a PR whose lock merge drops an entry `next`'s statements still need (`lock_drops`), naming each:
each PR the sync merges is judged, a new one or one merged again above a
dropped or moved member, never one in the prefix it keeps, so a refusal never rebuilds what `next` keeps; a merge
with the message, tree and parent tree of one judged under the same `main` takes that verdict (`Queue._gated`).  The warn
gates (drive.toml's `warn_gates`: `tools/verity/check/gates/markdown_caps.py`, a Markdown file a merge puts over its cap) judge
the same merge and keep nothing out: what one says is kept (the state's `warned`, by PR and head) and drive tells the PR's
owner once a head (`Driver.tell_warned`), who fixes it in their next PR.  A PR whose
merge a gate can't judge (it exits neither 0 nor 1, as the unread gate does while no facts of the Lean it reads are published) isn't
left out: it waits, recorded (the state's `waiting`) with its mark kept, as the PRs that contain it do, and each sync tries it
again; a gate that refuses it as well leaves it out.  A ready mark
admits a PR: the sync runs no preflight and waits on none, and its merge onto `next` (with the gates and lock drops above, and
the codemod and the move below) is the only check before the PR joins it; what a merge breaks past that, drive's landing
compile gate (`Driver._compiles`) and the after-landing run find.  The PRs of drive's priority list (`Queue.priority`), with
those they contain, go first of what a sync merges anew.  A PR
that contains one left out waits, recorded (the state's `waiting`) and named in that one's notice; `status` shows it waiting and
names the PR it contains and why that one is out, judging each PR after those it contains (else it shows
'admitted' for hours while one it contains conflicts with `main`).

Once the merged tree is on Lean's module system (it holds the codemod, `MODULES`, and its `PORTED_ROOTS` are modules:
`on_modules`, since the codemod may land before the port), the sync runs that codemod over the whole merged tree, in a
worktree of it, and what it ports goes into the merge commit, with each pin recording it (`ported`): git runs the `.lean`
merge driver only on a file both sides changed, so a `.lean` file either side added before the port (the PR, or `checked`
since the port's base) would stay unported, and a module importing it wouldn't build.  A codemod that fails leaves the PR
out with why.

Moves.  A layout move lands from `next` with no pause in landing.  A PR whose body has a `Move: tools/move/NAME.toml` line
(`PR.move`) is a move PR: its marked head is only the request.  Where the sync puts it, it is regenerated on the commit below it
(`checked`'s tip once the members below it land): `tools/move/rename.py --map NAME.toml` at the root of a worktree of it, one
commit on it with the PR's trailers and `Move: NAME.toml` (`_regenerate`), which `tools/verity/check/move_check.py` must
certify, else the PR is left out with why (a map the tip lacks too).  Both scripts, and `restack.py` below, are the landing
branch's, as its landing gates are (`move_scripts`), so a member below the move can't change what judges it; a member below
it that changes `tools/move/`'s scripts holds the move back until it lands, since the restack runs the move's base's.  It needs a ready
mark and a land verdict like any PR, and is a Lean change when it moves Lean, so a lock run follows its landing; it lands as
that one commit, under the after-landing run.  Each PR merged above a move, on `next` or among `checked`'s last
`LANDED_WINDOW` first-parent commits, whose head lacks it is merged across it (`_merge_through`): restacked onto it by
`tools/move/restack.py` when its change names a word of the move's map or its plain merge conflicts, so only its own
conflicts refuse it.  A restack runs in the background, one `restack.py` a head and at most drive.toml's `restack_jobs` at
once (`_restack`, the state's `restacking`), and its PR waits off `next` meanwhile (`RESTACKING`), a wait that refuses nothing
and rings no one; a later sync reads its result (the state's `restacks`) and merges the PR, drops the result once the PR's head
changed, and runs a job that ended with none (it died, or ran past `RESTACK_WAIT_S`) once more before that failure refuses the
PR, and one something else interrupted (SIGINT, SIGTERM, a restart of the pod) again without counting it.  A move or a restacked PR lands at a commit that isn't its head: the trailers keep it off `next` once
landed (`queue_log`), and drive closes it once `main` reaches it (`Driver._close_carried`).  The low-risk lane stops at a
move on `next`.  The preflight merges no move PR's head, and notes rather than refuses a conflict only in files a move changed.

Landing.  `drive` (`research.drive`) is the loop's tick that needs no agent: it lands `next`'s reviewed prefix, a Lean change
as any other, by a fast-forward of `checked` (no new merge commit), commits the lock records a lock run regenerates after a
Lean change, and checks `checked` after each landing,
naming the culprit of what that check fails (fixed forward, else reverted on `checked` unless a warn step failed: drive.toml's
`warn_steps`), and fast-forwarding `main` to each commit whose check passed every step on the merge gate; it tells
`CI` what it refuses.  A push refused because `checked` moved leaves it alone; drive's next tick takes the new `checked` only
if drive wrote it, and otherwise holds and rings `CI` and `TOP` (`Driver._unwritten`).  A PR landed on `checked` stays open on GitHub until `main` reaches it, and no sync admits it again.  What reaches
`main` and changes what a guarantee says sends the spec-change DM of what changed (`spec_alert`).

Signed verdicts.  A label's `by` is whatever its writer said.  With drive.toml's `signed_verdicts` on at `main`, drive counts a
ready mark or land verdict only beside a `label_auth` label that authenticates it (`research.labelauth`, `StoreLabels`): a
Cursor identity token over the label's SHA-512 nonce, minted when the label was written by the agent that holds its `by` in
`main`'s registry, or by a worker that holder delegated to (`research queue delegate`); `ready = false` (an ejection) and holds
count unsigned.  `ready`, `judge` and `review` sign each ready mark and land verdict they write whenever this agent can mint a
token, and with the flag on refuse to write one they can't sign, or one an agent that neither holds the handle nor has its
delegation would sign; the
`review` listing names each land verdict that doesn't count and why.  Drive refuses a tick when it can't read the registry or
the issuer's keys (the last set fetched, kept beside the queue's clone, covers an outage of the issuer), since judging every
mark absent would take every PR off `next`, and the flag is refused unless the low-risk lane can't land a change to the registry
or the code that judges a label (`SIGNED_GUARDS`).

Notices.  A refusal (a conflict, an ejection) is posted in the merge thread to `CI` and the owner's handle, once per PR head:
drive tells each conflict its sync finds, and `eject` posts as its ejector.  Nothing is written to the notes: a handoff note
there reached its owner only when the VM that ran the sync pushed its notes, inside that lane's commit.  What a sync leaves out
is told once per root cause instead, the PR that didn't merge or isn't admitted, naming every PR it takes out and tagging each
owner once, and again only when its head, those PRs or its reason changes (`Driver.tell_out`).

`--shadow` builds `queue-shadow` instead of `next`, never pushes `main` and sends no notices, so the queue can run beside the trains
with no effect on them; its `sync`, `land` and `drive` still run only on drive (`--dry-run` shows the same off it).

Files.  The queue works only in its own clone (`--workdir`, default ~/.research/queue/NAME for `--repo OWNER/NAME`, else
~/.research/queue/verity) of its remote (`--remote`, default https://github.com/OWNER/NAME.git for `--repo`, else the caller's
origin; one naming another repository on GitHub than `--repo` is refused: `remote_and_repo`), and fetches with forced refspecs: a
local copy of a branch that wasn't force-fetched once put a train on a stale base.  Only a clone the queue made has its origin
set to the remote (`Queue._clone`): it sets `research.queueclone` in each clone's git config, and those it made before that are
at ~/.research/queue/verity.  Any other checkout keeps its origin, so a push from it still goes where its owner meant, and one
whose origin is another repository than the remote is refused before anything in it changes.  `state.json` beside the clone
keeps ejections and the notices sent; losing it costs a retry.  One queue command runs at a time per workdir (`queue.lock`), and
one that only reads (`status`, `review` without N) holds it only while it clones and fetches, not while it reads labels.  The
clone authenticates as the caller's checkout does (`credential_args`): a cloud VM sets its credential helper in the checkout's
own config, which a clone elsewhere never reads.  Git never prompts, so a remote the queue can't read fails the command instead
of answering from what the clone fetched last.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import contextlib
import dataclasses
import difflib
import fcntl
import fnmatch
import functools
import hashlib
import importlib.util
import json
import math
import os
import posixpath
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
import weakref
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Collection, Iterable

from . import linear, quick_tier, waitcap
from .gates import git_env
from .store import ids, vocab

BRANCH, SHADOW_BRANCH = "next", "queue-shadow"
RULES = "ci/queue.toml"
#: where the rules are until the repository move puts `tools/verity/check/` at `ci/`
RULES_BEFORE_MOVE = "tools/verity/check/queue.toml"
#: drive's flags of record (`DriveFlags`, `--config`)
DRIVE_FLAGS = "tools/verity/check/drive.toml"
#: the pins tool, whose `PINS` names the generated files a merge rebuilds (`repinned`): after the repository move, and before it
PINS_FILES = ("ci/pins.py", "tools/verity/check/pins.py")
#: the codemod that puts the in-tree Lake packages' Lean on Lean's module system, and the one reading of a Lean header it reads
#: with: a merge into a tree on the module system runs the codemod over the merged tree (`ported`)
MODULES, HEADERS = "tools/verity/lean/modules.py", "tools/verity/lean/headers.py"
#: the root modules of Security's `Definitions` library and Flock's `Flock`, which the port makes modules: a tree holding
#: the codemod is on the module system when each is a module (`on_modules`)
PORTED_ROOTS = ("Security/Definitions.lean", "verity/protocols/flock/verifier/Flock.lean")
#: what the codemod's run over a tree reads besides the codemod and its header reader: a merge that changes none of these onto
#: a tree on the module system leaves its porting as it was (`ported`'s ``base``); pathspecs, each `*` crossing directories
PORT_INPUTS = ("*.lean", "*lakefile.toml", "*lakefile.lean", "*lake-manifest.json", "*lean-toolchain")
MARK_TIME_RE = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")
#: drive.toml's `after_landing_shards` for the shard planner's N: "auto", or "auto:MAX"
AUTO_SHARDS_RE = re.compile(r"auto(:[1-9]\d*)?")
IDENTITY = {"GIT_AUTHOR_NAME": "research queue", "GIT_AUTHOR_EMAIL": "queue@research.invalid",
            "GIT_COMMITTER_NAME": "research queue", "GIT_COMMITTER_EMAIL": "queue@research.invalid"}
DATES = ("GIT_AUTHOR_DATE", "GIT_COMMITTER_DATE")
#: a transfer from the remote under 1 KB/s for 2 min aborts: git has no timeout, and a stalled HTTPS fetch would hold a drive tick
#: for minutes, the next tick retrying it
STALL = ("-c", "http.lowSpeedLimit=1000", "-c", "http.lowSpeedTime=120")
#: `gh api`'s limit: gh has no timeout, and a listing of the open PRs can hold a drive tick on a dead connection for
#: minutes while the same call answers in a second from a fresh one
GH_TIMEOUT_S = 120
#: the pages of open PRs `GitHub.open_prs` asks for at once: `gh api --paginate` reads ~10 pages of 100 one after another,
#: seconds of every `research queue status`
GH_PAGES = 10
GH_PER_PAGE = 100
#: the lane's handle ends on a letter or digit, so a sentence's period (`Lane NAME.`) isn't part of it
LANE_RE = re.compile(r"^[ \t>*_]*Lane[:\s]+`?([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)`?", re.M)
REGISTRY = Path(".agents/skills/using-slack/registry.json")
AGENT_RE = re.compile(r"\bbc-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
TRAILER_RE = re.compile(r"^(PR|Head|Owner): (.+)$", re.M)
#: a move PR's declaration in its body (`PR.move`): the word map, under `tools/move/`, that `rename.py --map` runs
MOVE_RE = re.compile(r"^[ \t>*_]*Move[:\s]+`?(tools/move/[\w.-]+\.toml)`?[ \t.]*$", re.M)
#: a Linear issue's key, as `open --issue KEY` takes it
ISSUE_KEY_RE = re.compile(r"[A-Z][A-Z0-9]*-\d+")
#: the line `open --issue KEY` writes: `Refs` is one of Linear's non-closing magic words, so the PR links to the issue and its
#: merge doesn't move the issue to done
ISSUE_LINE = "Refs {key}"
#: the line `open --issue none:REASON` writes, and what `named_issue` reads as a body that names no issue on purpose
NO_ISSUE_LINE = "Issue: none: {reason}"
NO_ISSUE_RE = re.compile(r"^[ \t>*_]*Issue:[ \t]*none:[ \t]*(\S.*?)[ \t]*$", re.M)
#: the line naming a goal of the Goals page (`research.goals`) the PR serves: `open --issue KEY` writes one after `Refs KEY` for
#: each live goal KEY's labels name (`with_goals`), and a ready mark warns without one naming a live goal, or `NO_ISSUE_LINE`
#: (`goal_gate`)
GOAL_LINE = "Goal {id}"
GOAL_RE = re.compile(r"^[ \t>*_]*Goal:?[ \t]+`?([A-Za-z0-9](?:[\w./-]*[A-Za-z0-9])?)`?[ \t.]*$", re.M)
#: the Goals page's goals (id and status) as this machine last read them, which the goal gate reads when the page can't be read
GOALS_CACHE = Path("~/.research/cache/goals.json")
#: a key after any of Linear's magic words (closing, non-closing or relation), whatever its team
ISSUE_LINK_RE = re.compile(r"(?i)\b(?:refs?|references|part of|contributes to|towards?|relates to|related to|close[sd]?|closing|"
                           r"fix(?:e[sd])?|fixing|resolve[sd]?|resolving|complete[sd]?|completing|implement(?:s|ed|ing)?)"
                           r"[ \t:]+(?-i:([A-Z][A-Z0-9]*-\d+))\b")
#: the trailer of a move drive regenerated (`Queue._regenerate`), naming its map
MOVE_TRAILER_RE = re.compile(r"^Move: (tools/move/[\w.-]+\.toml)$", re.M)
#: the scripts a move runs, each as the landing branch has it (`move_scripts`): the move, its certificate, and carrying a branch
#: across it
RENAME, MOVE_CHECK, RESTACK = "tools/move/rename.py", "tools/verity/check/move_check.py", "tools/move/restack.py"
#: what those scripts import from their own checkout: `tools/move/`'s other scripts (rename.py's `layout`), and the pins list
#: move_check.py reads to tell a pin's rewritten digest from an edit
MOVE_DIR, MOVE_PINS = "tools/move/", "tools/verity/check/pins.py"
#: how long a restack job runs at most (`Queue._launch_restack`): `timeout` interrupts `restack.py` then, so it removes its
#: scratch worktrees, and kills it a minute on; a job with no exit written `RESTACK_LOST_S` after its start died (its process
#: gone, or its pid another's after a restart of the pod)
RESTACK_WAIT_S = 2 * 3600
RESTACK_LOST_S = RESTACK_WAIT_S + 300
#: a restack job: the rest of its arguments (`restack.py`'s copy and its own) with `--json`'s results in `RESTACK_OUT` in the
#: job's directory ($1)
RESTACK_OUT = "restack.json"
RESTACK_JOB = f'd=$1; shift; exec timeout -s INT -k 60 {RESTACK_WAIT_S} "$@" > "$d/{RESTACK_OUT}"'
#: the exits of a restack job something else interrupted (`restack_end`): SIGINT's and SIGTERM's, 128 + the signal as `timeout`
#: passes its command's on, which an operator, drive's stop or the pod's shutdown sends.  Not `timeout`'s own 124 at
#: `RESTACK_WAIT_S`, nor SIGKILL's 137, which the OOM killer and `timeout -k` send too
RESTACK_INTERRUPTED = ("130", "143")
#: why a PR waits off `next` while it is restacked in the background (`Restacking`): it refuses nothing, so nothing caches it and
#: drive tells no one (`drive.Driver.tell_out`)
RESTACKING = "restacking across the move"
#: how many first-parent commits of the landing branch a sync reads for the queue's landings (`queue_log`): the heads it
#: landed, and the moves a PR based before them is restacked through
LANDED_WINDOW = 2000
SLUG_RE = re.compile(r"github\.com[:/]+([^/]+/[^/]+?)(?:\.git)?/?$")
#: OWNER/NAME on GitHub, as `--repo` takes it
REPO_ARG_RE = re.compile(r"[\w.-]+/[\w.-]+")
#: the git config key the queue sets to `true` in each clone it makes (`Queue._clone`): only such a clone, or one at `WORKDIR`, has
#: its origin set to the queue's remote
CLONE_MARK = "research.queueclone"
#: the queue's clone with no --repo or --workdir: one there without `CLONE_MARK` is one the queue made before it set it (drive's)
WORKDIR = Path.home() / ".research" / "queue" / "verity"
AUTH_FAILED_RE = re.compile(r"could not read (Username|Password)|Authentication failed|unable to get password")
#: the repository the queue serves, `GitHub.repo` by default: a PR whose head is on another one is never marked or admitted
REPO = "compute-verification/verity"
#: the `author_association`s GitHub gives the organization's own people: a collaborator's (COLLABORATOR) is not one of them
MEMBER_ASSOCIATIONS = ("OWNER", "MEMBER")
#: the authors marked and admitted beside those: the Apps that open and land the team's PRs, and the repository's admin, whom
#: GitHub reports as a collaborator
AUTHORS = ("verity-agents[bot]", "verity-drive[bot]", "danielreuter")


class QueueError(Exception):
    pass


class Restacking(Exception):
    """A head a job in the background is carrying across a move (`Queue._restack`): its PR waits off `next` this sync, for
    this reason (`RESTACKING`)."""


def _git(repo: Path, *args: str, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    full = git_env(GIT_TERMINAL_PROMPT="0", **(env or {}))
    if env is IDENTITY:                     # the queue's merges carry the time they're made: a date a caller pins stops `log --since`
        for k in DATES:
            full.pop(k, None)
    with _worktrees_shared(repo, full) if _subcommand(args) == "fetch" else contextlib.nullcontext():
        r = subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, capture_output=True, text=True,
                           errors="surrogateescape", env=full)
    if check and r.returncode:
        raise QueueError(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()[-400:]}")
    return r


def _subcommand(args: tuple[str, ...]) -> str:
    """The git subcommand of ``args``: their first word past the `-c NAME=VALUE` settings."""
    i = 0
    while args[i:i + 1] == ("-c",):
        i += 2
    return args[i] if i < len(args) else ""


#: the clone's worktree lock, in its git directory: `tools/move/restack.py` adds and forgets each scratch worktree holding it
#: alone, because `git worktree add` writes a placeholder HEAD before the real one, and a fetch, whose connectivity check
#: reads every worktree's HEAD, holds it shared
WORKTREES_LOCK = "research-worktrees.lock"


@contextlib.contextmanager
def _worktrees_shared(repo: Path, env: dict[str, str]):
    common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=repo, capture_output=True, text=True,
                            env=env).stdout.strip()
    if not common:
        yield
        return
    with open(Path(repo, common) / WORKTREES_LOCK, "a") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        yield


def _out(repo: Path, *args: str) -> str:
    return _git(repo, *args).stdout.strip()


def on_first_parents(repo: Path, c: str, tip: str) -> bool:
    """Whether the commit ``c`` (full) is on ``tip``'s first-parent chain: one the branch at ``tip`` was at, not one only merged
    into it."""
    r = _git(repo, "rev-list", "--first-parent", "--parents", f"{c}..{tip}", check=False)
    if r.returncode:
        return False
    if lines := r.stdout.strip().splitlines():
        return lines[-1].split()[1:2] == [c]
    return _git(repo, "rev-parse", "--verify", "-q", f"{tip}^{{commit}}", check=False).stdout.strip() == c


def credential_args(repo: Path, remote: str) -> tuple[str, ...]:
    """`git -c` options that let the queue's own clone authenticate to `remote` as the checkout `repo` does: the `credential.*`
    settings its own config has for the remote (a cloud VM's helper lives there), else `gh auth git-credential` when no helper
    applies to the remote at all and `gh` is on PATH.  None when git's global config already serves it, or the remote isn't http."""
    if not re.match(r"https?://", remote):
        return ()
    found = _git(repo, "config", "--local", "--get-urlmatch", "credential", remote, check=False).stdout.splitlines()
    conf = dict(line.split(" ", 1) for line in found if " " in line)
    if not conf.get("credential.helper"):
        if _git(repo, "config", "--get-urlmatch", "credential.helper", remote, check=False).stdout.strip() or not (gh := shutil.which("gh")):
            return ()
        conf = {"credential.helper": f"!{shlex.quote(gh)} auth git-credential"}
    return ("-c", "credential.helper=", *(o for k, v in conf.items() for o in ("-c", f"{k}={v}")))


def redacted(url: str) -> str:
    """``url`` without the user and password it may carry, so a token in a remote's URL never reaches a message."""
    return re.sub(r"(?<=://)[^/@]*@", "", url)


def slug(url: str) -> str | None:
    """The OWNER/NAME a GitHub URL names (https or ssh, with or without `.git`, whatever user it carries), else None."""
    return (m := SLUG_RE.search(url)) and m.group(1)


def same_repository(a: str, b: str) -> bool:
    """Whether two remote URLs name one repository: on GitHub by OWNER/NAME in any case (`slug`), elsewhere (a path) by the URL
    itself."""
    sa, sb = slug(a), slug(b)
    return (sa or "").lower() == (sb or "").lower() if sa or sb else a.rstrip("/") == b.rstrip("/")


def remote_and_repo(remote: str | None, repo: str | None, cwd: Path) -> tuple[str, str]:
    """The remote the queue clones and fetches, and its OWNER/NAME on GitHub, from `--remote` and `--repo`: `--repo` alone takes
    `https://github.com/OWNER/NAME.git`, and with neither the remote is ``cwd``'s origin.  Refused when `--remote` names
    another repository on GitHub than `--repo` (the queue would read one's PRs against the other's pull refs), and when no
    OWNER/NAME can be told."""
    if repo is not None and not REPO_ARG_RE.fullmatch(repo):
        raise QueueError(f"--repo {repo!r}: OWNER/NAME on GitHub, as {REPO}")
    if repo and remote is None:
        return f"https://github.com/{repo}.git", repo
    remote = remote or _out(cwd, "remote", "get-url", "origin")
    named = slug(remote)
    if repo and named and named.lower() != repo.lower():
        raise QueueError(f"--remote {redacted(remote)} is {named} on GitHub, not --repo {repo}: pass --repo alone (its remote is "
                         f"https://github.com/{repo}.git), or a --remote of {repo}")
    if not (repo := repo or named):
        raise QueueError(f"can't tell OWNER/NAME from {redacted(remote)!r}: pass --repo")
    return remote, repo


# ---- the open PRs -------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PR:
    number: int
    head: str
    title: str = ""
    body: str = ""
    draft: bool = False
    labels: tuple[str, ...] = ()
    base: str = "main"
    branch: str = ""
    #: its head branch is another repository's (or one GitHub no longer has), never this one's to delete
    fork: bool = False
    #: GitHub's `head.repo.full_name`: None when the head's repository was deleted
    head_repo: str | None = REPO
    author: str = ""
    association: str = "MEMBER"

    @classmethod
    def from_api(cls, d: dict) -> "PR":
        repo = lambda side: ((d.get(side) or {}).get("repo") or {}).get("full_name")  # noqa: E731
        return cls(number=int(d["number"]), head=d["head"]["sha"], title=d.get("title") or "", body=d.get("body") or "",
                   draft=bool(d.get("draft")), labels=tuple(sorted(x["name"] for x in d.get("labels") or [])),
                   base=(d.get("base") or {}).get("ref", "main"), branch=d["head"].get("ref") or "",
                   fork=repo("head") != repo("base"), head_repo=repo("head"), author=(d.get("user") or {}).get("login") or "",
                   association=d.get("author_association") or "")

    def outside(self, repo: str) -> str | None:
        """Why the PR is never marked ready (`ready`, `judge`) nor admitted (`Queue.entries`), or None: its head is on another
        repository than ``repo`` (a fork's, or one deleted since), or its author is neither one of the organization's owners
        and members (`MEMBER_ASSOCIATIONS`) nor one of `AUTHORS`."""
        fix = "a maintainer opens it from a branch of this repository"
        if self.head_repo != repo:
            return f"its head is on {self.head_repo or 'a deleted repository'}, not {repo}: {fix}"
        if self.association not in MEMBER_ASSOCIATIONS and self.author not in AUTHORS:
            return (f"its author {self.author or '(none)'} is no owner or member of the organization "
                    f"({self.association or 'no association'}) and none of {', '.join(AUTHORS)}: {fix}")
        return None

    @property
    def owner(self) -> str | None:
        """The lane a `Lane {name}` line names, else the Cursor agent id the body carries (its PR footer), else None."""
        if m := LANE_RE.search(self.body):
            return m.group(1)
        if m := AGENT_RE.search(self.body):
            return m.group(0)
        return None

    @property
    def move(self) -> str | None:
        """The word map a `Move: tools/move/NAME.toml` line of its body names: a move PR, whose head is only the request, which
        the sync regenerates on the tip below it (`Queue._regenerate`)."""
        m = MOVE_RE.search(self.body)
        return m.group(1) if m else None


class GitHub:
    """The open PRs of a repository, through `gh api` (the cloud VMs' authenticated GitHub CLI)."""

    def __init__(self, repo: str):
        self.repo = repo

    def open_prs(self) -> list[PR]:
        """The open PRs, newest first: `GH_PAGES` pages at a time until one isn't full (a PR on two pages, as the listing
        shifted between them, counts once)."""
        out: dict[int, PR] = {}
        with concurrent.futures.ThreadPoolExecutor(GH_PAGES) as ex:
            first = 1
            while True:
                pages = list(ex.map(self._page, range(first, first + GH_PAGES)))
                for page in pages:
                    for d in page:
                        out.setdefault(int(d["number"]), PR.from_api(d))
                    if len(page) < GH_PER_PAGE:
                        return list(out.values())
                first += GH_PAGES

    def _page(self, n: int) -> list[dict]:
        try:
            r = subprocess.run(["gh", "api", f"repos/{self.repo}/pulls?state=open&per_page={GH_PER_PAGE}&page={n}"],
                               capture_output=True, text=True, timeout=GH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise QueueError(f"gh api repos/{self.repo}/pulls: no answer in {GH_TIMEOUT_S} s") from None
        if r.returncode:
            raise QueueError(f"gh api repos/{self.repo}/pulls: {(r.stderr or r.stdout).strip()[-300:]}")
        page = json.loads(r.stdout)
        return page if isinstance(page, list) else [page]

    def branch_tip(self, branch: str) -> str | None:
        """``branch``'s commit as git has it now (the refs API, which a push moves at once; a PR's head lags it by seconds), or
        None when it can't be read (a fork's branch, no answer)."""
        try:
            r = subprocess.run(["gh", "api", f"repos/{self.repo}/git/ref/heads/{branch}", "--jq", ".object.sha"], capture_output=True,
                               text=True, timeout=GH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return None
        return r.stdout.strip() or None if r.returncode == 0 else None

    def close(self, number: int, comment: str) -> None:
        """Close PR ``number`` with ``comment`` (drive's stale and unlandable rules: `research.stale`, `drive.Driver._unlandable`),
        as `gh` is authenticated here: drive's App."""
        try:
            r = subprocess.run(["gh", "pr", "close", str(number), "--repo", self.repo, "--comment", comment], capture_output=True,
                               text=True, timeout=GH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise QueueError(f"gh pr close {number}: no answer in {GH_TIMEOUT_S} s") from None
        if r.returncode:
            raise QueueError(f"gh pr close {number}: {(r.stderr or r.stdout).strip()[-300:]}")


def landing_waiter(repo: str | None = None, store_root: str | None = None) -> Callable[[list[int]], bool]:
    """Whether PRs wait only on landing: each is open, not held (the rules' hold label) and marked ready at its current head
    (`StoreLabels.ready`), as the Linear audit asks of an issue In Review.  The open PRs are read once, on the first call;
    `repo` defaults to the checkout's origin, and a PR GitHub or the store can't answer for doesn't wait.  When the open PRs
    can't be read at all (no checkout, gh failing), nothing waits: the audit then counts the issue as it would without this."""
    from .store.local import LocalStore

    state: dict = {}

    def waiting(numbers: list[int]) -> bool:
        if "heads" not in state:
            try:
                slug = repo or ((m := SLUG_RE.search(_out(Path.cwd(), "remote", "get-url", "origin"))) and m.group(1))
                hold = Rules().hold_label
                state["heads"] = {pr.number: pr.head for pr in GitHub(slug).open_prs() if hold not in pr.labels}
                state["labels"] = StoreLabels(LocalStore(store_root), log=lambda s: None)
                state["labels"].prefetch(state["heads"].items())
            except (QueueError, OSError) as e:
                print(f"linear audit: can't read the open PRs ({e}), so no issue counts as waiting on landing",
                      file=sys.stderr)
                state["heads"], state["labels"] = {}, None
        heads, labels = state["heads"], state["labels"]
        return bool(numbers) and all(n in heads and labels.ready(n, heads[n]) for n in numbers)

    return waiting


# ---- rules and labels ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Rules:
    hold_label: str = "hold"
    source: str = RULES
    #: ready marks older than this don't count (`--marked-after` overrides it), so every VM's queue admits what drive admits
    marked_after: str | None = None

    @classmethod
    def parse(cls, text: str, source: str = RULES) -> "Rules":
        doc = tomllib.loads(text)
        unknown = set(doc) - {"hold_label", "marked_after"}
        if unknown:
            raise QueueError(f"{source}: unknown keys {sorted(unknown)}")
        cut = doc.get("marked_after")
        if cut is not None and not (isinstance(cut, str) and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", cut)):
            raise QueueError(f"{source}: marked_after {cut!r}: a UTC time as YYYY-MM-DDTHH:MM:SSZ, as labels record it")
        return cls(doc.get("hold_label", "hold"), source, cut)


#: drive's ways of landing (`DriveFlags.land`): `check`, which landed trains on their own check, was deleted with
#: its code
LANDINGS = ("review",)
#: a landing or warn gate's path (`DriveFlags.landing_gates`, `DriveFlags.warn_gates`)
GATE_PATH_RE = re.compile(r"[\w.-]+(?:/[\w.-]+)*\.py")
#: the steps of a check (`tools/verity/check/check.py`), of which `DriveFlags.warn_steps` names some
CHECK_STEPS = ("preflight-lock", "preflight-lints", "preflight-fixtures", "lean-changed", "flock-circuit-build", "lean-build",
               "lean-unit-cut", "lean-agreement", "lean-audit", "lean-suites", "pytest", "circuit-check", "flock-rust-tests",
               "ratchet-gate")
#: an entry of `DriveFlags.lowrisk_excludes`: a directory from the repository's root ending in `/`, or a file name glob
LOWRISK_EXCLUDE_RE = re.compile(r"(?:[\w.-]+/)+|[\w.*?\[\]-]+")
#: an entry of `DriveFlags.bottom_ring`: a handle as a post that starts with it rings it (`slack.NAME_TOKEN_RE`)
HANDLE_RE = re.compile(r"@[a-z0-9][a-z0-9._-]*")
#: drive.toml's `executor` (off, shadow beside the after-landing run, or on in its place) and `verdict_carry` (off, shadow logged
#: only, or on written)
MODES = ("off", "shadow", "on")
#: drive.toml's `close_unlandable`: off, log (each close it would make, only logged) or on
CLOSE_UNLANDABLE = ("off", "log", "on")
#: the keys a flag of drive's command line can set (`DriveFlags.override`), each as `--KEY` with `_` as `-`
COMMAND_LINE = ("on", "marked_after", "regenerate_locks", "after_landing_no_cache")
#: drive.toml keys whose code was deleted: a drive deployed ahead of `main` (`drive.sh deploy SHA`) still reads them in
#: `main`'s drive.toml, so even a strict parse ignores them, with a note, rather than refuse every tick
DELETED_KEYS = ("max_lean_checks", "nonlean_first", "reviewed_first")
#: what every command says to a `--first`, refusing it: the landing order is drive's priority list (`Queue.priority`) alone
FIRST_RETIRED = ("--first is retired: drive's priority file (--priority-file) holds the landing order; list the PRs there, "
                 "first to land first")
#: what `DriveFlags.signed_verdicts` keeps off the low-risk lane: who holds each handle, and the code that judges a label
SIGNED_GUARDS = (str(REGISTRY), "tools/research/src/research/labelauth.py", "tools/research/src/research/queue.py",
                 "tools/research/src/research/drive.py")


@dataclass(frozen=True)
class DriveFlags:
    """`research queue drive`'s flags: the defaults, then `DRIVE_FLAGS` at `main`, read each tick (`drive_flags`: its one source,
    so a change to them is a reviewed commit and the next tick takes it), then each flag the command line gives of those
    `command_line` lets it set (`override`)."""
    on: str | None = None
    marked_after: str | None = None
    regenerate_locks: bool = False
    #: how `next` lands: "review", each member on a reviewer's verdict, onto `land_into`, whose tip the full check checks (`Driver`)
    land: str = "review"
    #: the branch a review landing fast-forwards and `next` is built on (`Queue.into`): `checked`, which `main` follows only to
    #: a commit whose after-landing run passed every step on the merge gate (`Driver._promote`), or `main` itself; None (unset):
    #: `main` (`into_at`)
    land_into: str | None = None
    #: the after-landing run reuses no pass but circuit-check's per-target ones (`check.py --no-cache --circuit-cache`)
    after_landing_no_cache: bool = False
    #: scripts a PR passes to be admitted (`gate_refusal`: its head at the ready mark's preflight, its merge onto `next` at the
    #: sync), each by its path from the repository's root, run as it is at `main`; drive's landing runs none
    landing_gates: tuple[str, ...] = ()
    #: scripts run where and as the landing gates are, whose refusal keeps no PR out: the ready mark prints it as a warning, and
    #: the sync admits the PR and keeps it for drive to tell its owner once a head (`Driver.tell_warned`)
    warn_gates: tuple[str, ...] = ()
    #: the check's steps whose red drive never reverts: it names the culprit and rings its lane and the owner of what failed
    #: once, to fix it forward in its next PR (`drive.Driver._warns`)
    warn_steps: tuple[str, ...] = ()
    #: every sync, members with a land verdict go below those waiting for one (`Driver._lift`), so a member waiting for its
    #: review holds only itself
    verdict_first: bool = True
    #: days an open PR unmarked at its head stays idle before it is stale (`research.stale`: `status` says so, drive rings its
    #: owner once a day), and before drive closes it; 0 is off
    stale_days: int = 0
    stale_close_days: int = 0
    #: PRs the stale rule never judges: their owners keep them open with a reason
    stale_keep: tuple[int, ...] = ()
    #: close each open PR nothing will land, its head carried by a landed PR of the same owner or its change already on
    #: `land_into` (`drive.Driver._unlandable`): "on"; "log", only log each it would close; "off"
    close_unlandable: str = "off"
    #: the after-landing run as N runs at once, shard K of ci.toml's plan with N shards each (`drive.Driver.start_postland`):
    #: 1, or a plan's N; it takes about its longest shard rather than every step on one slot; "auto" or "auto:MAX", the shard
    #: planner's N (`tools/verity/check/shards.py`), at most MAX (default its MAX_SHARDS), each shard `--shard auto:K/N`
    after_landing_shards: int | str = 1
    #: the keys drive's command line may override (`override`), of `COMMAND_LINE`; any other flag given there refuses the tick
    command_line: tuple[str, ...] = ("on",)
    #: each tick, after the reviewed prefix, land up to `lowrisk_per_tick` members of `next` whose change is `lowrisk` on their
    #: ready mark alone (`drive.Driver.land_lowrisk`); off: drive lands only on review
    lowrisk_lane: bool = True
    lowrisk_per_tick: int = 20
    #: what keeps a change off the low-risk lane (`lowrisk`): a directory from the root (ending in `/`), or a file name glob;
    #: none, the lane lands nothing
    lowrisk_excludes: tuple[str, ...] = ()
    #: the task executor after landing (`research.executor`, `drive.Driver`): "off"; "shadow", each tip's task graph enqueued beside
    #: the after-landing run and their verdicts compared (`research queue shadow-report`); "on", its verdicts decide, with a run
    #: of `drive.SLIM_STEPS` alone beside it
    executor: str = "off"
    #: the task executor's enqueue command, the words of one argv run with no shell (`research.executor.QueueTransport`): it
    #: puts `--priority N --check after-landing:SHA` on the executor's queue; none, each tip "shadow" or "on" enqueues fails, told
    executor_enqueue: tuple[str, ...] = ()
    #: a land verdict at an older head of a member's PR carries to its head when the commits since only merge trusted ones
    #: (`research.carry`, `drive.Driver._carry_verdicts`): "off"; "shadow", drive logs what it would carry and writes nothing;
    #: "on", it writes the land verdict at the head as `CI`
    verdict_carry: str = "off"
    #: who drive rings while `next`'s lowest member waits at its head for a land verdict, holding every landing above it
    #: (`drive.Driver._bottom`), and the minutes it waits before each ring
    bottom_ring: tuple[str, ...] = ("@ci", "@top")
    bottom_wait_min: int = 30
    #: count a ready mark or land verdict only when its `label_auth` companion authenticates it as its author's, or as an agent's
    #: its author delegated to (`research.labelauth`, `StoreLabels`); `ready`, `judge` and `review` then refuse to write one they
    #: can't sign.  On only while the low-risk lane can't change who holds a handle or how a label is judged (`SIGNED_GUARDS`)
    signed_verdicts: bool = False
    #: the restacks across a move the syncs run at once in the background, each `restack.py` on one head (`Queue._restack`)
    restack_jobs: int = 3

    @classmethod
    def parse(cls, text: str, source: str = "drive.toml", strict: bool = False) -> "DriveFlags":
        """``text``'s flags. ``strict`` (drive loading its own flags): any key this code doesn't read refuses, so a drive that
        lags `main` stops rather than run without a guard `main` added. Otherwise (a client reading `main`'s flags) such a
        key is ignored with a note, so a client older than `main` keeps working; `tests/test_drive_toml.py` holds `main`'s
        drive.toml to the strict parse, which is where a typo is caught."""
        try:
            doc = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise QueueError(f"{source}: not TOML: {e}") from None
        keys = [f.name for f in dataclasses.fields(cls)]
        for k in list(doc):
            if k not in keys:
                near = difflib.get_close_matches(k, keys, n=1)
                if strict and k not in DELETED_KEYS:
                    raise QueueError(f"{source}: unknown key {k!r}{f' (did you mean {near[0]!r}?)' if near else ''}: drive "
                                     f"reads only {', '.join(keys)}; rename or remove it, or update this research")
                print(f"{source}: ignoring key {k!r}, which this research doesn't read "
                      + ("(its code was deleted: remove it)" if k in DELETED_KEYS else "(a newer one may)"), file=sys.stderr)
                del doc[k]

        def bad(k: str, want: str) -> QueueError:
            return QueueError(f"{source}: {k} = {doc[k]!r}: {want}")

        def names(k: str) -> tuple[str, ...]:
            v = doc[k]
            if not (isinstance(v, list) and all(isinstance(x, str) and re.fullmatch(r"[\w.-]+", x) for x in v)):
                raise bad(k, "a list of names, as [\"a\", \"b\"]")
            return tuple(v)

        got: dict = {}
        if "on" in doc:
            if not (on := names("on")):
                raise bad("on", "at least one machine (train.sh's: a machine, or `auto`)")
            got["on"] = ",".join(on)
        if "after_landing_shards" in doc:
            v = doc["after_landing_shards"]
            if not (isinstance(v, str) and AUTO_SHARDS_RE.fullmatch(v)) and (type(v) is not int or v < 1):
                raise bad("after_landing_shards", "a whole number, 1 or a shard plan's N in tools/verity/check/ci.toml, or "
                                                  "\"auto\" or \"auto:MAX\" (the shard planner's N, at most MAX)")
            got["after_landing_shards"] = v
        for k, modes in (("executor", MODES), ("verdict_carry", MODES), ("close_unlandable", CLOSE_UNLANDABLE)):
            if k in doc:
                if doc[k] not in modes:
                    raise bad(k, f"one of {', '.join(modes)}")
                got[k] = doc[k]
        if "executor_enqueue" in doc:
            v = doc["executor_enqueue"]
            if not (isinstance(v, list) and v and all(isinstance(x, str) and x for x in v)):
                raise bad("executor_enqueue", "a command as a non-empty list of its words, as [\"python3\", \"enqueue.py\"]")
            got["executor_enqueue"] = tuple(v)
        if "marked_after" in doc:
            if not (isinstance(doc["marked_after"], str) and MARK_TIME_RE.fullmatch(doc["marked_after"])):
                raise bad("marked_after", "a UTC time as YYYY-MM-DDTHH:MM:SSZ, as labels record it")
            got["marked_after"] = doc["marked_after"]
        if "lowrisk_per_tick" in doc:
            if type(doc["lowrisk_per_tick"]) is not int or doc["lowrisk_per_tick"] < 1:
                raise bad("lowrisk_per_tick", "a whole number, 1 or more (`lowrisk_lane = false` turns the lane off)")
            got["lowrisk_per_tick"] = doc["lowrisk_per_tick"]
        if "restack_jobs" in doc:
            if type(doc["restack_jobs"]) is not int or doc["restack_jobs"] < 1:
                raise bad("restack_jobs", "a whole number, 1 or more")
            got["restack_jobs"] = doc["restack_jobs"]
        if "lowrisk_excludes" in doc:
            v = doc["lowrisk_excludes"]
            if not (isinstance(v, list) and v and all(isinstance(x, str) and LOWRISK_EXCLUDE_RE.fullmatch(x)
                                                         and ".." not in x.split("/") for x in v)):
                raise bad("lowrisk_excludes", "a non-empty list of directories from the repository's root, each ending in `/`, "
                                              "and file name globs, as [\"*.lean\", \"verity/core/\"]")
            got["lowrisk_excludes"] = tuple(v)
        if "bottom_ring" in doc:
            v = doc["bottom_ring"]
            if not (isinstance(v, list) and v and all(isinstance(x, str) and HANDLE_RE.fullmatch(x) for x in v)):
                raise bad("bottom_ring", "a non-empty list of handles, each `@` and its lower-case name, as [\"@ci\"]")
            got["bottom_ring"] = tuple(v)
        if "bottom_wait_min" in doc:
            if type(doc["bottom_wait_min"]) is not int or doc["bottom_wait_min"] < 1:
                raise bad("bottom_wait_min", "a whole number of minutes, 1 or more")
            got["bottom_wait_min"] = doc["bottom_wait_min"]
        for k in ("regenerate_locks", "verdict_first", "lowrisk_lane", "signed_verdicts"):
            if k in doc:
                if not isinstance(doc[k], bool):
                    raise bad(k, "true or false")
                got[k] = doc[k]
        if "land" in doc:
            if doc["land"] == "check":
                raise bad("land", "train landing was deleted with its code: drive lands on review only; set "
                                  "`land = \"review\"` or remove the key")
            if doc["land"] not in LANDINGS:
                raise bad("land", f"one of {', '.join(LANDINGS)}")
            got["land"] = doc["land"]
        if "land_into" in doc:
            from .merge import CHECKED

            if doc["land_into"] not in ("main", CHECKED):
                raise bad("land_into", f"\"{CHECKED}\" (main follows it to fully checked commits) or \"main\"")
            got["land_into"] = doc["land_into"]
        if "after_landing_no_cache" in doc:
            if not isinstance(doc["after_landing_no_cache"], bool):
                raise bad("after_landing_no_cache", "true or false")
            got["after_landing_no_cache"] = doc["after_landing_no_cache"]
        for k in ("landing_gates", "warn_gates"):
            if k in doc:
                v = doc[k]
                if not (isinstance(v, list) and all(isinstance(x, str) and GATE_PATH_RE.fullmatch(x) and ".." not in x.split("/")
                                                    for x in v)):
                    raise bad(k, "a list of Python scripts by their path from the repository's root, as "
                                 "[\"adapters/x/tools/gate.py\"]")
                got[k] = tuple(v)
        if "warn_steps" in doc:
            if not (isinstance(doc["warn_steps"], list) and all(x in CHECK_STEPS for x in doc["warn_steps"])):
                raise bad("warn_steps", f"a list of the check's steps, of {', '.join(CHECK_STEPS)}")
            got["warn_steps"] = tuple(doc["warn_steps"])
        for k in ("stale_days", "stale_close_days"):
            if k in doc:
                if type(doc[k]) is not int or doc[k] < 0:
                    raise bad(k, "a whole number of days, 0 for off")
                got[k] = doc[k]
        if "stale_keep" in doc:
            if not (isinstance(doc["stale_keep"], list) and all(type(n) is int and n > 0 for n in doc["stale_keep"])):
                raise bad("stale_keep", "a list of PR numbers, as [2692, 2693]")
            got["stale_keep"] = tuple(doc["stale_keep"])
        if "command_line" in doc:
            v = doc["command_line"]
            if not (isinstance(v, list) and all(x in COMMAND_LINE for x in v)):
                raise bad("command_line", f"a list of the keys a command-line flag sets, of {', '.join(COMMAND_LINE)}")
            got["command_line"] = tuple(v)
        if got.get("stale_close_days") and not 0 < got.get("stale_days", 0) <= got["stale_close_days"]:
            raise bad("stale_close_days", "0, or at least stale_days, which must be on: drive closes only a stale PR")
        if got.get("signed_verdicts") and got.get("lowrisk_lane", cls.lowrisk_lane) and \
                (open_ := [p for p in SIGNED_GUARDS if lowrisk([p], got.get("lowrisk_excludes", cls.lowrisk_excludes))]):
            raise bad("signed_verdicts", f"true lets a change to {' or '.join(open_)} land on its ready mark alone, so a lane "
                                         "could make itself another handle's holder or loosen the check: add each to "
                                         "`lowrisk_excludes`, or set `lowrisk_lane = false`")
        return cls(**got)

    def override(self, a: argparse.Namespace, where: str = DRIVE_FLAGS) -> "DriveFlags":
        """These flags with each one ``a`` (the command line) gives explicitly in its place; QueueError, naming each, when it
        gives one `command_line` (these flags', from ``where``) doesn't list."""
        given = {k: getattr(a, k) for k in COMMAND_LINE if getattr(a, k, None) is not None}
        if barred := [k for k in COMMAND_LINE if k in given and k not in self.command_line]:
            raise QueueError(f"{' and '.join('--' + k.replace('_', '-') for k in barred)}: {where} lets drive's command line set "
                             f"only {', '.join(self.command_line) or 'nothing'} (its `command_line`): commit the change to "
                             f"{DRIVE_FLAGS}, or drop the flag from drive's args")
        return dataclasses.replace(self, **given)

    def toml(self, k: str) -> str:
        """Key ``k``'s value as drive.toml writes it; "unset" for none."""
        v = getattr(self, k)
        if k == "on" and v is not None:
            v = v.split(",")
        if v is None:
            return "unset"
        if isinstance(v, bool):
            return "true" if v else "false"
        return json.dumps(list(v) if isinstance(v, tuple) else v)

    def line(self) -> str:
        return " ".join(f"{k}={','.join(map(str, v)) if isinstance(v, tuple) else v}"
                        for k, v in dataclasses.asdict(self).items())


def drive_flags(clone: Path, config: str | None = None, from_tree: bool = False) -> tuple[DriveFlags, str]:
    """Drive's flags: `DRIVE_FLAGS` at `main` in the queue's ``clone`` (fetched), parsed strictly, the defaults when `main` has
    none; with where they came from.  It is their one source: ``config`` (`--config PATH`: at `main`, or in this file system
    with ``from_tree``) is accepted only while every key it sets has `main`'s value, else QueueError names each key that
    differs, both values and the fix, so a file on drive's machine can't quietly override what `main` says."""
    sha = _out(clone, "rev-parse", "refs/remotes/origin/main")
    r = _git(clone, "show", f"{sha}:{DRIVE_FLAGS}", check=False)
    where = f"{DRIVE_FLAGS} at main {sha[:12]}"
    flags = DriveFlags.parse(r.stdout, where, strict=True) if r.returncode == 0 else DriveFlags()
    if r.returncode:
        where = f"the defaults ({DRIVE_FLAGS} isn't at main {sha[:12]})"
    if config is None:
        return flags, where
    if from_tree:
        try:
            text = Path(config).read_text()
        except OSError as e:
            raise QueueError(f"--config {config}: {e.strerror}") from None
    else:
        if Path(config).is_absolute() or config.startswith(".."):
            raise QueueError(f"--config {config}: a path from the repository's root, read at `main` (or --config-from-tree for "
                             "this file)")
        if (c := _git(clone, "show", f"{sha}:{config}", check=False)).returncode:
            raise QueueError(f"--config {config}: `main` ({sha[:12]}) has no such file: land it first, or pass --config-from-tree")
        text = c.stdout
    mine = DriveFlags.parse(text, f"--config {config}", strict=True)
    if differ := [k for k in tomllib.loads(text) if getattr(mine, k) != getattr(flags, k)]:
        raise QueueError(f"--config {config} differs from {where}, drive's one source of flags: "
                         + "; ".join(f"{k} = {mine.toml(k)} there, {flags.toml(k)} at main" for k in differ)
                         + f". Commit the change to {DRIVE_FLAGS}, or remove the key from {config}")
    return flags, where


def priority_list(path: Path) -> tuple[int, ...]:
    """Drive's priority list (`--priority-file`, `Queue.priority`), the landing order: the PR numbers ``path`` holds, separated
    by spaces or commas, in its order, first to land first, each once (a number repeated keeps its first place); empty when
    there is no such file.  QueueError, with the fix, when it can't be read or holds anything else."""
    try:
        if not path.exists():
            return ()
        text = path.read_text()
    except (OSError, UnicodeDecodeError) as e:
        raise QueueError(f"--priority-file {path} can't be read ({getattr(e, 'strerror', None) or e}): make it a text file of "
                         "PR numbers separated by spaces or commas, or remove it") from None
    words = [w for w in re.split(r"[\s,]+", text) if w]
    if bad := [w for w in words if not re.fullmatch(r"[1-9][0-9]*", w)]:
        raise QueueError(f"--priority-file {path} holds {', '.join(map(repr, bad[:5]))}, which "
                         f"{'is not a PR number' if len(bad) == 1 else 'are not PR numbers'}: write PR numbers separated by "
                         "spaces or commas")
    return tuple(dict.fromkeys(map(int, words)))


def drive_machines() -> tuple[set[str], dict[str, str]]:
    """The machines `research run --on` resolves (`remote.load_machines`: machines.toml and the pod registry's machines.d), as
    train.sh's runs reach them, and the names the registry refuses, with why."""
    from .pods import registry
    from .remote import load_machines

    return set(load_machines()), dict(registry.problems)


def drive_names(q: "Queue", flags: DriveFlags, push: bool = True) -> None:
    """Refuse a tick, with each fix, when a name drive's flags use doesn't exist (the queue's clone just fetched): the landing
    branch (`Queue.into`) on origin; with ``push``, that drive's credential may push it (`git push --dry-run` of the branch to
    itself: GitHub asks for write access to the repository before it answers, though a ruleset is judged only at a real push);
    each machine of `on` (`drive_machines`; `auto` is train.sh's); and each landing and warn gate's script at `main`."""
    main, missing = "refs/remotes/origin/main", []
    into = q.into
    if q._remote(q.workdir, "ls-remote", "--exit-code", "origin", f"refs/heads/{into}", ok=(0, 2)).returncode:
        missing.append(f"origin has no branch `{into}` (drive.toml's `land_into`): create it at `main` as drive's App (`git push "
                       f"origin refs/remotes/origin/main:refs/heads/{into}` in drive's checkout on the control pod), or "
                       f"commit `land_into = \"main\"` to {DRIVE_FLAGS}")
    elif push and (r := q._remote(q.workdir, "push", "--dry-run", "--no-verify", "--quiet", "origin",
                                  f"{q.base}:refs/heads/{into}", ok=None)).returncode:
        missing.append(f"drive's credential can't push `{into}` (`git push --dry-run`: {(r.stderr or r.stdout).strip()[-300:]}): "
                       "give drive's App contents write on the repository (infra: gh_app_token.sh's installation), or run "
                       "drive where its credential applies")
    if flags.on and (names := [m for m in flags.on.split(",") if m != "auto"]):
        try:
            known, refused = drive_machines()
        except (OSError, tomllib.TOMLDecodeError) as e:
            known, refused = set(), {m: f"the machine registry can't be read ({type(e).__name__}: {e})" for m in names}
        for m in names:
            if m not in known:
                missing.append(f"machine {m!r} of `on` doesn't resolve ({refused.get(m) or 'not in machines.toml or machines.d'}): "
                               f"register it (`research pods register {m} ...`), or remove it from `on` in {DRIVE_FLAGS} (or "
                               "drive's --on)")
    for key, kind in (("landing_gates", "landing gate"), ("warn_gates", "warn gate")):
        for g in getattr(flags, key):
            if _git(q.workdir, "cat-file", "-e", f"{main}:{g}", check=False).returncode:
                missing.append(f"{kind} {g} isn't at `main`: land it first, or remove it from `{key}` in {DRIVE_FLAGS}")
    if missing:
        raise QueueError("drive's flags name what doesn't exist, so this tick does nothing: " + "; ".join(missing))


#: how long a landing or warn gate may run before it counts as unable to judge (`run_gate`)
GATE_TIMEOUT_S = 600
#: the fix items of a gate's failure a refusal quotes
GATE_FIX_LINES = 15
#: the most gate verdicts the queue's state keeps for one `main` and one kind of gate (`Queue._gated`), the newest
GATE_VERDICTS = 1000
#: what the ready mark's preflight says after a warn gate's failure (`Preflight._gates`)
WARN_GATE_SAYS = "A warn gate refuses nothing: drive lands the PR and tells its owner, who fixes it in their next PR"
#: the most merges' records the queue's state keeps (`Queue._merge`), the most recently used: a few hundred bytes each, and a
#: sync merges a few hundred times
MERGES = 2000


def kept_verdicts(state: dict, key: str, main: str, gates: tuple[str, ...]) -> dict[str, list[str]]:
    """The gate verdicts the queue's state keeps under ``key`` (`Queue._gated`), while `main` and ``gates`` are those they were
    reached under; else none."""
    cache = state.get(key) or {}
    return (cache.get("of") or {}) if (cache.get("main"), cache.get("gates")) == (main, list(gates)) else {}


def flags_at(repo: Path, ref: str) -> DriveFlags:
    """drive.toml at ``ref``; the defaults when ``ref`` has none (QueueError when it can't be parsed)."""
    r = _git(repo, "show", f"{ref}:{DRIVE_FLAGS}", check=False)
    return DriveFlags.parse(r.stdout, f"{DRIVE_FLAGS} at {ref}") if r.returncode == 0 else DriveFlags()


def gates_at(repo: Path, ref: str, key: str = "landing_gates") -> tuple[str, ...]:
    """drive.toml's ``key`` (`landing_gates` or `warn_gates`) at ``ref``; none when ``ref`` has no drive.toml (QueueError when it
    isn't TOML or the gates are malformed).  Only that key is read: a branch's older code reads `main`'s drive.toml here (the
    ready mark's preflight), and a key added since must not stop it running the gates."""
    r = _git(repo, "show", f"{ref}:{DRIVE_FLAGS}", check=False)
    if r.returncode:
        return ()
    source = f"{DRIVE_FLAGS} at {ref}"
    try:
        gates = tomllib.loads(r.stdout).get(key)
    except tomllib.TOMLDecodeError as e:
        raise QueueError(f"{source}: not TOML: {e}") from None
    if gates is None:
        return ()
    return getattr(DriveFlags.parse(f"{key} = {json.dumps(gates)}\n", source), key)


def into_at(repo: Path, ref: str) -> str:
    """drive.toml's `land_into` at ``ref``, "main" when it has none.  Only that key is read, as `gates_at` reads its own: drive's
    config read names a bad key, and every `Queue` reads this one."""
    r = _git(repo, "show", f"{ref}:{DRIVE_FLAGS}", check=False)
    if r.returncode:
        return "main"
    source = f"{DRIVE_FLAGS} at {ref}"
    try:
        into = tomllib.loads(r.stdout).get("land_into")
    except tomllib.TOMLDecodeError as e:
        raise QueueError(f"{source}: not TOML: {e}") from None
    return "main" if into is None else DriveFlags.parse(f"land_into = {json.dumps(into)}\n", source).land_into


def signed_at(repo: Path, ref: str) -> bool:
    """drive.toml's `signed_verdicts` at ``ref``, false when it has none.  Only that key is read, as `into_at` reads its own (its
    guard reads two others, which drive's strict parse of `main` checks)."""
    r = _git(repo, "show", f"{ref}:{DRIVE_FLAGS}", check=False)
    if r.returncode:
        return False
    source = f"{DRIVE_FLAGS} at {ref}"
    try:
        on = tomllib.loads(r.stdout).get("signed_verdicts", False)
    except tomllib.TOMLDecodeError as e:
        raise QueueError(f"{source}: not TOML: {e}") from None
    if not isinstance(on, bool):
        raise QueueError(f"{source}: signed_verdicts = {on!r}: true or false")
    return on


def holders_at(repo: Path, ref: str) -> dict[str, str]:
    """{agent id: handle} of the registry at ``ref``: each agent id a handle's `holder` names holds that handle; none when ``ref``
    has no registry.  QueueError when it can't be parsed, as then no one's handle is known."""
    r = _git(repo, "show", f"{ref}:{REGISTRY}", check=False)
    try:
        reg = json.loads(r.stdout).get("handles", []) if r.returncode == 0 else []
    except (ValueError, AttributeError) as e:
        raise QueueError(f"{REGISTRY} at {ref} can't be parsed ({e}), so who holds each handle is unknown: fix it") from None
    return {a: h["handle"] for h in reg if isinstance(h, dict) and "handle" in h for a in AGENT_RE.findall(str(h.get("holder", "")))}


def run_gate(repo: Path, main: str, head: str, path: str) -> tuple[int | None, str]:
    """The landing or warn gate ``path`` as it is at ``main`` (so the change it judges can't weaken it), run as `python3 PATH
    MAIN HEAD --json` in ``repo``: (its exit, its output), or (None, why it didn't run)."""
    r = _git(repo, "show", f"{main}:{path}", check=False)
    if r.returncode:
        return None, f"{path} isn't at `main` {main[:12]}"
    # drive's TMPDIR is an NFS volume, where the gate script just run is often left as a .nfsXXXX silly-rename file
    with tempfile.TemporaryDirectory(prefix="queue-gate-", ignore_cleanup_errors=True) as d:
        script = Path(d) / Path(path).name
        script.write_text(r.stdout, errors="surrogateescape")
        try:
            p = subprocess.run([sys.executable, str(script), main, head, "--json"], cwd=repo, capture_output=True, text=True,
                               stdin=subprocess.DEVNULL, timeout=GATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return None, f"{path} ran over {GATE_TIMEOUT_S} s"
    return p.returncode, p.stdout if p.returncode in (0, 1) else (p.stderr or p.stdout).strip()[-400:]


def scripts_at(repo: Path, at: str, paths: Iterable[str]) -> dict[str, str]:
    """{path: blob} of the Python files at ``at`` of ``paths`` (a file, or a directory's own files when it ends in `/`)."""
    out = {}
    for line in _out(repo, "ls-tree", "-z", at, "--", *paths).split("\0"):
        meta, _, path = line.partition("\t")
        if path.endswith(".py") and meta.split()[1:2] == ["blob"]:
            out[path] = meta.split()[2]
    return out


def move_scripts(repo: Path, at: str) -> dict[str, str]:
    """{path: blob} of what judges a move at ``at``: `tools/move/`'s scripts (its maps are the moves' data, not code),
    `move_check.py` and the pins list it reads."""
    return scripts_at(repo, at, (MOVE_DIR, MOVE_CHECK, MOVE_PINS))


def write_scripts(repo: Path, scripts: dict[str, str], dest: Path) -> Path:
    """``scripts`` (`move_scripts`) written under ``dest`` at their paths, so each finds its imports beside it as in a checkout:
    ``dest``."""
    for path, blob in scripts.items():
        f = dest / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(_git(repo, "cat-file", "blob", blob).stdout, errors="surrogateescape")
    return dest


def gate_refusal(repo: Path, main: str, head: str, gates: Iterable[str],
                 run: Callable[[Path, str, str, str], tuple[int | None, str]] = run_gate,
                 onto: str | None = None, kind: str = "landing gate") -> tuple[list[str], list[str]]:
    """What ``gates``, each a ``kind`` (a landing gate, or a warn gate, whose failure refuses nothing), say of ``head`` against
    ``main`` (each judges the change from their merge base): (each failure, with its fix and how to run the gate; each gate
    that couldn't judge, with why).  ``head`` is a PR's head, or with ``onto`` its merge onto that branch, as the sync judges
    it: what the members below add counts too (a merge can take a file over its cap with the members below it, though its
    head alone is under).  A gate exits 0 to pass, and 1 with `--json`'s `{"failures": [{"why": ..., "fix": [...]}]}` to
    fail; anything else is unable to judge."""
    what, merged = (head[:12], "") if onto is None else (f"its merge onto `{onto}` ({head[:12]})", f" on its merge onto `{onto}`")
    rerun = "in the branch" if onto is None else f"on the branch merged with `origin/{onto}`"
    refused, unjudged = [], []
    for path in gates:
        rc, out = run(repo, main, head, path)
        if rc == 0:
            continue
        failures = None
        if rc == 1:
            with contextlib.suppress(ValueError, TypeError, KeyError, AttributeError):
                failures = [(str(f["why"]), [str(x) for x in (f.get("fix") or [])][:GATE_FIX_LINES])
                            for f in json.loads(out)["failures"]]
        if not failures:
            why = out if rc is None else f"exit {rc}" + (f": {out}" if rc != 1 else ": no failure it names")
            unjudged.append(f"the {kind} {path} can't judge {what} on `main` {main[:12]} ({why.strip()[-400:]})")
            continue
        for why, fix in failures:
            refused.append(f"the {kind} {path} at `main` {main[:12]}{merged}: {why}" + "".join(f"; {x}" for x in fix)
                           + f" (`python3 {path} origin/main HEAD` {rerun} runs it)")
    return refused, unjudged


def label_target(pr: int, head: str) -> str:
    return f"pr:{pr}@{head}"


def _who(name: str) -> str:
    return name.strip().lstrip("@").lower()


def unreviewed(labels: Iterable, owners: Iterable[str | None]) -> str | None:
    """Why the `reviewed` ``labels`` on one head don't land it, or None when they do: each reviewer's newest verdict counts, a
    hold by anyone keeps it, and a land counts only by a reviewer who is none of ``owners`` (the PR's owning lane or agent id,
    that owner's handle, its ready mark's author: `review_owners`), compared without case or a leading `@`."""
    mine = {_who(o) for o in owners if o}
    newest: dict[str, object] = {}
    for lab in sorted(labels, key=lambda lab: lab.ts):
        newest[_who(lab.by)] = lab
    if held := [f"{lab.by.lstrip('@')} ({lab.ref or 'no reason'})" for lab in newest.values() if lab.value == "hold"]:
        return f"held by {', '.join(held)}"
    if any(lab.value == "land" and by not in mine for by, lab in newest.items()):
        return None
    if own := [lab.by.lstrip("@") for by, lab in newest.items() if by in mine]:
        return f"no land verdict but its owner's own ({', '.join(own)}), which doesn't count"
    return "no land verdict"


#: a Lake package's lock, and the files of a package its Lean check reads
LEAN_LOCK = "lean-audit.json"
LEAN_SOURCES = (".lean", "lakefile.lean", "lakefile.toml", "lean-toolchain", "lake-manifest.json")


#: the analysis tree: a Lean change only here that touches no lock and no spec (`LEAN_SPEC`) is proofs, checked by the
#: after-landing run, a failure blaming it; Lean anywhere else runs as code (Flock's verifier and the ZK server, the PoUS
#: grader, the certifier program, the inference-only check, the remote audit's verifier), and a lock run follows its landing
LEAN_PROOFS_TREE = "Security/"
#: the spec: the Lean the guarantees' statements read, `Security/`'s Definitions library, and the Lake files of its package.
#: A change to it, like a lock's, is a Lean change with or without a lock change
LEAN_SPEC = ("Security/Definitions/", "Security/Definitions.lean", "Security/lakefile.toml", "Security/lake-manifest.json",
             "Security/lean-toolchain")


def lean_change(repo: Path, commit: str, base: str | None = None) -> bool:
    """Whether ``commit``'s own change (against its first parent, or ``base``) is a Lean change, whatever its PR's title, which
    drive's lock run follows on `checked` (`Driver._start_locks`: lean-audit alone, regenerating the records it leaves stale)
    and whose Lean cache `merge.lean_publish` publishes: it touches a lock or the spec (`LEAN_SPEC`), so neither a
    guarantee's statement nor a definition it reads reaches `main` without its record (the after-landing lean-audit,
    `--no-regenerate`, fails on a record difference), or a Lean source outside `LEAN_PROOFS_TREE` of a Lake package that has a
    lock, Lean that runs as code, whose lock (its `verifier.allow` among them) would go stale the same way.  A change to
    `Security/`'s other Lean alone is proofs: no lock run follows it, and a failure of the after-landing lean-audit blames
    it."""
    files = changed(repo, base or f"{commit}^1", commit)
    if any(Path(f).name == LEAN_LOCK or f.endswith(LEAN_SOURCES) and f.startswith(LEAN_SPEC) for f in files):
        return True
    if not (lean := [f for f in files if f.endswith(LEAN_SOURCES) and not f.startswith(LEAN_PROOFS_TREE)]):
        return False
    pkgs = {str(Path(f).parent) for f in _out(repo, "ls-tree", "-r", "--name-only", commit).splitlines() if Path(f).name == LEAN_LOCK}
    return any(p == "." or f.startswith(p + "/") for f in lean for p in pkgs)


def lowrisk(files: Iterable[str], excludes: Iterable[str]) -> bool:
    """Whether a change to ``files`` is low-risk, so drive lands it on its ready mark alone, with no land verdict and no wait in
    `next` (drive.toml's `lowrisk_lane`, `drive.Driver.land_lowrisk`): no file is under an ``excludes`` entry
    that ends in `/` (a directory from the repository's root: the verifiers under `verity/core/` and the soundness-path Python
    in `verity/ml/`) and no file's name matches an entry without one as a glob (`*.lean`, every `lean-audit.json`, the Lake
    files, `TRUSTED.sha256`).  ``files`` are what the PR's merge onto the landing branch's tip changes against that tip, not
    its own diff.  No excludes, nothing is low-risk; no file, nothing to land."""
    dirs, names = [], []
    for x in excludes:
        (dirs if x.endswith("/") else names).append(x)
    files = list(files)
    return bool(files and (dirs or names)) and not any(
        f.startswith(tuple(dirs)) or any(fnmatch.fnmatchcase(PurePosixPath(f).name, n) for n in names) for f in files)


#: the listings and GETs `StoreLabels.prefetch` makes at once: a fresh VM's store lacks every label of `next`'s ~500 members
#: (about 1,000 files), which one GET at a time took four minutes to pull; the ~900 open PRs' listings take about 4 s at 32
#: at once and 1 s at 128 (the remote's pool opens connections past its 64 idle ones)
PREFETCH_JOBS = 128


class StoreLabels:
    """Ready marks from the evidence store: `ready` labels on `pr:{n}@{head}`, pulled from the remote first.  Without that pull it
    raises: a cloud VM's shell that lost its store secrets has no remote, and its local labels are whatever it last pulled.
    With ``verify`` (a `labelauth.Verifier`: drive.toml's `signed_verdicts`), a ready mark or land verdict its `label_auth`
    companion doesn't authenticate is read as absent (`refusals` says why); `ready = false` and holds count as written.
    With ``change`` (`Queue.own_change`), a head with no `ready` (or `reviewed`) label of its own that counts takes those of
    each other head of its PR whose own change is the same, logging which (`_counted`): so a merge of `main` costs no new mark
    and no new review, while a head whose own diff differs takes none."""

    def __init__(self, store, log: Callable[[str], None] = print, since: str | None = None, verify=None,
                 change: Callable[[str], str | None] | None = None):
        self.store, self.log, self.since, self.verify, self.change = store, log, since, verify, change
        self._pulled: set[int] = set()
        self._newest: dict[tuple[int, str], object] = {}
        self._fresh: set[str] = set()
        #: the heads of each PR with labels here, by its label directories' prefix (`pr_12@`), listed again after a pull
        self._dirs: dict[str, list[str]] | None = None
        self._said: set[tuple[int, str, str]] = set()

    def prefetch(self, heads: Iterable[tuple[int, str]]) -> None:
        """Pull the labels of each `pr:{n}@{head}` from the store's remote at once, so `ready` answers each from them without a
        pull of its own (one listing per open PR, in turn, would be seconds of every drive tick); with ``change``, those of
        every head of each PR, one listing per PR as well, which a head without labels of its own may take.  A failed pull
        leaves each target to its own, which says why."""
        from .store import labels_sync

        heads = list(heads)
        targets, prs = [label_target(n, h) for n, h in heads], sorted({n for n, _ in heads})
        if not targets or getattr(self.store, "remote", None) is None:
            return
        try:
            if self.change is None:
                labels_sync.pull_targets(self.store, targets, jobs=PREFETCH_JOBS)
            else:
                labels_sync.pull_prefixes(self.store, [label_target(n, "") for n in prs], jobs=PREFETCH_JOBS)
        except Exception:  # noqa: BLE001
            return
        self._fresh.update(targets)
        if self.change is not None:
            self._pulled.update(prs)
            self._dirs = None

    def ready(self, pr: int, head: str) -> bool:
        """Whether the newest `ready` label on `pr:{n}@{head}` is `true`, written at or after ``since`` (`--marked-after`) when
        given.  `research queue eject` writes `ready = false` there, so an ejection holds on every VM's queue, at that head and at
        a new one whose own change is the same (`_counted`), until the owner marks it again (an ejection kept in the ejecting
        VM's state would be lost to drive, which re-admits)."""
        labs = sorted(self._counted(pr, head, vocab.READY)[0], key=lambda lab: lab.ts)
        self._newest[(pr, head)] = labs[-1] if labs else None
        return bool(labs) and str(labs[-1].value).lower() == "true" and (self.since is None or labs[-1].ts >= self.since)

    def reviews(self, pr: int, head: str) -> list:
        """The `reviewed` labels on `pr:{n}@{head}` (`research queue review`), pulled as `ready`'s are."""
        return self._counted(pr, head, vocab.REVIEWED)[0]

    def ready_by(self, pr: int, head: str) -> str | None:
        """Who wrote the newest `ready = true` label on `pr:{n}@{head}`, whenever it was written."""
        labs = [lab for lab in sorted(self._counted(pr, head, vocab.READY)[0], key=lambda lab: lab.ts) if str(lab.value).lower() == "true"]
        return labs[-1].by if labs else None

    def refusals(self, pr: int, head: str, key: str) -> list[str]:
        """Each ``key`` label on `pr:{n}@{head}` that ``verify`` reads as absent, with why; none without ``verify``."""
        return self._counted(pr, head, key)[1]

    def _counted(self, pr: int, head: str, key: str) -> tuple[list, list[str]]:
        """The ``key`` labels on `pr:{n}@{head}` that count, and why each other one doesn't.  With ``change``, a head none of
        whose own counts takes those of each other head of the PR with the same own change (``change``, never None for
        either), each judged on its own head's companions; the log names the heads they came from, once."""
        counted, refused = self._verified(pr, head, self._labels(pr, head, key))
        if counted or self.change is None or not (others := self._other_heads(pr, head, key)) or not (own := self.change(head)):
            return counted, refused
        came = []
        for old, labs in others.items():
            if self.change(old) == own:
                got, why = self._verified(pr, old, labs)
                counted, refused = [*counted, *got], [*refused, *why]
                came += [old] if got else []
        if came and (pr, head, key) not in self._said:
            self._said.add((pr, head, key))
            self.log(f"research queue: #{pr} at {head[:8]} takes the `{key}` labels at {', '.join(c[:8] for c in came)}: none "
                     f"of its own counts, and its own change is theirs (patch-id {own[:12]})")
        return counted, refused

    def _verified(self, pr: int, head: str, labs: list) -> tuple[list, list[str]]:
        """``labs``, labels on `pr:{n}@{head}`, that ``verify`` counts (each without it), and why each other one doesn't."""
        if self.verify is None:
            return labs, []
        from .labelauth import landing

        beside = self.store.labels(label_target(pr, head), key=vocab.LABEL_AUTH)
        counted, refused = [], []
        for lab in labs:
            if landing(lab) and (why := self.verify.why(lab, beside)):
                refused.append(f"{lab.key} = {lab.value} by {lab.by} at {lab.ts}: {why}")
            else:
                counted.append(lab)
        return counted, refused

    def marked_by(self, pr: int, head: str) -> str | None:
        """Who wrote the newest `ready` label `ready` read on `pr:{n}@{head}`."""
        lab = self._newest.get((pr, head))
        return None if lab is None else lab.by

    def marked_at(self, pr: int, head: str) -> str | None:
        """When the newest `ready` label `ready` read on `pr:{n}@{head}` was written."""
        lab = self._newest.get((pr, head))
        return None if lab is None else lab.ts

    def before_cutoff(self, pr: int, head: str) -> str | None:
        """When the newest `ready` label `ready` read on `pr:{n}@{head}` is `true` but older than ``since``, its time: a mark
        drive ignores, which `status` names so the owner re-marks or closes the PR."""
        lab = self._newest.get((pr, head))
        if lab is None or self.since is None or str(lab.value).lower() != "true" or lab.ts >= self.since:
            return None
        return lab.ts

    def _other_heads(self, pr: int, head: str, key: str) -> dict[str, list]:
        """{head: its ``key`` labels} for each head of pull request ``pr`` but ``head`` that has some, after pulling every head's
        labels from the store's remote (once per PR, or by `prefetch`); without that pull it raises, as `_labels` does."""
        from .store import labels_sync

        prefix = label_target(pr, "")
        if pr not in self._pulled:
            try:
                labels_sync.pull_prefix(self.store, prefix)
            except Exception as e:  # noqa: BLE001  (no pull: local labels alone may be stale)
                raise QueueError(f"STALE: the labels of {prefix}* weren't pulled from the store's remote ({type(e).__name__}: {e}), "
                                 f"so ready marks on #{pr}'s other heads here may be old") from None
            self._pulled.add(pr)
            self._dirs = None
        if self._dirs is None:
            root, self._dirs = Path(self.store.root) / "labels", {}
            for name in sorted(os.listdir(root)) if root.is_dir() else ():
                at, sep, old = name.rpartition("@")
                if sep:
                    self._dirs.setdefault(at + sep, []).append(old)
        out = {}
        for old in self._dirs.get(ids.target_dirname(prefix), ()):
            if old != head and ids.is_pr(label_target(pr, old)) and (labs := self.store.labels(label_target(pr, old), key=key)):
                out[old] = labs
        return out

    def _labels(self, pr: int, head: str, key: str) -> list:
        from .store import labels_sync
        from .store.local import StoreError

        target = label_target(pr, head)
        if target in self._fresh:
            return self.store.labels(target, key=key)
        try:
            labels_sync.read_through(self.store, [target], lambda s: self.log(f"research queue: {s}"), strict=True)
        except StoreError as e:
            raise QueueError(f"STALE: the labels of {target} weren't pulled from the store's remote ({e}), so ready marks here may "
                             f"be old; not answering from them") from None
        return self.store.labels(target, key=key)


# ---- the queue ----------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Member:
    """One PR on `next`: its merge commit, and the head it merged."""
    pr: int
    head: str
    owner: str
    commit: str


@dataclass
class Entry:
    """One open PR's standing in a sync."""
    pr: PR
    state: str = "ready"                  # ready | queued | waiting | refused
    why: str = ""
    after: tuple[int, ...] = ()           # open PRs whose current or an older head its head contains (`Queue.entries`)
    older: tuple[int, ...] = ()           # those of `after` whose current head it doesn't contain
    held: dict[int, str] = field(default_factory=dict)  # each of `after`: its head this head contains, current or the newest older
    position: int | None = None           # 1-based, on `next`
    commit: str | None = None
    idle_since: float | None = None       # stale (`research.stale`): idle since then

    def contains(self, n: int) -> str:
        """How a reason names ``n``, one of `after`."""
        return f"#{n}'s commits (at an older head)" if n in self.older else f"#{n}"

    def holds(self, n: int) -> str:
        """How a notice names ``n``, one of `after`, with the head of ``n``'s it holds (`held`), never a merge base: one of two
        branches that both merged `main` can be a commit of `main`."""
        return f"#{n}'s commits at its older head {self.held[n][:12]}" if n in self.older else f"#{n}'s head {self.held[n][:12]}"


def contained(repo: Path, base: str, head: str, heads: dict[str, int]) -> list[int]:
    """The PRs of ``heads`` ({head: PR number}) whose head ``head`` contains: one of its commits not on ``base``, ``head``
    itself included (the caller leaves its own PR out).  A PR goes on `next` after those it contains, and after those whose
    commits it holds at an older head (`Entry.after`, `Queue.entries`)."""
    return sorted({heads[h] for h in _out(repo, "rev-list", f"{base}..{head}").split() if h in heads})


def carried_heads(repo: Path, landed: str, head: str) -> list[tuple[int, str, str]]:
    """(N, H, C) for each commit C drive made on `next` for #N at head H (`queue_commit`) that ``head`` holds above
    ``landed``, the landing branch's tip, newest first, but those whose H ``landed`` holds (a `next` rebuilt since landed #N
    at a merge of its own): what a branch that merged `next` carries of the PRs on it."""
    log = _out(repo, "log", "--format=%H%x00%P%x00%B%x1e", "--grep=^Queue: #", f"{landed}..{head}")
    made = (queue_commit(*r.strip().split("\x00", 2)) for r in log.split("\x1e") if r.strip())
    return [(m.pr, m.head, m.commit) for m in made
            if m and _git(repo, "merge-base", "--is-ancestor", m.head, landed, check=False).returncode]


def stale_carried(repo: Path, carried: Iterable[tuple[int, str, str]], heads: dict[int, str],
                  admitted: Callable[[int], str | None], landed: str) -> dict[int, str]:
    """{N: the refusal} for each PR of ``carried`` (`carried_heads`) whose commits the head would land without it: #N is
    closed (not in ``heads``, each open PR's head), its current head doesn't hold the carried one (rebuilt without it), or it
    isn't admitted at its current head (``admitted``: why not, or None), as when its owner withdrew it after the branch
    merged `next`: a branch that merged `next` doesn't carry a PR's change the way a rebuild does (`Queue._carried`).  A
    head based on `main` that carries the merge of a PR closed unlanded is refused before this, as one of an old `next`
    (`old_next`), so for it a closed #N here landed at another commit (`queue_log`)."""
    out: dict[int, str] = {}
    for n, h, c in carried:
        if n in out:
            continue
        at, cur = f"it carries #{n}'s head {h[:12]} through `{BRANCH}`'s merge {c[:12]}", heads.get(n)
        fix = f"merge origin/main and #{n}'s current head instead of `{BRANCH}`, or drop #{n}'s commits"
        if cur is None:
            out[n] = f"{at}, and #{n} is closed: drop #{n}'s commits (rebuild the branch on origin/main, without `{BRANCH}`)"
        elif h != cur and _git(repo, "merge-base", "--is-ancestor", h, cur, check=False).returncode:
            out[n] = f"{at}, which neither `{landed}` nor #{n}'s current head {cur[:12]} holds: {fix}"
        elif (no := admitted(n)) is not None:
            out[n] = f"{at}, and #{n} isn't admitted at {cur[:12]} ({no}): {fix}"
    return out


def _brought(parents: dict[str, list[str]], heads: Iterable[tuple[int, str]]) -> dict[str, int]:
    """Each commit of ``parents`` (`rev-list --parents` of `next` above its base) that one of ``heads`` ((position on `next`,
    member head), lowest first) contains, with the position of the lowest that does: the member whose merge brings it."""
    at: dict[str, int] = {}
    for i, h in heads:
        todo = [h]
        while todo:
            if (c := todo.pop()) in parents and c not in at:
                at[c] = i
                todo += parents[c]
    return at


def above(repo: Path, base: str, heads: Iterable[str]) -> dict[str, tuple[list[str], set[str]]]:
    """{head: (`git rev-list --topo-order base..head`, `git rev-list --first-parent base..head`)} for each of ``heads``, from
    one `rev-list --parents` of them all: what one commit reaches that ``base`` doesn't is a walk of its parents that stops
    at ``base``'s, in git's topological order (`sort_in_topological_order`: a stack, each parent pushed once its last child
    in the walk is out), and its first-parent line the first parents to there.  Two `rev-list`s a PR took half a minute of
    every `research queue status`."""
    heads = list(dict.fromkeys(heads))
    if not heads:
        return {}
    r = subprocess.run(["git", "rev-list", "--parents", "--stdin"], cwd=repo, input="".join(f"{h}\n" for h in [*heads, f"^{base}"]),
                       capture_output=True, text=True, env=git_env(GIT_TERMINAL_PROMPT="0"))
    if r.returncode:
        raise QueueError(f"git rev-list --parents {base}.. ({len(heads)} heads): {(r.stderr or r.stdout).strip()[-400:]}")
    parents = {c: ps for c, *ps in (line.split() for line in r.stdout.splitlines())}
    out = {}
    for h in heads:
        below: set[str] = set()
        todo = [h] if h in parents else []
        while todo:
            if (c := todo.pop()) not in below:
                below.add(c)
                todo += [p for p in parents[c] if p in parents]
        children = dict.fromkeys(below, 0)
        for c in below:
            for p in parents[c]:
                if p in children:
                    children[p] += 1
        order, todo = [], [h] if below else []
        while todo:
            order.append(c := todo.pop())
            for p in parents[c]:
                if p in children:
                    children[p] -= 1
                    if not children[p]:
                        todo.append(p)
        line, c = set(), h
        while c in parents:
            line.add(c)
            c = parents[c][0] if parents[c] else ""
        out[h] = (order, line)
    return out


def based_on(pr: PR, prs: Iterable[PR]) -> str:
    """``pr``'s base branch, with the open PR whose head that branch is (`BRANCH (#P's head)`) when one is."""
    parent = next((p for p in prs if p.branch == pr.base and p.number != pr.number), None)
    return f"{pr.base} (#{parent.number}'s head)" if parent else pr.base


def _message(pr: PR) -> str:
    return f"Queue: #{pr.number} {pr.title}".rstrip() + f"\n\nPR: #{pr.number}\nHead: {pr.head}\nOwner: {pr.owner}\n"


def _move_message(pr: PR, tip: str, into: str, main: str) -> str:
    return (f"Queue: #{pr.number} {pr.title}".rstrip() + f"\n\n`python3 {RENAME} --map {pr.move}` at the root of {tip}, regenerated "
            f"by drive for #{pr.number} and certified by `{MOVE_CHECK} {tip[:12]} HEAD`, both as `{into}` {main[:12]} has them."
            f"\n\nPR: #{pr.number}\nHead: {pr.head}\nOwner: {pr.owner}\nMove: {pr.move}\n")


def queue_commit(commit: str, parents: str, body: str) -> Member | None:
    """The PR ``commit`` (its ``parents`` as `%P` gives them, its message ``body``) brings when the queue made it, else None:
    a merge of a PR onto `next` (`Queue: #N`, with its `PR`, `Head` and `Owner` trailers: `_message`), or a move drive
    regenerated, one commit with a `Move:` trailer too (`_move_message`).  Never a lander's commit, nor one whose `PR` isn't
    `#N` or whose `Head` isn't a full commit id, as no commit the queue made has."""
    t = {k: v.strip() for k, v in TRAILER_RE.findall(body)}
    if (len(parents.split()) > 1 or MOVE_TRAILER_RE.search(body)) and body.startswith("Queue: #") and {"PR", "Head", "Owner"} <= set(t) \
            and re.fullmatch(r"#\d+", t["PR"]) and re.fullmatch(r"[0-9a-f]{40}", t["Head"]):
        return Member(int(t["PR"][1:]), t["Head"], t["Owner"], commit)
    return None


class Queue:
    def __init__(self, workdir: Path, *, remote: str, github, ready: Callable[[int, str], bool],
                 store=None, shadow: bool = False, rules: Rules | None = None, log: Callable[[str], None] = print,
                 auth: tuple[str, ...] = (),
                 before_cutoff: Callable[[int, str], str | None] | None = None,
                 marked_by: Callable[[int, str], str | None] | None = None,
                 marked_at: Callable[[int, str], str | None] | None = None,
                 prefetch: Callable[[list[tuple[int, str]]], None] | None = None, verdict_first: bool | None = None,
                 land_into: str | None = None):
        self.workdir, self.remote, self.github, self.ready, self.store = Path(workdir), remote, github, ready, store
        #: the branch drive lands on and `next` is built on (`into`); None: drive.toml's `land_into` at `main`
        self.land_into, self._into = land_into, None
        #: whether the last `fetch` found `into` on origin, not `main` read in its place
        self.into_fetched = False
        self.shadow, self.log, self._rules, self.auth = shadow, log, rules, auth
        self.before_cutoff, self.marked_by, self.marked_at, self.prefetch = before_cutoff, marked_by, marked_at, prefetch
        #: whether drive's lift puts members with a land verdict below those without (`Driver._lift`); None: drive.toml's
        #: `verdict_first` at `main`
        self.verdict_first = verdict_first
        #: members `next` keeps that this sync moves below the rest, in this order, at the heads it holds (`Driver._lift`: those
        #: with a land verdict, above the priority list's); what `order` moved of them
        self.lift: list[int] = []
        self._lifted: list[int] = []
        #: members `next` keeps that this sync moves to its top, each with the PRs that contain it (`Driver._park`: those held
        #: by a verdict at the head `next` holds), so the members above them form the prefix that lands; what `order` moved
        self.park: list[int] = []
        self._parked: list[int] = []
        self.now: Callable[[], float] = time.time
        #: drive's priority list (`--priority-file`: `priority_file`, read again at each sync, `priority_list`), the landing
        #: order: the ready ones go at the bottom of `next` in its order (`order`); these PRs and those they contain
        #: (`_prioritized`) go first of what a sync merges anew
        self.priority: tuple[int, ...] = ()
        self.priority_file: Path | None = None
        self._prioritized: set[int] = set()
        #: releases the workdir's lock (`_lock`): a command that only reads calls it once its clone and fetches are done, so a
        #: listing that reads hundreds of members' labels doesn't refuse a verdict meanwhile
        self.unlock: Callable[[], None] = lambda: None
        self.branch = SHADOW_BRANCH if shadow else BRANCH
        self.state_path = self.workdir.parent / f"{self.workdir.name}.state.json"
        #: the files each merge commit of `next` changed (`touched_by`)
        self._touched: dict[str, set[str]] = {}
        #: main's registry as `handle` reads it (its handles, {agent id: handle}), until the next `fetch`
        self._registry: tuple[set, dict[str, str], set[str]] | None = None
        #: every open PR the last `status` listed, `main`-based or not (its entries leave out an unmarked PR on another branch)
        self.listed: list[PR] = []
        #: a sync's moves below the member it merges (`queue_log`'s, then each it regenerates), the heads it merges after it
        #: (`_restack` starts their restacks too), its regenerated moves and restacks by key (the state's `moves_made` and
        #: `restacks`, kept while a sync uses them: `_used`), and the restack jobs running by key (the state's `restacking`)
        self._moves: list[Move] = []
        self._upcoming: list[str] = []
        self._made: dict[str, dict] = {}
        self._restacks: dict[str, dict] = {}
        self._restacking: dict[str, dict] = {}
        self._used: set[str] = set()
        self._words_of: dict[str, set[str]] = {}
        #: how many restack jobs run at once (drive.toml's `restack_jobs`), and whether this sync starts any (a dry run doesn't)
        self.restack_jobs = DriveFlags.restack_jobs
        self._restack_starts = False
        #: each merge's record by its key (`_merge`, `_merge_key`; the state's `merges`), each stamped with the count of the last
        #: sync that used it (`_merge_serial`); the key of each (tip, head) and what every merge reads (`_merge_context`), both
        #: read once a sync; the record and head of each merge commit `_merge` made (`_lock_drops`); the merges a sync reused
        self._merge_records: dict[str, dict] = {}
        self._merge_serial = 0
        self._merge_keys: dict[tuple[str, str], str] = {}
        self._context: str | None = None
        self._merged_from: dict[str, tuple[str, str]] = {}
        self._merges_reused = 0
        #: each (landing branch, head)'s `own_change`, and those kept in the clone's git directory (at `_kept_at`) by merge base
        #: and head
        self._own: dict[tuple[str, str], str | None] = {}
        self._kept: dict[str, str | None] | None = None
        self._kept_at = Path()

    # -- the clone and its state

    def _remote(self, cwd: Path, *args: str, ok: tuple[int, ...] | None = (0,)) -> subprocess.CompletedProcess:
        """A git command that talks to the remote, with `auth` (`credential_args`); one that exits outside `ok` raises."""
        r = _git(cwd, *STALL, *self.auth, *args, check=False)
        if ok is not None and r.returncode not in ok:
            err = (r.stderr or r.stdout).strip()[-400:]
            hint = ("; no credential helper applies to the remote outside the caller's checkout: set one in git's global config, "
                    "or put an authenticated `gh` on PATH") if not self.auth and AUTH_FAILED_RE.search(err) else ""
            raise QueueError(f"git {' '.join(args)}: {err}{hint}")
        return r

    def _ours(self) -> bool:
        """Whether the workdir is a clone the queue made: one whose git config has `CLONE_MARK`, which the queue sets in each
        clone it makes, or `WORKDIR`, where it cloned before it set it."""
        return (_git(self.workdir, "config", "--local", "--get", CLONE_MARK, check=False).stdout.strip() == "true"
                or self.workdir.resolve() == WORKDIR.resolve())

    def _clone(self) -> None:
        """Clone the remote into the workdir, or bring the checkout there to the remote's state.  The origin of a clone the queue
        made (`_ours`) follows the remote; any other checkout keeps its own, and one whose origin names another repository
        than the remote is refused before anything in it changes."""
        if fresh := not (self.workdir / ".git").exists():
            self.workdir.parent.mkdir(parents=True, exist_ok=True)
            self._remote(self.workdir.parent, "clone", "-q", "--config", f"{CLONE_MARK}=true", self.remote, str(self.workdir))
        if fresh or self._ours():
            # the caller's origin as it is now: a token embedded in the URL the clone was made from rotates
            _git(self.workdir, "remote", "set-url", "origin", self.remote)
        elif not same_repository(origin := _git(self.workdir, "remote", "get-url", "origin", check=False).stdout.strip(),
                                 self.remote):
            here, there = redacted(origin), redacted(self.remote)
            raise QueueError(f"{self.workdir} is a checkout the queue didn't clone (its git config has no {CLONE_MARK}), whose "
                             f"origin, {here or 'unset'}, is another repository than the queue's remote {there}, and the queue "
                             f"never repoints such a checkout: pass --workdir a directory of its own, where the queue clones "
                             f"{there}" + (f", or --remote {here}, the repository that checkout is a clone of" if here else ""))
        # `.gitattributes` names these drivers; `git_env` gives them to the queue's own git, but without them in the clone's
        # config any other git in it (one by hand) merges the records line by line
        from .clone import Refused, set_drivers

        try:
            if (changed := set_drivers(self.workdir)) and not fresh:
                self.log(f"research queue: set the record merge drivers in {self.workdir}'s config, which lacked or held "
                         f"another value for {', '.join(changed)}")
        except Refused as e:
            raise QueueError(str(e)) from None
        _git(self.workdir, "merge", "--abort", check=False)
        _git(self.workdir, "reset", "-q", "--hard")
        # `status` and a listing `review` let the lock go once they've fetched and merge on here, and git's merge drivers read
        # their sides from `.merge_file_*` at the clone's root: removing those fails that merge ("Cannot lstat")
        _git(self.workdir, "clean", "-qfdx", "-e", "/.merge_file_*")

    def fetch(self, prs: Iterable[PR] = ()) -> dict[int, str]:
        """Forced refspecs only: `main`, the queue's branch when it exists, and each open PR's head as `refs/queue/pr/{n}`;
        return each fetched pull ref's commit, which can briefly differ from GitHub's API head, and none for a PR whose pull ref
        GitHub doesn't have (it waits, `fetch_prs`)."""
        prs = list(prs)
        self._registry, self._into = None, None
        specs = ["+refs/heads/main:refs/remotes/origin/main", *(f"+refs/pull/{p.number}/head:refs/queue/pr/{p.number}" for p in prs)]
        if self._remote(self.workdir, "fetch", "-q", "--prune", "origin", *specs, ok=None).returncode:
            # one pull ref GitHub hasn't made yet (a PR with only its `merge` ref) fails the whole
            # fetch, and every tick with it: fetch `main` alone, raising as before, then each PR's ref, leaving out what's missing
            self._remote(self.workdir, "fetch", "-q", "--prune", "origin", specs[0])
            for p, spec in zip(prs, specs[1:], strict=True):
                if (r := self._remote(self.workdir, "fetch", "-q", "origin", spec, ok=None)).returncode:
                    _git(self.workdir, "update-ref", "-d", f"refs/queue/pr/{p.number}", check=False)
                    self.log(f"research queue: #{p.number}'s pull ref can't be fetched, so it waits this sync: "
                             f"{(r.stderr or r.stdout).strip()[-200:]}")
        self.into_fetched = False
        if (into := self.into) != "main":
            # a landing branch the remote doesn't have reads as `main` here; drive's tick refuses without it (`drive_names`)
            if self._remote(self.workdir, "ls-remote", "--exit-code", "origin", f"refs/heads/{into}", ok=(0, 2)).returncode == 0:
                self._remote(self.workdir, "fetch", "-q", "origin", f"+refs/heads/{into}:{self.base}")
                self.into_fetched = True
            else:
                _git(self.workdir, "update-ref", self.base, "refs/remotes/origin/main")
        # the merge drivers (`clone.RECORD_DRIVER`) run from this checkout, so one that `main` has moved past judges merges with old
        # drivers (an old clone refuses a lean-audit.json conflict main's merge.py resolves).  Drive's checkout, `next`'s tip, contains `main` and stays
        if _git(self.workdir, "merge-base", "--is-ancestor", self.base, "HEAD", check=False).returncode:
            _git(self.workdir, "checkout", "-q", "--detach", "--force", self.base)
        # exit 2: no such branch; any other failure is the remote's, never an empty queue
        if self._remote(self.workdir, "ls-remote", "--exit-code", "origin", f"refs/heads/{self.branch}", ok=(0, 2)).returncode == 0:
            self._remote(self.workdir, "fetch", "-q", "origin", f"+refs/heads/{self.branch}:refs/remotes/origin/{self.branch}")
        else:
            _git(self.workdir, "update-ref", "-d", f"refs/remotes/origin/{self.branch}", check=False)
        if not prs:
            return {}
        refs = dict(line.split() for line in _out(self.workdir, "for-each-ref", "--format=%(refname) %(objectname)",
                                                  "refs/queue/pr/").splitlines())
        return {p.number: refs[ref] for p in prs if (ref := f"refs/queue/pr/{p.number}") in refs}

    def fetch_prs(self, prs: list[PR]) -> tuple[list[PR], dict[int, Entry]]:
        """Fetch ``prs`` and separate any whose API head disagrees with its pull ref, so one GitHub propagation race cannot stop
        the queue.  Such a PR waits outside ``next`` until a later snapshot agrees rather than admitting an unverified old head,
        unless its branch's tip is the API head."""
        fetched = self.fetch(prs)
        consistent, waiting = [], {}
        for p in prs:
            if fetched.get(p.number) != p.head and p.branch and self._head_from_branch(p):
                fetched[p.number] = p.head
            if p.number not in fetched:
                waiting[p.number] = Entry(p, state="waiting", why="GitHub has no pull ref for its head yet; retrying next sync")
                continue
            if fetched[p.number] == p.head:
                consistent.append(p)
                continue
            why = f"GitHub's API head {p.head[:12]} and pull ref {fetched[p.number][:12]} disagree; retrying next sync"
            waiting[p.number] = Entry(p, state="waiting", why=why)
            self.log(f"research queue: #{p.number}'s API head {p.head[:12]} and fetched pull ref "
                     f"{fetched[p.number][:12]} disagree; retrying next sync")
        return consistent, waiting

    def _head_from_branch(self, p: PR) -> bool:
        """Take ``p``'s head from its branch when the branch's tip is the API head: GitHub can leave refs/pull/N/head behind its
        branch and API head for hours."""
        ref = f"refs/queue/branch/{p.number}"
        r = self._remote(self.workdir, "fetch", "-q", "origin", f"+refs/heads/{p.branch}:{ref}", ok=None)
        tip = _git(self.workdir, "rev-parse", "-q", "--verify", ref, check=False).stdout.strip() if not r.returncode else ""
        _git(self.workdir, "update-ref", "-d", ref, check=False)
        if tip != p.head:
            return False
        _git(self.workdir, "update-ref", f"refs/queue/pr/{p.number}", tip)
        self.log(f"research queue: #{p.number}'s pull ref isn't at its API head {p.head[:12]}; took the head from its branch")
        return True

    def load_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {"ejected": {}, "conflicts": {}}

    def save_state(self, state: dict) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
        tmp.replace(self.state_path)

    def rules(self) -> Rules:
        if self._rules is None:
            self._rules = rules_at(self.workdir, "origin/main")
            if self._rules is None:
                raise QueueError(f"`main` has no {RULES}: pass --rules FILE")
        return self._rules

    @property
    def into(self) -> str:
        """The branch drive's review landing fast-forwards and `next` is built on: `land_into`, else drive.toml's at `main` as
        last fetched.  `checked`: `main` follows it only to each commit
        whose after-landing run passed every step on the merge gate (`Driver._promote`)."""
        if self.land_into is not None:
            return self.land_into
        if self._into is None:
            self._into = into_at(self.workdir, "refs/remotes/origin/main")
        return self._into

    @property
    def base(self) -> str:
        """`into` as fetched: the ref `next` is built on."""
        return f"refs/remotes/origin/{self.into}"

    @property
    def slug(self) -> str:
        """OWNER/NAME of the repository the queue serves (`GitHub.repo`; `REPO` without a GitHub, as the preflight's queue has)."""
        return getattr(self.github, "repo", None) or REPO

    def chain(self) -> list[Member]:
        """The members on the queue's branch above its base (`base`), oldest first; none when the branch is missing or the base
        isn't below it."""
        return chain_of(self.workdir, self.base, f"refs/remotes/origin/{self.branch}")

    def own_change(self, head: str, landing: str | None = None) -> str | None:
        """``head``'s own change (`own_change`) from its merge base with the landing branch as fetched (``landing``, else
        `base`), or None when either isn't in the clone or the branch has all of it: what a ready mark or land verdict carries
        to a new head by (`StoreLabels`).  A large diff takes about a second, so each is kept in the clone's git directory by
        merge base and head, the newest `OWN_KEEP`."""
        if (key := (landing or self.base, head)) in self._own:
            return self._own[key]
        if not (mb := _git(self.workdir, "merge-base", key[0], head, check=False).stdout.strip()):
            self._own[key] = None
            return None
        if self._kept is None:
            self._kept_at = self.workdir / _out(self.workdir, "rev-parse", "--git-common-dir") / "own-changes.json"
            self._kept = json.loads(self._kept_at.read_text()) if self._kept_at.is_file() else {}  # written whole, by `replace`
        if (at := f"{mb} {head}") not in self._kept:
            self._kept[at] = own_change(self.workdir, mb, head)
            self._kept = dict(list(self._kept.items())[-OWN_KEEP:])
            tmp = self._kept_at.with_name(f"{self._kept_at.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self._kept))
            tmp.replace(self._kept_at)
        self._own[key] = self._kept[at]
        return self._own[key]

    # -- admission

    def _marked_again(self, p: PR, ejected: dict) -> bool:
        """Whether ``p``'s owner marked it ready after its ejection at this head: the store's newest mark decides on every VM, so
        this queue's own record of the ejection can't hold a PR that mark admits: otherwise a PR re-marked at its ejected head
        would stay off `next` on the ejecting VM's queue only."""
        if self.marked_at is None or not self.ready(p.number, p.head):
            return False
        when, at = self.marked_at(p.number, p.head), ejected.get("at", "")
        if len(at) == len("2026-10-12T21:09Z"):
            at = at[:-1] + ":59Z"                       # recorded to the minute: a mark in that minute may predate the ejection
        return when is not None and when > at

    def entries(self, prs: list[PR], state: dict, base: str | None = None, landed: str | None = None,
                queued: str | None = None, open_numbers: Collection[int] | None = None,
                old_landed: str | None = None) -> dict[int, Entry]:
        """Each open PR's standing.  With ``open_numbers`` (every open PR's number), one whose head carries merges of an old
        `next` (`old_next`: of PRs neither among them nor landed) is refused, with the fix (`old_next_refusal`), as the
        preflight refuses its mark (`Preflight._old_next`).  A PR P goes after, and waits while it isn't admitted, each other open PR Q whose current
        head P's head contains, or an older head of Q's, by ancestry alone (never by PR number).  P holds an older head of Q
        when one of P's own commits (those in ``base``.. its head less those below the heads of open PRs it contains, and
        less those `next` already carries without P, below) is

        - a head the queue saw Q at, and the first open PR it saw there (`seen`), that Q's head still contains, as when P
          branched from Q and Q moved on unmarked (else P would land Q's unmarked commits under P's mark); of several PRs
          seen at that head, only those marked ready, when any is, since their marks cover it; or
        - one of Q's own commits on Q's first-parent line that P has only through a merge, as when P merged Q's branch and Q
          moved on (P waits for Q, and Q for nothing).

        What `next` (``queued``: the queue's branch as fetched unless given) already carries without P isn't P's own: each
        commit above ``landed`` that a member below P's merge brings (any member, when P isn't one), counting only members
        at their PR's current head that this call admits (`_brought`).  So a commit P shares with Q's older head through work
        admitted members already bring holds P for nothing, P's own merge never excuses its own commits, and a member that
        moved on or that this call leaves waiting carries nothing: the call judges again without it.  A head of Q's that
        P's head contains still holds P as before.

        Commits two PRs merely share wait for nothing: a third branch both merged (on neither first-parent line), or a fork
        point the queue never saw either PR at (on both), and neither does a head of Q's that P holds (`Entry.held`) on
        ``landed``, the landing branch's tip (`base` as fetched unless given: the preflight judges on `main`).  One walk of
        every head (`above`) and one `rev-list --parents` of `next`, not a merge base for each pair.

        ``old_landed``, the old path's `checked` while ``landed`` doesn't hold it: a head of Q's that P holds there landed under
        drive's rules, so P still goes after Q but doesn't wait for it, though Q moved on unmarked since."""
        rules, base = self.rules(), base or self.base
        landed = landed or self.base
        heads = {p.head: p.number for p in prs}
        out: dict[int, Entry] = {}
        below: dict[int, set[str]] = {}
        order: dict[int, list[str]] = {}                    # below, children before parents: the first held is the newest
        line: dict[int, set[str]] = {}
        if self.prefetch is not None:
            self.prefetch([(p.number, p.head) for p in prs if rules.hold_label not in p.labels and p.owner is not None])
        reach = above(self.workdir, base, (p.head for p in prs))
        old = {} if open_numbers is None else old_next(self.workdir, (p.head for p in prs if reach[p.head][0]), landed,
                                                       open_numbers)
        for p in prs:
            e = out[p.number] = Entry(p)
            ejected = state["ejected"].get(str(p.number))
            if why := p.outside(self.slug):
                e.state, e.why = "refused", why
            elif carried := [old[c] for c in reach[p.head][0] if c in old]:
                e.state, e.why = "refused", old_next_refusal(carried, self.into)
            elif rules.hold_label in p.labels:
                e.state, e.why = "waiting", f"held (`{rules.hold_label}` label)"
            elif p.owner is None:
                e.state, e.why = "waiting", "no owner: add a `Lane {name}` line to the PR body"
            elif ejected and ejected["head"] == p.head and not self._marked_again(p, ejected):
                e.state, e.why = "refused", f"ejected at {p.head[:8]}: {ejected['why']} (a ready mark re-admits it)"
            elif self.ready(p.number, p.head):
                pass
            elif self.before_cutoff and (when := self.before_cutoff(p.number, p.head)):
                e.state, e.why = "waiting", (f"ignored: marked before cutoff (its ready mark at {p.head[:8]} is from {when}, "
                                             f"before `marked_after`): re-mark it or close it")
            else:
                e.state, e.why = "waiting", f"not marked ready at {p.head[:8]}: `research queue ready {p.number} --on MACHINE` (or --local; --mark without the tier)"
            order[p.number], line[p.number] = reach[p.head]
            below[p.number] = set(order[p.number])
            e.held = {heads[h]: h for h in order[p.number] if h in heads and heads[h] != p.number}
            e.after = tuple(sorted(e.held))
        members = chain_of(self.workdir, landed, queued or f"refs/remotes/origin/{self.branch}")
        pos = {m.pr: i for i, m in enumerate(members) if m.pr in out and m.head == out[m.pr].pr.head}
        parents = {c: ps for c, *ps in map(str.split, _out(self.workdir, "rev-list", "--parents", f"{landed}..{members[-1].commit}")
                                           .splitlines())} if pos else {}
        first = {n: (e.state, e.why, e.after, e.held) for n, e in out.items()}
        # a held commit is above ``base``, so on ``landed`` only when that isn't ``base``
        on = functools.cache(lambda c: landed != base and not _git(self.workdir, "merge-base", "--is-ancestor", c, landed,
                                                                   check=False).returncode)
        on_old = functools.cache(lambda c: bool(old_landed) and not _git(self.workdir, "merge-base", "--is-ancestor", c,
                                                                          old_landed, check=False).returncode)
        seen = state.get("seen") or {}
        carriers = set(pos)
        while True:                                         # until every member counted as carrying commits is admitted
            brought = _brought(parents, sorted((pos[n], members[pos[n]].head) for n in carriers))
            for n, e in out.items():
                e.state, e.why, e.after, e.held = *first[n][:3], dict(first[n][3])
            own = {n: {c for c in below[n].difference(*(below[a] for a in e.after)) if brought.get(c, math.inf) >= pos.get(n, math.inf)}
                   for n, e in out.items()}
            branch: dict[str, list[int]] = {}               # commit: the PRs it is an own commit on the first-parent line of
            for n, cs in own.items():
                for c in cs & line[n]:
                    branch.setdefault(c, []).append(n)
            for n, e in out.items():
                older: dict[int, str] = {}
                for c in (c for c in order[n] if c in own[n]):
                    # a head several PRs were seen at is carried by the marked ones among them, whose marks cover it
                    at = [q for q in seen.get(c, ()) if q in out and c in below[q]]
                    for q in [*([q for q in at if out[q].state == "ready"] or at), *(branch.get(c, ()) if c not in line[n] else ())]:
                        if q != n:
                            older.setdefault(q, c)
                e.older = tuple(sorted(older))
                e.after = tuple(sorted(e.after + e.older))
                e.held |= older
            for e in out.values():
                e.after = tuple(a for a in e.after if not on(e.held[a]) and not self._carried(e.pr, out[a], state))
                e.older = tuple(a for a in e.older if a in e.after)
                e.held = {a: e.held[a] for a in e.after}
            changed = True
            while changed:                                  # a PR waits while a PR whose commits it contains isn't admitted
                changed = False
                for e in out.values():
                    if e.state == "ready" and (blocking := [a for a in e.after
                                                            if out[a].state != "ready" and not on_old(e.held[a])]):
                        e.state, e.why, changed = "waiting", f"contains {e.contains(blocking[0])}, which isn't admitted", True
            if not (fell := {n for n in carriers if out[n].state != "ready"}):
                return out
            carriers -= fell

    @staticmethod
    def _seen(state: dict, prs: list[PR], open_numbers: set[int]) -> None:
        """Record in ``state``'s `seen` ({head: PRs}) each head of ``prs`` no open PR was seen at before, for the PRs at it now,
        so a PR holding it later holds an older head of theirs (`entries`); a closed PR's records go."""
        seen = state["seen"] = {h: kept for h, ns in (state.get("seen") or {}).items()
                                if (kept := [n for n in ns if n in open_numbers])}
        new: dict[str, list[int]] = {}
        for p in prs:
            if p.head not in seen:
                new.setdefault(p.head, []).append(p.number)
        seen.update(new)

    def _carried(self, p: PR, a: Entry, state: dict) -> bool:
        """Whether ``p`` carries the change of ``a``, a PR whose head or own commits it contains (`entries`) and that isn't
        marked ready: the same owner
        rebuilt it as ``p``, or ``a``'s owner withdrew its mark (`eject`, or a `ready = false` label).  Then ``p`` doesn't wait
        for ``a`` and goes on without it below, so a lane that rebuilds a PR needn't drop the superseded one by hand, which no
        one's tooling may close."""
        if a.state == "ready" or a.pr.owner is None or self.rules().hold_label in a.pr.labels:
            return False
        who = _who(a.pr.owner)
        ejected = state["ejected"].get(str(a.pr.number)) or {}
        return (_who(p.owner or "") == who
                or (ejected.get("head") == a.pr.head and _who(ejected.get("by") or "") == who)
                or (self.marked_by is not None and not self.ready(a.pr.number, a.pr.head)
                    and not (self.before_cutoff and self.before_cutoff(a.pr.number, a.pr.head))    # then it's a `true`
                    and _who(self.marked_by(a.pr.number, a.pr.head) or "") == who))

    def _stale_carry(self, pr: PR, entries: dict[int, Entry]) -> str | None:
        """The preflight's (g) (`stale_carried`) of ``pr``'s head, judged by this sync's ``entries``, so a mark the preflight
        never saw (a label written directly) doesn't land a PR's commits without it either: the refusal, or None."""
        if not (carried := [c for c in carried_heads(self.workdir, self.base, pr.head) if c[0] != pr.number]):
            return None
        def why(n: int) -> str | None:
            if (e := entries[n]).state == "ready":
                return None
            return f"not marked ready at {e.pr.head[:8]}" if e.why.startswith("not marked ready") else e.why

        heads = {n: e.pr.head for n, e in entries.items()}
        return "; ".join(stale_carried(self.workdir, carried, heads, why, self.into).values()) or None

    def order(self, chain: list[Member], entries: dict[int, Entry], lift: Iterable[int] = (),
              park: Iterable[int] = ()) -> list[int]:
        """Members that keep their head, or move it to a descendant (`_same_or_fixed`), keep their order; the rest follow by PR
        number, those of drive's priority list and the PRs they contain first (`_prioritized`), each after the PRs it contains.
        The PRs of drive's priority list (`priority`, the landing order) that are ready go below all of them, in its order, so
        they land first.  The members of ``lift`` at the heads `next` holds, none of those, go above the priority list's and
        below the rest, each only after the PRs it contains (`_lifted`).  The members of ``park`` that stay at the head `next`
        holds (or at the admitted head while a new one waits for its mark: `_pending`), and every PR that contains one, go
        above all of these, in `next`'s order (`_parked`)."""
        ready = {n for n, e in entries.items() if e.state == "ready"}
        stays = {m.pr for m in chain if self._pending(m, entries)}
        kept = [m.pr for m in chain if m.pr in ready and self._same_or_fixed(m, entries) or m.pr in stays]
        at = {m.pr: m.head for m in chain}
        parked = {n for n in park if n in kept and (entries[n].pr.head == at[n] or n in stays)}
        grown = bool(parked)
        while grown:
            grown = False
            for n in ready.union(kept) - parked:
                if parked.intersection(entries[n].after):
                    parked.add(n)
                    grown = True
        out: list[int] = []
        placing: set[int] = set()

        def place(n: int) -> None:
            if n in out or n in placing:
                return
            placing.add(n)
            for a in entries[n].after:
                place(a)
            out.append(n)

        new = sorted(ready - set(kept))
        cold = {n for n in new if self._forces_cold(n)}
        self._lifted = []
        first = [n for n in self.priority if n in ready]
        below = set(first)
        for n in lift:
            if (n in kept and n not in parked and n not in below and entries[n].pr.head == at.get(n)
                    and all(a in self._lifted or a in below for a in entries[n].after)):
                self._lifted.append(n)
        self._parked = [n for n in kept if n in parked] + sorted(parked - set(kept))
        # a newly ready PR that makes circuit-check run cold goes above the warm ones, unless drive's priority list puts it first;
        # members already on `next` keep their place
        ahead = [n for n in new if n in self._prioritized]
        for n in (first + self._lifted + kept + ahead
                  + [n for n in new if n not in cold and n not in ahead] + sorted(cold.difference(ahead))):
            if n not in parked:
                place(n)
        for n in self._parked:
            place(n)
        return out

    def flag(self, name: str) -> bool:
        """drive.toml's ``name`` (`verdict_first`) as this queue takes it: its own setting, else `main`'s."""
        own = getattr(self, name)
        return own if own is not None else getattr(flags_at(self.workdir, "refs/remotes/origin/main"), name)

    def _same_or_fixed(self, m: Member, entries: dict[int, Entry]) -> bool:
        """Whether member ``m``'s PR is at the head `next` holds, or at a head that descends from it (a fix on top, or a merge of a
        PR below it): such a re-mark keeps its place, since its new head is merged and checked like any other, and a PR
        re-merged each time a stacked PR below it takes a fix isn't sent behind the rest of the queue."""
        new = entries[m.pr].pr.head
        return new == m.head or not _git(self.workdir, "merge-base", "--is-ancestor", m.head, new, check=False).returncode

    def _pending(self, m: Member, entries: dict[int, Entry]) -> bool:
        """Whether member ``m`` stays at its admitted head while its PR's new head, a descendant of it, waits for its ready mark:
        then the mark swaps the head in place (`_same_or_fixed`) instead of the PR, pushed one commit on, dropping out until
        its new mark and rejoining at the back.  Only while the admitted head is still marked: an ejection withdraws that
        mark."""
        e = entries.get(m.pr)
        if e is None or e.pr.head == m.head or e.state != "waiting" or not e.why.startswith("not marked ready"):
            return False
        return bool(self.ready(m.pr, m.head)) and self._same_or_fixed(m, entries)

    def _forces_cold(self, n: int) -> bool:
        r = _git(self.workdir, "diff", "--name-only", f"refs/remotes/origin/main...refs/queue/pr/{n}", check=False)
        return r.returncode == 0 and forces_cold(r.stdout.split())

    # -- building `next`

    def _merge(self, tip: str, pr: PR, members: list[Member], head: str | None = None) -> tuple[str | None, str]:
        """(``pr``'s head merged onto ``tip``, "") or (None, why not), as `git merge --no-ff` in a checkout of ``tip`` gives them,
        without the checkout: `git merge-tree` merges, with ``tip``'s merge drivers (`merged_tree`), and `git commit-tree`
        makes the merge commit, so a candidate that conflicts costs one call (merging each candidate in the worktree, with
        many of them conflicting, takes tens of minutes).  A merge whose every conflict is in a pin is kept, each pin rebuilt
        from the merged tree, unless both sides changed what a JSON pin holds besides its digests (`repinned`), and the
        merged tree's Lean takes the Lean module system once the tree has it
        (`ported`), which refuses the merge when it can't.  A git without `--attr-source` (before 2.40) raises: merging
        without it would leave out every PR whose only conflict is a pin (drive refuses its tick before, `Driver._old_git`).
        ``head``: the commit merged in its place, its head restacked through a move (`_merge_through`), under the same
        message.  What the merge comes to is recorded under what it reads (`_merge_key`, the state's `merges`): the same
        ``head`` merged again onto ``tip``'s tree takes the record, told as the merge was, and only `git commit-tree` runs."""
        head = head or pr.head
        rec = self._recorded(tip, head)
        if "conflict" in rec:
            return None, self._unmerged("", rec["conflict"], members)
        if "err" in rec:
            return None, self._unmerged(rec["err"], [], members)
        if rec.get("pins"):
            self.log(f"research queue: #{pr.number} conflicts only in pins: {', '.join(rec['pins'])} rebuilt from the merged tree")
        if rec.get("gone"):
            self.log(f"research queue: #{pr.number} deletes {', '.join(rec['gone'])}, which `{self.branch}` changed only by "
                     "porting it to the module system: deleted in the merge")
        if rec.get("tip_gone"):
            self.log(f"research queue: `{self.branch}` deletes {', '.join(rec['tip_gone'])}, which #{pr.number} changed only by "
                     "porting it to the module system: deleted in the merge")
        if rec.get("merged"):
            return tip, ""                                      # already merged: `merge --no-ff` makes no commit either
        if "why" in rec:
            return None, rec["why"]
        if files := rec.get("ported"):
            pins = rec.get("ported_pins")
            self.log(f"research queue: #{pr.number}'s merged tree holds Lean made before the module system: {', '.join(files)} "
                     f"ported by `{MODULES}`" + (f", and the pins {', '.join(pins)} rebuilt" if pins else ""))
        commit = _git(self.workdir, "commit-tree", rec["tree"], "-p", tip, "-p", head, "-m", _message(pr),
                      env=IDENTITY).stdout.strip()
        self._merged_from[commit] = (self._merge_key(tip, head), head)
        return commit, ""

    def _recorded(self, tip: str, head: str) -> dict:
        """``head`` merged onto ``tip`` as `_merged` tells it, from the record of that merge onto ``tip``'s tree (`_merge_key`)
        when there is one and the clone still holds its tree (a fresh clone or a prune drops it), else made and recorded;
        stamped with this sync's count (`_merge_serial`).  The sync's merges (`_merge`) and the low-risk lane's
        (`drive.Driver._clean_merge`) are one merge, and share the records."""
        key = self._merge_key(tip, head)
        rec = self._merge_records.get(key)
        if rec is not None and "tree" in rec and not self._has_tree(rec["tree"]):
            rec = None
        if rec is None:
            rec = self._merged(tip, head)
            if "err" not in rec and "why" not in rec:
                self._merge_records[key] = rec
        else:
            self._merges_reused += 1
        rec["at"] = self._merge_serial
        return rec

    def _keep_merges(self, state: dict) -> None:
        """Put the merges' records into ``state`` (its `merges`): those it holds with those this process made or took, the
        `MERGES` the latest syncs used."""
        had = state.get("merges") or {}
        merged = {**(had.get("of") or {}), **self._merge_records}
        self._merge_records = dict(sorted(merged.items(), key=lambda kv: -kv[1].get("at", 0))[:MERGES])
        state["merges"] = {"serial": max(self._merge_serial, int(had.get("serial") or 0)), "of": self._merge_records}

    def _merged(self, tip: str, head: str) -> dict:
        """``head`` merged onto ``tip``, as `_merge` records it: {"tree": the merged tree, ported, with "pins" rebuilt, "gone"
        the `.lean` files ``head`` deleted that `next` only ported and "tip_gone" the reverse (`taken`), and "ported" the files
        the codemod ported with "ported_pins" the pins recording them}, {"merged": True} when ``tip`` holds ``head``,
        {"conflict": the files} when it doesn't merge, and, never recorded, {"err": git's error} when `git merge-tree` fails
        otherwise and {"why": why} when porting fails (`ported`).  Empty lists are left out."""
        tree, files, r = merged_tree(self.workdir, tip, head, plain=False)
        if r.returncode == 129 and "attr-source" in r.stderr:
            raise QueueError(f"this git has no `--attr-source`, which merging `{self.branch}` needs: install git >= 2.40")
        if tree is None:
            return {"conflict": files} if files else {"err": (r.stderr or r.stdout).strip()[-200:]}
        rec: dict = {}
        if files:
            rec.update(zip(("pins", "gone", "tip_gone"), taken(self.workdir, tree, head, files)))
        elif not _git(self.workdir, "merge-base", "--is-ancestor", head, tip, check=False).returncode:
            return {"merged": True}
        tree, files, pins, why = ported(self.workdir, tree, base=tip)
        if tree is None:
            return {**rec, "why": why}
        return {k: v for k, v in {**rec, "tree": tree, "ported": files, "ported_pins": pins}.items() if v}

    def _merge_key(self, tip: str, head: str) -> str:
        """What `_merge` of ``head`` onto ``tip`` reads, as one digest: ``tip``'s tree (its files, its `.gitattributes` and the
        drivers' files `base_drivers` runs, its pins and its Lean, which `ported` and its base read), ``head``, their merge bases
        in git's order (what `git merge-tree` merges from, what `port_deletions` ports from, whether ``tip`` holds ``head``
        already, and `lock_drops`' fork), and what every merge of this sync reads (`_merge_context`).  ``tip``'s commit is
        not in it: a `checked` moved to a commit with the same tree leaves each merge above it as it was."""
        if (key := self._merge_keys.get((tip, head))) is None:
            tree = _out(self.workdir, "rev-parse", f"{tip}^{{tree}}")
            bases = _git(self.workdir, "merge-base", "--all", tip, head, check=False).stdout.split()
            key = hashlib.sha256("\0".join([self._merge_context(), tree, head, *bases]).encode()).hexdigest()
            self._merge_keys[(tip, head)] = key
        return key

    def _merge_context(self) -> str:
        """What every merge reads beside its commits, as one digest: this code (`queue.py`, `clone.py` with the drivers'
        commands, `RECORD_DRIVER`, and `gates.py`, whose `git_env` every git and script runs in), git's version and config as
        the merges see it (a driver's command, its merge settings, what the caller's environment sets; not a remote, a branch's
        upstream, credentials or the identity), and the interpreters the drivers (`python3`) and the porting
        (`sys.executable`) run in."""
        if self._context is None:
            from . import clone, gates

            config = [ln for ln in _git(self.workdir, "config", "--list", check=False).stdout.splitlines()
                      if not ln.startswith(("remote.", "url.", "http.", "credential.", "user.", "branch."))]
            drivers = subprocess.run(["python3", "-c", "import sys; print(sys.version)"], capture_output=True, text=True,
                                     env=git_env()).stdout
            parts = [*(Path(f).read_text() for f in (__file__, clone.__file__, gates.__file__)),
                     json.dumps(clone.RECORD_DRIVER, sort_keys=True), _out(self.workdir, "version"), sys.version, drivers, *config]
            self._context = hashlib.sha256("\0".join(parts).encode()).hexdigest()
        return self._context

    def _has_tree(self, tree: str) -> bool:
        return not _git(self.workdir, "cat-file", "-e", f"{tree}^{{tree}}", check=False).returncode

    def _lock_drops(self, tip: str, head: str, commit: str) -> list[str]:
        """`lock_drops` of ``commit``, ``head``'s merge onto ``tip``, kept in that merge's record (`_merge`) when ``head`` is
        what it merged: it reads ``tip``'s tree, ``head``, their merge base and the merged tree, which the record's key and
        tree fix.  A head restacked through a move merged another commit, and is judged afresh."""
        key, merged = self._merged_from.get(commit, ("", ""))
        if merged != head or (rec := self._merge_records.get(key)) is None:
            return lock_drops(self.workdir, tip, head, commit)
        if "drops" not in rec:
            rec["drops"] = lock_drops(self.workdir, tip, head, commit)
        return list(rec["drops"])

    # -- moves (the module's `Moves`)

    def _regenerate(self, tip: str, pr: PR, main: str) -> tuple[str | None, str]:
        """(move PR ``pr``'s move regenerated on ``tip`` and certified, "") or (None, why not), once per tip, landing tip ``main``
        and head (`_regenerated`; the state's `moves_made`)."""
        key = f"{tip}:{main}:{pr.move}:{pr.number}:{pr.head}"
        self._used.add(key)
        got = self._made.get(key) or {}
        if not got.get("why") and not (got.get("commit") and self._has(got["commit"])):
            commit, why = self._regenerated(tip, pr, main)
            got = self._made[key] = {"commit": commit} if commit else {"why": why}
        return got.get("commit"), got.get("why", "")

    def _regenerated(self, tip: str, pr: PR, main: str) -> tuple[str | None, str]:
        """`rename.py --map MAP` (``pr``'s `Move:` line, ``tip``'s map) run at the root of a worktree of ``tip``, committed as one
        commit on it with the PR's trailers and `Move: MAP` (`_move_message`), and certified by `move_check.py TIP COMMIT`;
        the PR's head is only the request.  Both scripts are the landing branch's tip ``main``'s (`move_scripts`), as its
        landing gates are, so a member below the move can't change what judges it; and a member below that changes
        `tools/move/`'s scripts holds the move back, since `restack.py` carries the members above it with its base's.  A map
        ``tip`` or a script ``main`` lacks, such a member, a script that fails, a map that renames nothing there, or a move
        `move_check.py` refuses is the reason."""
        scripts, into = move_scripts(self.workdir, main), self.into
        missing = [f"{p} at `{into}` {main[:12]}" for p in (RENAME, MOVE_CHECK) if p not in scripts]
        if _git(self.workdir, "cat-file", "-e", f"{tip}:{pr.move}", check=False).returncode:
            missing.insert(0, f"{pr.move} at {tip[:12]}")
        if missing:
            return None, (f"its move can't be regenerated: there's no {' or '.join(missing)}; land the map first, then mark the "
                          "move again")
        ours = move_scripts(self.workdir, tip)
        if changed := sorted(p for p in ours.keys() | scripts.keys()
                             if p.startswith(MOVE_DIR) and ours.get(p) != scripts.get(p)):
            return None, (f"the members below it change {', '.join(changed)} from `{into}` {main[:12]}: its move runs "
                          f"`{into}`'s scripts, and the members above it are carried across it with its base's, so it waits "
                          f"until those members land on `{into}`")
        with tempfile.TemporaryDirectory(prefix="queue-move-", ignore_cleanup_errors=True) as tmp:
            wt = Path(tmp, "wt")
            _git(self.workdir, "worktree", "add", "-q", "--detach", str(wt), tip)
            try:
                rename = write_scripts(self.workdir, scripts, Path(tmp, "rename")) / RENAME
                r = subprocess.run([sys.executable, str(rename), "--map", pr.move, "--root", str(wt)], cwd=wt,
                                   capture_output=True, text=True, env=_script_env())
                if r.returncode:
                    return None, f"`{RENAME} --map {pr.move}` failed on {tip[:12]} (exit {r.returncode}): {_tail(r)}"
                _git(wt, "add", "-A")
                tree = _out(wt, "write-tree")
                if tree == _out(self.workdir, "rev-parse", f"{tip}^{{tree}}"):
                    return None, (f"`{RENAME} --map {pr.move}` changes nothing on {tip[:12]}: the move is in it already, or "
                                  "its map moves nothing there")
                commit = _git(self.workdir, "commit-tree", tree, "-p", tip, "-m", _move_message(pr, tip, into, main),
                              env=IDENTITY).stdout.strip()
                # written afresh: the rename ran the tree's own pins.py, which could rewrite anything already written
                check = write_scripts(self.workdir, scripts, Path(tmp, "check")) / MOVE_CHECK
                c = subprocess.run([sys.executable, str(check), tip, commit], cwd=wt, capture_output=True, text=True,
                                   env=_script_env())
                if c.returncode:
                    return None, (f"its move regenerated on {tip[:12]} as {commit[:12]} isn't certified (`{MOVE_CHECK} "
                                  f"{tip[:12]} {commit[:12]}` exit {c.returncode}): {_tail(c, 4)}")
                self.log(f"research queue: #{pr.number}'s move ({pr.move}) regenerated on {tip[:12]} as {commit[:12]}: "
                         f"{_tail(c, 1)}")
                return commit, ""
            finally:
                _git(self.workdir, "worktree", "remove", "--force", str(wt), check=False)
                _git(self.workdir, "worktree", "prune", check=False)

    def _merge_through(self, tip: str, pr: PR, members: list[Member]) -> tuple[str | None, str]:
        """`_merge` of ``pr`` onto ``tip``, carried across each move below it (`_moves`) that its head lacks: restacked through
        them (`_restack`, `Restacking` until its restack ends) when its own change names a word of a move's map (`_sensitive`:
        a plain merge would leave a file it adds or a name it writes where the move no longer has it) or its plain merge
        conflicts, so a conflict that remains is its own."""
        if not (pending := [mv for mv in self._moves if not self._contains(mv.commit, pr.head)]):
            return self._merge(tip, pr, members)
        if not any(self._sensitive(pr.head, mv) for mv in pending):
            commit, why = self._merge(tip, pr, members)
            if commit is not None:
                return commit, why
        head, names = pr.head, ", ".join(f"{mv.commit[:12]} ({mv.map})" for mv in pending)
        for mv in pending:
            head, why = self._restack(head, mv, pr.head)
            if head is None:
                return None, f"it doesn't cross the move {mv.commit[:12]} ({mv.map}): {why}"
        commit, why = self._merge(tip, pr, members, head)
        if commit is None:
            return None, f"{why}, restacked through the move {names} as {head[:12]}"
        self.log(f"research queue: #{pr.number} at {pr.head[:8]} restacked through the move {names} as {head[:12]}")
        return commit, why

    def _sensitive(self, head: str, mv: Move) -> bool:
        """Whether ``head``'s own change (from its merge base with the move's base) names a word of ``mv``'s map, in a path or a
        line, as text in any case: a superset of what `rename.py` rewrites in it."""
        if mv.map not in self._words_of:
            r = _git(self.workdir, "show", f"{mv.commit}^1:{mv.map}", check=False)
            data = tomllib.loads(r.stdout) if not r.returncode else {}      # it parsed: its rename.py made the move from it
            words = {*data.get("words", {}), *(w for s in data.get("words_in", []) for w in s.get("words", {}))}
            self._words_of[mv.map] = {w.lower() for w in words if w}
        r = _git(self.workdir, "diff", "-U0", "--no-color", "--no-ext-diff", f"{mv.commit}^1...{head}", check=False)
        low = r.stdout.lower()
        return bool(r.returncode) or any(w in low for w in self._words_of[mv.map])

    def _restack(self, head: str, mv: Move, of: str) -> tuple[str | None, str]:
        """(``head`` carried across ``mv`` by `restack.py` onto the move, "") or (None, why not), once per head and move (the
        state's `restacks`), for PR head ``of``; `Restacking` until a job in the background has carried it (`_launch_restack`,
        the state's `restacking`).  The job that starts then starts one for each head merged after it this sync that lacks ``mv``
        (and has every move below it) and whose change names a word of its map, while fewer than `restack_jobs` run."""
        key = f"{head}:{mv.commit}"
        self._used.add(key)
        got = self._restacks.get(key)
        if got is None or (got.get("result") and not self._has(got["result"])):
            self._restacks.pop(key, None)
            if key not in self._restacking and self._launch_restack(head, mv, of):
                before = self._moves[:self._moves.index(mv)] if mv in self._moves else []
                for h in dict.fromkeys(self._upcoming):
                    k = f"{h}:{mv.commit}"
                    if h == head or k in self._restacks or k in self._restacking or self._contains(mv.commit, h) \
                            or not all(self._contains(b.commit, h) for b in before) or not self._sensitive(h, mv):
                        continue
                    if not self._launch_restack(h, mv, h):
                        break
            if (got := self._restacks.get(key)) is None:
                raise Restacking(self._restacking_why(key, mv))
        return got.get("result"), got.get("why", "")

    def _launch_restack(self, head: str, mv: Move, of: str, tries: int = 1, since: float | None = None) -> bool:
        """Start `restack.py HEAD --move-commit MOVE --script rename.py --script-arg=--map --script-arg=MAP --json`, the landing
        branch's copy run in the clone, in the background (`RESTACK_JOB`, `drive.detach`), unless this sync starts none or
        `restack_jobs` run: whether it started.  It runs the move's base's `tools/move/` scripts, which `_regenerated` held to
        the landing branch's when it made the move.  A landing branch without the script is the restack's failure."""
        key = f"{head}:{mv.commit}"
        if not self._restack_starts or len(self._restacking) >= self.restack_jobs:
            return False
        script = _git(self.workdir, "show", f"{self.base}:{RESTACK}", check=False)
        if script.returncode:
            self._restacks[key] = {"why": f"`{self.into}` has no {RESTACK}"}
            return False
        from .drive import LAUNCHES, detach

        root = Path(tempfile.gettempdir()) / LAUNCHES
        root.mkdir(parents=True, exist_ok=True)
        out = tempfile.mkdtemp(prefix="restack-", dir=root)
        path = Path(out, "restack.py")
        path.write_text(script.stdout, errors="surrogateescape")
        argv = ["sh", "-c", RESTACK_JOB, "sh", out, sys.executable, str(path), head, "--move-commit", mv.commit, "--script", RENAME,
                "--script-arg=--map", f"--script-arg={mv.map}", "--main", "refs/remotes/origin/main", "--json"]
        now = self.now()
        self._restacking[key] = {**detach(self.workdir, argv, _script_env()), "out": out, "of": of, "map": mv.map, "at": now,
                                 "since": now if since is None else since, "tries": tries}
        self._save_restacks()
        self.log(f"research queue: restacking {head[:12]} across the move {mv.commit[:12]} ({mv.map}) in the background"
                 + (f", try {tries}" if tries > 1 else ""))
        return True

    def _save_restacks(self) -> None:
        """The restack jobs running and the results read (the state's `restacking` and `restacks`) saved at once, as drive saves
        its launches: a sync that fails after starting or reading one neither starts it again nor loses its result."""
        self.save_state({**self.load_state(), "restacks": self._restacks, "restacking": self._restacking})

    def _restacking_why(self, key: str, mv: Move) -> str:
        """Why the PR whose restack is ``key`` waits off `next` (`RESTACKING`)."""
        from .drive import _hhmm

        what = f"{RESTACKING} {mv.commit[:12]} ({mv.map})"
        if job := self._restacking.get(key):
            return f"{what} since {_hhmm(job['since'])}"
        if not self._restack_starts:
            return f"{what}: a dry run starts no restack"
        return f"{what} once one of the {len(self._restacking)} restacks running ends"

    def _collect_restacks(self, wanted: set[str]) -> None:
        """Read each restack job that ended (`restack_end`) into `_restacks`, dropping its result once its PR's head is none of
        ``wanted``.  A job that ended with no result (it died, or ran past `RESTACK_WAIT_S`) runs once more, and then that is
        the restack's failure; one something else interrupted runs again without that counting as a try, so a stop or a
        restart of the pod never refuses a PR."""
        ended = False
        for key, job in list(self._restacking.items()):
            if (end := restack_end(job, self.now())) is None:
                continue
            ended = True
            del self._restacking[key]
            (head, move), (said, how, interrupted) = key.split(":"), end
            if job["of"] not in wanted or not self._has(head):
                self.log(f"research queue: the restack of {head[:12]} across the move {move[:12]} is dropped: "
                         + (f"its PR's head {job['of'][:8]} changed" if job["of"] not in wanted else "the clone lost it"))
            elif said is not None:
                self._restacks[key] = said
            elif interrupted:
                self.log(f"research queue: the restack of {head[:12]} across the move {move[:12]} was interrupted ({how}); "
                         "again, not counted as a try")
                self._launch_restack(head, Move(move, job["map"]), job["of"], job["tries"], job["since"])
            elif job["tries"] < 2:
                self.log(f"research queue: the restack of {head[:12]} across the move {move[:12]} ended with no result ({how}); "
                         "once more")
                self._launch_restack(head, Move(move, job["map"]), job["of"], job["tries"] + 1, job["since"])
            else:
                self._restacks[key] = {"why": f"{RESTACK} failed ({how})"}
        if ended:
            self._save_restacks()

    def _has(self, commit: str) -> bool:
        return not _git(self.workdir, "cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode

    def _contains(self, ancestor: str, commit: str) -> bool:
        return not _git(self.workdir, "merge-base", "--is-ancestor", ancestor, commit, check=False).returncode

    def _gated(self, main: str, gates: tuple[str, ...], tip: str, commit: str, pr: PR, before: dict[str, list[str]],
               verdicts: dict[str, list[str]], kind: str = "landing gate") -> tuple[list[str], list[str]]:
        """`gate_refusal` of ``commit``, ``pr``'s merge onto ``tip``, by ``gates`` (each a ``kind``), or the refusals ``before``
        (the last syncs' verdicts under this `main` and these gates) holds for a merge with the same message, tree and parent
        tree: a lift or a moved member merges every member above it again, and those whose members below are the same set have
        the same trees, each ~15 s of gates.  The gates read only `main`, the merge's trees and message, and the members below,
        each judged at its own merge.  What the gates say goes into ``verdicts``; a gate that couldn't judge is never kept."""
        trees = _out(self.workdir, "rev-parse", f"{tip}^{{tree}}", f"{commit}^{{tree}}").split()
        key = hashlib.sha256("\0".join([*trees, _message(pr)]).encode()).hexdigest()
        if key in before:
            verdicts[key] = before[key]
            return list(before[key]), []
        refused, unjudged = gate_refusal(self.workdir, main, commit, gates, onto=self.branch, kind=kind)
        if not unjudged:
            verdicts[key] = refused
        return refused, unjudged

    def _read_priority(self, state: dict) -> tuple[int, ...]:
        """Drive's priority list from `priority_file` (`priority_list`), in its order, logged when it changes (the state's
        `priority`); one it can't read is empty, its error logged once while it stays the same (`priority_error`)."""
        try:
            got = priority_list(self.priority_file)
        except QueueError as e:
            if state.get("priority_error") != str(e):
                self.log(f"research queue: {e}; drive's priority list is empty until the file is fixed")
            state["priority_error"], state["priority"] = str(e), []
            return ()
        state.pop("priority_error", None)
        if list(got) != state.get("priority", []):
            self.log(f"research queue: drive's priority list ({self.priority_file}), first to land first: "
                     + (" ".join(f"#{n}" for n in got) or "empty"))
            state["priority"] = list(got)
        return got

    def _unmerged(self, err: str, files: list[str], members: list[Member]) -> str:
        if not files:
            return f"doesn't merge onto `{self.branch}`: {err}"
        by = touched_by(self.workdir, members, files, self._touched)
        return f"conflicts with `{self.branch}` in {', '.join(files)}" + (f" (touched by {', '.join(by)})" if by else "")

    def handle(self, pr: PR) -> str | None:
        """The Slack handle of the lane that owns ``pr``, from main's registry (`.agents/skills/using-slack/registry.json`): the
        lane its body names, the agent id it carries, the ready mark's author, else the agent whose id ends in its branch's
        suffix (`cursor/<name>-4292`); None when none of them is a handle."""
        if (got := self._read_registry()) is None:
            return None
        names, holders, _ = got
        by = self.marked_by(pr.number, pr.head) if self.marked_by else None
        suffix = pr.branch.rpartition("-")[2] if "-" in pr.branch else ""
        tail = sorted({n for a, n in holders.items() if len(suffix) == 4 and a.endswith(suffix)})
        for c in (pr.owner, holders.get(pr.owner or ""), by, tail[0] if len(tail) == 1 else None):
            if c in names:
                return c
        return None

    def _read_registry(self) -> tuple[set, dict[str, str], set[str]] | None:
        """(the handles, {agent id: handle}, the Linear team keys its handles own) of `main`'s registry, read once per `fetch`: a
        review tick asks `handle` of each member of `next` (`Driver._owners`), and each `git show` is ~70 ms in drive's clone on
        the control pod's network volume, times over a hundred members.  None when it can't be parsed."""
        if self._registry is None:
            r = _git(self.workdir, "show", f"origin/main:{REGISTRY}", check=False)
            try:
                reg = json.loads(r.stdout).get("handles", []) if r.returncode == 0 else []
            except (ValueError, AttributeError):
                return None
            self._registry = ({h.get("handle") for h in reg if isinstance(h, dict)},
                              {a: h["handle"] for h in reg if isinstance(h, dict) and "handle" in h
                               for a in AGENT_RE.findall(str(h.get("holder", "")))},
                              {h["linear"] for h in reg if isinstance(h, dict) and isinstance(h.get("linear"), str)})
        return self._registry

    def issue_teams(self) -> set[str]:
        """The Linear team keys `main`'s registry gives its handles (`linear`), whose bare keys `named_issue` counts; empty when
        it can't be read."""
        return got[2] if (got := self._read_registry()) else set()

    def sync(self, dry_run: bool = False) -> tuple[list[Member], dict[int, Entry]]:
        """Rebuild `next` (see the module's `next`).  Off drive it refuses before anything (`_drive_only`)."""
        self._drive_only(self.branch, "queue sync", dry_run)
        self._clone()
        state = self.load_state()
        if self.priority_file is not None:
            self.priority = self._read_priority(state)
        open_prs = self.github.open_prs()
        prs, waiting = self.fetch_prs([p for p in open_prs if p.base == "main"])
        # a PR landed on `checked` stays open until `main` reaches it; a move or a head restacked through one lands at a commit
        # that isn't its head, which the queue's trailers name
        landed, self._moves = queue_log(self.workdir, self.base)
        prs = [p for p in prs if p.head not in landed
               and _git(self.workdir, "merge-base", "--is-ancestor", p.head, self.base, check=False).returncode]
        self._made, self._restacks, self._used = state.get("moves_made") or {}, state.get("restacks") or {}, set()
        self._restacking, self._restack_starts = state.get("restacking") or {}, not dry_run
        merges = state.get("merges") or {}
        self._merge_records = {**(merges.get("of") or {}), **self._merge_records}
        self._merge_serial, self._merges_reused = int(merges.get("serial") or 0) + 1, 0
        self._merge_keys, self._context, self._merged_from = {}, None, {}
        self._seen(state, prs, {p.number for p in open_prs})
        chain = self.chain()
        entries = self.entries(prs, state, open_numbers={p.number for p in open_prs})
        entries.update(waiting)
        listed = [n for n in self.priority if n in entries]
        self._prioritized = set(listed).union(*(entries[n].after for n in listed))
        pending = {m.pr: m.head for m in chain if self._pending(m, entries)}
        # what the last sync left out at its head is tried again above the rest: in the priority list's place or among members
        # `next` keeps, it would be left out again and every member above it merged again, each tick keeping none of them
        on = {m.pr for m in chain}
        out = {int(n) for n, c in state.get("conflicts", {}).items()
               if int(n) in entries and int(n) not in on and c["head"] == entries[int(n)].pr.head}
        out |= {n for n, e in entries.items() if n not in on and out.intersection(e.after)}

        def last(order: list[int]) -> list[int]:
            return [n for n in order if n not in out] + [n for n in order if n in out]

        main = _out(self.workdir, "rev-parse", self.base)
        order = last(self.order(chain, entries, self.lift, self.park))
        heads = {n: pending.get(n, entries[n].pr.head) for n in order}

        def kept(order: list[int]) -> int:
            want = [(n, heads[n]) for n in order]
            return next((i for i, (m, w) in enumerate(zip(chain, want)) if (m.pr, m.head) != w), min(len(chain), len(want)))

        k = kept(order)
        if k < max((i + 1 for i, n in enumerate(order) if n in self._lifted), default=0):
            self.log(f"research queue: {' '.join(f'#{n}' for n in self._lifted)} (reviewed) go to the bottom of `{self.branch}`, "
                     "below what waits for a review")
        parked = set(self._parked)
        low = next((i for i, m in enumerate(chain) if m.pr in parked), len(chain))
        if any(m.pr not in parked and m.pr in heads for m in chain[low:]):
            self.log(f"research queue: {' '.join(f'#{n}' for n in self._parked)} (held, or containing a held PR) go to the top "
                     f"of `{self.branch}`, so what was above them lands")
        members = chain[:k]
        tip = members[-1].commit if members else main
        dropped: set[int] = set()
        conflicts: dict[str, dict] = {}
        waits: dict[str, dict] = {}
        gates = gates_at(self.workdir, main) if order[k:] else ()
        warns = gates_at(self.workdir, main, "warn_gates") if order[k:] else ()
        before, verdicts = kept_verdicts(state, "gate_verdicts", main, gates), {}
        warn_before, warn_verdicts = kept_verdicts(state, "warn_verdicts", main, warns), {}
        open_heads = {p.number: p.head for p in open_prs}
        warned = {n: w for n, w in (state.get("warned") or {}).items() if open_heads.get(int(n)) == w["head"]}
        merged = False
        if tip != main:                                         # the moves among the members `next` keeps
            self._moves = queue_log(self.workdir, tip)[1]
        self._upcoming = [heads[n] for n in order[k:] if not entries[n].pr.move]
        if not dry_run:
            self._collect_restacks({*heads.values(), *open_heads.values()})

        def refuse(n: int, why: str) -> None:
            e = entries[n]
            e.state, e.why = "refused", why
            dropped.add(n)
            conflicts[str(n)] = {"head": e.pr.head, "why": why, "handle": self.handle(e.pr)}

        def wait(n: int, why: str) -> None:
            e = entries[n]
            e.state, e.why = "waiting", why
            dropped.add(n)
            waits[str(n)] = {"head": e.pr.head, "why": why, "handle": self.handle(e.pr)}
            if state.get("waiting", {}).get(str(n), {}).get("head") != e.pr.head:
                self.log(f"research queue: #{n} at {e.pr.head[:8]} waits off `{self.branch}`: {why}")

        for n in order[k:]:
            e = entries[n]
            if why := self._stale_carry(e.pr, entries):
                refuse(n, why)
                continue
            if blocked := [a for a in e.after if a in dropped]:
                wait(n, f"contains {e.contains(blocked[0])}, which didn't merge")
                continue
            if e.pr.move:
                commit, why = self._regenerate(tip, e.pr, main)
            else:
                try:
                    commit, why = self._merge_through(tip, e.pr, members)
                except Restacking as r:
                    wait(n, str(r))
                    continue
            merged = True
            # a regenerated move is `rename.py`'s output that `move_check.py` certified, lock texts included: what its map
            # rewrites in a premise or a frozen entry's reason is no drop
            if not e.pr.move and commit not in (None, tip) and (drops := self._lock_drops(tip, e.pr.head, commit)):
                commit, why = None, lock_drop_refusal(drops, self.branch)
            if commit is not None and gates:
                refused, unjudged = self._gated(main, gates, tip, commit, e.pr, before, verdicts)
                if unjudged and not refused:
                    wait(n, f"a landing gate can't judge it yet: {' | '.join(unjudged)}; retried next sync")
                    continue
                if refused or unjudged:
                    commit, why = None, " | ".join(refused + unjudged)
            if commit is None:
                refuse(n, why)
                continue
            if warns:
                said, unjudged = self._gated(main, warns, tip, commit, e.pr, warn_before, warn_verdicts, kind="warn gate")
                if said:
                    warned[str(n)] = {"head": e.pr.head, "why": " | ".join(said), "handle": self.handle(e.pr)}
                if unjudged:
                    self.log(f"research queue: #{n} at {e.pr.head[:8]}: {' | '.join(unjudged)}; a warn gate keeps no PR out")
            tip = commit
            members.append(Member(n, e.pr.head, e.pr.owner or "", commit))
            if e.pr.move:
                self._moves.append(Move(commit, e.pr.move))
        if merged:                  # drive runs train.sh and check.py from the worktree: `next`'s tip after a sync
            _git(self.workdir, "checkout", "-q", "--detach", tip)
        for i, m in enumerate(members, 1):
            entries[m.pr].state, entries[m.pr].position, entries[m.pr].commit = "queued", i, m.commit
        old = _git(self.workdir, "rev-parse", "--verify", "-q", f"refs/remotes/origin/{self.branch}", check=False).stdout.strip()
        if tip != old and not dry_run:
            self._remote(self.workdir, "push", "-q", f"--force-with-lease=refs/heads/{self.branch}:{old}", "origin",
                         f"{tip}:refs/heads/{self.branch}")
        self.log(f"research queue: `{self.branch}` at {tip[:12]}: {len(members)} PR(s) above `{self.into}`"
                 + (f", kept {k}, merged {len(members) - k}" if members else "")
                 + (f", merges reused for {self._merges_reused}" if self._merges_reused else "")
                 + (f", gates' verdicts reused for {reused}" if (reused := len(verdicts.keys() & before.keys())
                                                                  + len(warn_verdicts.keys() & warn_before.keys())) else "")
                 + (" (dry run: not pushed)" if dry_run else ""))
        if not dry_run:
            state["conflicts"], state["waiting"] = conflicts, waits
            state["moves_made"] = {k: v for k, v in self._made.items() if k in self._used}
            moves = {mv.commit for mv in self._moves}
            state["restacks"] = {k: v for k, v in self._restacks.items() if k in self._used or k.rpartition(":")[2] in moves}
            state["restacking"] = self._restacking
            for key, gs, kept, new in (("gate_verdicts", gates, before, verdicts), ("warn_verdicts", warns, warn_before,
                                                                                   warn_verdicts)):
                if gs:
                    newest = {**{h: v for h, v in kept.items() if h not in new}, **new}
                    state[key] = {"main": main, "gates": list(gs), "of": dict(list(newest.items())[-GATE_VERDICTS:])}
            state["warned"] = warned
            self._keep_merges(state)
            self.save_state(state)
        return members, entries

    def _drive_only(self, ref: str, verb: str, dry_run: bool) -> None:
        """Refuse a push of `main`, `checked` or `next` (`merge.DRIVE_ONLY`) here unless this is drive (`merge.on_drive`), whatever the caller:
        GitHub's rulesets don't stop verity-agents[bot].  `--shadow` gets no pass: its push of
        `queue-shadow` passes here, but `research queue sync`, `land` and `drive` refuse off drive with it as without, though
        it never pushes `main` or `next`."""
        from .merge import DRIVE_ONLY, off_drive, on_drive

        if not dry_run and ref in DRIVE_ONLY and not on_drive():
            raise QueueError(off_drive(ref, verb))

    def eject(self, pr: int, reason: str, by: str, sha: str | None = None) -> PR:
        """Withdraw ``pr``'s mark and keep it off the queue's branch; with ``sha``, only while that is still its head, so an
        eject that waits for drive's tick doesn't withdraw the mark of a head its owner pushed meanwhile."""
        self._clone()
        prs = {p.number: p for p in self.github.open_prs()}
        if pr not in prs:
            raise QueueError(f"#{pr} isn't an open PR")
        if sha and not prs[pr].head.startswith(sha):
            raise QueueError(f"#{pr} is at {prs[pr].head[:12]} now, not {sha[:12]}: its owner pushed since, so nothing is ejected; "
                             f"the new head is checked on its own")
        state = self.load_state()
        state["ejected"][str(pr)] = {"head": prs[pr].head, "why": reason, "by": by, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if self.store is not None and not self.shadow:
            from .store.cli import _label_write_through

            target = label_target(pr, prs[pr].head)
            self.store.add_label(target, vocab.READY, False, by, ref=reason)
            said, lost = _label_write_through(self.store, target)
            self.log(f"research queue: #{pr}'s ready mark at {prs[pr].head[:12]} is withdrawn ({target}); {said}")
            if lost:
                raise QueueError(f"#{pr}'s withdrawn mark didn't reach the store's remote, so other queues may still admit it: {said}")
        self.save_state(state)
        return prs[pr]

    def tell_ejected(self, p: PR, reason: str, by: str, post: Callable[..., bool] | None = None) -> int:
        """A hand ejection's notice, posted as ``by`` in the merge thread to `CI` and the lane that owns ``p`` (only a handle's
        holder may post as it).  Drive's
        own ejections are told by `Driver.refuse`.  1, with the text to send, when the post failed."""
        from .drive import CI, _merge_thread, merge_thread_post

        owner = self.handle(p)
        to = tuple(dict.fromkeys((CI, *([f"@{owner}"] if owner else []))))
        text = (f"{' '.join(to)} queue: #{p.number} at {p.head[:8]} is ejected from `{self.branch}` by {by}: {reason}. "
                f"Fix it and re-mark it; the next sync rebuilds `{self.branch}` without it")
        if self.shadow:
            self.log(f"research queue: would tell {' '.join(to)}: {text} [shadow]")
            return 0
        if (post or merge_thread_post)(text, to, sender=by):
            self.log(f"research queue: told {' '.join(to)} as @{by}: #{p.number} ejected")
            return 0
        print(f"research queue: #{p.number} is ejected, but posting its notice as @{by} failed, so no one was told. Send it as "
              f"your own handle: research msg send --as HANDLE --thread {_merge_thread()} --to {' '.join(to)} --text {text!r}",
              file=sys.stderr)
        return 1

    # -- status

    def status(self, stale: Callable[[list[PR]], dict[int, float]] | None = None, close_days: int = 0,
               fetched: Callable[[], tuple[Callable[[list[PR]], dict[int, float]] | None, int]] | None = None,
               loop: bool = False) -> dict[int, Entry]:
        """Each open PR's entry; with ``stale`` (`research.stale.stale_prs` over the open PRs), each stale one says so, an unmarked
        PR based on another branch among them, and ``close_days`` (drive's `stale_close_days`) when drive closes it.
        ``fetched``, called once the clone is fetched and before anything reads it, returns those two instead: the command reads
        its cutoff and drive.toml at `main` there, so it fetches once.  With ``loop``, as the verified queue admits each onto
        `main` (`vqueue_loop.World.ready`, which reads no `next`), a head on `main` landed; else as drive does, with each
        member's place on `next`, a head on `checked` landed there, and the refusals of drive's last sync."""
        self._clone()
        state = self.load_state()
        open_prs = self.listed = self.github.open_prs()
        prs, waiting = self.fetch_prs([p for p in open_prs if p.base == "main"])
        self.unlock()
        if fetched is not None:
            stale, close_days = fetched()
        main = _out(self.workdir, "rev-parse", "refs/remotes/origin/main")
        prefetch = self.prefetch
        if prefetch is not None:                # one pull for every label below, not one each for admission, off-main and stale
            prefetch([(p.number, p.head) for p in open_prs])
            self.prefetch = None
        try:
            old = _git(self.workdir, "rev-parse", "-q", "--verify", "refs/remotes/origin/checked^{commit}", check=False).stdout.strip()
            onto = {"base": main, "landed": main, "queued": main, "old_landed": old or None} if loop else {}
            entries = self.entries(prs, state, open_numbers={p.number for p in open_prs}, **onto)
            entries.update(waiting)
            entries.update(self.off_main(open_prs))
            idle = stale(open_prs) if stale else {}
        finally:
            self.prefetch = prefetch
        by_number = {p.number: p for p in open_prs}
        for n, since in idle.items():
            p = by_number[n]
            e = entries.setdefault(n, Entry(p, state="waiting", why=f"based on {based_on(p, open_prs)}, and not marked ready at "
                                                                    f"{p.head[:8]}"))
            e.idle_since = since
            e.why = (f"stale, idle since {time.strftime('%Y-%m-%d', time.gmtime(since))}"
                     + (f" (drive closes it {close_days} days idle; push, mark it ready or close it)" if close_days else "")
                     + f": {e.why}")
        conflicts, waits = ({}, {}) if loop else (state.get("conflicts", {}), state.get("waiting", {}))
        if loop:
            reach = above(self.workdir, main, (e.pr.head for e in entries.values() if e.state == "ready"))
            for e in entries.values():
                if e.state == "ready" and not reach[e.pr.head][0]:
                    e.state, e.why = "queued", "landed: its head is on `main`"
        else:
            for i, m in enumerate(self.chain(), 1):
                if m.pr in entries and entries[m.pr].pr.head == m.head:
                    entries[m.pr].state, entries[m.pr].position, entries[m.pr].commit = "queued", i, m.commit
            if self.into != "main":
                landed = queue_log(self.workdir, self.base)[0]
                reach = above(self.workdir, self.base, (e.pr.head for e in entries.values() if e.state == "ready"))
                for e in entries.values():
                    if e.state == "ready" and (e.pr.head in landed or not reach[e.pr.head][0]):
                        e.state, e.why = "queued", (f"landed on `{self.into}`: it reaches `main` when drive's full check of a "
                                                    f"`{self.into}` commit at or above it passes every step on the merge gate")
        files_of = self.conflicts_with(main, [e.pr.head for n, e in entries.items() if e.state == "ready" and not e.pr.move
                                              and conflicts.get(str(n), {}).get("head") != e.pr.head
                                              and waits.get(str(n), {}).get("head") != e.pr.head])
        for n in contained_first(entries):
            e = entries[n]
            if e.state == "ready" and (out := [a for a in e.after if a in entries and entries[a].state in ("refused", "waiting")]):
                e.state, e.why = "waiting", f"contains {e.contains(out[0])}, which is {entries[out[0]].state}: {entries[out[0]].why}"
            elif e.state == "ready" and (c := conflicts.get(str(n))) and c["head"] == e.pr.head:
                e.state, e.why = "refused", f"{c['why']} (at the last sync)"
            elif e.state == "ready" and (w := waits.get(str(n))) and w["head"] == e.pr.head:
                e.state, e.why = "waiting", f"{w['why']} (at the last sync)"
            elif e.state == "ready" and not e.pr.move and (files := files_of[e.pr.head]):
                e.state, e.why = "refused", (f"conflicts with `main` in {', '.join(files)}: merge `main` into the branch and "
                                             f"re-mark it")
            elif e.state == "ready":
                e.why = ("admitted: the verified queue builds it into a batch onto `main` once it has a land verdict, or on its "
                         "mark alone when it is low-risk")
        return entries

    def conflicts_with(self, main: str, heads: Iterable[str]) -> dict[str, list[str]]:
        """{head: `conflicting_files` of it onto ``main``} for each of ``heads``.  A merge of an old head takes about a second,
        so the answers are kept beside the clone for ``main``, the clone's HEAD (whose merge drivers the merge runs) and
        `clone.RECORD_DRIVER`, and those not kept yet are merged `os.cpu_count()` at once."""
        from .clone import RECORD_DRIVER

        path = self.workdir.parent / f"{self.workdir.name}.conflicts.json"
        key = {"main": main, "at": _out(self.workdir, "rev-parse", "HEAD"), "drivers": RECORD_DRIVER}
        doc = json.loads(path.read_text()) if path.is_file() else {}            # written whole, by `replace`
        got = dict(doc.get("of") or {}) if doc.get("key") == key else {}
        if todo := [h for h in dict.fromkeys(heads) if h not in got]:
            with concurrent.futures.ThreadPoolExecutor(os.cpu_count() or 1) as ex:
                got.update(zip(todo, ex.map(lambda h: conflicting_files(self.workdir, main, h), todo)))
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"key": key, "of": got}))
            tmp.replace(path)
        return got

    def off_main(self, prs: list[PR]) -> dict[int, Entry]:
        """The open PRs among ``prs`` based on a branch other than `main` that are marked ready at their head, each waiting with
        the fix: drive admits only PRs based on `main`, so without this such a PR's mark went unseen until its parent landed and
        someone retargeted it."""
        off = [p for p in prs if p.base != "main"]
        if off and self.prefetch is not None:
            self.prefetch([(p.number, p.head) for p in off])
        return {p.number: Entry(p, state="waiting", why=f"based on {based_on(p, prs)}, not main: retarget it to main")
                for p in off if self.ready(p.number, p.head)}


#: what a change must touch to move many circuit-check targets' cache keys: the circuits package (every Definition imports its
#: library), the checker, the lock and the workspace's member list (installed distributions), a sidecar (the registries), and
#: what every target's entry reads or the registries import (an entry can read tens of files of vLLM's `program/` and these
#: packages, so a change only to `verity_vllm/pipeline/` and `config.py` reruns hundreds of targets)
_COLD_PREFIXES = ("verity/circuits/", "tools/verity/circuit_check/", "adapters/vllm/verity_vllm/",
                  "verity/protocols/pouw/verifier/", "verity/protocols/pous/", "verity/protocols/flock/prover/python/",
                  "verity/commitments/", "verity/definitions/private_circuits/", "benchmarks/numerical/python/")


def contained_first(entries: dict[int, Entry]) -> list[int]:
    """``entries``' numbers, each after the PRs it contains (`Entry.after`), so a PR is judged after what it contains."""
    out: list[int] = []
    seen: set[int] = set()

    def place(n: int) -> None:
        if n in seen:
            return
        seen.add(n)
        for a in entries[n].after:
            if a in entries:
                place(a)
        out.append(n)

    for n in sorted(entries):
        place(n)
    return out


def forces_cold(files: Iterable[str]) -> bool:
    """Whether a change to ``files`` makes the next check's circuit-check run cold (hours, not seconds), so a sync merges it
    above the warm PRs (`Queue.order`), and what drive takes as circuit-check's inputs: the culprit of a red one
    (`Driver._input_rule`), and what doesn't land while it is red on `checked` (`Driver._circuit_held`)."""
    for f in files:
        if f.endswith(".md") or "/tests/" in f:
            continue
        if f.startswith(_COLD_PREFIXES) or f in ("uv.lock", "pyproject.toml") or f.endswith(".circuit-check.py"):
            return True
    return False


def merged_tree(repo: Path, base: str, head: str, *, plain: bool = True,
                config: tuple[str, ...] | None = None) -> tuple[str | None, list[str], subprocess.CompletedProcess]:
    """``head`` merged onto ``base`` by `git merge-tree` (no worktree touched), with ``base``'s merge drivers (`--attr-source`)
    run from ``base``'s tools (`base_drivers`, unless ``config`` gives git's `-c` options; drive's clone sits at `checked`'s
    tip, whose own `merge.py` predates one a member of `next` brings): (the merged tree, the conflicted files, `git
    merge-tree`'s result).  A merge whose every conflict is in a pin `repinned` rebuilds, or in a `.lean` file one side
    deleted that the other changed only by porting it (`port_deletions`), gives the tree with each pin rebuilt and each such
    file deleted, and those files (`taken` tells them apart); one that doesn't merge gives None.  A git without `--attr-source`
    (before 2.40) merges without the drivers, or with ``plain`` False gives None and its refusal (exit 129)."""
    if config is None:
        with base_drivers(repo, base) as config:
            return merged_tree(repo, base, head, plain=plain, config=config)
    r = _git(repo, *config, f"--attr-source={base}", "merge-tree", "--write-tree", "--name-only", "--no-messages", base, head,
             check=False)
    if r.returncode == 129 and "attr-source" in r.stderr:
        if not plain:
            return None, [], r
        r = _git(repo, *config, "merge-tree", "--write-tree", "--name-only", "--no-messages", base, head, check=False)
    if r.returncode == 1:
        files = sorted({f for f in r.stdout.splitlines()[1:] if f})
        tree, rest = port_deletions(repo, r.stdout.split()[0], base, head, files)
        return (repinned(repo, tree, base, head, rest) if rest else tree), files, r
    return (None if r.returncode else r.stdout.split()[0]), [], r


def port_deletions(repo: Path, tree: str, ours: str, theirs: str, conflicted: list[str]) -> tuple[str, list[str]]:
    """(``tree``, a merge of ``theirs`` onto ``ours``, with each conflicted `.lean` file one side deleted and the other
    changed only by porting it deleted, the files still conflicted).  Git runs no merge driver on a modify/delete, so a
    deletion made before the port conflicts with the port's rewrite of the file either way round: a branch made before the
    port that deletes a file the tip only ported, or a branch carrying the port merged onto a tip that deleted a file it only
    ported.  Without the port the merge is the deletion.  The porting side must be on the module system (`on_modules`), and
    its text the port of a merge base's (`_only_ported`); with several merge bases (a criss-cross history), one base whose
    text ports to it is enough.  A file changed beyond porting it from every base stays conflicted."""
    lean = [f for f in conflicted if f.endswith(".lean")]
    if not lean:
        return tree, conflicted
    bases = _git(repo, "merge-base", "--all", ours, theirs, check=False).stdout.split()
    sides = _blobs(repo, [f"{c}:{f}" for c in (*bases, ours, theirs) for f in lean])
    take: list[str] = []
    for porter, deleter in ((ours, theirs), (theirs, ours)):
        held = [f for f in lean if f"{porter}:{f}" in sides and f"{deleter}:{f}" not in sides
                and any(f"{b}:{f}" in sides for b in bases)]
        if held and on_modules(repo, porter)[0]:
            take += _only_ported(repo, porter, bases, held, sides)
    if not take:
        return tree, conflicted
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        env = {"GIT_INDEX_FILE": str(Path(tmp, "index"))}
        _git(repo, "read-tree", tree, env=env)
        _git(repo, "update-index", "--force-remove", *take, env=env)
        return _git(repo, "write-tree", env=env).stdout.strip(), [f for f in conflicted if f not in take]


def _only_ported(repo: Path, side: str, bases: list[str], files: list[str], sides: dict[str, bytes]) -> list[str]:
    """Those of ``files`` whose text in ``side`` is what ``side``'s `MODULES` makes of one of ``bases``' text of it
    (`_codemod`, over ``side``'s tree with that base's texts in place), ``sides`` holding each `<commit>:<path>` blob: one
    codemod run for each base whose texts of the files not yet taken no earlier run had."""
    out: list[str] = []
    tried: set[tuple[tuple[str, bytes], ...]] = set()
    for b in bases:
        todo = [f for f in files if f not in out and f"{b}:{f}" in sides]
        if not todo or (texts := tuple((f, sides[f"{b}:{f}"]) for f in todo)) in tried:
            continue
        tried.add(texts)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            env = {"GIT_INDEX_FILE": str(Path(tmp, "index"))}
            _git(repo, "read-tree", side, env=env)
            for f in todo:
                mode, _, sha = _out(repo, "ls-tree", b, "--", f).split("\t", 1)[0].split()
                _git(repo, "update-index", "--cacheinfo", f"{mode},{sha},{f}", env=env)
            new, _, _, _ = _codemod(repo, _git(repo, "write-tree", env=env).stdout.strip())
        port = _blobs(repo, [f"{new}:{f}" for f in todo]) if new else {}
        out += [f for f in todo if port.get(f"{new}:{f}") == sides[f"{side}:{f}"]]
    return out


def taken(repo: Path, tree: str, head: str, files: list[str]) -> tuple[list[str], list[str], list[str]]:
    """(the pins, the `.lean` files ``head`` deleted, those the tip deleted) among ``files``, the conflicts `merged_tree`
    resolved into ``tree`` merging ``head``: a pin is rebuilt in it, and a file `port_deletions` took isn't in it, deleted by
    the side that doesn't hold it."""
    held = _blobs(repo, [f"{c}:{f}" for c in (tree, head) for f in files])
    gone = [f for f in files if f"{tree}:{f}" not in held]
    return ([f for f in files if f"{tree}:{f}" in held], [f for f in gone if f"{head}:{f}" not in held],
            [f for f in gone if f"{head}:{f}" in held])


#: a path in a merge driver's command (`tools/verity/lean/merge.py`), relative to the top of the worktree git runs it in
_DRIVER_PATH = re.compile(r"(?<![\w./-])[\w.-]+(?:/[\w.-]+)+")


@contextlib.contextmanager
def base_drivers(repo: Path, base: str):
    """`git -c` options that run each merge driver ``base``'s `.gitattributes` names from ``base``'s tools: its command as the
    sync's clone sets it (`RECORD_DRIVER`), else as ``repo``'s config does, with each path in it that ``base`` holds moved to
    a temporary directory holding that path's directory as ``base`` has it (a driver loads files beside it), removed on exit.
    Otherwise git runs a driver from ``repo``'s worktree: a branch's older checkout runs its own `tools/verity/lean/merge.py`,
    and one from before `--caps` errors, so git reports a conflict in `tests/test_lean_packages.py` that the sync never sees."""
    from .clone import RECORD_DRIVER

    attrs = _git(repo, "show", f"{base}:.gitattributes", check=False)
    names = dict.fromkeys(m for ln in attrs.stdout.splitlines() if not ln.lstrip().startswith("#")
                          for m in re.findall(r"(?:^|\s)merge=(\S+)", ln)) if not attrs.returncode else {}
    cmds = {k: c for k in (f"merge.{n}.driver" for n in names)
            if (c := RECORD_DRIVER.get(k) or _git(repo, "config", "--get", k, check=False).stdout.strip())}
    with tempfile.TemporaryDirectory(prefix="research-drivers-", ignore_cleanup_errors=True) as tmp:
        held = {s.split(":", 1)[1] for s in _blobs(repo, [f"{base}:{p}" for c in cmds.values() for p in _DRIVER_PATH.findall(c)])}
        files = []
        for d in sorted({posixpath.dirname(p) for p in held}):
            ls = _git(repo, "ls-tree", "-z", base, "--", f"{d}/").stdout
            files += [e.split("\t", 1)[1] for e in ls.split("\0") if e and e.split()[1] == "blob"]
        for spec, data in _blobs(repo, [f"{base}:{f}" for f in files]).items():
            (out := Path(tmp) / spec.split(":", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
        def moved(m: re.Match) -> str:
            return shlex.quote(str(Path(tmp) / m[0])) if m[0] in held else m[0]

        yield tuple(o for k, c in cmds.items() for o in ("-c", f"{k}={_DRIVER_PATH.sub(moved, c)}"))


def conflicting_files(repo: Path, base: str, head: str) -> list[str]:
    """The files a merge of ``head`` onto ``base`` would leave conflicted, so `status` names a conflict on any VM, not only on
    the one whose last sync recorded it; [] when it merges cleanly.  It merges as `Queue._merge` does (`merged_tree`): a
    conflict only in pins the sync rebuilds (`repinned`) counts as none (a conflict over PoUS's TRUSTED.sha256, which the sync
    merges, refuses nothing), and so does one over a `.lean` file one side deletes that the other only ported
    (`port_deletions`)."""
    tree, files, _ = merged_tree(repo, base, head)
    return [] if tree else files


def pins_at(repo: Path, ref: str) -> dict[str, tuple[str | None, str]]:
    """{path: (key, under)} of each `Pin` in the pins tool's `PINS` on ``ref`` (`PINS_FILES`), read from its text with `ast`,
    since research imports nothing from Verity; {} when ``ref`` holds none or it doesn't parse."""
    for path in PINS_FILES:
        if (r := _git(repo, "show", f"{ref}:{path}", check=False)).returncode == 0:
            break
    else:
        return {}
    try:
        [calls] = [n.value.elts for n in ast.parse(r.stdout).body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Tuple)
                   and any(isinstance(t, ast.Name) and t.id == "PINS" for t in n.targets)]
        out = {}
        for c in calls:
            args = {**dict(zip(("path", "key", "under"), c.args)), **{k.arg: k.value for k in c.keywords}}
            path, key, under = (ast.literal_eval(args[a]) for a in ("path", "key", "under"))
            out[path] = (key, under)
        return out
    except (SyntaxError, ValueError, KeyError, AttributeError):
        return {}


def _blobs(repo: Path, specs: list[str]) -> dict[str, bytes]:
    """{spec: its blob's bytes} for each of ``specs`` (`<tree>:<path>`) that names a blob, by one `git cat-file --batch`."""
    out = subprocess.run(["git", "cat-file", "--batch"], cwd=repo, input="".join(f"{s}\n" for s in specs).encode(),
                         capture_output=True, env=git_env(GIT_TERMINAL_PROMPT="0"), check=True).stdout
    got, i = {}, 0
    for s in specs:
        j = out.index(b"\n", i)
        head, i = out[i:j].split(), j + 1
        if head[-1] in (b"missing", b"ambiguous"):
            continue
        if head[1] == b"blob":
            got[s] = out[i:i + int(head[2])]
        i += int(head[2]) + 1
    return got


def repinned(repo: Path, tree: str, ours: str, theirs: str, conflicted: list[str]) -> str | None:
    """The merged ``tree`` with each conflicted pin rebuilt, when every path in ``conflicted`` is a pin (`pins_at` ``ours``)
    whose content besides its digests the sides agree on or only one of them changed (`_kept`; a `sha256sum` listing is all
    digests); else None.  A pin records the union of the paths both sides' versions record, OURS's in its order then
    THEIRS's new ones, less those the merged tree lacks, each with the sha256 of its blob there; a JSON pin keeps the rest of
    the text of the side that changed it, OURS's when the sides agree on it.  `pins.py --merged` would run each pin's command
    instead, which needs a checkout and Lake: two PRs that each changed a pinned input both changed the pin, and drive would
    eject the second (as over the PoUS TRUSTED.sha256)."""
    pins = pins_at(repo, ours)
    if not conflicted or any(p not in pins for p in conflicted):
        return None
    bases = _git(repo, "merge-base", "--all", ours, theirs, check=False).stdout.split()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        env = {"GIT_INDEX_FILE": str(Path(tmp, "index"))}
        _git(repo, "read-tree", tree, env=env)
        for path in conflicted:
            key, under = pins[path]
            texts = _blobs(repo, [f"{c}:{path}" for c in (ours, theirs, *bases)])
            if not (sides := [texts[s] for s in (f"{ours}:{path}", f"{theirs}:{path}") if s in texts]):
                return None
            recorded: dict[str, str] = {}
            try:
                for text in (b.decode() for b in sides):
                    pairs = ([ln.split("  ", 1)[::-1] for ln in text.splitlines() if ln.strip()] if key is None
                             else json.loads(text)[key].items())
                    recorded |= {p: h for p, h in pairs if p not in recorded}
                kept = sides[0] if key is None else _kept(key, texts.get(f"{ours}:{path}"), texts.get(f"{theirs}:{path}"),
                                                          [texts.get(f"{b}:{path}") for b in bases])
            except (ValueError, KeyError, TypeError, AttributeError):
                return None
            if kept is None:
                return None
            full = {p: posixpath.normpath(posixpath.join(under, p)) for p in recorded}
            found = _blobs(repo, [f"{tree}:{f}" for f in full.values()])
            digests = {p: hashlib.sha256(found[f"{tree}:{f}"]).hexdigest() for p, f in full.items() if f"{tree}:{f}" in found}
            text = ("".join(f"{h}  {p}\n" for p, h in digests.items()) if key is None
                    else _with_json(kept.decode(), key, digests))
            blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=repo, input=text.encode(), capture_output=True,
                                  env=git_env(GIT_TERMINAL_PROMPT="0"), check=True).stdout.decode().strip()
            mode = _out(repo, "ls-tree", tree, "--", path).split(" ", 1)[0] or "100644"
            _git(repo, "update-index", "--add", "--cacheinfo", f"{mode},{blob},{path}", env=env)
        return _git(repo, "write-tree", env=env).stdout.strip()


def _kept(key: str, ours: bytes | None, theirs: bytes | None, bases: list[bytes | None]) -> bytes | None:
    """The side of a conflicted JSON pin whose text a merge keeps, from each side's and each merge base's (None: no pin there),
    by what each holds besides its digests (the object less ``key``): OURS when the sides agree on it, else the side that alone
    changed it from every base; None when both changed it (two PRs that each added vectors), since keeping either side's
    text drops the other's change, or when the side kept has no pin."""
    def rest(text: bytes | None) -> dict | None:
        return None if text is None else {k: v for k, v in json.loads(text).items() if k != key}

    o, t, was = rest(ours), rest(theirs), [rest(b) for b in bases]
    if o == t or (was and all(w == t for w in was)):
        return ours
    return theirs if was and all(w == o for w in was) else None


def _with_json(text: str, key: str, value: dict) -> str:
    """``text``, a JSON object written with `indent=1`, with its top-level ``key`` set to ``value`` and every other byte kept."""
    doc = json.loads(text)
    if m := re.search(rf"^([ \t]*){re.escape(json.dumps(key))}:\s*(?=\{{)", text, re.M):
        old, end = json.JSONDecoder().raw_decode(text, m.end())
        if old == doc[key]:
            return text[:m.end()] + json.dumps(value, indent=1).replace("\n", "\n" + m.group(1)) + text[end:]
    return json.dumps({**doc, key: value}, indent=1) + "\n"


def ported(repo: Path, tree: str, base: str | None = None) -> tuple[str | None, list[str], list[str], str]:
    """(``tree``, a merge's tree, with its Lean on Lean's module system, the files that took the port, the pins that recorded
    one of them, "") or (None, [], [], why not).  Git runs the `.lean` merge driver, which ports a side made before the port,
    only on a file both sides changed, so a `.lean` file either side added before the port stays as it was written (a PR's,
    or one `checked` added since the port's base, under the port's own merge), and a module importing it doesn't build.  Only
    a tree on the module system (`on_modules`) is touched: that tree's `MODULES` runs over all of it, at the root of a
    worktree of it, and leaves a ported tree as it is; each pin's record of a file it rewrote moves with it
    (`_repinned_files`).  No cheaper test stands in for the run: a header check of every `.lean` file would read a script or a
    test fixture, which the codemod leaves alone, as a file to port.  ``base``, the tree a merge went onto, stands in for it
    where the merge changed nothing the codemod reads (`PORT_INPUTS`, the codemod and its reader): the merged tree's Lean is
    ``base``'s, ported as ``base`` was.  The run takes about 20 s on drive's control pod, and a sync merges hundreds of times.
    The codemod, or the header reader, failing is the reason."""
    on, why = on_modules(repo, tree)
    if not on:
        return (None if why else tree), [], [], why
    if base is not None and not _git(repo, "diff", "--quiet", base, tree, "--", *PORT_INPUTS, MODULES, HEADERS,
                                     check=False).returncode:
        return tree, [], [], ""
    new, files, pins, why = _codemod(repo, tree)
    return new, list(files), list(pins), why


#: `on_modules`' reading, run apart from the queue as the codemod is: the tree's `HEADERS` (stdin's "headers") says whether each
#: of stdin's "roots" has a header that starts with `module`
_ROOTS_READ = ("import json, sys, types\nd = json.load(sys.stdin)\nm = types.ModuleType('headers')\n"
               "exec(compile(d['headers'], 'headers.py', 'exec'), m.__dict__)\n"
               "print(json.dumps([(h := m.header(s)) is not None and h.module for s in d['roots']]))\n")


def on_modules(repo: Path, tree: str) -> tuple[bool, str]:
    """(whether ``tree`` is on Lean's module system, "") or (False, why it can't tell).  It is when it holds `MODULES` and
    `HEADERS` and each of `PORTED_ROOTS` is a module as that `HEADERS` reads it (`header(src).module`, as `MODULES` reads
    one).  The port's tooling may land before the port, so a tree holding the codemod whose roots aren't modules isn't on it,
    and its merges are git's alone: the codemod run there would port the whole tree."""
    specs = [f"{tree}:{p}" for p in (MODULES, HEADERS, *PORTED_ROOTS)]
    blobs = _blobs(repo, specs)
    if len(blobs) < len(specs):
        return False, ""
    doc = {"headers": blobs[specs[1]].decode(errors="replace"),
           "roots": [blobs[s].decode(errors="replace") for s in specs[2:]]}
    r = subprocess.run([sys.executable, "-c", _ROOTS_READ], input=json.dumps(doc), capture_output=True, text=True,
                       env=_script_env())
    if r.returncode:
        return False, (f"`{HEADERS}` failed to read the headers of {', '.join(PORTED_ROOTS)} (exit {r.returncode}): "
                       f"{_tail(r, 3)}")
    return all(json.loads(r.stdout.splitlines()[-1])), ""


@functools.lru_cache(maxsize=256)
def _codemod(repo: Path, tree: str) -> tuple[str | None, tuple[str, ...], tuple[str, ...], str]:
    """`ported`'s run of ``tree``'s `MODULES` over it, once a tree: a sync merges a member again onto an unchanged tree."""
    commit = _git(repo, "commit-tree", tree, "-m", "ported", env=IDENTITY).stdout.strip()
    with merged_worktree(repo, commit) as wt:
        r = subprocess.run([sys.executable, MODULES], cwd=wt, capture_output=True, text=True, env=_script_env())
        if r.returncode:
            return None, (), (), (f"`{MODULES}` failed on the merged tree (exit {r.returncode}): {_tail(r, 3)}; port by hand "
                                  "what it names")
        _git(wt, "add", "-u")
        diff = _git(wt, "diff", "--cached", "-z", "--no-renames", "--name-only", commit).stdout
        if not (files := tuple(f for f in diff.split("\0") if f)):
            return tree, (), (), ""
        pins = _repinned_files(repo, wt, tree, files)
        _git(wt, "add", "-u")
        return _out(wt, "write-tree"), files, pins, ""


def _repinned_files(repo: Path, wt: Path, tree: str, files: tuple[str, ...]) -> tuple[str, ...]:
    """The pins (`pins_at` ``tree``) that recorded one of ``files`` at its digest in ``tree``, each rewritten in ``wt``, a
    worktree of ``tree`` where those files changed, to record its digest there, every other byte kept."""
    before = _blobs(repo, [f"{tree}:{f}" for f in files])
    moved = {f: (hashlib.sha256(before[f"{tree}:{f}"]).hexdigest(), hashlib.sha256((wt / f).read_bytes()).hexdigest())
             for f in files if f"{tree}:{f}" in before and (wt / f).is_file()}
    out = []
    for path, (key, under) in pins_at(repo, tree).items():
        if not (pin := wt / path).is_file():
            continue
        text = new = pin.read_bytes().decode(errors="surrogateescape")
        for f, (old, now) in moved.items():
            rel = posixpath.relpath(f, under) if under else f
            new = (re.sub(rf"(?m)^{old}(?=  {re.escape(rel)}$)", now, new) if key is None
                   else re.sub(rf'({re.escape(json.dumps(rel))}\s*:\s*"){old}"', rf'\g<1>{now}"', new))
        if new != text:
            pin.write_bytes(new.encode(errors="surrogateescape"))
            out.append(path)
    return tuple(out)


def rules_at(repo: Path, ref: str) -> Rules | None:
    """The admission rules on ``ref`` (`RULES`, else `RULES_BEFORE_MOVE`), or None when it holds neither."""
    for path in (RULES, RULES_BEFORE_MOVE):
        r = _git(repo, "show", f"{ref}:{path}", check=False)
        if r.returncode == 0:
            return Rules.parse(r.stdout, path)
    return None


def chain_of(repo: Path, base: str, ref: str) -> list[Member]:
    """The members on ``ref`` above ``base``, oldest first, from the trailers of its first-parent merges; none when ``ref`` is
    missing, ``base`` isn't below it, or a commit on that chain isn't the queue's."""
    if _git(repo, "rev-parse", "--verify", "-q", ref, check=False).returncode:
        return []
    if _git(repo, "merge-base", "--is-ancestor", base, ref, check=False).returncode:
        return []
    out = []
    log = _out(repo, "log", "--first-parent", "--reverse", "--format=%H%x00%B%x1e", f"{base}..{ref}")
    for rec in filter(None, (x.strip() for x in log.split("\x1e"))):
        commit, body = rec.split("\x00", 1)
        t = dict(TRAILER_RE.findall(body))
        if not {"PR", "Head", "Owner"} <= set(t):
            return []                                   # a commit the queue didn't make: rebuild from `main`
        out.append(Member(int(t["PR"].lstrip("#")), t["Head"], t["Owner"], commit))
    return out


def queue_merges(repo: Path, base: str, ref: str) -> list[Member]:
    """The members on ``ref``'s first-parent chain that ``base`` doesn't contain, oldest first, from their merges' trailers,
    passing over each commit without them.  `chain_of` finds none on a chain that crosses one (from `main`, the lock commits
    drive adds to `checked`) or whose ``base`` isn't below ``ref`` (a `checked` a landing moved off `next` until the sync
    rebuilds `next` on it)."""
    out = []
    log = _out(repo, "log", "--first-parent", "--reverse", "--format=%H%x00%B%x1e", f"{base}..{ref}")
    for rec in filter(None, (x.strip() for x in log.split("\x1e"))):
        commit, body = rec.split("\x00", 1)
        t = dict(TRAILER_RE.findall(body))
        if {"PR", "Head", "Owner"} <= set(t):
            out.append(Member(int(t["PR"].lstrip("#")), t["Head"], t["Owner"], commit))
    return out


@dataclass(frozen=True)
class Move:
    """A move drive regenerated (`Queue._regenerate`), on `next` or landed: its commit, one on its base, and its word map."""
    commit: str
    map: str


def queue_log(repo: Path, ref: str, n: int = LANDED_WINDOW) -> tuple[set[str], list[Move]]:
    """(the heads the queue merged, the moves drive regenerated, oldest first) on ``ref``'s last ``n`` first-parent commits, by
    their trailers: a PR landed at a head ``ref`` doesn't contain (a regenerated move's, or one restacked through a move) is
    landed all the same."""
    heads, moves = set(), []
    log = _git(repo, "log", "--first-parent", f"-n{n}", "--format=%H%x00%B%x1e", ref, check=False).stdout
    for rec in filter(None, (x.strip() for x in log.split("\x1e"))):
        commit, body = rec.split("\x00", 1)
        t = dict(TRAILER_RE.findall(body))
        if {"PR", "Head"} <= set(t):
            heads.add(t["Head"].strip())
            if m := MOVE_TRAILER_RE.search(body):
                moves.append(Move(commit, m.group(1)))
    return heads, moves[::-1]


#: how many of the merges it found a refusal for an old `next` names (`old_next_refusal`)
OLD_NEXT_SHOWN = 3


def old_next(repo: Path, heads: Iterable[str], landed: str, open_prs: Collection[int]) -> dict[str, Member]:
    """{commit: the PR it brings} of each queue commit (`queue_commit`) one of ``heads`` reaches that ``landed``, the landing
    branch's tip, doesn't, whose PR is none of ``open_prs`` (numbers) and never landed: its head is neither on ``landed`` nor
    one the queue landed there at another commit (`queue_log`: a move's, a restacked one's).  Each is a merge of an old
    `next`, one rebuilt since (a demo branch that merged it carries hundreds), and brings the change of a PR closed
    unlanded, which a head carrying it would land with its own.  A head stacked on an open PR, or one that merged `next`,
    carries only merges of PRs that are open or landed."""
    heads = list(dict.fromkeys(heads))
    if not heads:
        return {}
    env = git_env(GIT_TERMINAL_PROMPT="0")
    r = subprocess.run(["git", "log", "--stdin", "--grep=^Queue: #", "--format=%H%x00%P%x00%B%x1e"], cwd=repo,
                       input="".join(f"{h}\n" for h in [*heads, f"^{landed}"]), capture_output=True, text=True,
                       errors="surrogateescape", env=env)
    if r.returncode:
        raise QueueError(f"git log {landed}.. ({len(heads)} heads): {(r.stderr or r.stdout).strip()[-400:]}")
    found = {m.commit: m for rec in r.stdout.split("\x1e") if rec.strip()
             if (m := queue_commit(*rec.strip().split("\x00", 2))) and m.pr not in open_prs}
    if not found:
        return {}
    done = queue_log(repo, landed)[0]
    if not (maybe := sorted({m.head for m in found.values()} - done)):
        return {}
    have = subprocess.run(["git", "cat-file", "--batch-check=%(objectname)"], cwd=repo, input="".join(f"{h}\n" for h in maybe),
                          capture_output=True, text=True, env=env).stdout.splitlines()
    off = subprocess.run(["git", "rev-list", "--ignore-missing", "--stdin"], cwd=repo,
                         input="".join(f"{h}\n" for h in [*maybe, f"^{landed}"]), capture_output=True, text=True, env=env)
    if off.returncode:
        raise QueueError(f"git rev-list {landed}.. ({len(maybe)} heads): {(off.stderr or off.stdout).strip()[-400:]}")
    on = {h for h in have if not h.endswith(" missing")}.difference(off.stdout.split())
    return {c: m for c, m in found.items() if m.head not in done and m.head not in on}


def old_next_refusal(old: list[Member], into: str) -> str:
    """Why a head carrying ``old``, merges of an old `next` (`old_next`), is neither marked nor admitted, with the fix."""
    shown = ", ".join(f"{m.commit[:12]} (#{m.pr})" for m in old[:OLD_NEXT_SHOWN])
    more = f" and {len(old) - OLD_NEXT_SHOWN} more" if len(old) > OLD_NEXT_SHOWN else ""
    return (f"its head carries {len(old)} merge{'s' if len(old) > 1 else ''} of an old `{BRANCH}` that `{into}` doesn't hold, "
            f"of PRs closed unlanded ({shown}{more}), which would land with it: retarget the PR to the demo branch it builds on, "
            f"or restack its own commits onto `main` (a branch from `origin/main` with each cherry-picked, and the branch of each "
            f"open PR it needs merged, never `{BRANCH}`) and mark that head")


def move_of(repo: Path, commit: str) -> str | None:
    """The word map of ``commit`` when it is a move drive regenerated (its `Move:` trailer), else None."""
    m = MOVE_TRAILER_RE.search(_git(repo, "log", "-1", "--format=%B", commit, check=False).stdout)
    return m.group(1) if m else None


def _script_env() -> dict[str, str]:
    """The environment a move's script runs in: the queue's identity for the commits it makes, and no bytecode written into the
    tree it runs on (`rename.py` refuses a dirty one)."""
    env = git_env(GIT_TERMINAL_PROMPT="0", PYTHONDONTWRITEBYTECODE="1", **IDENTITY)
    for k in DATES:
        env.pop(k, None)
    return env


def _tail(r: subprocess.CompletedProcess, n: int = 6) -> str:
    return " | ".join(((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-n:])


def restack_end(job: dict, now: float) -> tuple[dict | None, str, bool] | None:
    """A restack job's end (`Queue._launch_restack`), its directories removed: (what `restack.py` said of its head, as a sync
    caches it, {"result": the restacked commit} or {"why": what it says}, else None; how the job ended, with its output's tail,
    and why its results don't read when it exited 0; whether something else interrupted it: an exit of `RESTACK_INTERRUPTED`,
    or its process gone with no exit written, a restart of the pod).  None while it runs, up to `RESTACK_LOST_S` after its
    start."""
    from .drive import _job_end

    if (end := _job_end(job, RESTACK_OUT)) is None and now - job["at"] <= RESTACK_LOST_S:
        return None
    if not isinstance(end, tuple):
        shutil.rmtree(job["dir"], ignore_errors=True)
    unread = ""
    try:
        (x,) = json.loads(Path(job["out"], RESTACK_OUT).read_text())
        said = ({"result": x["result"]} if x["status"] in ("clean", "up-to-date") else
                {"why": f"{RESTACK} conflicts at its {x.get('step')} merge in {', '.join(x.get('files') or [])}"}
                if x["status"] == "conflicted" else
                {"why": f"{RESTACK}: {x['status']}: {((x.get('error') or '').splitlines() or [''])[0][:300]}"})
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
        said, unread = None, f"; its results don't read ({type(e).__name__}: {e})"
    shutil.rmtree(job["out"], ignore_errors=True)
    if end is None:
        return said, f"no exit written {RESTACK_LOST_S // 60} min after its start", False
    if isinstance(end, str):
        return said, end, True
    rc, tail, _ = end
    stopped = f", stopped {RESTACK_WAIT_S // 60} min after its start" if rc == "124" else ""
    tail = " | ".join(tail.splitlines()[-6:])
    return (said, f"exit {rc}{stopped}" + (f": {tail}" if tail else "") + (unread if rc == "0" else ""),
            rc in RESTACK_INTERRUPTED)


def changed(repo: Path, base: str, commit: str) -> set[str]:
    return set(_out(repo, "diff", "--name-only", "--no-renames", base, commit).split())


def built_on(repo: Path, ref: str, into: str = "origin/main") -> tuple[str, list[Member]]:
    """(the base ``ref`` was built on, its members oldest first), whether or not ``into`` (the queue's `base`) has moved since;
    ("", []) when ``ref`` is missing or none of its first-parent chain is in ``into``."""
    if _git(repo, "rev-parse", "--verify", "-q", ref, check=False).returncode:
        return "", []
    # not `merge-base`: a member stacked on a PR that has since landed on `main` gives a second, wrong common ancestor
    # (which empties the chain and cancels a disjoint check)
    base = next((c for c in _out(repo, "rev-list", "--first-parent", ref).split()
                 if not _git(repo, "merge-base", "--is-ancestor", c, into, check=False).returncode), "")
    if not base:
        return "", []
    out = []
    for rec in filter(None, (x.strip() for x in _out(repo, "log", "--first-parent", "--reverse",
                                                      "--format=%H%x00%B%x1e", f"{base}..{ref}").split("\x1e"))):
        commit, body = rec.split("\x00", 1)
        t = dict(TRAILER_RE.findall(body))
        if not {"PR", "Head", "Owner"} <= set(t):
            return base, []
        out.append(Member(int(t["PR"].lstrip("#")), t["Head"], t["Owner"], commit))
    return base, out


def touched_by(repo: Path, members: list[Member], files: list[str], cache: dict[str, set[str]] | None = None) -> list[str]:
    """`#N` for each member whose merge changed one of ``files``; ``cache`` keeps each merge's files across calls (a sync names
    the members behind each of its conflicts)."""
    cache = {} if cache is None else cache
    for m in members:
        if m.commit not in cache:
            cache[m.commit] = set(_out(repo, "diff", "--name-only", f"{m.commit}^1", m.commit).split())
    return [f"#{m.pr}" for m in members if set(files) & cache[m.commit]]


# ---- command line -------------------------------------------------------------------------------------------------------------------

def _lock(workdir: Path):
    workdir.parent.mkdir(parents=True, exist_ok=True)
    f = open(workdir.parent / f"{workdir.name}.lock", "w")  # held until the process exits
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise QueueError(f"another `research queue` holds {f.name}") from None
    return f


def _render(entries: dict[int, Entry]) -> str:
    rank = {"queued": 0, "ready": 1, "waiting": 2, "refused": 3}
    rows = sorted(entries.values(), key=lambda e: (rank[e.state], e.position or 0, e.pr.number))
    lines = []
    for e in rows:
        where = f"{e.position:>3}" if e.position else "  -"
        lines.append(f"{where}  #{e.pr.number:<5} {e.pr.head[:8]}  {e.state:<8} {(e.pr.owner or '-')[:24]:<24} {e.why}")
    return "\n".join(lines) or "no open PRs"


#: the quick tier, run from the checkout's root: the suites the PR changes since `--changed REV`, without the tests marked `slow`
SUITES = quick_tier.SUITES
QUICK_TIER = ["uv", "run", "--locked", "--all-packages", "--extra", "torch-cpu", "python", SUITES, "--quick"]
#: the Lake packages whose builds the suites' tests run, built before `suites.py` as `check` builds them before those suites: each
#: test worker runs `lake build` there, and workers building one from cold at once fail (`lake build` exit 1, a binary missing)
LEAN_BUILDS = ("verity/protocols/flock/verifier",)


def allowed(repo: Path, base: str | None) -> Callable[[dict], bool] | None:
    """What drive's policy at ``base`` (the quick tier's base, a commit of `main`: `research.policy.at`) allows of a failed suite's
    row, as drive judges it after landing; None when it allows nothing there or can't be read, so every failure refuses."""
    from . import policy

    pol = policy.at(repo, base) if base else None
    if not isinstance(pol, policy.Policy) or not pol.allow:
        return None
    now = time.time()
    return lambda row: pol.suite(str(row.get("name")), row, now).covered


class QuickTier:
    """`suites.py --quick --changed {merge base with main}` at a PR's head, from a clean checkout (`quick_tier.tier`, judged by
    `quick_tier.judge`, with what drive's policy on `main` allows: `allowed`): one worktree of the caller's repository per machine under `root`, checked out at the head and cleaned of
    everything but its `.venv` and the `.lake` of each of `LEAN_BUILDS` for each PR (one PR at a time: `{name}.lock`), so its
    environment syncs only what moved and Lake rebuilds only what changed.  Each of `LEAN_BUILDS` is built first (with the PATH's
    `lake`, else elan's; none, or a failed build, refuses).  Passes are cached by input (`suites.py`), so a re-run after a failure
    runs only what failed."""

    #: the worktree's name after the hash of the caller's git common dir: a remote tier ships its own (`-ship`), so a local tier,
    #: which holds its worktree's lock for the whole tier, never holds up a remote one's checkout from the same repository
    WORKTREE_SUFFIX = ""

    def __init__(self, repo: Path, remote: str, root: Path | None = None):
        self.repo, self.remote = Path(repo), remote
        self.root = Path(root or Path.home() / ".research" / "queue" / "ready")

    def __call__(self, pr: PR) -> str | None:
        """None when the tier passes at ``pr``'s head; else what failed, for the refusal."""
        with self._checkout(pr) as (wt, base):
            if base is None:
                return f"{pr.head[:12]} isn't on {self.remote} (pushed again since?)"
            out = self.root / "logs" / f"{pr.number}-{pr.head[:12]}"
            return quick_tier.judge(quick_tier.tier(wt, out, pr.head, base, QUICK_TIER, LEAN_BUILDS), pr.head,
                                    covers=allowed(self.repo, base))

    @contextlib.contextmanager
    def _checkout(self, pr: PR):
        """Under the worktree's lock: the worktree at ``pr``'s head, cleaned, and the head's merge base with `main` (None when the
        head isn't on the remote)."""
        common = Path(_out(self.repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
        wt = self.root / (hashlib.sha256(str(common).encode()).hexdigest()[:12] + self.WORKTREE_SUFFIX)
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / f"{wt.name}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _git(self.repo, "fetch", "-q", self.remote, "+refs/heads/main:refs/queue/ready/main",
                 f"+refs/pull/{pr.number}/head:refs/queue/ready/pr")
            if _git(self.repo, "cat-file", "-e", f"{pr.head}^{{commit}}", check=False).returncode:
                yield wt, None
                return
            _git(self.repo, "worktree", "prune")
            if not (wt / ".git").exists():
                shutil.rmtree(wt, ignore_errors=True)
                _git(self.repo, "worktree", "add", "-q", "--detach", "--force", str(wt), pr.head)
            else:
                _git(wt, "checkout", "-q", "--detach", "--force", pr.head)
            _git(wt, "clean", "-qffdx", "-e", "/.venv", *(x for d in LEAN_BUILDS for x in ("-e", f"/{d}/.lake")))
            yield wt, _out(self.repo, "merge-base", "refs/queue/ready/main", pr.head)


#: what a remote quick tier may take once it holds a check slot (`slot.py --run-max`), how long it may wait in the machine's line for
#: one first (`slot.py --wait-max`; the run's own timeout on the machine is the two together), and how much longer the caller waits
#: for its records to say it ended (custody's upload, a machine briefly unreachable) before it gives up on the run
REMOTE_TIMEOUT_S = 3 * 3600
REMOTE_WAIT_S = 6 * 3600
REMOTE_GRACE_S = 1800
REMOTE_POLL_S = 30
#: slot.py's exit when no check slot came within --wait-max (its NOT_STARTED): the tier never ran
SLOT_NOT_STARTED = 75
#: the remote tier's command, slot.py from the shipped tree as the launcher of a machine with /etc/vy/runs-via-slot accepts it
#: (pods/nebius/slot_only.py's `through_slot`, which the suites that read this module may not read)
SLOT = ("python3", "tools/verity/check/slot.py")
#: `research queue ready`'s exit when the tier never started: nothing failed, and nothing is marked
READY_NOT_STARTED = 3
#: what `place` asks a research pool for when node 2 is down: any slot of the pool fits, and the tier takes the slot's cores
SPILL_CPUS, SPILL_MEM_GB = 4, 16
#: the slot list of a pool on a CPU node, which slot.py takes when its run names one (`MACHINE:POOL` from `place`)
POOL_SLOTS = "/workspace/research/locks/{pool}/slots"
#: a lane's own test run, which no landing needs (a PR lands on `checked` on a review; the full check after decides `main`)
LANE_RUN = ('`research run --on auto --cwd source -- bash -lc "uv sync -q --all-packages --extra torch-cpu && '
            'uv run tools/verity/check/suites.py --changed $(git fetch -q origin main && git merge-base HEAD FETCH_HEAD) --quick"`')


class NotStarted(str):
    """Why a quick tier never ran (no check slot within its wait, no machine to run it on): not a failure of the PR's head."""


class StillGoing(str):
    """`ready --on`'s caller stopped waiting at research.waitcap's cap with the tier's run still going: nothing failed or was
    cancelled, and `research queue judge RUN` marks the PR once the run has ended."""


def place(spec: str, repo: Path, read: Callable[[list[str]], dict] | None = None,
          spill: Callable[[Path], tuple[tuple[str, str] | None, str]] | None = None) -> tuple[str | None, str]:
    """`--on auto` (check.py's AUTO) or `--on M1,M2`: the machine `check.py --on` would take for a lane check now (`choose` over
    each machine's `read_slots`, from the caller's checkout), with the Lean audit slots left out, since the quick tier takes none.
    With `--on auto` and none of AUTO readable (node 2 stopped), a CPU node's research pool as `research run --on auto` spills to
    (``spill``, research.auto), as `MACHINE:POOL`. None when nothing could be placed. And why, with each machine skipped."""
    from .store import tools_registry

    path = tools_registry.resolve("tools/verity/check/check.py", lambda p: (repo / p).is_file()) or "tools/verity/check/check.py"
    s = importlib.util.spec_from_file_location("queue_check", repo / path)
    check = importlib.util.module_from_spec(s)
    sys.modules[s.name] = check
    s.loader.exec_module(check)
    names = list(check.AUTO) if spec == "auto" else [n for n in spec.split(",") if n]
    docs = {n: ({**d, "lean": {"pools": None}} if isinstance(d, dict) else d) for n, d in (read or check.read_slots)(names).items()}
    got, why, skipped = check.choose(names, docs)
    why += "".join(f"; skipped {s}" for s in skipped)
    if got is None and spec == "auto":
        target, how = (spill or _spill)(repo)
        if target is not None:
            return f"{target[0]}:{target[1]}", f"{how}, as none of {', '.join(names)} could be read ({why})"
        why += f"; no research pool either: {how}"
    return got, why


def _spill(repo: Path) -> tuple[tuple[str, str] | None, str]:
    """The research pool `research run --on auto` would send a small job to now (research.auto), or None and why not."""
    from . import auto

    try:
        target, why = auto.choose(auto.read(repo, SPILL_CPUS), SPILL_CPUS, SPILL_MEM_GB)
    except auto.Unplaceable as e:
        return None, str(e)
    return (target, why) if target is not None else (None, why)


def _launch(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True)


class RemoteQuickTier(QuickTier):
    """The quick tier as a `research run` on machine ``on`` (`auto` or `M1,M2`: the one `place` picks when the tier starts): a
    worktree of its own (`WORKTREE_SUFFIX`), checked out and cleaned as `QuickTier` does, is the run's `--source`, and the worktree's
    lock is held only while it is checked out and shipped, so several PRs' tiers run at once.  On the machine the run waits for a
    free check slot (the shipped tree's `tools/verity/check/slot.py`, as `check.py --record --on` takes one: the run's command is
    slot.py's, which a machine with /etc/vy/runs-via-slot requires) and runs `quick_tier.py` (sent with the run) in that tree.
    The caller polls the run's records (`research fetch`) until it ends and judges the facts in its result.json as the local tier's are judged; a run that ends without them, with facts from another run
    or head, or other than `done` with exit 0, is a failure.  A run that waited `wait_s` for a slot and gave up never started
    (`NotStarted`), and neither did one with no machine to run on.  The tier has `timeout_s` from taking its slot (`--run-max`), so
    the run on the machine is bounded by the two together (`--timeout`), and the caller's wait by that plus REMOTE_GRACE_S of
    ``clock`` time (a fetch can take its ssh's own timeout, so polls are not counted), past which it ends with the poll under way.
    The caller waits at most ``max_wait_s`` (research.waitcap's cap) of that: then the run goes on and the wait ends `StillGoing`."""

    WORKTREE_SUFFIX = "-ship"

    def __init__(self, repo: Path, remote: str, on: str, *, by: str = "", project: str = "verity", root: Path | None = None,
                 timeout_s: float = REMOTE_TIMEOUT_S, wait_s: float = REMOTE_WAIT_S, max_wait_s: float = waitcap.WAIT_CAP_S,
                 poll_s: float = REMOTE_POLL_S, launch: Callable[[list[str], Path], subprocess.CompletedProcess] = _launch,
                 fetch: Callable[[Path], dict] | None = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, place: Callable[[str, Path], tuple[str | None, str]] = place):
        super().__init__(repo, remote, root)
        self.on, self.by, self.project, self.poll_s = on, by, project, poll_s
        self.timeout_s, self.wait_s, self.max_wait_s = timeout_s, wait_s, max_wait_s
        self.launch, self.fetch, self.sleep, self.clock, self.place = launch, fetch, sleep, clock, place

    def __call__(self, pr: PR) -> str | None:
        from . import runs as R

        on, pool = self.on, ""
        if on == "auto" or "," in on:
            got, why = self.place(on, self.repo)
            if got is None:
                return NotStarted(f"--on {on}: {why}")
            print(f"research queue: --on {on}: {got}: {why}", flush=True)
            on, _, pool = got.partition(":")
        slot = (["env", f"CHECK_SLOTS={POOL_SLOTS.format(pool=pool)}"] if pool else []) + list(SLOT)
        job = Path(quick_tier.__file__)
        rid = R.new_run_id()
        with self._checkout(pr) as (wt, base):
            if base is None:
                return f"{pr.head[:12]} isn't on {self.remote} (pushed again since?)"
            args = [*(x for d in LEAN_BUILDS for x in ("--lean", d)), "--head", pr.head, "--base", base, "--", *QUICK_TIER]
            opts = [*(["--by", self.by] if self.by else []), "--wait-max", f"{self.wait_s:g}", "--run-max", f"{self.timeout_s:g}"]
            r = self.launch([sys.executable, "-m", "research", "run", "--id", rid, "--on", on, "--project", self.project,
                             "--source", str(wt), "--cwd", "source", "--send", str(job), "--require-result",
                             "--timeout", f"{self.wait_s + self.timeout_s:g}", "--custody-ttl", str(math.ceil(self.wait_s + self.timeout_s + REMOTE_GRACE_S)),
                             "--declared-output", "out/*", "--", *slot, *opts,
                             "--", "sh", "-c", f'exec python3 "$RESEARCH_RUN_DIR/inputs/{job.name}" {shlex.join(args)}'], wt)
        if r.returncode:
            return f"`research run --on {on}` failed (exit {r.returncode}): {(r.stderr or r.stdout or '').strip()[-400:]}"
        print(f"research queue: the quick tier runs as {rid} on {on}, in {'a slot of its ' + pool + ' pool' if pool else 'a check slot'}; waiting for it to end "
              f"(`research inspect {rid}`)", flush=True)
        d = R.runs_root() / rid
        status, why = self._wait(d, on)
        return why or judge_run(d, rid, on, pr.head, status, allowed(self.repo, base))

    def _wait(self, d: Path, on: str) -> tuple[dict, str | None]:
        """The run's status.json once it says the run ended, from its records fetched every `poll_s` until the wait's deadline; or why
        the wait gave up."""
        from . import remote
        from . import runs as R

        status: dict = {}
        bound = self.wait_s + self.timeout_s + REMOTE_GRACE_S
        limit = min(bound, self.max_wait_s)
        deadline = self.clock() + limit
        fetch = self.fetch or (lambda d: remote.fetch(d, timeout=deadline - self.clock()))
        while self.clock() < deadline:
            self.sleep(min(self.poll_s, deadline - self.clock()))
            try:
                rec = fetch(d)
            except Exception as e:  # noqa: BLE001  (a fetch that fails is a poll that saw nothing)
                rec = {"last_fetch_error": {"stderr": f"{type(e).__name__}: {e}"}}
            status = _json_file(d / "status.json")
            if status.get("state") in R.TERMINAL:
                return status, None
            if rec.get("launcher_exited") or rec.get("runner_gone"):
                # a fetch copies status.json before it probes the runner, which can record its end and exit in between
                # (a run that passed can read as gone): fetch once more before calling the run lost
                try:
                    fetch(d)
                except Exception:  # noqa: BLE001
                    pass
                status = _json_file(d / "status.json")
                if status.get("state") in R.TERMINAL:
                    return status, None
            if gone := rec.get("launcher_exited"):
                return status, f"run {d.name}'s runner on {on} exited before the run ended ({gone.get('last_line') or 'no log line'})"
            if gone := rec.get("runner_gone"):
                return status, f"run {d.name} on {on} ended without recording it: {gone['why']}"
        if limit < bound:
            return status, StillGoing(waitcap.still_going(
                f"the quick tier's run {d.name} on {on} ({status.get('state') or 'without a status'})",
                f"research queue judge {d.name} --by {self.by or 'LANE'}", limit))
        return status, (f"run {d.name} on {on} didn't end within {bound:g} s (last seen "
                        f"{status.get('state') or 'without a status'}): `research inspect {d.name}`")


def tier_report(d: Path, rid: str, on: str, status: dict):
    """Run ``rid``'s result.json (its records in ``d``, its `status.json` ``status``) as a `quick_tier` report, and None; or None and
    why it isn't one."""
    from .result import check

    res = check(d, rid)
    if not res.present and status.get("rc") == SLOT_NOT_STARTED:
        return None, NotStarted(f"run {rid} on {on} waited for a check slot until its --wait-max and never ran the tier "
                                f"(`research inspect {rid}`)")
    if not res.present:
        return None, f"run {rid} on {on} ended {status.get('state')} (exit {status.get('rc')}) without a result.json (`research inspect {rid}`)"
    if not res.valid:
        return None, f"run {rid}'s result.json is unreadable: {'; '.join(res.errors)[:400]}"
    if (kind := res.doc.get("kind")) != quick_tier.KIND:
        return None, f"run {rid}'s result is a {kind or 'result with no kind'}, not a {quick_tier.KIND} report"
    return res, None


def judge_report(res, rid: str, on: str, head: str, status: dict, covers: Callable[[dict], bool] | None = None) -> str | None:
    """None when the report ``res`` of run ``rid``, ended with ``status``, shows the tier passed at ``head``, its failures all
    ones ``covers`` allows (`allowed`); else what failed.  The run judged itself with no policy: `done`, exit 0 and validation
    passed when it found no failure, and `done`, exit 1 and validation failed when every one it found is allowed."""
    tier, output = res.doc.get("tier"), f"its output is in run {rid}'s log on {on}"
    if (why := quick_tier.judge(tier, head, output, covers)) is not None:
        return why
    want = ("done", 0, "passed") if quick_tier.judge(tier, head, output) is None else ("done", 1, "failed")
    if (status.get("state"), status.get("rc"), res.validation_status) != want:
        return (f"run {rid}'s facts show a pass, but it ended {status.get('state')} (exit {status.get('rc')}) with validation "
                f"{res.validation_status}")
    return None


def judge_run(d: Path, rid: str, on: str, head: str, status: dict, covers: Callable[[dict], bool] | None = None) -> str | None:
    """None when ended run ``rid`` (records in ``d``) reports a pass of the quick tier at ``head`` (``covers``: `judge_report`);
    else what failed: a run that ends without the facts, with facts from another run or head, or other than as it judged
    itself, is a failure."""
    res, why = tier_report(d, rid, on, status)
    return why or judge_report(res, rid, on, head, status, covers)


def _json_file(p: Path) -> dict:
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def same_tree_mark(store, repo: Path, remote: str, pr: PR) -> str | None:
    """Another head of ``pr`` marked ready whose tree is ``pr.head``'s (a merge of `main` that changes no file), or None.  The tier
    at that head ran these exact files, and when a clean merge of `main` leaves the tree as it was, the files the new head changes
    against `main` are among those the marked head changed against its merge base, so the tier would run no suite it didn't.  None
    when either commit isn't here after fetching the PR's head (a rebase drops the old one), and QueueError when the PR's labels
    can't be pulled from the store's remote."""
    _fetch_heads(repo, remote, [pr])
    if not (want := _tree(repo, pr.head)):
        return None
    marked = StoreLabels(store)._other_heads(pr.number, pr.head, vocab.READY)
    return next((old for old, labs in sorted(marked.items()) if any(str(lab.value).lower() == "true" for lab in labs)
                 and _tree(repo, old) == want), None)


def same_tree_prs(repo: Path, remote: str, prs: list[PR], head: str) -> list[PR]:
    """The PRs of ``prs`` whose head has ``head``'s tree, after fetching their heads (none when ``head`` isn't here)."""
    _fetch_heads(repo, remote, prs)
    want = _tree(repo, head)
    return [p for p in prs if want and _tree(repo, p.head) == want] if prs else []


def _fetch_heads(repo: Path, remote: str, prs: list[PR]) -> None:
    """Each PR's head into a ref of its own, so a tier's fetch under the worktree's lock never meets this one."""
    if prs:
        _git(repo, "fetch", "-q", remote, *(f"+refs/pull/{p.number}/head:refs/queue/same-tree/{p.number}" for p in prs), check=False)


def _tree(repo: Path, commit: str) -> str:
    return _git(repo, "rev-parse", "-q", "--verify", f"{commit}^{{tree}}", check=False).stdout.strip()


#: the own changes `Queue.own_change` keeps: the heads of open PRs whose mark or verdict another head may take number some hundreds
OWN_KEEP = 4096


def own_change(repo: Path, base: str, head: str) -> str | None:
    """The whitespace-kept patch-id (`git patch-id --verbatim`: the stable algorithm without `--stable`'s stripping of every
    whitespace) of ``head``'s diff from ``base`` (its merge base with the landing branch: `Queue.own_change`), with full blob
    ids, so a binary file's new content counts; None for an empty diff, a commit that isn't here, or a git before 2.39, which
    has no `--verbatim`.  A merge of `main` that resolves no conflict leaves it as it was, as does a merge of `next` once the
    members it merged have landed; an edit of the change, if only of its whitespace (a line's indentation, the spaces in a
    string), a conflict resolved into it, or a change of a line its diff shows as context changes it."""
    diff = subprocess.run(["git", "diff-tree", "-r", "-p", "--full-index", base, head], cwd=repo, capture_output=True,
                          env=git_env(GIT_TERMINAL_PROMPT="0"))
    if diff.returncode or not diff.stdout:
        return None
    got = subprocess.run(["git", "patch-id", "--verbatim"], cwd=repo, input=diff.stdout, capture_output=True).stdout.split()
    return got[0].decode() if got else None


#: the invariant tests the preflight runs on the merged tree, from the root, each a common way a train breaks: a module string
#: naming a module only an unlanded PR adds, stale Lake pins, a doc path, a workspace member without its REAL_ROOTS entry, and
#: `check`'s preflight-lints on test_repository.py (a report-genre name)
PREFLIGHT_TEST_PATHS = ("tests/test_module_strings.py", "tests/test_pins.py", "tests/test_doc_paths.py",
                        "tests/test_repository.py", "tools/research/tests/test_pythonpath.py")


def _by_suite(paths: tuple[str, ...]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """`paths` grouped by the directory pytest runs each in (the part before `tests/`), each relative to it."""
    groups: dict[str, list[str]] = {}
    for path in paths:
        d, sep, rest = path.rpartition("tests/")
        groups.setdefault(d.rstrip("/") or ".", []).append(sep + rest)
    return tuple((d, tuple(fs)) for d, fs in groups.items())


#: `PREFLIGHT_TEST_PATHS` by the directory pytest runs in
PREFLIGHT_TESTS = _by_suite(PREFLIGHT_TEST_PATHS)
#: lints the preflight runs on the merged tree only when the change reaches what they judge (a path starting with, or holding
#: a directory, of the prefix): they take about 20 s and 40 s, and a train that fails one of them loses its hour-long check
PREFLIGHT_LINTS = (("tests/test_no_wall_clock.py", ("tests/",)), ("adapters/vllm/tests/lint", ("adapters/vllm/",)))
#: suites whose collection is no cheap check, each with why; the preflight names them and leaves them to the train's check
PREFLIGHT_UNCOLLECTED: dict[str, str] = {}
#: the merged tree's environment, as the quick tier's `uv run --locked` has it (a fresh tree syncs from uv's cache in seconds)
PREFLIGHT_SYNC = ("uv", "sync", "--locked", "--all-packages", "--extra", "torch-cpu", "-q")
#: what the merged tree's checks may take together; what hasn't finished by then is named and refuses nothing.  Every suite's
#: collection but circuit-check's takes about 18 s on 4 workers of a 4-core VM; `PREFLIGHT_LINTS` add up to a minute
PREFLIGHT_CHECKS_S = 120.0
#: collections at once: four torch-importing `--collect-only` runs reach about 8 GB, half of a marker's 16 GB VM
PREFLIGHT_JOBS = 2
#: below this share of memory available, no suite is collected, so a mark never OOMs the VM's other agents (the shell hook's `AGENT_GUARD_MIN_AVAIL`)
PREFLIGHT_MIN_FREE = 0.25
#: the Lean step's bounds (`Preflight._lean`): the most modules it builds (those the change touches and every module importing one,
#: transitively, each rebuilt since an import changed), and the seconds its builds may take, on the merged tree and, for what fails
#: there, on `next`; past either it notes why and refuses nothing (`LeanTimedOut`).  With `PREFLIGHT_CHECKS_S` the whole
#: preflight stays within 5 minutes, and a warm tree rebuilds a few changed modules and their importers in a minute or two
PREFLIGHT_LEAN_MODULES = 40
PREFLIGHT_LEAN_S = 180.0
#: where the Lean step builds, whatever the mark's tier, when this machine keeps no warm Lean tree: the first of these nodes that
#: starts the build, each keeping `build` Lean slots, which `research run --on auto` doesn't place by (it takes a pool, whose
#: cold tree builds nothing in the step's budget).  The one place to move it
PREFLIGHT_LEAN_ON = ("vy-nebius-cpu-9", "vy-nebius-cpu-5")
#: the `research run` pool it takes there: none, since the node's `build` Lean slots are the gate; a node without them would take
#: its `research` pool
PREFLIGHT_LEAN_POOL: str | None = None
#: the merged tree's builder (`lean_changed.py --build FILE...`, on a `build` Lean slot's warm tree), and its last line's prefix
LEAN_BUILDER, LEAN_BUILD_LINE = "tools/verity/lean/lean_changed.py", "lean-build: "
#: `lean_slot.py`'s SLOTS: a machine whose file lists a `build` pool keeps warm Lean trees; an agent VM has none
LEAN_SLOTS = Path(os.environ.get("LEAN_SLOTS", "/workspace/research/locks/lean-slots"))
#: the Lake packages the Lean step reads: a directory with a lakefile and a `lean-audit.json`, the auditor's fixtures aside
LEAN_FIXTURES = "tools/verity/lean/"
#: the temporary worktrees' prefix (`merged_worktree`), `<tmp>/research-merged-<pid>-*`: one whose process is gone is swept (a kill
#: skips `finally`)
MERGED_PREFIX = "research-merged-"
#: where a merged worktree goes when its caller names no root (`merged_worktree`): `main` sets it from --preflight-root, so
#: drive's other merged trees (`ported`'s codemod) leave its network-volume TMPDIR too, where each add and remove takes minutes
MERGED_ROOT: Path | None = None
#: prints, as JSON, the paths of the suites `suites.py`'s `touched` selects for the change from commit argv[1] to tree argv[2]
#: (`suites`) and, for each, the fetched registry entries under its inputs (`fixtures`), as `suites.py` picks what to fetch before
#: it runs them: the repository's own suite reads tracked files only, so it needs none
_TOUCHED = ("import json, sys; sys.path.insert(0, 'tools/verity/check'); import suites as S; pkgs = S.members(); "
            "chosen = S.touched(S.discover(), pkgs, sys.argv[1], sys.argv[2], S.locked()); "
            "ins = {s: S.inputs_of(s, pkgs) for s in chosen}; own = {s: [p for p in ps if p != '.'] for s, ps in ins.items()}; "
            "fx = S.fetched_fixtures(sys.argv[2], {p for ps in own.values() for p in ps}); "
            "print(json.dumps({'suites': [s.path for s in chosen], 'fixtures': {s.path: sorted(p for p in fx "
            "if S.covered(p, own[s], S.dropped(s, ins[s]))) for s in chosen}}))")
#: what the preflight's fetch of its suites' fixtures (`research data fetch-fixtures`) may take of `PREFLIGHT_CHECKS_S`: the
#: machine's store keeps every blob it fetches, so a later preflight here copies them from it in seconds
PREFLIGHT_FETCH_S = 45.0
#: a line of pytest's short summary: `FAILED tests/x.py::test_y - why`, `ERROR tests/x.py - why` (a collection error)
def free_memory() -> float | None:
    """The share of this machine's memory available (`/proc/meminfo`'s MemAvailable over MemTotal), or None where unreadable."""
    try:
        info = dict(ln.split(":", 1) for ln in Path("/proc/meminfo").read_text().splitlines() if ":" in ln)
        return int(info["MemAvailable"].split()[0]) / int(info["MemTotal"].split()[0])
    except (OSError, KeyError, ValueError, ZeroDivisionError):
        return None


_SUMMARY_RE = re.compile(r"^(?:FAILED|ERROR) (.+?)(?: - (.*))?$", re.M)
_LOCK_STALE = "needs to be updated"


class Ran(str):
    """A preflight note saying what ran.  Every other note names a check that didn't run or couldn't judge, and the preflight's
    result lists it under "checks skipped", never as nothing refused."""


@dataclass
class Touched:
    """What `touched_suites` gave.  ``failed``: each failed suite in `suites.py --json`'s shape (`quick_tier.ROW_KEYS`, status
    failed, with each failing test's summary line under `summary`); ``suites``: those that ran here; ``ids``: the runs launched;
    ``notes``: what kept a suite or a launch from running; ``tail``: the launch's output; ``fixtures``: each selected suite's
    fetched fixtures (`LocalSuites`)."""
    suites: list[str] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    tail: str = ""
    fixtures: dict[str, list[str]] = field(default_factory=dict)


def failed_row(suite: str, failures: dict[str, str]) -> dict:
    """Suite ``suite``'s failures, {node id: its summary line}, as `suites.py --json` reports a failed suite."""
    return {"name": suite, "suite": suite, "status": "failed", "failed_tests": list(failures), "log": None,
            "summary": list(failures.values())}


def pytest_failures(run: Callable[[list[str], Path], tuple[subprocess.CompletedProcess | None, str]],
                    py: str | tuple[str, ...], cwd: Path, args: list[str]) -> dict[str, str] | str:
    """{node id: its summary line} of what failed in `pytest ARGS` in ``cwd`` (empty: passed), or why it didn't run.  ``py`` is
    an interpreter's path, or the argv that starts one."""
    r, why = run([*((py,) if isinstance(py, str) else py), "-P", "-m", "pytest", "-q", "-p", "no:cacheprovider", *args], cwd)
    if r is None:
        return why
    if r.returncode in (0, 5):                                   # 5: nothing collected
        return {}
    found = {m.group(1): m.group(0) for m in _SUMMARY_RE.finditer(r.stdout)}
    return found or {f"exit {r.returncode}": f"pytest exited {r.returncode}: {(r.stdout + r.stderr).strip()[-300:]}"}


@contextlib.contextmanager
def worktrees_locked(repo: Path):
    """``repo``'s worktrees to this caller alone: train.sh's lock (`<git common dir>/train.lock`), so drive's launches at once
    and train.sh never add, remove or prune a worktree beside another (two at once fail with "Could not write new index
    file")."""
    common = Path(_out(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    with open(common / "train.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


@contextlib.contextmanager
def merged_worktree(repo: Path, commit: str, root: Path | None = None):
    """A detached worktree of ``repo`` at ``commit`` in a temporary directory under ``root`` (else `MERGED_ROOT`, else the
    temporary directory; `MERGED_PREFIX`), removed with it whatever happens; one a dead process left is swept first."""
    root = Path(root or MERGED_ROOT or tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    for d in root.glob(f"{MERGED_PREFIX}*"):
        pid = d.name.removeprefix(MERGED_PREFIX).split("-", 1)[0]
        if pid.isdigit() and int(pid) != os.getpid() and not Path(f"/proc/{pid}").exists():
            shutil.rmtree(d, ignore_errors=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"{MERGED_PREFIX}{os.getpid()}-", dir=root))
    wt = tmp / "tree"
    try:
        with worktrees_locked(repo):
            _git(repo, "worktree", "prune", check=False)
            _git(repo, "worktree", "add", "-q", "--detach", str(wt), commit)
        yield wt
    finally:
        with worktrees_locked(repo):
            _git(repo, "worktree", "remove", "--force", str(wt), check=False)
            shutil.rmtree(tmp, ignore_errors=True)
            _git(repo, "worktree", "prune", check=False)


@dataclass
class LocalSuites:
    """`touched_suites` here, as the preflight runs them: `pytest ARGS` (`--collect-only`) in each suite `suites.py`'s own `touched`
    selects (run with the tree's ``python``; `PREFLIGHT_UNCOLLECTED` left out), or in ``suites`` alone when given, ``jobs`` at once,
    each through ``run`` (the caller's budget); none while under `PREFLIGHT_MIN_FREE` of memory is available.  First ``fetch``
    (the tree, the paths; gives the paths still missing and why) brings the fetched fixtures under those suites' inputs into the
    tree (`_TOUCHED`'s, or ``fixtures`` with ``suites``), since a test may read one at import; a suite whose fixtures it couldn't
    fetch isn't collected, and one note says so."""
    python: str
    run: Callable[[list[str], Path], tuple[subprocess.CompletedProcess | None, str]]
    jobs: int = PREFLIGHT_JOBS
    free: Callable[[], float | None] = free_memory
    suites: list[str] | None = None
    args: tuple[str, ...] = ("--collect-only",)
    fetch: Callable[[Path, list[str]], tuple[set[str], str]] | None = None
    fixtures: dict[str, list[str]] = field(default_factory=dict)

    def __call__(self, wt: Path, commit: str, base: str) -> Touched:
        out = Touched(fixtures=dict(self.fixtures))
        suites = self.suites
        if suites is None:
            r, why = self.run([self.python, "-P", "-c", _TOUCHED, base, _out(wt, "rev-parse", f"{commit}^{{tree}}")], wt)
            try:
                said = json.loads(r.stdout.strip().splitlines()[-1]) if r is not None and r.returncode == 0 else None
                suites, out.fixtures = (said["suites"], said["fixtures"]) if said is not None else (None, {})
            except (ValueError, IndexError, KeyError, TypeError):
                suites = None
            if suites is None:
                out.notes.append(f"no suite collected: `suites.py` couldn't say which the change touches "
                                 f"({why or (r.stderr or r.stdout).strip()[-200:]})")
                suites = []
            out.notes += [f"{s} not collected: {PREFLIGHT_UNCOLLECTED[s]}" for s in suites if s in PREFLIGHT_UNCOLLECTED]
            suites = [s for s in suites if s not in PREFLIGHT_UNCOLLECTED]
        todo = [s for s in suites if (wt / s).is_dir()]
        if todo and (f := self.free()) is not None and f < PREFLIGHT_MIN_FREE:
            out.notes.append(f"no suite collected: {f:.0%} of memory available, under {PREFLIGHT_MIN_FREE:.0%}; the after-landing "
                             "run still collects them")
            todo = []
        need = sorted({p for s in todo for p in out.fixtures.get(s, ())})
        if need and self.fetch is not None:
            missing, why = self.fetch(wt, need)
            if blocked := [s for s in todo if missing & set(out.fixtures.get(s, ()))]:
                shown = ", ".join(sorted(missing)[:3]) + (f" and {len(missing) - 3} more" if len(missing) > 3 else "")
                out.notes.append(f"{', '.join(blocked)} not judged: {len(missing)} of the {len(need)} fixture(s) the suites read "
                                 f"weren't fetched ({why or 'no source held them'}: {shown}); the after-landing run fetches them")
                todo = [s for s in todo if s not in blocked]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(self.jobs, len(todo) or 1))) as pool:
            for s, res in zip(todo, pool.map(lambda s: pytest_failures(self.run, self.python, wt / s, list(self.args)), todo)):
                if isinstance(res, str):
                    out.notes.append(f"collect:{s} not checked: {res}")
                    continue
                out.suites.append(s)
                if res:
                    out.failed.append(failed_row(s, res))
        if self.suites is None:
            out.notes.append(Ran(f"collected {len(todo)} suite(s)" + (f" ({', '.join(todo)})" if len(todo) <= 6 else "")))
        return out


def touched_suites(repo: Path, commit: str, base: str, how: LocalSuites, *, wt: Path | None = None,
                   root: Path | None = None) -> Touched:
    """The suites the change from ``base`` to ``commit`` (a merged tree) touches, run at ``commit`` by ``how`` within the
    caller's budget (`LocalSuites`: the preflight, a PR's head merged onto `next`).  In ``wt`` when given (checked out at
    ``commit``), else in a temporary worktree of ``repo`` at ``commit`` (`merged_worktree`), removed once ``how`` returns.  The
    failures come back in `suites.py --json`'s shape, in `Touched.failed`.  A worktree that can't be made is a note, and
    nothing runs."""
    if wt is not None:
        return how(wt, commit, base)
    try:
        with merged_worktree(repo, commit, root) as tree:
            return how(tree, commit, base)
    except QueueError as e:
        why = f"no worktree of {commit[:12]}: {e}"
        return Touched(notes=[why], tail=why)


_IMPORT = re.compile(r"(?:public\s+)?(?:meta\s+)?import\s+(?:all\s+)?(.*)")


def lean_imports(text: str) -> list[str]:
    """The modules a Lean file's header imports (`import M`, `public import M`, `meta import M`, `import all M`), read up to its
    first command, past comments (nested `/- -/` too), `module` and `prelude`."""
    out, depth = [], 0
    for line in text.splitlines():
        s = line.strip()
        while s:
            if depth:
                i, j = s.find("/-"), s.find("-/")
                if j < 0 and i < 0:
                    s = ""
                elif i >= 0 and (j < 0 or i < j):
                    depth, s = depth + 1, s[i + 2:]
                else:
                    depth, s = depth - 1, s[j + 2:].strip()
            elif s.startswith("/-"):
                depth, s = 1, s[2:]
            else:
                break
        if not s or s.startswith("--") or s in ("module", "prelude"):
            continue
        if not (m := _IMPORT.fullmatch(s)):
            break
        out += m.group(1).split("--", 1)[0].split()
    return out


@dataclass
class LakePackage:
    """A Lake package of a tree: its directory (``path``, from the root), where its modules' names start (``src``, its `srcDir`),
    its libraries' roots (a module under one is what `lake build +M` builds) and the packages it requires by path (its manifest's
    `path` entries, the inherited ones too)."""
    path: str
    src: str
    roots: tuple[str, ...]
    requires: tuple[str, ...]

    def module(self, f: str) -> str | None:
        rel = posixpath.relpath(f, self.src)
        return None if rel.startswith("../") or not rel.endswith(".lean") else rel.removesuffix(".lean").replace("/", ".")

    def builds(self, module: str) -> bool:
        return any(module == r or module.startswith(r + ".") for r in self.roots)


def lake_packages(wt: Path, files: list[str]) -> dict[str, LakePackage]:
    """The Lake packages of the tree ``wt`` holding ``files``: each directory with a `lean-audit.json` and a lakefile, outside
    `LEAN_FIXTURES`."""
    have = set(files)
    out = {}
    for f in files:
        if not f.endswith("/lean-audit.json") or f.startswith(LEAN_FIXTURES):
            continue
        d = f.rsplit("/", 1)[0]
        src, roots, requires = ".", [], []
        try:
            if f"{d}/lakefile.toml" in have:
                doc = tomllib.loads((wt / d / "lakefile.toml").read_text())
                src = doc.get("srcDir", ".")
                roots = [r for lib in doc.get("lean_lib", []) for r in (lib.get("roots") or [lib.get("name")]) if isinstance(r, str)]
                requires = [r["path"] for r in doc.get("require", []) if isinstance(r.get("path"), str)]
            elif f"{d}/lakefile.lean" in have:
                text = (wt / d / "lakefile.lean").read_text()
                src = (m.group(1) if (m := re.search(r'srcDir\s*:=\s*"([^"]*)"', text)) else ".")
                roots = re.findall(r"lean_lib\s+«?([\w.]+)»?", text)
            else:
                continue
            if f"{d}/lake-manifest.json" in have:
                manifest = json.loads((wt / d / "lake-manifest.json").read_text())
                requires += [p["dir"] for p in manifest.get("packages", []) if p.get("type") == "path" and isinstance(p.get("dir"), str)]
        except (OSError, ValueError, AttributeError, TypeError):
            continue
        out[d] = LakePackage(d, posixpath.normpath(posixpath.join(d, src)), tuple(roots),
                             tuple(sorted({posixpath.normpath(posixpath.join(d, p)) for p in requires} - {d})))
    return out


@dataclass
class LeanGraph:
    """The Lean step's view of a merged tree: {file: module} of each `.lean` file of the packages a change reaches, its imports
    among them, and ``targets``, {file: module} of what the step builds."""
    module: dict[str, str] = field(default_factory=dict)
    imports: dict[str, list[str]] = field(default_factory=dict)
    targets: dict[str, str] = field(default_factory=dict)

    def upstream(self, f: str) -> list[str]:
        """The files ``f``'s module imports, transitively, among those the graph holds."""
        files = {m: g for g, m in self.module.items()}
        seen, todo = set(), list(self.imports.get(self.module.get(f, ""), []))
        while todo:
            if (m := todo.pop()) in seen or m not in files:
                continue
            seen.add(m)
            todo += self.imports.get(m, [])
        return sorted(files[m] for m in seen)


def lean_graph(wt: Path, files: list[str], changed: list[str]) -> LeanGraph:
    """What the Lean step builds for a change to ``changed`` (paths from the root, deleted ones too) on the tree ``wt`` holding
    ``files``: for each Lake package a changed `.lean` file belongs to (`lake_packages`, the deepest holding it), the changed
    modules and every module importing one, transitively (`lean_imports`), among that package's own files and those of every
    package requiring it by path; only modules under a library's root, which `lake build +M` names.  Empty without a Lean change."""
    pkgs = lake_packages(wt, files)
    owner = lambda f: max((p for p in pkgs if f.startswith(p + "/")), key=len, default=None)    # noqa: E731
    touched = {p for f in changed if f.endswith(".lean") and (p := owner(f)) and not f.endswith("/lakefile.lean")}
    graph = LeanGraph()
    if not touched:
        return graph
    scope = {p for p, pkg in pkgs.items() if p in touched or touched & set(pkg.requires)}
    for f in files:
        if f.endswith(".lean") and not f.endswith("lakefile.lean") and (p := owner(f)) in scope and (m := pkgs[p].module(f)):
            graph.module[f] = m
            try:
                graph.imports[m] = lean_imports((wt / f).read_text(errors="replace"))
            except OSError:
                graph.imports[m] = []
    start = {m for f in changed if (p := owner(f)) in touched and f.endswith(".lean") and (m := pkgs[p].module(f))}
    importers: dict[str, set[str]] = {}
    for m, imps in graph.imports.items():
        for i in imps:
            importers.setdefault(i, set()).add(m)
    reached, todo = set(start), list(start)
    while todo:
        for m in importers.get(todo.pop(), ()):
            if m not in reached:
                reached.add(m)
                todo.append(m)
    graph.targets = {f: m for f, m in graph.module.items() if m in reached and pkgs[owner(f)].builds(m)}
    return graph


@dataclass
class LeanBuilt:
    """What the Lean step's build gave: {file: its first Lean error} for each file that failed, and what it said otherwise; where
    it built, and why each node before that one didn't."""
    failed: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    where: str = ""
    passed: list[str] = field(default_factory=list)


class LeanTimedOut(str):
    """Why the Lean step built nothing within its time: a note under "checks skipped", refusing nothing.  Any other reason it
    built nothing (a plain string) refuses the mark, naming the fix."""


#: the fix a Lean step that couldn't build names, with where it builds
LEAN_UNBUILT_FIX = "mark again once {on} builds them in its `build` Lean slots (`research status` there says why not)"


def run_group(argv: list[str], cwd: Path, left: float) -> tuple[int | None, str]:
    """``argv``'s exit status and output (stderr after stdout), in a process group of its own that is killed whole once ``left``
    seconds pass (a `lake build` it started included, which would otherwise go on in a slot's tree), and then None."""
    p = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace",
                         stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        out, err = p.communicate(timeout=max(left, 0))
        return p.returncode, out + err
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
        return None, out + err


#: `research run`'s line naming the run it started (or refused), as `verity lean try` reads it
RUN_STARTED_RE = re.compile(r"^research: run (\S+) (?:launched|refused) on ", re.M)


@dataclass
class LakeBuild:
    """The Lean step's builds (`Preflight._lean`): the tree's `LEAN_BUILDER` (`lean_changed.py --build FILE...`), which builds the
    files' modules on the warm tree of a node's `build` Lean slot as `verity lean try` builds one, and ends with one
    `LEAN_BUILD_LINE`.  It runs here when this machine's Lean slot file (``slots``) lists a `build` pool, in a process group
    killed at the step's deadline; else, given ``on`` (`ready`, whatever its tier: `PREFLIGHT_LEAN_ON`), as a `research run` of the
    tree on the first of those nodes that builds them, on its `build` Lean slot, or in ``pool`` (a slot named for ``by``) when one
    is given, all within the same deadline (an agent VM builds nothing itself: the team skill, "Machines").  A node that can't
    start the build (no free `build` slot: each node but the last takes one with `--no-wait`, rather than wait in line; a run
    refused, drained or unreachable; no result line; every package skipped) passes it to the next.  A build that didn't end by
    the deadline is `LeanTimedOut`; when no node built them (or there is no warm tree and no ``on``), a plain string naming each
    node's reason and the fix, which refuses the mark."""
    on: tuple[str, ...] = ()
    pool: str | None = PREFLIGHT_LEAN_POOL
    by: str | None = None
    slots: Path = LEAN_SLOTS
    run: Callable[[list[str], Path, float], tuple[int | None, str]] = run_group
    runs: Callable[[], Path] | None = None
    clock: Callable[[], float] = time.monotonic

    def warm(self) -> bool:
        try:
            return any(ln.split("#")[0].split()[:1] == ["build"] for ln in self.slots.read_text().splitlines())
        except OSError:
            return False

    def __call__(self, wt: Path, files: list[str], left: float) -> LeanBuilt | str:
        if left <= 0:
            return LeanTimedOut("the Lean step's budget was spent before it started")
        if not (wt / LEAN_BUILDER).is_file():
            return f"the merged tree has no {LEAN_BUILDER}, which builds them"
        if self.warm():
            rc, text = self.run([sys.executable, LEAN_BUILDER, "--build", *files], wt, left)
            res = self._read(rc, text, "here", left)
            return res if isinstance(res, (LeanBuilt, LeanTimedOut)) else f"{res}: {LEAN_UNBUILT_FIX.format(on='this node')}"
        if not self.on:
            return (f"this machine keeps no warm Lean tree (no `build` pool in {self.slots}; an agent VM builds nothing) and was "
                    f"given no node to build on: `ready --on MACHINE` builds them on {' or '.join(PREFLIGHT_LEAN_ON)}")
        end, tried = self.clock() + left, []
        for i, node in enumerate(self.on):
            if (rest := end - self.clock()) <= 0:
                return LeanTimedOut("; ".join([*tried, f"no time was left to build on {node}"]))
            rc, text, where = self._remote(node, wt, files, rest, wait=i == len(self.on) - 1)
            res = self._read(rc, text, where, rest)
            if isinstance(res, LeanTimedOut):
                return LeanTimedOut("; ".join([*tried, res]))
            if isinstance(res, LeanBuilt):
                res.passed = tried
                return res
            tried.append(res)
        return f"{'; '.join(tried)}: {LEAN_UNBUILT_FIX.format(on=' or '.join(self.on))}"

    @staticmethod
    def _read(rc: int | None, text: str, where: str, left: float) -> LeanBuilt | str:
        """What one build gave: `LeanBuilt`, `LeanTimedOut` when it didn't end (``rc`` None), else why it built nothing."""
        if rc is None:
            return LeanTimedOut(f"its build {where} hadn't ended after {left:.0f} s (a wait in its pool's line counts)"
                                f"{': ' + text if text else ''}" if where != "here" else
                                f"its build here was still going after {left:.0f} s{': ' + text if text else ''}")
        doc = next((ln[len(LEAN_BUILD_LINE):] for ln in reversed(text.splitlines()) if ln.startswith(LEAN_BUILD_LINE)), None)
        try:
            doc = json.loads(doc) if doc is not None else None
        except ValueError:
            doc = None
        if not isinstance(doc, dict):
            return f"its build {where} exited {rc} without a result line: {text.strip()[-300:]}"
        built, skipped = doc.get("built") or {}, doc.get("skipped") or {}
        notes = [f"{p} not built {where}: {why}" if where != "here" else f"{p} not built: {why}" for p, why in skipped.items()]
        failed = {f: str(e) for f, e in (doc.get("failed") or {}).items()}
        notes += [f"`lake build` in {p} exited {c} with no Lean error in a file" for p, c in built.items()
                  if c and not any(f.startswith(p + "/") for f in failed)]
        if not built and not failed:
            return "; ".join(notes) or f"its build {where} built nothing (exit {rc})"
        return LeanBuilt(failed, notes, where)

    def _remote(self, node: str, wt: Path, files: list[str], left: float, wait: bool = True) -> tuple[int | None, str, str]:
        """`LEAN_BUILDER --build` as a `research run` of ``wt``'s HEAD on ``node``, waited for within ``left`` (without ``wait``,
        `--no-wait`: no free `build` slot there builds nothing); its exit status (None: it didn't end), its stdout, and where it
        ran."""
        from . import runs as R

        secs = f"{max(left - 10, 1):.0f}"
        argv = [sys.executable, "-m", "research", "run", "--on", node, *(["--pool", self.pool, "--pool-wait", secs] if self.pool else []),
                *(["--as", self.by] if self.by else []), "--project", "verity", "--source", str(wt), "--cwd", "clone",
                "--timeout", secs, "--wait", "--wait-every", "5", "--wait-max", secs, "--",
                "python3", LEAN_BUILDER, *([] if wait else ["--no-wait"]), "--build", *files]
        rc, said = self.run(argv, wt, left)
        where = f"on {node}"
        if not (m := RUN_STARTED_RE.search(said)):
            return rc, said.strip()[-300:] if rc is not None else "", where
        rid, where = m.group(1), f"on {node} (run {m.group(1)})"
        self.run([sys.executable, "-m", "research", "fetch", rid, "--on", node, "--all"], wt, 60)
        log = (self.runs or R.runs_root)() / rid / "stdout.log"
        text = log.read_text(errors="replace") if log.is_file() else ""
        rc = None if rc == waitcap.STILL_GOING else rc
        return rc, text or (said.strip()[-300:] if rc is not None else ""), where


#: the fix `lock_drop_refusal` names for each entry a lock merge drops
LOCK_DROP_FIX = "restore it verbatim from `{onto}` in a lock-only commit (`git show origin/{onto}:PATH`) and mark that head"
#: how much of a dropped premise's text `lock_drops` shows
LOCK_DROP_SHOWN = 60


def _unowned(rec):
    return {k: v for k, v in rec.items() if k != "owner"} if isinstance(rec, dict) else rec


def _texts(section: str, entries: dict, g: str) -> list[str]:
    """The texts of ``g``'s entry in a lock's `shape` ``section``: its premises (`inline`), or its reason (`frozen`)."""
    if g not in entries:
        return []
    v = entries[g]
    if section == "inline":
        return list(v) if isinstance(v, dict) else []
    return [v if isinstance(v, str) else json.dumps(v, sort_keys=True)]


#: a name as a lock's recorded signature prints it (`check.py`'s `NAMES`)
LOCK_NAME = re.compile(r"(?<![\w.'!?])([^\W\d][\w'!?]*(?:\.[^\W\d][\w'!?]*)*)")


def _held_to_form(lock: dict) -> set[str]:
    """The theorems a lock's `claims.witnesses` (`fails`, `meets`), `witnesses` (`nontrivial`, `honest`) and `claims.generic`
    name, which `check.py` holds to their form instead of `shape` (its `claim_witnesses` and `generic_claims`): it reads no
    `shape.frozen` entry of theirs."""
    claims = lock.get("claims") if isinstance(lock.get("claims"), dict) else {}
    ws, gs, generic = (d if isinstance(d, dict) else {}
                       for d in (claims.get("witnesses"), lock.get("witnesses"), claims.get("generic")))
    names = [n for w in ws.values() if isinstance(w, dict)
             for n in [w.get("fails"), *(w.get("meets") if isinstance(w.get("meets"), list) else [])]]
    for w in gs.values():
        if isinstance(w, dict):
            n = w.get("nontrivial")
            names += [n] if isinstance(n, str) else n if isinstance(n, list) else []
            names += w.get("honest") if isinstance(w.get("honest"), list) else []
    return {n for n in [*names, *generic.values()] if isinstance(n, str)}


def _read_hashes(lock: dict) -> dict[str, str]:
    """{definition: its hash} over the lock's `reads`."""
    out: dict[str, str] = {}
    for group in (lock.get("reads") or {}).values():
        if isinstance(group, dict) and isinstance(defs := group.get("definitions"), dict):
            out.update(defs)
    return out


def dropped_entries(base: dict, merged: dict) -> list[str]:
    """Each entry of the lock ``base`` (`next`'s) that ``merged`` (the head's lock merged onto it) drops though `next`'s
    statements still need it: a premise of `shape.inline` ({guarantee: {premise as printed: why}}), as `shape.inline[G]:
    PREMISE`, and a key of `shape.frozen`, as `shape.frozen[G]`.  ``merged`` may drop a guarantee's entry when it no longer
    lists the guarantee under `guarantees` (removed), or restated it: it records the guarantee otherwise than ``base`` does,
    its owner aside, or a definition its signature names has another hash in both (a guarantee lifted into a `Guarantees`
    definition is restated there; `check.py --update` keeps a premise only while the statement prints it, and the head's
    records run, or the lock run after its landing, checks its lock).  It may drop an entry whose text (a premise, a frozen
    entry's reason) it holds under a key of the section ``base`` lacks (renamed), a guarantee's premises while it adds as
    many others under that guarantee (re-keyed at a restated statement, as `check.py`'s `paired` keys them), a frozen
    entry's reason while it keeps the key (reworded: `check.py` reads the key and the reason's category, not its wording),
    and the frozen entry of a theorem ``merged`` holds to its form (`_held_to_form`), which `check.py` never reads."""
    listed, was = merged.get("guarantees") or {}, base.get("guarantees") or {}
    old, new = _read_hashes(base), _read_hashes(merged)
    inert = _held_to_form(merged)

    def restated(g: str) -> bool:
        names = LOCK_NAME.findall(" ".join((listed[g] if isinstance(listed[g], dict) else {}).get("signature") or []))
        return _unowned(listed[g]) != _unowned(was.get(g)) or any(
            old[n] != new[n] for n in names if n in old and n in new)

    out = []
    for section in ("inline", "frozen"):
        b, m = ((d.get("shape") or {}).get(section) or {} for d in (base, merged))
        if not isinstance(b, dict) or not isinstance(m, dict):
            continue
        moved = {t for g in m if g not in b for t in _texts(section, m, g)}
        for g in b:
            if g not in listed or restated(g):
                continue
            had, kept = _texts(section, b, g), _texts(section, m, g)
            gone = [t for t in had if t not in kept and t not in moved]
            if section == "frozen":
                out += [f"shape.frozen[{g}]" for _ in gone if g not in m and g not in inert]
            elif len(gone) > len([t for t in kept if t not in had]):
                out += [f"shape.inline[{g}]: " + (t if len(t) <= LOCK_DROP_SHOWN else t[:LOCK_DROP_SHOWN] + "…") for t in gone]
    return out


def lock_drops(repo: Path, base: str, head: str, merged: str | None = None) -> list[str]:
    """`dropped_entries` of each lock (`lean-audit.json`, `LEAN_FIXTURES` aside) that ``head``'s change from its merge base with
    ``base`` touches, from ``base`` to ``merged`` (a commit or tree; by default ``head`` merged onto ``base`` as the sync merges
    it, with ``base``'s merge drivers, so `tools/verity/lean/merge.py` merges the locks), each prefixed by the lock's path.  A
    drop ``head``'s own lock already makes against that merge base is no refusal: its records run, or the lock run after its
    landing, regenerates the lock, and `check.py --update` keeps a premise only while a statement prints it, so a premise it
    dropped is printed nowhere at the head (a stale entry); only what the merge alone drops is named.  A lock either side lacks
    is the head's to add or remove, one that doesn't parse is named, and a diff or merge git can't make finds nothing: the lock
    run after its landing still fails on a dropped premise."""
    r = _git(repo, "diff", "--name-only", "--no-renames", f"{base}...{head}", check=False)
    locks = [f for f in r.stdout.splitlines() if posixpath.basename(f) == LEAN_LOCK and not f.startswith(LEAN_FIXTURES)]
    if r.returncode or not locks:
        return []
    if merged is None:
        argv = ("merge-tree", "--write-tree", "--name-only", "--no-messages", base, head)
        m = _git(repo, f"--attr-source={base}", *argv, check=False)
        if m.returncode == 129 and "attr-source" in m.stderr:
            m = _git(repo, *argv, check=False)
        if m.returncode not in (0, 1) or not m.stdout.split():
            return []
        merged = m.stdout.split()[0]
    fork = _git(repo, "merge-base", base, head, check=False).stdout.strip()
    blobs = _blobs(repo, [f"{c}:{f}" for f in locks for c in (base, merged, fork, head) if c])
    out = []
    for f in locks:
        if f"{base}:{f}" not in blobs or f"{merged}:{f}" not in blobs:
            continue
        try:
            was, now, had, has = (json.loads(blobs[k]) if k in blobs else None
                                  for k in (f"{base}:{f}", f"{merged}:{f}", f"{fork}:{f}", f"{head}:{f}"))
        except ValueError as e:
            out.append(f"{f}: not JSON on `{BRANCH}`, merged, at the merge base or at the head ({e}), so what it drops can't be "
                       "told")
            continue
        if isinstance(was, dict) and isinstance(now, dict):
            own = _own_drops(had, has)
            out += [f"{f}: {d}" for d in dropped_entries(was, now) if d not in own]
    return out


def _own_drops(fork: object, head: object) -> set[str]:
    """The `shape.inline` premises `dropped_entries` finds from the lock at the merge base (``fork``) to the head's own
    (``head``), both parsed: what the head's regeneration dropped itself, empty when either is missing.  A frozen key is no
    regeneration's to drop, so its drop stays refused."""
    if not (isinstance(fork, dict) and isinstance(head, dict)):
        return set()
    return {d for d in dropped_entries(fork, head) if d.startswith("shape.inline[")}


def lock_drop_refusal(drops: list[str], onto: str = BRANCH) -> str:
    """The refusal of a head whose lock merge onto ``onto`` drops ``drops`` (`lock_drops`), one string naming each and the fix."""
    return (f"its lock merge onto `{onto}` drops {len(drops)} entr{'y' if len(drops) == 1 else 'ies'} `{onto}`'s statements still "
            f"need: {'; '.join(drops)}: {LOCK_DROP_FIX.format(onto=onto)}")


class Preflight:
    """`ready`'s preflight, before the mark (and before the quick tier): refused, each refusal naming its fix, when the head
    (a) conflicts with `next` outside the pins (`merged_tree`, as drive's sync merges, no worktree, with the merge drivers of the
    commit it merges onto, `base_drivers`, never the caller's checkout's), naming the paths and the PRs on `next` that touched them;
    (b) merged onto `next`, fails `pins.py --stale`, a test of `PREFLIGHT_TEST_PATHS`, a lint of `PREFLIGHT_LINTS` the change
    reaches, or `pytest --collect-only` in a suite whose inputs
    the change touches (`touched_suites` with `LocalSuites`, from `next` to the merged tree), or the merged tree's `uv sync --locked`
    finds its lock stale; (c) contains another open PR's head that isn't admitted (no ready mark at that head, held, or no
    owner), naming each; (d) fails one of drive.toml's `landing_gates` at `main` (`_gates`, which logs what its `warn_gates`
    say as warnings), with the gate's fix; or (e) merged
    onto `next`, breaks the build of a Lean module that is or imports, transitively, a module it changes, in that module's package
    or one requiring it by path (`lean_graph`), built by ``lean`` (`LakeBuild`, on a node's warm Lean tree) within `lean_s`, naming the
    module, its first error and the PR on `next` that owns it (`touched_by`), or for a module of its own, those that changed what it
    imports; or (f) merged onto `next` as the sync merges it (`lock_drops`), drops from a lock it changes a `shape.inline` premise or
    a `shape.frozen` key `next`'s statements still need (`dropped_entries`), naming each and the fix (`LOCK_DROP_FIX`); or (g)
    carries, through a merge of `next` (`carried_heads`), a PR's head that isn't on the landing branch and that the PR closed,
    rebuilt without, or isn't admitted at (`stale_carried`), naming each and the fix.  A train most often breaks on one of (a) to
    (c), and a Lean break is most often one of (e) or (f).

    `next` is fetched fresh; a PR already on it is merged onto the commit below its merge there, as drive's next sync would
    rebuild it, found by its trailers among the commits of `next` the landing branch doesn't hold, past those without them
    (`_fetch`).

    The checks of (b) and (e) run in a temporary worktree of the caller's repository at the merge (`merged_worktree`), (e)
    first, within `lean_s`, while the tree is clean (a `research run` ships it), then (b) with its own environment (`sync`;
    ``python`` stands in for its interpreter), under `budget_s` together, and the worktree is removed whatever happens.  Before
    (b), the fetched fixtures under the collected suites' inputs come into it (``fetch``, by default `_fetch_fixtures`: through
    ``store``, within `PREFLIGHT_FETCH_S` of the budget); a suite whose fixtures can't be fetched (no credentials, no source) is
    named once and not judged.  A failure that `next` has too (a pin already stale there, a module two PRs on `next` already
    broke) isn't the PR's: each failed check runs again at `next` in the same worktree, within the same budget, and only what
    fails on the merge alone refuses.  What keeps a check from running (no `next` to fetch, the budget spent, labels that can't
    be pulled, more than `lean_modules` Lean modules, a Lean build that didn't end in time) is said under "checks skipped", never
    as "nothing refused", and refuses nothing: the preflight saves a revert, and the after-landing run still checks the head.  A
    Lean build that couldn't run or built nothing (no node to build on, a run that didn't start) refuses, naming the fix: the
    modules it reaches are few enough to build, and an unbuilt one is how a head that breaks `next`'s Lean gets marked.  Drive's
    sync reads none of it: a ready mark admits the PR, and its merge onto `next` checks it again only for a conflict, a lock's
    drops and the landing gates."""

    def __init__(self, repo: Path, remote: str, store, *, sync: tuple[str, ...] | None = PREFLIGHT_SYNC, python: str | None = None,
                 budget_s: float = PREFLIGHT_CHECKS_S, jobs: int = PREFLIGHT_JOBS, clock: Callable[[], float] = time.monotonic,
                 root: Path | None = None, log: Callable[[str], None] = print,
                 free: Callable[[], float | None] = free_memory,
                 fetch: Callable[[Path, list[str], float], tuple[set[str], str]] | None = None,
                 lean: Callable[[Path, list[str], float], LeanBuilt | str] | None = None,
                 lean_s: float = PREFLIGHT_LEAN_S, lean_modules: int = PREFLIGHT_LEAN_MODULES):
        self.repo, self.remote, self.store, self.sync, self.python = Path(repo), remote, store, sync, python
        self.budget_s, self.jobs, self.clock, self.log, self.free = budget_s, jobs, clock, log, free
        self.fetch = fetch or self._fetch_fixtures
        self.lean, self.lean_s, self.lean_modules = lean, lean_s, lean_modules
        self.root = Path(root or tempfile.gettempdir())

    def __call__(self, pr: PR, prs: list[PR]) -> str | None:
        """None when the head may be marked; else the refusal: before any check when the sync would refuse the head for the
        merges of an old `next` it carries (`_old_next`), since its fix makes another head."""
        start = self.clock()
        try:
            base, onto, members = self._fetch(pr)
        except QueueError as e:
            self.log(f"research queue: preflight: #{pr.number} at {pr.head[:12]}: checks skipped: all ({e}); the after-landing "
                     "run still tests the head")
            return None
        if why := self._old_next(pr, prs):
            return why
        notes: list[str] = []
        found = self._contained(pr, prs) + self._gates(pr, notes)
        tree, files, r = None, [], None
        if not pr.move:
            tree, files, r = merged_tree(self.repo, base, pr.head)
        if pr.move:
            notes.append(f"the merged tree's checks: a move ({pr.move}), which drive's sync regenerates on the tip below it and "
                         f"refuses there unless {MOVE_CHECK} certifies it")
        elif tree is None and files and (moves := self._only_moved(base, files)):
            notes.append(f"the merged tree's checks: it conflicts with {onto} in {', '.join(files)}, files only the move "
                         f"{', '.join(moves)} changed, so drive's sync restacks it through the move and refuses it there if a "
                         "conflict remains")
        elif tree is None and files:
            by = touched_by(self.repo, members, files)
            found.insert(0, f"it conflicts with {onto} in {', '.join(files)}" + (f" (touched by {', '.join(by)})" if by else "")
                         + ": merge `main` once those land, or settle it with their owners; never merge `next` into the branch "
                           "(it carries unlanded PRs)")
            notes.append("the merged tree's checks: it doesn't merge")
        elif tree is None:
            notes.append(f"the merged tree's checks: `git merge-tree` exited {r.returncode} ({(r.stderr or r.stdout).strip()[-200:]})")
        else:
            if files:
                pins, gone, tip_gone = taken(self.repo, tree, pr.head, files)
                if pins:
                    notes.append(Ran(f"conflicts only in pins: {', '.join(pins)}, rebuilt from the merged tree"))
                if gone:
                    notes.append(Ran(f"deletes {', '.join(gone)}, which {onto} changed only by porting it to the module "
                                     "system: deleted in the merge"))
                if tip_gone:
                    notes.append(Ran(f"{onto} deletes {', '.join(tip_gone)}, which the head changed only by porting it to the "
                                     "module system: deleted in the merge"))
            tree, files, pins, why = ported(self.repo, tree, base=base)
            if tree is None:
                found.append(why)
                notes.append("the merged tree's checks: its Lean doesn't take the module system")
            else:
                if files:
                    notes.append(Ran(f"Lean made before the module system: {', '.join(files)} ported by `{MODULES}`"
                                     + (f", and the pins {', '.join(pins)} rebuilt" if pins else "")))
                if drops := lock_drops(self.repo, base, pr.head, tree):
                    found.append(lock_drop_refusal(drops, onto.split("`")[1]))
                try:
                    found += self._merged(pr, base, tree, onto, notes, members)
                except QueueError as e:
                    notes.append(f"the merged tree's checks ({e})")
        skipped = [n for n in notes if not isinstance(n, Ran)]
        took = "; ".join([f"{self.clock() - start:.0f} s", *(n for n in notes if isinstance(n, Ran))])
        if found:
            return (f"the preflight of its head merged onto {onto} refused it ({took}"
                    + (f"; checks skipped: {'; '.join(skipped)}" if skipped else "") + "):\n  - " + "\n  - ".join(found)
                    + "\n  (`--no-preflight` skips the preflight: the lander's)")
        said = f"checks skipped: {'; '.join(skipped)}; the checks that ran passed" if skipped else "nothing refused"
        self.log(f"research queue: preflight: #{pr.number} at {pr.head[:12]} merged onto {onto}: {said} ({took})")
        return None

    def _old_next(self, pr: PR, prs: list[PR]) -> str | None:
        """The sync's refusal (`Queue.entries`) of a head based on `main` that carries merges of an old `next` (`old_next`, on
        the landing branch as `_fetch` fetched it, ``prs`` the open PRs), else None.  One based on another branch isn't judged:
        drive never admits it (`off_main_note`).  What git can't read refuses nothing, said in the log: the sync judges it."""
        if pr.base != "main":
            return None
        into = into_at(self.repo, "refs/queue/preflight/main")
        if _git(self.repo, "rev-parse", "-q", "--verify", f"refs/queue/preflight/{into}^{{commit}}", check=False).returncode:
            into = "main"
        try:
            old = old_next(self.repo, [pr.head], f"refs/queue/preflight/{into}", {p.number for p in prs})
        except QueueError as e:
            self.log(f"research queue: preflight: whether #{pr.number}'s head carries an old `{BRANCH}` isn't known ({e}); the "
                     "sync judges it")
            return None
        return old_next_refusal(list(old.values()), into) if old else None

    def _only_moved(self, base: str, files: list[str]) -> list[str]:
        """The moves drive regenerated below ``base`` (`queue_log`) that ``files``, a merge's conflicts, are all files of, when
        they are; else none."""
        moves = queue_log(self.repo, base)[1]
        moved = set().union(*(changed(self.repo, f"{mv.commit}^1", mv.commit) for mv in moves))
        return [f"{mv.commit[:12]} ({mv.map})" for mv in moves] if moves and set(files) <= moved else []

    def _fetch(self, pr: PR) -> tuple[str, str, list[Member]]:
        """The commit to merge the head onto, how to name it, and `next`'s members up to it.  The members are the merges of `next`
        that the landing branch's tip (`main`'s when that can't be fetched) doesn't contain, by their trailers, past the commits
        without them (`queue_merges`); the commit is the one below the PR's newest merge among them, else `next`'s tip, or with no
        `next`, or one that doesn't hold `main` when `main` is the landing branch (no queue above it), that landing tip."""
        _git(self.repo, *STALL, "fetch", "-q", self.remote, "+refs/heads/main:refs/queue/preflight/main",
             f"+refs/pull/{pr.number}/head:refs/queue/preflight/pr")
        missing = lambda: _git(self.repo, "cat-file", "-e", f"{pr.head}^{{commit}}", check=False).returncode   # noqa: E731
        if missing() and pr.branch:         # GitHub's pull ref lags a push by seconds; the branch has the head `ready` read
            _git(self.repo, *STALL, "fetch", "-q", self.remote, f"+refs/heads/{pr.branch}:refs/queue/preflight/branch", check=False)
        if missing():
            raise QueueError(f"{pr.head[:12]} isn't on {self.remote} (pushed again since?)")
        main, into = _out(self.repo, "rev-parse", "refs/queue/preflight/main"), "main"
        if (land := into_at(self.repo, "refs/queue/preflight/main")) != "main" and not _git(
                self.repo, *STALL, "fetch", "-q", self.remote, f"+refs/heads/{land}:refs/queue/preflight/{land}", check=False).returncode:
            main, into = _out(self.repo, "rev-parse", f"refs/queue/preflight/{land}"), land
        r = _git(self.repo, *STALL, "ls-remote", "--exit-code", self.remote, f"refs/heads/{BRANCH}", check=False)
        if r.returncode == 2:
            _git(self.repo, "update-ref", "-d", f"refs/queue/preflight/{BRANCH}", check=False)
            return main, f"`{into}` at {main[:12]} (no `{BRANCH}`)", []
        if r.returncode:
            raise QueueError(f"git ls-remote {self.remote}: {(r.stderr or r.stdout).strip()[-200:]}")
        _git(self.repo, *STALL, "fetch", "-q", self.remote, f"+refs/heads/{BRANCH}:refs/queue/preflight/{BRANCH}")
        tip = _out(self.repo, "rev-parse", f"refs/queue/preflight/{BRANCH}")
        if into == "main" and _git(self.repo, "merge-base", "--is-ancestor", main, tip, check=False).returncode:
            return main, f"`{into}` at {main[:12]} (`{BRANCH}` at {tip[:12]} doesn't hold it)", []
        members = queue_merges(self.repo, main, tip)
        if own := next((m for m in reversed(members) if m.pr == pr.number), None):
            below = _out(self.repo, "rev-parse", f"{own.commit}^1")
            return below, f"`{BRANCH}` below its merge of #{pr.number} at {own.head[:8]} ({below[:12]})", members[:members.index(own)]
        return tip, f"`{BRANCH}` at {tip[:12]}", members

    def _contained(self, pr: PR, prs: list[PR]) -> list[str]:
        """(c): each open PR whose head this head contains and that the sync wouldn't admit, so that it holds this PR behind it,
        with the fix.  The sync's own admission judges it (`Queue.entries` over this head, taken as marked, and the PRs it
        contains): not marked ready at that head, ejected there, held, with no owner, or containing such a PR itself, unless this
        PR carries its change (`Queue._carried`) or that head is on the landing branch (`land_into`'s tip), so such a mark is
        refused here rather than taken and left for the next sync to say "contains #N, which isn't admitted".  (g): each PR
        whose head this head carries through a merge of `next` that drive's sync leaves off for it (`stale_carried`)."""
        main = "refs/queue/preflight/main"
        heads = {p.head: p.number for p in prs}
        land = f"refs/queue/preflight/{into_at(self.repo, main)}"
        landed = land if not _git(self.repo, "rev-parse", "-q", "--verify", f"{land}^{{commit}}", check=False).returncode else main
        carried = [c for c in carried_heads(self.repo, landed, pr.head) if c[0] != pr.number]
        inside = set(contained(self.repo, main, pr.head, heads)) - {pr.number}
        if not inside and not carried:
            return []
        rules = rules_at(self.repo, main) or Rules()
        labels = StoreLabels(self.store, log=lambda s: print(s, file=sys.stderr), since=rules.marked_after)
        q = Queue(self.repo, remote=self.remote, github=None, ready=lambda n, h: (n, h) == (pr.number, pr.head) or labels.ready(n, h),
                  store=self.store, rules=rules, log=self.log, before_cutoff=labels.before_cutoff, marked_by=labels.marked_by,
                  marked_at=labels.marked_at, prefetch=labels.prefetch)
        labels.change = functools.partial(q.own_change, landing=landed)
        judged = inside | {n for n, _, _ in carried}
        try:
            entries = q.entries([pr, *(p for p in prs if p.number in judged)], {"ejected": {}}, base=main, landed=landed,
                                queued=f"refs/queue/preflight/{BRANCH}")
        except QueueError as e:
            self.log(f"research queue: preflight: can't tell whether the open PRs this head contains are admitted ({e})")
            return []

        def why(n: int) -> str:
            e, lab = entries[n], labels._newest.get((n, entries[n].pr.head))
            if lab is not None and str(lab.value).lower() == "false" and lab.ref:
                return f"ejected at {e.pr.head[:8]} by {lab.by}: {lab.ref}"
            return f"not marked ready at {e.pr.head[:8]}" if e.why.startswith("not marked ready") else e.why

        held = [n for n in entries[pr.number].after if entries[n].state != "ready"]
        stale = stale_carried(self.repo, (c for c in carried if c[0] not in held), {n: e.pr.head for n, e in entries.items()},
                              lambda n: None if entries[n].state == "ready" else why(n), landed.rsplit("/", 1)[-1])
        return [f"it contains {entries[pr.number].holds(n)}, and #{n} isn't admitted ({why(n)}), so drive holds this PR "
                f"behind it: mark #{n} ready first, so it lands first, or drop #{n}'s commits from this branch (if this PR carries "
                f"#{n}'s change, #{n}'s owner withdraws its mark: `research data label pr:{n}@{entries[n].pr.head[:12]} ready false "
                f"--by OWNER`)" for n in held] + list(stale.values())

    def _gates(self, pr: PR, notes: list[str]) -> list[str]:
        """(d): what drive.toml's `landing_gates` at `main` refuse (`gate_refusal`).  A gate that can't judge is a note: the sync
        that would merge the head onto `next` runs the gates again, leaves out a head they refuse, and holds one they can't
        judge waiting, its mark kept, until a sync's gates can.  What its `warn_gates` say is logged as a warning and refuses
        nothing: the sync admits the head, and drive tells its owner."""
        main = _out(self.repo, "rev-parse", "refs/queue/preflight/main")
        try:
            gates, warns = gates_at(self.repo, main), gates_at(self.repo, main, "warn_gates")
        except QueueError as e:
            notes.append(f"no landing gate run ({e})")
            return []
        refused, unjudged = gate_refusal(self.repo, main, pr.head, gates)
        notes += unjudged
        for why in gate_refusal(self.repo, main, pr.head, warns, kind="warn gate")[0]:
            self.log(f"research queue: preflight: warning: #{pr.number} at {pr.head[:12]}: {why}. {WARN_GATE_SAYS}")
        return refused

    def _merged(self, pr: PR, base: str, tree: str, onto: str, notes: list[str], members: list[Member] | None = None) -> list[str]:
        """(b) and (e), in a temporary worktree at the merge of the head onto ``base``, ``tree`` (`merged_tree`)."""
        commit = _out(self.repo, "commit-tree", tree, "-p", base, "-p", pr.head, "-m", f"preflight: #{pr.number} onto {base[:12]}")
        fixtures: dict[str, list[str]] = {}
        out: list[str] = []
        with merged_worktree(self.repo, commit, self.root) as wt:
            graph, lean_failed, lean_deadline = self._lean(wt, base, tree, notes, out)
            deadline = self.clock() + self.budget_s
            failed = self._checks(wt, base, tree, None, notes, deadline, fixtures)
            if not any(failed.values()) and not lean_failed:
                return out
            _git(wt, "checkout", "-q", "--detach", "--force", base)
            _git(wt, "clean", "-qffdx", "-e", "/.venv")
            again = (self._checks(wt, base, tree, {k for k, v in failed.items() if v}, [], deadline, fixtures)
                     if any(failed.values()) else {})
            lean_again = self._lean_at_base(wt, lean_failed, lean_deadline) if lean_failed else {}
        for name, fails in failed.items():
            if fails and name not in again:
                notes.append(f"{name} failed on the merged tree and couldn't run on {onto} to compare: {'; '.join(fails.values())}")
            elif new := [line for k, line in sorted(fails.items()) if k not in again[name]]:
                out.append(self._finding(name, new))
        if isinstance(lean_again, str):
            notes.append(f"Lean: {', '.join(sorted(lean_failed))} failed to build on the merged tree and couldn't be built on {onto} "
                         f"to compare ({lean_again}): {'; '.join(lean_failed.values())}")
        else:
            changed = set(_out(self.repo, "diff", "--name-only", "--no-renames", base, tree).split())
            cache: dict[str, set[str]] = {}
            out += [self._lean_finding(f, line, graph, members or [], f in changed, cache)
                    for f, line in sorted(lean_failed.items()) if f not in lean_again]
        return out

    def _lean(self, wt: Path, base: str, tree: str, notes: list[str], found: list[str]) -> tuple[LeanGraph, dict[str, str], float]:
        """(e): the modules of the merged tree in ``wt`` that the change from ``base`` touches and every module importing one, transitively
        (`lean_graph`), built by ``lean`` (`LakeBuild`) within `lean_s`: the graph, {file: its first Lean error} of each that failed,
        and the step's deadline.  Past `lean_modules` modules, or a build that didn't end in time (`LeanTimedOut`), a note and
        nothing failed; a build that couldn't run or built nothing refuses, into ``found``, naming the fix."""
        deadline = self.clock() + self.lean_s
        if self.lean is None:
            return LeanGraph(), {}, deadline
        changed = _out(self.repo, "diff", "--name-only", "--no-renames", base, tree).split()
        graph = lean_graph(wt, _out(wt, "ls-files").splitlines(), changed)
        if not graph.targets:
            return graph, {}, deadline
        if len(graph.targets) > self.lean_modules:
            notes.append(f"Lean: not built: {len(graph.targets)} modules are or import what it changes, over the "
                         f"{self.lean_modules} the preflight builds; the lock run after its landing builds them")
            return graph, {}, deadline
        res = self.lean(wt, sorted(graph.targets), deadline - self.clock())
        if isinstance(res, LeanTimedOut):
            notes.append(f"lean build timed out ({len(graph.targets)} module(s) not built: {res})")
            return graph, {}, deadline
        if isinstance(res, str):
            found.append(f"Lean: {len(graph.targets)} module(s) that are or import what it changes couldn't be built: {res}")
            return graph, {}, deadline
        notes += [f"Lean: {n}" for n in res.notes]
        notes.append(Ran(f"Lean: built {len(graph.targets)} module(s) {res.where or 'here'}, {len(res.failed)} failed"
                         + (f" (not built before that: {'; '.join(res.passed)})" if res.passed else "")))
        return graph, res.failed, deadline

    def _lean_at_base(self, wt: Path, failed: dict[str, str], deadline: float) -> dict[str, str] | str:
        """Which of the files that ``failed`` on the merged tree fail on `next` too (checked out in ``wt``), or why that couldn't be
        built: a file `next` lacks fails only on the merge."""
        there = sorted(f for f in failed if (wt / f).is_file())
        if not there:
            return {}
        res = self.lean(wt, there, deadline - self.clock())
        return res if isinstance(res, str) else res.failed

    def _lean_finding(self, f: str, line: str, graph: LeanGraph, members: list[Member], own: bool,
                      cache: dict[str, set[str]]) -> str:
        """The refusal for ``f``, which fails to build on the merged tree and not on `next`, naming the PR on `next` that owns it
        (`touched_by`), or, when it is the head's own (``own``) or no member's, those that changed what it imports."""
        mod, stack = graph.module.get(f, f), "merge or stack on that PR, never merge `next` (it carries unlanded PRs)"
        if owners := [] if own else touched_by(self.repo, members, [f], cache):
            return f"Lean: `{mod}` ({f}), of {', '.join(owners)} on `{BRANCH}`, fails to build with this head merged: {line}: {stack}"
        whose = "this head's own" if own else "on `main`"
        if ups := touched_by(self.repo, members, graph.upstream(f), cache):
            return (f"Lean: `{mod}` ({f}), {whose}, fails to build with this head merged: {line}: it imports what "
                    f"{', '.join(ups)} on `{BRANCH}` changed: {stack}")
        return f"Lean: `{mod}` ({f}), {whose}, fails to build with this head merged and not on `{BRANCH}`: {line}: fix the branch"

    @staticmethod
    def _finding(name: str, lines: list[str]) -> str:
        shown = "; ".join(lines[:5]) + (f"; and {len(lines) - 5} more" if len(lines) > 5 else "")
        kind, _, where = name.partition(":")
        if kind == "sync":
            return (f"`uv sync --locked` on the merged tree: {shown}: the merge leaves `uv.lock` stale; run `uv lock` on the branch "
                    f"once `main` has the other side, and push it")
        if kind == "pins":
            return (f"`pins.py --stale` on the merged tree: {shown}: regenerate each with its command, never by hand "
                    f"(`python3 tools/verity/check/pins.py --regen`); when the head alone is fresh, the other side is a PR on "
                    f"`next`: regenerate once it lands")
        if kind == "tests":
            return (f"the invariant tests in {where} fail on the merged tree: {shown}: reproduce on `git merge origin/next` in a "
                    f"scratch worktree, fix the branch, push and mark the new head")
        return (f"`pytest --collect-only` in {where} fails on the merged tree: {shown}: a test reads what another queued PR "
                f"changes or what only an unlanded PR adds; make it read what `main` and `next` hold, or wait for that PR")

    def _run(self, argv: list[str], cwd: Path, deadline: float) -> tuple[subprocess.CompletedProcess | None, str]:
        """``argv`` in ``cwd`` with what's left of the budget, outside the caller's environment (`quick_tier.CALLER_ENV`,
        pytest's own variables); and why it didn't finish."""
        left = deadline - self.clock()
        if left <= 0:
            return None, "the budget was spent before it started"
        env = {k: v for k, v in os.environ.items() if k not in quick_tier.CALLER_ENV and not k.startswith("PYTEST_")}
        env["COLUMNS"] = "300"                                         # pytest's summary lines, whole
        try:
            return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, errors="replace", timeout=left), ""
        except subprocess.TimeoutExpired:
            return None, f"still running when the {self.budget_s:g} s budget ran out"
        except OSError as e:
            return None, f"{argv[0]} didn't start ({e})"

    def _fetch_fixtures(self, wt: Path, paths: list[str], deadline: float) -> tuple[set[str], str]:
        """The fetched registry entries ``paths`` brought into ``wt`` by `research data fetch-fixtures` (`suites.py`'s fetch, every
        byte checked against its id) through ``store``, whose blobs stay there for the next preflight on this machine, within
        `PREFLIGHT_FETCH_S` of what's left by ``deadline``: the paths still missing, and why."""
        left = min(deadline - self.clock(), PREFLIGHT_FETCH_S)
        if left <= 0:
            return set(paths), "the budget was spent before the fetch started"
        root = getattr(self.store, "root", None)
        argv = [sys.executable, "-P", "-m", "research", "data", "fetch-fixtures", *(["--store", str(root)] if root else []),
                "--repo", str(wt), "--json", *(str(wt / p) for p in paths)]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, errors="replace", stdin=subprocess.DEVNULL, timeout=left,
                               check=False)
        except subprocess.TimeoutExpired:
            return set(paths), f"`research data fetch-fixtures` still running after {left:.0f} s"
        except OSError as e:
            return set(paths), f"`research data fetch-fixtures` didn't start ({e})"
        try:
            rep = json.loads(r.stdout)
            missing = {e["path"] for e in rep["entries"] if e["status"] in ("missing", "stale")}
        except (ValueError, KeyError, TypeError):
            return set(paths), f"`research data fetch-fixtures` exited {r.returncode}: {(r.stderr or r.stdout).strip()[-200:]}"
        if not missing:
            return missing, ""
        return missing, ("no source here: no reachable private remote, no public URL" if not rep.get("sources")
                         else f"not on {', '.join(rep['sources'])}")

    def _checks(self, wt: Path, base: str, tree: str, only: set[str] | None, notes: list[str],
                deadline: float, fixtures: dict[str, list[str]]) -> dict[str, dict[str, str]]:
        """{check: {what failed: its line}} for the checks of (b) on the tree checked out in ``wt`` (``only`` those named), each
        that ran by ``deadline``; why each other didn't run goes to ``notes``.  ``fixtures`` gets each collected suite's fetched
        fixtures on the first pass (``only`` None), and a later pass fetches them again."""
        want = (lambda name: True) if only is None else (lambda name: name in only)
        out: dict[str, dict[str, str]] = {}

        def got(name: str, res: dict[str, str] | str) -> None:
            if isinstance(res, str):
                notes.append(f"{name} not checked: {res}")
            else:
                out[name] = res

        if want("pins"):
            if (wt / "tools/verity/check/pins.py").is_file():
                r, why = self._run([sys.executable, "tools/verity/check/pins.py", "--stale"], wt, deadline)
                got("pins", why if r is None else {} if r.returncode == 0 else
                    {ln.split(" is stale", 1)[0]: ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()}
                    or {f"exit {r.returncode}": f"pins.py --stale exited {r.returncode}"})
        if self.sync is not None:
            r, why = self._run(list(self.sync), wt, deadline)
            text = "" if r is not None and r.returncode == 0 else why or (r.stderr or r.stdout).strip()[-300:]
            if r is not None and (not text or _LOCK_STALE in text) and want("sync"):
                out["sync"] = {"uv.lock": text.splitlines()[-1]} if text else {}
            if text:
                if r is None or _LOCK_STALE not in text:
                    notes.append(f"the merged tree's tests not run: `{' '.join(self.sync)}` failed ({text})")
                return out
        py = self.python or str(wt / ".venv" / "bin" / "python")
        run = lambda argv, cwd: self._run(argv, cwd, deadline)                         # noqa: E731
        for d, files in PREFLIGHT_TESTS:
            if want(f"tests:{d}") and (present := [f for f in files if (wt / d / f).is_file()]):
                got(f"tests:{d}", pytest_failures(run, py, wt / d, present))
        lints = [(path, reach) for path, reach in PREFLIGHT_LINTS if want(f"tests:{path}") and (wt / path).exists()]
        changed = _out(self.repo, "diff", "--name-only", base, tree).split() if lints else []
        for path, reach in lints:
            if any(f.startswith(r) or f"/{r}" in f for f in changed for r in reach):
                d, _, rest = path.rpartition("tests/")
                got(f"tests:{path}", pytest_failures(run, py, wt / (d.rstrip("/") or "."), ["tests/" + rest]))
        names = None if only is None else [n.partition(":")[2] for n in only if n.startswith("collect:")]
        fetch = lambda wt_, paths: self.fetch(wt_, paths, deadline)                    # noqa: E731
        res = touched_suites(self.repo, tree, base, LocalSuites(py, run, self.jobs, self.free, names, fetch=fetch,
                                                                fixtures=fixtures), wt=wt)
        fixtures.update(res.fixtures)
        notes += res.notes
        failed = {row["suite"]: dict(zip(row["failed_tests"], row["summary"])) for row in res.failed}
        for s in res.suites:
            out[f"collect:{s}"] = failed.get(s, {})
        return out


class MarkOnly:
    """`ready --mark`: the owner's word alone, with no quick tier, so marking needs no machine and no wait; the after-landing
    run tests the head.  A ready post with no mark behind it admits nothing, so the mark is what an owner gives."""

    def __call__(self, pr: PR) -> None:
        return None


#: the result kinds whose `passes` ({Lean inputs key: package}) say a package passed Lean on those inputs: the Lean fast path's
#: (`lean_changed.py --records`, each package whose records passed) and a check's, a shard's or an after-landing run's (its
#: `lean-audit` step's, `check.py`'s `lean_passes`: each package audited on the commit's own lock with the kernel replay)
FAST_KIND, CHECK_KINDS = "lean-fast/v1", ("check/v1", "check-shard/v1", "check-deferred/v1")
PASS_KINDS = (FAST_KIND, *CHECK_KINDS)
#: the Lean check's step in a check's result
LEAN_AUDIT_STEP = "lean-audit"
#: an attempt whose result may list passes (`index_lean_results`): a check (`check.py`) or a records run
PASS_RUN_RE = re.compile(r"(?:^|[/\s])check\.py\b|lean_changed\.py\b.*\s--records\b")
#: the attempts the records gate reads from the store's remote for passes: those started in the lookback before now
PASSES_LOOKBACK_S = 3 * 86400
#: what keys a package's Lean inputs at a commit (`lean_keys`): that commit's own `KEY_COMMAND --inputs-key`, run from the Python
#: files of `KEY_CODE` there, the directories it loads its helpers from
KEY_COMMAND = "tools/verity/lean/lean_changed.py"
KEY_CODE = ("tools/verity/lean/", "tools/verity/check/", "tools/verity/graph/verity_graph/")
KEY_TIMEOUT_S = 600
#: the fast path's command, which `RecordsGate` names for each package it can't vouch for
RECORDS_COMMAND = ("research run --on auto --mem-gb 32 --max-min 120 --project verity --source . --cwd clone --declared-output "
                   "'lean-audit/*' -- python3 tools/verity/lean/lean_changed.py --records {pkg} --update")
#: why `RecordsGate` warns and never refuses, which each of its warnings says
RECORDS_AFTER_LANDING = ("the lock run after its landing regenerates the records, and what it can't regenerate fails that run, "
                         "a red whose culprit, the PR, is fixed forward or reverted")
#: how many commits below a head `RecordsGate` follows one that only merged the landing branch or only changed other packages' locks
RECORDS_WALK = 8


def _blob(repo: Path, commit: str, path: str) -> bytes | None:
    r = subprocess.run(["git", "cat-file", "blob", f"{commit}:{path}"], cwd=repo, capture_output=True, env=git_env(), check=False)
    return r.stdout if r.returncode == 0 else None


def lean_keys(repo: Path, commit: str, pkgs: Iterable[str] | None = None) -> dict[str, str]:
    """{package: its Lean inputs key at ``commit``} for each of ``pkgs`` (by default every package with a lock there, the
    auditor's fixtures aside): the key a records run or a Lean check of a commit with the same Lean inputs lists among its
    `passes`, printed by ``commit``'s own `KEY_COMMAND --inputs-key` from git's objects in ``repo`` (no checkout, no build), its
    code (`KEY_CODE`'s Python files at ``commit``) written to a temporary directory.  {} when ``commit`` has no such command or
    no package; QueueError when it fails."""
    if pkgs is None:
        pkgs = [posixpath.dirname(f) for f in _out(repo, "ls-tree", "-r", "-z", "--name-only", commit).split("\0")
                if f.endswith("/" + LEAN_LOCK) and not f.startswith(LEAN_FIXTURES)]
    code = scripts_at(repo, commit, KEY_CODE)
    if not (pkgs := sorted(set(pkgs))) or KEY_COMMAND not in code:
        return {}
    with tempfile.TemporaryDirectory(prefix="queue-keys-", ignore_cleanup_errors=True) as d:
        tool = write_scripts(repo, code, Path(d)) / KEY_COMMAND
        if "--inputs-key" not in tool.read_text(errors="replace"):
            return {}
        try:
            r = subprocess.run([sys.executable, str(tool), "--inputs-key", *pkgs, "--head", commit, "--repo", str(Path(repo).resolve())],
                               cwd=d, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=KEY_TIMEOUT_S,
                               env=git_env(GIT_TERMINAL_PROMPT="0", PYTHONDONTWRITEBYTECODE="1"))
        except subprocess.TimeoutExpired:
            raise QueueError(f"`{KEY_COMMAND} --inputs-key` at {commit[:12]} ran over {KEY_TIMEOUT_S} s") from None
    if r.returncode:
        raise QueueError(f"`{KEY_COMMAND} --inputs-key` at {commit[:12]} exited {r.returncode}: "
                         f"{(r.stderr or r.stdout).strip()[-300:]}")
    bad = QueueError(f"`{KEY_COMMAND} --inputs-key` at {commit[:12]} printed no {{package: key}}: {r.stdout.strip()[-200:]}")
    try:
        keys = json.loads(r.stdout)
    except ValueError:
        raise bad from None
    if not isinstance(keys, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in keys.items()):
        raise bad
    return keys


def lean_passed(store, pkg: str, key: str) -> tuple[str, str, str] | None:
    """Whether package ``pkg`` passed Lean on the inputs whose key (`lean_keys`) is ``key``: (the run, its result's kind, the
    commit it ran on) of the newest attempt, of any commit, that ended by itself on a clean checkout and whose result, of one of
    `PASS_KINDS`, lists ``key`` for ``pkg`` among its `passes`; None when none does.  One indexed query of the store's index,
    which holds what this machine has read of the store (`index_lean_results` reads the newest results from its remote)."""
    rows = store.index.sql(
        "SELECT t.id, a.kind, t.source_commit FROM artifact_meta m JOIN artifacts a ON a.id = m.id "
        "JOIN attempt_outputs o ON o.artifact = m.id AND o.name = 'result' JOIN attempts t ON t.id = o.attempt "
        f"WHERE m.key = ? AND m.value = ? AND a.kind IN ({', '.join('?' * len(PASS_KINDS))}) AND t.state IN ('done', 'failed') "
        "AND t.rc BETWEEN 0 AND 127 AND t.source_dirty = 0 ORDER BY t.start_utc DESC, t.id DESC LIMIT 1",
        (f"passes.{key}", pkg, *PASS_KINDS))
    return (str(rows[0][0]), str(rows[0][1]), str(rows[0][2])) if rows else None


def index_lean_results(store, since: str, jobs: int = 16) -> list[str]:
    """Index the result of each attempt this store holds since ``since`` (a run id or its prefix) that may list passes
    (`PASS_RUN_RE`: a check, or a records run) and that its index lacks, the manifest fetched from the store's remote
    (`get_manifest`, ``jobs`` at a time), so `lean_passed` finds their `passes`.  What can't be read, a line each."""
    rows = store.index.sql("SELECT id, manifest_json FROM attempts WHERE id > ? AND source_dirty = 0 "
                           "AND (manifest_json LIKE '%check.py%' OR manifest_json LIKE '%lean_changed.py%')", (since,))
    want: dict[str, None] = {}
    errs: list[str] = []
    for rid, doc in rows:
        try:
            att = json.loads(doc)
            argv, art = " ".join(map(str, att.get("argv") or [])), (att.get("outputs") or {}).get("result")
        except (ValueError, TypeError, AttributeError) as e:
            errs.append(f"{rid}: its manifest can't be read ({type(e).__name__})")
            continue
        if isinstance(art, str) and PASS_RUN_RE.search(argv) and store.index.artifact(art) is None:
            want[art] = None

    def get(art: str) -> str | None:
        try:
            store.get_manifest(art)
        except Exception as e:  # noqa: BLE001  (a result that can't be read lists no pass)
            return f"{art}: {type(e).__name__}: {e}"
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        errs += [e for e in ex.map(get, want) if e]
    return errs


class RecordsGate:
    """`ready`'s records gate, before the preflight and the mark, which warns and never refuses: a head whose change against the
    landing branch (`into_at` `main`: `checked` under `land_into = "checked"`, else `main`, whose locks the PRs that landed there
    were held to; from their merge base) touches a Lake package's lock (`lean-audit.json`) samples a warning, naming the run that
    would record it, for each such package unless the store holds a pass of it on the head's Lean inputs (`lean_passed`): a
    records run or a Lean check, of any commit, whose result lists the package's Lean inputs key at the head among its
    `passes`.  The lock run after its landing regenerates the records (`RECORDS_AFTER_LANDING`), so nothing here holds the mark.
    The key covers every file of the package and of each package it reads (`lean_changed.closure_inputs`: its lock, tests and
    docs included), their toolchains and the checker, so a pass on it is a pass of the head's lock wherever it ran: a PR that
    carries another PR's Lean change needs no run of its own.  A records run lists a package whose record its `--update` wrote
    under that record's key, so the commit of exactly the records one run kept over the commit it ran on is vouched for with no
    run of its own, and the commit it ran on isn't.  A run that failed a package lists no pass of it, and the warning names the
    package's failures (`_failures`).

    Two heads are vouched for by the commit below them, followed `RECORDS_WALK` commits at most: one that only merged the
    landing branch into a vouched commit, every file it changed being that branch's whole but a lock, which is that branch's or
    `tools/verity/lean/merge.py`'s merge of the two sides (the fast path reports `path: reused` for it: a candidate, which the
    lock run after its landing rebuilds and regenerates where it differs); and one whose one parent is vouched and that changes
    only other packages' locks over it, which a package's records don't read.

    The store is read from its remote (`refresh(attempts_only=True, since=...)` and `index_lean_results`: the runs of the last
    `PASSES_LOOKBACK_S`, beside the passes this machine read before); a store with no usable remote, or one that can't be listed,
    vouches for nothing, and the warning says why, as it does for a head whose changed locks can't be told."""

    def __init__(self, repo: Path, remote: str, store, *, log: Callable[[str], None] = print,
                 now: Callable[[], float] = time.time, lock_merge: Path | None = None):
        self.repo, self.remote, self.store, self.log, self.now = Path(repo), remote, store, log, now
        self.lock_merge = lock_merge or self.repo / "tools" / "verity" / "lean" / "merge.py"
        self._merges: dict[str, bool] = {}
        #: {commit: {package: its Lean inputs key there}} (`lean_keys`)
        self._keys: dict[str, dict[str, str]] = {}
        self.notes: list[str] = []
        #: the landing branch the head's change is judged against (`_locks`)
        self.into = "main"

    def __call__(self, pr: PR) -> str | None:
        """None when it vouches for each lock the head changes; else the warning, naming for each package it doesn't vouch for
        the command that would record it."""
        try:
            locks = self._locks(pr)
        except QueueError as e:
            return (f"the records gate can't tell which Lean locks it changes ({e}), so it vouches for none: "
                    f"{RECORDS_AFTER_LANDING}")
        if not locks:
            return None
        try:
            self._read()
        except Exception as e:  # noqa: BLE001  (no store, no remote, a failed listing: vouch for none, never from what's here)
            return (f"it changes the Lean lock of {', '.join(locks)}, and the store's records runs can't be read "
                    f"({type(e).__name__}: {e}; the remote is R2_ENDPOINT and R2_BUCKET, or ~/.research/store.toml), so the "
                    f"records gate vouches for none: {RECORDS_AFTER_LANDING}")
        missing = []
        for pkg in locks:
            if how := self._vouch(pr.head, pkg, 0):
                self.log(f"research queue: records gate: {pkg}: {how}")
            else:
                missing.append(pkg)
        if not missing:
            return None
        lines = []
        for pkg in missing:
            lines += self._failures(pr.head, pkg)
            lines.append(f"to record {pkg} before it lands, start `{RECORDS_COMMAND.format(pkg=pkg)}`; when its result keeps "
                         f"records (`lean-audit/record/<path>/{LEAN_LOCK}`, {pkg}'s and each requirer's it regenerated), commit "
                         f"exactly those over the locks of their paths and mark that commit, which needs no run of its own")
        notes = f" ({'; '.join(self.notes[:3])})" if self.notes else ""
        return (f"it changes the Lean lock of {', '.join(missing)} against `{self.into}`, and the store holds no pass of "
                f"{'it' if len(missing) == 1 else 'each'} on {pr.head[:12]}'s Lean inputs (`{KEY_COMMAND} --inputs-key`) by a "
                f"records run or a Lean check of any commit, nor below a head that only merged `{self.into}` or changed only other "
                f"packages' locks{notes}: {RECORDS_AFTER_LANDING}:\n  - " + "\n  - ".join(lines))

    def _locks(self, pr: PR) -> dict[str, str]:
        """{package: its lock's path} for each lock the head's change against the landing branch (`self.into`) touches and the
        head still has, the auditor's fixtures (`LEAN_FIXTURES`) aside: no records run reads one (`lean_changed.py --records`
        matches no package there), and the controls write their records. A head stacked on a PR since landed there, which also
        merged `main`, has several merge bases with it, and from the stacked-on PR's head everything that merge brought counts
        as the head's own: so a file is the head's change only where it differs from every merge base."""
        _git(self.repo, *STALL, "fetch", "-q", self.remote, "+refs/heads/main:refs/queue/records/main",
             f"+refs/pull/{pr.number}/head:refs/queue/records/pr")
        self.into = into_at(self.repo, "refs/queue/records/main")
        _git(self.repo, *STALL, "fetch", "-q", self.remote, f"+refs/heads/{self.into}:refs/queue/records/base")
        if _git(self.repo, "cat-file", "-e", f"{pr.head}^{{commit}}", check=False).returncode:
            raise QueueError(f"{pr.head[:12]} isn't on {self.remote} (pushed again since?)")
        bases = _out(self.repo, "merge-base", "--all", "refs/queue/records/base", pr.head).split()
        files = set.intersection(*(set(_out(self.repo, "diff", "--name-only", "--no-renames", b, pr.head).splitlines())
                                   for b in bases)) if bases else \
            set(_out(self.repo, "diff", "--name-only", "--no-renames", f"refs/queue/records/base...{pr.head}").splitlines())
        return {posixpath.dirname(f) or ".": f for f in sorted(files) if posixpath.basename(f) == LEAN_LOCK
                and not f.startswith(LEAN_FIXTURES) and _git(self.repo, "cat-file", "-e", f"{pr.head}:{f}", check=False).returncode == 0}

    def _read(self) -> None:
        """Bring this store's attempts of the last `PASSES_LOOKBACK_S`, and the results among them that may list passes, up to
        date from its remote."""
        if self.store.remote is None:
            raise QueueError("this machine's store has no remote, so the runs other machines published aren't here")
        since = time.strftime("r%Y%m%d-%H%M%S", time.gmtime(self.now() - PASSES_LOOKBACK_S))
        rep = self.store.refresh(attempts_only=True, since=since)
        if errs := rep.get("errors"):
            self.notes.append(f"{len(errs)} attempt(s) on the remote couldn't be read, e.g. {errs[0]}")
        if errs := index_lean_results(self.store, since):
            self.notes.append(f"{len(errs)} result(s) that may list passes couldn't be read, e.g. {errs[0]}")

    def _key(self, commit: str, pkg: str) -> str | None:
        """``pkg``'s Lean inputs key at ``commit`` (`lean_keys`, every package's at once), None when it has none (said)."""
        if commit not in self._keys:
            try:
                self._keys[commit] = lean_keys(self.repo, commit)
            except QueueError as e:
                self._keys[commit] = {}
                self.notes.append(f"the Lean inputs of {commit[:12]} can't be keyed ({e})")
            else:
                if not self._keys[commit]:
                    self.notes.append(f"{commit[:12]} has no `{KEY_COMMAND} --inputs-key` to key its Lean inputs by: merge "
                                      f"`{self.into}` into it")
        return self._keys[commit].get(pkg)

    def _vouch(self, commit: str, pkg: str, depth: int) -> str | None:
        """Why ``pkg``'s lock at ``commit`` passed Lean, or None: a pass on its Lean inputs there (`lean_passed`), or one below a
        commit that only merged the landing branch or only changed other packages' locks."""
        if (key := self._key(commit, pkg)) and (got := lean_passed(self.store, pkg, key)):
            rid, kind, at = got
            what = f"records run {rid}" if kind == FAST_KIND else f"the Lean check {rid}"
            return f"{what} of {at[:12]} passed it on {commit[:12]}'s Lean inputs (key {key[:12]})"
        if depth >= RECORDS_WALK:
            return None
        parents = _out(self.repo, "rev-list", "--parents", "-n", "1", commit).split()[1:]
        if len(parents) == 2 and self._main_merge(commit, *parents):
            how = self._vouch(parents[0], pkg, depth + 1)
            return how and f"{commit[:12]} only merges `{self.into}`'s {parents[1][:12]} into {parents[0][:12]}, where {how}"
        if len(parents) != 1:
            return None
        files = _out(self.repo, "diff", "--name-only", "--no-renames", parents[0], commit).splitlines()
        lock = f"{pkg}/{LEAN_LOCK}" if pkg != "." else LEAN_LOCK
        if not files or lock in files or any(posixpath.basename(f) != LEAN_LOCK for f in files):
            return None
        how = self._vouch(parents[0], pkg, depth + 1)
        return how and f"{commit[:12]} changes only other packages' locks over {parents[0][:12]}, where {how}"

    def _main_merge(self, commit: str, ours: str, theirs: str) -> bool:
        """Whether ``commit`` only merges the landing branch's ``theirs`` into ``ours``: each file it changes over ``ours`` is
        that branch's whole, a lock aside, which may be `merge.py`'s merge of the two sides instead."""
        if commit not in self._merges:
            ok = _git(self.repo, "merge-base", "--is-ancestor", theirs, "refs/queue/records/base", check=False).returncode == 0
            base = _out(self.repo, "merge-base", ours, theirs) if ok else ""
            for f in _out(self.repo, "diff", "--name-only", "--no-renames", ours, commit).splitlines() if ok else []:
                mine = _blob(self.repo, commit, f)
                if mine == _blob(self.repo, theirs, f):
                    continue
                if not (posixpath.basename(f) == LEAN_LOCK and mine is not None and self._merged_lock(base, ours, theirs, f, mine)):
                    ok = False
                    break
            self._merges[commit] = ok
        return self._merges[commit]

    def _merged_lock(self, base: str, ours: str, theirs: str, path: str, mine: bytes) -> bool:
        """Whether ``mine`` is `tools/verity/lean/merge.py`'s merge (this checkout's) of ``path`` at ``base``, ``ours``, ``theirs``."""
        tool = self.lock_merge
        if not tool.is_file():
            return False
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            sides = []
            for name, c in (("base", base), ("ours", ours), ("theirs", theirs)):
                (p := Path(d) / name).write_bytes(_blob(self.repo, c, path) or b"")
                sides.append(str(p))
            try:
                r = subprocess.run([sys.executable, str(tool), *sides, path], cwd=self.repo, capture_output=True, timeout=300,
                                   check=False)
                return r.returncode == 0 and json.loads(Path(sides[1]).read_text()) == json.loads(mine)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                return False

    def _failures(self, head: str, pkg: str) -> list[str]:
        """What the runs of ``head`` say against ``pkg``, the newest first, down to the newest finished records run that names
        it: a Lean check or a records run that didn't finish, a Lean check whose result lists no pass of it, and a records run
        that passed it only on the record its `--update` wrote (which the head doesn't hold), failed it or refused."""
        out = []
        rows = self.store.index.sql("SELECT id, state, rc, manifest_json FROM attempts WHERE source_commit = ? AND source_dirty = 0 "
                                    "ORDER BY start_utc DESC, id DESC", (head,))
        for rid, state, rc, doc in rows:
            try:
                art = (json.loads(doc).get("outputs") or {}).get("result")
                meta = (self.store.index.artifact(art) or {}).get("manifest", {}).get("meta") or {} if isinstance(art, str) else {}
            except (ValueError, TypeError, AttributeError) as e:
                out.append(f"run {rid} of {head[:12]}: its result can't be read ({type(e).__name__}: {e}), so it says nothing")
                continue
            step = rec = None
            if meta.get("kind") in CHECK_KINDS:
                step = next((s for s in meta.get("steps") or [] if isinstance(s, dict) and s.get("step") == LEAN_AUDIT_STEP), None)
            elif meta.get("kind") == FAST_KIND:
                rec = (meta.get("records") or {}).get(pkg)
            if not isinstance(step, dict) and not isinstance(rec, dict):
                continue
            what = f"the Lean check {rid}" if step is not None else f"records run {rid}"
            if state not in ("done", "failed") or not isinstance(rc, int) or not 0 <= rc <= 127:
                out.append(f"{what} of {head[:12]} didn't finish (state {state}, rc {rc}), so it vouches for nothing")
                continue
            if step is not None:
                out.append(f"{what} of {head[:12]} lists no pass of {pkg} on the head's lock among its result's `passes` (its "
                           f"`lean-audit` step {step.get('status')})")
                continue
            if rec.get("outcome") == "regenerated" and rec.get("record"):
                out.append(f"records run {rid} of {head[:12]} passed {pkg} only on the record its `--update` wrote, which differs "
                           f"from the head's lock: commit its `{rec['record']}` over `{pkg}/{LEAN_LOCK}`")
                break
            if rec.get("outcome") in ("failed", "refused"):
                out.append(f"records run {rid} of {head[:12]} {rec['outcome']} {pkg} (`research status {rid}` and its "
                           f"`lean-audit/` reports say why)")
                break
        return out


def ejection(store, pr: PR) -> str | None:
    """Why ``pr``'s current head was ejected, when the newest `ready` label on it is `research queue eject`'s `false` (its reason in
    `ref`), pulled from the store's remote first; else None, and None when that pull fails: this saves a revert, and the after-landing
    run checks the head."""
    try:
        labs = StoreLabels(store, log=lambda s: print(s, file=sys.stderr))._labels(pr.number, pr.head, vocab.READY)
    except QueueError as e:
        print(f"research queue ready: can't tell whether #{pr.number}'s head was ejected ({e}); the after-landing run still checks it",
              file=sys.stderr)
        return None
    last = labs[-1] if labs else None
    if last is None or str(last.value).lower() != "false" or not last.ref:
        return None
    return f"{last.ref} (ejected by {last.by} at {last.ts})"


#: how many times `ready --sha` reads GitHub's open PRs, and the seconds between reads, before refusing a head GitHub doesn't show
SHA_READS, SHA_WAIT_S = 6, 5.0


def off_main_note(pr: PR, prs: list[PR]) -> str | None:
    """What `ready` tells the owner of ``pr`` when it isn't based on `main` (drive admits only PRs that are), or None."""
    if pr.base == "main":
        return None
    parent = next((p for p in prs if p.branch == pr.base and p.number != pr.number), None)
    until = f"until #{parent.number} lands, one stack level per landing" if parent else "until it is"
    return (f"#{pr.number} is based on {based_on(pr, prs)}, not main: drive won't see it {until}. Retarget it to main (drive orders "
            f"stacked PRs itself: a PR containing another waits for it); retargeting keeps the head, so a ready mark at "
            f"{pr.head[:12]} counts from then on, and a new push needs a new mark unless its own change is the same")


def ready(github, store, number: int | None, by: str, tier: Callable[[PR], str | None], again: bool = False,
          preflight: Callable[[PR, list[PR]], str | None] | None = None, sha: str | None = None,
          sleep: Callable[[float], None] = time.sleep, as_json: bool = False,
          records: Callable[[PR], str | None] | None = None, goals: Callable[[PR], str | None] | None = None) -> int:
    """`research queue ready PR`: run the quick tier at the PR's current head (`tier`, a `RemoteQuickTier` or a `QuickTier`), and once
    it passes write `ready = true` on `pr:{n}@{that head}` and push it to the store's remote.  A head whose tree is that of another
    head marked ready (`same_tree_mark`) is marked without running the tier again.  A head drive or the lander ejected is refused
    unless ``again`` (a re-mark at an ejected head would fail a second train on the same line).  Before any of it, a PR
    `PR.outside` refuses (its head on another repository than ``github``'s, or an author the organization doesn't have) is
    refused, then ``goals`` (`goal_gate`) warns, on stderr, of a PR whose body names no live goal of the Goals page and no
    `Issue: none: REASON`, and goes on, then ``records`` (`RecordsGate`) warns, as ``goals`` does, of a head that changes a Lean
    lock no records run passed, of it or of a commit with the same inputs for that package (the lock run after its landing
    regenerates the records), then ``preflight`` (`Preflight`; None with `--no-preflight`) refuses a head that would break
    `next`, with the fix.  With ``sha`` (`--sha`) it marks only that head: GitHub's head lags a push by seconds (`git push &&
    research queue ready N --mark` can mark the old head), so it reads again, up to `SHA_READS` times, and refuses when GitHub
    still has another head.  A PR based
    on another branch is marked all the same, and told that drive won't see it until it is based on `main` (`off_main_note`).
    With ``as_json`` (`--json`) the rest goes to stderr, and stdout gets one object: pr, head, base, exit and warning (that note)."""
    with contextlib.redirect_stdout(sys.stderr) if as_json else contextlib.nullcontext():
        rc, pr, prs = _ready(github, store, number, by, tier, again, preflight, sha, sleep, records, goals)
        note = off_main_note(pr, prs) if pr is not None else None
        if note:
            print(f"research queue ready: {note}", file=sys.stderr)
    if as_json:
        print(json.dumps({"pr": number, "head": pr and pr.head, "base": pr and pr.base, "exit": rc, "warning": note}))
    return rc


def _ready(github, store, number: int | None, by: str, tier: Callable[[PR], str | None], again: bool,
           preflight: Callable[[PR, list[PR]], str | None] | None, sha: str | None,
           sleep: Callable[[float], None], records: Callable[[PR], str | None] | None = None,
           goals: Callable[[PR], str | None] | None = None) -> tuple[int, PR | None, list[PR]]:
    if not sha and number is not None and (p := next((x for x in github.open_prs() if x.number == number), None)) and p.branch \
            and (tip := getattr(github, "branch_tip", lambda _b: None)(p.branch)):
        sha = tip                                           # a mark seconds after a push would preflight GitHub's old head
    for i in range(SHA_READS if sha else 1):
        prs = github.open_prs() if number is not None else []
        pr = next((p for p in prs if p.number == number), None)
        if pr is None or not sha or pr.head.startswith(sha):
            break
        if i + 1 < SHA_READS:
            sleep(SHA_WAIT_S)
    if pr is None:
        print(f"research queue ready: {'#' + str(number) if number is not None else 'a PR number'} is not an open PR", file=sys.stderr)
        return 2, pr, prs
    if why := pr.outside(github.repo):
        print(f"research queue ready: #{pr.number} is not marked ready: {why}", file=sys.stderr)
        return 1, pr, prs
    if sha and not pr.head.startswith(sha):
        print(f"research queue ready: #{pr.number} is not marked: GitHub has its head at {pr.head[:12]}, not {sha[:12]}, after "
              f"{SHA_READS} reads {SHA_WAIT_S:g} s apart. Push {sha[:12]} to the PR's branch (or pass the head GitHub shows), then "
              "run this again", file=sys.stderr)
        return 1, pr, prs
    if goals is not None and (why := goals(pr)):
        print(f"research queue ready: warning: #{pr.number} at {pr.head[:12]}: {why}", file=sys.stderr)
    if not again and (why := ejection(store, pr)):
        print(f"research queue ready: #{pr.number} is not marked ready at {pr.head[:12]}: that head was ejected: {why}. Push the fix "
              f"and mark the new head; if the ejection was wrong (a machine fault), say so in the merge thread and pass --again",
              file=sys.stderr)
        return 1, pr, prs
    if records is not None and (why := records(pr)):
        print(f"research queue ready: warning: #{pr.number} at {pr.head[:12]}: {why}", file=sys.stderr)
    if preflight is not None and (why := preflight(pr, prs)):
        print(f"research queue ready: #{pr.number} is not marked ready at {pr.head[:12]}: {why}", file=sys.stderr)
        return 1, pr, prs
    if (repo := getattr(tier, "repo", None)) is not None:
        try:
            old = same_tree_mark(store, repo, tier.remote, pr)
        except QueueError as e:
            print(f"research queue: {e}; running the quick tier", file=sys.stderr)
            old = None
        if old:
            print(f"research queue: #{pr.number}'s head {pr.head[:12]} has the tree of {old[:12]}, marked ready after its quick tier "
                  f"passed: the same files, so the tier isn't run again", flush=True)
            return _mark(store, pr, by, None, "ready"), pr, prs
    if isinstance(tier, MarkOnly):
        print(f"research queue: #{pr.number} at {pr.head[:12]}: marked on {by}'s word, with no quick tier", flush=True)
        return _mark(store, pr, by, None, "ready"), pr, prs
    print(f"research queue: the quick tier at #{pr.number}'s head {pr.head[:12]}: `suites.py --quick` on the suites it changes", flush=True)
    return _mark(store, pr, by, tier(pr), "ready"), pr, prs


def judge(github, store, ref: str, by: str, on: str | None = None, fetch: Callable[[Path], dict] | None = None,
          runs_root: Path | None = None, repo: Path | None = None, url: str | None = None,
          records: Callable[[PR], str | None] | None = None, goals: Callable[[PR], str | None] | None = None) -> int:
    """`research queue judge RUN`: what `ready --on` does once its run ends, for a quick-tier run that already ended (its waiting
    client gone): fetch its records (`research fetch`; a run another VM launched is adopted from its machine, ``on`` or each), judge
    them as `RemoteQuickTier` does, at the head the tier ran at, and mark the open PR at that head ready, or, when none is there now
    and ``repo`` (a clone of ``url``) is given, the one open PR whose head has that head's tree (`same_tree_mark`'s reason).
    Refused when the run can't be fetched, hasn't ended, isn't a quick tier, failed, or no open PR is at that head or tree now;
    ``goals`` (`goal_gate`) and ``records`` (`RecordsGate`) warn, as in `ready`."""
    from . import cli, remote
    from . import runs as R

    def refuse(why: str) -> int:
        print(f"research queue judge: {why}", file=sys.stderr)
        return 1

    try:
        d = cli._run_dir(R.runs_root(runs_root), ref, on)
        rec = (fetch or remote.fetch)(d)
    except (FileNotFoundError, LookupError, remote.Refused) as e:
        return refuse(str(e))
    rid, machine, status = d.name, rec.get("machine") or "its machine", _json_file(d / "status.json")
    if status.get("state") not in R.TERMINAL:
        err = (rec.get("last_fetch_error") or {}).get("stderr")
        return refuse(f"run {rid} hasn't ended (last seen {status.get('state') or 'without a status'}"
                      f"{'; its machine is unreachable: ' + err.strip()[-200:] if err else ''}): `research inspect {rid}`")
    res, why = tier_report(d, rid, machine, status)
    if why:
        return refuse(why)
    head = tier.get("head") if isinstance(tier := res.doc.get("tier"), dict) else None
    open_prs = github.open_prs()
    prs = [p for p in open_prs if p.head == head]
    if not prs and repo is not None and isinstance(head, str):
        if len(prs := same_tree_prs(repo, url, open_prs, head)) == 1:
            print(f"research queue: #{prs[0].number}'s head {prs[0].head[:12]} has the tree of {head[:12]}, where run {rid} ran the "
                  f"quick tier: the same files", flush=True)
    if len(prs) != 1:
        return refuse(f"run {rid} ran the quick tier at {str(head)[:12]}, " + (
            f"the head of each of {', '.join('#' + str(p.number) for p in prs)}: mark one with `research queue ready`" if prs else
            "which is no open PR's head now, nor its tree (pushed again since, or closed?)"))
    if why := prs[0].outside(github.repo):
        return refuse(f"#{prs[0].number} is not marked ready: {why}")
    if goals is not None and (why := goals(prs[0])):
        print(f"research queue judge: warning: #{prs[0].number} at {prs[0].head[:12]}: {why}", file=sys.stderr)
    if records is not None and (why := records(prs[0])):
        print(f"research queue judge: warning: #{prs[0].number} at {prs[0].head[:12]}: {why}", file=sys.stderr)
    covers = allowed(repo or Path.cwd(), tier.get("base") if isinstance(tier, dict) else None)
    return _mark(store, prs[0], by, judge_report(res, rid, machine, head, status, covers), "judge")


def _mark(store, pr: PR, by: str, failed: str | None, command: str) -> int:
    """`ready = true` on `pr:{n}@{head}`, pushed to the store's remote, unless the quick tier ``failed`` there (the refusal), or
    ``store``, a `labelauth.Signer` (`label_signer`), refuses to write a mark it can't sign."""
    from .labelauth import AuthError
    from .store.cli import _label_write_through

    if isinstance(failed, StillGoing):
        print(f"research queue {command}: #{pr.number} is not marked ready at {pr.head[:12]} yet. {failed}", file=sys.stderr)
        return waitcap.STILL_GOING
    if isinstance(failed, NotStarted):
        print(f"research queue {command}: #{pr.number} is not marked ready at {pr.head[:12]}: the quick tier never started: {failed}. "
              f"Nothing failed: run it again (`--on auto` takes the machine with a free check slot). A PR lands on `checked` on a review, "
              f"and reaches `main` on the full check after, so the quick tier isn't required: `ready N --mark --by LANE` marks it; to run its tests anyway: {LANE_RUN}", file=sys.stderr)
        return READY_NOT_STARTED
    if failed is not None:
        print(f"research queue {command}: #{pr.number} is not marked ready at {pr.head[:12]}: the quick tier failed: {failed}. Push a "
              f"fix, or run it again: passes are cached by input, so a re-run on the same cache runs only what failed", file=sys.stderr)
        return 1
    target = label_target(pr.number, pr.head)
    try:
        store.add_label(target, vocab.READY, True, by)
    except AuthError as e:
        print(f"research queue {command}: #{pr.number} is not marked ready at {pr.head[:12]}: {e}", file=sys.stderr)
        return 1
    said, lost = _label_write_through(store, target)
    print(f"research queue: #{pr.number} is ready at {pr.head[:12]} ({target}); {said}")
    return 1 if lost else 0


def review(q: Queue, store, labels: StoreLabels, pr: int | None, verdict: str | None, ref: str | None, by: str | None,
           sha: str | None = None, as_json: bool = False) -> int:
    """`research queue review`: without ``pr``, list the open PRs onto `main` marked ready at their head and not on `main` that no
    land verdict at their head lands (`unreviewed`), oldest mark first: those the verified queue waits on a reviewer for (a
    low-risk one lands on its mark alone, so its row says it); with it, write ``verdict`` (`land` or `hold`, ``ref`` its note or
    reason) by ``by`` on the PR's head on `next` (or ``sha``, or its open head when it isn't on `next`), refusing its owner.  Without
    ``sha`` it refuses when GitHub has a head other than `next`'s, as the reviewer may have read either; a short ``sha`` is the
    known head or clone commit it starts.  The list names each land verdict ``labels`` doesn't authenticate
    (`StoreLabels.refusals`); ``store``, a `labelauth.Signer` (`label_signer`), refuses a verdict it can't sign."""
    from .labelauth import AuthError
    from .store.cli import _label_write_through

    q._clone()
    opened = {p.number: p for p in q.github.open_prs()}
    if pr is None:
        onto = [p for p in opened.values() if p.base == "main"]
        q.fetch(onto)
        q.unlock()
        main = _out(q.workdir, "rev-parse", "refs/remotes/origin/main")
        here = [p for p in onto if not _git(q.workdir, "cat-file", "-e", f"{p.head}^{{commit}}", check=False).returncode
                and _git(q.workdir, "merge-base", "--is-ancestor", p.head, main, check=False).returncode]
        labels.prefetch([(p.number, p.head) for p in here])
        rows = []
        for p in here:
            if not q.ready(p.number, p.head):
                continue
            if (why := unreviewed(labels.reviews(p.number, p.head), review_owners(q, labels, p.number, p.head, p.owner, opened))) is None:
                continue
            if refused := labels.refusals(p.number, p.head, vocab.REVIEWED):
                why += f"; not authenticated: {'; '.join(refused)}"
            base = _out(q.workdir, "merge-base", main, p.head)
            rows.append({"pr": p.number, "head": p.head, "owner": p.owner, "lean": lean_change(q.workdir, p.head, base),
                         "files": sorted(changed(q.workdir, base, p.head)), "why": why,
                         "marked": q.marked_at(p.number, p.head) if q.marked_at else None})
        rows.sort(key=lambda r: (r["marked"] or "", r["pr"]))
        if as_json:
            print(json.dumps(rows, indent=1))
        else:
            for r in rows:
                shown = ", ".join(r["files"][:8]) + (f" (+{len(r['files']) - 8} more)" if len(r["files"]) > 8 else "")
                print(f"#{r['pr']} {r['head'][:12]} {r['owner']} Lean change: {'yes' if r['lean'] else 'no'}, marked "
                      f"{r['marked'] or '?'}: {r['why']}\n    {shown}")
            print(f"research queue review: {len(rows)} marked PR(s) not on `main` lack a land verdict")
        return 0
    q.fetch()
    chain = q.chain()
    m = next((m for m in chain if m.pr == pr), None)
    live = (p := opened.get(pr)) and ((p.branch and getattr(q.github, "branch_tip", lambda _b: None)(p.branch)) or p.head)
    if sha:
        # a full SHA is checked too: a mistyped one would record a verdict on a commit that doesn't exist, which never counts
        known = {h for h in (m and m.head, live) if h and h.startswith(sha)}
        r = _git(q.workdir, "rev-parse", "--verify", "-q", f"{sha}^{{commit}}", check=False)
        if len(known) != 1 and r.returncode:
            raise QueueError(f"review: --sha {sha} is neither #{pr}'s head on `{q.branch}` nor on GitHub, nor a commit the queue's "
                             f"clone has: pass {'the' if len(sha) < 40 else 'its'} full 40-character SHA")
        sha = known.pop() if len(known) == 1 else r.stdout.strip()
    if sha is None and m and live and live != m.head:
        raise QueueError(f"review: #{pr} is on `{q.branch}` at {m.head[:12]}, but GitHub has its head at {live[:12]}, pushed since "
                         f"drive's last sync: pass --sha with the head you read ({live} or {m.head})")
    head, owner = (sha or m.head, m.owner) if m and (sha in (None, m.head)) else (sha, None)
    if owner is None:
        if (open_pr := opened.get(pr)) is None:
            raise QueueError(f"review: #{pr} is neither on `{q.branch}` nor open")
        head, owner = head or open_pr.head, open_pr.owner
    if not owner or _who(by) in {_who(o) for o in review_owners(q, labels, pr, head, owner, opened) if o}:
        raise QueueError(f"review: #{pr} is {by}'s own PR (its owner, its owner's handle or its ready mark's author; or it has no "
                         "owner), and an owner's verdict doesn't count: a reviewer who isn't its owner reviews it")
    target = label_target(pr, head)
    try:
        store.add_label(target, vocab.REVIEWED, verdict, by, ref=ref or None)
    except AuthError as e:
        raise QueueError(f"review: #{pr}'s `{verdict}` at {head[:12]} isn't written: {e}") from None
    said, lost = _label_write_through(store, target)
    print(f"research queue: #{pr} reviewed `{verdict}` at {head[:12]} by {by} ({target}); {said}")
    return 1 if lost else 0


def delegate(store, agent: str, lane: str, days: int, holders: dict[str, str] | None,
             mint: Callable[[str], str] | None = None, now: Callable[[], float] = time.time) -> int:
    """`research queue delegate AGENT --by LANE [--days N]`: as LANE's holder, let the Cursor agent AGENT (a worker) authenticate
    ready marks and land verdicts as LANE for ``days`` (at most `labelauth.DELEGATION_MAX_DAYS`): a `label_delegation` label on
    the agent's delegation target, pushed to the store's remote, which `labelauth.Signer` copies into each companion it writes.
    Refused, nothing written, when this agent isn't LANE's holder in ``holders`` (the registry: its delegation would never count)."""
    from . import labelauth
    from .store.cli import _label_write_through

    if not AGENT_RE.fullmatch(agent):
        raise QueueError(f"delegate: {agent!r} isn't a Cursor agent id (bc-, then a UUID)")
    if not 1 <= days <= labelauth.DELEGATION_MAX_DAYS:
        raise QueueError(f"delegate: --days {days}: 1 to {labelauth.DELEGATION_MAX_DAYS}")
    until = time.strftime(labelauth.TS_FORMAT, time.gmtime(now() + days * 86400))
    try:
        value = labelauth.delegation(lane, agent, until, mint or labelauth.cursor_mint)
        me = labelauth.unverified_claims(value["token"]).get("cloud_agent_id")
    except labelauth.AuthError as e:
        raise QueueError(f"delegate: {agent} can't be delegated {lane}: {e}") from None
    if holders is None or not isinstance(me, str) or _who(holders.get(me, "")) != _who(lane):
        raise QueueError(f"delegate: this agent ({me}) doesn't hold {lane} in main's registry ({REGISTRY}), so its delegation "
                         f"would never count: {lane}'s holder runs it")
    target = labelauth.delegation_target(agent)
    store.add_label(target, vocab.LABEL_DELEGATION, value, lane)
    said, lost = _label_write_through(store, target)
    print(f"research queue: {agent} signs as {lane} until {until}, while this agent holds {lane} in main's registry "
          f"({target}); {said}")
    return 1 if lost else 0


def label_signer(store, repo: Path | None):
    """``store`` as the `labelauth.Signer` `ready`, `judge` and `review` write through: required, with the registry's holders, when
    drive.toml's `signed_verdicts` is on at `origin/main` in ``repo`` (the caller's checkout); otherwise it signs what it can."""
    from . import labelauth

    main, holders = "refs/remotes/origin/main", None
    if on := repo is not None and signed_at(repo, main):
        holders = holders_at(repo, main)
    return labelauth.Signer(store, labelauth.cursor_mint, required=on, holders=holders,
                            log=lambda s: print(f"research queue: {s}", file=sys.stderr))


def label_verifier(repo: Path, ref: str, keys: Path, log: Callable[[str], None] = print):
    """The `labelauth.Verifier` of drive.toml's `signed_verdicts`: the registry at ``ref`` in ``repo`` names who holds each handle,
    and the issuer's keys are fetched (kept at ``keys`` for an outage of the issuer).  QueueError when either can't be had: no
    mark could be judged, and judging every one absent would take every PR off `next`."""
    from . import labelauth

    holders = holders_at(repo, ref)
    try:
        return labelauth.Verifier(labelauth.issuer_keys(keys, log=log), holders)
    except labelauth.AuthError as e:
        raise QueueError(f"no ready mark or land verdict can be authenticated (drive.toml's `signed_verdicts`): {e}") from None


def review_owners(q: Queue, labels: StoreLabels, pr: int, head: str, owner: str | None,
                  opened: dict[int, PR] | None = None) -> set[str | None]:
    """Whose verdict on ``pr`` at ``head`` doesn't count (`unreviewed`): its owner (`PR.owner`, or its merge's `Owner`), that
    owner's Slack handle through the registry (`Queue.handle`), and who wrote its ready mark (`StoreLabels.ready_by`)."""
    p = (opened or {}).get(pr) or PR(pr, head, body=f"Lane {owner}")
    return {owner, q.handle(p), labels.ready_by(pr, head)}


def _move_main(q: Queue, sha: str, verb: str, passed: str, wait: Callable[[], str | None] = lambda: None) -> int:
    """Fast-forward `main` to ``sha``, a commit on the landing branch's first-parent chain (`checked`, `Queue.into`), beside
    drive's tick: in drive's clone as the tick left it, read only (the branches' tips from the remote, nothing fetched, and the
    push to the remote's URL, so no ref of the clone moves under the tick), with a plain push, never forced, which a `main` moved
    meanwhile refuses.  Nothing to do when `main` already holds ``sha``, or while ``wait`` says why not yet.  Drive's next tick
    takes the new `main` (`Driver._main_moved`)."""
    if q.into == "main":
        raise QueueError(f"{verb}: drive lands on `main` itself (drive.toml's `land_into`), so no `checked` leads it")
    if not (q.workdir / ".git").exists():
        raise QueueError(f"{verb}: drive has no clone at {q.workdir} yet: its first tick makes it")
    has = lambda c: not _git(q.workdir, "cat-file", "-e", f"{c}^{{commit}}", check=False).returncode  # noqa: E731
    holds = lambda a, b: not _git(q.workdir, "merge-base", "--is-ancestor", a, b, check=False).returncode  # noqa: E731

    def main_tip() -> str | None:
        out = q._remote(q.workdir, "ls-remote", q.remote, "refs/heads/main", ok=None).stdout.split()
        return out[0] if out else None

    tips = {ref: c for c, ref in (line.split() for line in q._remote(
        q.workdir, "ls-remote", q.remote, "refs/heads/main", f"refs/heads/{q.into}").stdout.splitlines())}
    main, tip = tips.get("refs/heads/main"), tips.get(f"refs/heads/{q.into}")
    if missing := [f"`{n}` {c[:12]}" for n, c in (("main", main), (q.into, tip)) if c and not has(c)]:
        raise QueueError(f"{verb}: {' and '.join(missing)} isn't in drive's clone: pushed outside drive, which takes it next tick")
    if not main or not tip or not on_first_parents(q.workdir, sha, tip):
        raise QueueError(f"{verb}: {sha[:12]} is not on `{q.into}`'s first-parent chain" + (f" (at {tip[:12]})" if tip else ""))
    if holds(sha, main):
        print(f"research queue: {verb}: `main` {main[:12]} holds {sha[:12]} already: nothing to move")
        return 0
    if not holds(main, sha):
        raise QueueError(f"{verb}: `main` {main[:12]}, pushed outside drive, isn't below {sha[:12]}: drive's next tick merges "
                         f"it into `{q.into}` (`_absorb`)")
    if why := wait():
        print(f"research queue: {verb}: `main` stays at {main[:12]}: {why}")
        return 0
    q._drive_only("main", verb, False)
    if (p := q._remote(q.workdir, "push", "-q", "--no-verify", q.remote, f"{sha}:refs/heads/main", ok=None)).returncode:
        if (now := main_tip()) and has(now) and holds(sha, now):
            print(f"research queue: {verb}: `main` moved to {now[:12]} meanwhile, which holds {sha[:12]}: nothing to move")
            return 0
        raise QueueError(f"{verb}: pushing `main` {main[:12]} to {sha[:12]} was refused: "
                         f"{(p.stderr or p.stdout).strip()[-200:]}")
    print(f"research queue: {verb}: `main` fast-forwarded from {main[:12]} to {sha[:12]}, {passed}; drive's next tick takes it")
    return 0


def baseline(q: Queue, sha: str, by: str, steps: str | None = None) -> int:
    """`research queue baseline SHA --by ci [--steps STEP,...]`: fast-forward `main` to SHA, a commit on `checked`'s first-parent
    chain whose after-landing steps are known green (every one by default; with --steps, those), at once (`_move_main`): it
    waits for no tick, takes no lock of drive's and writes nothing of drive's state, and drive's next tick takes `main` there as
    verified (`Driver._main_moved`), so the review landing lands on it.  Refused without `--by ci`, off drive
    (`/data/infra/drive.sh queue baseline SHA --by ci` on the control pod), for --steps that leave out a step on the merge gate
    at SHA (`drive.gating`: not circuit-check), and for a SHA off that chain or below `main`; nothing to do when `main` holds it."""
    from .drive import AFTER_LANDING, gating

    if _who(by) != "ci":
        raise QueueError(f"baseline: a green `main` is recorded only with --by ci, not --by {by}")
    green = [x for x in (steps or "").split(",") if x] or list(AFTER_LANDING)
    if unknown := [x for x in green if x not in AFTER_LANDING]:
        raise QueueError(f"baseline: --steps {', '.join(unknown)}: not after-landing steps ({', '.join(AFTER_LANDING)})")
    if not (q.workdir / ".git").exists() or not (
            full := _git(q.workdir, "rev-parse", "--verify", "-q", f"{sha}^{{commit}}", check=False).stdout.strip()):
        raise QueueError(f"baseline: {sha} is no commit drive's clone has")
    if left := [s for s in gating(q.workdir, full) if s not in green]:
        raise QueueError(f"baseline: --steps leaves out {', '.join(left)}, on the merge gate at {full[:12]}: `main` moves only to "
                         "a commit that passed every step on it")
    return _move_main(q, full, "baseline", f"green on every step on the merge gate by {by}")


def promote(q: Queue, states: Callable[[Path, list[str]], dict] | None = None,
            evidence: Callable[[Path, list[str]], list[Path]] | None = None) -> int:
    """`research queue promote`: fast-forward `main` to the tip of drive's after-landing run (`postland["run"]` in drive's
    state, read and never written) once each of its runs, every shard's twins, has ended and it passed every step on the merge
    gate there, as drive's tick judges it (`Driver.not_promotable`), the way `baseline` moves it (`_move_main`): `main` moves
    within minutes of the run's end, not at drive's next tick.  Nothing to do without such a run, while it runs, once it failed
    on the gate, or when `main` holds its tip; a loop on drive's host runs it (`/data/infra/drive.sh queue promote`)."""
    from .drive import Driver, evidence as run_evidence, run_states

    if not (run := ((q.load_state().get("drive") or {}).get("postland") or {}).get("run")):
        print("research queue: promote: no after-landing run in drive's state: nothing to move")
        return 0
    d = Driver(q, on="", states=states or run_states, evidence=evidence or run_evidence)
    return _move_main(q, run["tip"], "promote", f"whose after-landing run {' '.join(run.get('ids') or ())} passed every step on "
                      "the merge gate", lambda: d.not_promotable(run))


def accept_checked(q: Queue, sha: str, by: str) -> int:
    """`research queue accept-checked SHA --by ci`: record the landing branch (`checked`, `Queue.into`) at SHA, the tip it is at,
    among the tips drive takes (`drive["checked_tips"]`) though drive didn't write it, so drive's ticks, held on it
    (`Driver._unwritten`), go on from it.  Refused without `--by ci`, for a SHA that isn't the branch's tip, and off drive
    (`/data/infra/drive.sh queue accept-checked SHA --by ci` on the control pod)."""
    from .drive import CHECKED_TIPS

    if _who(by) != "ci":
        raise QueueError(f"accept-checked: a `{q.into}` drive didn't write is accepted only with --by ci, not --by {by}")
    q._clone()
    q.fetch()
    full = _git(q.workdir, "rev-parse", "--verify", "-q", f"{sha}^{{commit}}", check=False).stdout.strip()
    tip = _out(q.workdir, "rev-parse", q.base)
    if full != tip:
        raise QueueError(f"accept-checked: {sha} is not `{q.into}`'s tip, {tip[:12]}: only the commit it is at is accepted")
    state = q.load_state()
    tips = state.setdefault("drive", {}).setdefault("checked_tips", [])
    tips[:] = [*(t for t in tips if t != tip), tip][-CHECKED_TIPS:]
    q.save_state(state)
    print(f"research queue: `{q.into}` {tip[:12]} accepted by {by}: drive's next tick goes on from it")
    return 0


def drive_shard_add(q: Queue, tip: str, k: int, rid: str) -> int:
    """`research queue drive-shard-add TIP K RUN_ID`: count RUN_ID among shard K's twins (``groups``) of the after-landing run of
    `checked` at TIP (``ids`` too, so drive reads its state), a run launched by hand in place of a twin that waits or was lost: the
    shard passes once one of its twins does (`Driver._every_shard`).  Drive waits while any twin runs or waits, so cancel the
    twin it replaces after adding it (before, a shard whose every twin was stopped launches the whole run again).  Refused unless TIP (full, or 7+ hex digits of it) is that
    run's tip with its shards' ids known, for a K outside 1..N (an unsharded run is shard 1/1), for a RUN_ID it already
    has, and unless RUN_ID's own record says it is that run's shard: a `check.py` run of TIP's commit (`merge._ran_on`: its
    attempt's clean source, else its run record's), with `--shard K/N` (none for an unsharded run) and, when it names them,
    `--deferred` the run's steps, and for an auto plan's shard (`--shard auto:K/N`), that plan among its inputs, so a passing
    run of a mistyped id never verifies a commit nobody checked."""
    from . import merge
    from .drive import AUTO_PLAN, shard_spec

    pl = ((state := q.load_state()).get("drive") or {}).get("postland") or {}
    run = pl.get("run")
    if not run or len(tip) < 7 or not run["tip"].startswith(tip):
        raise QueueError(f"drive-shard-add: {tip} is not the tip of the after-landing run ("
                         f"{run['tip'][:12] if run else 'none runs'})")
    if run.get("shards") or "launch" in run:
        raise QueueError(f"drive-shard-add: the after-landing run of {run['tip'][:12]} is still launching: add to it once drive "
                         "has read its run ids")
    groups = run.get("groups") or [run["ids"]]
    if not 1 <= k <= len(groups):
        raise QueueError(f"drive-shard-add: shard {k} is not one of the run's {len(groups)} (1..{len(groups)})")
    if rid in run["ids"]:
        raise QueueError(f"drive-shard-add: {rid} is already a run of the after-landing run of {run['tip'][:12]}")
    if (got := merge.fetch_run(rid, [])) is None:
        raise QueueError(f"drive-shard-add: {rid}'s run record can't be read (`research status {rid}`): its commit and shard "
                         "are unknown")
    cmd = [str(x) for x in got[1].get("command") or ()]
    if not any(x.endswith("check/check.py") for x in cmd):
        raise QueueError(f"drive-shard-add: {rid} is no check.py run (its command: {' '.join(cmd)[:200] or 'unrecorded'})")
    if not merge._ran_on(q.store, rid, run["tip"], merge.fetch_run):
        raise QueueError(f"drive-shard-add: {rid} checked {(got[1].get('tree_sha') or 'an unrecorded commit')[:12]}, not the "
                         f"after-landing run's {run['tip'][:12]}")
    want = shard_spec(run, k) if "groups" in run else None
    if (has := cmd[cmd.index("--shard") + 1] if "--shard" in cmd[:-1] else None) != want:
        raise QueueError(f"drive-shard-add: {rid} is {f'shard {has}' if has else 'no shard'}, not "
                         f"{f'shard {want}' if want else 'unsharded'} of the after-landing run of {run['tip'][:12]}")
    if run.get("plan"):
        try:
            d = merge.fetch_run(rid, [f"inputs/{AUTO_PLAN}"])[0]
            plan = str(json.loads((d / "inputs" / AUTO_PLAN).read_text())["digest"])
        except (TypeError, OSError, ValueError, KeyError) as e:
            plan = f"unreadable ({type(e).__name__})"
        if plan != run["plan"]:
            raise QueueError(f"drive-shard-add: {rid}'s auto plan ({AUTO_PLAN} among its inputs) is {plan[:12]}, not the "
                             f"after-landing run's {run['plan'][:12]}")
    steps = pl.get("steps") or []
    if "--deferred" in cmd[:-1] and (named := cmd[cmd.index("--deferred") + 1].split(",")) != steps:
        raise QueueError(f"drive-shard-add: {rid} ran --deferred {','.join(named)}, not the after-landing run's "
                         f"{','.join(steps)}")
    if "groups" in run:
        run["groups"][k - 1].append(rid)
    run["ids"].append(rid)
    q.save_state(state)
    print(f"research queue: {rid} counts as a twin of shard {k}/{len(groups)} of the after-landing run of `{q.into}` "
          f"{run['tip'][:12]}")
    return 0


def broker_gh(top: Path) -> str:
    """The gh to open PRs with: the checkout's broker wrapper (`<git common dir>/verity-auth/bin/gh`, which trades the agent's
    identity for an installation token) when it has one, else gh on PATH."""
    common = _git(top, "rev-parse", "--path-format=absolute", "--git-common-dir", check=False).stdout.strip()
    wrapper = Path(common) / "verity-auth" / "bin" / "gh" if common else None
    return str(wrapper) if wrapper and wrapper.is_file() else (shutil.which("gh") or "gh")


def issue_line(spec: str) -> str:
    """The body line `open --issue SPEC` writes: `Refs KEY` for a Linear key (`ISSUE_KEY_RE`), `Issue: none: REASON` for
    `none:REASON`.  Anything else, or `none:` with no reason, is refused (`QueueError`)."""
    if spec.startswith("none:"):
        if not (reason := spec.removeprefix("none:").strip()):
            raise QueueError("open: --issue none:REASON needs the reason this PR has no Linear issue")
        return NO_ISSUE_LINE.format(reason=reason)
    if not ISSUE_KEY_RE.fullmatch(spec):
        raise QueueError(f"open: --issue {spec!r} is neither a Linear key (like ENG-123) nor none:REASON")
    return ISSUE_LINE.format(key=spec)


def with_issue(body: str, spec: str) -> str:
    """``body`` with `issue_line(spec)` as its last line, unless a line of it already is that line."""
    line = issue_line(spec)
    return body if line in body.splitlines() else f"{body.rstrip()}\n\n{line}\n"


def issue_app(key: str) -> tuple[linear.Linear, str] | Exception:
    """The Linear app and the id of issue ``key``, resolved before anything is sent; the failure instead when the app can't be
    asked (no credentials, Linear down), which `open` warns of (`with_goals`) and `link` reports (`link_issue`).  A key Linear
    has no issue for raises QueueError."""
    try:
        app = linear.Linear()
        return app, app.issue_id(key)
    except linear.NoIssue as e:
        raise QueueError(f"--issue {key}: {e}: give the PR's issue, or none:REASON; nothing sent") from None
    except (linear.LinearError, OSError) as e:
        return e


def link_issue(key: str, url: str, app: tuple[linear.Linear, str] | Exception | None = None) -> str:
    """Link the PR at ``url`` to the Linear issue ``key`` through the Linear app (`Linear.link_pr`): its `Refs KEY` line alone
    links nothing.  ``app`` is `issue_app(key)` when the caller resolved it already.  What it linked; a failure raises
    QueueError naming the command that links it later (`research queue link N --issue KEY`)."""
    got = app or issue_app(key)
    try:
        if isinstance(got, Exception):
            raise got
        return f"{url} {'linked' if got[0].link_pr(got[1], url) else 'was linked already'} to {key} through the Linear app"
    except (linear.LinearError, OSError) as e:
        n = (m := re.search(r"/pull/(\d+)$", url)) and m.group(1)
        raise QueueError(f"{url} is not linked to {key} in Linear ({e}); link it with "
                         f"`research queue link {n or 'N'} --issue {key}`") from None


def goal_statuses(cache: Path | None = None) -> dict[str, str]:
    """Each goal on the Goals page (`research.goals`), its id to its status, written to ``cache`` (`GOALS_CACHE`) at each read.
    When the page can't be read, the goals last written there, whatever their age, said in one line on stderr; with neither,
    QueueError naming the outage, to which each caller adds its escape."""
    from . import goals
    from .slack import SlackError

    path = (cache or GOALS_CACHE).expanduser()
    try:
        got = {str(g["id"]): str(g.get("status")) for g in goals.Route().call()}
    except (SlackError, OSError, ValueError, KeyError, TypeError) as e:
        try:
            kept = json.loads(path.read_text())
            got = {str(g["id"]): str(g["status"]) for g in kept["goals"]}
        except (OSError, ValueError, KeyError, TypeError):
            raise QueueError(f"the Goals page can't be read ({e}) and this machine has no copy of it ({path})") from None
        print(f"research queue: the Goals page can't be read ({e}): using its {len(got)} goals as this machine read them at "
              f"{kept.get('read') or 'an unrecorded time'} ({path})", file=sys.stderr)
        return got
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"read": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                   "goals": [{"id": k, "status": v} for k, v in sorted(got.items())]}, indent=1) + "\n")
        tmp.replace(path)
    except OSError as e:
        print(f"research queue: the Goals page's goals aren't cached at {path} ({e})", file=sys.stderr)
    return got


def live_goals(names: Iterable[str], status: dict[str, str]) -> list[str]:
    """Those of ``names`` that are a live goal's id: on the Goals page (``status``, `goal_statuses`) and not done; sorted."""
    return sorted({n for n in names if status.get(n, "done") != "done"})


def with_goals(body: str, key: str, app: tuple[linear.Linear, str] | Exception,
               read: Callable[[], dict[str, str]] | None = None) -> str:
    """``body``, `with_issue`'s for Linear issue ``key``, with a `Goal <id>` line after its `Refs KEY` line for each live goal
    (`live_goals`) a label of the issue names (its name is the goal's id), or as it is when it has `Goal` lines of its own.  A
    warning on stderr, never a refusal: when the issue's labels can't be read (``app`` is `issue_app`'s failure) or name no
    live goal, when ``read`` (`goal_statuses`) has no goals, and for each `Goal` line of the body that names another."""
    def warn(why: str) -> None:
        print(f"research queue open: warning: --issue {key}: {why}", file=sys.stderr)

    try:
        if isinstance(app, Exception):
            raise app
        labels = app[0].issue_labels(key)
    except (linear.LinearError, OSError) as e:
        warn(f"the Linear app can't read its labels ({e}): no `Goal` line written")
        return body
    try:
        status = (read or goal_statuses)()
    except QueueError as e:
        warn(f"{e}: no `Goal` line written")
        return body
    if not (live := live_goals(labels, status)):
        done = [n for n in labels if status.get(n) == "done"]
        warn(f"no label of it names a live goal of the Goals page (its labels: {', '.join(labels) or 'none'}"
             f"{'; done goals among them: ' + ', '.join(done) if done else ''}): label {key} with the goal it serves "
             f"(its id on the page, `research goals show`)")
    for g in GOAL_RE.findall(body):
        if g not in live:
            why = (f"a goal no label of {key} names" if g not in labels else "a done goal" if status.get(g) == "done" else
                   "no goal of the Goals page")
            warn(f"the body's `Goal {g}` names {why}" + (f": name one of {key}'s live goals ({', '.join(live)}), or drop the "
                                                        "line and `open` writes them" if live else ""))
    if GOAL_RE.search(body) or not live:
        return body
    lines = body.splitlines()
    at = lines.index(ISSUE_LINE.format(key=key)) + 1
    return "\n".join([*lines[:at], *(GOAL_LINE.format(id=g) for g in live), *lines[at:]]) + "\n"


def goal_gate(pr: PR, read: Callable[[], dict[str, str]] | None = None) -> str | None:
    """The goal gate of `ready` and `judge`, a warning that never refuses: None when ``pr``'s body names a live goal
    (`live_goals`) on a `Goal <id>` line, or no issue on purpose on an `Issue: none: REASON` line, which reads no page; else why
    not, with the fix, which they print to stderr before they go on."""
    if NO_ISSUE_RE.search(pr.body):
        return None
    fix = (f"add a `Goal <id>` line naming the live goal it serves (its id on the Goals page, `research goals show`), or an "
           f"`Issue: none: REASON` line, to its body (`research queue edit {pr.number} --body-file F`)")
    if not (named := GOAL_RE.findall(pr.body)):
        return f"its body names no goal: {fix}"
    try:
        status = (read or goal_statuses)()
    except QueueError as e:
        return f"{e}: its `Goal` lines go unchecked"
    if live_goals(named, status):
        return None
    dead = "; ".join(f"{g} is done" if g in status else f"the Goals page has no goal {g}" for g in dict.fromkeys(named))
    return f"its body names no live goal ({dead}): {fix}"


def named_issue(body: str, teams: Iterable[str] = ()) -> str | None:
    """The Linear issue ``body`` names: a key after a Linear magic word (`Refs ENG-123`, any team's), else the first bare key
    of one of ``teams`` (the registry's Linear team keys: `SHA-512` names no issue), else `none: REASON` from its `Issue:
    none:` line; None when it names none."""
    if m := ISSUE_LINK_RE.search(body):
        return m.group(1)
    if teams and (m := re.search(rf"\b(?:{'|'.join(map(re.escape, sorted(teams)))})-\d+\b", body)):
        return m.group(0)
    if m := NO_ISSUE_RE.search(body):
        return f"none: {m.group(1)}"
    return None


def _unnamed(prs: Iterable[PR], teams: Iterable[str]) -> str:
    """`status`'s section of the open PRs whose body names no Linear issue (`named_issue`), with the fix; empty when none."""
    prs = sorted(prs, key=lambda p: p.number)
    rows = [p for p in prs if named_issue(p.body, teams) is None]
    if not rows:
        return ""
    head = (f"\n\nno Linear issue named ({len(rows)} of {len(prs)} open PRs; add a `Refs KEY` or `Issue: none: REASON` line "
            "with `research queue edit N --body-file F`):")
    return head + "".join(f"\n  #{p.number:<5} {(p.owner or '-')[:24]:<24} {p.title}" for p in rows)


def open_pr(top: Path, repo: str, head: str, title: str, body: str, *, base: str = "main", sha: str | None = None,
            draft: bool = False, gh: str | None = None, run=subprocess.run) -> str:
    """Open a PR for ``head``, a branch already pushed to origin, and return its URL.  Refuses (`QueueError`) when the branch
    isn't on origin, when ``sha`` is given and the branch isn't at it (the owner's pinned head), and when an open PR already has
    that head, so a relayed request can never open a duplicate or a moved head."""
    found = _git(top, "ls-remote", "origin", f"refs/heads/{head}", check=False).stdout.split()
    if not found:
        raise QueueError(f"open: no branch {head!r} on origin: push it first")
    if sha and not found[0].startswith(sha):
        raise QueueError(f"open: origin's {head} is at {found[0][:12]}, not {sha}: the head moved; re-check it with its owner")
    call = _gh_call("open", top, gh, run)
    existing = json.loads(call("pr", "list", "--repo", repo, "--head", head, "--state", "open", "--json", "number,url") or "[]")
    if existing:
        raise QueueError(f"open: {head} already has PR #{existing[0]['number']} ({existing[0]['url']}); nothing opened")
    out = call("pr", "create", "--repo", repo, "--head", head, "--base", base, "--title", title, "--body", body,
               *(["--draft"] if draft else []))
    return out.strip().splitlines()[-1]


def _gh_call(verb: str, top: Path, gh: str | None, run) -> Callable[..., str]:
    """`gh ARGS` in ``top`` (the broker's gh by default, `broker_gh`), returning its stdout; a failure raises, naming ``verb``."""
    gh = gh or broker_gh(top)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

    def call(*args: str) -> str:
        r = run([gh, *args], cwd=top, capture_output=True, text=True, env=env, timeout=120)
        if r.returncode:
            raise QueueError(f"{verb}: gh {args[0]} {args[1]}: {(r.stderr or r.stdout).strip()[-300:]}")
        return r.stdout

    return call


#: the markup a Cursor agent's PR tool keeps in a body: it rewrites only the text between these markers, so any text above them
#: (the body `open` gave the PR) stays, and its footer, the agent's link whose id `PR.owner` reads, follows them
CURSOR_BODY = ("<!-- CURSOR_AGENT_PR_BODY_BEGIN -->", "<!-- CURSOR_AGENT_PR_BODY_END -->")
CURSOR_FOOTER_RE = re.compile(r'<div><a href="https://cursor\.com/agents/bc-[^"]*cursor_ref=pr_footer')


def edited_body(old: str, new: str) -> str:
    """``new`` as the whole of a PR body that was ``old``: none of ``old``'s text stays, only the platform's markup, which is
    kept deliberately.  ``new`` goes between Cursor's markers when ``old`` has them, so the tool's next update replaces it
    rather than leaving it above, and ``old``'s footer follows it (else a PR keeps the body it was opened with above every
    later one).  Markers or a footer ``new`` brings along are dropped."""
    begin, end = CURSOR_BODY
    foot = next((old[m.start():].strip() for m in reversed(list(CURSOR_FOOTER_RE.finditer(old)))
                 if end not in old or m.start() > old.rfind(end)), "")
    if m := CURSOR_FOOTER_RE.search(new):
        new = new[:m.start()]
    text = new.replace(begin, "").replace(end, "").strip("\n")
    if begin in old and end in old:
        text = f"{begin}\n{text}\n{end}"
    return text + (f"\n\n{foot}" if foot else "") + "\n"


def edit_pr(top: Path, repo: str, number: int, body: str, *, title: str | None = None, dry_run: bool = False,
            gh: str | None = None, run=subprocess.run) -> tuple[str, str]:
    """Replace PR ``number``'s body with ``body`` outright (`edited_body`), and its title with ``title`` when given, and return
    its URL and the body it now has; with ``dry_run``, change nothing.  Refused (`QueueError`) when the PR's body names its
    owner on a `Lane NAME` line and the new one names another owner or none: an edit never moves a PR to another lane or out
    of the queue."""
    call = _gh_call("edit", top, gh, run)
    cur = json.loads(call("pr", "view", str(number), "--repo", repo, "--json", "body,url"))
    text = edited_body(cur.get("body") or "", body)
    now = PR(number, "", body=text).owner
    if (was := LANE_RE.search(cur.get("body") or "")) and _who(now or "") != _who(was.group(1)):
        raise QueueError(f"edit: #{number}'s new body names owner {now or 'none'}, and the PR's is {was.group(1)}: keep its "
                         f"`Lane {was.group(1)}` line; nothing edited")
    if not dry_run:
        call("pr", "edit", str(number), "--repo", repo, *(("--title", title) if title else ()), "--body", text)
    return cur.get("url") or f"#{number}", text


def _stats(a: argparse.Namespace) -> int:
    """`research queue stats` (`queue_stats`): each landed PR's head from `main`'s merges onto `next`, its ready mark from the
    store, the landing check's records fetched with `research status --also`."""
    from . import queue_stats
    from .merge import fetch_run
    from .store.local import LocalStore

    top = Path(_out(Path.cwd(), "rev-parse", "--show-toplevel"))
    _git(top, *STALL, "fetch", "-q", "origin", "main", check=False)
    heads = {}
    for body in _out(top, "log", "--first-parent", "--format=%B%x1e", "-n", "5000", "origin/main").split("\x1e"):
        t = dict(TRAILER_RE.findall(body))
        if "PR" in t and "Head" in t:
            heads.setdefault(int(t["PR"].lstrip("#")), t["Head"])
    labels = StoreLabels(LocalStore(a.store), log=lambda s: None)

    def marked(pr: int):
        head = heads.get(pr)
        if head is None:
            return None
        labels.ready(pr, head)
        at = labels.marked_at(pr, head)
        return queue_stats.ts(at) if at else None

    def run_dir(rid: str) -> Path:
        # a landing's check has ended (--refresh); one name a fetch, since slot.json beside steps.jsonl fetched neither
        got = fetch_run(rid, ["steps.jsonl"], top, refresh=True)
        if got and not (got[0] / "slot.json").is_file():
            fetch_run(rid, ["slot.json"], top, refresh=True)
        return got[0] if got else Path("/nonexistent")

    return queue_stats.main(a.log, a.hours, a.until, marked, run_dir, a.json)


def _checked_heads(a: argparse.Namespace) -> int:
    from .merge import Refused, checked_heads
    from .store import bucket_index as BI
    from .store.local import LocalStore

    idx = BI.reader(LocalStore(a.store).root)
    if idx is None:
        print(f"research queue checked-heads: no store index: set ${BI.READER_ENV} (store_private's read-only URL) or build one "
              "here with `research data index`", file=sys.stderr)
        return 1
    try:
        heads = checked_heads(Path(_out(Path.cwd(), "rev-parse", "--show-toplevel")), a.tip, idx)
    except (QueueError, Refused, RuntimeError) as e:
        print(f"research queue checked-heads: {e}", file=sys.stderr)
        return 1
    if a.out is not None:
        a.out.write_text("".join(c + "\n" for c, _ in reversed(heads)))
        print(f"research queue checked-heads: wrote {len(heads)} checked train heads (newest first) to {a.out}, an "
              "optional verifier --main file (first-parent by default)", file=sys.stderr)
        return 0
    print(json.dumps([{"commit": c, "run": r} for c, r in heads], indent=1) if a.json else "\n".join(c for c, _ in heads))
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["vqueue"]:                  # the verified queue's loop, with options of its own (`vqueue_loop.main`)
        from .vqueue_loop import main as vqueue_main

        return vqueue_main(argv[1:])
    ap = argparse.ArgumentParser(prog="research queue", description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["status", "sync", "eject", "ready", "judge", "drive", "open", "edit", "link", "stats",
                                            "checked-heads", "review", "baseline", "promote", "accept-checked",
                                            "drive-shard-add", "delegate",
                                            "shadow-report", "vqueue"])
    ap.add_argument("pr", nargs="?", help="eject, ready, review, edit, link: the PR number; judge: the quick tier's run id; baseline: the "
                    "commit of `checked` ci knows is green, which `main` moves to; "
                    "accept-checked: `checked`'s tip, which drive didn't write; drive-shard-add: the after-landing run's tip; "
                    "delegate: the Cursor agent id to delegate --by's handle to")
    ap.add_argument("shard_run", nargs="*", metavar="K RUN_ID", help="drive-shard-add: the shard and the run to count in it")
    ap.add_argument("--reason", help="eject: why, for the owner's notice")
    ap.add_argument("--steps", help="baseline: the after-landing steps green at SHA (default every one), which must cover every "
                    "step on the merge gate there")
    ap.add_argument("--by", default=os.environ.get("RESEARCH_LANE"),
                    help="eject, ready, judge: who ejects it, the handle its notice is posted as (default merge-queue, which no "
                         "one holds, so the notice fails) or says it's ready (no default: the owner's lane), "
                         "default $RESEARCH_LANE")
    ap.add_argument("--shadow", action="store_true", help=f"build `{SHADOW_BRANCH}`, never push `main`, send no notices")
    ap.add_argument("--dry-run", action="store_true", help="sync/drive: decide and report, push and launch nothing, tell no "
                                                           "one; open, edit: say what it would send, and send nothing")
    ap.add_argument("--workdir", type=Path, default=None, help="the queue's own clone (default: ~/.research/queue/NAME for --repo "
                                                              "OWNER/NAME, else ~/.research/queue/verity)")
    ap.add_argument("--remote", default=None, help="the repository to clone (default: https://github.com/OWNER/NAME.git for --repo "
                                                   "OWNER/NAME, else this checkout's origin); one naming another repository on "
                                                   "GitHub than --repo is refused")
    ap.add_argument("--repo", default=None, help="OWNER/NAME on GitHub (default: from the remote's URL)")
    ap.add_argument("--rules", type=Path, default=None, help=f"the admission rules (default: {RULES} on `main`)")
    ap.add_argument("--store", default=None, help="evidence store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--json", action="store_true", help="status: as JSON; ready: one JSON object on stdout (pr, head, base, exit, "
                    "warning), the rest on stderr")
    ap.add_argument("--on", metavar="MACHINE", help="ready: run the quick tier as a `research run` there, in a check slot (`auto` or "
                                                    "M1,M2: the one with a free check slot, as `check.py --on` picks), waiting for it "
                                                    f"{waitcap.WAIT_CAP_S} s at most (then exit {waitcap.STILL_GOING}, the run going "
                                                    "on: `judge RUN` marks the PR once it ends); judge: look "
                                                    "for a run another VM launched there only")
    ap.add_argument("--local", action="store_true", help="ready: run the quick tier here")
    ap.add_argument("--mark", action="store_true", help="ready: mark it with no quick tier (the after-landing run tests it)")
    ap.add_argument("--again", action="store_true", help="ready: mark a head that was ejected (only when the ejection was wrong)")
    ap.add_argument("--preflight-root", type=Path, default=None, metavar="DIR",
                    help="ready, drive: where merged trees go, ready's preflight's and every other (default: the temporary "
                         "directory); a tree must share a filesystem with uv's cache for its sync to hardlink (drive on the "
                         "control pod, whose TMPDIR is a network volume, passes the container disk)")
    ap.add_argument("--no-preflight", action="store_true",
                    help="ready: skip the preflight (the lander's): the head merged onto a fresh `next` must merge cleanly, contain "
                         "no open PR that isn't admitted, and pass `pins.py --stale`, the invariant tests (module strings, pins, "
                         "doc paths, pythonpath) and `pytest --collect-only` of the suites its change touches, in a temporary "
                         f"worktree, in about {PREFLIGHT_CHECKS_S:g} s, and build the Lean modules that are or import what it "
                         f"changes (at most {PREFLIGHT_LEAN_MODULES}, in {PREFLIGHT_LEAN_S:g} s, on a warm Lean tree: here on a "
                         f"node, else on the first of {', '.join(PREFLIGHT_LEAN_ON)} that starts the build, whatever the tier; "
                         "one no node can build refuses the mark, one not "
                         "built in time is a skipped check); the records gate (which warns of a changed Lean lock with no records "
                         "run or Lean check, of any commit, that passed its package on the head's Lean inputs, by their key "
                         "`tools/verity/lean/lean_changed.py --inputs-key`) still runs")
    ap.add_argument("--regenerate-locks", action=argparse.BooleanOptionalAction,
                    help="drive: after a landing that changes Lean, a lock or the spec, run lean-audit alone on `checked`'s tip "
                         "(the lock run) and commit the lock records it regenerated onto `checked` (`Regenerated:`), which "
                         "`research merge`'s gate accepts on that run; `main` moves only to a commit with them")
    ap.add_argument("--marked-after", metavar="YYYY-MM-DDTHH:MM:SSZ", help="status/sync/drive: admit only ready marks written at "
                                                                           "or after this time (default: `marked_after` in the rules file)")
    ap.add_argument("--first", help=argparse.SUPPRESS)
    ap.add_argument("--priority-file", type=Path, metavar="PATH",
                    help="sync/drive: drive's priority list, the landing order: a file of PR numbers separated by spaces or "
                         "commas, first to land first (a number repeated keeps its first place), read again at each sync (none: "
                         "an empty list; one it can't read: an error logged once, and an empty list); its ready PRs go at the "
                         "bottom of `next` in its order, so an edit reorders `next` from the first PR whose place changed; "
                         "these PRs and those they contain go first of "
                         "what the sync merges anew")
    ap.add_argument("--after-landing-no-cache", action=argparse.BooleanOptionalAction,
                    help="drive: run the after-landing check reusing no pass but circuit-check's per-target ones "
                         "(check.py --no-cache --circuit-cache)")
    for old in ("--max-running", "--train-size"):
        ap.add_argument(old, type=int, metavar="N", help="drive: accepted and ignored (train landing was deleted)")
    ap.add_argument("--land", action="store_true", help="review N: the verdict that lands it")
    ap.add_argument("--hold", metavar="REASON", help="review N: the verdict that keeps it, and everything above it, on `next`")
    ap.add_argument("--ref", help="review N --land: the note or reason the verdict rests on")
    ap.add_argument("--config", metavar="PATH",
                    help=f"drive: a flags file (`DriveFlags`) at `main` in the queue's clone, accepted only while every key it "
                         f"sets has the value of {DRIVE_FLAGS} at `main`, drive's one source of flags; else the tick refuses, "
                         "naming each key")
    ap.add_argument("--config-from-tree", action="store_true", help="drive: read --config's PATH from this file system instead")
    ap.add_argument("--head", help="open: the pushed branch to open a PR for")
    ap.add_argument("--sha", help="open: refuse unless origin's branch is at this commit (the owner's pinned head); ready: mark only "
                                  "this head, refusing when GitHub still shows another after a few reads (push && ready, safely); "
                                  "eject: only while this is the PR's head, so an eject that waits for drive's tick never hits a "
                                  "head pushed meanwhile")
    ap.add_argument("--title", help="open: the PR's title; edit: its new title")
    ap.add_argument("--body-file", type=Path, help="open: the PR's body, verbatim but for --issue's last line (its first line "
                                                   "`Lane NAME` names the owner); "
                                                   "edit: the PR's whole new body, in place of all its text")
    ap.add_argument("--issue", metavar="KEY|none:REASON",
                    help="open (required): the PR's Linear issue, a key like ENG-123, written into the body as a `Refs KEY` line "
                         "and linked to the PR through the Linear app (Linear's GitHub integration is off for this "
                         "repository, so the line alone links nothing), with a `Goal <id>` line after it for each live goal "
                         "of the Goals page a label of the issue names (a warning on stderr when none does); or none:REASON, written as an "
                         "`Issue: none: REASON` line, for a PR with no issue; link (required): the KEY to link the open PR to")
    ap.add_argument("--base", default="main", help="open: the base branch (default main)")
    ap.add_argument("--tip", default="origin/main", help="checked-heads: whose first-parent line (default origin/main)")
    ap.add_argument("--out", type=Path, metavar="FILE", help="checked-heads: write the checked train heads newest first "
                    "to FILE (an optional source for the verifier's --main file) instead of listing them on stdout")
    ap.add_argument("--draft", action="store_true", help="open: as a draft (default: ready for review)")
    ap.add_argument("--log", type=Path, help="stats: drive's log (its stdout, every tick's)")
    ap.add_argument("--tips", type=int, default=10, metavar="N", help="shadow-report: the newest N `checked` tips drive compared "
                    "with the executor's verdicts (default 10)")
    ap.add_argument("--hours", type=float, default=24.0, help="stats: the window's length (default 24)")
    ap.add_argument("--days", type=int, default=7, help="delegate: how long the delegation lasts (default 7, at most 14)")
    ap.add_argument("--until", metavar="YYYY-MM-DDTHH:MM:SSZ", help="stats: the window's end (default now)")
    a = ap.parse_args(argv)
    if a.command == "vqueue":
        ap.error("vqueue comes first, its options after it: research queue vqueue --help")
    if a.first is not None:
        ap.error(FIRST_RETIRED)
    global MERGED_ROOT
    MERGED_ROOT = a.preflight_root or MERGED_ROOT
    if a.command == "stats":
        if a.log is None:
            ap.error("stats needs --log FILE (drive's log)")
        return _stats(a)
    if a.command == "checked-heads":
        return _checked_heads(a)
    if a.command == "shadow-report":
        from .executor import shadow_records, shadow_report
        from .store.local import LocalStore

        text, _ = shadow_report(shadow_records(LocalStore(a.store), log=lambda s: print(s, file=sys.stderr)), max(a.tips, 1))
        print(text)
        return 0
    if a.issue is not None and a.command not in ("open", "link"):
        ap.error("--issue is for open and link only")
    if a.command == "link":
        if not (a.pr and a.pr.isdigit() and a.issue and ISSUE_KEY_RE.fullmatch(a.issue)) or a.dry_run:
            ap.error("link needs the PR number and --issue KEY (its Linear issue, like ENG-123), and has no --dry-run")
        try:
            repo = remote_and_repo(a.remote, a.repo, Path.cwd())[1]
            print(f"research queue link: {link_issue(a.issue, f'https://github.com/{repo}/pull/{a.pr}')}")
            return 0
        except (QueueError, OSError) as e:
            print(f"research queue: {e}", file=sys.stderr)
            return 1
    if a.command == "open":
        if not (a.head and a.title and a.body_file and a.issue):
            ap.error("open needs --head BRANCH, --title T, --body-file F and --issue KEY (the PR's Linear issue, like ENG-123, "
                     "or none:REASON)")
        try:
            body = with_issue(a.body_file.read_text(), a.issue)
            top = Path(_out(Path.cwd(), "rev-parse", "--show-toplevel"))
            repo = remote_and_repo(a.remote, a.repo, top)[1]
            app = issue_app(a.issue) if ISSUE_KEY_RE.fullmatch(a.issue) else None
            if app is not None:
                body = with_goals(body, a.issue, app)
            if a.dry_run:                                   # --dry-run never opens a real PR
                link = f"; would link it to {a.issue} through the Linear app" if app else ""
                print(f"research queue open: would open {a.head}{'@' + a.sha[:12] if a.sha else ''} onto {a.base} in {repo}"
                      f"{' as a draft' if a.draft else ''}: {a.title!r} ({len(body)} bytes of body, ending "
                      f"{body.rstrip().splitlines()[-1]!r}{link}; dry run)")
                return 0
            url = open_pr(top, repo, a.head, a.title, body, base=a.base, sha=a.sha, draft=a.draft)
        except (QueueError, OSError) as e:
            print(f"research queue: {e}", file=sys.stderr)
            return 1
        print(url)
        if app:
            try:
                print(f"research queue open: {link_issue(a.issue, url, app)}")
            except QueueError as e:                         # the PR is open: a link that fails doesn't fail `open`
                print(f"research queue: {e}", file=sys.stderr)
        return 0
    if a.command == "edit":
        if not (a.pr and a.pr.isdigit() and a.body_file):
            ap.error("edit needs the PR number and --body-file F")
        try:
            top = Path(_out(Path.cwd(), "rev-parse", "--show-toplevel"))
            repo = remote_and_repo(a.remote, a.repo, top)[1]
            url, body = edit_pr(top, repo, int(a.pr), a.body_file.read_text(), title=a.title, dry_run=a.dry_run)
            titled = f" and its title with {a.title!r}" if a.title else ""
            if a.dry_run:
                print(f"research queue edit: would replace {url}'s body{titled} with ({len(body)} bytes; dry run):\n{body}",
                      end="")
            else:
                print(f"research queue edit: {url}'s body replaced ({len(body)} bytes){titled}")
            return 0
        except (QueueError, OSError) as e:
            print(f"research queue: {e}", file=sys.stderr)
            return 1
    if a.priority_file and a.command not in ("sync", "drive"):
        ap.error("--priority-file is for sync and drive only")
    if a.marked_after and not MARK_TIME_RE.fullmatch(a.marked_after):
        ap.error(f"--marked-after {a.marked_after!r}: a UTC time as YYYY-MM-DDTHH:MM:SSZ, as labels record it")
    if (a.config or a.config_from_tree) and a.command != "drive":
        ap.error("--config and --config-from-tree are for drive only")
    if a.config_from_tree and not a.config:
        ap.error("--config-from-tree needs --config PATH")
    if a.dry_run and a.command not in ("sync", "drive"):
        ap.error(f"--dry-run is for sync and drive only: `{a.command}` would act for real")
    if a.command == "ready" and bool(a.on) + a.local + a.mark != 1:
        ap.error("ready takes one of --on MACHINE (the quick tier as a run there, in a check slot), --local (here) and --mark (no tier)")
    if a.no_preflight and a.command != "ready":
        ap.error("--no-preflight is for ready only")
    if a.command == "judge" and not a.pr:
        ap.error("judge needs the quick tier's run id")
    if a.command in ("ready", "judge") and not a.by:
        ap.error(f"{a.command} writes the owner's ready mark: pass --by <lane> (or set $RESEARCH_LANE)")
    if (a.land or a.hold is not None or a.ref) and not (a.command == "review" and a.pr):
        ap.error("--land, --hold and --ref are for review N only")
    if a.command == "review" and a.pr and (a.land == (a.hold is not None) or not a.by):
        ap.error("review N records one verdict: --land [--ref NOTE] or --hold REASON, with --by <reviewer lane> (or $RESEARCH_LANE)")
    if a.command == "baseline" and not (a.pr and a.by):
        ap.error("baseline SHA moves `main` to a green commit of `checked`: pass the commit and --by ci")
    if a.command == "promote" and a.pr is not None:
        ap.error("promote takes no argument: it moves `main` to the tip of drive's after-landing run, once it passed")
    if a.command == "accept-checked" and not (a.pr and a.by):
        ap.error("accept-checked SHA records `checked`'s tip as one drive takes: pass the commit and --by ci")
    if a.shard_run and a.command != "drive-shard-add":
        ap.error(f"unrecognized arguments: {' '.join(a.shard_run)}")
    if a.command == "drive-shard-add" and not (a.pr and len(a.shard_run) == 2 and a.shard_run[0].isdigit()):
        ap.error("drive-shard-add TIP K RUN_ID: the after-landing run's tip, the shard's number and the run to count in it")
    if a.command == "delegate" and not (a.pr and a.by):
        ap.error("delegate AGENT lets a Cursor agent (a worker) sign as your handle: pass its agent id and --by <your lane>")
    a.by = a.by or "merge-queue"
    if a.command not in ("judge", "baseline", "accept-checked", "drive-shard-add", "delegate") and a.pr is not None:
        if not a.pr.isdigit():
            ap.error(f"{a.command}: {a.pr!r} is not a PR number")
        a.pr = int(a.pr)
    from .store.local import LocalStore

    try:
        remote, repo = remote_and_repo(a.remote, a.repo, Path.cwd())
        workdir = a.workdir or (WORKDIR.with_name(repo.split("/")[1].lower()) if a.repo else WORKDIR)
        store = LocalStore(a.store)
        here = _git(Path.cwd(), "rev-parse", "--show-toplevel", check=False).stdout.strip()
        main_ref, keys = "refs/remotes/origin/main", workdir.parent / f"{workdir.name}.issuer-keys.json"
        if a.command == "delegate":
            return delegate(store, a.pr, a.by, a.days, holders_at(Path(here), main_ref) if here else None)
        if a.command in ("judge", "ready"):
            store = label_signer(store, Path(here) if here else None)
        if a.command == "judge":
            return judge(GitHub(repo), store, a.pr, a.by, on=a.on, repo=Path(here) if here else None, url=remote,
                         records=RecordsGate(Path(here) if here else Path.cwd(), remote, store), goals=goal_gate)
        if a.command == "ready":
            top = Path(_out(Path.cwd(), "rev-parse", "--show-toplevel"))
            tier = MarkOnly() if a.mark else RemoteQuickTier(top, remote, a.on, by=a.by) if a.on else QuickTier(top, remote)
            pre = None if a.no_preflight else Preflight(top, remote, store, lean=LakeBuild(on=PREFLIGHT_LEAN_ON, by=a.by),
                                                             root=a.preflight_root)
            return ready(GitHub(repo), store, a.pr, a.by, tier, again=a.again, preflight=pre, sha=a.sha, as_json=a.json,
                         records=RecordsGate(top, remote, store), goals=goal_gate)
        labels = StoreLabels(store, log=lambda s: print(s, file=sys.stderr), since=a.marked_after)
        q = Queue(workdir, remote=remote, github=GitHub(repo), ready=labels.ready, store=store, shadow=a.shadow,
                  rules=Rules.parse(a.rules.read_text(), str(a.rules)) if a.rules else None,
                  auth=credential_args(Path.cwd(), remote),
                  before_cutoff=labels.before_cutoff, marked_by=labels.marked_by, marked_at=labels.marked_at,
                  prefetch=labels.prefetch)
        if a.command in ("baseline", "promote"):
            from .merge import DRIVE_ENV, on_drive

            if not on_drive():
                raise QueueError(f"{a.command}: `main` moves only from drive's clone, and this isn't drive (no {DRIVE_ENV}=1): run "
                                 f"`/data/infra/drive.sh queue {' '.join([a.command, *([a.pr, '--by', a.by] if a.pr else [])])}` there")
            # no lock: a tick holds the clone's for its whole run, and these only read the clone and drive's state
            return baseline(q, a.pr, a.by, a.steps) if a.command == "baseline" else promote(q)
        lock = _lock(workdir)
        q.unlock = lock.close
        # weakly: `q` holds `labels`' methods, so a strong reference back would keep `q`, and the workdir's lock it holds, past
        # `main`'s return
        own = weakref.WeakMethod(q.own_change)
        labels.change = lambda head: own()(head)
        if ignored := [f"--{k.replace('_', '-')}" for k in ("max_running", "train_size") if getattr(a, k) is not None]:
            q.log(f"research queue drive: {' and '.join(ignored)} ignored: train landing was deleted; drop them from drive's args")
        if a.command == "drive":
            q._clone()
            q.fetch()
            base, where = drive_flags(q.workdir, a.config, a.config_from_tree)
            flags = base.override(a, where)
            if not flags.on:
                raise QueueError(f"drive needs a machine for the checks it launches: set `on` in {DRIVE_FLAGS} at `main`, or pass "
                                 "--on MACHINE")
            q.verdict_first, q.restack_jobs = flags.verdict_first, flags.restack_jobs
            q.land_into = flags.land_into
            a.marked_after = flags.marked_after
            labels.since = a.marked_after
            q.log(f"research queue drive: flags from {where}: {flags.line()}")
            if flags.signed_verdicts:
                labels.verify = label_verifier(q.workdir, main_ref, keys, q.log)
        if not a.marked_after and a.command != "status":
            if a.rules is None:
                q._clone()
                q.fetch()
            labels.since = q.rules().marked_after
        if a.command == "status":
            from .stale import stale_prs

            def at_main() -> tuple[Callable[[list[PR]], dict[int, float]], int]:
                if not a.marked_after:
                    labels.since = q.rules().marked_after
                r = _git(q.workdir, "show", f"refs/remotes/origin/main:{DRIVE_FLAGS}", check=False)
                got = DriveFlags.parse(r.stdout, f"{DRIVE_FLAGS} at main") if r.returncode == 0 else DriveFlags()
                return (lambda prs: stale_prs(q, prs, q.now(), got.stale_days,
                                              lambda n, h: [lab.ts for lab in labels.reviews(n, h)], keep=got.stale_keep),
                        got.stale_close_days)

            entries = q.status(fetched=at_main, loop=True)
            teams = q.issue_teams()
            if a.json:
                print(json.dumps([{"pr": e.pr.number, "head": e.pr.head, "owner": e.pr.owner, "draft": e.pr.draft, "state": e.state,
                                   "why": e.why, "position": e.position, "commit": e.commit, "after": list(e.after),
                                   "issue": named_issue(e.pr.body, teams),
                                   "stale": e.idle_since is not None,
                                   "idle_since": None if e.idle_since is None else
                                   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(e.idle_since))}
                                  for e in sorted(entries.values(), key=lambda e: e.pr.number)], indent=1))
            else:
                print(_render(entries) + _unnamed(q.listed, teams))
            return 0
        if a.command == "review":
            if a.pr is None and here and signed_at(Path(here), main_ref):
                labels.verify = label_verifier(Path(here), main_ref, keys, lambda s: print(f"research queue: {s}", file=sys.stderr))
            return review(q, label_signer(store, Path(here) if here else None) if a.pr else store, labels, a.pr,
                          "land" if a.land else "hold" if a.pr else None, a.hold or a.ref, a.by, sha=a.sha, as_json=a.json)
        from .merge import DRIVE_ENV, off_drive, on_drive

        if a.command == "accept-checked":
            if not on_drive():
                raise QueueError(f"accept-checked: drive's state is the control pod's, and this isn't drive (no {DRIVE_ENV}=1): "
                                 f"run `/data/infra/drive.sh queue accept-checked {a.pr} --by {a.by}` there")
            return accept_checked(q, a.pr, a.by)
        if a.command == "drive-shard-add":
            if not on_drive():
                raise QueueError(f"drive-shard-add: drive's state is the control pod's, and this isn't drive (no {DRIVE_ENV}=1): "
                                 f"run `/data/infra/drive.sh queue drive-shard-add {a.pr} {' '.join(a.shard_run)}` there")
            return drive_shard_add(q, a.pr, int(a.shard_run[0]), a.shard_run[1])

        if a.command == "sync" and not (a.dry_run or on_drive()):
            raise QueueError(off_drive(q.branch, "queue sync"))
        if a.command == "drive":
            q._drive_only("main", "once", a.dry_run)          # its landings push `main`, its syncs `next`
        if a.command in ("sync", "drive"):
            q.priority_file = a.priority_file
        if a.command == "sync":
            q.sync(dry_run=a.dry_run)
            return 0
        if a.command == "drive":
            from .drive import Driver

            drive_names(q, flags, push=not (a.dry_run or a.shadow))
            Driver(q, on=flags.on, regenerate_locks=flags.regenerate_locks, after_landing_no_cache=flags.after_landing_no_cache,
                   reviews=labels.reviews, ready_by=labels.ready_by,
                   stale_days=flags.stale_days, stale_close_days=flags.stale_close_days,
                   stale_keep=flags.stale_keep, close_unlandable=flags.close_unlandable,
                   after_landing_shards=flags.after_landing_shards,
                   lowrisk_lane=flags.lowrisk_lane, lowrisk_per_tick=flags.lowrisk_per_tick,
                   lowrisk_excludes=flags.lowrisk_excludes, executor=flags.executor,
                   executor_enqueue=flags.executor_enqueue, warn_steps=flags.warn_steps,
                   verdict_carry=flags.verdict_carry, bottom_ring=flags.bottom_ring,
                   bottom_wait_min=flags.bottom_wait_min).tick(dry_run=a.dry_run)
            return 0
        if a.pr is None or not a.reason:
            ap.error("eject needs a PR number and --reason")
        # no sync here, on drive or off: drive's tick rebuilds `next` with its priority list and its flags
        told = q.tell_ejected(q.eject(a.pr, a.reason, a.by, sha=a.sha), a.reason, a.by)
        print(f"research queue: #{a.pr}'s mark is withdrawn; drive's next tick rebuilds `{q.branch}` without it (only drive's "
              "tick pushes it)")
        return told
    except QueueError as e:
        print(f"research queue: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
