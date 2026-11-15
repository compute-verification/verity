# Platform (draft): the team's rules and skills, to leave verity

A draft, not a rule in force. It holds the text that leaves verity's `AGENTS.md` and skills when the team's machinery
moves to the platform repository (compute-verification/infra, to be renamed), and how a verity checkout keeps reading it
after. Until that move lands, each rule here stands where verity states it today. Text taken from verity is quoted word
for word; a section says where it came from.

## What moves

- Code, as `tests/tree.toml` plans it: `infra/`, `tools/research/` and `tools/agent-guard/`. `tools/move/` stays, its
  maps being verity's records.
- Skills: `team`, `coordinating`, `using-slack`, `asking-daniel`, `writing-goals`, `team-shapes`, `process-design`,
  `smell-check`, `writing-runs`, and `friction` but for its protocol and claim rulings, which stay in verity word for
  word.
- Data: `.agents/skills/using-slack/registry.json` and `slack-app.json`, each to a data path of its own first, since
  tools and tests outside the skill read them.
- The lane contract and the parts of the testing, lean-proofs and kernel-engineering skills that run on the team's
  machines (below).

## How a verity checkout reads it after

Platform publishes one Cursor plugin from a team marketplace (`.cursor-plugin/marketplace.json` at its root), installed
as Required with auto-refresh. The plugin holds the skills above, one always-applied rule with the sections below, and
the agent-guard hooks. A cloud agent keeps the plugin version it started with. Fallback, where a plugin doesn't load:
the environment installs the same files at VM start, as it installs `/opt/agent-guard/` today. To show before the
switch: a throwaway plugin loading in a cloud agent started from the API and from an automation.

## The team's rule (always applied)

### The team's own machinery (from verity `AGENTS.md`, Map and commands)

- The team's own tooling and machines, which nothing in `verity/` reads: `tools/research/` (runs, the evidence store,
  messaging, pods), `tools/agent-guard/`, `tools/move/` and `infra/` (the team skill). They stay here, and this
  repository is public whole; new infra goes here too. The `platform` repository (compute-verification/infra) is where
  the team's development, coordination and CI harness live: the website, the executor (its `infra/`) and its `.cursor`.

The team's agents run these remotely (the team skill).

### Landing (from verity `AGENTS.md`, Checking and merging)

A part's verdict is reused while a hash of its declared inputs, the base image and its runner is unchanged, so a
docs-only change reruns nothing; a nightly uncached check must agree, and a disagreement is fixed in the check. Every commit of
`main` is checked whole, before it lands or, for a plain candidate under `--gate verdict`, right after: the verified queue's
loop (`research queue vqueue`, ci's) merges the marked PRs that have a land
verdict, are low-risk or are certified moves onto `main` as a candidate, checks it (by default graph2's tasks, with lean-agreement and
ratchet-gate run beside them; under `--gate check`, every job in shards), and fast-forwards `main` only to a candidate with
no new failure: each gating task passed, failed only where an unexpired allowance of policy.toml at its base covers, or
fails on its base too. Under `--gate verdict` a plain candidate lands on its members' land verdicts and the quick tier,
`main`'s whole check follows each landing, and a red check reverts the one culprit or, with no culprit, sends plain
candidates back to the whole check until `main` is green; a candidate is plain when each member has a land verdict of its
own, is no move and touches nothing drive.toml's `verdict_excludes` names, nor any entry of its `lowrisk_excludes` that
`verdict_plain` doesn't list (with either of the first two empty, none is plain), so every other candidate (Lean, a lock, a move, a verifier or device, the soundness path, the landing code
itself) keeps the whole check before landing. A failure the candidate's base shares blames none of its PRs;
any other's suspects are the PRs whose change touches what it names (its files, the files declaring a Lean declaration it
names), every PR when none does: a lone suspect is ejected, several are bisected, and in the same tick the candidate and
those above it are rebuilt without them. A candidate holding a Lean, lock or spec change runs lean-audit alone first (its records run) and is checked on the
commit of the lock records it regenerated, so `main` moves only to regenerated records. What drive.toml lets warn (its
`warn_gates` and `warn_steps`), a step off the gate and a lock's bookkeeping hold nothing, and the owner fixes it in their
next PR: a warn gate's failure is printed at the ready mark, a red off the gate that its base's run doesn't share is rung to
who acts on it under the loop's `--gate check` or `verdict` (to no one under graph2), and a lock's bookkeeping is told to `lean` and each
owner a warning names. How a PR lands, from its ready mark to `main`, and the loop's rule for each case: the team skill's
Landing.

An owner closes its own PR once `main` holds its change or another PR carries it, with one line naming the commit or PR
that holds it. Nobody is asked first, and a superseded PR is not left open as a record: the closed PR is the record.

### Working principles for the team (from verity `AGENTS.md`, Working principles)

- **Something broken becomes a benchmark** (the team's part). Several workers then drive the numbers down, each with its
  own approach, branch and `research run` compute. Each measured run carries a label with its numbers, and the best one
  that passes the guard lands.
- **A survey is answered at once.** A question sent to several people for their answers (a survey, a what-if, a price)
  asks each to answer on reading it, with what they know now, and sets no later deadline. An answer that lacks a figure
  says "not known" and the hour it will be known. Each answer is passed on as it arrives, not held for the rest.

### Where the team's writing lives (from verity `AGENTS.md`, Where writing goes)

- **Where those live:** results in the evidence store, why a line changed in git, and open work, decisions and reports
  outside the repository (the team skill).

### Skills (from verity `AGENTS.md`, Skills)

The team's own agents start with `team` (machines, landing, lanes, and where work, notes and evidence live), then
`coordinating`, `using-slack`, `asking-daniel`, `writing-goals` (your goals in the plan Daniel reads) and `team-shapes`
(which team a piece of work gets, each shape with what its uses measured, and how any agent tries a different practice).
`smell-check` (before you call something served, end to end, proved or credited), `writing-runs` (a job for `research
run`), `friction` (when something is in your way, and the standing rulings) and `process-design`.

## Running verity's tests on the team's machines (from verity's testing skill, Run tests)

Suites run remotely through `research run --on auto` (AGENTS.md "Tests"). Its launch says where the job went, why, and
when it should start (`research inspect RUN` says it again on the VM that launched it), and warns when that is over 10
minutes away (`research/auto.py`). A machine that can start no run takes no job: the launch passes over it for the next,
saying which and why. Give it `--cwd source`, so the job runs in the shipped tree rather than its run directory, and
`uv sync` that tree first: `suites.py` refuses another checkout's environment. `research status RUN` checks back on it.

~~~sh
R="research run --on auto --cwd source -- bash -lc"
S="uv sync -q --all-packages --extra torch-cpu && uv run tools/check/suites.py"
$R "$S"                         # every suite whose inputs changed since it last passed there
$R "$S flock verity"            # these suites (a package name, a directory, or a unique part of one)
$R "$S flock -- -k tanh -x"     # arguments after -- go to pytest
$R "$S --quick"                 # without the tests marked slow
python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD   # here, in a second, before the pre-mark run: the no-growth lints on your change
$R "$S --changed $(git fetch -q origin main && git merge-base HEAD FETCH_HEAD) --quick"   # the pre-mark run; the mark's tier runs it on the head merged onto main
$R "$S --fresh"                 # ignore the cache, Lake's artifact cache too: what the suites test is compiled
uv run tools/check/suites.py --list                    # each suite's inputs, and whether a pass is cached (local: reads only)
~~~

To record a check of your clean commit as an Attempt, the whole check (every job on the merge gate: circuit-check and
ratchet-gate, off it, only with `--deferred circuit-check,ratchet-gate`) or some of its steps (`--deferred lean-agreement`),
launch `uv run python tools/check/check.py --record --on auto --pool research [--deferred STEP,...]` from your
checkout. It starts its own `research run` in a slot of the research pool `research run --on auto` would spill to (the CPU
nodes' check slots take train checks only), so inside a `research run` it is refused, naming this command.

Locally, while editing, one named test file or `-k` selection of under about a minute and 1 GB is fine:
`cd verity/protocols/proofs/flock/prover && uv run pytest tests/test_layouts.py -k round_trip`. It doesn't cache and doesn't load the read guard. The shell
hook refuses a local pytest that names no file or `-k`; `AGENT_GUARD_LOCAL_OK=1` is the opt-in.

From the same skill's other sections:

- The control pod exports `main`'s registry every 5 minutes, so a fixture registered with `research data put` needs no
  ask, and `research data fetch-fixtures` fetches by hand (as before `cargo test`).
- Inside `check` and `research run --on` the evidence store is reachable only through the run's custody key.
- `research run` starts every job with `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16` unless the environment already names it.
- adapters/vllm's heavy circuit-neutrality tier runs on a 256-CPU node (`research pods list`). Its fast tier runs after
  each landing (`post_train.py`), and a row it finds moved rings the vLLM adapter's owner.
