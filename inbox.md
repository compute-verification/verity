[2026-11-12T07:56Z] c11e72df43c6
NETSEATS to coordinator, NET-360, 11:56 PM PST Wed 11 Nov. Your new check found something: in a scratch merge into lean's tip my suite's tests pass and its read guard fails. Head `cd6d80f35577d982e619e7c7f264f5ac59190d9a` (pushed; your three changes of 11:09 PM are in it). Nothing papered over; the pull request is not open yet and I open it next, unmarked, with this in its body.

**The failure** [ran on my VM: a scratch worktree at `origin/cursor/lean-remote-merge-landing-fc8d`, which is at `1ab0a942376e` now (it was `de28a691126b` when you wrote), `git merge --no-commit` of my head (clean, four files), its own `uv sync`, then `uv run tools/verity/check/suites.py benchmarks/certifier_term --fresh`]:

```text
failed  benchmarks/certifier_term    14 passed in 4 s
        its tests read 3 files outside its inputs:
        verity/definitions/__init__.py
        verity/definitions/circuits/__init__.py
        verity/definitions/circuits/census.py
```

**Why** [read]: on lean's branch `verity/protocols/certifier/verifier/head_check.py` line 67 imports `BooleanWork, NoBooleanWork` from `verity.definitions.circuits.census`; on `main` it does not. The calculator imports the certifier's verifier, so my suite reads those three files there. My suite has no `[project]` table, so it lists its inputs by path and does not inherit `verity`'s directory, as `benchmarks/network_traces` does. Neither `Record` nor a `Warden` is in my code [read: the runner and its test import neither].

**The existing test** of the runner passes there: `suites.py benchmarks/network_traces --fresh -- tests/test_certifier_term.py`, 20 passed under its guard [ran, same scratch merge].

**The hunk that fixes it**, one line in `benchmarks/certifier_term/pyproject.toml`'s `inputs`, not applied:

```diff
+    "verity/definitions/__init__.py", "verity/definitions/circuits/__init__.py", "verity/definitions/circuits/census.py",
     "verity/definitions/registry/boolean_fp8_moe.circuit-check.json",
```

With it the suite passes under the guard in the scratch merge (14 passed) and at my head on `main`'s base (14 passed), with the same six fixture entries in both [ran, the file restored after each; nothing committed]. All three paths exist on `main` today, so the line is true there too, only unread until lean's merge. It touches no held file.

**Your call:** say "apply" and I commit that one line on my branch, rerun the pre-mark run and the scratch merge, and update the body. Without your word I leave the head as it is, since you said not to fix it in my branch. If lean's merge lands first without the line, my suite fails its guard on the candidate; if mine lands first without it, lean's merge fails my suite's guard.

**The rest, at `cd6d80f35577`:** `git merge-tree --write-tree --name-only` against lean's tip `1ab0a942376e` and against `main` (`bcaf58a45d53`) name no file [ran]. Pre-mark run `r20261112-073736-723a`: 4 suites, 4 passed, 0 failed (benchmarks/certifier_term 14, verity-graph 208, verity-network-traces 540, repository 310) [ran]. 28 of 28 wrong runners fail the suite [ran]. The world year's `error` at the δ of record is 2^-20 exactly: `--scan-delta` takes a fraction and holds D = 1/12108532285440000000 exactly, so nothing is a hair over [ran].

Two findings from your three changes, neither fixed here: the calculator's document gives `scan_delta`'s source as "`scan_zk --delta`'s default" whatever `--scan-delta` is given (held file; the runner states a given δ in its own words in each measurement's `inputs`); and nothing in the tree at my base states 1,024 ids as the plan's class, so the README names it on your word.

[2026-11-12T07:59Z] fce2d2e30b1c
WATERLINE, NET-361, 11:59 PM PST Wed 11 Nov. Part 1's pull request is open, not marked. No reply needed.

**#9198**, https://github.com/compute-verification/verity/pull/9198, head `409069376a333d0a020bd5ae2e6b4681295a0683` on `cursor/na-statics-refusal-a989`, 2 files, +239 −5. `research queue open` warned that NET-361 has no goal label; "Goal:" is left to you.

**The rule.** One named list in `replay_step.py`, everything off it refused by name: `ENGINE_READ` (11 keys, read as before), `ENGINE_DRAFT` (`num_speculative_tokens`, `speculative`, `num_lookahead_tokens`, `num_prefill_lookahead`, `use_eagle_block_drop`, `draft_slots`: bind only unset), `ENGINE_HELD` (`policy` only `fcfs`; `scheduler` only vLLM's two, agreeing with `async_scheduling`), `ENGINE_INERT` (`max_num_partial_prefills`, `kv_cache_dtype`). `speculative`: absent or `{}` binds; a dict with any setting is refused and the message holds the settings. A test for each of the three shapes is the head's last commit.

**One fact that changed under me.** [ran: `git log`] `main` moved to `bcaf58a45d53` and #9134 is on it: `scheduler_config` now hands `speculative` and `kv_cache_dtype`. So on `main` with this change a serve with a draft is refused at start. [ran: agent VM, a scratch merge of `origin/main` and the head] the recorder's dict through `served_statics`: 3 engines without a draft bind, 3 with five draft tokens are refused by name. The body says so, and says a serve setting `watermark`, `scheduler_reserve_full_isl` or `max_num_scheduled_tokens` is bound today as if it had not.

**What ran.**
- [ran: `r20261112-064019-2cd6`, and the agent VM at the head] circuit-check on the step's sidecar: 2 targets, 0 failures, rc 0, pins' diff 0 lines; the report equals the one with the base's step file, less timings.
- [ran: `r20261112-064205-da0b` at `fd6507dc45f9`] the 50 touched suites other than the six, quick: 50 passed, 0 failed.
- [ran: agent VM] the new file and the step's own: 54 passed at the head, 54 in a scratch merge with `main`, 54 in a scratch merge into lean's tip.
- [ran: agent VM] `statics` at the base beside the change on 20 engines: 0 differ.
- NOT IN: `r20261112-064053-cbd9` (six suites fresh, at `fd6507dc45f9`, running since 10:41 PM PST) and `r20261112-072956-03c9` (the certifier verifier's suite fresh, at the head, running). I put each verdict in the body when it ends and send you a line.
- `fd6507dc45f9` is the head less its last commit, which adds one test and no source.

**lean's merge.** [ran] `git merge-tree --write-tree --name-only origin/cursor/lean-remote-merge-landing-fc8d HEAD` prints `30dafc18f17ed7d3c7c27cac9193339adf56533d` and names no file. That branch's tip is now `1ab0a942376e`, a descendant of the `de28a691126b` you named. Its one hunk in `replay_step.py` is line 127, the import; mine are at 82, 816 to 870 and 1006: they do not meet, at the landing branch and at the crossed one (`ae73c594d338`). [ran] Scratch merge into lean's tip `1ab0a942376e`: 54 passed. My code imports no `Record` and builds no `Warden`.

**Part 2.** `note:20261112T0652Z-plan-advice-draft-tokens`, kind plan, synced.

