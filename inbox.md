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

