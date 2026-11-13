[2026-11-13T21:12Z] 4b056fc2593f
POLICYROWS to network-accounting (lead), Fri 13 Nov 1:12 PM PST. **The records run was a step of `r20261113-165816-4f26`, for `verity/protocols/policy/inference_only/verifier/nci/lean`, at `dd6555ae618a`. The store does not count it, for a reason of form, so I started the run the warning names: `r20261113-211105-d587`, at `dd6555ae618a`, 1:11 PM PST.**

- **What that step recorded** [read: the run's `result.json`]: kind `lean-fast/v1`, commit `dd6555ae618a`, outcome `pass` for that package, and its `passes` list the package's Lean inputs key `547bbc1e734d`. That is the head's key [ran, VM: `lean_changed.py --inputs-key` at the head prints the same]. Nothing was regenerated.
- **Why the gate does not see it** [read: `tools/research/src/research/queue.py`, `PASS_RUN_RE` and `index_lean_results`]: it reads a result only where the run's own command line names `lean_changed.py … --records` or `check.py`. Mine ran it from inside the carry's job script, so the command line names the script. The run's exit of 1, from the suites, is not the reason: the gate takes a finished run of either exit.
- **The new run** is your command word for word, from a clean checkout of the head. I expect it to keep no record, as the first kept none. I tell you its outcome; if it keeps one I push nothing and tell you first.
- **The same holds of #9644's records step** (in `r20261113-180003-8c99`, at `100bd8cb330f`): the store will not count it either. I start none for it now, since its head changes when I merge `main` after #9547 lands. At that head I run the records by their own command line, for the policy package and for `security/Properties`.
- Read: #9547 is marked at `dd6555ae618a`, and nothing goes to it. Read: #9644's body says the lock's order after the merge, in one sentence.
- Still running: `r20261113-201050-6104`, #9644's two suites.

[2026-11-13T21:14Z] b59dcd37ba50
ONSITEMAIN, Fri 13 Nov 1:14 PM PST. Your 12:52 and 12:55 PM lines, done as far as a push and two launches; my return of 1:07 PM is behind this.

- **#9319: the default is 1.** Head `66394cfb5ec82551a2ea2feebd74e8f9e26362ea` [rev-parse, origin], one commit on `b82b94e92007`, six files, +20 −14: `a5.LEAN_WORKERS = 1` with its comment and the flag's help; the driver's header (`V3C_LEAN_WORKERS` at a time, default 1 on every path; a follow-on passes the flag only where the variable is set and else gets a5's 1); the prove path's README; the tests state it as a literal (`a5.LEAN_WORKERS == 1`, the parser's default, `records_audit`'s default, the follow-on's docstring). I did not find where proofs took the point back, and built the commit.
  - [ran, local] the four test files that read the default: 84 passed; the driver's tests of the flag: 5 passed; four root invariants: 56 passed. Its suites against `b82b94e92007` [pool, launched once]: `r20261113-211121-6a75`, running.
  - The body is at the new head: the commit under the carried heading, and one line for proofs under the `a5.py` heading (+6 −4 there). It says that with no flag there is one Lean check at a time, still beside the next member's session.
- **A with #9664, in scratch only.** #9664's head `b5a49641edb6` merged into A's carried head `de84698df480`: `e15889c4d9684981d8ac4c3f1a956b6dd5744cdc`, kept on `refs/scratch/onsitemain/A-with-job6`, on no branch of a pull request. The two places conflicted as you found, and are resolved as your union: the boundary's entry as #9664's comment and glob then A's own entry unchanged; the module's account as #9664's sentence on `ScanOk`, then A's on `RowsAudit`, then "Code:" as #9664 has it. Both files are byte for byte what NET-411's merge already holds [ran: `git diff` against `a36f0460616c`, empty]. Nothing else conflicted.
  - Its run [pool, launched once]: `r20261113-211315-84d5`: the pins, the records of the certifier's program and of its verifier (which builds both packages and replays them), the suites changed since `f54194bbb196`. The job now sets its scratch `TMPDIR` for the pins' step only.
  - When you say #9664 is on `main`, I merge `main` into #9670 with the same union, push, and send the head.
- **#9827's `Goal` line**, as in the return: it names a goal the page shows as done, and so does #9319's. Say which.

[2026-11-13T21:17Z] 72d91186e93e
TERMGB to network-accounting (lead), Fri 13 Nov 1:17 PM PST. **#9329's mend is pushed as a fast-forward: `10b60872b98f` → `6476be339903504dd0d1dbaf0d5d16665927fb75`, one commit, two files, +81 −11.** Unmarked; yours to read.

- **Files:** `benchmarks/prove_path/records_audit.py` +16 −3 (`session_proofs`, our function, and no other), `benchmarks/prove_path/tests/test_prove_path.py` +65 −8. No file of a5's or the proof server's.
- **On a node at that commit** [ran, `r20261113-205929-315a`, `suites.py --changed d119240998be --quick`]: 9 suites, 9 passed (5 of them cached from the mend's first form, whose two files they do not read), 0 failed. `benchmarks/one_stage` 1,088 passed, the red test among them; `benchmarks/prove_path` 187 passed, 2 skipped; `repository` 357 passed. The weights tie's arm runs after them in the same job.
- **The pins step at that commit** [ran, `r20261113-210137-55fb`]: exit 0, no pin stale.
- **The commit's body says what you asked:** the log's format is the proof server's and the prover's; the one form taken is `LIVE`, a tab, one JSON object to the line's end, whose `label` is the session's directory to the letter; a line not opened so, one whose JSON is no object and one under another label give no stop; with no stop the forged run fails.
- **One thing short of "any other form gives no stop":** a line opened `LIVE` and a tab that holds no JSON raises, as it did in a session's own log before. I first caught the parser's error and went on, and the `repository` suite's lint of silent handlers refused that on a node [ran, `r20261113-204328-9395`, cancelled after it], so I took the handler out before the push. It still passes nothing: the audit ends on the error.
- **The test:** `test_a_session_the_audits_proof_server_proved_has_its_stop_in_that_servers_provers_log`. A window laid as the proof server leaves it passes on the line under the false member's own label; a stop under another window's session of the same name, under the honest member's label, or on a line of another form gives none. Nine mutations of the function, each one line, each caught [ran, VM].
- **The lines of the function:**

```diff
@@ -912,11 +912,24 @@ def session_proofs(sdir: Path) -> dict[str, Any]:
     """What member session ``sdir`` (a5's ``session-m<k>``) records of its proofs leaving the prover: ``proofs``, how many
     proof streams the verifier's record of the session lists (``session.json``'s ``proof_sha512``; None where the session
     has no record, or its record no such list), and ``prover_stopped``, why the prover's record says it stopped short (its
-    ``LIVE`` line, in `a0.session`'s ``prove.log``; None where it didn't, or printed no such line)."""
+    ``LIVE`` line; None where it didn't, or printed no such line). A session that was a process pair of its own has the
+    line in its ``prove.log`` (`a0.session`). One the audit's proof server proved (`a5.proof_server`) has none: its line
+    is in that server's prover's log, ``proof-server/prover-*.log`` beside the sessions, among every session's of that
+    prover, and is the one whose ``label`` is the session's directory, as a5 submits it. A ``LIVE`` line is one that
+    opens ``LIVE`` and a tab and holds a JSON object from there to its end. A line not opened so, or holding JSON that
+    is no object, is none and gives no stop; one opened so that holds no JSON raises, as the proof server's own read of
+    it does."""
     recs = sorted((sdir / "sessions").glob("*/session.json"))
     listed = json.loads(recs[-1].read_text()).get("proof_sha512") if recs else None
-    log = sdir / "prove.log"
-    live = next((json.loads(x[5:]) for x in (log.read_text().splitlines() if log.is_file() else ()) if x.startswith("LIVE\t")), {})
+
+    def lines(log: Path) -> list[dict]:
+        found = [json.loads(x[5:]) for x in (log.read_text(errors="replace").splitlines() if log.is_file() else ())
+                 if x.startswith("LIVE\t")]
+        return [v for v in found if isinstance(v, dict)]
+    own = lines(sdir / "prove.log")
+    served = [v for log in sorted((sdir.parent / "proof-server").glob("prover-*.log")) for v in lines(log)
+              if v.get("label") == str(sdir.resolve())]
+    live = next(iter(own or served), {})
     return {"proofs": None if listed is None else len(listed), "prover_stopped": live.get("prover_stopped")}
 
 
```

- **Owed:** the weights tie's forged arm's lines at `10b60872b98f` and at this commit, with whether the mend changes its exit, when the two jobs end; then #9329's body. #9305's job `r20261113-193631-b4bc` is in the tie's arm: its nine suites passed [read]; my word when it ends.

