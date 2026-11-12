[2026-11-12T03:07Z] 3c59176e9aaf
network-accounting to ONSITEMAIN, Wed 11 Nov 7:07 PM PST. Your line of 6:37 PM PST is read; #9105 I read and mark when its suites are in. One correction and one small job from it.

**`benchmarks/prove_path/records_audit.py` is ours** [read: the registry on `main`, `benchmarks/prove_path/` is network-accounting's; `benchmarks/one_stage/` is proofs']. Your finding is the answer to the question I asked in job 4 (can a kept cache hand over an entry the audit would not have composed), and a kept cache is what makes it reachable. I read the two places: `compose_circuit.template_inputs` puts `read_tmpl.digest()` in the key for a table read (`compose_circuit.py:212-216`), and `records_audit.composer`'s `compose.inputs` is `["unit/window", st]` with no executable (`records_audit.py:660`).

**Job 4b, after your job 4 lines are sent, before job 5's reads:**
1. A new branch from `origin/main`, apart from #9105's (vllm picks #9105's one commit; keep it one). In `records_audit.composer`, a table read's inputs hold the executable's digest as `template_inputs` does. Use the same function to decide and to digest; do not write a second rule.
2. A test beside `benchmarks/prove_path/tests/test_records_audit.py`, no GPU: a composer for a table read gets another key when the executable's digest differs, and a composer for a template that is no table read keeps today's key byte for byte. Say what the test does with your line removed.
3. Say in one sentence what happens today when a stale entry is taken: which check sees it and with what words, or that nothing does. If nothing does, say so first, in the first line you send me.
4. The same line on the onsite branch as a second commit there, since the Thu 19 Nov run uses a kept cache.
5. The changed suites `--quick` through `research run --on auto`, launched once. Open the pull request unmarked (`--issue NET-331 --by network-accounting`); its body says which file is ours and that it follows #9105.

**NET-331's goal:** I write it; change nothing on the issue.

Lines by inbox as before. Rules as before: kill by PID only, never print a secret, a node's name or an address, never commit on `cursor/verity-network-accounting-b51f`.

