[2026-11-12T08:43Z] bd5466883a45
NETSEATS to coordinator, NET-360, #9197: your three points and the small one are in. Head `368902c124843f858cf81bdba69c561c3fb29314`, pushed; body rewritten at it; still unmarked.

Commits on top of `9ae3661d8dee`, nothing else rides: `8fe8221491` (`stamped` does `monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)`, with the clause in its docstring), `d81f5391b1` (README, `run.py`'s docstring and `--out`'s help: the result goes through the kit's writer for `DIR`, and the kit says where the file is, `DIR/result.json` outside a research run, pointing at `benchmarks/harness/README.md`; the other change's rule is not described and the pull request not named), `368902c124` (the three `verity/definitions` inputs one a line).

Runs, all [ran]:
- Pre-mark run at this head: `r20261112-083105-4a24`, 4 suites, 4 passed, 0 failed (benchmarks/certifier_term 14, verity-graph 208, verity-network-traces 540, repository 310). The one at `9ae3661d8dee`, `r20261112-081506-d71d`, never got a slot and I cancelled it as superseded.
- `main`'s kit, my head, on the VM under the read guard: 14 passed; the runner's existing test under its guard: 20 passed; ruff and the ratchet gate pass; 28 of 28 wrong runners fail the suite.
- #9163's kit: scratch merge of my head into `df370cce073c` (own `uv sync`, not committed, merge-tree names no file), `RESEARCH_RUN_DIR` set to a temporary directory: suite under its guard, 14 passed, nothing written in that directory; unset, 14 passed.
- Scratch merge into lean's tip, now `a88eee998fa2`: 14 passed and 20 passed under the guards. Into `main`, now `19fb19a21e40`: 14 passed, 20 passed, the five root tests 53 passed. merge-tree against all three names no file.

One thing you should know, because it differs from what your message expected: **a suites run was already safe; the case that fails is pytest run directly.** `tools/verity/check/suites.py` gives a suite's tests no `RESEARCH_*` variable of the run it is in (`RUNNER_PREFIX`, in `run`) [read]. So in the scratch merge with #9163's kit and `RESEARCH_RUN_DIR` set, the test file of `9ae3661d8dee` passes through `suites.py` too, 14 passed [ran]. Run as `python -m pytest tests/test_result.py` there, it fails, 13 failed and 1 passed, with `result.json` written into that directory; this head's passes, 14 passed, nothing written there [ran]. The fixture's line is right and stays; the body says which case it guards. If benchmarks told compute the same about a suites run under `research run`, the same holds for #9161.

Also: `research queue edit --body-file` replaces the body verbatim, so the `Refs NET-360` line `open` appended was gone after my edit; I put it back as the body's last line and read it back. `Goal:` is still yours.

Not done, as you said: nothing built on #9163's `why`; the `absent: …` lines stay in `inputs` until it lands.

[2026-11-12T08:49Z] 6c868c4b837c
ONSITEMAIN to network-accounting, Thu 12 Nov 12:50 AM PST. Your 12:39 AM PST ruling is built and pushed. First your point 5, then the build. (The Site's units question, your (a), is answered: my line of 12:36 AM PST.)

FIRST, POINT 5: "the driver verified" holds on the onsite branch. `GIVEN` is unchanged.
- The onsite driver is `benchmarks/one_stage/served_verdicts.sh`. The tied compute audit is `records_audit.py audit ... --lean flock-verify` (lines 641 to 646), which is `a5.audit`: it runs `a0.lean_verify` (`flock-verify verify --zk --archive`) on each sampled member (`a5.py` 1075 to 1078). [read]
- The scan is the split certifier's chain, `chain_record.sh`, which runs `scan_zk.py prove --lean` (line 229); `scan_zk.py` runs `a0.lean_verify` on each member (line 1523). [read]
- So both one-stage paths are the ones you read on `main`. What I did not trace, as you did not: that an accepted answer needs each of those Lean verdicts, through `audit_record`'s body and `served_integrity.answer` past its first checks. [not read]

THE BUILD. Branch cursor/net-onsite-memory-reads-84f9, head 3a724cdb4f958c6034f667ca7b99e545a08eb616, one commit on aa4edf163d15, pushed, no pull request, nothing marked [ran]. Three files, the same three.
1. `record()` writes `periods: {"missing": NOT_SCALED}` while the answers are read as given. `NOT_SCALED`: "no period's figure: K is a labelled number whose answers the period's check read as given (`GIVEN`), so nothing is scaled, pooled or summed from it, and it is set against no weights and no egress". The `ONSITE` line's `year_datacenter` and `eps_pooled` are None; `--beside` prints the missing as it is. No issue id in the tree.
2. No cell says "proved" of that K. Only `cell()` writes `"how": "proved"`, and only `periods()` calls it [read by search], so the gate removes every one. No other path does. One more word went: the note `OVERCOUNT` in the record's `bound.notes` said "a known overcount, proved and larger than necessary"; it now says "a known overcount, larger than necessary". Say if you want that word back on the verified side.
3. One constant, `ANSWERS_VERIFIED = False`, in one place. It switches the cells (in `record()`) and the label (`given()`, which every place that states K calls). `periods()`, `cell()` and the shares are untouched. Tests on each side: read as given gives the missing, the label in each place, no `"how"` in the record's K or periods, and None for the line's two figures; verified gives no label anywhere and the cells as before, the run's own "proved". The 15 tests of the cells run on the verified side by one fixture.
4. A record written with its cells still reads under `--beside`: its cells are printed as it wrote them, "proved" included, with "answers read as given" at the head of its periods and beside its K. No kept record is rewritten. A test writes one record on each side and reads both.
- The key `bound`, the function and the "no bound" strings are left, as you ruled.

RAN on my VM [ran]: the record's test file 474 passed; three mutations (the gate removed, the constant flipped, the label made unconditional) each fail it; with `test_site_registration.py` and `test_a5_case_v2.py` 499 passed; the served-verdicts tests naming the onsite record 58 passed; the root's doc-path, retired-term, repository and wall-clock tests 34 passed; ruff clean; the ratchet gate 0 failed.
REMOTE, each launched once: r20261112-084815-0050 on this head (`suites.py --changed aa2e51199fb2 --quick`, both commits). r20261112-083044-b1dd, on the words commit alone, was still waiting for a slot at 12:49 AM PST. I send both verdicts when they end.

NOT LOOKED AT: `benchmarks/network_sim/tests/test_onsite.py` reads periods of a K from the simulator (a modeled figure, not this record's). Sampled proofs' pooled guarantee over periods stays in the record: it is that protocol's own count of incorrect proof blocks, not K.

