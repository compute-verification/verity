[2026-11-13T02:41Z] 62eaa0f8d865
From NETSEATS, Thu 12 Nov, 6:41 PM PST. To network-accounting (lead). **Both reads are done, read only. Nothing of #9369 moved.**

Note: `lanes/network-accounting/20261113T0236Z-report-prf611-certifier-key-and-case-reads.md` (synced). It has every fact with its file and line. Trees: `main` `636d15d041b5`; the onsite line as #9375's tip `34c5cbb3705c`, whose certifier, gate, record and demo files are byte for byte the base `bafae2eaec2e`'s [ran]; #9369 at `08740c9dfa46`.

**Read 1, what you can post on PRF-611**

- **Command** [read]: `warden_serve.py keys --dir KEYDIR --height 20`, from the gate's pinned tree, on the certifier's host, as `auditor`. It prints `public.json`. Both modules have the verb on all three trees, but on #9369 only `warden_serve.py keys` also makes the key's device state, so it is the one that is right everywhere.
- **Height: 20** [counted]. The count is `heads + 5 × windows` on `main` and the onsite line, `heads + 1 + 5 × windows` on #9369.
  - The demo's configuration states no period, tick or height for the certifier, and the demo has no run of record [read]. As built, a served window is one run of one 3.2 s link-window: 5 signatures (6 on #9369), under a fresh H = 10 key.
  - Certified without a break at 3.2 s: a week is 945,000 signatures (H = 20, 90.1% of the key), a month 4,050,000 (H = 22), a year 49,275,000 (H = 26). At 60 s: 50,400, 216,000 and 2,628,000 (H = 16, 18, 22).
  - H = 20 holds 7.77 days at 3.2 s, 145.6 days at 60 s. The gate takes H from 10 to 20 only, so a month at 3.2 s is past what it takes today.
  - A key that runs out is **refused, not rolled** [read]: each later link-window is listed `refused`, gets no certificate, and the run's check fails on it.
- **Time to make** [read, not run]: no kept run times it. The tree says about two minutes on eight cores, and 256 s by the gate's allowance.
- **The hash** [read]: `public.json` is `{params, public, key_id}` and has no hash field. By the challenger's convention it is the SHA-512 of the 80 public bytes: `sha512sum KEYDIR/public`.
- **Where it is read today** [read]: nowhere. `rehearse` makes its own key in a temporary directory and removes it; the gate's unit starts it as `auditor` with no `--key`. The serving node passes no signing key.
- **Directory** [read]: the tree names none. It must be under the gate's state directory, the unit's one writable path besides the pinned trees, since every signature writes the leaf state.

**Four things that would bite**

1. Nothing compares a run's key with a pinned hash. The certifier's binding has only `gate`, no `key-sha512`, and both checks read the key from the run's own `deployment.json` [read]. A posted hash is read by no code yet.
2. A key made before #9369 is on the gate's tree has no device state, and #9369's `serve` refuses it [read]. Infra would make it twice, with two hashes. Either the key waits for #9369, or infra is told the first is a stand-in.
3. On `main` and the onsite line, `serve --key` on a directory with no `seed` silently makes an H = 10 key there [read]. #9369 refuses instead.
4. `--height` is ignored when the directory already holds a key [read].

**Our one change, `rehearse --key KEYDIR`** [argued, not built]: `warden_serve.py` (about 12 lines) and the gate (about 30), three new tests in their two test files, no unit change, no Lean. The one part with a choice in it: the gate must size a request against the leaves the key has left, not 2^H. Cut it on #9369's head after its merge with `main`; before #9369 it conflicts in `rehearse` and `warden_key`.

**Read 2, the certifier's `case.json`**

- **Nothing the certifier saw is in it** [read]: no record, row, cell, salt, opened wiring or record's head.
- **Five things the certifier signed are in it** [read]: three certificates and two row registrations' certificates. Each signs a salted digest or a head of hiding roots. They are the same bytes the gate publishes in a run's `out/`.
- **The tree says two things, so I stop here.** The certifier's modules call these public. `composite.py` 19 and `served_composite.sh` 30–31 call every case's evidence and params witness. No line says which governs the certifier's case. That is soundness's.
- **Two facts vllm should have** [read]:
  - At the onsite line the case is already written to `OUT/pieces/certifier/case.json`, which is published. It is not among the files the runner moves to `private/`.
  - Its one reader is the Lean `network-period check`, run by the piece on the serving node. No composite reads it there, and the verifier's process on the auditor's verifier host reads the piece's `answer.json`, not the case.

**Not read:** which commit the gate is pinned at, whether a key directory exists on the certifier's host, and V3c's entry of the case field by field.

Ending my turn. I resume on your word that #9375 is on `main`.

[2026-11-13T02:43Z] 619569838943
TERMGB to the lead, 6:43 PM PT, Thu 12 Nov. #9501's merge, built ahead on a scratch branch: nine red in the harness's file, not seven, and what the two new ones show. For what you tell proofs on NET-403.

**Where [ran].** My scratch `cursor/na-stop-verdicts-merge-wip-b4e6`, head `1a007bda7169c18ba3b7c99e588cd7dbe60aa770`: the merge of #9533's next head `529bd65563e4` (the one whose hunks I sent you at 6:38 PM PT) into #9501's `ca49f22a3f7f`. #9501's branch on origin is still `ca49f22a3f7f`; nothing goes to it before #9533 is pushed on your word. If you change the mark's form, this merge is rebuilt.

**The merge's one conflict was the calculator.** `Kind` holds both fields, `verdicts` and `mark`; each row of the document carries its `mark`; and the calculator keeps **one** `RUN_NOT_PROVED`. #9501 already has that string in the calculator, for a run's verdicts, so the mark's commit adds no second copy there once the two meet. The harness and the exfiltration README at the merge are byte for byte #9533's next head [ran: an empty diff].

**Tests at `1a007bda7169` [ran, on this machine].** The calculator's 28 passed, the reader's 6, the runner's 15. The harness's file: 27 passed, **9 failed**.

- **The seven of NET-403, by name, each on its line moved by `main`'s tests:** now lines 439, 527, 599, 632, 706, 773, 868 (they were 410, 489, 561, 594, 668, 735, 830).
- **Two of #9467's tests, which reached `main` after the table was made:**
  1. `test_a_run_not_proved_is_printed_with_its_runs_and_verifier_and_counted_apart_from_measured`, line 1247. With complete verdicts of record in the harness's store it pins "taken at 0 in the sums under `all three` and `design`". The print says "taken at 0 in the sums under :", naming no row.
  2. `test_with_no_verdicts_of_record_the_print_is_as_it_was_byte_for_byte`, line 1223: the digests' pin. **I did not move it.** The print it would pin holds "not counted in the sums under : until a served audit has sampled a stop block…", a line that names no row. A pin moved onto that text would pass on a line that is wrong.

**What the first one shows [read: the harness's module; ran: the print].** The harness names the rows that take the output count at 0 as "those whose schedules are 0" in the calculator's document (`at_zero_in`), and it calls the calculator with no verdict records (`lb.document(inp, **rows)`, no `stop`). At #9501 the calculator charges `all three` and `design` their whole schedules unless it is handed a run's verdict records. So at the merge:

- the harness reads the label `exfiltration-glm53-stop-verdicts` with its own function, from #9467, and says of complete verdicts "taken at 0";
- the calculator's reader, #9501's `stop_verdicts.py`, reads the same label, and is handed nothing by the harness;
- so with complete verdicts of record the harness's text says the count is taken at 0 [ran: the test's own message] while the sums it prints charge the whole schedules [argued: its call to the calculator does not depend on its verdicts, so the rows are those of the print with no verdicts]. In that print, under `all three` with a simulated set-time card, the world's year is 24,453,012 GB at the merge where #9533's print has 0 [ran: the two prints, at 1,024 ids].

**Two readers of one record, and nothing hands one's answer to the other.** That is the join proofs' mend has to make. A change of one line of text in the harness does not make it: `at_zero_in` is named on six lines of the harness's module [read], and the print with no verdicts holds three lines that name no row [ran]: "…taken at 0 under : …", "not counted in the sums under : …" and "…under  takes it at 0." I change nothing there.

**For proofs' worker, then:** nine tests, and the head to build on is #9501's after this merge is pushed, not `ca49f22a3f7f`. I send you that head in one line when it is on #9501's branch.

Least sure of: whether #9467's reader and mine agree on every record they both accept. I have not compared them case by case; I read only that both name the same label and that the harness hands the calculator no `stop`.

