[2026-11-12T23:36Z] bf6895c2fc20
KEY, Thu 12 Nov 3:35 PM PST.

**#9474's new head is merged: #9502 is at ea620a89cf, pushed. The nci package's Lean check passes there and the proof suites are running at it. A slip first: my turn restarted on a new machine at about 3:03 PM with two things not on origin; both are restored and pushed, and no verdict was lost.**

Slips, at the hour I know of them:
- **The restart.** The docstring commit 177e9c77dc was not pushed (I was waiting on its tries). I redid the same three edits as 4ac888e689. The served branch `cursor/trn147-served-cases-10f7` had never been pushed either. Its commit dbe1d8ede3 was still in vy-nebius-cpu-14's git; I took it back as a pack (RAN, r20261112-231318-250d) and pushed it, hash unchanged. I named that node because the tree was there.
- **A run's command line.** My earlier `research status RUN` checks printed the table's command column, about 45 characters of it, several times. No secret was in it. I now read named fields only.
- **Runs cancelled with no verdict.**
  - r20261112-214628-0ad4: the suites at 5757d4f6ee, never started (cpu-6's memory was reserved for 91 minutes).
  - r20261112-214623-2624: the security proofs package's check at 5757d4f6ee, 86 minutes in, two heads behind.
  - Five launched at 4ac888e689 about a minute before PROBE's 3:16 PM line reached me, each cancelled within five minutes: the suites (r20261112-231654-9dc4), the nci check (r20261112-231723-49f7) and the three proof tries (r20261112-231737-4e25, -9a39, -52be).
- A wrong `uv run` left a `.venv` and a `uv.lock` in the network package's directory; removed, never committed.

The head, ea620a89cf (4ac888e689 plus the merge of b23c853a35):
- RAN locally: `ratchet_gate.py origin/main HEAD` fails on exactly 3 sites, all #9474's (`RecordCheckMain.lean` 3 and 15, `lakefile.toml` 57), and none of mine.
- RAN locally: `test_nci_record_difftest.py` and `test_nci_record_check.py`, 17 passed, 1 slow deselected.
- RAN, r20261112-232414-cced: the nci package's Lean check. Last line: `AUDIT: PASS; reports in …/lean-audit`. The lock's runs ran, the record difftest's Lean half among them.
- RAN, r20261112-232515-59b3: `RecordInputs.lean`. Last line: "axiom probe: none of its 14 theorems uses an axiom beyond propext, Classical.choice and Quot.sound".
- RAN at 4ac888e689: `Record.lean` and `RecordText.lean`, lean exit 0 each (r20261112-230723-5c05, r20261112-230723-42fa).
- Running at ea620a89cf:
  - `RecordRead.lean` (r20261112-232515-0fba) and `RecordTemplates.lean` (r20261112-232515-de9b);
  - the security proofs package's Lean check (r20261112-232445-9cb4);
  - the pre-mark suites, the same 57 in five groups: r20261112-232245-e582, r20261112-232311-85ba, r20261112-232338-3ab1, r20261112-232351-a95f, r20261112-232403-d24c.

Served cases:
- RAN: all twelve at dbe1d8ede3 have ended. The compiled form's rate verdict and its count equal the reference's on every field in every case (r20261112-212235-e658, r20261112-212342-43a8, r20261112-212342-1920). The table is in the PR body.
- READ (`git diff dbe1d8ede3 ea620a89cf`): that tree's `Record.lean` and `RecordText.lean` are the head's but for docstrings. Its command, script and reference are not the head's. It has #9474's two earlier commits, whose script switched off the reference's two leaf-by-leaf expansions itself. The head has `form_only` and the restated `own-rows` rule.
- So I launched the same twelve again on the head's own reference and command: r20261112-233108-21af, r20261112-233205-6c00, r20261112-233259-86da, at efd15349eb (ea620a89cf plus the script's `--format` option, pushed, never a PR). The earlier runs took 80 to 92 minutes and peaked at 35 to 36 GiB each (RAN, their cgroup peaks); these ask 44 GB.
- The `--format` option is on my served branch only. It adds 33 lines to #9474's script, and I tried it on the 15 made cases that state formats (RAN locally). If you want the format cases reproducible from `main`, say so and I send PROBE the diff.

The PR body is current. I end this turn now, as the hook requires, and a timer brings me back in 20 minutes on a new machine. Everything I hold is on origin, in the runs, or in my note `20261112T2243Z-handoff-KEY`, which I update before I stop.

