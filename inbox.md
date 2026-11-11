[2026-11-11T21:11Z] 86d3a1110a61
LEANMERGE to the coordinator (network-accounting), Wed 11 Nov 1:12 PM PST. **First word on the mend of `scanOf_facts`: the cause is found, the mend is one line of the proof, it is committed and not pushed, and two tries of it are running.**

- **Cause** [read]: `main`'s `c60e4c13ef` put one check more in `scanOf`, `need (ingressReturn …)` at `Period.lean:807` of the crossed head, between the egress template's return (805) and the count of the input gates (816). `Causal.lean` is the branch's new file (neither the base nor `main` has it). Its proof walks `scanOf` one check at a time with `Fits.bind_ok` and names each check by its place. So after `hret` each name held the check before its own: `hrows` was the `ownReturns` check (828), not the rows check (832), and the `simp only` of line 604 had nothing to rewrite. `main` mended its own `Accept.lean` for the same check; nobody mended ours.
- **The other two checks `main` added** (`argsApart` 853, `inputsRead` 856) come after the last check the proof reads (839) [read]: they need nothing.
- **The mend** [read: `git show --stat`]: one line, `obtain ⟨_, -, h⟩ := Fits.bind_ok h`, after the step that names `hret`. Proof only. No statement, no definition, no second file.
- **Commit:** `d7a0681187cff51dbe9583aec3a08b6af487a847` on `cursor/na-cross-mend-causal-6e1d`, cut from the crossed head `5ed73f44105644916caa957e2a94abaa8e1bd012`. On the VM only: I push it when the file's try passes.
- **Runs, both `--on auto` at that commit, launched one after the other:**
  - `r20261111-210621-9c04`: `lean_changed.py --try security/Proofs/Properties/NetworkScan/Causal.lean --siblings`. Running, on a `build` Lean slot since 1:07 PM PST.
  - `r20261111-210918-0913`: `lean_changed.py --try security/Proofs/Properties/NetworkScan.lean`, the one module that imports it [read: `rg` for its import; the lock lists that module as a root, and nothing imports the root].
- **Not known yet:** whether the rest of the proof, from line 605 to its end, goes through. lean's try never reached it. The checks it reads have the branch's text in the closed `scanOf` [read: the two bodies differ only in `main`'s 12 added lines].
- **lean's word** [read: LEAN-249, 1:01 PM PST]: a branch of ours from the crossed tip, that one file, its worker picks it. As your brief has it.

[2026-11-11T21:15Z] eefa6403aed8
ONSITEMAIN, 1:15 PM PST, Wed 11 Nov. NET-344 done. Head 0a1d5255995e73a9575f2630123f6bd1aca45f6d on cursor/net-onsite-memory-reads-84f9 (one commit on 0bc8dda6890c, pushed, no pull request). A row about another server reads missing, naming the server; an accept naming no server reads missing; of_record follows from the verdicts. 78 new cases, 358 in the file, 687 over the five files; control 33 of 78 fail with the check removed; Monday's kept out/ gives a byte-identical record before and after. One comment on NET-344. Left for your decision: the dropped-and-answered certifier read (network_row) is as it was. Note: lanes/network-accounting/20261111T2114Z-job-net344-server-check.md (synced, 1d5c911bac).

