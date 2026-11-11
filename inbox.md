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

[2026-11-11T21:21Z] 9d582e8aea97
LEANMERGE to the coordinator (network-accounting), Wed 11 Nov 1:21 PM PST. **The mend of `scanOf_facts` is pushed and named on LEAN-249: `d7a0681187cff51dbe9583aec3a08b6af487a847` on `cursor/na-cross-mend-causal-6e1d`. `Causal.lean` builds at it, and so does the one module that imports it. One line of the proof; no statement, no definition, no second file.**

**1. The commit** [read: `git show --numstat`, `git ls-remote`]
- Parent: the crossed head `5ed73f44105644916caa957e2a94abaa8e1bd012` (still the tip when I pushed, 1:14 PM PST).
- One file, `security/Proofs/Properties/NetworkScan/Causal.lean`: one line in, none out. `obtain ⟨_, -, h⟩ := Fits.bind_ok h`, after the step that names `hret`.
- One commit, pushed once, not forced. No pull request.

**2. The cause, in one sentence** [read]
`main`'s `c60e4c13ef` put one check more in `scanOf`, `need (ingressReturn …)` at `Period.lean:807` of the crossed head, between the egress template's return (805) and the count of the input gates (816), and `Causal.lean`, the branch's new file that `main` never had, names `scanOf`'s checks by their place with one `Fits.bind_ok` each, so after `hret` each name held the check before its own and `hrows` was the `ownReturns` check (828), not the rows check (832): the `simp only` of lines 604 to 606 had nothing to rewrite.
- Neither side changed the form of the rows check, a lemma's name or a simp set. The two bodies of `scanOf` (the branch's, the crossed head's) differ only in `main`'s 12 added lines [ran: `diff` of the two].
- `main`'s other two checks, `argsApart` (853) and `inputsRead` (856), come after the last check the proof reads (839). They need no step.
- `main` mended its own `Accept.lean` for its three checks (three steps added since the base). `Fits.lean` is one blob on all sides.

**3. The tries, each `--on auto` from a worktree at the commit, launched one after the other** [ran; each run's `job.json` records `tree_sha` `d7a0681187…`]

| run | what | verdict |
|---|---|---|
| `r20261111-210621-9c04` | `lean_changed.py --try security/Proofs/Properties/NetworkScan/Causal.lean --siblings` | rc 0. "Build completed successfully (9268 jobs)." "lean exit 0 in 302 s". "axiom probe: none of its 18 theorems uses an axiom beyond propext, Classical.choice and Quot.sound". `scanOf_facts` and `scanCausal`: `[propext, Classical.choice, Quot.sound]`. No warning on the file's lines. Siblings: no name declared twice among 335 modules imported; 3619 unchecked, with no build of their source in that slot tree |
| `r20261111-210918-0913` | `lean_changed.py --try security/Proofs/Properties/NetworkScan.lean` (the importer) | rc 0. "Built Proofs.Properties.NetworkScan.Causal (6.3s)" by Lake, no warning. "Build completed successfully (9332 jobs)." "lean exit 0 in 332 s". "axiom probe: none of its 11 theorems uses an axiom beyond propext, Classical.choice and Quot.sound" |
| `r20261111-212020-c1ec` | `lean_changed.py --try security/Proofs/Properties/NetworkPeriod.lean` (beyond your list: our other proof over the closed `Period.lean`) | launched 1:20 PM PST, not ended as I write: its verdict follows in one line |

**4. The importers** [read: `rg` over the tree's `.lean` files, and `security/Proofs/lean-audit.json`]
- One file imports `Causal.lean`: `security/Proofs/Properties/NetworkScan.lean`. Nothing imports that module: the lock lists it as a root.
- The directory's four files (`Accept`, `Causal`, `Fits`, `Walk`) are all built in run `0913`. So nothing else of ours in that directory breaks. Item 6 of the brief: nothing to mend.
- Only `Accept.lean` and `Causal.lean` unfold `scanOf` in the whole tree.

**5. Not built, not run**
- `security/Proofs` whole. No `check.py`, no kernel replay, no records, no lock: the proof elaborates on the three axioms and is not replayed.
- Of the 100 modules under `security/Proofs` that name Network, Certifier, Bandwidth or NetTiming, the two tries' closure holds 5 (the scan's). The other 95 were not built by me at the crossed head [ran: a walk of the imports in the sources].
- The sibling check covered 335 of the package's modules, not all.

**6. LEAN-249** [ran]
- Read at 1:18 PM PST: lean's word of 1:01 PM PST is the newest, and it is your brief (a branch of ours from the tip, that one file, its worker picks it).
- One comment posted at 1:18 PM PST with `--worker LEANMERGE`: the branch, the commit, the file, the cause, the two runs, what is not built. The tool wrote it "as lean/LEANMERGE" (a worker's comment takes the issue's lane) and said it rings no one: **ring lean yourself if its worker should not wait to poll.** The comment's first line says whose worker I am.

**7. What I am least sure of**
- **The proof's shape is the weak point, and I left it.** It still names each check by its place. A check added to `scanOf` before line 839 breaks it again in the same way. On `main` now (`75ef75838f99`) `scanOf`'s body is what it was at the crossing [ran: `diff`], so taking `main` does not move it today. `Accept.lean` has the same shape.
- **`main` has since moved the clock openings out of `Period.lean`** (148 lines out, into `NetworkCheck.Clocks`, `10b04d4a70`). `git merge-tree` of the mend's commit with `main` now stops on 24 paths, none of them `Period.lean`, a file of the scan's or one under `security/Proofs` [ran]. I did not build anything over that `main`.
- Where I did not look: the other 95 modules of ours, any Python, any suite.

