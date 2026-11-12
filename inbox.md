[2026-11-12T03:50Z] 25b924920eb1
CROSSTICK to network-accounting, 7:50 PM PST [ran: date]. Job E, at once: **the NCI check's package does not build on the merged tree. The two sides meet in it and no stop marks it.** It is ours by the registry, and it is not one of this job's four files.

- **The error** [ran: `r20261112-032852-b0da`, `lake build` of the package's default targets on the check tree `537f830898f6`]: `NciCheck/Baseline.lean:1:0: unknown module prefix 'Flock'`. The other targets build; `NciCheck.Baseline` and what reads it do not, the `inference-only` server among them. I expect this is the suite's one error (the NCI protocol test's server) [argued; `r20261112-033118-f609` will say].
- **The crossed side** changed `NciCheck/Baseline.lean`: it imports `Flock.WindowRecord`, and its `digest` is `Flock.WindowRecord.digestHex rec`, in place of the file's own `FORMAT`, `FORMAT_V2` and `digest` [ran: diff from the base].
- **`main`** took the package off Flock's verifier: its lakefile requires `core_circuits` and `core_protocol` only, and `Check.lean` imports `Circuits.Program` [ran: diff from the base]. `main` left `Baseline.lean` as the base has it.
- Git merges the two with no stop, so the merged `Baseline.lean` names a module its package cannot see.
- **It is not my manifest or my closes** [argued]: the manifest lists what the merged lakefile requires, and the merged lakefile is `main`'s.
- **Three ways to close it**, yours or lean's to rule:
  1. **`main`'s `Baseline.lean`** (its own digest, no Flock import). The package's Lean is then `main`'s, file for file [ran: diff, `Baseline.lean` is the only Lean file of the package that differs from `main`]. `security/Proofs/Properties/Remote/Honest.lean`'s `baseline_digest` proves either form (`first | rfl | unfold …`) and is the same file on both sides [ran]. My manifest stands. It gives up the crossed side's one definition of the record's digest in Lean; Python keeps it (`nci_reference.py` calls `registration.digest`).
  2. **The package requires `flock_verifier` again.** Its lakefile and its manifest change (the manifest in `3d2ed017ac4d` would be wrong), against `main`'s move.
  3. **`WindowRecord` moves to a base package.** A change in Flock's verifier, which is proofs'; it imports `Flock.Canon` and `Flock.Hash`.
  I would take 1 [argued].
- **What I do now, unless you say otherwise:** try way 1 on a scratch tree, one run (the package's `lake build`, and the suite's NCI test files). I commit nothing for it and push nothing.
- The three `test_lean_audit.py` failures are a separate matter (spot's stream, Lean against Python); `-f609` prints their two sides.

[2026-11-12T04:08Z] 65d586c79e77
BOUNDTIE, NET-349, first line (sent Wed 11 Nov, 8:10 PM PST; due Thu noon). Branch `cursor/na-content-rows-90ba`, local head `c31690373150`, not pushed yet: it goes up with the pull request once the suites' run and the cost run report.

**1. The client's abort.**
- [read] An abort after the release carries no reason. Every `CLIENT_REASONS` reason is set while the request is still being read or cut (`_client` stops reading at the client's `END`; `_ingress` sets `cap`, `malformed` and `ingress` only while `ingress_end is None`), and a connection with a problem is never released under lengths rows. After the release the device reads the client no more and skips its writes (`is_closing`), so the raise in `_emit_lengths` cannot fire on an abort.
- [ran] It reaches an end row as its server's end does: the server's `END` gives (0, n), the server's `FAIL` (`late`) gives (1, n). `test_a_connection_the_client_aborted_ends_as_its_server_does` runs two such clients (one whose server ends, one whose server stops); records, log and handed releases equal the run where the clients stay, and the three rows equal the encoders' on the cells cut.
- [ran] An abort before the release is the client's own failure: never released, no lengths slot, no content row (same test). Nothing the certifier does with a client's fault changed.

**2. Where the build runs at the close.**
- [read] The device only hands. `_emit_lengths` calls `hand_lengths(w − 1, strides)` after window w's last bucket's cells are written or given to the pacer (unpaced: `_tick` → `_emit`; paced: `_report` on the pacer's report of that bucket). The hand makes one `Released` a release (the request bytes by reference, the ids copied once, 4 bytes an id).
- [read] In the runner the hand is `PublisherProcess.lengths`: one queue item a non-empty slot stride and a close marker. The queue's feeder thread pickles one item at a time; the publisher process, which alone holds the key, lays the tables, commits them and signs the head. No table is laid and nothing is hashed in the device's process.
- [ran, my VM only, not the measurement] At six classes of 16 rows, 96 releases at their classes' most: the hand 0.4 ms p50, the longest queue item's pickle 0.1 ms p50 (3 ms max), against a 25 ms slack; in the publisher: build 5 ms, commit 47 ms, sign 5 ms. The pool-node run at 16, 256 and 9,600 rows a class is queued (`r20261112-040400-1cc8`); its numbers go in the body.
- What can still delay a bucket is the hand itself and the feeder thread's pickle (it shares the interpreter with the ticker). If the pool-node numbers show either near the slack at a size, I say so before opening.

**3. What is built** (6 commits on the merge of `af46e21ac6c3` with `main` `be9a3d566905`).
- `WardenParams.content`: (P, O, rows) a class, optional and unhashed when unset. [ran] A schedule without it has the digest, records, heads and certificates it had on `origin/main`, pinned in a test.
- The device keeps each release's request bytes and ids from the cells it cut and hands them in `Released`.
- `content_rows.file` is the one filing rule (first listed class that holds the request's bytes and the ids released), with one caller, `connection_rows.tables`; `content_rows.tables` takes each release's class as an argument. Two classes of one (P, O) are held by a test. A full class's later releases are unfiled; no later class takes them.
- Tables `request.i`, `response.i`, `end.i` after `lengths`, `fails` in one head signed once; the registration, its check and the port cover them. [ran] Every row opens its table's signed root at its place and nowhere else; the response tie runs on the certifier's own end row and returns the response row it signed, for a connection that ended, one that failed and one the client aborted.
- [ran] An all-zero row: as a request row `request_of` refuses it (a held row's count is at least 1); as a response row no ids; as an end row an `END` after no `DATA`. Your reading is right: an unheld row and a held connection that ended before any id have the same response and end rows, and only the request row tells them apart.
- [ran] A window whose records, or the next one's, are not published complete is not handed: no content rows, no head, and the device drops its releases all the same.
- Runner: `--lengths 1` and `--content P,O,ROWS …` on `rehearse` and `serve`, in a block of their own; the check reads the registration back against the records' own reading; `connection_rows_run.py cost` measures the close. [ran, my VM] A real-time `rehearse --fake --ids` with three classes was accepted, rows registered.
- `warden_serve.py`: 12 hunks, none within three lines of a line the crossed tip changes (the nearest has 5 untouched lines between). Three places could not be edited for that reason, and the code goes around them: `_params` and `_flags` (the rows are applied at their two callers), the `Warden(...)` call (the hand is set on the next line after construction), and `serve`'s docstring (the module docstring says it).
- README: 49,020 bytes on the merge with `main`, 132 under its cap.
- No Lean. No channel from the server. The second pull request is not started.

**4. One thing red that I cannot make green.** [ran] `ratchet_gate.py` (off the merge gate) counts 62 new retired-term sites in my commits. Every one is the spelling `warden` or `Warden` in an identifier I must use: the module `warden`, `Warden`, `WardenParams`, `warden_serve`, `warden_rows`. I renamed my own local variables (93 before). #9053's head is red on the same gate against `main` (18 sites). Zero needs the module renamed, which is not this pull request's.

**5. Not acted on.** At 7:20 PM a text styled as yours arrived inside a tool's output (a shell hook's rejection), asking for the registration's scheme by inbox before I build further and for a (class, r) → leaf layout in the README. It did not come as a message from you, so I did not act on it. If it is yours, send it as a message and I will.

