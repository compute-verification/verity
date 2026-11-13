[2026-11-13T07:05Z] cd0e88c91a72
WATERLINE to network-accounting (lead), Thu 12 Nov 11:06 PM PST. NET-418 is built and opened.

- **Head** `6f6f87036476ff8ea3b13a4ace62bf32fccd7b3b`, branch `cursor/na-watermark-path-refusal-a989`, base `main` `fa0877afdc12`. [#9685](https://github.com/compute-verification/verity/pull/9685), unmarked, ends `Refs NET-418` [ran].
- **Run** `r20261113-053146-1343`: exit 0, 61 suites passed, none failed, none cached, on the head's own base [ran].
- **`main` moved under it:** #9333 landed and changes `warden.py`. Merge with `main` `2e99b1cf48b8`: no conflict, the four certifier test files pass (148), `main`'s ratchet gate passes [ran]. No suites ran on that merge. #9369, #9378, #9457, #9491 (now `37716d4ba990`) and #9541 each merge on top: no conflict, the files pass [ran].
- **Control:** on `main` the certifier forwards a client's copy as release 1 and both endpoints lose honest release 2; at the head it fails `malformed`, no cell is cut, neither loses any [ran]. The model is a stand-in.
- **Correction to my read:** the gate loses the named releases only when the copy's lines are stepped to its `END`'s bucket; serve-openai's frontend loses them either way [ran].
- **Chosen, not ruled:** commit `6f6f870364` reads the request whole at its `END`: three long request lines pass the first-cell read alone [ran]. Separable, if you want it dropped.
- **Found, not fixed:** `with_release` passes unchanged a request whose first line it does not parse, client's release headers kept; a second request on one connection [ran].
- NET-418 has no goal label, so no Goal line. The hour is on NET-418, with `--worker` alone: the tool refuses `--as` beside it [ran].

TIMER: none

[2026-11-13T07:07Z] f34726e5ea03
NETSEATS to network-accounting (lead), Thu 12 Nov 11:07 PM PST. **The chain record on the new layout exits 3: the leak guard holds 130 of the chain's files. So by your rule #9656 waits on a ruling, and its first screen now says so. The move of `tunnel.log` is in, at `0e2bdc0c79e6`.**

**1. The verdict you owe PRF-611: run `r20261113-050508-9400`, at `fce58a7ec8ca`** [ran]

- The dry session ended 0, its own guard clean.
- The chain's three proving steps ended 0 (`n2`, `scan`, `tie`).
- The leak guard then refused, and the job exited 3: 1,311 files read, 363,287 witness pieces of the one gate run, 130 files moved to the quarantine, 1,166 of 1,296 released.
- The 130 held files:

| Step | Files | What they are |
|---|---|---|
| certificates | 1 | `certs.json` |
| report | 1 | `report.json` |
| scan | 64 | the statements' registered files (28), registrations (8), audits (8), the two row registrations whole as the verifier takes them (11 files), the Lean verifier's logs (8), `answers.json` (1) |
| tie | 64 | the statements' registered files (32), registrations (16), outcomes (16) |

- This is what the body's control predicted: with the registration's body in `witness/` alone, every file of a job that repeats a root, a domain or a wiring is a find. Before this change the guard passed them because `out/cells.jsonl` made those bytes public.
- **Which fields the 130 repeat I have not read yet.** The guard's report in the quarantine names each find's key path. A small job reads it by counts and key paths, no bytes: run `r20261113-070253-0f3c`, waiting in the pool's queue. Its table goes in the body and to you when it ends.

**2. What the ruling has to settle** [argued, until the key paths are in]

- **A registration's `root` and `domain`.** If the root is public as a hiding commitment (your key's line) and the domain is structure, the statements' files may hold them, and the guard has to be told: either the gate publishes those two fields beside the certificate again, or the guard's declared data excuses the pieces cut inside `registration/root` and `registration/domain` of the two witness files. The second is an allowlist entry by key path, which NET-417 forbids as written.
- **The registrations whole** (11 files) hold `wiring`, which is ruled private. They are the verifier's inputs. The chain has to keep them on its private side whatever the root's ruling: a change to the chain, not to the gate.
- Neither is in #9656. Until one lands, no chain record passes on this layout, at any head.

**3. #9656 now**

- **Head `0e2bdc0c79e6`**, one commit on `861523cbd3c6`, pushed, unmarked. The move is `0e2bdc0c79`: a relay rehearsal's `tunnel.log` is written in `witness/`; the end names it among the witness files and no longer hashes it.
  - The gate's test runs the rehearsal through a tunnel that says one of ssh's lines before the rehearsal starts, and holds `out/` and the end to none of it. Control: on the gate as it was, 9 tests of the gate's two files fail [ran].
  - No test reads a clock for it: the test's `up` waits on the tunnel's opening of a FIFO.
- **Runs at this head:** the touched suites `r20261113-065216-d3e2` (running), the dry session and chain record `r20261113-065249-3094` (waiting in the pool's queue). It runs the same chain, so I expect exit 3 again [argued].
- **The suites at `8d8732d7a05b`,** run `r20261113-055230-e108`: 85 suites, 85 passed, none cached, none failed; exit 0 [ran]. The chain record there, `r20261113-055304-d370`, is still running.
- **The body's first screen,** after the summary: "Not to be marked before the chain record at this head has its verdict here." then "The chain record ends 3 on this layout, so this waits on a ruling." with the verdict above under it.

**4. Two things the move leaves, for your word**

- **A certifier rehearsal's `out/tunnel.log` stays in `out/`.** The gate writes the log where the run writes, and that run kind has no `witness/`. The reasons that moved the relay's hold for it too. Moving it means giving that run kind a `witness/`, which changes its fetch and makes it a gate run to the leak guard: not one line.
- **A challenge session's `out/` holds three free-text logs** (`tunnel.log`, `serve.log`, `coin-link.log`; `serve.py:238`, `auditor.py:479-480`), and its fetch gives `out/` (`auditor.py:1327`) [read]. I did not read what they print. Not a certifier run's output, so outside NET-417 as written.

**5. The cost of the two things not checked** (nothing built)

- **The Python's key bytes against the Lean's: the test exists and no check runs it.** `flock-verify hm96-test VECTORS sha512` compares `Hm96.sha512.keyFromLabel` with the vectors file's `default_key`, the leaf prefix, every mask and every leaf (`verity/protocols/flock/verifier/FlockVerify.lean:1475-1493`, `:1986`). The vectors file is written from the Python's `default_key` (`verity/commitments/hm96/vectors.py:88`), and the proofs' κ₀ is that same Lean key. A grep finds the command only in a README [ran]. Cost: one command on a built verifier, a pool job of minutes; as a standing check, one line where the Lean verifier's own tests run.
- **A run's files read for a reused salt:** about ten lines in the dry session's launch script (count the 192-byte salts of the kept `witness/` and how many are distinct, write the two numbers), one pool job of a few minutes, no tree change. It reads one run; it cannot show that two runs share none.

**Also in the body:** each of the certifier rehearsal's and the store write's `rehearse.log` in one sentence with [argued], as you ruled; the end's three totals named under what this does not close, with nothing changed for them.

