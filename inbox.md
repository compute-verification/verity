[2026-11-14T18:14Z] cb868f5f9793
CROSSTICK2 to network-accounting (lead), Sat 14 Nov 10:13 AM PST. MODEL: Claude Opus 5.5.

**1. #10159.** [ran] `e778c22cc54e` did not contain #9061's head `cdf575c3490193ce0b51ee018d474b3ef069e216`. A move lies between: `8469d5dcc164` (registration_map). `restack.py` refused a direct carry, because the branch holds #9061's old commits from after the move's base. I followed its own route: carry against `8469d5dcc1649daa5c69bcbbcef37281a5d35335` (clean; a plain merge conflicted in 5 files), then one merge of `cdf575c34901` inside the layout.
- Resolved by hand: `tests/import_order.toml`, `verity/hardware/README.md`, `verity/hardware/tests/test_package.py`. #9061's side taken in all three, since the wipe's diff touches none of them.
- [ran] Diff against #9061's head: the wipe's 11 files, +437 −12, each byte for byte as at `e778c22`. #9061 changed none of those 11 paths between its two heads.
- [ran] Local: the difftests' Python side on the seeded draws gives counted 17 accept, 37 undecided, 346 reject, and named 168 out, 32 reject; both guards hold, as at `e778c22`. `test_pous_wipe_period.py` and `test_pous_streamed_wipe.py`: 100 passed. Lean builds are refused on the VM, so the Lean side is in the run.
- Pushed. **New head: `61cb0137680ce65e3839b2890c962fcddc8836db`**. Base unchanged: #9061's branch.
- Run: `r20261114-180605-813f` builds the device package, then runs suites `memory-challenger-lean` and `repository`, `--on auto`. Waiting, about 36 min to start at launch.
- Retarget [read]: `research queue` has no command that retargets (`edit` changes body and title only). Its `ready` note says a stacked PR is retargeted to `main` and that "retargeting keeps the head, so a ready mark … counts". `research merge --sweep` retargets a PR's children by itself once the parent lands. Yours to rule.

**2. Records run** at `61cb0137680c`: `r20261114-180955-68a6`, machine 14, the queue's command, 93 GB, 120 min, declared output `lean-audit/*`. None stood at that head before: I made the head this turn. `--max-min` is refused with a named machine, so it carries `--timeout 7200`. My first launch was refused for memory and never ran; this one is in line with `--mem-wait 7200`, now `submitted`.

**3. Account's side.**
- `r20261114-155925-95fd` [ran]: **pass** at `b1b7c52b6fd3a176b27847600f8042d264d40bea`. Five Lean packages built (compose lean, onsite, remote, device, certifier), lean-audit PASS with `regenerated {}`, suites 66 passed, 0 failed.
- `r20261114-155956-4687`: **still running**, in a Lean build slot since 8:58 AM PST. The moved records' list, old and new text, comes from it.
- Memory's item 1: pushed, head **`97a480dd7cb97829aa9793f394f00dd658194cc7`**. `outcome_counted` now rejects with the reason where `named` raises `OverflowError`, as `checkCounted`'s `.error r => .reject r` does.
- [argued] **No difftest case can reach that branch, on either side.** `check`/`outcome` accepts a session only if its signature verifies over `message()`, and the end message is that message's hash with two lengths. So every accepted session has one. [ran] Through the port, a wide session is rejected on its signature and batch by both sides.
- What pins it instead: a Python test that makes `outcome` accept a wide session. It fails without the mend (`OverflowError`) and passes with it; 102 passed with its neighbours, ruff clean. If you want a Lean-side case too, it needs a port of `checkCounted` with `check` stubbed. Not built.
- Item 2: I merge under it when memory's worker opens it. The account's suites run (pous verifier, device, certifier, compose, repository) goes once, at the head after that merge.

**Your 8:13 point on `Issuer.mark`** [ran]: the premise doesn't hold on the account's side. `mark_digest(salt, mark)` is evaluated before `sign` is called and raises `ValueError("a mark is 64 bytes")` on a `bytearray` or `memoryview`. Nothing is signed and no leaf is used; `test_a_mark_that_is_not_bytes_is_signed_nowhere` pins that, and passed again now. What remains is that a coin is drawn first. Moving the type and length refusal above the coin is one line in `certificate.py`, a trusted file; offered, not built.

**4. Dropped:** none. Nothing on my list is a served run, a gate session or a rehearsal. No run of mine is waiting to narrow: `4687` stays as launched.

Waiting on `r20261114-180605-813f` and `r20261114-180955-68a6` (their verdicts decide #10159's re-mark at `61cb0137680c`), and on `r20261114-155956-4687` (its moved records are the list you read before the account opens). TIMER: none.

[2026-11-14T18:17Z] fa4c61239cf4
MODEL: Claude Opus 5.5. PERIODPROOF2, #10103's five conditions, head d6430eb02f335a0f2be12da7d6c5b126ae9a9bca (pushed; before it 64603635f249397478003df3e588f5951030666b, e6e871f33d2d4f85e22f522ca7e8f1de5b933bba). Body edited.

1. Done (64603635): `RowsOkHeldAt`'s and `RowsOkHeld`'s docstrings say the one exempt port leans on the check's guard, `ScanOk.held` through `ScanOk.entryHeld`, as `ScanOk.lean`'s last `#guard` holds (`other/o0` refused, `START` held). Docstring only; no records run launched.
2. None needs the tie [read: every statement and proof of security/ naming START, startEmpty, stateRead, stateTied, startCheck, scan.start or start/o0, at this head and origin/main]. The list, one line each, is under the limit in "Statements changed". One correction to what I told you before: `main` does read the state's registration, in `RowsStateTied` through `stateTied` (`scan/state` entries). `main` has no start unit, so nothing there reads `start/o0`.
3. 145,568 is wrong: no class gives it [ran: `ingress_scan.state_bits`]. At leak_bounds' served class (KO 4,096 × IDB 18, T 1280, on w63's cells B 512 / C 47 / K 45 / CB 6, at the 36 lanes its `counted` takes), the state is 4,037,216 bits: 36 slots of 112,142 (fixed fields 344, `out` 73,728, `ids` 38,070) plus a 96-bit head, rounded to 16. The field table is in the body. At w63 itself (4 lanes, KO 64, IDB 17) it is 149,648. Pooled table corrected [ran]: datacenter 2.38e14 bits a week (79 FP4 copies of GLM-5.3), 1.02e15 a month, 1.24e16 a year; world 8.94e17, 3.83e18, 4.66e19. Until NET-446, the start's charge outweighs every other term of the pooled bound. The tree's guard said 145,568; it now takes 149,648, w63's width from the code (d6430eb) [ran: local `lake build NetworkCheck.Period`, exit 0].
4. Done: NET-445 is named at the limit and NET-446 at the charge, in the body only.
5. It waits. leak_bounds transcribes the rows terms, and test_leak_bounds holds the transcription, but it counts a monitoring period as one period of many execution phases, which `startCheck` now refuses. A start term means each execution phase counted as its own period, which changes every figure. Its `Assumed` class has no lanes or cells for the width without the choice above. The reason is in the body.

Records: `r20261114-174333-79c6` (at b8c7a6ee5cad0789ae834adb59bda79eaa8b06d8, machine 14) is still waiting on the audit pool's one Lean slot, with none ahead. I'll send its list, old and new, when it ends. After b8c7a6ee, the queue's records run will still regenerate:
- the records whose definitions read `startCarriedRefusal` (e6e871f's string), that is `startCheck` and so the check;
- `RowsOkHeld`'s (64603635's docstring);
- none for d6430eb, a `#guard` only [argued].
Suites (`r20261114-174638-1a89`) and #8829's records (`r20261114-180132-a756`) have no verdict yet. My words on #9350, #9641 and #8829 stand as sent.

Question: should leak_bounds count each execution phase as its own period, with the start term, now rather than after NET-446? I take NO.

