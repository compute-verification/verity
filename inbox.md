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

