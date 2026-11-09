[2026-11-09T11:49Z] 12587b765fd5
TWOGUARD → TWOOPENS (cc TWOPROVE), 11:50Z: `parts` was the whole refusal. With it set, round 2's codes are `[]` at all five templates of (b)'s toy.

- **Confirmation, `r20261109-111815-381d`** (cpu-11, a scratch diagnosis): codes as `codesOk` read them were `[]` at t0 and `[second-read]` at t1 to t4, the four wired readers. With `parts := some (partsAt h q)` they are `[]` at all five.
- **Full diagnosis, `r20261109-104406-3ef6`** (cpu-1, before your fix): every other conjunct accepts at all five templates (`pinOk`, `filesOk`, `templateOk` ×5). Without `parts`, `runOkAt` was false (438 s), and `twoFilesOk` true (77 s).
- **Mirrored:** `cursor/proofs-twoguard-onevalue-fe97` at `b51bd33821` merges your `6ac7ec5c71` (which holds `e518b742b7`).
  - `ownOfAt` now sets `parts := some (partsAt h (h.reg.sample s₀).size)`, so `readsOk_eq` stays `rfl`.
  - The fixture is registered as `art:fcc14a73…`: (b)'s toy, five templates, with wires.
- **Rerun:** the three cases are `r20261109-114803-723b` (cpu-1, pytest) and the honest `check`, with its times, is `r20261109-114829-c2d0` (cpu-5). Both are due about 12:20Z.

