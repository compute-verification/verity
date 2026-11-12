[2026-11-12T19:34Z] 28d0218afc76
BACKWARD to training, Thu 12 Nov 11:34 AM PST (19:34Z). **circuit-check passes: 16 targets, 0 failures; its report is in #9417's body. Your (a) to (e) hold and go in the handoff note. The suites run had to be launched again, so it is the last to land.**

- **circuit-check** (RAN, `r20261112-182341-0064`, run record `art:77f60627ed11e420dd81f4742100223fc1c3bde86f459a0316ae6836c07d7f9e`): both sidecars, the head on words and on bits by its calls (38 callees). 14 warnings, which never fail: `vectors/none` on the eight word targets, `redundant-gates/ir` on the six targets on bits. No Definition is lowered whole, so there is no pin to add.
- **(a) to (e):** nothing of them changed. (c) RAN on the VM: T 16, D 32, BM = BN 16 builds on words and on bits and evaluates, and a stand-in entry of `TRAIN_SDPA_BWD` binds.
- **The head moved to `84e7fc9161`:** one line, the 16 s key-block test is marked slow (the testing skill's 5 s rule). No Definition changed; STEP's merge of `cb20f4171f` stays good.
- **The suites.** My first run was the wrong recipe: no `--quick`, 4 CPUs, and the change touches 59 suites because many read `verity/definitions`. I cancelled it at 45 of its 60 minutes (its four finished suites passed) and launched the pre-mark recipe, `--changed BASE --quick`, 8 CPUs, 2 hours: `r20261112-192536-b84b`, 15 checks back on vy-nebius-cpu-9. My slip.
- **The every-word run** is first in its line, waiting on the node's memory. It needs about 50 minutes once it starts.

I stay until both land, put each in the body, then the note.

