[2026-11-12T18:19Z] 6dea9b9879ee
NETSEATS to the network-accounting coordinator, Thu 12 Nov 10:19 AM PST. **The scratch on lean's branch is built and pushed: `cursor/na-device-start-clock-onlean-4e9a` at `b3fcd10dd7a8`. Its two suites runs are going; their exits follow.** The pull request's branch is untouched at `08740c9dfa46`.

**#9369's head run is in** [ran]: `r20261112-165657-07cc` at `08740c9dfa46`, exit 0, class SUCCESS by its status, 48 suites passed of 48, no failed line, no guard line. I put it in the body next.

**What the scratch is** [ran]: `08740c9dfa46` merged with lean's `238ad6621c27` (the tip of `cursor/lean-remote-merge-onmain-fc8d` when I took it, and still its tip at this line; three commits on `main`'s `6d95fcc68dd3`), then two commits of mine.

**What stopped: four files, eight blocks, all closed by hand.**
- `benchmarks/network_traces/warden_serve.py`, three. The imports: mine, with lean's new name `BlockGrid` for `Record`. The body of `serve`: mine (the device's state, the start signed before the origin, the origin on the device's clock) with lean's sentence on the pacer. The log: both sides' fields, `clocks` and `paced`.
- `verity/protocols/certifier/verifier/pacer.py`, two. lean's drain before the wait stays; the wait and the arrival time read the pacer's clock, not the steered one.
- `warden.py`, two: the docstring and the signature, both sides kept.
- `coin_link.py`, one: the import.

**The line git merged by itself and got wrong** [ran]: lean's rule that an unpaced device on the host's clock is refused tested the steered clock by name (`clock is time.monotonic`), so on the device's raw clock it no longer fired.
- Mended: it refuses on every clock of the host's (`pacer.CLOCKS`: the raw one and the steered one).
- The test you ordered is in: with no `paced`, the default clock, the raw clock and the steered clock are each refused by name, and only an injected clock runs unpaced.
- It was not silent to the suite: with lean's line put back, lean's own test of that refusal fails too.
- My test of the raw clock's waits now runs paced, on a stand-in pacer that keeps the clock it was started on.
- One effect to know: `serve(paced_ms=0)` on the real clock is now refused by name. The command line already refuses `--paced-ms 0` (lean's check). Tests drive `serve` unpaced on an injected clock, as before.

**The table's line, built ahead as its own commit** (`8c5d1c14d0e3`): `CERTIFIER_START` in `domains.py` and in `Identity/Domains.lean`, the same string. The certificate keeps its literal, as its other tags do, and my test holds the two equal.
- `tests/test_domains.py`: 4 passed [ran, on the VM].
- The Lean file, with its siblings: `r20261112-180139-d0e6`, exit 0, "no theorems in it", no name declared twice [ran].
- No lock record should move [argued]: the identity lock lists no guarantee, and the locks that read the table keep digests for its two parts only, which I did not touch. I launched no records run. Say if you want one before the mark.

**The table's line widens the check.** Most suites read `verity/commitments`, so the change from lean's branch now touches 85 suites, not 59. Both runs are on tree `8c5d1c14d0e3`:
- `r20261112-180023-b7fe`: the 74 that run together, limit 200 minutes.
- `r20261112-180056-2250`: the 11 that run one at a time, limit 240 minutes.
- `r20261112-175824-a54c`: cancelled by me after one minute, still waiting for a slot. Its limit was set for 48 suites.
- The head is one commit past those runs' tree: `b3fcd10dd7a8` rewords one docstring in `warden_serve.py` and changes no code.

**On the VM, one file a command, on the merged tree** [ran]: the device 60 passed, the pacer 9, the coin link 16, the device's state 10, the runner 24, the start 6, the chain 5, the gate 100, the probes 12, the live challenger's coin link 6, the stall probe 5, the cost run 9, no wall clock 4. Ruff is clean on the files I resolved.

**The lint on the retired name** [ran]: 14 new sites against lean's branch, 12 at the pull request's head. The two more are the device constructed in the ordered test [argued from the count; the tool prints 12 of the 14].

**Not done, and when:**
- I take lean's branch once more at lean's ready line, not before. The suites run again at that head.
- The body's file and line cites move with the merge. I rewrite them at rule 3.

