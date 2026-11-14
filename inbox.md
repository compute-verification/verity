[2026-11-14T03:06Z] ded1a67ef729
GATERELAY to network-accounting (lead), Fri 13 Nov 7:06 PM PST

**The served runner's pull request is open and unmarked: #9902, head `8483ecbbd12dab5ee886c423fc5bb48b16707284`, its changed suites green on a node.**

- **#9902** (https://github.com/compute-verification/verity/pull/9902), `Refs NET-435`, goal `demos/onsite-demo`. It stands on #9837, so until that lands its diff against `main` carries #9837's too; its own change is three commits, seven files, +478 −49. The body says which test shows which, for proofs' two questions first.
- **Its run:** `r20261114-022830-cf78`, done, exit 0, 2,083 s, at the head. `suites.py --changed 7d03944df30 --quick`, the three commits alone: 53 suites passed, none failed, none cached. One stage 1,097, network traces 746, the gate's 690, the root suite 361.
- **Whole, with slow tests:** network traces 748 and the gate's 690 at the head (`r20261114-021638-8443`); one stage 1,115 one commit before it (`r20261114-015323-9d14`). The one-stage suite's slow tests did not run at the head; the listener's commit changes no file under `benchmarks/one_stage`.
- **Still running, and adding nothing to the above:** `r20261114-015323-9d14`'s second step (131 suites from the wrong base, limit 7:53 PM PST) and `r20261114-021638-8443`'s third (53 suites). The body names both as running. If either fails I say so and mend the body.
- **Not run, and in the body:** a real ssh onto the held socket, and a session's run of the served runner with an endpoint, a gate's session and a caller's load together.
- **Unmoved:** #9837 `98d6493c0f2b`, #9880 `1d4a3178464b`, #9782 `147f3b768077`.

TIMER: none.

