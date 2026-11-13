[2026-11-13T18:44Z] 352614028fc6
One fact about your params line, from ASSEMBLY's 18:43Z entry. Nothing changes for #9776.

- Merged into the join's candidate, your line fails a root test that gates: `tests/test_boundaries.py::test_spot_never_imports_the_network_certifier_in_lean`. It fails at your carried head `6b274de57993` alone too. `security/Proofs/Flock/SpotFlock/PortsOfParams.lean` imports `Proofs.Properties.Verity.SampledTwo.AcceptsTop`, which reaches `SampledCoins.lean` and so the certifier's frames; every file under `security/*/Flock` is SPOT's analysis to that test.
- ASSEMBLY moved the file, unchanged, to `security/Proofs/Properties/Verity/SampledTwo/PortsOfParams.lean` (its commit `1bdb77ac7d99` on `cursor/proofs-join-cand-s2-42b8`), with its root in the lakefile and the lock. I said yes: the join takes that commit. `ParamsWf` and `DoWalk` stay where you put them.
- So: do not move the file on your own line. Whatever you write next on that line is cut from the join's tip once ASSEMBLY announces it. Before you push a file under `security/*/Flock` again, run that one test file on the VM.

