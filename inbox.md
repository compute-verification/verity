[2026-11-14T20:48Z] c5c6233236b9
## Coordinator, Sat 14 Nov 20:48Z: notice: quiet window on the certifier's files until 22:30Z

From the proofs lead: top declared a QUIET WINDOW on the network certifier's files, from 20:50Z to 22:30Z or until network-accounting says on TOP-424 that its first step is on the branch, whichever is first (its integrator is merging a 2,800-commit branch that stops in about 150 files, mostly the certifier's). Until then push NOTHING into `architecture` that changes a file under `verity/protocols/accounting/certifier/`, `security/**/Certifier/**`, `security/**/NetworkCertifier/**`, `benchmarks/network_traces/`, `benchmarks/network_e2e/` or `benchmarks/prove_path/`. If a rename map of yours reaches those paths, hold that push (or exclude those paths from this run and rerun the map over them after the window); pushes that touch none of them go on as usual. Say in your next note if you held something.

