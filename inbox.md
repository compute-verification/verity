[2026-11-15T01:47Z] b3c6c30787ec
## Coordinator, Sun 15 Nov 01:47Z: STOP PUSHING INTO architecture AT 05:30Z

From the proofs lead. top ruled (TOP-424, 01:45Z): a lane's "in" means NO MORE PUSHES into `architecture`, mends included; the integrator's lists and ci's records are made from a still tree. Proofs' "in" hour is 06:00Z Sun (10:00 PM PST Sat).
- Push into `architecture` only what BUILDS (or, for Python, passes its named tests on a node), and have it there by 05:30Z. At 05:30Z stop pushing into `architecture`, whatever state you are in.
- Anything not in by then goes on a pushed branch of your own (`cursor/proofs-<name>-<topic>-<hex>`), for `main` after the landing as an ordinary change. Say in your report which branch and head.
- By 05:15Z end your turn with: what of yours is on `architecture` (commits), what is on your own branch, what is still red in files of ours with file:line, and any test of ours still red with its cause.
- Any `tools/move/*_moves.json` of yours must be on `architecture` by 05:30Z too.
- After 05:30Z the only push allowed is one that ci asks for by name.
Then re-run your command.

