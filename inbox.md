[2026-11-11T01:28Z] 5b14b48817da
ONSITEMAIN, NET-327, state, Tue 10 Nov 5:28 PM PST. I wait on one run: `r20261111-012503-b840`.

**(c): 66 files, seven items, the base's side in each.** Whose, by the registry [my matching]: 36 ours, 30 not.
1. Period check: 2, ours.
2. a5 and the v2 case: 10, none ours: proofs 7 (`benchmarks/one_stage/`), vllm 3 (`verity/ml/windows/`).
3. The service: 2, proofs' (`spot/`).
4. The ingress tie: 8, ours.
5. Drivers: 12, ours but `tests/test_retired_terms.py` (top).
6. Composites: 6, proofs' by the registry.
7. Security: 26: ours 15, lean 10, proofs 1.

**Your message of 5:18 PM PST reached me after the merge commit was pushed.** So there are two branches:
- **Read this one:** `cursor/net-onsite-port-e83a` at `5d82c83ee4b7`, two ordinary commits over the base `f63a7b76334d`, each naming `00d79b22e4d2` and its files.
- Superseded, not rewritten: `cursor/net-onsite-port-eebd` at `e7cda23b3919`, the merge commit. Same tree, byte for byte [read: both tree hashes]. Do not merge it.

**Carried:** six files, the rows audit: `NetworkCertifier/Rows.lean`, `NetworkCheck/RowsAudit.lean`, `RowsMain.lean` (by `modules.py`), the two roots' imports, the `network-rows` executable.

**Independent of (c) 1:** of `Period` it reads `Schedule`, `Schedule.digest`, `Schedule.allocation`, `scheduleOf`, `records`. Staging changes none; none is one of your four. [read] It binds the base's digest, which has the egress form in it.

**Nothing calls or tests it here**: its readers on staging are (c) 5's drivers. The run shows only that it builds.

**Runs:**
- `r20261111-011715-8cef`, first branch: cancelled by me while waiting; nothing built.
- `r20261111-012503-b840`, at `5d82c83ee4b7`, cpu-3, waiting for a slot: both Lake packages, then the network and certifier suites. No PoUS suite: my cut. Expect exit 1 from the base's own two `compile_slice_v6` failures. [read: LEAN-249]

**Not done:** `verity lean try` (nothing carried is under `Security/`); no lock; the base's new head was not named at 5:17 PM PST.

**Your question:** I expect the maps to fix names in most clean files and to turn most of (a) and (b) clean, but not to make more files carriable: each clean file I read is stopped by a (c) file it reads, not by a name.

**Next:** read the run; one fix if the first error is in the three new files; merge the new head when named, unless it holds the reorganization.

**Least sure:** whether the audit elaborates (names checked, types not); whether it is worth carrying alone; the sort is by text.

**Note:** `lanes/network-accounting/20261111T0118Z-state-net327-onsiteport.md`.

