[2026-11-12T06:44Z] fc99ee817972
LEANMERGE to network-accounting (coordinator), 10:44 PM PST Wed 11 Nov. NET-354, pull request 9154: the second commit is pushed.

**Head:** `df61accc41bbc5c925111075bd5bb6c2763be6c4` on `cursor/net354-certifier-mark-5d2a` [ran: `git ls-remote` shows origin's branch at it]. Two commits on `main`'s `62bf29346f76`: `5181e37f0973` (the mark certificate, unchanged) and `df61accc41bb` (the bytes). CROSSTICK can cut on it.

**What the second commit holds** [ran: the test file alone, 27 passed locally]
- `Certificate.encode() -> bytes`: `u16be len(I) ‖ I ‖ u16be len(link) ‖ link (UTF-8) ‖ u8 direction (0 egress, 1 ingress) ‖ u64be execution phase ‖ digest (64) ‖ u32be len(signature) ‖ signature`. It raises `ValueError` on a certificate the layout cannot hold: a field of another type, a key id or link past 65,535 bytes, an execution phase that is not a u64, a digest that is not 64 bytes.
- `Certificate.decode(data) -> Certificate`, a classmethod, the inverse. Its refusals, each a `ValueError` whose text says which:
  - a short buffer: `a certificate's bytes end inside its <part>: ...` (the part is `key id's length`, `link's length`, `direction`, `execution phase`, `digest` or `signature's length`);
  - a length that does not fit: `a certificate's <key id|link|signature> does not fit: it is N bytes by its length, and M are left`;
  - trailing bytes: `N bytes are left after a certificate's signature`;
  - also `a certificate's direction code is 0 or 1, not N` and `a certificate's link is not UTF-8`.
- `service.py` and `lean_audit.py` are not edited [ran: `git diff --stat` names two files, `certificate.py` and its test]. `CertifiedMark` gained nothing. Nothing in Lean; `Domains.lean` did not move.
- Three tests: the layout restated with `struct` and the inverse, for a record's certificate at each end, a clock's and a mark's; the same bytes inside `service.encode`'s leaf (after the head and the 64-byte salt, to the end) and inside `lean_audit.encode`'s (kind byte 2, u64 length, then that leaf); each refusal by name, and every proper prefix of a certificate's bytes refused.
- A control I ran outside the tree [ran]: with `encode` writing the execution phase before the direction, all three tests fail; with a `decode` that drops trailing bytes, the refusal test fails.

**One thing that differs from the two encoders, for when they adopt it:** `encode` refuses a digest that is not 64 bytes; `service.encode` and `lean_audit.encode` write whatever `c.digest` holds [read]. No certificate the issuer signs has such a digest (`sign` builds it from a SHA-512), so the bytes agree wherever both give bytes.

**The merge with lean's crossed tip** [ran: `git merge-tree --write-tree --name-only origin/cursor/lean-remote-merge-crossed-fc8d HEAD`, the tip at `ae73c594d338`]: it exits 1 with 54 conflict lines, none of them in my four files. `certificate.py` and its test are listed as `Auto-merging`; `domains.py` and `Identity/Domains.lean` are not named at all. In the merged tree both Python files parse, hold no conflict marker, and carry `BlockGrid` on the grid import and `Issuer.certify`'s signature beside my `encode`, `decode` and `mark` [ran]. The conflicts are the crossing's own (Lean files and lakefiles), not mine to look into.

**Ratchet** [ran]: `ratchet gate 62bf29346f76..df61accc41bb: 1 change(s), 0 failed`. Per file, base to head: retired terms `domains.py` 44 to 44, `Identity/Domains.lean` 36 to 36, `certificate.py` 10 to 10, the test file 1 to 1; lane text 0 to 0 for all four.

**Runs launched on `df61accc41bb`, each once, `--on auto`:**
- `r20261112-064235-56b3`: the certifier verifier's suite (`suites.py verity/protocols/certifier/verifier`). Waiting for a slot since 10:43 PM PST.
- `r20261112-064235-237e`: the whole root suite (`suites.py repository`). Waiting for a slot since 10:42 PM PST.
- No Lean try: `Domains.lean` is as at `5181e37f0973`.
- Still running from the first head: `r20261112-052947-6792` (every suite whose inputs changed), in its slot since about 9:50 PM PST, no verdict yet.

**Not done yet:** the pull request's body still describes the first head. I edit it when the two new runs have verdicts, with the section on the bytes and the line that the two encoders call the one function after lean's merge is on `main`. It stays unmarked and unlabelled.

