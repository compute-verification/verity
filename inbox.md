[2026-11-13T02:14Z] 3ed1911ae863
FIELDS, Thu 18:14 PST

**Job 2's records changed only in what guarantees read, all in `security/Properties/lean-audit.json`. No statement text or premise moved, so I committed them; #9565's head is `66a3543f46` and its body lists them.**

- RAN r20261113-003947-5236 (rc 0, at `e57681171b`): `lean_changed.py --records verity/circuits/lean --update`.
  - `security/Proofs` PASS.
  - `verity/circuits/lean`, `flock/verifier`, `certifier/verifier/lean` and `security` PASS with their records unchanged.
  - `security/Properties`: reads only. `Flock.Program.fieldsOf` is new (the guarantees reading `Circuits.Program` gain it, and `ServedComplies` reads it). `NciCheck.RecordText.fieldsOf` is gone. `tmplOf`, `TemplateOf`, `outRow` and `OutLeaf` changed, read by `ServedComplies`.
- RAN r20261113-004139-3407 (rc 0): the check's package full check, with 671 cases and 0 differences. Its lock changes only in the `runs` entry.
- RAN r20261113-004039-9eba (suites at `e57681171b`): the check slot's one-hour limit stopped it (rc 130). 7 suites had passed and none failed; `verity-vllm` had run 55 minutes without finishing.
- In flight at `66a3543f46`:
  - r20261113-021149-517d: records for `verity/protocols/spot/verifier/lean`, which 5236 did not cover.
  - r20261113-021149-636b: `suites.py --changed 7e70df68bd --quick`, with a three-hour limit.
- A correction to my previous message: its header said 17:45, but it was sent at 17:10.
- #9561 is unchanged since d299; it waits on your mark.

