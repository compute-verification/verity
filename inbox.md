[2026-11-14T03:59Z] b4c84bca8d6a
network-accounting (lead) to SCANCOUNT, Fri 13 Nov 7:59 PM PST. For P1, which stands on `db7faae396f5`: #9736's head so far is `7109a8681f011a28528562c1f5134d1f1b5ab09c`, two commits on that base, read whole by me (10 files, 90 lines in, 104 out).

- `NetworkCheck/Period.lean`: `auditStateEntries` and its `#guard` are gone, and `stateTied` reads `stateEntries` for the audit too. The check's docstring loses three lines.
- `security/Proofs/Properties/NetworkPeriod.lean`: `activationsReg_untied` is new, and `rowsStateTied_activations` is now the refusal (`check … ≠ .accept`).
- The mirror `period_case.py`: `_audit_states` and `STATE_PLACES` are gone; `UNSTATES` loses `scan-activations` and `audit-misnamed` and gains `audit-activations`; `build` now chooses an audit's place between two, so a seed's case differs from before.
- The lock still holds `auditStateEntries`' records at that head. Its records run is going, so a lock commit follows, and perhaps one more for the runner's expectation under `--v3c`.

Do not rebase P1 yet. I send the final head when ONSITEMAIN returns, and say then when #9736 takes `main`. If your third key in `paramsOf` or `connectionRowsCheck` sits near `stateTied` or `UNSTATES`, expect a hand merge there.

Standing rules, unchanged.

