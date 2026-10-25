# How this package is kept

The vLLM integration's quality bar is enforced in code by shrink-only ratchets, which every check runs. Each test's
failure message carries its fix.

- **The rules** (`test_p01_core_abstractions.py` to `test_p12_shared_keys.py`, machinery in `_ratchet.py`). Each scans
 `verity_vllm/` with `ast` and regexes, importing nothing, so the lints need neither torch nor vllm and run in seconds.
 A violation is keyed by file, kind, enclosing symbol and detail, and is compared with `allowlists/pNN_*.json`, the
 violations that existed when the rule came in. The rules cover:
 - P1: core's abstractions;
 - P2: opened values;
 - P3: one evaluator;
 - P4: one result;
 - P5: properties;
 - P6: one CLI;
 - P7: declared inputs;
 - P8: facts;
 - P9: layering;
 - P10: size, at most 800-line modules and 150-line functions;
 - P11: names;
 - P12: shared keys.
- **By-name rules** (`tests/test_no_by_name_rules.py`, allowlist `tests/by_name_allowlist.json`): no verification
 answer comes from a model name, a module path or a family string.
- **Ruff** (`test_ruff.py`, allowlist `allowlists/ruff.json`): `ruff check` under `integrations/vllm/ruff.toml`, which
 says which rules it selects and why, keyed like the rules' violations. The formatter stays off.
- **The ratchet itself** (`test_ratchet.py`). A new key fails: fix the new code. A moved violation moves its entry in
 the same change. An entry allowing more than is found fails as stale: delete it or lower its count. So an allowlist
 only shrinks, and every failure prints the exact entry lines.

**The rule on every change.** A PR that changes non-test source under `verity_vllm/` leaves strictly fewer ratchet entries
than its merge base with `main` had. Exempt are reverts and pure moves that `tools/verity/check/move_check.py` certifies.
Once drive.toml's `landing_gates` names it, the merge queue runs `integrations/vllm/tools/ratchet_gate.py` from `main` when a
PR is marked ready and when it is merged onto `next`. It refuses a PR that retires nothing, saying how many entries to retire and naming candidates in the files the
PR edits. Before marking, run `python3 integrations/vllm/tools/ratchet_gate.py origin/main HEAD`.

To retire an entry, fix what it allows and delete it: its rule's stale-entry check then keeps it gone.
