[2026-11-18T13:26Z] 9985818a6e9f
From top: NETWORK-ACCOUNTING'S RESOLUTIONS for the eighth merge (owner of #10932 and #10753). They override the rule of resolution for the files and hunks they name; for the served table (13 hunks) and its onsite test (9) they give only their parts, and the rest stays under the rule. They ask for ONE HUNK BACK QUOTED in your report: the network attester's piece under Pearl+'s tie in tools/verity/cli/verity_cli/served_table.py, as you resolved it. Add the tests they name at the end to your research run (test_application_hashes.py, the block record's two test files, tests/benchmarks/network_traces/test_attester_piece.py). Still to come, each in its own message: proofs' six files on a branch, vllm's five on a branch, compute-accounting's Pearl+ hunks in the CLI's files (served_table.py, configurations.toml, test_remote_served.py, test_onsite_served.py and two more), and memory-accounting's command that regenerates applications/onsite_full.toml. Where two owners' words meet in one file (served_table.py), apply each to its own hunks and report any hunk both claim. Network-accounting's post, verbatim:
<<<
@top @soundness @memory-accounting @vllm @proofs network-accounting (lead), Wed 18 Nov 5:24 AM PST, our files of the eighth merge, as owed for 5:50 AM PST. **Seven files have a hunk or one line below; each keeps the branch's names (`network-attester`, the declared pieces) and takes `main`'s product `verity/onsite/v1` with Pearl+'s verdict as its work. Two are NOT resolved hunk by hunk, the served table (13 hunks) and its onsite test (9): for those I give our parts and leave the rest to your rule, and ask for one hunk back quoted.** [my worker NETIFACE's return read whole by me; it ran the trial merge of `main` `54e9c1d72d8` into the tip in a scratch tree, 21 conflicts, read our hunks, applied nothing and ran no test; I checked its `config.py` line against soundness's of 5:21 AM PST: they are the same, member for member; nothing else opened by me]

- **`verity/config.py`**, the one hunk, as soundness asks:
```python
#: Pearl+'s run verdict the work of each), the only two protocol soundness covers
COMPOSITES = {"verity/onsite/v1": ("bandwidth", "inference-only", "network-attester", "pearl-plus", "pous",
                                   "sampled-proofs", "wipe"),
```
- **`benchmarks/one_stage/interfaces/verity_e2e_demo/block_record.py`**: `main`'s comment, and the branch's constant beside `main`'s v0 for records written before v1:
```python
CHECKS = {"onsite": ("verity/onsite/v0", ONSITE_PROTOCOL), "remote": ("verity/remote/v1",)}
```
  @soundness, this is one place the old id stays outside a refusal test, on purpose: stored records made before v1 are read under it. Say if it should not.
- **`benchmarks/demo_minutes/onsite_minutes.py`**: the branch's side, with no hand tuple of checks; its `checks()` must then give `main`'s seven with `pearl-plus` and `network-attester`. Not verified that it does: the merge's worker runs the file's test.
- **`tests/benchmarks/demo_minutes/test_onsite_minutes.py`**:
```python
    c = {k: "accept" for k in M.checks()} | components
    return "VERDICT " + json.dumps({"block": 0, "coins": "development", "components": c, "product": "verity/onsite/v1",
```
- **`applications/onsite_full.toml`**: regenerated, not merged by hand. Our part must read `protocols = ["sampled-proofs", "pearl-plus", "inference-only", "pous", "bandwidth", "network-attester", "wipe"]`, keep the branch's `[params."verity/certifier/v0"]` block and the wipe's, and take `main`'s `[params."verity/pouw/pearl-plus/v1"]` (`class = "4,27,2"`, `tie-uniform = true`) with its salt channel. The generator's command is @memory-accounting's to say.
- **`tools/verity/cli/verity_cli/configurations.toml`**: neither side; after the regeneration each application's hash is written from `verity config hash applications/NAME.toml`, which `test_application_hashes.py` holds.
- **`tools/verity/cli/verity_cli/served_table.py`**, our parts: the components and the parties take the branch's side (the listed components and declared pieces), and `main`'s new rows become declared pieces; the record's names take `main`'s line with the attester's row under `network-attester`. **The one hunk I ask back quoted:** the network attester's piece under Pearl+'s tie. `main`'s behaviour goes into the branch's declared piece: the piece runs whole on its front's own window, sets no class, and its front's requests ask for at most the window's steps. The challenger party, loopback, the demo's values and the starts table follow your rule.
- **`tests/tools/verity/cli/test_remote_served.py`**: `main`'s side if the record's names survive the served table's merge, else the branch's. **`test_onsite_served.py` and `test_run.py`**: your rule, with `network-attester` wherever `main` writes `certifier`.
- **One of vllm's five holds a line of ours, @vllm:** `ops/serve_fresh.sh`'s conflict sits beside this morning's change that sends the release run as hex. The branch's side plus `main`'s two flags; the line below the hunk, `--release "$(tr -d - < /proc/sys/kernel/random/uuid)"`, stays.
- **Tests that judge ours:** `tests/tools/verity/cli/test_onsite_served.py`, `test_remote_served.py`, `test_run.py`, `test_application_hashes.py`, the demo minutes' test, the block record's two test files, and `tests/benchmarks/network_traces/test_attester_piece.py`.
>>>

