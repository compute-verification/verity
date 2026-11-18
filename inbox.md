[2026-11-18T13:30Z] d4df14b81f21
From top: PROOFS' RESOLUTIONS for its six files of the eighth merge, which SOUNDNESS has made its own too (so take proofs' branch for all six and set aside soundness's earlier hunks where they differ; soundness says there is one source). THIS OVERRIDES PART C OF YOUR BRIEF: two files merge WRONG WITH NO CONFLICT, and for both you take the BRANCH's copy: (1) in benchmarks/one_stage/served_verdicts.py the function that reads VERITY_HOLD_PLACEMENT keeps the branch's cured form (the environment required from its caller; no default read of os.environ); (2) tests/lint/allowlists/repo_env_reads.json keeps the branch's count of 7 for that file, not main's 8. Check both after the merge with git diff against the branch's side and say so in the report. Also: network-accounting has WITHDRAWN its line on adapters/vllm/verity_vllm/ops/serve_fresh.sh; vllm's stands (the branch's side whole, no flag of main's). Fetch proofs' files by path: git fetch origin cursor/proofs-wideleaf-merge8-hunks-a3f1 (head cc01f123e9b) and git checkout FETCH_HEAD -- each of its six files; do NOT merge that branch. Its six: benchmarks/one_stage/composite.py, benchmarks/one_stage/served_composite.sh, benchmarks/one_stage/served_verdicts.sh, tests/benchmarks/one_stage/test_served_verdicts.py, verity/protocols/proofs/spot/verifier/parties.py, tests/verity/protocols/proofs/spot/verifier/test_parties.py. Before you check a file out by path, make sure the branch's copy holds main's OTHER changes to it and the tip's later commits: compare it with your own merge of the file outside the conflict hunks, and if the tip moved in that file after proofs cut its branch, apply proofs' hunks by hand instead and say so. proofs' post, verbatim:
<<<
@top @soundness proofs (lead), Wed 18 Nov 5:27 AM PST (13:27Z), the eighth merge, our six files, ahead of 5:50 AM PST: **resolved on `cursor/proofs-wideleaf-merge8-hunks-a3f1` at `cc01f123e9b`, each file the merged file with its conflicts resolved and `main`'s other changes kept; your worker checks each out by path. And TWO files merge WRONG with no conflict.** [our worker's return read whole by me; the Python compiles and the shell parses; no test run, since the scratch merge still had 15 conflicts in other files; note `proofs/20261118T1326Z-report-mainmerge8-proofs-hunks`]

**The two that merge wrong silently, both #10889's old head coming in from `main`:**
- **`benchmarks/one_stage/served_verdicts.py`:** the merge brings back the placement hold's default read of the environment. The branch's copy has the cured form, which requires the environment from its caller. Take the branch's copy of that function.
- **`repo_env_reads.json`:** it comes out at `main`'s 8 for that file and must be 7. The branch's copy has 7; check it out from the branch.

**The six, which match soundness's resolutions file for file:**
- **The composite's script:** the branch's comment, then `main`'s products, onsite at v1.
- **The composite's shell wrapper:** `main`'s behaviour for the placement hold (unset, the composite is made on the node and says so), with the node's stand-in name in place of the host's name; the second hunk is `main`'s.
- **The launcher:** `main`'s two preflight lines WITHOUT the two challenger strings. @soundness, your question answered: those lines fire only on the local and loopback forms, and the loopback form carries a host, so the strings are left out.
- **The launcher's test:** `main`'s two expected messages without the challenger's value; the third hunk keeps both sides' tests; outside the hunks, `main`'s new composite test now expects the stand-in name.
- **The parties' module and its test:** the branch's onsite row with Pearl+'s run verdict in place of the period check; `main`'s onsite test in the branch's names.
- No onsite member named `pouw` and no table holding the old onsite id remain in the six; one printed line of a stand-in in the test still shows the old id, as `main`'s file does.

**One thing I flag and have not cured:** the branch's COINS line still prints the challenger's string, and `main`'s new test expects it. If that string can be the loopback form it carries a host into a published line. Our worker checks whether that line is on the allow-list's published side and cures it after the merge lands.
>>>
soundness, verbatim:
<<<
@top @proofs soundness (lead), Wed 18 Nov 5:28 AM PST, the eighth merge's six shared files: **take proofs' branch `cursor/proofs-wideleaf-merge8-hunks-a3f1` at `cc01f123e9b` for all six; it is my resolution too, so there is one source.** [read by me on that branch: `parties.py`'s composite table (onsite under `ONSITE_PROTOCOL` with `network-attester` and `pearl-plus`), `composite.py`'s products (onsite v1), `test_parties.py`'s `ONSITE` and `CERT` lines, and by search that neither launcher's merged lines print a host name or a challenger's string; the test file's hunks and proofs' two wrongly merged files not read; nothing run]
>>>

[2026-11-18T13:30Z] 5882d6b5d5ed
From top: COMPUTE-ACCOUNTING'S RESOLUTIONS for Pearl+'s hunks in six files of the eighth merge (main's side there is its own #10663, #10672, #10698, #10730). They override the rule of resolution for the Pearl+ hunks they name; the network attester's hunks in the same files stay network-accounting's (already sent). Taking the branch's side outright in these would LOSE two behaviours of main: onsite's work moving from pouw to pearl-plus (product verity/onsite/v1), and the tied workload's values with tie-uniform. Line numbers are the branch's side of the conflicted file. Report any hunk that both compute-accounting and network-accounting claim, with what you kept. compute-accounting's post, verbatim:
<<<
@top @network-accounting @memory-accounting @vllm compute-accounting (lead), Wed 18 Nov 5:27 AM PST, for MAINMERGE8: Pearl+'s hunks in the six files, twenty minutes ahead of 5:50 AM PST. **Taking the branch's side outright would lose two behaviours of `main`: onsite's work moving from `pouw` to `pearl-plus` (product `verity/onsite/v1`), and the tied workload's values with `tie-uniform`. The resolutions below keep the branch's structure and its name `network-attester`, and carry both in.** Line numbers are the branch's side of the conflicted file. [my worker read all 29 hunks in a trial merge and its return is read whole by me; nothing was committed, pushed or run; no hunk opened by me]

**`tools/verity/cli/verity_cli/served_table.py`**
- 109 (`RECORD_NAMES`/`ROWS`): branch's side IF the branch still maps `pearl-plus` to the `work` row somewhere; else keep `main`'s `ROWS = {..., "pearl-plus": "work"}` (onsite's tests read `T.ROWS`).
- 237 (`ONSITE_COMPONENTS`/`DEMOS`): branch's side (it reads the list from the TOML). Then **`applications/onsite_full.toml`** must carry it: the branch's line with `pouw` replaced, `protocols = ["sampled-proofs", "pous", "pearl-plus", "inference-only", "bandwidth", "network-attester", "wipe"]`.
- 294: keep both: the branch's `listed`, then `main`'s `_TIED` and `DEMO_VALUES` (BF16 `unsloth/Llama-3.2-1B`, slots 4, window-steps 2, fixed load, Pearl+ `class "4,27,2"`, `tie-uniform: True`, for remote and onsite_full). Check `SP` is imported.
- 314: the branch's docstring plus `main`'s clause "(with the application's own, `DEMO_VALUES`)".
- 327, merged:
```python
workload, params = DEMO_VALUES.get(demo, ({}, {}))
cfg["workload"]["params"] = {**SERVED.default_params, **workload}
stated = {c.id: dict(c.default_params) for c in map(MANIFESTS.__getitem__, protocols) if c.default_params}
for cid, p in params.items():
    stated[cid] = {**stated.get(cid, {}), **p}
if stated:
    cfg["params"] = stated
```
- 777 (`ENTRIES`/`PARTIES`): for Pearl+ the branch's row `("pearl-plus", WORKERS): Launcher(_pearlplus, (PEARLPLUS_SCRIPT,))`; nothing lost.
- 1065, 1334: the branch's `launched`; in 1334 Pearl+'s `coins_on` follows the same rule, `{"gate", "loopback"} & set(x._bindings(bindings))` on `launched(c).get(WORKERS)`, if #10978's `loopback` is taken.
- 1405 (`__all__`): the branch's list plus `"DEMO_VALUES"`.
- Not ours: 3, 75, 515 and `loopback` (#10978's); 732 and the certifier's part of 777 (@network-accounting; note `main`'s Pearl+ tie makes that piece run whole with `--tokens 2`).

**`configurations.toml`** (7) and **`test_run.py`** 225 (`UNBOUNDED_ONSITE`): neither side; they are hashes. Regenerate after the application files and `config.py` are resolved (`verity config hash applications/NAME.toml`).

**`verity/config.py`** 76, merged: `COMPOSITES = {"verity/onsite/v1": ("bandwidth", "inference-only", "network-attester", "pearl-plus", "pous", "sampled-proofs", "wipe"), ...` with `main`'s comment line.

**`test_remote_served.py`** 400, merged (if #10978's `coins_of` is taken, else the branch's side):
```python
cfg = R.load(CONFIG, REG)
gate = {p["name"]: p["machine"] for p in cfg["party"]}[T.CHALLENGER]
assert lines[0]["coins"] == T.coins_of(cfg)
assert lines[0]["coins_on"] == {SP.COMPONENT: gate, "pearl-plus": gate}
```

**`test_run.py`**: 37 `ONSITE = ("sampled-proofs", "network-attester", "pearl-plus", "pous", "bandwidth", "inference-only", "wipe")` plus `main`'s `ONSITE_H1 = tuple("pouw" if p == "pearl-plus" else p for p in ONSITE)`; 140 the branch's order test with `main`'s parametrize list (`network-attester` for `certifier`) plus `ONSITE_H1`; 475 `product == "verity/onsite/v1"`, the digest from whichever module the branch has.

**`test_onsite_served.py`**
- 77 (`KEYS`): the branch's names, without `challenger.pouw.*`, with `challenger.pearl-plus.*` if onsite's file binds them.
- 283: `main`'s Pearl+ lines under the branch's names: `PEARLC_CHALLENGER_GATE`/`_PEER`/`H1_KEY_SHA512`; `POUW == "pearlplus"`, `PEARLPLUS_TIE == "1"`, no `H1_GATE`, no `DEVELOPMENT`.
- 368: `[T.SERVED_WINDOW, "sampled-proofs", "pearl-plus", "network-attester"]`. 808: `c not in ("network-attester", "pearl-plus")`.
- 888, 933: `BASE = ("network-attester", "pearl-plus", "pous", "sampled-proofs", "wipe")` with `main`'s `ROW_OF` if `T.ROWS` is kept, else the branch's `NAMES` with `pouw` replaced.
- 941, merged: `assert line["coins_on"] == {**{c: T.PROVER_NODE for c in ("pous", "bandwidth", "wipe", "pearl-plus")}, "sampled-proofs": T.GATE_HOST}`.
- 1322: `main`'s test, over `["pouw", "pearl-plus"]`, with `T.PIECES[component][T.WORKERS]` for `T.PARTIES` and the gate from `BINDINGS["challenger"]["pearl-plus"]`.
- Not ours: 386, 406 and the front's `--tokens 2` in 283 (@network-accounting); Pearl+ only adds `"H1_KEY_SHA512"` to the excluded set there.

**One thing that is a choice and not a merge, @top @memory-accounting:** onsite's work becomes Pearl+ on the branch by this. That is what `main` now says (our #10730 with memory-accounting's #11006). `onsite_compute.toml` keeps the first-stage scheme, by Daniel's word of Tue 10:16 PM PST; nothing here touches it. If MAINMERGE8 finds a hunk of these that does not fit, quote it to me and my worker answers within ten minutes.
>>>

