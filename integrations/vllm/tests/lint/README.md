# How this package is kept

The vLLM integration's quality bar is enforced in code by shrink-only ratchets, which every check runs. Each test's
failure message carries its fix.

- **The rules** (`test_p01_core_abstractions.py` to `test_p16_cli_callers.py`, machinery in `_ratchet.py`). Each scans
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
 - P12: shared keys;
 - P13: duplicate bodies;
 - P14: one way;
 - P15: no versioned twins;
 - P16: every CLI command has a caller.
- **Typed records** (`test_typed_records.py`, allowlist `allowlists/typed_records.json`): a `.get(k) or []` or `or {}`,
 counted per package, and a schema id literal (`<namespace>/<name>/vN`) outside `verity_vllm/records/`. Read the record
 through its type in `verity_vllm/records/` instead, whose `load` refuses a missing or unknown field by name. A hash's
 domain-separation tag reads like one but names no record: a module-level constant named `*_TAG` or `*_DOMAIN` holds it as
 a plain literal, which the rule skips.
- **Recorded outcomes** (`test_silent_excepts.py`, allowlist `allowlists/silent_excepts.json`): in `verity_vllm/check/` and
 `verity_vllm/commit/`, a caught exception is raised or recorded (an `error` result, a reason, an `unreadable: …` entry in
 the record), never dropped with `pass`, `continue` or a default. A guard never fails open: a handler that skips what it
 can't read, or puts a default where a value was, shrinks the evidence a verdict is decided on and leaves no trace of it.
 `tests/lint/test_repo_silent_excepts.py` holds every other area to the same rule (below).
- **By-name rules** (`tests/test_no_by_name_rules.py`, allowlist `tests/by_name_allowlist.json`): no verification
 answer comes from a model name, a module path or a family string.
- **Ruff** (`test_ruff.py`, allowlist `allowlists/ruff.json`): `ruff check` under `integrations/vllm/ruff.toml`, which
 extends the root `ruff.toml` (the rules every package selects, and why), keyed like the rules' violations. The formatter stays off. A PR that fixes or adds a
 ruff finding in `integrations/vllm` (a stale `noqa` deleted is one) edits `ruff.json` in the same PR.
- **The ratchet itself** (`test_ratchet.py`). A new key fails: fix the new code. A moved violation moves its entry in
 the same change. An entry allowing more than is found fails as stale: delete it or lower its count. So an allowlist
 only shrinks, and every failure prints the exact entry lines.

**The rule on every change: no growth.** No PR adds a ratchet
exception or raises a recorded count: against its merge base with `main` (the merge of its merge bases, when a branch that
merged both `main` and a since-landed branch has several), no ratchet holds more entries and no entry's `count` is higher. An entry may move with the violation it allows. A kind whose entries record counts (typed_records' or-empty per
package, P07's) is held by its total instead: a key may rise, or a new one appear, when the same kind's other keys fall by
at least as much, so a move from one package to another passes, but the kind's total may not grow. P10's count is a size,
not a number of findings, so it is never pooled. A change to a lint's own code or thresholds is judged like source: the base's lint code, run on the PR's tree, must find nothing the PR's allowlists don't cover, so loosening a
lint to drop entries fails. Ruff's lint code is `test_ruff.py`, `integrations/vllm/ruff.toml`, the root `ruff.toml` it
extends and the root `dev` group's ruff pin, so the gate reruns it with the base's configs at the base's pin. The gate holds
the repository's ruff lint the same way (`tests/lint/test_repo_ruff.py`, list `tests/lint/allowlists/repo_ruff.json`: every
tracked file outside this lint's own), except that it counts findings per (file, rule), an entry
`{"file", "rule", "count"}` pooled by rule, and a count above what a tree finds doesn't fail: its list may hold the findings of
PRs queued on `next` that a tree doesn't have yet, and `python3 tests/lint/test_repo_ruff.py --prune` lowers what a tree no
longer finds. Exempt are reverts and pure moves that `tools/verity/check/move_check.py` certifies. A feature
PR owes no cleanup; dedicated cleanup lanes pay the exceptions down. `tools/verity/check/drive.toml`'s `landing_gates` names
it, so the merge queue runs `integrations/vllm/tools/ratchet_gate.py` from `main` when a PR is marked ready and when it is merged onto `next`,
and refuses a PR that grows a ratchet (naming the entries it added) or that the base's lints refuse (naming each violation).
It lints the PR's own Python too, since drive lands before the suites run: ruff, under `main`'s
root `ruff.toml`, on each `*.py` file the PR adds or modifies outside this lint's files, before and after, and
a PR whose (file, rule) count grows is refused, naming each new finding `file:line RULE message`. Fix the code; the list
never takes a new finding. (file, rule) is the repository lint's key too, so a finding moved to another function passes both.
A renamed file keeps its findings, but the repository lint reads no history, so its entries move with it
(`python3 tests/lint/test_repo_ruff.py --move OLD NEW`, in the same change): a rename that leaves them behind is refused,
naming each entry and that command. A file the PR doesn't touch is never linted. If ruff can't run where drive runs the
gate, the gate can't judge, and the sync leaves the PR out.
It holds direct environment reads the same way, so that configuration is read in one place.
`tests/lint/test_repo_env_reads.py` counts each tracked `*.py` file's reads (`os.environ[...]`, `.get`, `.setdefault`, a
`.pop` whose value is used, an `in` test, `os.getenv`, through any alias) outside `integrations/vllm/` (P07 is its settings
rule), against `tests/lint/allowlists/repo_env_reads.json` (`{"file", "count"}`, each file's largest count
over `main`, `next` and the admitted PRs), and fails only when a file reads more than its entry allows. A settings registry
is exempt: a module named `settings.py`, or one the list marks `{"file", "registry": true}`. The gate refuses a PR that adds
a read to a file it changes, naming each, with the fix: read it through the area's settings registry (a `Registry` of
`verity/settings.py`'s machinery; a distribution's, as `tools/config/verity_config/settings.py` says), or create one. A mark
counts from the PR's base, so it lands before the reads it exempts, and a rename moves the file's entry (`python3 tests/lint/test_repo_env_reads.py --move OLD NEW`), as for ruff.
It holds except handlers that swallow what they catch the same way, so that every outcome is recorded.
`tests/lint/test_repo_silent_excepts.py` counts, in the same files, each handler, typed or broad, whose every statement is
`pass`, `continue`, `break`, a bare constant, or a `return` or assignment of a default (nothing, `None`, a constant other than
a string or `False`, a name other than the caught exception's, an empty container), against
`tests/lint/allowlists/repo_silent_excepts.json`, built like the env-read list. The gate refuses a PR that adds one to a file it
changes, naming each, with the fix: re-raise; or record the outcome in the area's typed record or log; or test for the
expected case and let the rest raise (a narrower handler that still swallows still counts). A rename moves its entry
(`python3 tests/lint/test_repo_silent_excepts.py --move OLD NEW`). A repository list (ruff's or a counted lint's) may
raise a key, in any file, up to what its lint, as the PR's base has it, finds there in the base's tree: a list that records
its base's findings adds none, so one left behind by PRs that were on `next` before its lint landed catches up, while growth
past the base's findings is refused as before. A counted lint that comes to count more records its list in the same PR
(the lane-text lint's `--record`, run on `main` and on `next`, raises entries and lowers none):
the gate takes a rise at a file the PR leaves untouched (on `main`, or one a PR on `next` brings) from a PR that edits that
lint's module, since the PR adds no text there, and the source check still holds each file a later PR touches. A rise at a
file the PR edits is held as before.
It holds lane text the same way, in every area, this integration's included: `tests/lint/test_repo_lane_text.py` counts, in
the prose of each file it reads (comments and docstrings in code, Lean comments, every Markdown file; never a string literal
or a data file), dated attributions and ISO dates, `note:` ids, Notion URLs, Linear ids, PR numbers, agent ids, Slack
timestamps, worker codenames (board ids, rounds, lanes, handles), research run ids, red-team finding ids and host names,
against `tests/lint/allowlists/repo_lane_text.json`.
AGENTS.md, ARCHITECTURE.md, GLOSSARY.md and the skills hold none, though AGENTS.md and the skills may name a handle as an
owner and the machine a step runs on. The gate refuses a PR that adds one to a file it changes,
naming each, with where the text goes: history to git, work to Linear, rulings to Notion, results to the store, a
finding's rule in place of its id, and a run or host a tool reads into a data file it declares. It takes
over P11's docstring patterns (board ids, lane tags, rounds and dates) for the whole tree.
Before marking, run `python3 integrations/vllm/tools/ratchet_gate.py origin/main HEAD`.

**A `# noqa` is an exception too.** In `integrations/vllm`, a change may add no more
`noqa` comments than it removes. The gate counts them with no allowlist, and refuses a change that adds more, naming each
new line. Lines are matched by their stripped text, so a `noqa` that moves with its code counts as kept. Fix the finding
instead, or delete a stale `noqa` elsewhere in the same change: deleting one is a retirement.

To retire an entry, fix what it allows and delete it: its rule's stale-entry check then keeps it gone. Each lint's
`test_no_new*` tests are the ones the gate runs, under drive's Python (with the pinned ruff, for the ruff lint), so
`test_ratchet.py` holds every lint to having one that takes no fixture, to a `NAME` that is a string literal, and to
importing only the standard library and this directory.
