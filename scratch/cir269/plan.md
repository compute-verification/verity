# CIR-269: circuit-check is handed the catalog it checks — the sized plan

CATALOGHAND, for circuits' coordinator. Started Thu 12 Nov 2026 6:07 PM PST [02:07Z], on `main` `636d15d041b5`
(`git fetch` of 5:59 PM PST). Scratch only: nothing here is a pull request.

Marks: **[read]** = I read it on `main` `636d15d041b5` (or the named head) in this session, by `rg`, `git diff` or the
file itself; **[inferred]** = my reading of code I did not run; **[ran]** = output of a command I ran. Nothing here built
the catalog.

## 4. The answer for top (first, as asked)

One worker, doing nothing else, from Fri 13 Nov morning:

- **First pull request open: Fri 13 Nov** (PR 1, `call_scope.specialization` handed the registry, about 30 lines).
- **Landing before the move:** last pull request on `main` **Sat 14 Nov at the earliest, Sun 15 Nov as planned**,
  before Sunday night's move. What sets the day: **@vllm's read of the adapter's sidecar** (PR 2a) and PR 2b's
  candidate check, the one landing that moves `cache.version`, so its circuit-check suite session builds the catalog
  cold (the coordinator's 30 to 37 minutes of one slot; minutes, not a day). Then the move exposes **no** site:
  `CIRCUIT_CHECK_IMPORTS` never grows, and the move changes no versioned circuit-check file [inferred, section 3].
- **Landing after the move:** last pull request on `main` **Tue 17 Nov at the earliest** (Mon 16 Nov only if the
  restack is done Monday morning and both reads are same-day). What sets it: **the move on Sunday night and the
  restack on Monday**, after which PR 2a and 2b are rebased onto `adapters/vllm/verity_vllm/definitions/`, and PR 2b
  also shrinks `CIRCUIT_CHECK_IMPORTS` (`tests/test_boundaries.py`, @top's file: a third owner's read). Two cold builds
  instead of one: the move's own (it then has to edit `sidecars.py`, `units.py`, `cache.py`, `cli.py`) and PR 2b's.
- **A day earlier** (before the move: Sat 14 Nov): the coordinator agrees the hook names with @vllm on Fri 13 Nov, so
  PR 2a is read the day it opens; PR 2a folded into PR 2b (one PR, two owners' reads, one queue tick); PR 2b in the same
  candidate as #9340, which moves `cache.version` too (one cold build for both).
- **Later:** #9340 slipping (it edits `cache.py`, `cli.py` and `units.py`, so PR 2b rebases onto it; no line of mine
  overlaps its hunks [read]); #9375 slipping (`cache.py`, `sidecars.py`); a caller of `call_scope.specialization` in
  vLLM or benchmarks that relied on its lazy load and fails in that suite (PR 1); CIR-234's narrower key landing midway
  (PR 2b rebases onto a changed `key`); a red circuit-check session on `main` unrelated to this change, which PR 2b's cold
  build would meet first.

## 1. The sites

Each line number is on `main` `636d15d041b5` [read]. "Who calls it": every path into circuit-check goes through
`targets.load_registries()` first: the step (`cli.run`, `circuit-check --all`), the graph's per-target tasks (`cli.run`
on one target), the suite's session (`tests/conftest.py` builds `targets.suite()`, which runs `binding_files()` →
`load_registries()`) and `prebuild` (`cli.PREBUILD`, in a process of its own) [read: `rg load_registries`].
"Served alone": what it takes is the served part's alone, or also needed for Definitions that stay under `verity/`.

| # | File:line | What it takes | Who calls it | Served alone? |
| -- | -- | -- | -- | -- |
| 1 | `targets.py:83` | the catalog: `import verity.definitions.registry as R`, then every module of it (`pkgutil.iter_modules`, `quarantine` aside) | `load_registries`: every path above | served alone |
| 2 | `targets.py:87` | the kernels' registration as a side effect: `from verity.definitions.numerics import rows, twins` | `load_registries`: every path | served alone |
| 3 | `targets.py:474` | a kernel's twin: `numerics.kernel_registry.registered()` → each `check_targets(rng)` | `_kernel_roots`, a root function in `ROOT_FUNCTIONS`: the catalog (`suite`, `catalog_units`), so the step, the session and each graph task of a `_kernel_roots` unit; `test_shard.py:297`, `test_cache.py:1436` name the unit | served alone (the vLLM kernels' self-check specialisations) |
| 4 | `targets.py:484` | a table: `registry.tables.tanh_rn_table()`, `TANH_RN_REGIONS` | `_tanh_rn_probe` ← `TABLE_PROBES` ← `table_probe` ← `checks.py:474-477` (a correspondence of `TableRead27x2_v1`, `TanhF32Rn_v2`): any check of those two Definitions | served alone (both are defined in `registry/tables.py` and `registry/operand_widths.py` [read]) |
| 5 | `targets.py:493` | a table: `registry.tables.TANH_RN_LO` | `_tanh_rn_word` ← `TABLE_PROBES["TanhF32Rn_v2"]`: as 4 | served alone |
| 6 | `targets.py:634` | the replay's row families: `numerics.rows.ROWS` | `word_call_families` ← `call_families`, `is_call` (the partition invariant's "is this a Call"), `units.call_families` (the records path) | served alone |
| 7 | `targets.py:762` | the Boolean table's two lookups: `boolean._load()` (exact, by name) | `boolean_call_candidates` ← `is_call`, `units.call_families:362` | **both**: the table is built from `verity.definitions.ops.boolean` (stays) and `registry.catalog.load()` (moves) (`boolean.py:104-112` [read]) |
| 8 | `targets.py:782` | a Definition's Boolean version made on demand: `boolean.made_on_demand` | `on_demand_boolean_families` ← `boolean_call_candidates`, `certified_boolean_calls`, `boolean_call_families`, `units.call_families:359` | **both**: `Const<w>[0x..]` from `ops.boolean.scalar` (stays); `Gather*`, `BitAtx*` from `registry.boolean_gather` (moves) (`boolean.py:170-183` [read]) |
| 9 | `targets.py:810` | a Definition's Boolean version: `boolean.boolean_version` | `certified_boolean_calls` ← `is_call` | **both** (as 7) |
| 10 | `targets.py:851` | a Definition's Boolean version: `boolean.NoBooleanVersion`, `boolean_version` | `boolean_family` ← `boolean_call_families`, `units.call_families:360,370` | **both** (as 7) |
| 11 | `targets.py:863` | the Boolean table's two lookups: `boolean._load` | `boolean_links` ← `units.call_families:364` | **both** (as 7) |
| 12 | `sidecars.py:259` | a composite's lift: `registry.lifted.lift_composite` | `composite_lift` ← `named` ← `pin_owner` ← `pin_file`, `write_pins`: any check that looks up or writes a pin of a `Lifted[<composite>]_v<k>` id | **both** [inferred]: it lifts any registered composite, `ops`' included |
| 13 | `units.py:515` | a Definition's Boolean version made on demand: `boolean.made_on_demand` | `_replay` ← `plan`: the records path (a session or a step taking records replays what a binding registered) | **both** (as 8) |
| F | `verity/definitions/call_scope.py:17` | the catalog: `registry.catalog.load()` → `(REGISTRY, errors)`, inside `specialization` | `word.check_calls` (`word.py:602`, default `specialize=`), `cross_call.check_program` (`cross_call.py:340`, default), `targets.py:192,273` and `checks.py:1727` (as `W.specialization`), and outside circuit-check: `adapters/vllm/tools/popread.py:787`, `verity_vllm/acquire/sources/call_boundary_plan.py:289`, `verity_vllm/pipeline/serving_templates.py:249`, `verity_vllm/pipeline/program_graph.py:448` (as `word_query.specialization`), `benchmarks/one_stage/a11_kinds.py:293`, `benchmarks/harness/verity_bench/prove_records.py:317`, `benchmarks/bool_build_cost/run.py:61`; tests `verity/definitions/tests/test_call_scope.py`, `test_word.py`, `benchmarks/harness/tests/test_prove.py:1488` | **both**: `load("registry")` only imports `registry/`; what resolves is any registered family, `ops`' included |
| N1 | `cache.py:429` | names two functions by module: `_ENUMERATORS = {("verity.definitions.registry.catalog", "load"), ("verity.definitions.boolean", "_load")}` | `_enumerator` ← `_enumerating`: the tracer, in every recording, decides a `pkgutil` listing is an enumeration held by its package's modules | served alone (both functions move) |
| N2 | `cli.py:1326` | writes `from verity.definitions.numerics import twins` into `PREBUILD`, then `twins.preload()` | `prebuild`, run by the step before the tracer starts | served alone |
| N3 | `sidecars.py:45` | `PACKAGES = ("verity", "verity.circuits", "verity.definitions.kernels", "verity.core")` | `_roots` ← `discover` | **names no served module** [read]. After the move the served sidecars are found under `TREES` (`adapters/*/*` with `__init__.circuit-check.py`, which `verity_vllm` has [read]). Nothing to change |

Found beside them, not in the brief [read]:

- `heavy.json`: 23 lines name a served sidecar's unit (`verity.definitions.registry.<m>.circuit-check#...`). Data, no
  key's input (`cache.version` leaves it out, `cache.py:2080,2094`): the move rewrites them, or those bindings lose their
  measured peaks and schedule as unmeasured.
- Prose only (no key moves: `cache._code` hashes the AST with docstrings cut, and comments aren't in it): `cache.py:12`
  and `:176`, docstrings of `targets.py` (`:628`, `:778`, `:835`, `:850`, `:860`), `units.py:494`, `README.md:16`.
- circuit-check's tests name the served part on 45 lines of 14 files [ran: `rg -c`]. `CIRCUIT_CHECK_IMPORTS` reads
  `src/` only (`test_boundaries.py:859`), so they're the move's rewrite and no part of this plan.

**`checks.py`: confirmed, with one correction to the note.** Its imports of `verity.definitions` are `word` (`:1113`,
`:1207`, `:1284`, `:1691`) and `cross_call` (`:1310`, `:1707`) only [read], both staying. It imports none of the
served part. After top's ruling, the files of `src/` that import the served part are three: `targets.py`, `sidecars.py`
and `units.py`. `cache.py` and `cli.py` name it in strings that `test_circuit_check_imports_no_workspace_distribution_but_verity`
doesn't see (it reads imports, import-call strings and `"module:attr"` strings, `_imports`, `test_boundaries.py:395-416` [read]). After
the move they still break at run time.

## 2. The interface

What a distribution has to hand circuit-check, from the table: (i) "load your catalog" (1, 2, F), (ii) the names of the
functions that enumerate it (N1), (iii) the Boolean table: `boolean_version`, `NoBooleanVersion`, `made_on_demand`, its two
lookups (7 to 11, 13), (iv) the replay's row families (6), (v) its kernels' check roots (3), (vi) its table probes (4,
5), (vii) a composite's lift (12), (viii) the models to build before tracing (N2).

### A (recommended). The adapter's registering sidecar hands named hooks, in `request_program`'s form

- **What the adapter provides, and where:** a few names in the one file circuit-check already finds it by,
  `adapters/vllm/verity_vllm/__init__.circuit-check.py` (@vllm), each a few lines delegating to the served modules (before
  the move, `verity.definitions.*`; after it, `verity_vllm.definitions.*`): `load_catalog()` (lines 83 to 87: the
  registry's modules and `numerics.rows`, `twins`; returns the import errors), `CATALOG_ENUMERATORS` (the `(module,
  qualname)` pairs of `_ENUMERATORS`), `boolean_version`, `NoBooleanVersion`, `made_on_demand`, `boolean_links()`,
  `row_families()`, `kernel_roots()`, `table_probes()`, `lift_composite`, `preload()`. None of them is binding-shaped
  (`_X_roots`, `_X_bindings`), so `binding_files()` takes them as it takes `request_program` today [read:
  `targets.py:382`].
- **How circuit-check reads them:** `targets._handed(name)` beside `_read_by` (`targets.py:328`): the one binding file
  that defines it, else refused by name; a union over every binding file for `row_families`, `kernel_roots` and
  `table_probes`. `_kernel_roots` and `TABLE_PROBES` stay in `targets.py` under their own names and call the hooks, so no
  unit name changes. `cache._ENUMERATORS` is set at run time (`CC.enumerators(...)`) by `load_registries` before it calls
  `load_catalog`, and `cli.PREBUILD` calls `targets.preload()`. `call_scope.specialization(fam, statics, reg=None)` takes
  the registry and loads nothing; `None` means `verity.circuits.defs.REGISTRY` as loaded.
- **How a session and the step find it:** they don't change. Every path calls `load_registries()` → `sidecars.discover()`,
  which finds the adapter's package by `TREES` already. No argument, no variable, no entry point. One step is new:
  `load_registries` loads the registering sidecar (`__init__.circuit-check.py`, which imports nothing at its top
  [read]) before the registries rather than after them. That's the one ordering change, and the trial's to try.
- **Sites cleared:** all 13, F, N1 and N2. N3 needs nothing.
- **A record's key, and `cache.version`:** the switch edits `cache.py`, `cli.py`, `sidecars.py` and `units.py`, so
  `cache.version` moves once. It also edits `targets.py`'s `common` unit, which every record holds (`cache.py:1939,1943`
  [read]), in the same landing: one whole-catalog build for both. After it, a hook's edit moves that sidecar's `common`
  in the records that ran it, not `cache.version`. At the move, the hooks' imports change path only. `_digest_ast` keys
  an import by the names it binds (`cache.py:1062-1069` [read]) and `_moved` follows a moved module by its content, so
  the move keeps those records [inferred: from the docstrings, not run].

### B. One catalog object behind a protocol in `verity`, named by an entry point

- **What the adapter provides:** one object (`verity_vllm.definitions.catalog:CATALOG`) implementing a `Catalog`
  protocol that `verity` defines (the same eight needs as methods), named in `adapters/vllm/pyproject.toml` under
  `[project.entry-points."verity.catalog"]`. circuit-check reads `importlib.metadata.entry_points(group="verity.catalog")`.
  `call_scope.specialization` takes a `Catalog` too.
- **Sites cleared:** the same.
- **How found:** by the installed distribution's metadata, so it needs a `uv sync` after the edit. The tree's one entry
  point today is vLLM's own plugin (`adapters/vllm/pyproject.toml:17` [read]).
- **A record's key:** the entry point lives in `.venv`'s `dist-info`, outside the tree. A record wouldn't hold a changed
  entry point unless the tracer learns to [inferred]. `cache.version` moves once, as in A.

**Recommendation: A.** It is the mechanism the tree already has: `request_program` and `_vocabulary_roots` from vLLM's
sidecar, and `struct_slots_split` from Flock's `struct_slots.circuit-check.py` (the code behind the #9069 precedent;
#9069 itself changed only READMEs [read]). Every path finds it with no new argument. Each hook is a file the tracer
already holds, so the cache needs no new kind of input. B adds a protocol to `verity` and an input outside the tree
that the cache can't see, for a second distribution that doesn't exist yet.

## 3. The pull requests

Owners from `research msg owners` [ran]: `tools/verity/circuit_check/**` @circuits, `verity/definitions/**` @proofs,
`adapters/vllm/verity_vllm/__init__.circuit-check.py` @vllm, `adapters/vllm/verity_vllm/definitions/**` (after the
move) @vllm, `tests/test_boundaries.py` @top.

| PR | Files (owner) | Rests on | Before the move? | Under `src/circuit_check`? |
| -- | -- | -- | -- | -- |
| **1. `call_scope.specialization` is handed the registry** | `verity/definitions/call_scope.py`, `tests/test_call_scope.py`, `tests/test_word.py` (@proofs); the callers that relied on its lazy load pass a loaded registry: `adapters/vllm/tools/popread.py`, `verity_vllm/pipeline/serving_templates.py` (@vllm), `verity_vllm/acquire/sources/call_boundary_plan.py`, `benchmarks/bool_build_cost/run.py` (@circuits), `benchmarks/one_stage/a11_kinds.py` (@proofs), `benchmarks/harness/verity_bench/prove_records.py`, `benchmarks/harness/tests/test_prove.py` (@benchmarks). `serving_templates.py:536` and `a11_kinds.py:277` load the catalog already [read]; whether the others do, I didn't read | nothing | **yes, and it must**: after the move, line 17 would be `verity` importing `verity_vllm`, which `verity`'s boundary tests refuse (`FORBIDDEN_EXTERNAL`, `test_boundaries.py:200`, read by `test_core_unit_imports_only_inward`, `:461` [read the list; inferred that it fires there]) | no. circuit-check's three callers run after `load_registries`. Records that read `call_scope.py` miss once [inferred: the targets that ran a word or cross-call check] |
| **2a. The adapter's sidecar hands circuit-check its catalog** | `adapters/vllm/verity_vllm/__init__.circuit-check.py` (@vllm): the hooks of section 2 A, unused until 2b | nothing | yes (the hooks import `verity.definitions.*`) | no. Records holding that sidecar's `common` miss once [inferred: the vocabulary and row targets] |
| **2b. circuit-check reads them, and names no served module** | `targets.py`, `sidecars.py`, `units.py`, `cache.py`, `cli.py`; tests `test_targets.py` (a test that a hook is read from a stand-in sidecar, and refused by name when none or two define it), `test_circuit_check.py` (its `TABLE_PROBES` uses at `:550-586`, `:773`) (@circuits) | 2a; #9340 and #9375 on `main` first (same files: rebase) | yes. The move then exposes nothing, and `CIRCUIT_CHECK_IMPORTS` never grows | **yes**: `cache.version` and `targets.py#common`, one whole-catalog build |
| *(after the move instead)* 2b′ | as 2b, on the moved paths, plus `tests/test_boundaries.py` (@top): the list's entries for `targets.py`, `sidecars.py` and `units.py` lose `verity_vllm` | the move and Monday's restack | only after | **yes**, and the move's own landing already moved `cache.version` once |

**Landings that move `cache.version`: 1 before the move (2b), 2 after it** (the move's own edits of `sidecars.py`,
`units.py`, `cache.py`, `cli.py`, then 2b′). Folding 2a into 2b keeps the count at 1.

Where they meet the work in flight [read: `git diff origin/main...HEAD` of each head]:

- **#9375** (head `34c5cbb3705c`) changes none of my lines. In `sidecars.py` its hunk is `named` (`:271`, `:281`:
  `Registry` → `DefinitionTable`), 12 lines below site 12's import at `:259`; no textual conflict, but 2b edits
  `composite_lift` beside it. In `cache.py` its hunk is `_instrument` (`:739-763`), not `:429`. It also adds two served
  modules with sidecars (`registry/linear_cut.py`, `registry/pearl_c_tile.py`), which hook 1 (`load_catalog`) picks up
  as today's line 83 does. Its own edits of `by_calls.py`, `cache.py` and `sidecars.py` move `cache.version` once,
  tonight.
- **#9340** (head `3aefb20f1941`, label off) changes `cache.py` (`:73`, `:2242-2777`, `:3083`), `cli.py` (`:13`,
  `:369-959`, `:1419-1621`), `units.py` (`:23-160`, `:429-452`, `:609-700`, `:1037-1069`) and `tests/conftest.py`. None
  of those hunks is `cache.py:429`, `cli.py:1326` or `units.py:515`. 2b rebases onto it. Landing in one candidate with it
  makes one cold build of two.
- **CIR-234** (no worker until Mon 16 Nov) changes what `cache.key` holds. If it lands first, 2b's cold build may be
  narrower (only targets that loaded the edited modules). If 2b lands first, CIR-234's table counts it as one of the
  landings it would have narrowed. They meet in `cache.py`'s `version`/`key`, which 2b doesn't edit.
- **The move** (top's, Sunday night): with 2a and 2b on `main`, it edits the adapter's sidecar's imports, `heavy.json`'s
  23 keys and prose, and no versioned circuit-check file [inferred].
