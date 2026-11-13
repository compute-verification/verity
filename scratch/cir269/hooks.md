# CIR-269: the hooks circuit-check is handed

Written Thu 12 Nov 6:43 PM PST [Fri 02:43Z]. The code is on `cursor/circuits-cir269-trial-6f04` at `4ffa2070b011`, one scratch
commit on top of the trial's `f94e9e91b89e` (both based on `main` `636d15d041b5`). It is not a PR.

## What this is

circuit-check (`tools/verity/circuit_check/`) imports the served circuits by name: 13 imports in its source, plus two
names it holds as data. Those circuits are the kernel registry, its fast evaluators and the Boolean table under
`verity/definitions/`. Top's move takes them to `adapters/vllm/verity_vllm/definitions/`. After the move, every one of
those names would be an import of the vLLM adapter from inside `verity`'s own tool, which the boundary rule forbids.

The fix is the one circuit-check already uses for the request Program's reader (`request_program`): the package that
owns the code hands circuit-check a function, by name, in a sidecar file. circuit-check finds that sidecar without
naming the package. This note says where that sidecar is, what it hands, and the one ordering question the first trial
left open.

## The place: (a), a sidecar of the catalog's own module

The file is `verity/definitions/registry/catalog.circuit-check.py`, beside `registry/catalog.py`.

- `sidecars.discover` finds it: `verity` is in `PACKAGES`, and `catalog.py` is the module it belongs to. After the move
  it sits at `adapters/vllm/verity_vllm/definitions/registry/catalog.circuit-check.py`, under the served map's
  `registry/` row. It is found there through `TREES`, since `verity_vllm` registers itself.
- vLLM's `__init__.circuit-check.py` is not touched. The trial's `row_families` in it is reverted.
- It can be read before the catalog is imported. That is why (a) holds, and the next section is the result.

## Loading it before the catalog: tried, and it works

`load_registries` used to import every module with a sidecar first. That loop now imports `registry.catalog` itself,
since `catalog.py` has a sidecar. So any hook the loop needs has to be read before the loop runs. A hook can't come from
`binding_modules()` either: that loads every sidecar, and most sidecars import their registry module at their top level.

The mechanism:

- `sidecars.handing(name)` parses each binding file, using `_top_level_names`, memoized per file by `_defines`. It loads
  only the files whose top level defines `name`, by path, under the usual `<module>.circuit-check` name. It imports no
  module and loads no other sidecar.
- `targets._handed(name)` takes the one it finds, memoized in `_HANDS`. When none or two define the name, it raises a
  `ValueError` that names the hook and the sidecars found.
- The catalog's sidecar imports nothing at its top level: each function imports what it hands.
- `load_registries` now runs in this order:
  1. `CC.enumerators(_handed("enumerators")())`;
  2. `load_catalog = _handed("load_catalog")`;
  3. the loop over `SC.discover()`, which imports each module with a sidecar;
  4. `load_catalog()`;
  5. `CC.registries(...)`, `REGISTERED` and `declare_by_calls()`, as before.
- The import order is unchanged: modules with a sidecar first, then the registry's top-level modules (not `quarantine`),
  then `numerics.rows` and `twins`. `load_catalog` repeats those lines exactly. It does not call `catalog.load`, which
  differs in two ways: it recurses into `device_pins/`, and it records import errors where these lines raise.
- The tracer still holds the listing by the listed modules, because `load_catalog` is among the handed enumerators.
  `load_registries`' own frame is still recognized in `cache._enumerator` for its imports.

Run locally on the scratch commit:

- `tools/verity/circuit_check/tests/test_targets.py`: **26 passed** in 16.16 s, and again in 10.81 s after the ruff fix.
  That is the trial's 15, less the trial's one row-family test, plus 12 new:
  - 10 parametrized tests, one per hook: on this tree exactly one sidecar defines it, a stand-in with one is read, and
    stand-ins with none or two are refused by name.
  - The order `load_registries` follows, with stand-ins: it reads `enumerators`, calls it, holds the result, reads
    `load_catalog`, imports each sidecar's module, and only then calls `load_catalog`.
  - In a fresh interpreter, reading all 10 hooks on the real tree leaves no module of `verity.definitions.registry`,
    `.numerics` or `.boolean` in `sys.modules` (printed `[]`). This is the real ordering result: the hooks are read
    before any catalog module is imported.
- `tests/test_boundaries.py`: **58 passed** in 39.81 s.
- `test_cache.py::test_each_enumerator_only_imports_what_it_lists`, which I changed: 1 passed. It now holds the handed
  enumerators to the rule, `load_catalog` among them. It no longer checks `load_registries`, which lists no package now.
- Top's staged `test_boundaries.py` from `f054fb84b31`, with its new `sidecars`, `targets` and `units` entries cut back to
  `verity_flock`: the two circuit-check tests passed. The control, the same file against the source before this commit,
  **also passed**. Before the move, `verity.definitions.*` is `verity`'s own package, so that test can't see these
  imports yet. The evidence that none are left is `rg 'verity\.definitions\.(boolean|numerics|registry)'` over
  circuit-check's `src/`: what's left is three docstring or comment lines and `heavy.json`'s unit names, which are data
  and move with the modules.
- ruff is clean on every changed file.

## The hooks

There are ten. `targets.HOOKS` names them, each with what it returns. All are in `catalog.circuit-check.py`.

| name | signature | returns | clears |
|---|---|---|---|
| `enumerators` | `() -> set[tuple[str, str]]` | the functions (module, qualified name) that list a package only to import its modules: `catalog.load`, `boolean._load`, and its own `load_catalog` | N1 (`cache.py:429`, `_ENUMERATORS`, now filled at run time by `cache.enumerators`) |
| `load_catalog` | `() -> None` | nothing; it imports every module of the registry, and `numerics.rows` and `twins` | `targets.py:83`, `:87` |
| `preload` | `() -> dict[str, Any]` | `twins.preload()`: every C++ model, built | N2 (`cli.py:1326`; `PREBUILD` now runs `targets.preload()`) |
| `kernels` | `() -> list[Any]` | `kernel_registry.registered()` | `:474` (`_kernel_roots` keeps its name and its seeding) |
| `tanh_rn` | `() -> tuple[Callable, Any, int]` | `tables.tanh_rn_table`, `TANH_RN_REGIONS`, `TANH_RN_LO` | `:484`, `:493` (`TABLE_PROBES` keeps its name and entries) |
| `row_families` | `() -> set[str]` | `set(rows.ROWS)` | `:634` |
| `boolean_table` | `() -> tuple[dict, dict, list[str]]` | `boolean._load()`: exact, by_name, errors | `:762`, `:863` |
| `made_on_demand` | `(fid: str) -> Any` | `boolean.made_on_demand(fid)` | `:782`, `units.py:515` |
| `boolean_version` | `() -> tuple[Callable, type[Exception]]` | `boolean_version` and `NoBooleanVersion`, so `boolean_family` still catches only that error | `:810`, `:851` |
| `lift_composite` | `(defn: Any) -> Any` | `lifted.lift_composite(defn)` | `sidecars.py:259` |

That is 13 sites, plus N1 and N2. The line numbers are `main`'s: they are the same at `636d15d041b5` and at today's
`8c11a1d5abff`.

- **F** (`call_scope.py:17`, its lazy `catalog.load()`) is not a hook. PR 1 clears it: `specialization` takes `reg`,
  defaulting to `REGISTRY` as registered, and each caller that relied on the lazy load loads the catalog itself.
- **N3** (`sidecars.PACKAGES`) names no served module and stays.

## Flock's 30 function-level imports: sized, not built

The same mechanism would carry them, since `handing` doesn't care whose sidecar it reads. The 30 imports are `from
verity_flock import <module>` inside functions, split ambient 2, checks 18, sidecars 1, structural 5 and targets 4. They
reach 14 modules and **49 distinct attributes**. Of those, 18 are `ir_lower`'s alone (counted by an AST walk of the five
files).

There is also one name held as data: the string `("verity_flock.ir_lower", "_boolean_build")` in
`cache._WORD_ENUMERATORS` (`cache.py:1396`).

The sidecar would be `verity_flock/__init__.circuit-check.py`. `discover` finds it inside `verity`'s tree, as it finds
`verity_flock.struct_slots`'. Its owner is proofs.

What stops it is not ordering:

- **What would be handed is half of Flock's lowering API.** Of the 49 names, judged by their names, 5 are classes (`Lowering`, `NoPiece`,
  `Tmpl`, `gf2.Circuit`, `CircuitTypeTable`), 9 are constants (`PIECES`, `TC_STEPS`, `FORMAT`, `DISPLAY`, three table
  widths, `TEMPLATES`, `lowerings.registry`), and 3 are private (`_Read`, `_boolean_ops`, `_width`).
- **Handing them adds nothing.** Hooks one per attribute (49 functions), or one per module (14, a re-export in disguise),
  would turn imports into names circuit-check looks up as strings. That is no less coupling, and it loses static
  checking.
- **checks.py is Flock's own check.** Its lowering and template checks hold each Definition's Boolean circuit to Flock's
  lowering, so checks.py's 18 imports are the check itself. The boundary test's own remedy fits them: "if the check is
  the distribution's own, run it from that distribution's command and suite". That is a split of `checks.py` between
  circuit-check and Flock's suite, and it is proofs' decision and proofs' files, not a CIR-269 hook.
- **Cost, inferred:** 4 or 5 hooks would cover ambient, sidecars and targets (7 imports). Structural's 5 and checks.py's
  18 need that decision first.

## What I trust least

That memoizing `_HANDS` per process is safe under the cache's tracer. A hook's function runs inside a target's trace from
the catalog sidecar's file, so that sidecar's `#common` would be in every record that called it. I read this from
`cache.py:1939,1943`; I have not seen it in a real recorded run. PR 2's cold whole-catalog check is the run that shows it.
