# Inbox for the fine-grained check swarm (G0 core, G1 test files, G2 Lean, G3 circuits, L1 module system)

Read at every hourly checkpoint: git fetch -q origin cursor/top-graph-inbox && git show origin/cursor/top-graph-inbox:inbox/graph.md
Newest entries first. The contract is inbox/graph-contract.md on this branch.

- 22:00Z (top): Daniel wants this as parallel as possible, under one plan. Changes:
  - Two more workers. G2b (LEAN-148, bc-bf7ecfc2): the Lean runners (`runners/lean_module.py`, `runners/lean_lock.py`).
    L1b (LEAN-147, bc-d92d333f): the Lean check replays private proof bodies (fixtures with `sorry` and an axiom in
    non-exposed proofs), branched from #6819's head.
  - G2: you now own the Lean families only (`families/lean_module.py`, `families/lean_lock.py`, their tests and the
    listing numbers). Leave the runners to G2b; agree the task fields with it through what you push.
  - L1: you now own the codemod, the ported tree's build and its fixes, the lock diff (`check.py --update` leaves every
    `lean-audit.json` unchanged), the before/after numbers and the landing plan. L1b owns review check 1 (replay of
    private bodies); lean owns checks 3 and 4 (header parsers, cache artifacts) as LEAN-149. Call the Lean check
    verified when L1b's fixtures pass on your ported tree.
  - G0: two additions for Daniel's dashboard. `verity graph check --local` also writes events/v1 lines (purpose
    `adhoc`), so a family run by the reference executor gives dashboard data before the executor's events exist. As
    soon as listing works, publish `verity graph --json` of main's tip to the store
    (`research data put --kind verity-graph --file F --preserve`) and post the `art:` id on CI-337.
  - Tonight's executor run uses a commit that holds main plus your branches (a PR head is fine): nothing has to land
    on `checked` first. Still mark your PRs on time; ci puts them first in its order.
  - The plan: one Linear project, "Fine-grained check: one graph, one executor", milestones M1 (tonight) to M5 (Fri).
    Your issue sits in M1 with its "blocks" links: CI-337 blocks G1, G2, G2b and G3; L1b blocks L1.

- 21:36Z (top): reviewers posted what their review checks, on your issue (read its comments); build to pass it.
  - G2 (LEAN-145): take module inputs from `tools/verity/lean/module_keys.py` (imports, `[[input_file]]`/`[[input_dir]]`,
    toolchain, manifest); don't write a second header parser. A lean-lock task's inputs are its package's modules plus
    every path dependency's, as `lean_changed.py --records` walks them (Security/Properties requires Security).
  - L1 (LEAN-146), five checks: replay loads private bodies (fixtures: a `sorry` and an `axiom` used only in non-exposed
    proofs); `check.py --update` on the ported tree leaves every `lean-audit.json` byte-identical; `module_keys.py`,
    `lean_changed.py` and `tests/test_lean_vocabulary.py` learn `module`, `public import`, `meta import`, `import all`,
    with a test per form; `cache.py get` restores a ported package with zero recompiles; it lands as an idempotent
    codemod rerun over each open Lean PR's head.
  - G3 (CIR-148): per-binding tasks keep the read-set cache keys (#6455, #6657) and the records step (#6663); the
    catalog artifact equals what each xdist worker builds today.
  - G0 (CI-337): land `contract.py` and `tests/key_vectors.json` early. lean's LEAN-140 cache (#6819) lands with its own
    keys and switches `module_keys.py` to `verity_graph.contract`'s key once those are on main (Mon 18:00Z).
  - Reviews: when you mark, put the PR number and head in your Linear checkpoint; top rings the reviewer (lean for G2/L1,
    ci for G0/G1, circuits for G3, verdict within 2 h).

- 21:21Z (top): workers: G0 bc-dcd57c5d, G1 bc-c7049025, G2 bc-c6fea9c7, G3 bc-0d1264f9, L1 bc-22ae8c0d. Executor side is infra's (INF-409). Issues: TOP-145 parent.
- 21:20Z (top): swarm launched. G0 pushes the skeleton to cursor/top-graph-core-a001 by 22:30Z; G1-G3 merge it into their branches once it is there.## 22:20Z Sun 1 Nov (top): G1, push the family early

- G1: infra's executor is live on the pod (key vectors pass, result cache, events/v1). Tonight's test-file run starts the moment your branch lists the test-file family, finished by 06:00Z. Push `families/test_file.py` and a runner that works, on your branch, as soon as `verity graph --json --family test-file` lists, even before the 50-file comparison. Post the branch and SHA on CI-338.
- infra's executor branch: compute-verification/infra `cursor/infra-executor-98f5` (graph2.py). A top worker adds read staging and a node cache there (INF-410).



## 23:55Z Sun 1 Nov (top): G0 marks #6847; Lean tasks read public parts only

- G0 (bc-dcd57c5d, CI-337): #6847 is red on `tools/research/tests/test_pythonpath.py`: its root list lacks `tools/verity/graph`, which your core adds. G2 fixed it on #6846's branch; take the same fix, then mark #6847 by 02:00Z. #6846 can't be marked while it contains an unmarked #6847.
- G0: two flags from G2, in #6846's body. (1) Under the module system each import becomes five read entries, about 8.9M in all, over the core's 60 CPU-s listing budget; the decision below takes it back to one or two per import. (2) `verity graph run` spends 5.5-12.8 CPU-s listing for each module it runs. Reply on CI-337 with how the core lists once per run.
- Decision for G2 (bc-c6fea9c7), G2b (bc-bf7ecfc2), L1 (bc-22ae8c0d), and lean to confirm, from E2's report on INF-410:
  - A module task reads its imports' public outputs only: `.olean`, plus `.ir.sig`/`.ir` where it compiles code. It never reads `.olean.private`.
  - The runner's replay imports at the public level.
  - Each module task writes its declarations' axiom sets as an output. A module's axiom check unions its own with the sets its imports recorded.
  - This keeps the Lean check: every declaration is replayed once, in its own module's task, and axioms are checked transitively. It also makes a proof-only edit rerun one module. Before L1's port no file is a `module`, so until then every dependent of a changed `.olean` reruns. INF-410's done-condition holds from L1 on.
  - G2: the listing change is a follow-up PR on #6846; mark #6846 as it is. G2b: the runner writes and reads axiom sets. lean: confirm the guarantee on LEAN-148 by Mon 06:00Z. All of it lands with K2, Mon 18:00Z.

## 02:00Z Mon 2 Nov (top): ci's follow-up on #6863 goes to G1B; who is marking what

- G1B (bc-3b986727, CI-345): ci's land verdict on #6863 came with a follow-up for your stacked PR. Rename `families/test_file.py` and `runners/test_file.py` to names that don't match `test_*.py`, and drop the exemption #6863 added to `tests/test_repository.py` for `test_*.py` files outside every suite that define no test. Say so in your PR body, so ci's reviewer sees it.
- G0's #6847 and G1's #6863 have ci's land verdicts (00:46Z) and sit first in `--first`. G0 works on K2's list-once-per-run on `cursor/top-graph-core-k2-83a1` (its 01:26Z reply on CI-337).
- G3 (#6858 at f04dd31279) and L1b (#6872 at a6ea6784da) were resumed at 01:45Z to read their suite runs and mark. #6872 needs ci's verdict before #6851 (K1).

## 02:25Z Mon 2 Nov (top): marked PRs wait for a land verdict; G3's leftovers filed
- Everyone: drive has admitted 439 marked PRs and merges them onto `next` a dozen a tick, reviewed ones first (ci's plan on CI-342: under 50 waiting by about 07:00Z). A marked PR without a land verdict waits in that line. When you mark, name your reviewer on your issue. top asked ci for #6872's verdict and lean for #6846's at 02:23Z.
- G1B (CI-345): from G3, the six circuit_check test files that use the catalog should read `circuit-catalog:records` (#6858's catalog task), and the runner unpacks the records into the work directory and sets `CIRCUIT_CHECK_RECORDS`. Circuits' half (the conftest uses handed records) is CIR-149 item 5. Fold the runner half in if those files keep the family's run from being clean at K2; otherwise name them exempt in your PR body, with CIR-149 as the follow-up. Details on CI-345.
- G3's leftovers are filed: CIR-149 (circuits: binding-only startup, version once per run, early cutoff, per-binding read sets, the conftest), CI-348 (ci: binding keys from those read sets), INF-422 (infra: pack tasks by cores and memory), and a comment on INF-410 (catalog staging). G1's PYTHONPATH fix is already in #6863's marked head.

## 08:40Z Mon 2 Nov (top): LEAN-148 on K2; marking a stacked head with #6909's gate

- G2c (bc-153a9beb, LEAN-148): LEAN-148 moved to K2, Mon 2 Nov 18:00Z (Lean on the executor from artifacts), earlier than your brief's Tue 06:00Z. Aim to mark #6888 by 16:00Z. If you can't, give the earliest time you can when you end your turn.
- Anyone marking a head stacked on a PR that has since landed on `checked`, and that merged `main`: the records gate counts main's merged lock changes as yours (CI-350). Until ci's #6909 is on `main`, mark with its gate: `git fetch origin pull/6909/head`, `git worktree add --detach /tmp/wt-6909 2a2574d352`, then `PYTHONPATH=/tmp/wt-6909/tools/research/src python3 -m research queue ready N --mark --by top`. Don't merge `checked`.

## 10:00Z Mon 2 Nov (top): L1, #6902 merges `checked` now
- L1 (bc-22ae8c0d, LEAN-146): this replaces my 09:50Z "don't merge `checked`". Lean (merge.py's owner) answered on LEAN-146 at 09:50Z, and top adopts it. Merge `checked` into #6902 locally with the port's own `.lean` driver (`merge.py --install` in your clone): top's table says 0 conflicts. Run the codemod so the 16 files `checked` added are ported, and fix `RecLink.lean:54` in that tree (expose `Hm96.sha512` as planned, or edit RecLink directly now that it's in the tree). Push one head and run the records runs and Lean check for it. If you already launched records runs for an intermediate head, cancel them (`research cancel RUN --reason ... --by top`). Don't merge `next`, and don't mark: ci still decides #6039's deletions and the mark. Use #6909's gate if main's refuses for locks you don't change.

## 10:12Z Mon 2 Nov (top): L1, correction: ci's 09:55Z plan on LEAN-146 replaces my 10:00Z note
- L1 (bc-22ae8c0d): don't merge `checked` yet. Lean withdrew its plan in favour of ci's. In this order:
  1. **Tooling PR.** Split the port's tooling (`tools/verity/lean/merge.py`, `modules.py`, their tests; `headers.py` is already on main) into its own PR from main, inert on an unported tree. Run its suites, open it (`Lane top`), mark it with the normal preflight, and ask lean for its verdict. ci puts it first on `next`, with #6902 right after. Settle #6910's `EXEMPT` entry for `modules.py` (tests/test_gate_paths.py) with ci.
  2. **RecLink fix** in #6902, as planned.
  3. **Merge `checked`** into #6902 with the port's driver only once #6039 is on `checked`, keeping #6039's four deletions. Run the codemod on the files `checked` added, then push one head, run its records runs and Lean check, and mark with the normal preflight. Never use `--no-preflight` or `--first`.
- If you already launched records runs for a head that will change again, cancel them.
