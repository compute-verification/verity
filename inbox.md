[2026-11-16T04:12Z] 5b737296abb5
model: claude-opus-5-5-high (Claude Opus 5.5); subscriptions removed: none held

training reader-clean, 08:12 PM PST: Part 1 ready on the scratch branch (pushed)

Part 1 is done on scratch branch cursor/training-reader-scratch-af01 (4a0fe119668 the READMEs, then 1f6172ad82a dropping two stray uv.lock files the first commit took in; both on origin/main d8876d61d61, and they will be cherry-picked as one commit onto the carried head).

Counts (top's sweep scan.py), before and after:
- Training's Markdown sites: 9 -> 0.
- Training's sites in all files: 680 -> 671.
- The whole tree: 42,968 -> 42,959.
The remaining 671 are Lean (610: declarations and the rho symbol, which stay) and Python and TOML (59, Part 2).

What changed, in the 8 Markdown files (7 edited; the rules README needed nothing):
- benchmarks/training_slowdown/README.md:
  - It opens with the Glossary's slowdown (a ratio of training costs, bounded below by T_min/T_ref through a necessary subcomputation's count and a block's yield).
  - The record's slowdown and S = p x it are called the paused attack search's own measurement. "Grading a config" says S is that search's scoring, neither the slowdown nor its lower bound. Nothing the code does was deleted.
  - Commands: the python -m commands now say to run with `uv run --project ../.. python -m training_slowdown.MODULE`. A bare `uv run` in that directory makes its own environment without numpy, which is why six of them failed.
  - `--keep-checkpoints` (gone) is now `--keep-weights`, with the e/ and checkpoints/ layout run.py writes.
  - "Reads" names adapters/torch/verity_torch for the training-step frontend, and verity_vllm only for the Definitions catalog.
  - "window" for the periods' first span is now the execution phase.
- benchmarks/slowdown/README.md:
  - 𝒜′ -> 𝒜; 𝒜_phys -> the physical model bounding 𝓕; envelope/𝒜_a/δ_a -> the subprotocol's admissible set 𝒜_p/δ_p.
  - The certifier subprotocol -> network certification; per-window δ -> per-execution-phase δ.
  - Monitored share -> registered share (`monitored` named); every unit -> every proof block; adversary -> malicious developer; time to the target -> time to the task.
- benchmarks/inference_only/README.md: dropped the line about `--served`, which no longer exists; PoUW row -> PoUW verdict; a unit -> a block; a cheating change and the prover -> a malicious developer.
- benchmarks/nci/README.md:
  - server window -> server block; every unit -> every proof block.
  - attacker, cheater and a determined actor -> malicious developer; the defender -> the auditor; an auditor checking citations -> a reviewer.
  - "under monitoring" -> on registered hardware; a cited Notion page -> the published statement.
- benchmarks/circuit_properties/README.md: a submitter -> a malicious developer.
- security/Definitions/Slowdown/Capability/README.md: "cap", meaning the block yield q, -> yield. These were the lint's 9 sites.
- adapters/torch/verity_torch/frontend/README.md: the names benchmarks/training_slowdown imports now include export_steps and train_trace's CaptureRefused, ATTENTION and BACKWARD.

Commands:
- Every CPU-only, short documented command was run to its help or first output, and runs. That covers training_slowdown's library, of_record, run, capture.circuit, gates, program_reading, as_math, grade and grader; verity bench training-slowdown, inference_only and circuit_properties; slowdown run.py and periods.py, including the min-cut example; and nci bracket, kappa, tiers, fit, statement, library table/data/query and render.
- nci render.py needs `research data fetch-fixtures benchmarks/nci/sources` first, which the README says.
- Node and GPU commands were checked flag by flag against their argparse: the queued attack, run.py --device cuda, inference_only --from/--program/--circuit/--log/--served-class, and grader HANDOVER. All match.
- `verity bench inference_only --from DIR`, tools/check/suites.py and adapters/torch are named as on main.

What stays, with its reason:
- "the inference-only policy" and a policy (c, k): the Glossary's current terms. The carried head's Glossary leaves them; workload classification is the protocol-group sense, which training's text doesn't use.
- The data "windows" of a start's stream (`windows = "fresh"`, `read_windows`, the record's `windows`): a config and record key in a sense no Glossary row retires.
- "window record" and `window_circuit`: the Glossary's narrower sense keeps them.
- `T_0` in the padded size ratio |A|/T_0: the code's quantity (`Slowdown.padded`).
- Plain limits: the memory cap, theta_cap, the caps on rates. nci's rho_T/rho_PoS/rho_NCI. The cheating share χ (tiers.py's name). The `attacker` convention (a key).
- Matrix and table rows; `Certifier` and other code names.

Checks:
- Root lints: tests/test_doc_paths.py, tests/test_retired_terms.py, tests/test_tree.py and tools/move/tests/test_maps.py pass (43), with no warning on training's files.
- tests/lint: its named files pass (64). test_repo_ruff wasn't run, since no Python changed.
- Suites: run r20261116-035209-ae49 is done, rc 0. All 6 named suites passed, none cached: nci 132, slowdown 682, inference_only 63, circuit-properties 157, verity-torch 705, training_slowdown 803.

For top: main's AGENTS.md Map still says "Target, not yet: a torch adapter beside it, which the vLLM adapter sits on." under adapters/, though adapters/torch/ exists. I left it alone.

Part 2 has started on the same branch. 0cbf57117bf covers 17 files of docstrings, comments and printed prose:
- hardware multiple, overhead and time to the target -> slowdown and time to the task;
- per-window δ -> per-execution-phase δ; monitored -> registered and adversary -> malicious developer, in prose;
- nci's fit.py server window -> server block; the Notion page citations in statement.py, tiers.py and test_tiers.py are gone.
Record strings, JSON values (the label "every unit proved"), keys and identifiers stay. Its after-suites run is next, then the rest of the families. Then cursor/training-reader-clean-4207 from the head, re-checking paths against the move.

