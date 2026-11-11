#!/usr/bin/env python3
"""`tools/move/layout_maps.py [--form moves|one|both] [--ref REF] [--targets FILE] [--created FILE]`: write the layout's maps.

    python3 tools/move/layout_maps.py            # regenerates the layout's map on HEAD

It reads `layout_maps.json` beside it (the layout's assignment at one commit: each moved component's new directory and
files, the files archived and the files that stay) and the files REF tracks, and writes the layout as one `layout.py`
map beside itself, where `stage.py` reads a map:

    tools/move/layout_one_map.toml       every row of the layout

`--form moves` writes the same rows as one map a move instead, numbered in the order they run
(`tools/move/layout_NN_NAME_map.toml`), and `--form both` writes both: the two forms move every file to the same place.

It writes nothing unless every check below passes, so it is the one rerun when `main` moves: a tracked file it cannot
place stops it, by name. Then:

    python3 tools/move/layout.py --map tools/move/layout_one_map.toml --dry-run

THE RULE for a path inside a directory. The layout gives each component a directory, which ends in `verifier/`,
`prover/` or `device/` where the component is that part. A moved file's new path is

    its component's new directory / the name of the directory it is in today / its path below that directory

where "the directory it is in today" is the one a person names it by (`named`: the Python package, Lake package or
benchmark directory), and that name is left out where the new directory, less its part, already ends in it
(`verity/core/protocols/pouw/audit.py` goes to `verity/protocols/pouw/verifier/audit.py`, not
`.../verifier/pouw/audit.py`). So a component keeps its own subpath, nothing is renamed, and no new path keeps a `core`
or `ml` segment. An archived file goes to `archive/` under its whole path today.

A module's circuit-check sidecar and its pins (`X.circuit-check.py`, `X.circuit-check.json`) go where `X.py` goes,
whichever component they are in: circuit-check finds a module's bindings beside it.

A SUITE MOVES WHOLE. A `pyproject.toml` that configures pytest is a suite, and every file under its test directories
goes where the pyproject goes (or stays where it stays), whichever component it is in: no map splits a suite, and
splitting one is its owner's change after the moves. The script names each place where that puts one component's
tests in another's directory ("with its suite"), and the trust check lets a suite's tests go with it. A test whose
code the layout sends into another suite goes into that suite with it, by a row of `EXCEPT` for the test. An archived
test stays archived.

AN EMPTIED PACKAGE LEVEL. The `__init__.py` of a package level the moves empty (every other file under it leaves, and
none arrives) stays where it is and gets no row, whichever component it is in and whether or not the assignment
archives it; the script names each one, for a commit that deletes them after the moves. A row to `archive/` would
rename the package, and every mention of it that is left, to `archive.…`; a row that took it along by the rule would
rename the package under the modules other rows take out of it.

EXCEPTIONS, each a row of `EXCEPT`, `KEEP` (a component, a file or a directory that stays), `DROP` (a file a move
deletes) or `ALIASES` (a name a moved module is imported by that its path does not give) with its reason: where the rule gives two files one path or two paths
that differ only by case, puts a file nobody trusts in a trusted directory, takes a module out of its import package,
or gives a path nobody would look for. A row the rule itself now gives is reported as doing nothing.

A FILE THE DATA DOESN'T KNOW (tracked at REF, added since the data was cut) goes with the files beside it: a directory
that goes one way takes it along; in a directory several components share it goes with the one component that holds
the other files of that directory, archived files aside, and is printed; where that isn't one component the script
stops and names it, for a row in `NEW`. A path of the data that REF no longer tracks is dropped and reported.

KINDS, each treated one way:
- Python packages. A whole directory is one directory row; a directory whose files part ways gets one row a file (or a
  subdirectory). `[modules]` is what `layout.py` derives from `[paths]` and `[roots]`, so imports, dotted names and
  `-m` names follow the files. `layout.py` adds an empty `__init__.py` for each new package level a moved module needs
  and no moved file brings (its "files created"; `--created FILE` lists them).
- Lake packages. Each moves whole, by a directory row at or above its root or a `[tokens]` row for it, so a lakefile's
  or a manifest's relative `path` to it follows; a Lake package inside another has rows of its own. Lean module names
  don't change, and no `.lean` file's bytes do (`EXCLUDE`): a module's source is in its build key, so a path rewritten
  in a comment would compile the module again, and every module that imports it. A `.lean` file that writes a path
  outside a comment is rewritten, by a row of `LEAN_PATHS`.
- Distributions with their own `pyproject.toml`. The pyproject moves with its component and the distribution keeps its
  name and import name (`ROOTS`).
- Tests go with their suite; plain files go with their component, by the rule.

A DIRECTORY NO ROW MOVES WHOLE is named in `[tokens]`, so a reference to the directory itself (a suite's declared
input, a boundary's glob) is rewritten and none is left naming a directory that is gone. In a map a move, the map that
takes the directory's `pyproject.toml` (else its `__init__.py`, its `lakefile.toml`, else most of its files) speaks
for it. Where one map empties a directory and its files part ways, a reference reads as the directory its files keep
together in but for the archived ones, a file with a row of its own and, in a Lake package, a Lake package inside it
with a row of its own (`EXCEPT`), which goes its own way; else as the one new directory that holds all
of them but the archived, where that is narrower than a top-level directory; else, in the one map, as where the
directory's package goes. A directory a move's map speaks for reads the same in the one map: a reference to it, and a
bare `Path(__file__).parent` in a file under it, then follow the files of the move that holds most of it, as they do
when the moves run one by one. `layout.py` then wants each reference to a directory that splits acknowledged in
`[split]`, by file and directory. The one
map's `[split]` rows are what `layout.py`'s own scan of the worktree finds (minutes, and REF must be HEAD); nobody has
read them, and each says so. `verity/core` and `verity/ml` themselves get no row: no one new directory holds them.

ORDER. The one map has none. The maps a move run in the order of their names: each names today's paths only, no map
has a row at or under another map's row, and no destination lies under any map's row, so the files land the same in
any order. Module names do not: a map that moves a package's `__init__.py` renames the package, and would rename with
it the modules another map takes out of that package. So that other map runs first: the order is derived from the
maps, their names carry it, and each map's header names the maps that run before it.

RETIRED. The one map's `[retired]` names every directory it empties that no directory row holds, the outermost of
them: every tracked file under it leaves and none arrives. `layout.py` refuses a tracked file under one that has no
row, so a file that lands there after the map was cut stops the move by name, where it would have stayed behind
unsaid. A directory with a row needs none: the row carries the file. A directory whose emptied `__init__.py` stays
for the later commit is not emptied by the map and is not named. In a move's map `[retired]` is empty: a directory
two maps share can be retired by neither while each map passes alone.

CHECKS (the script fails on any): every tracked file is one the data places or `NEW` names; every file the layout
moves or archives has exactly one destination, and each map moves exactly its own files; no destination is claimed
twice or is a file that stays; no two paths of the moved tree differ only by case; no new path outside `archive/`
keeps a `core` or `ml` segment; every tracked file under `verity/core` or `verity/ml` moves, but for the emptied
`__init__.py`s; no file of a component the auditor doesn't trust lands under a trusted directory, but for `UNTRUSTED`'s
and a suite's tests; no `pyproject.toml` lies between the root and a workspace member whose sources name the workspace
(uv finds a member's workspace at the first `pyproject.toml` above it, so `uv lock` would fail on the moved tree); every
test file outside `archive/` is under a suite's test directory after the moves; every moved
Lake package moves whole; `[modules]` gives every moved module its new name and renames no module that stays; the
order has no cycle.
"""
import argparse
import collections
import json
import os
import posixpath
import re
import subprocess
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import layout as L  # noqa: E402
import layout_case  # noqa: E402

#: (today's path: a file, or a directory ending in "/"; its new path; why the rule is not followed)
EXCEPT = [
    ("verity/ml/flock/verifier/", "verity/protocols/flock/agreement/",
     "the agreement with upstream's verifier, which nobody trusts: the rule puts its scripts in "
     "`verity/protocols/flock/verifier/`, the trusted root of the Lean verifier's Lake package, and its README on that "
     "package's"),
    ("verity/ml/flock/python/verity_flock/", "verity/protocols/flock/prover/python/verity_flock/",
     "a module goes with its import package: eleven bench drivers are modules of `verity_flock`, which the rule would "
     "leave in a second `verity/protocols/flock/python/verity_flock/` with no `__init__.py`, outside the distribution "
     "that keeps the name"),
    ("verity/core/service/spot/", "verity/protocols/spot/verifier/lean/",
     "SPOT's Lake package: the rule puts its `pyproject.toml` on the Python verifier's in "
     "`verity/protocols/spot/verifier/`; the Lean takes `lean/`, as each protocol's Lean is its `lean/`"),
    ("verity/core/service/vstar/recursive/", "verity/protocols/recursion/verifier/recursive/",
     "the recursive audit's Lake package: the rule gives `verity/protocols/recursion/verifier/vstar/recursive/`, where "
     "`vstar/` names only the command package's directory it sits in today"),
    ("verity/core/protocols/record.py", "verity/hardware/record.py",
     "the rule gives `verity/hardware/protocols/record.py`: a `protocols/` segment that names nothing, and a package "
     "whose `__init__.py` no moved file brings"),
    ("verity/core/protocols/registration.py", "verity/hardware/registration.py",
     "as `record.py`: the rule gives `verity/hardware/protocols/registration.py`"),
    ("verity/core/protocols/profile.py", "verity/protocols/spot/verifier/profile.py",
     "the integrity profile: the rule gives `verity/protocols/spot/verifier/protocols/profile.py`, a second `protocols/` "
     "segment that names nothing"),
    ("integrations/vllm/verity_vllm/", "adapters/vllm/verity_vllm/",
     "a module stays in its import package: the rule names a module of a subpackage by that subpackage and gives "
     "`adapters/vllm/commitments/`, beside `verity_vllm` and in no package, for `verity_vllm/commitments/`. The adapter's "
     "directory moves whole and the distribution keeps the name `verity_vllm`"),
    ("verity/ml/commitments/leaves.py", "adapters/vllm/verity_vllm/commitments/leaves.py",
     "a module goes into an import package: the rule gives `adapters/vllm/commitments/`, beside `verity_vllm` and "
     "in no package, where it would be imported as `adapters.vllm.commitments.leaves`"),
    ("verity/ml/commitments/vllm_v1/", "adapters/vllm/verity_vllm/commitments/vllm_v1/",
     "as `leaves.py`: the commitment scheme named for vLLM goes into `verity_vllm`"),
    ("verity/core/devices/memory_challenger/", "verity/protocols/pous/device/",
     "the memory challenger's package, one directory row: the rule puts it in `device/memory_challenger/`, beside the "
     "module `memory_challenger.py`, two things under one import name. The row takes along one archived file inside "
     "the package's Lake package (`lean/scripts/replay.py`), which a row of its own would split"),
    ("verity/core/protocols/remote_bandwidth/", "verity/protocols/poub/verifier/",
     "PoUB's reference, session and Lean verifier, one directory row: the rule names the protocol twice "
     "(`poub/verifier/remote_bandwidth/`)"),
    ("examples/README.md", "demos/README.md",
     "the demos' index: the rule gives `demos/dither_full_util/examples/README.md`, inside one demo"),
    ("verity/core/service/vstar/", "verity/protocols/recursion/verifier/vstar_cli/",
     "V*'s command package: the rule gives `verity/protocols/recursion/verifier/vstar/`, the name V*'s own Lean has in "
     "Flock's verifier (`VStar/`, which stays in that Lake package); `vstar_cli/` says this one is the command"),
    ("verity/core/protocols/certifier/lean/", "verity/protocols/certifier/device/program/lean/",
     "the certifier program's Lake package: the rule puts it on `device/lean/`, where the certifier device's Lake "
     "package goes (two lakefiles, manifests, locks and toolchains on one path)"),
    ("verity/ml/commitments/tests/test_leaves.py", "adapters/vllm/tests/commitments/test_leaves.py",
     "the test of `leaves.py`, which goes into the adapter: it goes into the adapter's suite with the module it "
     "tests; left with its own suite, a test under `verity/` would import an adapter"),
    ("verity/ml/commitments/tests/test_vllm_v1.py", "adapters/vllm/tests/commitments/test_vllm_v1.py",
     "the test of `vllm_v1/`, which goes into the adapter: with the code it tests, as `test_leaves.py`"),
    ("examples/h200/e2e_h200/served_weights.py", "benchmarks/torch_gpu/p2_stage/served_weights.py",
     "the served weights' reader goes beside the GPU stage that reads it; the demo's other files stay"),
    ("examples/h200/interfaces/", "benchmarks/one_stage/interfaces/",
     "the drawn-unit replay bundle and the block record, with their schemas, fixtures and suite, go beside the one-stage "
     "audit, their main reader, which is live; the demo's other files stay"),
    ("verity/core/protocols/pyproject.toml", "verity/protocols/tests/pyproject.toml",
     "the suite of the protocols' own two tests, whole, in a directory with no workspace member below it (its tests go "
     "with it, to `verity/protocols/tests/tests/`): the rule gives `verity/protocols/pyproject.toml`, between the root "
     "and Flock's prover, and uv finds a member's workspace at the first `pyproject.toml` above it. `verity/hardware/`, "
     "where `record.py` goes, has a suite of its own on that path"),
    ("verity/core/protocols/pous/wipe_stream.c", "verity/protocols/pous/verifier/wipe_stream.c",
     "the streamed wipe's native fill goes beside `wipe_stream.py`, the one module that builds it, by the name of the file "
     "beside itself; the rule parts the two (`device/wipe_stream.c`)"),
]
#: an EXCEPT row whose files go with another move than their component's: {the row's path today: the move}
EXCEPT_MOVE = {
    "verity/ml/flock/python/verity_flock/": "flock_prover",
    "verity/core/devices/memory_challenger/": "pous",
    "verity/core/protocols/remote_bandwidth/": "poub",
    "examples/h200/e2e_h200/served_weights.py": "archive",
    "examples/h200/interfaces/": "archive",
}
#: components the layout gives a directory that stay where they are: {component id: why}
KEEP = {
    "bench:31": "the sampling-plan calculators and figures stay in `benchmarks/`: moved under `verity/`, two of their "
                "scripts would read `benchmarks/census` by path",
    "bench:63": "PoUW's calculators stay in `benchmarks/pouw/`: nothing under `verity/` or an integration imports them, "
                "and one of them loads `benchmarks/sampling_plan`",
    "bench:30": "the recursion end-to-end run stays in `benchmarks/`: `verity bench NAME` runs `benchmarks/NAME`'s runner, "
                "which its `run.py` is, and a directory under `demos/` is a demo with a `demo.toml`, which it has not",
    "benchmarks/network_traces/":
        "its files find each other by bare name, with the directory on `sys.path`, and a move leaves a bare name as "
        "written: the directory stays whole, its archived files too, until its owner imports by package name and splits it",
}
#: a name a moved module is imported by that its path does not give: {the name: (the move, the module's new name, why)}
ALIASES = {
    "e2e_h200.served_weights":
        ("archive", "benchmarks.torch_gpu.p2_stage.served_weights",
         "its readers put `examples/h200` on `sys.path` and import it as `e2e_h200.served_weights`"),
}
#: files a move deletes: {path: (the move, why)}
DROP = {
    "benchmarks/pouw/contents/paths/np_kernel.toml":
        ("archive", "the entry of `benchmarks/pouw`'s contents for `np_kernel/`, which goes to `archive/`"),
    "benchmarks/pouw/contents/paths/clear.toml":
        ("pouw", "the entry of `benchmarks/pouw`'s contents for `clear/`, which leaves `benchmarks/pouw`"),
    "benchmarks/pouw/contents/filler/receipts.toml":
        ("pouw", "the entry of `benchmarks/pouw`'s filler table for `../receipts/`, which leaves `benchmarks/pouw`"),
}
#: files of a component the auditor doesn't trust that land under a trusted directory: {component id: why it may}
UNTRUSTED = {
    "core:zk-server": "the ZK server's half of Flock's Lake package (`Flock.Firewall`), which the developer trusts and the "
                      "auditor doesn't: a Lake package moves whole",
    "core:zk-server.release": "the ZK server's release side, which the developer trusts and the auditor doesn't. Its Lake "
                              "package requires the core protocol interface, which requires Flock's verifier, so it does "
                              "not go inside Flock's directory: the layout puts it with the recursion layer",
    "edge:tools.config": "the settings package (`verity_config`), which the layout puts with the Configuration it reads",
    "ml:windows": "Window, what an adapter hands Verity, which no verifier imports: the layout puts it under the "
                  "commitments its windows are committed in, as a package and a suite of its own (`windows/`)",
}
#: tracked files the data doesn't know, in a directory several components share, and files it places otherwise:
#: {path, or directory ending in "/": the component id it goes with, "stays" or "archived"}
NEW = {
    "verity/ml/commitments/history.py": "ml:commitment-schemes",
    "verity/ml/commitments/tests/test_history.py": "ml:commitment-schemes",
    "archive/AGENTS.md": "stays",                      # what `archive/` is, there before the archive's files arrive
    "verity/core/protocols/spot/schedule.py": "core:spot-python",       # `second_stage.py` imports it
    "verity/core/service/flock/Flock/TmplLines.lean": "core:flock-lean-verifier",
    "verity/ml/definitions/registry/dsa_topk_mask_cut.py": "ml:definitions-registry",
    "verity/ml/definitions/registry/dsa_topk_mask_cut.circuit-check.py": "ml:definitions-registry",
    "verity/ml/definitions/registry/dsa_topk_mask_cut.circuit-check.json": "ml:definitions-registry",
    "verity/ml/definitions/tests/test_dsa_topk_mask_cut.py": "ml:definitions-registry",
    "verity/ml/definitions/registry/fa2_train.py": "ml:definitions-torch",          # a training step's attention
    "verity/ml/definitions/registry/fa2_train.circuit-check.py": "ml:definitions-torch",
    "verity/ml/definitions/registry/boolean_fa2_train.py": "ml:definitions-torch",  # its Boolean version, as `boolean_torch_words`
    "verity/ml/definitions/registry/boolean_fa2_train.circuit-check.py": "ml:definitions-torch",
    "verity/ml/definitions/tests/test_fa2_train.py": "ml:definitions-torch",
    "verity/core/service/flock/Flock/RegBound.lean": "core:flock-lean-verifier",
    "verity/ml/flock/tests/test_fmaxf_commit.py": "ml:flock-lowering",
    "verity/ml/commitments/gates/sha512_stream.py": "ml:commitment-gates",          # SHA-512 over a stream, as gates
    "verity/ml/commitments/gates/sha512_stream.circuit-check.py": "ml:commitment-gates",
    "verity/ml/commitments/gates/sha512_stream.circuit-check.json": "ml:commitment-gates",
    "verity/ml/commitments/tests/test_boolean_sha512_stream.py": "ml:commitment-gates",
    "verity/ml/commitments/keyed.py": "ml:commitment-schemes",          # keyed leaves, a scheme beside `history.py`
    "verity/ml/commitments/tests/test_keyed.py": "ml:commitment-schemes",
    "verity/ml/crypto/transport.py": "ml:aead",        # the AEAD's transport, tested in the commitments' suite
    "verity/ml/commitments/tests/test_transport.py": "ml:aead",
    "verity/ml/definitions/registry/linear_cut.py": "ml:definitions-registry",
    "verity/ml/definitions/registry/linear_cut.circuit-check.py": "ml:definitions-registry",
    "verity/ml/definitions/registry/linear_cut.circuit-check.json": "ml:definitions-registry",
    "verity/ml/definitions/tests/test_linear_cut.py": "ml:definitions-registry",
    "verity/ml/definitions/registry/pearl_c_tile.py": "ml:definitions-registry",
    "verity/ml/definitions/registry/pearl_c_tile.circuit-check.py": "ml:definitions-registry",
    "verity/ml/definitions/registry/pearl_c_tile.circuit-check.json": "ml:definitions-registry",
    "verity/ml/circuits/ops/tests/test_boolean_rolled.py": "ml:ops-boolean",        # tests `ops/boolean/rolled.py`
    "verity/ml/flock/tests/test_rolled_lowering.py": "ml:flock-lowering",           # the roll's lowering
    "verity/ml/definitions/tests/test_sparse_mla_combine_columns.py": "ml:definitions-registry",   # tests `sparse_mla_split`
    "verity/ml/definitions/numerics/token_select_rows.py": "ml:definitions-numerics",              # a row kernel, as `stop_rows.py`
    "benchmarks/memory/": "stays",                     # a benchmark of its own
    "benchmarks/pouw/pearlplus_lean/": "stays",        # a benchmark's Lean, in a directory part of which is archived
    "benchmarks/network_traces/tests/test_scan_zk_v3c.py": "stays",
    "benchmarks/pouw/np_kernel/": "archived",          # its owner archives the kernel: only its test imports it,
    "benchmarks/pouw/tests/test_np_kernel.py": "archived",              # and the test goes with it
}
#: directories written as one row though a file of theirs is archived, so that a file added to the directory, or one it
#: fetches, goes with it: {the directory today: why}
WHOLE = {
    "benchmarks/nci/": "the inference-only policy's benchmark fetches `sources/` and gains files; one row carries them",
    "integrations/vllm/": "the adapter gains files with every change to it; one row carries them",
}
#: references to a directory a map splits that are right after the move: (the move, "old file: old directory", why)
SPLIT = [
    ("flock_prover", f"verity/ml/flock/live/src/{name}.rs: verity",
     "`../../../..` from the crate's directory (`CARGO_MANIFEST_DIR`) is the repository's root. The crate is one level "
     "deeper in `verity/protocols/flock/prover/live` than in `verity/ml/flock/live`, and the rewrite adds the one `..` "
     "that keeps it the root: it reads the path from the file's directory, as `verity`, which is as much deeper")
    for name in ("circuit", "firewall", "lookup", "sha512_native")
]
#: import name -> its package directory today; a module outside every root is named by its path from the root
ROOTS = {
    "verity": "verity", "verity_difftest": "verity/core/difftest/python/verity_difftest",
    "verity_flock": "verity/ml/flock/python/verity_flock", "verity_numerical": "benchmarks/numerical/python/verity_numerical",
    "verity_vllm": "integrations/vllm/verity_vllm", "research": "tools/research/src/research",
    "circuit_check": "tools/verity/circuit_check/src/circuit_check", "cluster": "infra/cluster/src/cluster",
    "verity_cli": "tools/verity/cli/verity_cli", "verity_config": "tools/config/verity_config",
    "verity_machine_model": "benchmarks/pouw/machine_model/verity_machine_model", "datacenter": "benchmarks/datacenter/datacenter",
}
#: the moves, in the order asked for where nothing else decides: (map name, what it is)
MOVES = [
    ("circuits", "the circuit language, to `verity/circuits`"),
    ("definitions", "the Definitions and their host kernels, to `verity/definitions`"),
    ("interface", "the core protocol interface, to `verity/protocols/interface`"),
    ("flock_verifier", "Flock's Lean package, to `verity/protocols/flock/verifier`"),
    ("flock_prover", "Flock's prover, to `verity/protocols/flock/prover`"),
    ("flock", "Flock's tests, pod scripts and agreement with upstream, to `verity/protocols/flock`"),
    ("spot", "sampled proofs, to `verity/protocols/spot`"),
    ("recursion", "V*'s command package, sampled proofs' recursive audit and the ZK server's release side, to "
                  "`verity/protocols/recursion`"),
    ("pouw", "PoUW, to `verity/protocols/pouw`"),
    ("pous", "PoUS, to `verity/protocols/pous`"),
    ("poub", "PoUB, to `verity/protocols/poub`"),
    ("certifier", "network certification, to `verity/protocols/certifier`"),
    ("inference_only", "the inference-only policy, to `verity/protocols/inference_only`"),
    ("remote", "remote Verity and the composition of core protocols, to `verity/protocols/remote`"),
    ("onsite", "onsite Verity, to `verity/protocols/onsite`"),
    ("composite", "the auditor host's challenger service, to `verity/composite`"),
    ("commitments", "commitments, hashes, signatures, the challenge derivation and Window, to `verity/commitments`"),
    ("hardware", "World, Site and registration, to `verity/hardware`"),
    ("configuration", "the Configuration, the wire and the settings package, to `verity/configuration`"),
    ("difftest", "the difftest harness, to `verity/difftest`"),
    ("verity", "the claims' ids and the reporting grid, to `verity`"),
    ("vllm", "the vLLM adapter, with the commitment scheme named for vLLM, to `adapters/vllm`"),
    ("demos", "the demos, to `demos`"),
    ("benchmarks", "the GPU probes that are Definitions today, to `benchmarks`"),
    ("archive", "what the layout archives, to `archive`"),
]
EXCLUDE = [
    "fixtures/**", "**/fixtures/**", "**/*vectors*.json", "**/*vectors*.txt.gz", "**/CHANGELOG.md", "*.patch",
    "**/TRUSTED.sha256", "**/lake-manifest.json.lock", "tools/move/**", "archive/**",
    "benchmarks/numerical/reports/**", "benchmarks/numerical/tests/bench/data/**", "**/price_twins.json",
    "verity/ml/circuits/silicon/captures/**", "verity/ml/circuits/silicon/hawkeye/**",
    "verity/definitions/silicon/captures/**", "verity/definitions/silicon/hawkeye/**",
    "integrations/vllm/manifests/**", "integrations/vllm/workloads/**", "integrations/vllm/data/**",
    "integrations/vllm/verity_vllm/pins/**", "tools/verity/check/tests/test_suites.py",
    "tools/verity/check/tests/test_move_check.py", "tools/research/src/research/store/moves.json",
    "benchmarks/tc_probe_fp4/fp4_emulation_1ad1aaa2.py", "benchmarks/network_e2e/tests/data/**",
    "benchmarks/hash/dither/j10b_ref.py", "benchmarks/hash/dither/j10ca_ref.py", "benchmarks/hash/dither/j10hb.py",
    "benchmarks/hash/dither/j10rtx.py", "benchmarks/hash/dither/j10served.py",
    "integrations/vllm/tools/migrate/superseded_quarantine.py", "integrations/vllm/tests/migrate/test_superseded_quarantine.py",
    "adapters/vllm/manifests/**", "adapters/vllm/workloads/**", "adapters/vllm/data/**", "adapters/vllm/verity_vllm/pins/**",
    "adapters/vllm/tools/migrate/superseded_quarantine.py", "adapters/vllm/tests/migrate/test_superseded_quarantine.py",
    "**/*.lean",
]
#: the `.lean` files the rewrite still reads: each writes a path outside a comment, by today's path and by its new one
LEAN_PATHS = [
    "verity/ml/flock/tests/lean/RandomnessVectors.lean", "verity/protocols/flock/prover/tests/lean/RandomnessVectors.lean",
    "verity/core/protocols/spot/tests/lean/ExfiltrationVectors.lean",
    "verity/protocols/spot/verifier/tests/lean/ExfiltrationVectors.lean",
    "verity/core/protocols/spot/tests/lean/ExfiltrationTwoVectors.lean",
    "verity/protocols/spot/verifier/tests/lean/ExfiltrationTwoVectors.lean",
]
INCLUDE = ["fixtures/artifacts.json", "**/fixtures/README.md", "**/fixtures/**/README.md", "**/lake-manifest.json",
           "integrations/vllm/manifests/definitions.json", "adapters/vllm/manifests/definitions.json", *LEAN_PATHS]
NAMESAKE = ("pyproject.toml", "__init__.py", "lakefile.toml")
MAP_NAME = re.compile(r"layout_(\d\d_\w+|one)_map\.toml")
#: the two forms of the layout
FORMS = ("moves", "one")
ONE = ("layout", "the whole layout")
TEST_FILE = re.compile(r"(?:.*/)?(?:test_[^/]*|[^/]*_test)\.py")
#: where no suite collects a test file and none need (`tests/test_repository.py`'s `UNSUITED_TEST_DIRS`)
UNSUITED = ("verity/ml/kernels/", "archive/")
SCANNED = ("found by `layout.py`'s own scan and read by nobody: the reference follows the directory's `[tokens]` row, "
           "where its package goes, and part of the directory goes elsewhere")
SIDECAR = re.compile(r"(.+)\.circuit-check\.(?:py|json)")


def die(msg):
    raise SystemExit("layout_maps: " + msg)


def git(repo, *args):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if r.returncode:
        die(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def named(p):
    """The directory a person names `p` by today: its Python package, Lake package or benchmark directory."""
    parts = p.split("/")
    stem, parts = parts[-1], parts[:-1]
    if not parts:
        return "."
    t = parts[0]

    def take(n):
        return "/".join(parts[:n])
    if t == "Security":
        if len(parts) == 1:
            return "Security"
        if parts[1] in ("Definitions", "Proofs", "Properties"):
            if len(parts) >= 3 and parts[1] == "Proofs" and parts[2] == "Properties":
                if len(parts) >= 4:
                    return take(4)
                return take(3) + "/" + stem[:-5] if stem.endswith(".lean") else take(3)
            if len(parts) >= 3:
                return take(3)
            return take(2) + "/" + stem[:-5] if stem.endswith(".lean") else take(2)
        return take(2)
    if t == "verity":
        if len(parts) >= 3 and parts[1] == "core" and parts[2] in ("protocols", "service", "primitives", "devices"):
            if len(parts) >= 5 and parts[2] == "protocols" and parts[3] == "network" and parts[4] in ("nci", "lean"):
                return take(5)
            return take(4) if len(parts) >= 4 else take(3)
        if len(parts) >= 3 and parts[1] == "ml" and parts[2] == "protocols":
            return take(4) if len(parts) >= 4 else take(3)
        if len(parts) >= 4 and parts[1] == "ml" and parts[2] == "circuits" and parts[3] in ("ops", "silicon"):
            return take(4)
        return take(3) if len(parts) >= 3 else take(len(parts))
    if t == "integrations":
        if len(parts) >= 4 and parts[2] == "verity_vllm":
            return take(4)
        return take(3) if len(parts) >= 3 else take(len(parts))
    if t == "tools":
        if len(parts) >= 3 and parts[1] == "verity":
            return take(3)
        return take(2) if len(parts) >= 2 else "tools"
    if t in ("benchmarks", "examples", "infra", "fixtures"):
        return take(2) if len(parts) >= 2 else t
    return t


def common_dir(paths):
    """The deepest directory that holds every one of `paths`."""
    out = []
    for segs in zip(*(p.split("/")[:-1] for p in paths)):
        if len(set(segs)) != 1:
            break
        out.append(segs[0])
    return "/".join(out)


def move_of(cdir):
    """The move (map) that fills a component's new directory."""
    s = cdir.strip("/").split("/")
    if s[0] == "verity":
        if len(s) == 1:
            return "verity"
        if s[1] == "protocols":
            return f"flock_{s[3]}" if s[2] == "flock" and len(s) > 3 else s[2]
        return s[1]
    return s[1] if s[0] == "adapters" else s[0]


def ancestors(d):
    out = []
    while d:
        d = posixpath.dirname(d)
        if d:
            out.append(d)
    return out


def holds(row, p):
    return p == row or (row.endswith("/") and p.startswith(row))


def excepted(p):
    """`p`'s EXCEPT row (the longest that holds it), its new path by it and that row's move if it names one, or Nones."""
    best = None
    for old, new, _ in EXCEPT:
        if holds(old, p) and (best is None or len(old) > len(best[0])):
            best = (old, new)
    return (None, None, None) if best is None else (best[0], best[1] + p[len(best[0]):], EXCEPT_MOVE.get(best[0]))


def own_row(p):
    """`p`'s EXCEPT row where that row is one a map always writes for itself, else None: a file's own row, or the row
    of a directory that leaves an archived directory, which then still goes to `archive/` as one row."""
    k, _, move = excepted(p)
    return k if k is not None and (k == p or move == "archive") else None


def kept(p):
    """Whether a path row of `KEEP` (a file, or a directory ending in "/") holds `p`."""
    return any(holds(k, p) for k in KEEP if "/" in k)


def place(data, tracked, say):
    """{tracked file: a component's id, "stays" or "archived"}: by the data's paths, `NEW`, or the files beside it."""
    entry = {}
    for key, paths in [(c["id"], c["paths"]) for c in data["components"]] + [("archived", data["archived"]), ("stays", data["stays"])]:
        for p in paths:
            if p in entry:
                die(f"layout_maps.json names {p} twice: for {entry[p]} and for {key}")
            entry[p] = key
    known = {c["id"] for c in data["components"]} | {"stays", "archived"}
    for p, key in NEW.items():
        if key not in known:
            die(f"NEW: {p} goes with {key}, which is no component of layout_maps.json")
        entry[p] = key
    used, owner, unknown = set(), {}, []
    for p in sorted(tracked):
        e, d = (p if p in entry else None), posixpath.dirname(p)
        while e is None and d:
            e, d = (d + "/" if d + "/" in entry else None), posixpath.dirname(d)
        if e is None:
            unknown.append(p)
        else:
            used.add(e)
            owner[p] = entry[e]
    beside = collections.defaultdict(set)              # directory -> who holds the data's files directly in it
    for p, key in owner.items():
        if p in entry and p not in NEW:
            beside[posixpath.dirname(p)].add(key)
    stuck = []
    for p in unknown:
        keys = beside.get(posixpath.dirname(p), set())
        live = keys - {"archived"}                     # the archive speaks for no directory: a file beside an archived one
        if len(live) == 1:                             # goes with the one component that still lives there
            owner[p] = next(iter(live))
            say(f"new since the data was cut: {p} goes with the other files of its directory ({owner[p]})")
        else:
            stuck.append(f"{p}: " + ("its directory holds no file the data knows" if not keys else
                                     "its directory's files are " + ", ".join(sorted(keys))))
    if stuck:
        die("tracked files the data doesn't place, each in a directory several components share: give each a row in NEW "
            "(a component's id, \"stays\" or \"archived\"), or one row for its directory\n  " + "\n  ".join(stuck))
    for e in sorted(set(entry) - used):
        say(f"dropped, {e} is no longer tracked: a path of {entry[e]}")
    return owner


def plan(data, owner, say, suites=()):
    """({today's path: (new path, component id, move)} for every file a map moves, the emptied `__init__.py`s that stay,
    the files that go with their suite and not by the rule). `suites` is [(a suite's pyproject.toml, its test directories)]."""
    comps = {c["id"]: c for c in data["components"]}
    parts = set(data["parts"])
    out, plain = {}, {}
    for p, key in owner.items():
        row, new, mv = excepted(p)
        if key == "stays" or key in KEEP or kept(p):
            rule, move = None, None
        elif key == "archived":
            rule, move = "archive/" + p, "archive"
        else:
            c = comps[key]
            cdir = c["to"]
            if c["part"] and not cdir.endswith("/" + c["part"]):
                die(f"{key}: its part is {c['part']} and its directory {cdir} does not end in it")
            segs = cdir.split("/")
            while len(segs) > 1 and segs[-1] in parts:
                segs = segs[:-1]
            nd = named(p)
            if not (nd == "." or p.startswith(nd + "/")):
                die(f"{p}: not under the directory it is named by, {nd}")
            below = p if nd == "." else p[len(nd) + 1:]
            base = nd.split("/")[-1]
            rule, move = f"{cdir}/{below if nd == '.' or base == segs[-1] else base + '/' + below}", move_of(cdir)
        if row is not None:
            plain.setdefault(row, []).append(rule == new)
            if (mv or move) is None:
                die(f"{p}: the layout keeps it, so its EXCEPT row needs a move in EXCEPT_MOVE")
            out[p] = (new, key, mv or move)
        elif rule is not None:
            out[p] = (rule, key, move)
    for p in sorted(owner):                           # a module's circuit-check sidecar and its pins go where the module goes
        if (m := SIDECAR.fullmatch(p)) and (mod := m.group(1) + ".py") in owner and excepted(p)[0] is None:
            there = out.get(mod)
            new = None if there is None else f"{posixpath.dirname(there[0])}/{posixpath.basename(p)}"
            if (out.get(p) or (None,))[0] != new:
                say(f"with its module: {p} goes to {new or 'nowhere (its module stays)'}, not {(out.get(p) or ('nowhere',))[0]}")
                if new is None:
                    del out[p]
                else:
                    out[p] = (new, owner[p], there[2])
    tests = {t: pp for pp, tdirs in suites for t in tdirs}      # a suite moves whole: its tests go where its pyproject goes
    with_suite, suited = collections.Counter(), set()
    for p in sorted(owner):
        t = posixpath.dirname(p)
        while t and t not in tests:
            t = posixpath.dirname(t)
        if not t or (owner[p] == "archived" and not kept(p)) or excepted(p)[0] == p:   # a row of its own: with the code it tests
            continue
        pp = tests[t]
        pdir = posixpath.dirname(pp)
        if t == pdir and not TEST_FILE.fullmatch(p):   # a suite that collects its own directory: only its test files
            continue
        there = out.get(pp)
        new = p if there is None else posixpath.dirname(there[0]) + p[len(pdir):]
        was = out[p][0] if p in out else p
        if new != was:
            with_suite[(owner[p], posixpath.dirname(was), posixpath.dirname(new))] += 1
            suited.add(p)
            if new == p:
                del out[p]
            else:
                out[p] = (new, owner[p], there[2])
    for (key, was, new), n in sorted(with_suite.items()):
        say(f"with its suite: {n} file{'s' if n > 1 else ''} of {key} go{'' if n > 1 else 'es'} to {new}, not {was}")
    lands = {posixpath.dirname(n) for n, _, _ in out.values()}
    lands |= {a for d in list(lands) for a in ancestors(d)}
    under = collections.Counter(a for p in owner if p not in out for a in ancestors(posixpath.dirname(p)))
    under.update(posixpath.dirname(p) for p in owner if p not in out)
    inside = collections.defaultdict(list)             # directory -> the tracked files anywhere under it
    for p in owner:
        for a in ancestors(p):
            inside[a].append(p)
    emptied = []
    for p in sorted(owner, key=lambda p: -p.count("/")):      # the deepest first: a level is empty once those below it are
        d = posixpath.dirname(p)
        if posixpath.basename(p) != "__init__.py" or d in lands or excepted(p)[0] is not None:
            continue
        if p not in out:
            empty = under[d] == 1                      # it stays, and every other file under it leaves
        else:
            there, move = posixpath.dirname(out[p][0]) + "/", out[p][2]
            with_it = [q for q in inside[d] if q != p and q in out and out[q][0].startswith(there)]
            # it goes by the rule and no other file of the package goes with it; or another map takes a module of the
            # package to the same place (that map runs first, `layout.py` creates this `__init__.py` for it, and the row
            # would land on the created file); or it is archived while a module of its package is not, and the row would
            # rename the package, and every mention of it that is left, to `archive.…`
            live = any(q.endswith(".py") and q != p and q in out and out[q][2] != "archive" for q in inside[d])
            empty = ((len(inside[d]) > 1 and not with_it) or any(q.endswith(".py") and out[q][2] != move for q in with_it)
                     or (move == "archive" and live))
        if empty:
            if p in out:
                del out[p]
            else:
                for a in (d, *ancestors(d)):
                    under[a] -= 1
            emptied.append(p)
    for old, new, _ in EXCEPT:
        if old not in plain:
            say(f"dropped, EXCEPT's row {old} holds no tracked file")
        elif all(plain[old]):
            say(f"EXCEPT's row {old} does nothing now: the rule gives {new}")
    for key in KEEP:
        if "/" not in key and key not in comps:
            say(f"KEEP's row {key} does nothing now: the layout no longer moves it")
    return out, sorted(emptied), suited


def compress(mine, tracked_dirs, own=()):
    """The rows of one map: `mine` is {old file: new file}; `tracked_dirs` is {directory: (files in it, subdirectories)}
    over every tracked file. A directory is one row where every tracked file under it is the map's and keeps its path
    below it; otherwise its files are rows and its subdirectories are looked at in turn. A path of `own` (a file, or a
    directory ending in "/") is always written by itself and the directory that holds it may still be one row: the
    longest row wins."""
    whole = {}
    own = {k for k in own if any(holds(k, p) for p in mine)}

    def walk(d):
        files, subs = tracked_dirs[d]
        target = set()
        for f in files:
            p = f"{d}/{f}" if d else f
            if p in own:
                continue
            new = mine.get(p)
            target.add(None if new is None or posixpath.basename(new) != f else posixpath.dirname(new))
        for s in subs:
            n = walk(f"{d}/{s}" if d else s)
            if (f"{d}/{s}/" if d else f"{s}/") not in own:
                target.add(None if n is None or posixpath.basename(n) != s else posixpath.dirname(n))
        whole[d] = target.pop() if len(target) == 1 else None
        return whole[d]
    walk("")
    rows = {}

    def emit(d):
        if d and whole[d] is not None:
            rows[d + "/"] = whole[d] + "/"
            return
        files, subs = tracked_dirs[d]
        for f in files:
            p = f"{d}/{f}" if d else f
            if p in mine:
                rows[p] = mine[p]
        for s in subs:
            emit(f"{d}/{s}" if d else s)
    emit("")
    for k in own:
        if k.endswith("/"):
            emit(k[:-1])
        else:
            rows[k] = mine[k]
    return rows


def tokens_for(move, dest, tracked_dirs, rows, emptied=(), one=False):
    """The directories no row moves whole that this map speaks for, those several maps share and those it empties by
    rows that part ways: {old directory: new directory}. `one`: the map is the whole layout."""
    out = {}
    under = collections.defaultdict(list)              # directory -> tracked files below it
    for d, (files, _) in tracked_dirs.items():
        for f in files:
            p = f"{d}/{f}" if d else f
            a = d
            while a:
                under[a].append(p)
                a = posixpath.dirname(a)
    dir_rows = [k.rstrip("/") for k in rows if k.endswith("/")]
    for d, ps in under.items():
        if any(d == r or d.startswith(r + "/") for r in dir_rows):
            continue
        mine = [p for p in ps if p in dest and dest[p][2] == move]
        if not mine or all(dest[p][0].startswith("archive/") for p in mine):
            continue                                   # the archive speaks for no directory: what stays keeps its name
        files = tracked_dirs[d][0]
        name = next((n for n in NAMESAKE if n in files), None)
        if name is not None:
            if dest.get(f"{d}/{name}", (None, None, None))[2] != move:
                continue
        elif 2 * len(mine) <= len(ps):
            continue
        new = {dest[p][0][: -len(p[len(d):])] if dest[p][0].endswith(p[len(d):]) else None for p in mine}
        if len(new) == 1 and (n := new.pop()) is not None:
            out[d + "/"] = n + "/"
        elif all(p in emptied or dest.get(p, (None, None, None))[2] in (move, "archive") for p in ps):
            # The map empties the directory and its files part ways. `layout.py` looks only for paths that begin with a
            # row or a `[tokens]` key, so without one a reference to the directory itself (a suite's declared input, a
            # boundary's glob) is neither rewritten nor noted and names a directory that is gone. It reads as the one
            # new directory that holds every file of the old one but the archived, where that is narrower than a
            # top-level directory;
            # else, in the one map, as where its package goes, each such reference acknowledged in `[split]`.
            def inner(p, lake="lakefile.toml" in files):   # in a Lake package inside this one that has a row of its own
                k = excepted(p)[0]
                return (lake and k is not None and k.endswith("/") and k.startswith(d + "/") and k != d + "/"
                        and "lakefile.toml" in tracked_dirs.get(k.rstrip("/"), ((),))[0])
            rest = [p for p in mine if not dest[p][0].startswith("archive/") and own_row(p) is None and not inner(p)]
            whole = {dest[p][0][: -len(p[len(d):])] if dest[p][0].endswith(p[len(d):]) else None for p in rest}
            common = posixpath.commonpath([posixpath.dirname(dest[p][0]) for p in rest or mine])
            if len(whole) == 1 and (n := next(iter(whole))) is not None:
                # all but the archived, a file with a row of its own and a Lake package inside this one keep together
                out[d + "/"] = n + "/"
            elif "/" in common:
                out[d + "/"] = common + "/"
            elif one and name is not None and posixpath.basename(dest[f"{d}/{name}"][0]) == name:
                out[d + "/"] = posixpath.dirname(dest[f"{d}/{name}"][0]) + "/"
    return out


def retire(tracked, moved, drop, rows):
    """The outermost directories a map empties that none of its directory rows holds, each ending in "/": every tracked
    file under one is moved (`moved`, {today's path: new path}) or dropped, and no new path lies under it."""
    stays, arrives = set(), {a for n in moved.values() for a in ancestors(n)}
    for p in tracked:
        if p not in moved and p not in drop:
            stays.update(ancestors(p))
    out = []
    for d in sorted({a for p in tracked for a in ancestors(p)} - stays - arrives):      # a directory before those under it
        if not any((d + "/").startswith(k) for k in (*out, *rows) if k.endswith("/")):
            out.append(d + "/")
    return out


def scan_splits(repo, lay_data):
    """The references to a directory the map splits that `layout.py` would rewrite, by its own scan of the worktree."""
    import pathlib
    run = L.Run(pathlib.Path(repo), L.Layout(lay_data), True)
    moves = run.plan()
    run.index(moves)
    run.move(moves)
    run.create()
    run.rewrite_all()
    return dict(run.splits)


def toml_str(s):
    return json.dumps(s, ensure_ascii=False)


def wrap(prefix, text, width=118):
    out, line = [], prefix
    for w in text.split():
        if len(line) + len(w) + 1 > width and line.strip("# "):
            out.append(line.rstrip())
            line = prefix
        line += w + " "
    out.append(line.rstrip())
    return "\n".join(out)


def render(fname, v, before, one=False):
    rows = v["rows"]
    lines = [wrap("# ", f"The layout move of {v['what']}: what `tools/move/layout.py --map tools/move/{fname}` moves and "
                        f"rewrites. {len(v['mine'])} files in {len(rows)} rows. Written by `tools/move/layout_maps.py`, "
                        "which states the rule: edit that, not this.")]
    lines.append(wrap("# ", "A moved file's new path is its component's new directory, then the name of the directory it is in "
                            "today, then its path below that; the name is left out where the new directory, less `verifier/`, "
                            "`prover/` or `device/`, already ends in it. A directory is one row where all of it goes one way; a "
                            "directory this move shares with another is one row a file. Nothing is edited or deleted."))
    if v["reasons"]:
        lines.append("# Rows that depart from that rule:")
        for old, new, why in v["reasons"]:
            lines.append(wrap("#   ", f"`{old}` to `{new}`: {why}.").replace("#   ", "# - ", 1))
    lines.append(wrap("# ", ("This one map is the whole layout: no order, and nothing runs before or after it." if one else
                             "The layout's maps run in the order of their names. " +
                             ("This one runs after " + ", ".join(f"`{b}`" for b in before) + ": it renames a package that "
                              "those maps take modules out of, and would rename their modules with it."
                              if before else "This one renames no package another map takes modules out of, so its place "
                              "among them is free.")) + " The format is `layout_map.toml`'s."))
    fmt = lambda d: "{ " + ", ".join(f"{k} = {toml_str(x)}" for k, x in d.items()) + " }"   # noqa: E731
    lines += ["", "[roots]", f"old = {fmt(ROOTS)}", f"new = {fmt(v['roots_new'])}", "", "[paths]"]
    lines += [f"{toml_str(k)} = {toml_str(x)}" for k, x in sorted(rows.items())]
    if v["tokens"]:
        lines += ["", "[tokens]", "# directories no row moves whole, as a reference to one reads after the move"]
        lines += [f"{toml_str(k)} = {toml_str(x)}" for k, x in sorted(v["tokens"].items())]
    if v["split"]:
        lines += ["", "[split]", "# references to a directory this move splits that stay right: \"old file: old directory\" = why"]
        lines += [f"{toml_str(k)} = {toml_str(x)}" for k, x in sorted(v["split"].items())]
    lines += ["", "[retired]"]
    if v["retired"]:
        lines += [wrap("# ", "directories this map empties that no directory row holds: `layout.py` refuses a tracked file "
                             "under one that has no row"),
                  "paths = [", *(f"  {toml_str(p)}," for p in v["retired"]), "]"]
    else:
        lines.append("paths = []")
    lines += ["", "[drop]",
              *(wrap("# ", f"`{p}`: {DROP[p][1]}.") for p in v["drop"]),
              "paths = [" + ", ".join(toml_str(p) for p in v["drop"]) + "]", "", "[rewrite]",
              "# Never rewritten: records of what was (the earlier moves and their maps, changelogs, research's recorded moves, the",
              "# archive), files whose bytes a digest pins, and Lean sources. A module's source is in its build key, so a path",
              "# rewritten in a comment would compile the module again, and every module that imports it. A `.lean` file that",
              "# writes a path outside a comment is still rewritten (`include`); lakefiles and Lake manifests are too.",
              "exclude = [" + ",\n           ".join(", ".join(toml_str(x) for x in EXCLUDE[i:i + 4]) for i in range(0, len(EXCLUDE), 4)) + "]",
              "include = [" + ",\n           ".join(", ".join(toml_str(x) for x in INCLUDE[i:i + 3]) for i in range(0, len(INCLUDE), 3)) + "]",
              "", "[modules]",
              "# the renames [paths] and [roots] imply (`layout.py --modules`)"]
    lines += [f"{toml_str(k)} = {toml_str(x)}" for k, x in sorted(v["modules"].items())]
    if v["aliases"]:
        lines += ["", "[aliases]", "# a name a moved module is imported by that its path does not give"]
        for k, x in sorted(v["aliases"].items()):
            lines += [wrap("# ", f"`{k}`: {ALIASES[k][2]}."), f"{toml_str(k)} = {toml_str(x)}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", default="HEAD", help="the commit whose files the maps move (default: HEAD)")
    ap.add_argument("--data", default=os.path.join(HERE, "layout_maps.json"))
    ap.add_argument("--out", default=HERE, help="where the two forms' directories go (default: tools/move)")
    ap.add_argument("--form", choices=[*FORMS, "both"], default="one",
                    help="`one`: every row as one map (the default); `moves`: one map a move")
    ap.add_argument("--targets", help="write the table of where each moved component goes here, tab-separated")
    ap.add_argument("--created", help="write here the `__init__.py`s `layout.py` creates over all the maps, one a line")
    a = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(HERE))
    data = json.load(open(a.data))
    head = git(repo, "rev-parse", a.ref).strip()
    tracked = {p for p in git(repo, "ls-tree", "-r", "--name-only", "-z", a.ref).split("\0") if p}
    notes = []
    owner = place(data, tracked, notes.append)
    suites = []                                        # as `tests/test_repository.py` reads them: a pyproject that configures pytest
    for pp in sorted(p for p in tracked if posixpath.basename(p) == "pyproject.toml"):
        ini = tomllib.loads(git(repo, "show", f"{a.ref}:{pp}")).get("tool", {}).get("pytest", {}).get("ini_options")
        if ini is not None:
            suites.append((pp, [posixpath.normpath(posixpath.join(posixpath.dirname(pp), t)) for t in ini.get("testpaths", ["."])]))
    dest, emptied, suited = plan(data, owner, notes.append, suites)
    comps = {c["id"]: c for c in data["components"]}

    # ---- checks on the plan
    by_new = collections.defaultdict(list)
    for p, (new, _, _) in dest.items():
        by_new[new].append(p)
    twice = {n: ps for n, ps in by_new.items() if len(ps) > 1}
    if twice:
        die("two files, one destination (add a row to EXCEPT):\n  " + "\n  ".join(f"{n}: {ps}" for n, ps in sorted(twice.items())[:40]))
    onto = sorted(n for n in by_new if n in tracked and n not in dest)
    if onto:
        die("a destination is a file that stays:\n  " + "\n  ".join(onto[:40]))
    bad = sorted(n for n in by_new if not n.startswith("archive/") and {"core", "ml"} & set(n.split("/")[:-1]))
    bad += sorted(p for p, (n, _, _) in dest.items() if n.startswith("archive/") and n != "archive/" + p)
    if bad:
        die("a new path keeps a `core` or `ml` segment, or an archived file loses its path:\n  " + "\n  ".join(bad[:40]))
    same = sorted(p for p, (n, _, _) in dest.items() if p == n)
    if same:
        die("a row that moves nothing:\n  " + "\n  ".join(same[:40]))
    stay = sorted(p for p in tracked if p.startswith(("verity/core/", "verity/ml/")) and p not in dest and p not in emptied)
    if stay:
        die("under verity/core or verity/ml with no destination:\n  " + "\n  ".join(stay[:40]))
    trusted_dirs = [t.strip("/") for t in data["trusted_dirs"]]
    untrusted, with_suite = collections.defaultdict(list), collections.Counter()
    for p, (n, key, _) in dest.items():
        if key in comps and not comps[key]["trusted"] and any(n == t or n.startswith(t + "/") for t in trusted_dirs):
            if p in suited:                            # a suite's tests go with it, whoever's they are
                with_suite[(key, posixpath.dirname(n))] += 1
            else:
                untrusted[key].append(f"{p} -> {n}")
    if set(untrusted) - set(UNTRUSTED):
        die("files of a component the auditor doesn't trust land under a trusted directory (a row in EXCEPT, or in "
            "UNTRUSTED with why it may):\n  " + "\n  ".join(f"{k} ({comps[k]['name']}): {len(v)} files, {v[0]}"
                                                            for k, v in sorted(untrusted.items()) if k not in UNTRUSTED))
    unknown = sorted({m for _, _, m in dest.values()} - {m for m, _ in MOVES})
    if unknown:
        die(f"moves with no entry in MOVES: {unknown}")
    after_all = (tracked - set(dest) - set(DROP)) | set(by_new)   # the tree after the moves
    by_case = layout_case.case_sets(after_all)
    if by_case:
        die("paths of the moved tree that differ only by case, which a checkout that ignores case merges (a row in EXCEPT):\n  "
            + "\n  ".join(" and ".join(sorted(v)) for v in by_case[:40]))
    new_of = lambda p: dest[p][0] if p in dest else p   # noqa: E731
    members = tomllib.loads(git(repo, "show", f"{a.ref}:pyproject.toml")).get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])
    cut = []                                           # uv finds a member's workspace at the first pyproject.toml above it
    for m in sorted(members):
        pp = f"{m.rstrip('/')}/pyproject.toml"
        if pp in tracked and any(isinstance(s, dict) and s.get("workspace") for s in tomllib.loads(
                git(repo, "show", f"{a.ref}:{pp}")).get("tool", {}).get("uv", {}).get("sources", {}).values()):
            cut += [f"{new_of(pp)}: below {d}/pyproject.toml" for d in ancestors(posixpath.dirname(new_of(pp)))
                    if f"{d}/pyproject.toml" in after_all]
    if cut:
        die("a workspace member whose sources name the workspace lands below another pyproject.toml, where `uv lock` no "
            "longer finds the workspace (a row in EXCEPT for the pyproject.toml above it):\n  " + "\n  ".join(cut))
    testpaths = [new_of(pp)[: -len("pyproject.toml")].rstrip("/") + t[len(posixpath.dirname(pp)):] for pp, ts in suites for t in ts]
    defines = set(git(repo, "grep", "-lE", r"^(async\s+)?def\s+test|^class\s+Test", a.ref, "--", "*.py").replace(a.ref + ":", "").split())
    unsuited = sorted(new_of(p) for p in tracked if TEST_FILE.fullmatch(p) and p in defines and not p.startswith(UNSUITED)
                      and not any(t in (".", "") or new_of(p).startswith(t + "/") for t in testpaths))
    unsuited = [n for n in unsuited if not n.startswith(UNSUITED)]
    if unsuited:
        die("test files that no suite collects after the moves (a suite moves whole, with its pyproject.toml):\n  " + "\n  ".join(unsuited))
    for p in emptied:
        notes.append(f"stays, the `__init__.py` of a package level the moves empty, for a commit that deletes it after them: {p}")

    # ---- the maps
    tracked_dirs = collections.defaultdict(lambda: ([], set()))
    for p in sorted(tracked):
        d = posixpath.dirname(p)
        tracked_dirs[d][0].append(posixpath.basename(p))
        while d:
            tracked_dirs[posixpath.dirname(d)][1].add(posixpath.basename(d))
            d = posixpath.dirname(d)
    tracked_dirs = {d: (fs, sorted(ss)) for d, (fs, ss) in tracked_dirs.items()}
    def build(dest, moves, one, agree=None):
        """The maps of one form: {move: its map}, the order they run in and what each runs after. `agree` is the
        `[tokens]` rows of the other form's maps, which this form's rows for the same directories take."""
        maps = {}
        for move, what in moves:
            mine = {p: n for p, (n, _, m) in dest.items() if m == move}
            if not mine:
                die(f"{move}: no file")
            rows = compress(mine, tracked_dirs, ({own_row(p) for p in mine} - {None})
                            | {p for p in mine if mine[p].startswith("archive/") and any(holds(k, p) for k in WHOLE)})
            # the archive speaks for no directory
            tokens = {} if move == "archive" else tokens_for(move, dest, tracked_dirs, rows, emptied, one)
            for k, n in (agree or {}).items():         # a directory a move's map speaks for reads the same in the one map
                if tokens.get(k, n) != n:
                    notes.append(f"[tokens] {k} reads {n} in the one map, as in its move's map, not {tokens[k]}")
                    tokens[k] = n
            roots_new = {}
            for name, d in ROOTS.items():
                if not any(p.startswith(d + "/") for p in tracked):
                    die(f"ROOTS: {name}'s directory {d} holds no tracked file")
                if f"{d}/__init__.py" not in mine:         # the package stays, whatever leaves it
                    roots_new[name] = d
                    continue
                inside = [p for p in tracked if p.startswith(d + "/")]
                new = {mine[p][: -len(p[len(d):])] if p in mine and mine[p].endswith(p[len(d):]) else None for p in inside}
                if len(new) != 1 or None in new:
                    die(f"{move}: it moves the package {name} ({d}) in part, or to two places: {sorted(map(str, new))}")
                roots_new[name] = new.pop().rstrip("/")
            lay_data = {"roots": {"old": dict(ROOTS), "new": roots_new}, "paths": rows, "tokens": tokens, "split": {},
                        "rewrite": {"exclude": EXCLUDE, "include": INCLUDE}}
            modules = L.derived_modules(L.Layout(lay_data))
            lay_data["modules"] = modules
            split = {} if one else {k: w for m, k, w in SPLIT if m == move and k.split(": ")[0] in tracked}
            lay_data["split"] = split
            drop = sorted(p for p, (m, _) in DROP.items() if (one or m == move) and p in tracked)
            if set(drop) & set(dest):
                die(f"{move}: DROP deletes a file a row moves: {sorted(set(drop) & set(dest))}")
            lay_data["drop"] = {"paths": drop}
            gone = retire(tracked, mine, drop, rows) if one else []
            lay_data["retired"] = {"paths": gone}
            aliases = {k: n for k, (m, n, _) in sorted(ALIASES.items()) if one or m == move}
            for k, n in aliases.items():
                if n not in modules.values():
                    die(f"{move}: ALIASES' row {k} names {n}, which no row of the map makes")
            lay_data["aliases"] = aliases
            lay = L.Layout(lay_data)
            got = {p: lay.file(p) for p in tracked if lay.file(p) is not None}
            if got != mine:
                diff = sorted(set(got.items()) ^ set(mine.items()))
                die(f"{move}: its rows move other than its files: {diff[:10]}")
            for p, n in mine.items():
                if p.endswith(".py") and (mo := L.module_name(p, ROOTS)) and (mn := L.module_name(n, roots_new)) and lay.module(mo) != mn:
                    die(f"{move}: {p} -> {n}: its module {mo} would be renamed {lay.module(mo)}, not {mn}")
            reasons = [(o, n, w) for o, n, w in EXCEPT if any(holds(o, p) for p in mine)]
            maps[move] = {"what": what, "rows": rows, "tokens": tokens, "split": split, "modules": modules,
                          "roots_new": roots_new, "lay": lay, "mine": mine, "reasons": reasons, "data": lay_data, "drop": drop,
                          "aliases": aliases, "retired": gone}

        # ---- across the maps: no row over another map's files or over a destination, Lake packages whole
        keys = [(m, k) for m in maps for k in maps[m]["rows"]]
        for m, k in keys:
            for m2, k2 in keys:
                if m != m2 and (k == k2 or (k.endswith("/") and k2.startswith(k))):
                    die(f"{m}'s row {k} holds {m2}'s row {k2}")
        for m, v in maps.items():
            hit = sorted(n for n in by_new if v["lay"].file(n) is not None)
            if hit:
                die(f"{m}: a row of it would move a destination again: {hit[:5]}")
        lakefiles = sorted(p for p in dest if posixpath.basename(p) in ("lakefile.toml", "lakefile.lean"))
        for p in lakefiles:
            if p:
                root, m = posixpath.dirname(p), dest[p][2]
                new_root = posixpath.dirname(dest[p][0])
                there = maps[m]["lay"].directory(root)[0]
                # a Lake package inside this one goes its own way
                inner = tuple(posixpath.dirname(q) + "/" for q in lakefiles if q.startswith(root + "/") and posixpath.dirname(q) != root)
                if there != new_root or any(n != new_root + q[len(root):] for q, n in maps[m]["mine"].items()
                                            if q.startswith(root + "/") and not (inner and q.startswith(inner))):
                    die(f"the Lake package at {root} does not move whole to {new_root} by a row or a token of {m} (it gives {there})")

        # ---- the order: B runs after A when B's renames would rename a module A moves
        after = {m: set() for m in maps}
        for b, vb in maps.items():
            for a_, va in maps.items():
                if a_ == b:
                    continue
                for p in va["mine"]:
                    if p.endswith(".py") and (mo := L.module_name(p, ROOTS)) and vb["lay"].module(mo) != mo:
                        after[b].add(a_)
                        break
                for n in va["mine"].values():
                    if n.endswith(".py") and (mn := L.module_name(n, va["roots_new"])) and vb["lay"].module(mn) != mn:
                        die(f"{b}'s [modules] would rename {mn}, the new name of a module {a_} moves")
        for p in sorted(tracked - set(dest)):
            if p.endswith(".py") and (mo := L.module_name(p, ROOTS)):
                for b, vb in maps.items():
                    if vb["lay"].module(mo) != mo:
                        die(f"{b}'s [modules] would rename {mo}, a module no map moves ({p})")
        order, left = [], [m for m, _ in moves]
        while left:
            ready = [m for m in left if not (after[m] - set(order))]
            if not ready:
                die(f"the order has a cycle among {left}: " + "; ".join(f"{m} after {sorted(after[m] - set(order))}" for m in left))
            order.append(ready[0])
            left.remove(ready[0])
        if one:                # the one map splits directories: `layout.py`'s own scan names each reference, after every check
            if git(repo, "rev-parse", "HEAD").strip() != head:
                die(f"the one map's [split] rows are read from the worktree, which is not at {a.ref}")
            why = {k: w for _, k, w in SPLIT}
            maps[ONE[0]]["split"] = {k: why.get(k, SCANNED) for k in scan_splits(repo, maps[ONE[0]]["data"])}
        return maps, order, after

    built = {}
    by_move = build(dest, MOVES, False)                # always: the one map's `[tokens]` rows agree with the moves' maps'
    if a.form in ("moves", "both"):
        built["moves"] = by_move
    if a.form in ("one", "both"):
        agree = {}
        for move, v in by_move[0].items():
            for k, n in v["tokens"].items():
                if agree.setdefault(k, n) != n:
                    die(f"[tokens] {k}: two moves' maps speak for it, as {agree[k]} and as {n}")
        built["one"] = build({p: (n, key, ONE[0]) for p, (n, key, _) in dest.items()}, [ONE], True, agree)
        differ = sorted(k for k, n in built["one"][0][ONE[0]]["tokens"].items() if agree.get(k, n) != n)
        if differ:
            die("[tokens] rows of the one map that differ from a move's map's: " + ", ".join(differ))

    # ---- write: a move's map carries its place in the order in its name
    os.makedirs(a.out, exist_ok=True)
    fnames = {}
    for form, (maps, order, after) in built.items():
        one = form == "one"
        fname = {m: "layout_one_map.toml" if one else f"layout_{i:02d}_{m}_map.toml" for i, m in enumerate(order, 1)}
        fnames[form] = fname
        for old in os.listdir(a.out):                  # this form's maps of an earlier run, under names it no longer writes
            if MAP_NAME.fullmatch(old) and (old == "layout_one_map.toml") == one and old not in fname.values():
                os.remove(os.path.join(a.out, old))
        for move in order:
            with open(os.path.join(a.out, fname[move]), "w") as f:
                f.write(render(fname[move], maps[move], [fname[b] for b in order if b in after[move]], one))
    form = "moves" if "moves" in built else "one"     # the table and the counts below are of the maps a move, where written
    maps, order, after = built[form]
    fname = fnames[form]
    if form == "one":
        dest = {p: (n, key, ONE[0]) for p, (n, key, _) in dest.items()}
    if a.targets:
        by_comp = collections.defaultdict(list)
        for p, (n, key, m) in dest.items():
            by_comp[(m, key)].append((p, n))
        with open(a.targets, "w") as f:
            f.write("\t".join(["map", "component", "name", "files", "where its files are today", "target path",
                               "by directory: today -> target (files)"]) + "\n")
            for (m, key), pairs in sorted(by_comp.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1])):
                by = collections.Counter()
                for p, n in pairs:
                    i = 0
                    while i < min(len(p), len(n)) and p[-1 - i] == n[-1 - i]:
                        i += 1
                    tail = p[len(p) - i:]
                    tail = tail[tail.find("/") + 1:] if "/" in tail else posixpath.basename(p)
                    by[(p[: len(p) - len(tail)].rstrip("/") or ".", n[: len(n) - len(tail)].rstrip("/") or ".")] += 1
                name = comps[key]["name"] if key in comps else {"archived": "what the layout archives"}.get(key, "a file the layout keeps")
                f.write("\t".join([fname[m], key, name.replace("\t", " "), str(len(pairs)), common_dir([p for p, _ in pairs]) or ".",
                                   common_dir([n for _, n in pairs]) or ".",
                                   "; ".join(f"{o} -> {n} ({k})" for (o, n), k in sorted(by.items()))]) + "\n")
    created = set()                                    # as `layout.py`'s `create` does, over all the maps
    for move in order:
        v = maps[move]
        for p, n in v["mine"].items():
            if n.endswith(".py") and f"{posixpath.dirname(p)}/__init__.py" in tracked:
                root = next((d for d in sorted(v["roots_new"].values(), key=len, reverse=True) if n.startswith(d + "/")), None)
                d = posixpath.dirname(n)
                while root is not None and d and (d == root or d.startswith(root + "/")):
                    if f"{d}/__init__.py" not in after_all:
                        created.add(f"{d}/__init__.py")
                    d = posixpath.dirname(d)
    if a.created:
        with open(a.created, "w") as f:
            f.write("".join(p + "\n" for p in sorted(created)))
    moved = {key for _, key, _ in dest.values() if key in comps}
    if "one" in built:
        v = built["one"][0][ONE[0]]
        size = os.path.getsize(os.path.join(a.out, "layout_one_map.toml"))
        print(f"the one map: {len(v['mine'])} files, {len(v['rows'])} rows ({sum(1 for k in v['rows'] if k.endswith('/'))} "
              f"directories), {len(v['tokens'])} tokens, {len(v['split'])} split rows, {len(v['modules'])} module renames, "
              f"{size} bytes")
    print(f"{len(order)} maps at {a.ref} {head[:12]} (the data's cut: {data['commit'][:12]}): {len(dest)} files "
          f"({sum(1 for _, k, _ in dest.values() if k in comps)} in {len(moved)} components that move, "
          f"{sum(1 for n, _, _ in dest.values() if n.startswith('archive/'))} to archive/, "
          f"{sum(1 for n, k, _ in dest.values() if k not in comps and not n.startswith('archive/'))} by a row of EXCEPT alone), "
          f"{sum(len(v['rows']) for v in maps.values())} rows, {sum(len(v['tokens']) for v in maps.values())} tokens")
    for move in order:
        v = maps[move]
        print(f"  {fname[move]}: {len(v['mine'])} files, {len(v['rows'])} rows "
              f"({sum(1 for k in v['rows'] if k.endswith('/'))} directories), {len(v['tokens'])} tokens, {len(v['split'])} "
              f"split rows, {len(v['modules'])} module renames" + (f"; after {', '.join(sorted(after[move]))}" if after[move] else ""))
    for key, v in sorted(untrusted.items()):
        print(f"  untrusted under a trusted directory, allowed by UNTRUSTED: {key}, {len(v)} files ({v[0]})")
    for (key, d), n in sorted(with_suite.items()):
        print(f"  a suite's tests under a trusted directory: {n} of `{key}` ({comps[key]['name']}) in `{d}/`")
    print(f"  `layout.py` creates {len(created)} `__init__.py` over the maps (--created FILE lists them)")
    for n in notes:
        print("  " + n)


if __name__ == "__main__":
    main()
