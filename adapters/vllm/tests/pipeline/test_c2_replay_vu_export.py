"""Which process keeps a replay's population for the VU export: the Commit, from where it makes the export's hook
(`c2_replay.vu_export`), and no process whose replay has no hook.  The replay driver keeps a population only while its
`keep_population` is on, importing `pipeline.vu_store` turns it on, and only the hook's `after_replay` takes what is kept.  Each
case imports its module in a process of its own: the test process has loaded `vu_store` for other tests, so its switch is on."""
from __future__ import annotations

import functools
import json
import subprocess
import sys

import pytest

#: the modules of the commands that replay a sealed bundle on CPU (`commit-replay`, `tp-commit-replay`): their replay has no
#: export hook, so nothing would take what it kept
NO_HOOK = ("verity_vllm.pipeline.cpu_replay", "verity_vllm.pipeline.tp.commit_replay")

#: argv[1] is imported; what the process holds then, and again once `c2_replay.vu_export` has made a hook as the Commit makes it
CHILD = ("import importlib, json, sys\n"
         "from types import SimpleNamespace\n"
         "importlib.import_module(sys.argv[1])\n"
         "import verity_vllm.check.replay.driver as SR\n"
         "import verity_vllm.pipeline.c2_replay as C2\n"
         "def held():\n"
         "    return {'vu_store': 'verity_vllm.pipeline.vu_store' in sys.modules, 'keep_population': SR._KEEP[0]}\n"
         "imported = held()\n"
         "C2.vu_export(SimpleNamespace(vu_export_dir=None), 0, SimpleNamespace(run_root=b''), None, {}, None)\n"
         "print(json.dumps({'imported': imported, 'hook_made': held()}))\n")


@functools.cache
def held(module: str) -> dict[str, dict[str, bool]]:
    """A fresh process that imports `module`, as its command does: whether `vu_store` is loaded and the driver's
    `keep_population` is on after the import (`imported`), and after the Commit's hook is made there (`hook_made`)."""
    child = subprocess.run([sys.executable, "-c", CHILD, module], capture_output=True, text=True, check=True)
    return json.loads(child.stdout)


@pytest.mark.parametrize("module", NO_HOOK)
def test_a_process_whose_replay_has_no_export_hook_keeps_no_population(module: str) -> None:
    """With the switch on and no hook, each replay's indices and VUs stay in the driver after it returns, until the next replay's
    or the process's end.  Importing the replay's own modules must leave the switch off and `vu_store` unloaded."""
    assert held(module)["imported"] == {"vu_store": False, "keep_population": False}


@pytest.mark.parametrize("module", NO_HOOK)
def test_the_population_is_kept_from_where_the_commit_makes_the_export_s_hook(module: str) -> None:
    """The Commit makes the hook before its replay runs, so the export still reuses that replay's population."""
    assert held(module)["hook_made"] == {"vu_store": True, "keep_population": True}
