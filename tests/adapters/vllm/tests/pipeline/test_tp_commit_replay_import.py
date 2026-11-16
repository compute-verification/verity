"""`verity_vllm.pipeline.tp.commit_replay` asks for the replay driver where a replay starts (`main`, `replay_rank`), never as
the module loads.  The driver loads Definitions of Verity's registry, and each enters the Definition table as it is built: a
process that imports this module for a helper (`_of_record`, `registered_roles`) and starts no replay must find the table as it
left it, since `interface.resolve` and `codec.decode_program` answer for an id by whether the table holds it."""
from __future__ import annotations

import json
import subprocess
import sys

MODULE = "verity_vllm.pipeline.tp.commit_replay"
DRIVER = "verity_vllm.check.replay.driver"

#: argv[1] is imported with every import statement at its own top level recorded: the names each asks for
CHILD = ("import builtins, importlib, json, sys\n"
         "module, asked, real = sys.argv[1], [], builtins.__import__\n"
         "def recording(name, globals=None, locals=None, fromlist=(), level=0):\n"
         "    if (globals or {}).get('__name__') == module and sys._getframe(1).f_code.co_name == '<module>':\n"
         "        asked.extend([name, *(f'{name}.{x}' for x in fromlist or ())])\n"
         "    return real(name, globals, locals, fromlist, level)\n"
         "builtins.__import__ = recording\n"
         "importlib.import_module(module)\n"
         "builtins.__import__ = real\n"
         "print(json.dumps(sorted(set(asked))))\n")


def test_the_module_does_not_ask_for_the_replay_driver_as_it_loads():
    child = subprocess.run([sys.executable, "-c", CHILD, MODULE], capture_output=True, text=True, check=False)
    assert child.returncode == 0, child.stderr
    asked = json.loads(child.stdout.splitlines()[-1])
    assert f"{MODULE.rsplit('.', 1)[0]}.c2" in asked, "the recorder saw none of the module's own imports"
    assert DRIVER not in asked

