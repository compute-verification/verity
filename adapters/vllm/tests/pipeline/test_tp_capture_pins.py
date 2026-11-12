"""`verity_vllm.pipeline.tp.capture` sets the engine's pins (`engine_profile.apply_env`) as it is imported, below the imports of
its own modules.  vLLM, `huggingface_hub` and `torch._inductor.config` read those switches once, as they are imported, so the
modules above the pins load none of them and read none of the pinned names: the pins are what every reader finds.  Run in a
Python of its own, since this one has the module loaded."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from verity_vllm.engine import env

NAMES = sorted({*env.ENGINE, "HF_HOME", "HF_HUB_OFFLINE"})

_IMPORT = """
import json
import os
import sys

from verity_vllm.engine import env

NAMES = {*env.ENGINE, "HF_HOME", "HF_HUB_OFFLINE"}
READERS = ("vllm", "huggingface_hub", "transformers", "torch._inductor.config")
read, at_the_pins = [], []
environ = type(os.environ)
getitem, walk, apply_engine = environ.__getitem__, environ.__iter__, env.apply_engine


def reads(self, name):
    read.append(name)
    return getitem(self, name)


def walks(self):
    read.extend(NAMES)
    return walk(self)


def pins(hf_home):
    at_the_pins.append({"read": sorted(NAMES & set(read)), "loaded": [m for m in READERS if m in sys.modules]})
    return apply_engine(hf_home)


environ.__getitem__, environ.__iter__, env.apply_engine = reads, walks, pins
import verity_vllm.pipeline.tp.capture  # noqa: E402,F401

print(json.dumps({"at_the_pins": at_the_pins[:1], "in_force": {k: os.environ[k] for k in env.ENGINE}}))
"""


def test_the_pins_are_set_before_anything_reads_a_pinned_name_or_loads_a_module_that_does():
    """Every read of one name (`get`, `[]`, `in`, `setdefault`, `os.getenv`) goes through `os.environ`'s item lookup, and a copy or a
    walk of the environment, which reads every name whether set yet or not, through its iterator; both are recorded from before the
    module's first import to the moment it sets the pins.  The child runs in the adapter's directory, so the module it imports is
    this tree's."""
    inherited = {k: v for k, v in os.environ.items() if k not in NAMES}
    r = subprocess.run([sys.executable, "-c", _IMPORT], env=inherited, cwd=Path(__file__).resolve().parents[2], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.splitlines()[-1]) == {"at_the_pins": [{"read": [], "loaded": []}], "in_force": env.ENGINE}
