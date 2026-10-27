"""Every population program a template of this package's catalog makes keeps its descriptor id and digest, held to the values
recorded in `population_recorded.json` (written on main before a unit's population gained its ports, `_instance_ports`):
`TemplateInstances` v1 at N = 1 and 3 and v0 at N = 3 over each complete composite root of every circuit-check sidecar under
`verity/ml/circuits` (the Gemm family's word and Boolean templates, the served rows' coordinates among them), and
`TemplatePopulations` v1 and v0 over each consecutive pair of them.

A change meant to change these programs rewrites the record (``python test_population_recorded.py --write``) and says so;
any other change leaves it as it is. The programs are built in a child process: the sidecars import modules that register
kernels, which `test_evaluation`'s registry test would otherwise see in the same worker."""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from verity.ml.circuits.codec import descriptor_id, program_digest
from verity.ml.circuits.defs import CompositeDefinition, SpecializedDefinition
from verity.ml.circuits.population import mixed_population_program, population_program

PACKAGE = Path(__file__).resolve().parents[1]
RECORD = Path(__file__).with_name("population_recorded.json")
ABOUT = ("TemplateInstances / TemplatePopulations programs over verity/ml/circuits' circuit-check catalog: per program, the "
         "SHA-512 of its descriptor id and its program digest; checked and written by test_population_recorded.py")
#: the population programs' own families, which the sidecars bind as roots too: not templates of a population here
POPULATIONS = ("TemplateInstances", "TemplatePopulations")


def _templates() -> list[Any]:
    """Each complete composite root of the package's sidecars, by sidecar path and root function, once each."""
    out: dict[str, Any] = {}
    for p in sorted(PACKAGE.rglob("*.circuit-check.py")):
        spec = importlib.util.spec_from_file_location(f"_population_recorded_{len(out)}_{p.stem.replace('.', '_')}", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        for name in sorted(n for n in vars(m) if n.startswith("_") and n.endswith("_roots")):
            for r in getattr(m, name)():
                if isinstance(r, SpecializedDefinition) and isinstance(r.defn, CompositeDefinition) and r.complete \
                        and r.signature()[0] and not r.defn.name.startswith(POPULATIONS):
                    out.setdefault(descriptor_id(r), r)
    return list(out.values())


def _entry(prog: Any) -> dict[str, str]:
    return {"id_sha512": hashlib.sha512(descriptor_id(prog.fn).encode()).hexdigest(), "digest": program_digest(prog)}


def programs() -> dict[str, dict[str, str]]:
    ts = _templates()
    out = {}
    for t in ts:
        for n, v in ((1, 1), (3, 1), (3, 0)):
            out[f"TemplateInstances_v{v}|N={n}|{descriptor_id(t)}"] = _entry(population_program(t, n, version=v))
    for a, b in itertools.pairwise(ts):
        for v in (1, 0):
            out[f"TemplatePopulations_v{v}|1,2|{descriptor_id(a)}|{descriptor_id(b)}"] = \
                _entry(mixed_population_program([(a, 1), (b, 2)], version=v))
    return out


@pytest.mark.slow
def test_every_recorded_population_keeps_its_id_and_digest():
    recorded = json.loads(RECORD.read_text())["programs"]
    now = json.loads(subprocess.run([sys.executable, __file__, "--print"], capture_output=True, text=True,
                                    check=True).stdout)
    gone = sorted(set(recorded) - set(now))
    changed = sorted(k for k in recorded if k in now and now[k] != recorded[k])
    assert not gone and not changed, (
        f"population programs changed ({len(changed)}) or no longer built ({len(gone)}): {(changed + gone)[:5]}. "
        "population.py must keep every existing population program byte for byte; a change meant to change them rewrites "
        f"{RECORD.name} with `python {Path(__file__).name} --write` and says so in its PR")
    assert len(recorded) >= 300


if __name__ == "__main__" and sys.argv[1:] == ["--write"]:
    RECORD.write_text(json.dumps({"about": ABOUT, "programs": programs()}, indent=1, sort_keys=True) + "\n")
elif __name__ == "__main__" and sys.argv[1:] == ["--print"]:
    print(json.dumps(programs()))
