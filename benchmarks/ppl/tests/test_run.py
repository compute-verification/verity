import argparse
import json
import re

import pytest

import models
import report
import run
import spec
import tables
import windows


def test_arms_for_adds_each_scopes_plain_arm_once():
    names = [a.name for a in run.arms_for(["adopted-linears", "adopted", "int8-linears"])]
    assert names == ["adopted-linears", "adopted", "int8-linears", "int8-all"]
    with pytest.raises(ValueError, match="share a name"):
        run.arms_for(["adopted", "adopted"])


def test_seeds_are_a_prefix():
    assert run.parse_seeds("0,1,2") == (0, 1, 2) and run.parse_seeds("0") == (0,)
    for bad in ("1,2", "0,2", "0,1,2,3"):
        with pytest.raises(ValueError, match="prefix"):
            run.parse_seeds(bad)


def submit(**kw) -> list[str]:
    a = argparse.Namespace(**{"model": "llama-3.1-8b", "set": "c4", "ctx": 2048, "gpus": 0, "seeds": "0,1,2",
                              "plugin": None, "spec": ["adopted"], "max_min": 120, "campaign": "cmp-13",
                              "device": "cuda", "threads": 0, "on": run.CPU_NODE, "mem_gb": 96, **kw})
    return run.submit_command(a, run.LOCAL_DATA)


def test_submit_is_a_research_run_on_the_node_under_a_lease():
    cmd = submit()
    assert cmd[:4] == ["research", "run", "--on", run.NODE] and cmd[-3:] == ["--", "bash", "benchmarks/ppl/job.sh"]
    sent = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--send"]
    assert sent == [str(run.LOCAL_DATA / d["file"]) for d in windows.DATASETS.values()]
    env = dict(cmd[i + 1].split("=", 1) for i, x in enumerate(cmd) if x == "--env")
    assert env["PPL_GPUS"] == "1" and env["PPL_ARGS"].endswith("--gpus 1 --seeds 0,1,2 adopted")
    assert "--model llama-3.1-8b --set c4 --ctx 2048" in env["PPL_ARGS"]


def test_a_cpu_submit_is_a_research_run_on_a_cpu_node_with_no_lease():
    cmd = submit(device="cpu", threads=32, on="vy-nebius-cpu-4", mem_gb=256, max_min=300)
    assert cmd[:4] == ["research", "run", "--on", "vy-nebius-cpu-4"] and cmd[-3:] == ["--", "bash", "benchmarks/ppl/cpu_job.sh"]
    assert cmd[cmd.index("--mem-gb") + 1] == "256" and "--mem-wait" in cmd and cmd[cmd.index("--timeout") + 1] == "300m"
    env = dict(cmd[i + 1].split("=", 1) for i, x in enumerate(cmd) if x == "--env")
    assert env == {"PPL_ARGS": "--model llama-3.1-8b --set c4 --ctx 2048 --seeds 0,1,2 --device cpu --threads 32 adopted"}


def test_70b_leases_two_gpus_and_never_more():
    env = dict(x.split("=", 1) for x in submit(model="llama-3.1-70b") if x.startswith("PPL_"))
    assert env["PPL_GPUS"] == "2" and "--gpus 2" in env["PPL_ARGS"]
    with pytest.raises(SystemExit, match="at most 2 GPUs"):
        submit(gpus=3)


def test_the_readme_holds_no_table():
    text = run.README.read_text()
    found = [f"README.md:{i}: {line}" for i, line in enumerate(text.splitlines(), 1)
             if re.match(r"\s*\|.*\|\s*$", line)]
    assert not found, ("a table belongs in `run.py tables` (tables.py, one section a table, read from the code and "
                       "data it describes), and the README says which section prints it:\n" + "\n".join(found))
    assert "run.py tables" in text


def test_tables_render_the_registry_the_specs_and_the_lock(capsys):
    out = tables.render()
    steps = spec.catalogue()
    assert all(f"| `{k}(" in out for k in steps) and all(v["source"] for v in steps.values())
    assert all(f"| `{p.stem}` |" in out for p in spec.SPECS.glob("*.toml"))
    assert all(f"| `{m}` |" in out for m in models.MODELS)
    assert report.baseline_table(json.loads(run.BASELINE.read_text())) in out
    assert run.main(["tables", "windows"]) == 0
    assert re.findall(r"^## (.*)", capsys.readouterr().out, re.M) == ["The window lists"]
    with pytest.raises(SystemExit, match="no section nope"):
        run.main(["tables", "nope"])
