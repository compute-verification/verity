import argparse

import pytest

import run
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
                              "plugin": None, "spec": ["adopted"], "max_min": 120, "campaign": "cmp-13", **kw})
    return run.submit_command(a, run.LOCAL_DATA)


def test_submit_is_a_research_run_on_the_node_under_a_lease():
    cmd = submit()
    assert cmd[:4] == ["research", "run", "--on", run.NODE] and cmd[-3:] == ["--", "bash", "benchmarks/ppl/job.sh"]
    sent = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--send"]
    assert sent == [str(run.LOCAL_DATA / d["file"]) for d in windows.DATASETS.values()]
    env = dict(cmd[i + 1].split("=", 1) for i, x in enumerate(cmd) if x == "--env")
    assert env["PPL_GPUS"] == "1" and env["PPL_ARGS"].endswith("--gpus 1 --seeds 0,1,2 adopted")
    assert "--model llama-3.1-8b --set c4 --ctx 2048" in env["PPL_ARGS"]


def test_70b_leases_two_gpus_and_never_more():
    env = dict(x.split("=", 1) for x in submit(model="llama-3.1-70b") if x.startswith("PPL_"))
    assert env["PPL_GPUS"] == "2" and "--gpus 2" in env["PPL_ARGS"]
    with pytest.raises(SystemExit, match="at most 2 GPUs"):
        submit(gpus=3)


def test_readme_with_replaces_only_the_baseline():
    text = "a\n<!-- baseline -->\nold\n<!-- /baseline -->\nb\n"
    assert run.readme_with("new", text) == "a\n<!-- baseline -->\nnew\n<!-- /baseline -->\nb\n"
