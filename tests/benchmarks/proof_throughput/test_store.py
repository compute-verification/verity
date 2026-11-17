import json

from verity_proof_throughput import store


def custody(tmp_path, run_id="r20261017-000000-abcd"):
    d = tmp_path / "requests" / run_id / "custody"
    d.mkdir(parents=True)
    (d / "store.toml").write_text("[remote]\n")
    (d / "cred.json").write_text(json.dumps({"AWS_ACCESS_KEY_ID": "id", "AWS_SECRET_ACCESS_KEY": "secret"}))
    (tmp_path / "runs" / run_id).mkdir(parents=True)
    return {"RESEARCH_RUN_DIR": str(tmp_path / "runs" / run_id), "RESEARCH_RUN_ID": run_id, "HOME": str(tmp_path / "home")}


def test_a_recorded_run_hands_its_custody_key_to_the_store_cli(tmp_path, monkeypatch):
    env = custody(tmp_path)
    monkeypatch.setenv("HOME", env["HOME"])
    got = store.store_env(env)
    assert got["RESEARCH_STORE_CONFIG"] == str(tmp_path / "requests" / env["RESEARCH_RUN_ID"] / "custody" / "store.toml")
    assert got["AWS_ACCESS_KEY_ID"] == "id" and "AWS_SESSION_TOKEN" not in got
    assert got["PYTHONPATH"].split(":")[0] == str(store.RESEARCH_SRC)


def test_a_configured_store_or_no_run_keeps_the_environment(tmp_path, monkeypatch):
    env = custody(tmp_path)
    monkeypatch.setenv("HOME", env["HOME"])
    assert "AWS_ACCESS_KEY_ID" not in store.store_env({**env, "RESEARCH_STORE_CONFIG": "/etc/store.toml"})
    assert store.custody_dir({"HOME": env["HOME"]}) is None
    assert "RESEARCH_STORE_CONFIG" not in store.store_env({"HOME": env["HOME"]})
