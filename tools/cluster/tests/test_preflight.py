"""`cluster preflight`: each check from a node's sectioned output, the node's input checker run here on a made-up store, hub
and model cache, and the CLI with a fake SSH runner."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys

from test_status import DF, T, node2_snapshot

from cluster import cli, nebius1, nebius2, preflight, status

ART = "f286e8fbcb44530f7e5a9544b47c38a6311a9799f1f7e4e4197ab688d0ce9611"
REV = "25319945aa7bc3ad1e8b3b1d6c4a5e0f9d2c7b18"
R2 = {k: "dummy-not-a-credential" for k in preflight.R2_ENV}
MAIN = preflight.main
WRITES = re.compile(r"\b(patch|delete|apply|create|edit|label|annotate|scale|cordon|drain|rm|mv|kill|tee|chmod|chown|touch|"
                    r"mkdir|ln|cp|systemctl|crontab)\b|>(?!/dev/null|&1)")


def out(**sections: tuple[int, str]) -> str:
    """What `script`'s command prints: each section's marker, its output and its exit status."""
    return "".join(f"{preflight.MARK}:{k}\n{body}{'' if body.endswith(chr(10)) or not body else chr(10)}"
                   f"{preflight.MARK}-rc:{rc}\n" for k, (rc, body) in sections.items())


def kueue(*stopped: str) -> str:
    qs = [{"metadata": {"name": n}, "spec": {"stopPolicy": "Hold" if n in stopped else "None"}}
          for n in ("deployments-gpu", "circuits-gpu", *stopped)]
    return json.dumps({"items": qs})


def state(mtime: int, snap: dict | None = None) -> str:
    return f"/workspace/pouw/infra/cluster/live/state.json {mtime}\n" + json.dumps((snap or node2_snapshot())["state"])


def by_check(rows: list[dict]) -> list[tuple[str, bool]]:
    return [(r["check"], r["ok"]) for r in rows]


def test_refs_take_art_model_and_hf_refs_only_once_and_refuse_anything_else():
    assert preflight.refs([f"art:{ART}", "model:Qwen/Qwen3-4B@main", f"hf:unsloth/gemma-2-2b@{REV}", f"art:{ART}"]) == \
        [f"art:{ART}", "model:Qwen/Qwen3-4B@main", f"hf:unsloth/gemma-2-2b@{REV}"]
    for bad in ("art:abc", f"art:{ART.upper()}", "model:Qwen3-4B@main", "hf:../../etc@x", "hf:a/b@c;rm -rf /", "lean:x/y@z",
                f"art:{ART} ", "model:a/b"):
        try:
            preflight.refs([bad])
        except preflight.Refused as e:
            assert repr(bad)[1:-1] in str(e) or bad in str(e)
        else:
            raise AssertionError(f"{bad!r} was taken")


def test_the_script_is_one_read_only_command_with_a_section_per_check():
    n1 = preflight.script(nebius1.NODE, [f"art:{ART}"], custody=True)
    n2 = preflight.script(nebius2.NODE, [f"art:{ART}"], custody=True)
    assert [m for m in re.findall(rf"{preflight.MARK}:(\w+)", n1)] == ["kueue", "df", "inputs", "custody"]
    assert [m for m in re.findall(rf"{preflight.MARK}:(\w+)", n2)] == ["state", "df", "inputs", "custody"]
    assert "get secret research-r2 -o name" in n1 and "-o yaml" not in n1 and "jsonpath" not in n1
    assert "ssh -o BatchMode=yes -o ConnectTimeout=5 vy-n1 'sudo -n k3s kubectl get secret research-r2 -o name'" in n2
    assert status.COMMANDS["state"] in n2 and "/workspace/jobs/hf/hub" in n2 and "/workspace/hf/hub" in n1
    assert preflight.script(nebius1.NODE, [], custody=False).count(preflight.MARK) == 4, "only kueue and df without either"
    for cmd in (n1, n2):
        assert not WRITES.search(cmd), cmd
    assert not re.search(r"open\([^)]*['\"][wax+]|os\.(remove|unlink|rename|replace|mkdir|makedirs|symlink|chmod)", preflight.INPUTS)


def test_sections_split_the_output_by_marker_with_each_exit_status():
    got = preflight.sections(out(df=(0, DF), kueue=(1, "error: the server doesn't have a resource type"), inputs=(0, "")))
    assert got == {"df": (0, DF), "kueue": (1, "error: the server doesn't have a resource type\n"), "inputs": (0, "")}
    assert preflight.sections("ssh banner only\n") == {}


def test_node1_passes_when_kueue_answers_names_its_stopped_queues_and_holds_the_secret():
    rows = preflight.checks(nebius1.NODE, "research@n1", out(kueue=(0, kueue("circuits"))), None, T, [], False, {})
    assert by_check(rows) == [("ssh", True), ("state", False)], "no df section fails state"
    o = out(kueue=(0, kueue("circuits")), df=(0, DF), custody=(0, "secret/research-r2"))
    rows = preflight.checks(nebius1.NODE, "research@n1", o, None, T, [], True, {})
    assert by_check(rows) == [("ssh", True), ("state", True), ("custody", True)]
    assert rows[1]["detail"] == "Kueue answers; stopped ClusterQueues: circuits; /workspace 70% used"
    assert rows[2]["detail"] == "node 1 holds Secret research-r2"
    o = out(kueue=(1, "The connection to the server was refused"), df=(0, DF),
            custody=(1, 'Error from server (NotFound): secrets "research-r2" not found'))
    rows = preflight.checks(nebius1.NODE, "research@n1", o, None, T, [], True, {})
    assert by_check(rows) == [("ssh", True), ("state", False), ("custody", False)]
    assert rows[1]["detail"] == "Kueue doesn't answer: The connection to the server was refused"
    assert "doesn't show Secret research-r2: Error from server (NotFound)" in rows[2]["detail"]


def test_node2_state_is_fresh_by_its_files_mtime_and_held_back_by_a_timed_window():
    quiet = out(state=(0, state(T + 597)), df=(0, DF), custody=(0, "secret/research-r2"))
    rows = preflight.checks(nebius2.NODE, "research@n2", quiet, None, T + 600, [], True, {})
    assert by_check(rows) == [("ssh", True), ("state", True), ("custody", True)], "its last record is 10 min old, its file 3 s"
    assert rows[1]["detail"] == "the agent's ledger state is seq 479, 3s old; /workspace 70% used"
    assert rows[2]["detail"] == "node 1, which node 2 reaches as vy-n1, holds Secret research-r2"
    stale = preflight.checks(nebius2.NODE, "research@n2", quiet, None, T + 600 + status.STALE_S, [], False, {})[1]
    assert stale == {"check": "state", "ok": False,
                     "detail": "the agent's ledger state is seq 479, 1m old: is the agent running?; /workspace 70% used"}
    timed = node2_snapshot()
    timed["state"]["allocations"][0]["job"]["quiet"] = True
    held = preflight.checks(nebius2.NODE, "n2", out(state=(0, state(T, timed)), df=(0, DF)), None, T + 90, [], False, {})[1]
    assert held["ok"] is True and "a timed window holds the node" in held["detail"]
    missing = preflight.checks(nebius2.NODE, "research@n2", out(state=(1, ""), df=(0, DF)), None, T, [], False, {})[1]
    assert missing == {"check": "state", "ok": False, "detail": "no agent state: no output"}


def test_custody_passes_on_the_callers_r2_credentials_by_name_without_printing_them():
    o = out(kueue=(0, kueue()), df=(0, DF), custody=(1, "sudo: a password is required"))
    rows = preflight.checks(nebius1.NODE, "research@n1", o, None, T, [], True, R2)
    assert rows[-1] == {"check": "custody", "ok": True,
                        "detail": "R2 credentials in the caller's environment (R2_ENDPOINT, R2_BUCKET, AWS_ACCESS_KEY_ID, "
                                  "AWS_SECRET_ACCESS_KEY)"}
    partial = {**R2, "AWS_SECRET_ACCESS_KEY": ""}
    rows = preflight.checks(nebius1.NODE, "research@n1", o, None, T, [], True, partial)
    assert rows[-1]["ok"] is False and "sudo: a password is required" in rows[-1]["detail"]
    assert "dummy-not-a-credential" not in json.dumps(rows)


def test_without_ssh_every_node_check_fails_and_names_why():
    rows = preflight.checks(nebius2.NODE, "research@n2", None, "ReadError: Permission denied (publickey).", T,
                            [f"art:{ART}"], True, {})
    assert by_check(rows) == [("ssh", False), ("state", False), ("input", False), ("custody", False)]
    assert rows[0]["detail"] == "research@n2: ReadError: Permission denied (publickey)."
    assert rows[2]["detail"] == f"art:{ART}: not read: no SSH"
    assert preflight.checks(nebius2.NODE, "d", None, "x", T, [], True, R2)[-1]["ok"] is True


def _node(tmp_path, materialized: bool) -> dict:
    """A store, model cache and hub on disk, in the shapes the nodes hold them."""
    manifests, models, hub = tmp_path / "manifests", tmp_path / "models", tmp_path / "hub"
    manifests.mkdir()
    body = b'{"files": []}\n'
    good = hashlib.sha256(body).hexdigest()
    (manifests / f"{good}.json").write_bytes(body)
    (manifests / f"{ART}.json").write_bytes(b"not the bytes its id hashes\n")
    (models / "Qwen--Qwen3-4B@main").mkdir(parents=True)
    (models / "Qwen--Qwen3-4B@main" / ".entry.json").write_text("{}")
    (models / "Qwen--Qwen3-8B@main").mkdir(parents=True)
    repo = hub / "models--unsloth--gemma-2-2b"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(REV + "\n")
    snap = repo / "snapshots" / REV
    snap.mkdir(parents=True)
    (repo / "blobs").mkdir()
    (repo / "blobs" / "abc").write_text("{}")
    if materialized:
        (snap / "config.json").write_text("{}")
    else:
        os.symlink(repo / "blobs" / "abc", snap / "config.json")
    bare = hub / "models--org--noconfig" / "snapshots" / REV
    bare.mkdir(parents=True)
    linked = hub / "models--org--linked" / "snapshots" / REV
    linked.mkdir(parents=True)
    (linked / "config.json").write_text("{}")
    os.symlink(tmp_path / "gone", linked / "model.safetensors")
    return {"manifests": str(manifests), "models": str(models), "hub": str(hub), "materialized": materialized, "good": good}


def _inputs(args: dict, refs: list[str]) -> list:
    r = subprocess.run([sys.executable, "-c", preflight.INPUTS, json.dumps({**args, "refs": refs})],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def test_the_input_checker_hashes_manifests_finds_model_entries_and_reads_hf_snapshots(tmp_path):
    args = _node(tmp_path, materialized=True)
    refs = [f"art:{args['good']}", f"art:{ART}", f"art:{'0' * 64}", "model:Qwen/Qwen3-4B@main", "model:Qwen/Qwen3-8B@main",
            "hf:unsloth/gemma-2-2b@main", f"hf:unsloth/gemma-2-2b@{REV}", "hf:unsloth/gemma-2-2b@v9", f"hf:org/noconfig@{REV}",
            f"hf:org/linked@{REV}"]
    got = {ref: (ok, detail.replace(str(tmp_path), "")) for ref, ok, detail in _inputs(args, refs)}
    assert got == {
        f"art:{args['good']}": (True, f"/manifests/{args['good']}.json hashes to its id"),
        f"art:{ART}": (False, f"/manifests/{ART}.json hashes to "
                              f"{hashlib.sha256(b'not the bytes its id hashes' + bytes([10])).hexdigest()}, not its id"),
        f"art:{'0' * 64}": (False, f"no manifest /manifests/{'0' * 64}.json"),
        "model:Qwen/Qwen3-4B@main": (True, "/models/Qwen--Qwen3-4B@main/.entry.json is there"),
        "model:Qwen/Qwen3-8B@main": (False, "/models/Qwen--Qwen3-8B@main/.entry.json is missing"),
        "hf:unsloth/gemma-2-2b@main": (True, f"/hub/models--unsloth--gemma-2-2b/snapshots/{REV} has config.json, no symlinks"),
        f"hf:unsloth/gemma-2-2b@{REV}": (True, f"/hub/models--unsloth--gemma-2-2b/snapshots/{REV} has config.json, no symlinks"),
        "hf:unsloth/gemma-2-2b@v9": (False, "/hub/models--unsloth--gemma-2-2b/snapshots/v9 is missing"),
        f"hf:org/noconfig@{REV}": (False, f"/hub/models--org--noconfig/snapshots/{REV} has no config.json"),
        f"hf:org/linked@{REV}": (False, f"/hub/models--org--linked/snapshots/{REV} holds 1 symlinks; "
                                        "this node's jobs read materialized snapshots"),
    }


def test_node1s_hub_may_link_into_its_blobs_but_no_link_may_dangle(tmp_path):
    args = _node(tmp_path, materialized=False)
    got = {ref: (ok, detail.replace(str(tmp_path), ""))
           for ref, ok, detail in _inputs(args, ["hf:unsloth/gemma-2-2b@main", f"hf:org/linked@{REV}"])}
    assert got == {"hf:unsloth/gemma-2-2b@main": (True, f"/hub/models--unsloth--gemma-2-2b/snapshots/{REV} has config.json, "
                                                        "1 symlinks into its blobs"),
                   f"hf:org/linked@{REV}": (False, f"/hub/models--org--linked/snapshots/{REV} has 1 dangling symlinks")}


def test_input_rows_come_from_the_checker_and_a_ref_it_didnt_check_fails(tmp_path):
    args = _node(tmp_path, materialized=True)
    body = json.dumps(_inputs(args, [f"art:{args['good']}"]))
    o = out(kueue=(0, kueue()), df=(0, DF), inputs=(0, body))
    rows = preflight.checks(nebius1.NODE, "n1", o, None, T, [f"art:{args['good']}", "model:a/b@c"], False, {})
    assert by_check(rows) == [("ssh", True), ("state", True), ("input", True), ("input", False)]
    assert rows[3]["detail"].startswith("model:a/b@c: not checked: ")
    crashed = out(kueue=(0, kueue()), df=(0, DF), inputs=(1, "python3: command not found"))
    rows = preflight.checks(nebius1.NODE, "n1", crashed, None, T, [f"art:{ART}"], False, {})
    assert rows[-1] == {"check": "input", "ok": False, "detail": f"art:{ART}: not checked: python3: command not found"}


def _cli(capsys, monkeypatch, argv, reply, env=None):
    sent: list[tuple[str, str]] = []

    def runner(dest):
        def run(cmd):
            sent.append((dest, cmd))
            if isinstance(reply, Exception):
                raise reply
            return reply
        return run

    monkeypatch.setattr(preflight, "clock", lambda: T)
    monkeypatch.setattr(preflight, "main", lambda node, inputs, custody, as_json:
                        MAIN(node, inputs, custody, as_json, runner=runner, env=env or {}))
    rc = cli.main(argv)
    return rc, capsys.readouterr().out, sent


def test_cli_preflight_prints_one_line_per_check_and_exits_0_only_when_all_pass(capsys, monkeypatch):
    ok = out(state=(0, state(T - 2)), df=(0, DF), inputs=(0, json.dumps([[f"art:{ART}", True, "m hashes to its id"]])),
             custody=(0, "secret/research-r2"))
    rc, text, sent = _cli(capsys, monkeypatch, ["preflight", "--node", "nebius-2", "--input", f"art:{ART}", "--custody"], ok)
    assert rc == 0 and sent == [(status.HOSTS[nebius2.NODE], preflight.script(nebius2.NODE, [f"art:{ART}"], True))]
    assert text.splitlines() == [
        f"ok    ssh      {status.HOSTS[nebius2.NODE]} answered the caller's key",
        "ok    state    the agent's ledger state is seq 479, 2s old; /workspace 70% used",
        f"ok    input    art:{ART}: m hashes to its id",
        "ok    custody  node 1, which node 2 reaches as vy-n1, holds Secret research-r2",
        "preflight vy-nebius-2: all 4 checks passed"]
    bad = ok.replace('true, "m hashes', 'false, "m hashes')
    rc, text, _ = _cli(capsys, monkeypatch, ["preflight", "--node", "nebius-2", "--input", f"art:{ART}", "--json"], bad)
    doc = json.loads(text)
    assert rc == 1 and (doc["schema"], doc["node"], doc["t"], doc["ok"]) == ("cluster/preflight/v1", nebius2.NODE, T, False)
    assert [c["check"] for c in doc["checks"] if not c["ok"]] == ["input"]


def test_cli_preflight_refuses_a_malformed_ref_before_reaching_the_node_and_fails_when_ssh_does(capsys, monkeypatch):
    rc, text, sent = _cli(capsys, monkeypatch, ["preflight", "--node", "nebius-1", "--input", "hf:x/y@z;reboot"], "")
    assert (rc, sent) == (2, []) and "not an art:<sha256>, model:ORG/NAME@REV or hf:ORG/NAME@REV ref: hf:x/y@z;reboot" in text
    rc, text, _ = _cli(capsys, monkeypatch, ["preflight", "--node", "nebius-1"],
                       status.ReadError("ssh: connect to host 81.85.2.165 port 22: Connection timed out"))
    assert rc == 1 and text.splitlines()[0] == (f"FAIL  ssh      {status.HOSTS[nebius1.NODE]}: ReadError: ssh: connect to host "
                                                "81.85.2.165 port 22: Connection timed out")
    assert text.splitlines()[-1] == "preflight vy-nebius-1: 2 of 2 checks failed"
