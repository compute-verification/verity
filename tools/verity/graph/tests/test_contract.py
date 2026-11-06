"""The contract's digests and key rule reproduce `key_vectors.json`, the vectors Verity and the executor share.

    python tools/verity/graph/tests/test_contract.py --write    # rewrite key_vectors.json from CASES (a contract change only)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from verity_graph import contract

VECTORS = Path(__file__).resolve().parent / "key_vectors.json"
FILES = {"a.py": b"print('a')\n", "b/c.lean": b"theorem t : True := trivial\n", "empty": b"",
         "tests/\u00fc.py": "\u00fc\n".encode(), "tests/\uff21.py": "\uff21\n".encode(), "tests/\U0001f600.py": "\U0001f600\n".encode(),
         "uv.lock": b"version = 1\n", "pyproject.toml": b"[project]\nname = \"x\"\n"}
ENVS = {"py": ["uv.lock", "pyproject.toml"], "none": []}
OLEAN = {"olean": contract.digest(b"olean"), "ilean": contract.digest(b"ilean")}


def _task(tid: str, kind: str, **kw) -> dict:
    return {"id": tid, "kind": kind, "argv": ["verity", "graph", "run", tid], "vars": {}, "inputs": [], "reads": [],
            "outputs": [], **kw}


A = _task("test-file:a.py", "test-file", inputs=["a.py"])
#: (name, task, env, read digests), each a case the key rule must get right
CASES = [
    ("inputs only", A, "py", {}),
    ("inputs and outputs, unsorted", _task("lean-module:b/c.lean", "lean-module", inputs=["b/c.lean", "a.py"],
                                           outputs=[{"name": "olean", "path": "out/c.olean"},
                                                    {"name": "ilean", "path": "out/c.ilean"}]), "py", {}),
    ("no inputs, outputs or vars", _task("build:nothing", "build", argv=["true"]), "none", {}),
    ("one read, of one of two outputs", _task("lean-module:d.lean", "lean-module", inputs=["empty"],
                                              reads=[{"task": "lean-module:b/c.lean", "output": "olean"}]),
     "py", {"lean-module:b/c.lean": OLEAN}),
    ("reads of two tasks, unsorted", _task("test-file:a.py", "test-file", inputs=["a.py"],
                                           reads=[{"task": "z:t", "output": "x"}, {"task": "a:t", "output": "y"},
                                                  {"task": "a:t", "output": "b"}]),
     "py", {"z:t": {"x": contract.digest(b"x")}, "a:t": {"y": contract.digest(b"y"), "b": contract.digest(b"b")}}),
    ("unicode paths, sorted by code point", _task("test-file:tests/\U0001f600.py", "test-file",
                                                  inputs=["tests/\U0001f600.py", "tests/\uff21.py", "tests/\u00fc.py"],
                                                  outputs=[{"name": "r\u00e9sum\u00e9", "path": "out/\u00fc\U0001f600.txt"}]),
     "py", {}),
    ("vars with escapes and unicode keys", _task("test-file:a.py", "test-file", inputs=["a.py"],
                                                 vars={"\uff21": "1", "\U0001f600": "2", "": "",
                                                       "a": "x\"y\\z\n\t\u0001\u007f<>&\u2028"}), "py", {}),
    ("argv with escapes, not normalized", _task("test-file:a.py", "test-file", inputs=["a.py"],
                                                argv=["sh", "-c", "echo \"$X\" > out/a\\b", "e\u0301", "\u00e9"]), "py", {}),
    ("inputs only, another env", A, "none", {}),
    ("inputs only, with the fields the key leaves out", {**A, "id": "test-file:elsewhere", "about": "run a.py", "env": "py",
                                                         "resources": {"cores": 2, "mem_gb": 3, "disk_gb": 4},
                                                         "timeout_s": 60, "priority": 3, "key": None}, "py", {}),
]


def file_digests() -> dict[str, str]:
    return {p: contract.digest(b) for p, b in FILES.items()}


def env_digests() -> dict[str, str]:
    return {n: contract.env_digest(sp, file_digests()) for n, sp in ENVS.items()}


def vectors() -> dict:
    fd, ed = file_digests(), env_digests()
    keys = []
    for name, task, env, reads in CASES:
        doc = contract.key_doc(task, fd, reads, ed[env])
        keys.append({"name": name, "task": task, "env_digest": ed[env], "read_digests": reads,
                     "canonical": contract.canonical(doc).decode(), "key": contract.key(task, fd, reads, ed[env])})
    return {"files": [{"path": p, "bytes_hex": b.hex(), "digest": fd[p]} for p, b in FILES.items()],
            "envs": [{"name": n, "spec": sp, "digest": ed[n]} for n, sp in ENVS.items()], "keys": keys}


@pytest.fixture(scope="module")
def frozen() -> dict:
    return json.loads(VECTORS.read_text())


def test_file_and_env_digests_reproduce_the_vectors(frozen):
    for f in frozen["files"]:
        assert contract.digest(bytes.fromhex(f["bytes_hex"])) == f["digest"], f["path"]
    fd = {f["path"]: f["digest"] for f in frozen["files"]}
    for e in frozen["envs"]:
        assert contract.env_digest(e["spec"], fd) == e["digest"], e["name"]


def test_keys_reproduce_the_vectors(frozen):
    fd = {f["path"]: f["digest"] for f in frozen["files"]}
    assert len(frozen["keys"]) >= 10
    for v in frozen["keys"]:
        doc = contract.key_doc(v["task"], fd, v["read_digests"], v["env_digest"])
        assert contract.canonical(doc).decode() == v["canonical"], v["name"]
        assert contract.key(v["task"], fd, v["read_digests"], v["env_digest"]) == v["key"], v["name"]
        assert v["key"] == "sha512:" + hashlib.sha512(b"verity-key/v1\n" + v["canonical"].encode()).hexdigest(), v["name"]


def test_the_listing_s_keyer_is_the_key(frozen):
    fd = {f["path"]: f["digest"] for f in frozen["files"]}
    key = contract.keyer(fd)
    for v in frozen["keys"]:
        assert key(v["task"], v["read_digests"], v["env_digest"]) == v["key"], v["name"]
    for p in fd:
        t = _task("test-file:x", "test-file", inputs=[p], argv=['"inputs":[', p], vars={"inputs": '"inputs":['})
        assert key(t, {}, "e") == contract.key(t, fd, {}, "e"), p


def test_the_vectors_are_the_cases(frozen):
    """The file is what CASES make: a change to either is a change to the contract, made with `--write`."""
    assert frozen == vectors()


def test_canonical_json_is_the_one_the_contract_names():
    """Sorted keys, no spaces, raw UTF-8, Python's escapes for control characters, nothing else escaped."""
    v = {"b": ["\u00e9", "\U0001f600", "<>&", "\u2028"], "a": {"y": "\"\\\n\t\u0001\u007f", "x": []}}
    want = '{"a":{"x":[],"y":"\\"\\\\\\n\\t\\u0001\u007f"},"b":["\u00e9","\U0001f600","<>&","\u2028"]}'
    assert contract.canonical(v) == want.encode("utf-8")


def test_unicode_paths_sort_by_code_point_not_utf16(frozen):
    v = next(v for v in frozen["keys"] if v["name"].startswith("unicode"))
    got = [p for p, _ in json.loads(v["canonical"])["inputs"]]
    assert got == ["tests/\u00fc.py", "tests/\uff21.py", "tests/\U0001f600.py"]


def test_the_key_leaves_out_about_resources_timeout_priority_and_id():
    fd, ed = file_digests(), env_digests()
    other = {**A, "id": "test-file:b", "about": "x", "resources": {"cores": 9, "mem_gb": 9, "disk_gb": 9}, "timeout_s": 1,
             "priority": 7}
    assert contract.key(A, fd, {}, ed["py"]) == contract.key(other, fd, {}, ed["py"])
    assert contract.key(A, fd, {}, ed["py"]) != contract.key({**A, "vars": {"x": ""}}, fd, {}, ed["py"])


def test_a_read_enters_by_its_digest_alone():
    """Early cutoff: an artifact rebuilt to the same bytes leaves its readers' keys unchanged; other bytes change them."""
    fd, ed = file_digests(), env_digests()
    t = CASES[3][1]
    k = contract.key(t, fd, {"lean-module:b/c.lean": OLEAN}, ed["py"])
    same = {"lean-module:b/c.lean": {"olean": contract.digest(b"olean"), "ilean": contract.digest(b"other")}}
    assert contract.key(t, fd, same, ed["py"]) == k
    assert contract.key(t, fd, {"lean-module:b/c.lean": {"olean": contract.digest(b"v2")}}, ed["py"]) != k
    with pytest.raises(KeyError, match="isn't known yet"):
        contract.key(t, fd, {}, ed["py"])


def test_result_errors_name_what_is_wrong():
    ok = {"schema": contract.SCHEMA_RESULT, "task": "t:x", "key": None, "status": "pass", "seconds": 1.0, "cpu_s": 0.5,
          "peak_rss_gb": 0.1, "outputs": {}, "undeclared_reads": [], "detail": {}, "log": contract.digest(b"")}
    assert contract.result_errors(ok) == []
    assert contract.result_errors({**ok, "owner": "x"}) == ["field owner is not result/v1's"]
    assert contract.result_errors({**ok, "status": "skipped"}) == ["status 'skipped', not one of pass, fail, error"]
    assert contract.result_errors({**ok, "undeclared_reads": ["x"]}) == [
        "a task with an undeclared read is an error, never a result"]
    assert contract.result_errors(ok, {"id": "t:y", "outputs": []}) == ["the result is t:x's, not t:y's"]


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit(__doc__)
    VECTORS.write_text(json.dumps(vectors(), indent=1, ensure_ascii=False) + "\n")
