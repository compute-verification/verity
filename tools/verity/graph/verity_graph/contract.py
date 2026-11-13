"""The contract between Verity and an executor: the schema names, canonical JSON, file and env digests, the key rule, and
result/v1's shape (README.md).  `tests/key_vectors.json` holds vectors both sides reproduce."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA_GRAPH = "verity-graph/v2"
#: graph/v2 packed (`graph.pack`): the same document, each task's inputs and reads as runs of indices
SCHEMA_PACKED = "verity-graph/v2-packed"
SCHEMA_RESULT = "verity-result/v1"
SCHEMA_EVENTS = "verity-events/v1"
KEY_TAG = b"verity-key/v1\n"
ENV_TAG = b"verity-env/v1\n"
STATUSES = ("pass", "fail", "error")
#: result/v1's fields, each with the types its value may have
RESULT_FIELDS = {"schema": (str,), "task": (str,), "key": (str, type(None)), "status": (str,), "seconds": (int, float),
                 "cpu_s": (int, float), "peak_rss_gb": (int, float), "outputs": (dict,), "undeclared_reads": (list,),
                 "detail": (dict,), "log": (str, type(None))}
EVENTS = ("queued", "hit", "started", "finished", "errored", "cancelled")
PURPOSES = ("after-landing", "ready", "lean-check", "nightly", "adhoc")


def canonical(x) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(data: bytes) -> str:
    return "sha512:" + hashlib.sha512(data).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha512()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha512:" + h.hexdigest()


def env_digest(spec: list[str], file_digests: dict[str, str]) -> str:
    """An env's digest: SHA-512 over its spec files' paths and digests, sorted."""
    return digest(ENV_TAG + canonical(sorted([p, file_digests[p]] for p in spec)))


def key_doc(task: dict, file_digests: dict[str, str], read_digests: dict[str, dict[str, str]] | None, env_digest: str) -> dict:
    """What `key` hashes. ``read_digests`` is {task id: {output name: digest}}, each read task's result/v1 `outputs`; KeyError
    names a read whose digest it lacks."""
    reads = []
    for r in task["reads"]:
        got = (read_digests or {}).get(r["task"], {}).get(r["output"])
        if got is None:
            raise KeyError(f"{task['id']} reads {r['task']}'s {r['output']}, whose digest isn't known yet")
        reads.append([r["task"], r["output"], got])
    return {"env": env_digest, "kind": task["kind"], "argv": list(task["argv"]), "vars": dict(task["vars"]),
            "inputs": sorted([p, file_digests[p]] for p in task["inputs"]), "reads": sorted(reads),
            "outputs": sorted([o["name"], o["path"]] for o in task["outputs"])}


def key(task: dict, file_digests: dict[str, str], read_digests: dict[str, dict[str, str]] | None, env_digest: str) -> str:
    """The task's key: `about`, `resources`, `timeout_s` and `priority` stay out of it, and a read enters by its output's
    digest, so an artifact rebuilt to the same bytes leaves its readers' keys unchanged."""
    return digest(KEY_TAG + canonical(key_doc(task, file_digests, read_digests, env_digest)))


def keyer(file_digests: dict[str, str]):
    """`key` over ``file_digests``, the same digest, with each input's `[path, file digest]` encoded once for every task that
    reads it: the listing keys thousands of tasks that share most of their thousands of inputs."""
    pairs: dict[str, bytes] = {}

    def pair(p: str) -> bytes:
        if (b := pairs.get(p)) is None:
            b = pairs[p] = canonical([p, file_digests[p]])
        return b

    def of(task: dict, read_digests: dict[str, dict[str, str]] | None, env_digest: str) -> str:
        doc = key_doc({**task, "inputs": []}, file_digests, read_digests, env_digest)
        head, mark, tail = canonical(doc).partition(b'"inputs":[')
        h = hashlib.sha512(KEY_TAG + head + mark)
        h.update(b",".join([pair(p) for p in sorted(task["inputs"])]))
        h.update(tail)
        return "sha512:" + h.hexdigest()
    return of


def result_errors(res: object, task: dict | None = None) -> list[str]:
    """What makes ``res`` no result/v1 (of ``task``, when given): an empty list when it is one."""
    if not isinstance(res, dict):
        return [f"a result is a JSON object, not {type(res).__name__}"]
    errs = [f"no field {f}" for f in RESULT_FIELDS if f not in res]
    errs += [f"field {f} is not result/v1's" for f in res if f not in RESULT_FIELDS]
    errs += [f"{f} is {type(res[f]).__name__}" for f, ts in RESULT_FIELDS.items() if f in res and not isinstance(res[f], ts)]
    if errs:
        return errs
    if res["schema"] != SCHEMA_RESULT:
        errs.append(f"schema {res['schema']!r}, not {SCHEMA_RESULT}")
    if res["status"] not in STATUSES:
        errs.append(f"status {res['status']!r}, not one of {', '.join(STATUSES)}")
    if any(not isinstance(k, str) or not isinstance(v, str) or not v.startswith("sha512:") for k, v in res["outputs"].items()):
        errs.append("outputs is not {name: sha512 digest}")
    if any(not isinstance(p, str) for p in res["undeclared_reads"]):
        errs.append("undeclared_reads is not a list of paths")
    if res["undeclared_reads"] and res["status"] != "error":
        errs.append("a task with an undeclared read is an error, never a result")
    if task is not None:
        if res["task"] != task["id"]:
            errs.append(f"the result is {res['task']}'s, not {task['id']}'s")
        if extra := set(res["outputs"]) - {o["name"] for o in task["outputs"]}:
            errs.append(f"outputs {', '.join(sorted(extra))} aren't the task's")
    return errs
