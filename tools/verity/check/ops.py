"""What the check takes from `research`, the team's run, store and queue tooling: every file under `tools/verity/check/` reaches
research through this module and nowhere else (`tests/test_ops.py`), so the check can run from another repository than
research's by changing the module paths here alone.

    OPS = _load("ops")                       # a sibling loads it by path, as check.py loads its others
    store = OPS.LocalStore()                 # a name of USES or TESTS, looked up in research when it is read

A name is resolved each time it is read, never kept, so research is imported only by the code that needs it (the preflight runs
before anything but research may be installed) and a test that patches research's module is what the check then reads.
"""
from __future__ import annotations

import importlib

#: what the check's own code takes from research: {its name here: `module` (the module itself) or `module:attribute`}
USES = {
    "remote": "research.remote",
    "targets": "research.checkview:targets",
    "slot_only": "research.pods.nebius.slot_only",
    "rule_touches": "research.merge:rule_touches",
    "shard_set": "research.merge:shard_set",
    "new_run_id": "research.runs:new_run_id",
    "runs_root": "research.runs:runs_root",
    "custody_env": "research.store.custody:custody_env",
    "public_url": "research.store.fetchset:public_url",
    "_is_tracked": "research.store.fixtures:_is_tracked",
    "entries": "research.store.fixtures:entries",
    "fixture_id": "research.store.fixtures:fixture_id",
    "load_map": "research.store.fixtures:load_map",
    "tracked": "research.store.fixtures:tracked",
    "LocalStore": "research.store.local:LocalStore",
    "manifest_key": "research.store.local:manifest_key",
    "object_key": "research.store.local:object_key",
    "HttpRemote": "research.store.remote:HttpRemote",
    "load_remote": "research.store.remote:load_remote",
    "S3Remote": "research.store.remote_s3:S3Remote",
    "Tool": "research.store.tool:Tool",
}

#: what only its tests take beside those: the modules they patch or drive, and the names they compare with
TESTS = {
    "auto": "research.auto",
    "clone": "research.clone",
    "custody": "research.store.custody",
    "mem": "research.mem",
    "merge": "research.merge",
    "runpod": "research.pods.runpod",
    "queue": "research.queue",
    "result": "research.result",
    "runs": "research.runs",
    "store_local": "research.store.local",
    "store_remote": "research.store.remote",
    "REFUSED_KINDS": "research.merge:REFUSED_KINDS",
    "SHARD_KIND": "research.merge:SHARD_KIND",
    "plan_refusal": "research.merge:plan_refusal",
    "post_landing": "research.merge:post_landing",
    "unlike": "research.merge:unlike",
    "DRIVE_FLAGS": "research.queue:DRIVE_FLAGS",
    "RULES_BEFORE_MOVE": "research.queue:RULES_BEFORE_MOVE",
    "CI_JOBS": "research.queue:CI_JOBS",
    "DriveFlags": "research.queue:DriveFlags",
    "Rules": "research.queue:Rules",
    "Refused": "research.remote:Refused",
    "dump_map": "research.store.fixtures:dump_map",
    "FsRemote": "research.store.remote:FsRemote",
}


def __getattr__(name: str):
    where = USES.get(name) or TESTS.get(name)
    if where is None:
        raise AttributeError(f"ops: the check takes no {name!r} from research: name it in ops.py's USES (or TESTS, for a test's)")
    module, _, attr = where.partition(":")
    m = importlib.import_module(module)
    return getattr(m, attr) if attr else m
