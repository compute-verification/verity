"""`catalog.sqlite` (README §5): a disposable, rebuildable, LOCAL ONLY index over manifests/, attempts/, labels/.

Denormalised tables; add columns freely, the file is never uploaded and `research data reindex` recreates it.  Flattening:
nested objects get dotted keys, lists get numeric components (`families.0`), scalars are stored as text (`json.dumps` for
non-strings, so `true`, `null`, `1.5`); `--where meta.key=value` compares that text.  `sql()` opens a separate read-only
connection (`?mode=ro`) so the escape hatch cannot write.  A `replicas` table records `verified_utc` per (artifact, location);
it is the only state here that a rebuild cannot recover (verification is a remote round-trip, not a local fact).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .model import Label

SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts(id TEXT PRIMARY KEY, kind TEXT, bytes INTEGER, payload_type TEXT, manifest_json TEXT,
                                     local INTEGER, remote INTEGER, repo_path TEXT);
CREATE INDEX IF NOT EXISTS artifacts_kind ON artifacts(kind);
CREATE TABLE IF NOT EXISTS artifact_meta(id TEXT, key TEXT, value TEXT);
CREATE INDEX IF NOT EXISTS artifact_meta_kv ON artifact_meta(key, value);
CREATE INDEX IF NOT EXISTS artifact_meta_id ON artifact_meta(id);
CREATE TABLE IF NOT EXISTS refs(src TEXT, name TEXT, dst TEXT);
CREATE INDEX IF NOT EXISTS refs_src ON refs(src);
CREATE INDEX IF NOT EXISTS refs_dst ON refs(dst);
CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY, tool TEXT, tool_version TEXT, derivation TEXT, reuse_key TEXT, state TEXT, rc INTEGER,
                                    validation TEXT, start_utc TEXT, wall_s REAL, host TEXT, project TEXT, campaign TEXT, cost_usd REAL,
                                    source_commit TEXT, source_dirty INTEGER, manifest_json TEXT);
CREATE INDEX IF NOT EXISTS attempts_reuse ON attempts(reuse_key);
CREATE INDEX IF NOT EXISTS attempts_drv ON attempts(derivation);
CREATE INDEX IF NOT EXISTS attempts_tool ON attempts(tool);
CREATE TABLE IF NOT EXISTS attempt_params(attempt TEXT, key TEXT, value TEXT);
CREATE INDEX IF NOT EXISTS attempt_params_kv ON attempt_params(key, value);
CREATE TABLE IF NOT EXISTS attempt_conditions(attempt TEXT, key TEXT, value TEXT);
CREATE INDEX IF NOT EXISTS attempt_conditions_kv ON attempt_conditions(key, value);
CREATE TABLE IF NOT EXISTS attempt_outputs(attempt TEXT, name TEXT, artifact TEXT);
CREATE INDEX IF NOT EXISTS attempt_outputs_art ON attempt_outputs(artifact);
CREATE TABLE IF NOT EXISTS attempt_inputs(attempt TEXT, name TEXT, artifact TEXT);
CREATE INDEX IF NOT EXISTS attempt_inputs_art ON attempt_inputs(artifact);
CREATE TABLE IF NOT EXISTS labels(target TEXT, key TEXT, value TEXT, by TEXT, ts TEXT, ref TEXT, path TEXT PRIMARY KEY);
CREATE INDEX IF NOT EXISTS labels_target ON labels(target, key);
CREATE TABLE IF NOT EXISTS replicas(id TEXT, location TEXT, verified_utc TEXT, detail TEXT, PRIMARY KEY(id, location));
"""


def flatten(obj: Any, prefix: str = "") -> list[tuple[str, str]]:
    """Dotted (key, text) pairs for every scalar leaf.  Empty containers produce nothing."""
    if isinstance(obj, dict):
        out: list[tuple[str, str]] = []
        for k, v in obj.items():
            out += flatten(v, f"{prefix}.{k}" if prefix else str(k))
        return out
    if isinstance(obj, (list, tuple)):
        out = []
        for i, v in enumerate(obj):
            out += flatten(v, f"{prefix}.{i}" if prefix else str(i))
        return out
    return [(prefix, scalar_text(obj))]


def scalar_text(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v)


def _g(d: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


class Index:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)   # autocommit; explicit BEGIN where several writes must land together
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # ---- writes -------------------------------------------------------------------------------------------------------------------------

    def index_artifact(self, manifest_doc: dict, art_id: str, local: bool | None = None, remote: bool | None = None, repo_path: str | None = None) -> None:
        """Upsert one artifact.  `local`/`remote`/`repo_path` left None keep the current value (so a push can flip `remote` alone)."""
        p = manifest_doc.get("payload") or {}
        with self.db:
            self.db.execute("BEGIN")
            row = self.db.execute("SELECT local, remote, repo_path FROM artifacts WHERE id=?", (art_id,)).fetchone()
            cur = row or (0, 0, None)
            self.db.execute("INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?,?,?,?,?)",
                            (art_id, manifest_doc.get("kind"), p.get("bytes", 0) or 0, p.get("type"), json.dumps(manifest_doc, sort_keys=True),
                             int(cur[0] if local is None else local), int(cur[1] if remote is None else remote), cur[2] if repo_path is None else repo_path))
            self.db.execute("DELETE FROM artifact_meta WHERE id=?", (art_id,))
            self.db.executemany("INSERT INTO artifact_meta VALUES (?,?,?)", [(art_id, k, v) for k, v in flatten(manifest_doc.get("meta") or {})])
            self.db.execute("DELETE FROM refs WHERE src=?", (art_id,))
            rows = []
            for name, v in (manifest_doc.get("refs") or {}).items():
                for dst in (v if isinstance(v, list) else [v]):
                    rows.append((art_id, name, dst))
            self.db.executemany("INSERT INTO refs VALUES (?,?,?)", rows)

    def index_attempt(self, doc: dict) -> None:
        st, ex, src, tool = doc.get("status") or {}, doc.get("execution") or {}, doc.get("source") or {}, doc.get("tool") or {}
        aid = doc["id"]
        with self.db:
            self.db.execute("BEGIN")
            self.db.execute("INSERT OR REPLACE INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (aid, tool.get("name"), tool.get("version"), doc.get("derivation"), doc.get("reuse_key"), st.get("state"), st.get("rc"),
                             st.get("validation"), ex.get("start_utc"), ex.get("wall_s"), ex.get("host"), ex.get("project"), ex.get("campaign"),
                             _g(doc, "cost", "usd"), src.get("commit"), None if src.get("dirty") is None else int(bool(src.get("dirty"))),
                             json.dumps(doc, sort_keys=True)))
            for table, key in (("attempt_params", "params"), ("attempt_conditions", "conditions")):
                self.db.execute(f"DELETE FROM {table} WHERE attempt=?", (aid,))
                self.db.executemany(f"INSERT INTO {table} VALUES (?,?,?)", [(aid, k, v) for k, v in flatten(doc.get(key) or {})])
            for table, key in (("attempt_outputs", "outputs"), ("attempt_inputs", "inputs")):
                self.db.execute(f"DELETE FROM {table} WHERE attempt=?", (aid,))
                self.db.executemany(f"INSERT INTO {table} VALUES (?,?,?)",
                                    [(aid, n, a) for n, a in (doc.get(key) or {}).items() if isinstance(a, str)])

    def index_label(self, label: Label | dict, path: Path | str) -> None:
        d = label.to_doc() if isinstance(label, Label) else label
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?,?)",
                            (d["target"], d["key"], scalar_text(d.get("value")), d["by"], d["ts"], d.get("ref"), str(path)))

    def set_replica(self, art_id: str, location: str, verified_utc: str | None, detail: str | None = None) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO replicas VALUES (?,?,?,?)", (art_id, location, verified_utc, detail))

    def set_presence(self, art_id: str, *, local: bool | None = None, remote: bool | None = None, repo_path: str | None = None) -> None:
        sets, vals = [], []
        for col, v in (("local", local), ("remote", remote), ("repo_path", repo_path)):
            if v is not None:
                sets.append(f"{col}=?")
                vals.append(int(v) if isinstance(v, bool) else v)
        if sets:
            with self.db:
                self.db.execute(f"UPDATE artifacts SET {', '.join(sets)} WHERE id=?", (*vals, art_id))

    def rebuild(self, artifacts: Iterable[tuple[dict, str, bool, bool, str | None]], attempts: Iterable[dict], labels: Iterable[tuple[Label | dict, Path | str]]) -> dict:
        """Wipe every table and reload.  `replicas` is kept (it records verifications, which are not re-derivable locally)."""
        n = {"artifacts": 0, "attempts": 0, "labels": 0}
        with self.db:
            self.db.execute("BEGIN")
            for t in ("artifacts", "artifact_meta", "refs", "attempts", "attempt_params", "attempt_conditions", "attempt_outputs", "attempt_inputs", "labels"):
                self.db.execute(f"DELETE FROM {t}")
        for doc, aid, local, remote, repo_path in artifacts:
            self.index_artifact(doc, aid, local, remote, repo_path)
            n["artifacts"] += 1
        for doc in attempts:
            self.index_attempt(doc)
            n["attempts"] += 1
        for label, path in labels:
            self.index_label(label, path)
            n["labels"] += 1
        return n

    # ---- reads --------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _art_row(r: tuple) -> dict:
        return {"id": r[0], "kind": r[1], "bytes": r[2], "payload_type": r[3], "manifest": json.loads(r[4]), "local": bool(r[5]), "remote": bool(r[6]),
                "repo_path": r[7]}

    def artifact(self, art_id: str) -> dict | None:
        r = self.db.execute("SELECT * FROM artifacts WHERE id=?", (art_id,)).fetchone()
        return self._art_row(r) if r else None

    def artifacts(self) -> list[dict]:
        return [self._art_row(r) for r in self.db.execute("SELECT * FROM artifacts ORDER BY id")]

    def attempt(self, run_id: str) -> dict | None:
        r = self.db.execute("SELECT manifest_json FROM attempts WHERE id=?", (run_id,)).fetchone()
        return json.loads(r[0]) if r else None

    def attempts_by_reuse_key(self, key: str) -> list[dict]:
        return [json.loads(r[0]) for r in self.db.execute("SELECT manifest_json FROM attempts WHERE reuse_key=? ORDER BY start_utc, id", (key,))]

    def attempts_by_derivation(self, drv: str) -> list[dict]:
        return [json.loads(r[0]) for r in self.db.execute("SELECT manifest_json FROM attempts WHERE derivation=? ORDER BY start_utc, id", (drv,))]

    def attempts_by_output(self, art_id: str) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT DISTINCT attempt FROM attempt_outputs WHERE artifact=? ORDER BY attempt", (art_id,))]

    def select_artifacts(self, kind: str | None = None, where: dict[str, str] | None = None, label: tuple[str, str] | None = None,
                         attempt: str | None = None, tool: str | None = None, derivation: str | None = None) -> list[dict]:
        """Artifacts matching every given facet.  `where` keys are dotted meta keys (a leading `meta.` is accepted);
        `label` = (key, value) asserted by anyone; `attempt`/`tool`/`derivation` select outputs of matching attempts."""
        sql, args = "SELECT a.* FROM artifacts a WHERE 1=1", []
        if kind:
            sql += " AND a.kind=?"
            args.append(kind)
        for k, v in (where or {}).items():
            k = k[5:] if k.startswith("meta.") else k
            sql += " AND EXISTS (SELECT 1 FROM artifact_meta m WHERE m.id=a.id AND m.key=? AND m.value=?)"
            args += [k, v]
        if label:
            sql += " AND EXISTS (SELECT 1 FROM labels l WHERE l.target=a.id AND l.key=? AND l.value=?)"
            args += [label[0], scalar_text(label[1])]
        if attempt or tool or derivation:
            sql += (" AND EXISTS (SELECT 1 FROM attempt_outputs o JOIN attempts t ON t.id=o.attempt WHERE o.artifact=a.id"
                    + (" AND o.attempt=?" if attempt else "") + (" AND t.tool=?" if tool else "") + (" AND t.derivation=?" if derivation else "") + ")")
            args += [x for x in (attempt, tool, derivation) if x]
        return [self._art_row(r) for r in self.db.execute(sql + " ORDER BY a.id", args)]

    def labels(self, target: str, key: str | None = None) -> list[dict]:
        sql, args = "SELECT target, key, value, by, ts, ref, path FROM labels WHERE target=?", [target]
        if key:
            sql += " AND key=?"
            args.append(key)
        return [dict(zip(("target", "key", "value", "by", "ts", "ref", "path"), r)) for r in self.db.execute(sql + " ORDER BY ts, path", args)]

    def replicas(self, art_id: str) -> dict[str, dict]:
        return {r[0]: {"verified_utc": r[1], "detail": r[2]} for r in self.db.execute("SELECT location, verified_utc, detail FROM replicas WHERE id=?", (art_id,))}

    def sql(self, query: str, params: tuple = ()) -> list[tuple]:
        """Power-user escape hatch on a read-only connection."""
        ro = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=30)
        try:
            return ro.execute(query, params).fetchall()
        finally:
            ro.close()

    def sql_with_columns(self, query: str, params: tuple = ()) -> tuple[list[str], list[tuple]]:
        ro = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=30)
        try:
            cur = ro.execute(query, params)
            return [d[0] for d in cur.description or []], cur.fetchall()
        finally:
            ro.close()
