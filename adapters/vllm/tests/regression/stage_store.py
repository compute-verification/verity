"""Put the reference records into the evidence store as `fixture/v1` tree artifacts and freeze the snapshot `r19-reference-v1`.

    set -a; source ~/.config/verity/r2.env; set +a           # the control machine's R2 credentials (never copied to a pod)
    python -m tests.regression.stage_store --oracle /path/to/oracle-mirror [--role records|programs] [--rows 23,68] [--no-preserve]

One tree artifact per row dir of SMALL records (`--role records`: the mirror pulled from /vault/tmp/oracle, < 1 GB in total); the
Programs (`--role programs`: `build_request*/instances.json.gz`, 50-300 MB per row, up to 1.6 GB for #39) are a separate tree per row,
put from the machine that holds them (a pod with a minted temporary credential: `research data mint-credential --ttl 6h --env`).
Every put goes through `store_io.put_fixture`; the resulting `art:` ids are written into `fixtures.toml` (`[rows.<row>.artifacts]`)
and the snapshot id into its `snapshot_artifact`, so the resolver can fetch what a machine lacks.  Rows whose Programs are not put keep
`programs_state = "pending_upload"` (their sha256s are already frozen in `files.toml`).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from . import resolver, store_io
from .lift_expected import REFERENCE, SNAPSHOT_NAME


def _set_toml_value(text: str, table: str, key: str, value: str) -> str:
    """Set `key = "value"` inside `[table]` of fixtures.toml (the table header is written by lift_expected; replace or append the line)."""
    header = f"[{table}]"
    i = text.find(header + "\n")
    if i < 0:
        raise KeyError(table)
    j = text.find("\n[", i + len(header))
    block = text[i:j if j >= 0 else len(text)]
    line = f'{key} = "{value}"'
    if re.search(rf"^{re.escape(key)} = ", block, flags=re.M):
        block2 = re.sub(rf"^{re.escape(key)} = .*$", line, block, count=1, flags=re.M)
    else:
        block2 = block.rstrip("\n") + "\n" + line + "\n"
    return text[:i] + block2 + text[j if j >= 0 else len(text):]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--oracle", required=True, help="directory of <row>/... files to put (the records mirror, or a dir of per-row Program trees)")
    ap.add_argument("--role", default="records", choices=("records", "programs"))
    ap.add_argument("--rows", default=None, help="comma-separated row numbers (default: every row present under --oracle)")
    ap.add_argument("--no-preserve", action="store_true", help="local store only (no remote push)")
    ap.add_argument("--snapshot", action="store_true", help="freeze/refresh the snapshot over every artifact id now in fixtures.toml")
    a = ap.parse_args(argv)
    if not store_io.reachable(need_remote=not a.no_preserve):
        print("store not reachable here", file=sys.stderr)
        return 2
    want = {int(x) for x in a.rows.split(",")} if a.rows else None
    text = resolver.FIXTURES.read_text()
    oracle = Path(a.oracle)
    for row in resolver.rows():
        if want is not None and row.number not in want:
            continue
        d = oracle / row.key
        if not d.is_dir():
            continue
        n_files = sum(1 for p in d.rglob("*") if p.is_file())
        meta = {"name": f"r19-reference/{row.key}", "role": a.role, "row": row.number, "row_id": row.key, "class": row.cls,
                "reference": REFERENCE, "files_n": n_files, "index_schema": "verity/r19-oracle-index/v1"}
        art = store_io.put_fixture(d, meta, preserve=not a.no_preserve)
        print(f"[stage] #{row.number:>2} {a.role:<8} {art}  ({n_files} files)")
        text = _set_toml_value(text, f'rows."{row.key}".artifacts', a.role, art)
        if a.role == "programs":
            text = _set_toml_value(text, f'rows."{row.key}"', "programs_state", "uploaded" if not a.no_preserve else "local-store-only")
    resolver.FIXTURES.write_text(text)
    if a.snapshot:
        fx = resolver.load_fixtures()
        members = sorted({v for r in fx["rows"].values() for v in (r.get("artifacts") or {}).values() if v})
        snap = store_io.snapshot(SNAPSHOT_NAME, members, note=f"R19 reference records (v1 = {REFERENCE['commit'][:12]}, {REFERENCE['release']}); {len(members)} trees")
        print(f"[stage] snapshot {SNAPSHOT_NAME} = {snap} over {len(members)} members")
        text = resolver.FIXTURES.read_text()
        if re.search(r"^snapshot_artifact = ", text, flags=re.M):
            text = re.sub(r"^snapshot_artifact = .*$", f'snapshot_artifact = "{snap}"', text, count=1, flags=re.M)
        else:
            text = text.replace(f'snapshot = "{SNAPSHOT_NAME}"\n', f'snapshot = "{SNAPSHOT_NAME}"\nsnapshot_artifact = "{snap}"\n', 1)
        resolver.FIXTURES.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
