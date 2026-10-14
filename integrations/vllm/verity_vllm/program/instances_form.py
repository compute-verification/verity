"""[R17 mfast F2, COORD R17-34 (2) / M-0496, rev M-0500] the STORED form of `instances.jsonl` operand references.

`instances.jsonl` (one line per Compute instance) lists every operand's references.  The record form (`verity-gen/instances/v1`,
`operand_refs_form: runs`) writes every maximal run `["space", idx, lo, hi]` -- the node-space K/V gathers of a long prefill run by run:
54 M items / 1.63 GB on a B8 1024/128 row, of which every reader keeps the ~564 k parameter-space ("p") runs and discards the rest after
parsing it.  The compact form (`verity-gen/instances/v2`, `operand_refs_form: progressions`) writes node- AND parameter-space references
as progressions `["N"|"P", idx, lo, hi, dI, dL, dH, count]` (cb_a.compact's form, count >= 2; a single run stays `["n"|"p", idx, lo, hi]`).
"P" since the GLM-5.3 scattered weight reads: a transposed-weight operand is ~1e6 parameter-space runs, run by run under the first v2
form.  Lossless: `compact.runs_from_file_items` returns the record's run list, item for item.

Every reader accepts both forms and expands on read where it needs runs (compare.load_oracle keeps the "p" runs of either form;
tp.rank_match and bench.gen_batch_candidate expand); nothing caches an expansion by digest.  The writer is `run_config.persist_program`
(`VERITY_INSTANCES_FORM=runs|progressions`, default `runs` until coord rules the v2 schema the file of record).  The fold digest of
record (`program.json` digest == `fold_summary.digest`) is over the Program descriptor, not over `instances.jsonl` bytes: the same fold
has the same digest under both forms; only F-r16-24's tie fact `instances_sha256` (the bytes an oracle READ) differs, as it must.

[rev F-r17-12] this module REPLACES unit 7's compact-provenance sidecars (program.compact.json + instances.compact.jsonl written into the
fold dir by the record's producer and read by the fast loader after checking the SOURCES' digests): nothing the checker owned
authenticated the sidecar's content, and a forged sidecar with the true source hashes was accepted (M-0497).  Withdrawn; no reader of a
sidecar exists.  The compact form here is the file of record itself.

    verity-vllm instances-form convert --form progressions|runs SRC DST
    verity-vllm instances-form form FILE...
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

from verity_vllm.config import option

INSTANCES_FORMS = ("runs", "progressions")
SCHEMA_V1 = "verity-gen/instances/v1"
SCHEMA_V2 = "verity-gen/instances/v2"


def sha256_file(path: str) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def convert_instances(src: str, dst: str, form: str) -> dict:
    """Rewrite `src` (an instances.jsonl in either form) as `dst` in `form`.  Header: schema v1 + operand_refs_form "runs" | v2 +
    "progressions"; rows: every operand's items through compact.file_items_from_runs (-> progressions) or compact.runs_from_file_items
    (-> runs).  Lossless both ways (the greedy progression rule is a function of the run list); everything else on the row is copied
    verbatim.  Returns counts (rows, items in / out, bytes)."""
    from .compact import file_items_from_runs, runs_from_file_items
    if form not in INSTANCES_FORMS:
        raise ValueError(f"form {form!r}: expected one of {INSTANCES_FORMS}")
    n_rows = n_in = n_out = 0
    tmp = dst + ".tmp"
    with open(src) as f, open(tmp, "w") as g:
        for line in f:
            t = line.strip()
            if not t:
                continue
            o = json.loads(t)
            if "node" not in o:
                if str(o.get("schema", "")).startswith("verity-gen/instances/"):
                    o["schema"] = SCHEMA_V2 if form == "progressions" else SCHEMA_V1
                o["operand_refs_form"] = form
                g.write(json.dumps(o) + "\n")
                continue
            refs = o.get("operand_refs")
            if refs is not None:
                out = []
                for a in refs:
                    n_in += len(a)
                    b = file_items_from_runs([tuple(r) for r in runs_from_file_items(a)]) if form == "progressions" \
                        else [list(r) for r in runs_from_file_items(a)]
                    n_out += len(b)
                    out.append(b)
                o["operand_refs"] = out
            g.write(json.dumps(o, separators=(",", ":")) + "\n")
            n_rows += 1
    os.replace(tmp, dst)
    return {"form": form, "rows": n_rows, "items_in": n_in, "items_out": n_out, "bytes_in": os.path.getsize(src), "bytes_out": os.path.getsize(dst)}


def instances_form_of(path: str) -> str:
    """The stored form the file's header declares (a pre-F2 file declares none = runs)."""
    with open(path) as f:
        for line in f:
            t = line.strip()
            if t:
                o = json.loads(t)
                return o.get("operand_refs_form") or "runs"
    return "runs"


@dataclass(frozen=True, kw_only=True)
class ConvertOptions:
    "rewrite SRC instances.jsonl as DST in --form runs|progressions (lossless)"

    form: str = option("--form", choices=INSTANCES_FORMS)
    src: str = option("src")
    dst: str = option("dst")
    cmd: str = field(default="convert", init=False)


@dataclass(frozen=True, kw_only=True)
class FormOptions:
    "print the stored form each FILE's header declares"

    file: list[str] = option("file", nargs="+")
    cmd: str = field(default="form", init=False)


COMMANDS = {"convert": ConvertOptions, "form": FormOptions}


def main(a: ConvertOptions | FormOptions) -> int:
    import time
    if a.cmd == "convert":
        t0 = time.time()
        res = convert_instances(a.src, a.dst, a.form)
        res["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(res))
        return 0
    for p in a.file:
        print(json.dumps({"file": p, "operand_refs_form": instances_form_of(p), "bytes": os.path.getsize(p)}))
    return 0
