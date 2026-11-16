"""The stagers ``benchmarks/pouw/h1_prove/drawn.py`` hands `verity.protocols.resource_accounting.compute.pouw.prover.serving.h1.sampled`'s
``stage`` (its `Stager`), built on Flock's Python (`verity_flock`), which no part of verity but this lowering imports
(``tests/import_order.toml``): `stager`, K's flat statement recomposed (``--recompose``), and `FileStager`, a statement on
its committed circuit file, read and never composed."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from verity_flock import circuit as FC
from verity_flock import class_statement as CS

from verity.protocols.resource_accounting.compute.pouw.prover.circuit import h1window as HW
from verity.protocols.resource_accounting.compute.pouw.prover.serving.h1 import sampled as D


def stager(k: int, tile: Sequence[int], m: int, circuit: Path, bind: dict) -> CS.UnitStager:
    """K's flat statement's `UnitStager` (the form the Lean verifier accepts) on `h1window.statement_rows`, its circuit
    built once and written to `circuit`, bound to `bind` (``{"partition", "program"}``), as `pouw_h1_circuits` builds K's
    file."""
    t = D.template(k, tile, m)
    return CS.UnitStager(t, D.NAME, HW.statement_rows(t), HW.STATEMENT_OUT_ROWS, circuit=circuit, bind=bind)


class FileStager:
    """`UnitStager`'s ``stage`` on the committed circuit file `path` of the statement of Definition id `definition` over
    the rows `in_rows` and `out_rows` (`UnitStager`'s), read and never composed (the K 2048 J10-R composition takes some
    hundreds of GB): its META (its second line), its pin and class digest (`circuit.digest`, `circuit.class_digest`)
    streamed from the file, and each instance's outputs the word reference's, which ``outs`` gives. `circuit.write` reads
    only these of a circuit; the files it writes are the recomposed stager's byte for byte. J10-R's statement's rows are
    `h1window.statement_rows` of drawn's `template`; a lanes-form leaf's are drawn's `leaf_rows` over its file's template (the
    manifest's ``templates``)."""

    def __init__(self, path: Path, definition: str, in_rows: Sequence[tuple], out_rows: Sequence[tuple],
                 name: str = D.NAME) -> None:
        with open(path, "rb") as f:
            first, second = f.readline(), f.readline()
            meta = json.loads(second[5:])
            head = {k: v for k, v in meta.items() if k not in FC.BINDING_KEYS}
            h_pin = hashlib.sha512(first + second)
            h_cls = hashlib.sha512(first + b"META " + "".join(FC._COMPACT.iterencode(head)).encode() + b"\n")
            while b := f.read(1 << 26):
                h_pin.update(b)
                h_cls.update(b)
        self.meta, self.pin, self.class_sha512 = meta, h_pin.hexdigest(), h_cls.hexdigest()
        self.ports = FC._meta_ports(meta)
        self.row_ports = [r for r, *_ in in_rows]
        def port(nm: str) -> SimpleNamespace:
            return SimpleNamespace(name=nm, dtype="u16")
        content = json.dumps([definition, [list(r[:3]) + [list(r[3])] for r in in_rows],
                              [[s, list(ps)] for s, ps in out_rows]]).encode()
        outs = sorted({p for _s, ps in out_rows for p in ps})
        self.sub = SimpleNamespace(id=name, template=SimpleNamespace(id="unit-class"), inputs=[port(r) for r in self.row_ports],
                                   outputs=[port(f"o{j}") for j in outs])
        self.iset = SimpleNamespace(subcircuit=self.sub, name=name.replace("/", "-"),
                                    content_digest=hashlib.sha256(content).hexdigest())
        self.outs: dict[str, np.ndarray] | None = None

    def stage(self, rows: dict, out_dir: Path, salt_key: bytes, salts: dict, indices: Sequence[int], set_name: str,
              tables: dict, refs: dict, registered: dict) -> dict:
        iset = SimpleNamespace(**{**vars(self.iset), "name": set_name})
        n = len(rows[self.row_ports[0]])
        for r in self.row_ports:
            if not np.array_equal(np.asarray(tables[r], np.uint16)[np.asarray(refs[r])], np.asarray(rows[r], np.uint16)):
                raise ValueError(f"port {r}'s table rows at the instances' refs are not their rows")
        table_rows = {p.name: len(tables[p.name]) if p.name in tables else n for p in self.ports}
        salts = {**FC.stage_salts(iset, 0, n, self.ports, salt_key, table_rows), **salts}
        out_dir.mkdir(parents=True, exist_ok=True)
        FC.write("", iset, 0, n, out_dir / "inst.bin", out_dir / "pub.bin", salt_key,
                 rows={r: np.asarray(rows[r], np.uint16) for r in self.row_ports}, outs=self.outs, salts=salts,
                 public_words={}, digests=(self.pin, self.class_sha512), indices=[int(u) for u in indices],
                 share_rows=True, tables=tables, refs=refs, registered=registered, meta=self.meta)
        return {"meta": self.meta, "outs": self.outs, "salts": salts, "circuit_sha512": self.pin}
