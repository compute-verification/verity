"""[R10 owner 7] Independent parser for the guest's binary statement (`veritor/proofs/statement/v2` and `/v3`), written from the
format comment at the top of `zk/sp1/common/src/codec.rs` (read, not imported; nothing from `veritor.protocol.proofs.wire` is used).

Big-endian integers; str/bytes = u32 length + payload; list = u32 count + items; digest = 32 raw bytes.

  Statement   = MAGIC  str gate_set_id  digest gate_set_digest  u32 width  list<KindProgram> kinds  list<Obligation> obligations
  KindProgram = digest kind  u32 size  list<u32> ports  list<GateOp> gates  list<u32> outputs
  GateOp      = str op  list<Arg> args ; Arg = u8 space  u32 value
  Obligation  = digest session  digest compiled  u64 unit  u64 replay_unit  digest kind  list<CommitmentRef>  list<PositionRef|Entry>
                list<u32> inputs  list<u32> outputs
  CommitmentRef = u64 owner_plus_two  digest domain_id  digest root  u64 count
  PositionRef   = u32 commitment  u64 rank  u64 position  str schema  u8 has_expected [bytes expected]
  v3 Entry      = u8 0 PositionRef | u8 1 u32 commitment u64 rank u64 position str descriptor u64 start u64 count u64 stride

Usage: review_statement_parse.py <statement.bin> [--json out.json]   -> prints sha256, gate set, kinds, obligations with commitments/positions
and, for `verity-fa2/chunk/v1` trees, decodes the domain_id as the 8-word header template.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

MAGIC_V2 = b"veritor/proofs/statement/v2\0"
MAGIC_V3 = b"veritor/proofs/statement/v3\0"


class R:
    def __init__(self, b: bytes) -> None:
        self.b, self.i = b, 0

    def take(self, n: int) -> bytes:
        if self.i + n > len(self.b):
            raise ValueError("truncated")
        x = self.b[self.i:self.i + n]; self.i += n
        return x

    def u8(self) -> int: return self.take(1)[0]
    def u32(self) -> int: return struct.unpack(">I", self.take(4))[0]
    def u64(self) -> int: return struct.unpack(">Q", self.take(8))[0]
    def digest(self) -> str: return self.take(32).hex()
    def bytes_(self) -> bytes: return self.take(self.u32())
    def str_(self) -> str: return self.bytes_().decode()


def parse_position(r: R) -> dict:
    c, rank, pos = r.u32(), r.u64(), r.u64()
    schema = r.str_()
    has = r.u8()
    exp = r.bytes_().hex() if has else None
    return {"commitment": c, "rank": rank, "position": pos, "schema": schema, "expected": exp}


def parse_entry(r: R) -> dict:
    tag = r.u8()
    if tag == 0:
        return parse_position(r)
    if tag == 1:
        return {"range": True, "commitment": r.u32(), "rank": r.u64(), "position": r.u64(), "descriptor": r.str_(),
                "start": r.u64(), "count": r.u64(), "stride": r.u64()}
    raise ValueError(f"bad entry tag {tag}")


def parse_statement(data: bytes) -> dict:
    r = R(data)
    if data.startswith(MAGIC_V2):
        r.take(len(MAGIC_V2)); v = 2
    elif data.startswith(MAGIC_V3):
        r.take(len(MAGIC_V3)); v = 3
    else:
        raise ValueError("wrong magic")
    out = {"format": v, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "gate_set_id": r.str_(), "gate_set_digest": r.digest(),
           "width": r.u32()}
    kinds = []
    for _ in range(r.u32()):
        k = {"kind": r.digest(), "size": r.u32()}
        k["ports"] = [r.u32() for _ in range(r.u32())]
        gates = []
        for _ in range(r.u32()):
            op = r.str_()
            args = [(r.u8(), r.u32()) for _ in range(r.u32())]
            gates.append((op, args))
        k["gates_n"] = len(gates)
        k["gate_ops_histogram"] = {op: sum(1 for g in gates if g[0] == op) for op in sorted({g[0] for g in gates})}
        k["outputs"] = [r.u32() for _ in range(r.u32())]
        kinds.append(k)
    out["kinds"] = kinds
    obls = []
    for _ in range(r.u32()):
        o = {"session": r.digest(), "compiled": r.digest(), "unit": r.u64(), "replay_unit": r.u64(), "kind": r.digest()}
        o["commitments"] = [{"owner_plus_two": r.u64(), "domain_id": r.digest(), "root": r.digest(), "count": r.u64()} for _ in range(r.u32())]
        pe = parse_position if v == 2 else parse_entry
        o["positions"] = [pe(r) for _ in range(r.u32())]
        o["inputs"] = [r.u32() for _ in range(r.u32())]
        o["outputs"] = [r.u32() for _ in range(r.u32())]
        obls.append(o)
    out["obligations"] = obls
    if r.i != len(data):
        raise ValueError(f"{len(data) - r.i} trailing bytes")
    # decode fa2 chunk header templates
    for o in obls:
        for c in o["commitments"]:
            w = struct.unpack("<8I", bytes.fromhex(c["domain_id"]))
            if w[0] == 0x68326166:
                c["domain_id_as_fa2_header"] = {"tag": hex(w[0]), "ver_kind_src": hex(w[1]), "launch_tag": w[2], "chunk_index_placeholder": w[3],
                                                "chunk_words": w[4], "HB": w[5], "M_padded_words": w[6], "NB": w[7]}
    return out


def summary(st: dict) -> dict:
    trees = {}
    npos = 0
    for o in st["obligations"]:
        for c in o["commitments"]:
            trees.setdefault((c["root"], c["domain_id"]), c)
        npos += sum((p.get("count", 1) if p.get("range") else 1) for p in o["positions"])
    return {"format": st["format"], "sha256": st["sha256"], "bytes": st["bytes"], "gate_set_id": st["gate_set_id"], "kinds": len(st["kinds"]),
            "obligations": len(st["obligations"]), "positions_expanded": npos, "distinct_trees": len(trees),
            "trees": [dict(c, owner_plus_two=c["owner_plus_two"]) for c in trees.values()][:20],
            "schemas": sorted({(p.get("schema") or p.get("descriptor")).rsplit(":", 1)[0] for o in st["obligations"] for p in o["positions"]})}


def main(argv: list[str]) -> int:
    st = parse_statement(Path(argv[0]).read_bytes())
    s = summary(st)
    print(json.dumps(s, indent=1))
    if "--json" in argv:
        Path(argv[argv.index("--json") + 1]).write_text(json.dumps(st, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
