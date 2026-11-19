"""The credited block's inputs from a Pearl+ pass: `credited_inputs` fills every port of Pearl+'s tile unit
(`circuit.pearl_plus_tile`: `PearlPlusTileCoreV2`'s, then the leaf's) for one tile from the pass's directory, each value
with the opening that ties it to the pass's commitments.

- **a**, the tile's TM rows of A as FP32 words, each opened at the call's ``root_a`` (`opened.Opened.open`); rows past the
  call's ``real_rows`` are its zero filler.
- **b**, the weight's rows for the tile's columns, opened at the weight's root.
- **y**, the call's committed y on the tile's real rows only (`opened.Opened.open_y`): filler rows aren't served.
- the public rows: the salt, B's root, the call's index, the tile's rows and its column tile; E_B's lines, F_A, F_B and G
  recomputed from the scheme's seed (`pearl_c.PearlC.noise`'s ``seed_b``), as the verifier recomputes them.
- **cert**, the certificates of each (real row, column) pair, by replaying the tile through the scheme
  (`inert_bound.certificates` over the encoded rows' codes).
- the transcript leaf: its hm96 salt and path, the transcript root, its domain and the tile's position.

The values are the prover's, read from its own files; nothing here checks them beyond what opening them needs, so a pass
whose files are not what it committed fills ports the unit refuses (``ok`` 0, or a digest the leaf doesn't open)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from verity.circuits.defs import SpecializedDefinition, bind
from verity.commitments import Opening
from verity.protocols.resource_accounting.compute.pouw.prover import schemes as pouw_schemes
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import inert_bound, pc8
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_tile as T
from verity.protocols.resource_accounting.compute.pouw.prover.clear.clear_audit import TileOpening
from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c as C
from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c.manifest import manifest_epoch
from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c.opened import Files, Opened
from verity.protocols.resource_accounting.compute.pouw.verifier.audit import (
    HM96_SALT,
    TileRef,
    opened_value,
    tile_refs,
    transcript_domain,
)
from verity.protocols.resource_accounting.compute.pouw.verifier.coins import CoinCommitment
from verity.protocols.resource_accounting.compute.pouw.verifier.window import LIVE_SCHEME
from verity.protocols.proofs.spot.verifier.challenger.challenger import LiveCommitment


def _words(value: bytes) -> list[int]:
    return [int.from_bytes(value[4 * t:4 * t + 4], "little") for t in range(len(value) // 4)]


@dataclass(frozen=True)
class Credited:
    """One tile's ports of Pearl+'s tile unit, in `PearlPlusTileCoreV2`'s order then the leaf's, with the openings that tie
    a, b, y and the digest's leaf to the pass's roots. ``real`` is the tile's real rows (TR), the rows y holds."""

    ref: TileRef
    scheme: C.PearlC
    k: int
    n: int
    real: int
    a: list[list[int]]
    b: list[list[int]]
    y: list[list[int]]
    cert: list
    salt: bytes
    root: bytes
    unit: int
    rows: list[int]
    col: int
    eb: list
    fa: list
    fb: list
    g: list
    hsalt: bytes
    path: list[bytes]
    troot: bytes
    tdomain: bytes
    tpos: int
    opened: TileOpening
    y_opened: tuple[Opening, ...]
    t_opened: Opening

    @property
    def tm(self) -> int:
        return len(self.rows)

    @property
    def tn(self) -> int:
        return len(self.b)

    def statics(self) -> dict:
        """The unit's statics at this tile, TR its real rows."""
        d = self.scheme.device
        return dict(K=self.k, TM=self.tm, TR=self.real, TN=self.tn, N=self.n, RHO=inert_bound._rho(d.name, None),
                    RHOR=inert_bound.row_rho(self.scheme, self.k), DEV=d.name)

    def core(self) -> SpecializedDefinition:
        return bind(T.PearlPlusTileCoreV2, **self.statics())

    def core_args(self) -> list:
        return [self.a, self.b, self.y, self.cert, self.salt, self.root, self.unit, self.rows, self.col, self.eb, self.fa,
                self.fb, self.g]

    def reference(self) -> tuple[bytes | None, int]:
        """`pearl_plus_tile.reference` on these ports: (digest, ok)."""
        s = self.statics()
        return T.reference(s["DEV"], self.k, self.n, self.a, self.b, self.y, self.cert, self.salt, self.root, self.unit,
                           self.rows, list(self.ref.tile.cols), self.col, s["RHO"], s["RHOR"])

    def leaf_reference(self, digest: bytes) -> int:
        """`pearl_plus_tile.leaf_reference`: 1 iff ``digest``'s leaf under hsalt opens at tpos to the transcript root."""
        return T.leaf_reference(digest, self.hsalt, self.troot, self.tdomain, self.tpos, self.path)


def _coins(man: dict) -> CoinCommitment | LiveCommitment:
    live = isinstance(man.get("coins"), dict) and man["coins"].get("scheme") == LIVE_SCHEME
    return (LiveCommitment if live else CoinCommitment).from_json(man["coins"])


def credited_inputs(pass_dir: str | Path, tile_id: int) -> Credited:
    """Tile ``tile_id``'s ports of Pearl+'s tile unit from the pass in ``pass_dir`` (its ``manifest.json`` and the files it
    names), under the pass's own coin commitment: the prover fills its ports, and the verifier's statement holds the public
    ones to its own. The tile's real rows are its call's ``real_rows`` in the tile (y is opened on those alone, and a's
    later rows are the call's filler). ValueError for a tile the pass doesn't have, a scheme other than Pearl+'s
    (forming v1 under -h3), a tile that doesn't encode, or one with no real row."""
    base = Path(pass_dir)
    man = json.loads((base / "manifest.json").read_text())
    scheme = pouw_schemes.get(man["scheme"], **{k: tuple(v) if isinstance(v, list) else v
                                                for k, v in man.get("instance", {}).items()})
    if scheme.forming != "v1" or scheme.hashing != "h3":
        raise ValueError(f"{scheme.name}: Pearl+'s tile unit is forming v1 under -h3")
    epoch, _how = manifest_epoch(man, _coins(man))
    refs = tile_refs(scheme, epoch.workload)
    if not 0 <= tile_id < len(refs):
        raise ValueError(f"the pass has {len(refs)} tiles, and no tile {tile_id}")
    ref, hashing = refs[tile_id], scheme.commitment_hash
    files = Files(base)
    prover = Opened(man, files, refs)
    opened, y_opened = prover.open(epoch, tile_id), prover.open_y(epoch, tile_id)
    u, rec = epoch.workload.matmuls[ref.matmul], man["matmuls"][ref.matmul]
    k, n, tile = u.shape.k, u.shape.n, ref.tile
    tm, tn = len(tile.rows), len(tile.cols)
    a_rows = [opened_value(hashing, o.value) for o in opened.a]
    b_rows = [opened_value(hashing, o.value) for o in opened.b]
    root_a, root_b = bytes.fromhex(rec["root_a"]), epoch.weight_roots[u.weight]
    ctx = C.Context(epoch.salt, u, root_a, root_b)
    noise = scheme.noise(ctx)
    ops = scheme.encode(ctx, noise, a_rows, b_rows, tile)
    if ops is None:
        raise ValueError(f"tile {tile_id} doesn't encode: its operands are outside the scheme's domain")
    real = sum(i < rec["real_rows"] for i in tile.rows)
    if real == 0:
        raise ValueError(f"tile {tile_id} is filler throughout: it credits nothing, and the unit takes 1 to TM real rows")
    shape = pc8.Shape(tm, k, tn, tile=(tm, tn), dev=scheme.device.name)
    fa, fb, g = (pc8.basis_fa(shape, noise.seed_b), pc8.basis_fb(shape, noise.seed_b), pc8.gram_g(shape, noise.seed_b))
    _d, ng = inert_bound._promoted(scheme.device.name, k)
    cert = [[[c[2 * z:2 * z + 2] for z in range(ng - 1)]
             for c in (list(inert_bound.certificates(ops.a.codes[i], ops.b.codes[j], scheme.device.name))
                       for j in range(tn))]
            for i in range(tm)]
    c, t_opened = man["commitment"], opened.leaf
    return Credited(ref=ref, scheme=scheme, k=k, n=n, real=real,
                    a=[_words(r) for r in a_rows], b=[_words(r) for r in b_rows],
                    y=[_words(opened_value(hashing, o.value)) for o in y_opened], cert=cert,
                    salt=epoch.salt, root=root_b, unit=u.index, rows=list(tile.rows), col=tile.cols[0] // tn,
                    eb=[list(ops.e_b[j]) for j in range(tn)], fa=[fa[t * 32:t * 32 + 32] for t in range(k)],
                    fb=[fb[t * 32:t * 32 + 32] for t in range(k)], g=[g[p * 32:p * 32 + 32] for p in range(32)],
                    hsalt=t_opened.value[-HM96_SALT:], path=list(t_opened.path), troot=bytes.fromhex(c["root"]),
                    tdomain=transcript_domain(epoch, c["count"], hashing).domain_id, tpos=tile_id,
                    opened=opened, y_opened=y_opened, t_opened=t_opened)


__all__ = ["Credited", "credited_inputs"]
