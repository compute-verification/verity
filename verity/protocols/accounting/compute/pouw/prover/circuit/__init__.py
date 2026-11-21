"""`ncp-v2`: NCP-INT's noised GEMM as a `verity.circuits` Program, verified by SPOT.

PoUW's word primitives are `verity.circuits.ops.words` (every output at most 32 bits but `pc8`'s exact squares, which stay
inside a row unit), and their Boolean bodies `verity.circuits.ops.boolean.words`.

* `hashes`: SHA-512, SHAKE256 and TurboSHAKE128 in gates (`Sha512Compress`, `KeccakF1600`, `KeccakP1600R12`).
* `leaves`: the tile checks' committed row schemas (Flock's ports), Pearl-C's tile digest in gates, and the rows' native
  reference.
* `ncp2`: the unit templates; `weight(shape, kind)`, a weight's Y units once per run; `linear(shape, kind)`, one call's
  Program, reading them; `window(shape, calls, kind)`, calls on one weight in one root.
* `reference`: the native reference those must evaluate to.
* `anchors`: the anchored inputs' anchors, which the verifier builds (the tile checks' derived noise too, `TileAnchors`, with
  each unit's public input regions), and the execution phase's refusal of a repeated call index.
* `tile_units`: what the tile checks share: their emitter, the ok word, the digest families, the row cut's shape
  (`RowK`), the unit roles and the cut of a root into units (`units`, `unit_cut`).
* `pc8`: Pearl-C's (FP8, forming v1) tile check as a Program, cut into row and tile units, and its hiding Program
  (``Shape.hidden``), whose tiles return only their tile digest; the rows are Flock-committed ports and the anchored
  inputs public input regions (`PORTS`).  Its scheme has no entry in `SCHEMES` yet.  Pearl-C4's (`pc4`) is
  `verity.protocols.experimental.pearl_c4.prover`'s.
* `rowk`: the tile checks' row units cut along k into a head, one form per segment and a tail (``Shape.row_k``, opt-in), so each
  piece fits Flock's block.
* `compile`: PoUW as one pass over a served Program: `compile(program, scheme,
  groups)` replaces every registered matmul's dots (`MATMULS`) with the scheme's r × c tile units, each deriving its seed
  and noise in its own proof from the one coin, the salt, and checks the result; `rebuild` is the
  verifier's side. Its schemes are its own `SCHEMES`, not this module's.
* `seeds`: the seed and the noise blocks a unit derives (`PouwSeed`, `PouwNoiseBlock`, on words and on bits), and its two
  seeds where its receipt folds take keys (`PouwUnitSeeds`: seed_j and σ_R).
* `h1tile`: `eps1-sm120-h1fx1` as a compile-pass scheme, H1 + fx1 on a warp's tile under CP-W.
* `h1device`: H1's device records, one an architecture (sm_120, and sm_90 for the H200): the s8 step and its binding, and the
  contractions its kernels serve.
* `h1zy`: the H1 tile with its served z/y record checked (`PouwH1TileZy`): the record an input, committed
  before its round's coin, its z the tile's A Bᵀ and its y the served dequant (`PouwZyDequant`, the activation scale
  first, no bias).
* `h1window`: an H1 execution phase as one public Program of those tiles in fixed slots (`PouwH1Window`), the tile R × C a
  parameter and one tile template per distinct K (`PouwH1WindowTile`, its A and B rows ``a_k<K>`` and ``b_k<K>``),
  each linear's B rows registered once for all its calls; its template-instance partition derives its units, and
  `unit_reads` the rows each unit reads: the Program a served path registers its trees under, a mixed-K execution phase in one
  read.
* `live`: a round's served rows under rule A (`PouwLiveSlots_v2`), one unit a round: ok iff exactly its slots [0, m) name
  requests of the gate-signed request table, none more often than it can take in round s (its prompt at its prefill, one
  row a decode), none past its public max_tokens, for the public m that dense credit counts.
* `window_statement`: an execution phase's statement on its committed entry rows (`PouwWindowStatement`,
  `window.window_problem` in gates), one unit an execution phase: ok iff the P rows are entry rows of the layout, the
  calls, one filler, then empty entries, whose ranges tile [0, N) at the public registered level, every call's -h3 root
  the pass's public one and the calls' pass tiles in order.
* `pearl_c_call`, `pearl_c_forming`, `pearl_c_noise`, `pearl_c_tile`: Pearl-C -h3's rows and cells, its forming, its noise
  derivation and its tile's checked words as registered Definitions, each with its Boolean version, which
  `pearl_plus_tile`, Pearl+'s tile check, calls.
"""
from __future__ import annotations

from types import MappingProxyType

from verity.protocols.accounting.compute.pouw.verifier.kernel_schemes import of_family

#: the schemes that have a circuit here, each with the permutation P it pins: the kernel schemes' ``ncp-v2`` family
SCHEMES = MappingProxyType({k.name: k.parameters["perm"] for k in of_family("ncp-v2")})

__all__ = ["SCHEMES"]
