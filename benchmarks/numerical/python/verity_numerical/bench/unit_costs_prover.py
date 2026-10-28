"""Front F6's prover column: the inner Flock prover's wall time and peak memory per session at each bucket's nonzero
bound, fitted from recorded prover runs, and where a session stops fitting on one node.

    python -m verity_numerical.bench.unit_costs_prover --inventory unit-costs.json --out DIR

**Source runs** (:data:`SOURCE`): every ``flock-circuit prove --zk`` record of ``benchmarks/pouw/hidden_zk`` on the Nebius
nodes (CPU prover, Rust, upstream Flock b684b12 with Flock's patches; Intel Xeon 6776P, 192 CPUs, 1,716 GiB): the session's
instances, its block ``2^k_log`` bits per instance and ``D = 2^dense_m`` bits in all, the circuit text's bytes (one unit
netlist shared by the instances, plus the SHA-512 and hm96 glue), the prove's median time, the prover process's wall time
(circuit load included) and peak RSS (GiB), and the prover's threads.

**The law** (:func:`fit`): each of RSS, prove time and process wall time is affine in (D / 2^30, circuit bytes / GiB), by
least squares: RSS over every record, unweighted; the times over the 32-thread records, weighted by 1 / y² (relative
residuals). The fitted coefficients and every residual are in the output.

**A session** (:func:`session`) of B slots at bound Z, each slot one unit of at most Z nonzeros with ρ nonzeros per row:
rows = Z / ρ; ``k_log`` = max(22, ceil(log2(rows + 1)) + 2), the slot rounded up to a power of two and at most doubled again
by the SHA-512, hm96 and lookup ranges (every recorded block is at least 2^22); D = B 2^k_log; circuit bytes =
:data:`GLUE_BYTES` + the netlists of the session's distinct unit types (at most B; a session's slots may hold different
types), each at :data:`BYTES_PER_NNZ` per nonzero of what the circuit text holds (:func:`netlists`): a registered type's one
unit (GEMM's tc-step: 263,933 nonzeros, shared by its K/16 units), a token-select tree's distinct members, a generic type's
whole subcircuit; never more than Z nonzeros. ρ is the least nonzeros per committed row among the bucket's types (the
inventory's ``nnz / committed``). A block is at most 2^:data:`K_LOG_MAX` bits (``Stmt.InRange``), so a session needing more
is out of range: flock-circuit refuses it.

**Per workload** (:func:`audit`): the padding table's body in its dp4 buckets at B = 16 and its tail at B = 1
(``unit_costs_padding``); each shape's sessions (mean and the public 1 − 2⁻⁴⁰ count), its session's RSS and prove time, and
the prover time per audit, Σ sessions x session time (with the slot sumcheck below), on one prover. **The cap** is the
largest power of two Z at which a B = 16 session at the workload's least ρ is in range and fits one node's memory
(:data:`NODE_GIB`; :data:`GPU_GIB` for one GPU's, the CPU law taken as a bound on the CUDA prover's). A unit over the cap
is split below the subcircuit into ceil(z / cap) pieces, its pieces in a bucket of their own at the cap (sessions
ceil(p X / B) for X of its units drawn); every other unit keeps its dp4 bucket.

**dp4t** (:func:`menu_t`): dp4 places on nonzeros alone, so a rare unregistered type in a bucket of registered ones pads
every session's slots to its template. dp4t keys the body's buckets by (template class, Z): registered types by their
registered unit's template (every GEMM tc step one class, at every K), generic types by their whole subcircuit's, each
class within a factor of two; at most :data:`BUCKETS_T` buckets, the dp on Z within each class. A class drawn fewer than
:data:`RARE_DRAWS` times per audit is rare: each of its types a bucket of its own, at B = 1 or 16, with the public count at
its 1 − 2⁻⁴⁰ quantile, like the tail. The menu stays fixed per model. The caps are dp4's (the cap search's session holds
the workload's largest templates). The what-if registers every generic body type drawn at least :data:`RARE_DRAWS` times
as a tc-step-like unit.

**Staging** (:func:`staging`), once per template shape, apart from proving: building the circuit text and the instances
(``verity_flock.circuit``). Affine in the ANDs lowered, wall time and peak RSS each fitted (weighted by 1 / y²) on circuits'
staging ladder (:data:`STAGING`); a registered type lowers one unit, at least the measured registered staging
(:data:`REGISTERED_STAGING`), a generic type its whole subcircuit, a tree its distinct members.

**The slot sumcheck** (:data:`SLOT_SUMCHECK`; candidate 3): V* runs
the slot sumcheck's verifier and opens the committed bits, so each session's prover adds, apart from Flock's prove, K1 ns
per entry per rep x entries x reps seconds. A slot's entries are its type's *template*, the registered table: the
nonzeros of what the circuit text holds for the type (:func:`netlists`: a registered type's one unit, a tree's distinct
members, a generic type's whole subcircuit), at most Z. A session's entries are slots x its largest template's (the 16
slots run as one sumcheck and the type is hidden, so every slot is padded to the session's largest; F4's table pads a
template to n = 2^ν slots, whose padding pairs the kernel skips; proofs' ``vstar_c3.py`` takes the same shape). K1 is
measured per ν: a template of 2^ν entries takes the measurement at the least measured ν' ≥ ν (the last beyond). Its memory
is the committed bits of the bucket's stacked table (its T distinct templates, T padded to a power of two, each at the
largest's size; bytes per entry) plus the sumcheck's working set (bytes per entry x the session's entries). A session's
time is Flock's prove plus the sumcheck, its peak memory Flock's peak RSS plus the sumcheck's, and the cap and the prover
hours use those.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from . import unit_costs_padding as P

#: vy-nebius-1 and vy-nebius-2: ``free -g`` total (GiB)
NODE_GIB = 1716
#: one NVIDIA RTX PRO 6000 Blackwell Server Edition: 97,887 MiB
GPU_GIB = 97_887 / 1024
SLOTS = 16
#: flock-ir-unit/v2 netlist text per nonzero: the largest of four registered units' (``ir_lower.netlist``: GEMM tc-step 4.93,
#: RoPE pair 4.73, SiLU element 4.33, RMSNorm-Triton N=128 6.04)
BYTES_PER_NNZ = 6.04
#: the circuit text that is not a unit's netlist (SHA-512 compressions, hm96 rows, META): the smallest recorded circuit's,
#: Pc4RowATail_v1{K=256,KS=64} of 5,280 ANDs
GLUE_BYTES = 55_911_585
K_MIN_LOG = 22
#: ``Stmt.InRange``'s block bound (``live/src/circuit.rs``'s ``IN_RANGE_K_LOG``; ``verity_flock.circuit.K_MAX``)
K_LOG_MAX = 27
POW2 = tuple(range(16, 33))
CAP_SEARCH = tuple(range(16, 41))
#: ρ for the power-of-two table: RoPE's and RMSNorm's registered layouts (GEMM's is 27.9, attention's 27.9-29.6)
RHO_TABLE = 26.5
MENU_K = 4
#: ``dp4t``: at most this many body buckets, keyed by (template class, Z); a class drawn fewer than :data:`RARE_DRAWS` times
#: per audit (mean) is rare, each of its types a bucket of its own at B = 1 or 16 (whichever costs less at the public count)
BUCKETS_T = 8
RARE_DRAWS = 16
#: candidate 3's slot sumcheck, measured by proofs' K1 on vy-nebius-2's 32 threads (art:10599a37d2ec; upper bounds,
#: Lean builds loaded the node): ns per entry per rep at ν = 20 and 28, the middle of wide-Morton (the band's low end) and
#: uniform columns (its high end, the worst case); reps; the committed bits' bytes per entry (8; 64 encoded at rate 1/8);
#: and the working set per entry (peak RSS 37 GiB at ν = 28). F4's estimate before it: 13 ns (10-16) at ν = 20.
SLOT_SUMCHECK: dict[str, Any] = {
    "k1_ns": ((20, 15.2), (28, 26.1)), "band": (((20, 12.2), (28, 19.3)), ((20, 18.2), (28, 32.9))), "reps": 2,
    "bits_bytes": 8.0, "work_bytes": 37 * 2 ** 30 / 2 ** 28,
    "source": "measured, proofs' K1 r20261010-093619-c3ee (art:10599a37d2ec), upper bounds"}

#: (run, definition, instances, k_log, dense_m, circuit bytes, prove s (median), prover process wall s, prover peak RSS GiB,
#: threads): ``benchmarks/pouw/hidden_zk`` summaries on vy-nebius-1/2
HIDDEN_ZK: tuple[tuple[str, str, int, int, int, int, float, float, float, int], ...] = (
    ('r20261003-060202-b967', 'Pc4Tile_v1{K=1024,TM=1,TN=1}', 16, 22, 26, 100136943, 1.0987, 6.53, 1.39, 32),
    ('r20261003-060202-b967', 'Pc4Tile_v1{K=256,TM=1,TN=1}', 16, 22, 25, 70617623, 0.938, 5.7, 1.04, 32),
    ('r20261003-060227-1efd', 'Pc4RowA_v1{K=1024}', 16, 25, 29, 5760857510, 7.5869, 116.56, 20.16, 32),
    ('r20261003-060227-1efd', 'Pc4RowA_v1{K=256}', 16, 23, 27, 1384534519, 1.4038, 26.79, 5.52, 32),
    ('r20261003-060232-34df', 'Pc4RowB_v1{K=1024}', 16, 25, 29, 6581669534, 8.0521, 134.38, 21.78, 32),
    ('r20261003-060232-34df', 'Pc4RowB_v1{K=256}', 16, 24, 28, 2115332939, 6.0679, 58.93, 8.96, 32),
    ('r20261003-064840-b83a', 'Pc4RowA_v1{K=4096}', 16, 27, 31, 26151124996, 32.3803, 574.82, 87.05, 32),
    ('r20261003-064856-dc42', 'Pc4RowB_v1{K=4096}', 16, 27, 31, 26971971824, 31.7834, 600.02, 84.18, 32),
    ('r20261003-064901-8d6e', 'Pc4TileHidden_v1{K=1024,TM=1,TN=1}', 16, 22, 26, 108727949, 1.3135, 7.81, 1.42, 32),
    ('r20261003-064901-8d6e', 'Pc4TileHidden_v1{K=256,TM=1,TN=1}', 16, 22, 25, 78692059, 1.1261, 6.95, 1.07, 32),
    ('r20261003-064901-8d6e', 'Pc4TileHidden_v1{K=4096,TM=1,TN=1}', 16, 23, 27, 237376407, 2.1184, 13.49, 2.93, 32),
    ('r20261003-064901-8d6e', 'Pc4Tile_v1{K=4096,TM=1,TN=1}', 16, 23, 27, 228784891, 2.3568, 14.43, 2.9, 32),
    ('r20261003-071851-a2f3', 'Pc4Tile_v1{K=4096,TM=1,TN=1}', 256, 23, 31, 228784891, 11.2523, 42.06, 11.07, 32),
    ('r20261003-073948-0e7d', 'Pc4Tile_v1{K=4096,TM=1,TN=1}', 4096, 23, 35, 228784891, 283.247, 949.49, 129.92, 32),
    ('r20261003-102619-cab2', 'Pc4Tile_v1{K=4096,TM=4,TN=4}', 256, 25, 33, 3192819893, 30.0277, 150.22, 46.11, 32),
    ('r20261003-105542-b162', 'Pc4Tile_v1{K=4096,TM=8,TN=8}', 64, 27, 33, 13762384409, 44.209, 372.6, 78.18, 32),
    ('r20261003-215938-bdf1', "Pc8TileCap_v1{K=8192,TM=1,TN=1,N=1,RHO=1000,DEV='sm120'}", 8, 25, 28, 3905011930, 5.2177, 79.61, 15.38, 32),
    ('r20261003-215938-bdf1', "Pc8CapWord_v1{K=8192,DEV='sm120'}", 8, 25, 28, 3682754201, 5.3155, 81.85, 15.03, 32),
    ('r20261003-215938-bdf1', "Pc8Tile_v1{K=8192,TM=1,TN=1,DEV='sm120'}", 8, 25, 28, 934944359, 11.0517, 76.97, 8.75, 32),
    ('r20261003-215938-bdf1', "Pc8Tile_v1{K=8192,TM=2,TN=2,DEV='sm120'}", 8, 26, 29, 3718279248, 10.2693, 104.97, 21.62, 32),
    ('r20261004-003758-fc56', 'Pc4RowA_v1{K=4096}', 16, 27, 31, 23357804541, 26.9612, 459.41, 80.74, 32),
    ('r20261004-012846-3deb', 'Pc4RowB_v1{K=4096}', 16, 27, 31, 24178651369, 28.6005, 585.85, 79.28, 32),
    ('r20261004-032911-9751', 'Pc8CapRowDigest_v1{TN=64}', 8, 26, 29, 73409170, 14.7667, 73.66, 13.02, 32),
    ('r20261004-032911-9751', 'Pc8CapRowTally_v1{K=8192,TN=64}', 8, 26, 29, 174229833, 13.9842, 71.5, 13.27, 32),
    ('r20261004-032911-9751', "Pc8CapStrip_v1{K=8192,TM=64,TN=64,N=16384,RHO=1000,DEV='sm120'}", 8, 26, 29, 65255064, 16.5827, 81.29, 13.0, 32),
    ('r20261004-032911-9751', "Pc8CapTally_v1{K=8192,TM=64,TN=64,N=16384,RHO=1000,DEV='sm120'}", 8, 26, 29, 72451183, 12.9208, 64.39, 13.04, 32),
    ('r20261004-032911-9751', 'Pc8CapTile_v1{TM=64,TN=64}', 8, 26, 29, 186715558, 14.6531, 77.54, 13.35, 32),
    ('r20261004-032911-9751', "Pc8CapWordUnit_v1{K=8192,DEV='sm120'}", 8, 25, 28, 3615080965, 5.7912, 82.72, 14.68, 32),
    ('r20261004-065458-8301', 'Pc4RowAFirst_v1{K=256,KS=64}', 2, 22, 25, 306126257, 0.8181, 11.48, 1.66, 8),
    ('r20261004-065458-8301', 'Pc4RowAForm_v1{K=256,KS=64,J=1}', 2, 22, 25, 311011608, 4.45, 23.4, 1.68, 8),
    ('r20261004-065458-8301', 'Pc4RowAHead_v1{K=256}', 2, 22, 25, 74290889, 0.8439, 5.96, 1.07, 8),
    ('r20261004-065458-8301', 'Pc4RowATail_v1{K=256,KS=64}', 2, 22, 25, 55911585, 1.0986, 7.87, 0.98, 8),
    ('r20261004-065940-d834', 'Pc4RowAForm_v1{K=256,KS=64,J=2}', 2, 22, 25, 311011608, 0.8469, 11.87, 1.65, 8),
    ('r20261004-065940-d834', 'Pc4RowAHead_v1{K=256}', 2, 22, 25, 74290889, 0.8698, 14.55, 1.06, 8),
    ('r20261004-065940-d834', 'Pc4RowATail_v1{K=256,KS=64}', 2, 22, 25, 55911585, 0.9024, 13.3, 0.98, 8),
    ('r20261004-142622-188e', 'Pc4RowAForm_v1{K=14336,KS=2048,J=5}', 16, 26, 30, 10328215509, 40.653, 342.19, 38.04, 16),
    ('r20261004-142637-5819', 'Pc4RowAForm_v1{K=14336,KS=2048,J=6}', 16, 26, 30, 10328215509, 30.7314, 336.08, 37.77, 16),
    ('r20261004-161659-ccf7', 'Pc4RowAFirst_v1{K=256,KS=64}', 2, 22, 25, 306126257, 2.3429, 17.95, 1.65, 8),
    ('r20261004-161659-ccf7', 'Pc4RowAHead_v1{K=256}', 2, 22, 25, 74290781, 1.1397, 7.46, 1.07, 8),
    ('r20261004-163914-c6e3', 'Pc4RowAHead_v1{K=14336}', 16, 25, 29, 22456739417, 10.8955, 585.18, 65.85, 16),
    ('r20261004-212718-0d67', 'Pc4RowAFirst_v1{K=256,KS=64}', 2, 22, 25, 306126257, 0.9273, 10.19, 1.66, 8),
    ('r20261004-212718-0d67', 'Pc4RowAForm_v1{K=256,KS=64,J=1}', 2, 22, 25, 311011608, 0.8636, 9.04, 1.65, 8),
    ('r20261004-212718-0d67', 'Pc4RowAHead_v1{K=256}', 2, 22, 25, 74290889, 0.7688, 5.36, 1.07, 8),
    ('r20261004-212718-0d67', 'Pc4RowATail_v1{K=256,KS=64}', 2, 22, 25, 55911585, 0.8755, 5.73, 1.0, 8),
    ('r20261004-213401-c9f1', 'Pc4RowAForm_v1{K=14336,KS=2048,J=3}', 16, 26, 30, 10328215509, 13.7513, 207.61, 38.08, 18),
    ('r20261004-213401-c9f1', 'Pc4RowAForm_v1{K=14336,KS=2048,J=4}', 16, 26, 30, 10328215509, 13.6487, 227.09, 38.03, 18),
    ('r20261004-213401-c9f1', 'Pc4RowAHead_v1{K=14336}', 16, 25, 29, 22456739557, 8.9423, 438.11, 65.85, 18),
    ('r20261004-213401-c9f1', 'Pc4RowATail_v1{K=14336,KS=2048}', 16, 25, 29, 58626310, 7.3724, 36.3, 8.62, 18),
    ('r20261004-213406-1ca3', 'Pc4RowAFirst_v1{K=14336,KS=2048}', 16, 26, 30, 10321753902, 14.4287, 220.46, 36.74, 18),
    ('r20261004-213406-1ca3', 'Pc4RowAForm_v1{K=14336,KS=2048,J=1}', 16, 26, 30, 10328215509, 14.9206, 227.28, 38.02, 18),
    ('r20261004-213406-1ca3', 'Pc4RowAForm_v1{K=14336,KS=2048,J=2}', 16, 26, 30, 10328215509, 13.5812, 209.28, 38.13, 18),
    ('r20261004-213411-ee31', 'Pc4RowAForm_v1{K=14336,KS=2048,J=5}', 16, 26, 30, 10328215509, 14.1627, 222.12, 38.11, 16),
    ('r20261004-213416-7b3a', 'Pc4RowAForm_v1{K=14336,KS=2048,J=6}', 16, 26, 30, 10328215509, 13.642, 222.5, 38.22, 16),
    ('r20261004-230445-e566', 'Pc4RowAForm_v1{K=14336,KS=2048,J=3}', 16, 26, 30, 10328215509, 17.5762, 225.93, 38.04, 18),
    ('r20261004-230445-e566', 'Pc4RowAForm_v1{K=14336,KS=2048,J=4}', 16, 26, 30, 10328215509, 12.3578, 194.38, 38.03, 18),
    ('r20261004-230445-e566', 'Pc4RowAHead_v1{K=14336}', 16, 25, 29, 22456739557, 9.1976, 547.96, 65.85, 18),
    ('r20261004-230445-e566', 'Pc4RowATail_v1{K=14336,KS=2048}', 16, 25, 29, 58626310, 7.7063, 38.24, 8.62, 18),
    ('r20261004-230450-e202', 'Pc4RowAFirst_v1{K=14336,KS=2048}', 16, 26, 30, 10321753902, 15.693, 218.52, 36.6, 18),
    ('r20261004-230450-e202', 'Pc4RowAForm_v1{K=14336,KS=2048,J=1}', 16, 26, 30, 10328215509, 17.6504, 300.32, 38.05, 18),
    ('r20261004-230450-e202', 'Pc4RowAForm_v1{K=14336,KS=2048,J=2}', 16, 26, 30, 10328215509, 14.375, 221.08, 38.09, 18),
    ('r20261004-230455-eac8', 'Pc4RowAForm_v1{K=14336,KS=2048,J=5}', 16, 26, 30, 10328215509, 14.938, 222.25, 38.13, 16),
    ('r20261004-230501-ceb4', 'Pc4RowAForm_v1{K=14336,KS=2048,J=6}', 16, 26, 30, 10328215509, 15.1235, 221.66, 37.97, 16),
)
#: one GEMM tc-step unit's netlist (rows + A and B nonzeros), the registered unit of every gemm-coordinate K
TC_STEP_NNZ = 9473 + 120_833 + 133_627
#: registered GEMM-coordinate sessions, the column's check (``benchmarks/pouw/hidden_zk/registered.sh``, 16 instances, sm80
#: bf16, vy-nebius-1): K (each slot K / 16 tc-step units) and the session in :data:`HIDDEN_ZK`'s form. The verifier checks
#: every one; it refused their form (a unit reading 2 input groups) until its scope took the general instance form.
REGISTERED_SESSIONS: tuple[tuple[int, tuple[str, str, int, int, int, int, float, float, float, int]], ...] = (
    (16384, ('r20261010-084308-31a9', 'gemm-coordinate/tc-step/k16384/sm80-bf16', 16, 26, 30, 57_362_394, 15.5087, 78.98,
             17_677_104 / 2 ** 20, 32)),
    (32768, ('r20261010-084308-31a9', 'gemm-coordinate/tc-step/k32768/sm80-bf16', 16, 27, 31, 57_892_825, 27.7097, 150.65,
             34_913_896 / 2 ** 20, 32)),
)
SOURCE = HIDDEN_ZK + tuple(r for _, r in REGISTERED_SESSIONS)

#: circuits' staging ladder, ``verity_flock`` staging (trace, units, lanes, stage) of a random G-gate class, 2 instances, on
#: vy-cpu-2: (run, G, ANDs lowered, wall s, peak RSS GiB). G = 4096's ANDs are its report's rounded 17.5M.
STAGING: tuple[tuple[str, int, int, float, float], ...] = (
    ('r20261009-064739-e959', 256, 110_754, 25.57, 1.08),
    ('r20261009-064739-e959', 512, 351_170, 63.81, 3.055),
    ('r20261009-064739-e959', 1024, 1_225_698, 207.22, 10.834),
    ('r20261009-064739-e959', 2048, 4_548_610, 782.68, 41.611),
    ('r20261009-062139-dab2', 4096, 17_500_000, 3907.0, 163.5e9 / 2 ** 30),
)
#: registered staging measured (``benchmarks/pouw/hidden_zk/registered.sh``, 16 instances, vy-nebius-1): (run, lowering, K,
#: ANDs lowered (one tc-step unit), wall s, peak RSS GiB)
REGISTERED_STAGING: tuple[tuple[str, str, int, int, float, float], ...] = (
    ('r20261010-084308-31a9', 'gemm-coordinate/tc-step/k16384/sm80-bf16', 16384, 8260, 29.16, 893_960 / 2 ** 20),
    ('r20261010-084308-31a9', 'gemm-coordinate/tc-step/k32768/sm80-bf16', 32768, 8260, 33.74, 901_800 / 2 ** 20),
)


# --- the fit -----------------------------------------------------------------------------------------------------------------

def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def lstsq(xs: Sequence[Sequence[float]], ys: Sequence[float], ws: Sequence[float] | None = None) -> list[float]:
    """Weighted least squares for y = c0 + Σ c_j x_j."""
    ws = ws or [1.0] * len(ys)
    X = [[1.0, *x] for x in xs]
    m = len(X[0])
    A = [[sum(w * r[i] * r[j] for r, w in zip(X, ws)) for j in range(m)] for i in range(m)]
    b = [sum(w * r[i] * y for r, y, w in zip(X, ys, ws)) for i in range(m)]
    return _solve(A, b)


#: each law's features (0: D / 2^30, 1: circuit bytes / GiB, 2: one block's bits, 2^k_log / 2^30); prove time leaves out the
#: circuit text, whose coefficient fitted with it is negative and below its error
LAWS = {"rss_gib": (0, 1, 2), "prove_s": (0, 2), "wall_s": (0, 1, 2)}
#: the column as posted at b41137f42: every law over (D, circuit bytes), fitted on :data:`HIDDEN_ZK`
POSTED_LAWS = {"rss_gib": (0, 1), "prove_s": (0, 1), "wall_s": (0, 1)}


def features(D: float, cb: float, block: float) -> tuple[float, float, float]:
    return D / 2 ** 30, cb / 2 ** 30, block / 2 ** 30


def _law(xs: Sequence[Sequence[float]], ys: Sequence[float], use: Sequence[int], ws: Sequence[float] | None = None) -> list[float]:
    c = lstsq([[x[j] for j in use] for x in xs], ys, ws)
    out = [c[0], 0.0, 0.0, 0.0]
    for j, v in zip(use, c[1:]):
        out[1 + j] = v
    return out


def _at(c: Sequence[float], x: Sequence[float]) -> float:
    return c[0] + sum(a * v for a, v in zip(c[1:], x))


def fit(source: Sequence[tuple] = SOURCE, laws: Mapping[str, Sequence[int]] = LAWS) -> dict[str, Any]:
    """The three laws over ``laws``' features and each record's residuals."""
    xs = [features(2.0 ** r[4], r[5], 2.0 ** r[3]) for r in source]
    t32 = [i for i, r in enumerate(source) if r[9] == 32]
    coef = {"rss_gib": _law(xs, [r[8] for r in source], laws["rss_gib"]),
            **{k: _law([xs[i] for i in t32], [source[i][j] for i in t32], laws[k], [source[i][j] ** -2 for i in t32])
               for k, j in (("prove_s", 6), ("wall_s", 7))}}
    points = []
    for r, x in zip(source, xs):
        p = {"run": r[0], "definition": r[1], "instances": r[2], "k_log": r[3], "dense_m": r[4], "circuit_bytes": r[5],
             "prove_s": r[6], "wall_s": r[7], "rss_gib": r[8], "threads": r[9]}
        for k, c in coef.items():
            p[f"{k}_fit"] = _at(c, x)
        points.append(p)
    resid = {"rss_gib": max(abs(p["rss_gib"] - p["rss_gib_fit"]) for p in points),
             **{k: max(abs(p[k] / p[f"{k}_fit"] - 1) for p in points if p["threads"] == 32) for k in ("prove_s", "wall_s")}}
    return {"coef": coef, "points": points, "max_residual": resid,
            "law": "y = c0 + c1 D/2^30 + c2 (circuit bytes)/GiB + c3 2^k_log/2^30 (prove: c2 = 0); RSS GiB unweighted over "
                   "all records, prove and process wall seconds weighted by 1/y^2 over the 32-thread records"}


def check(coef: Mapping[str, Sequence[float]], before: Sequence[tuple] = HIDDEN_ZK) -> list[dict[str, Any]]:
    """Each registered session against what the column predicted for it: the fit on ``before`` (the column as posted) at its
    measured D and circuit bytes, and its bound Z under the posted session model (each slot its own netlist of Z nonzeros,
    ρ = :data:`RHO_TABLE`), next to this model's prediction (one tc-step netlist, ``coef``)."""
    old = fit(before, POSTED_LAWS)["coef"]
    out = []
    for K, r in REGISTERED_SESSIONS:
        Z = K // 16 * TC_STEP_NNZ
        x = features(2.0 ** r[4], r[5], 2.0 ** r[3])
        posted = session(Z, RHO_TABLE, old)
        now = session(Z, Z / (K // 16 * 9473), coef, r[2], nets=[BYTES_PER_NNZ * TC_STEP_NNZ])
        row = {"K": K, "run": r[0], "Z": Z, "k_log": r[3], "D": 2.0 ** r[4], "circuit_bytes": r[5],
               "measured": {"rss_gib": r[8], "prove_s": r[6], "wall_s": r[7]},
               "law_at_measured": {k: _at(old[k], x) for k in ("rss_gib", "prove_s", "wall_s")},
               "posted": {k: posted[k] for k in ("k_log", "D", "circuit_bytes", "rss_gib", "prove_s", "wall_s")},
               "now": {k: now[k] for k in ("k_log", "D", "circuit_bytes", "rss_gib", "prove_s", "wall_s")}}
        for k in ("rss_gib", "prove_s"):
            row[f"{k}_vs_posted"] = r[8 if k == "rss_gib" else 6] / posted[k] - 1
            row[f"{k}_vs_law"] = r[8 if k == "rss_gib" else 6] / row["law_at_measured"][k] - 1
        out.append(row)
    return out


# --- a session ---------------------------------------------------------------------------------------------------------------

def session(Z: float, rho: float, coef: Mapping[str, Sequence[float]], slots: int = SLOTS, bytes_per_nnz: float = BYTES_PER_NNZ,
            glue: float = GLUE_BYTES, nets: Sequence[float] | None = None,
            slot: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One session of ``slots`` slots at bound Z (the module docstring): its rows per slot, k_log, whether it is in range, D,
    circuit bytes, and the laws' RSS (GiB), prove and process wall seconds. ``nets``: the netlist bytes of the unit types
    it may hold (None: each slot a netlist of its own Z nonzeros). ``slot``: the slot sumcheck's coefficients
    (:data:`SLOT_SUMCHECK`; None: no slot sumcheck), adding its entries, seconds and GiB, and the session's total seconds
    and peak GiB."""
    rows = Z / rho
    k_log = max(K_MIN_LOG, math.ceil(math.log2(rows + 1)) + 2)
    D = slots * 2.0 ** k_log
    full = bytes_per_nnz * Z
    held = [full] * slots if nets is None else sorted(nets, reverse=True)[:slots]
    cb = glue + sum(min(n, full) for n in held)
    x = features(D, cb, 2.0 ** k_log)
    s = {"Z": Z, "slots": slots, "rho": rho, "rows": rows, "k_log": k_log, "in_range": k_log <= K_LOG_MAX, "D": D,
         "circuit_bytes": cb, **{k: _at(c, x) for k, c in coef.items()}}
    T = slots if nets is None else max(len(nets), 1)
    s["template"] = max((min(n, full) / bytes_per_nnz for n in held), default=Z)
    s["entries"] = entries = slots * s["template"]
    s["k1_ns"] = 0.0 if slot is None else k1(slot["k1_ns"], s["template"])
    s["sumcheck_s"] = s["k1_ns"] * 1e-9 * entries * (slot or {}).get("reps", 0)
    bits = 2 ** math.ceil(math.log2(T)) * s["template"]
    s["sumcheck_gib"] = 0.0 if slot is None else (slot["bits_bytes"] * bits + slot["work_bytes"] * entries) / 2 ** 30
    s["total_s"] = s.get("prove_s", 0.0) + s["sumcheck_s"]
    s["peak_gib"] = s.get("rss_gib", 0.0) + s["sumcheck_gib"]
    return s


def k1(ns: float | Sequence[tuple[int, float]], template: float) -> float:
    """K1 (ns per entry per rep) for a template of ``template`` entries: ``ns`` itself, or of ``ns``' (ν, ns) points the one
    at the least ν ≥ the template's ceil(log2), the last beyond."""
    if isinstance(ns, (int, float)):
        return float(ns)
    nu = math.ceil(math.log2(max(template, 2)))
    pts = sorted(ns)
    return next((v for m, v in pts if m >= nu), pts[-1][1])


def cap(rho: float, coef: Mapping[str, Sequence[float]], gib: float, slots: int = SLOTS, **kw: Any) -> float | None:
    """The largest power of two Z in 2^16..2^40 whose session is in range and whose peak (Flock's and the slot
    sumcheck's) fits ``gib`` (None when none does)."""
    fits = [2.0 ** j for j in CAP_SEARCH
            if (s := session(2.0 ** j, rho, coef, slots, **kw))["in_range"] and s["peak_gib"] <= gib]
    return max(fits) if fits else None


def netlists(inv: Mapping[str, Any], workload: str, bytes_per_nnz: float = BYTES_PER_NNZ) -> dict[str, dict[str, Any]]:
    """Each unit type's netlist in the circuit text (the module docstring): its kind, nonzeros, bytes and the ANDs staging
    lowers."""
    out = {}
    for r in inv["rows"]:
        c = r["subcircuit"].get("per_unit")
        if r["workload"] != workload or not c:
            continue
        ru, tree = r.get("registered_unit"), r.get("tree")
        if r.get("source") == "registered" and ru:
            kind, nnz, ands = "registered", ru["rows"] + ru["nnz_a"] + ru["nnz_b"], ru["ands"]
        elif tree:
            ms = list({m["member"]: m for m in tree["members"]}.values())
            kind, nnz, ands = "tree", sum(m["nnz"] for m in ms), sum(m["ands"] for m in ms)
        else:
            kind, nnz, ands = "generic", c["nnz"], c["ands"]
        out[r["unit_type"]] = {"kind": kind, "nnz": nnz, "bytes": bytes_per_nnz * nnz, "ands": ands}
    return out


# --- staging -----------------------------------------------------------------------------------------------------------------

def staging_fit(ladder: Sequence[tuple] = STAGING) -> dict[str, list[float]]:
    """Wall seconds and peak RSS GiB, each affine in ANDs / 10^6, weighted by 1 / y²."""
    xs = [[r[2] / 1e6] for r in ladder]
    return {"wall_s": lstsq(xs, [r[3] for r in ladder], [r[3] ** -2 for r in ladder]),
            "peak_gib": lstsq(xs, [r[4] for r in ladder], [r[4] ** -2 for r in ladder])}


def staging(net: Mapping[str, Any], law: Mapping[str, Sequence[float]],
            registered: Sequence[tuple] = REGISTERED_STAGING) -> dict[str, float]:
    """One template shape's staging: the law at the ANDs it lowers, a registered type's at least its measured staging."""
    out = {k: _at(c, [net["ands"] / 1e6]) for k, c in law.items()}
    if net["kind"] == "registered" and registered:
        out = {"wall_s": max(out["wall_s"], *(r[4] for r in registered)),
               "peak_gib": max(out["peak_gib"], *(r[5] for r in registered))}
    return out


# --- a workload's audit ------------------------------------------------------------------------------------------------------

def rhos(inv: Mapping[str, Any], workload: str) -> dict[str, float]:
    """Each unit type's nonzeros per committed row (its subcircuit's ``nnz / committed``)."""
    out = {}
    for r in inv["rows"]:
        c = r["subcircuit"].get("per_unit") or {}
        if r["workload"] == workload and c.get("committed"):
            out[r["unit_type"]] = c["nnz"] / c["committed"]
    return out


def _held(us: Sequence[P.Unit], nets: Mapping[str, Mapping[str, Any]] | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """``session``'s netlists for a bucket of ``us`` (none without ``nets``), and the bucket's largest staging."""
    if nets is None:
        return {}, None
    st = [dict(nets[u.label]["staging"], unit_type=u.label, kind=nets[u.label]["kind"]) for u in us
          if "staging" in nets.get(u.label, {})]
    return {"nets": [nets[u.label]["bytes"] for u in us if u.label in nets]}, max(st, key=lambda x: x["peak_gib"], default=None)


def shape(N: int, K: int, us: Sequence[P.Unit], Z: float, slots: int, pieces: int, rho: Mapping[str, float],
          coef: Mapping[str, Sequence[float]], nets: Mapping[str, Mapping[str, Any]] | None = None,
          **kw: Any) -> dict[str, Any]:
    """A bucket of ``us`` at bound Z, each unit ``pieces`` slots: its sessions (mean, public), their cost, and its
    largest staging."""
    n = sum(u.n for u in us)
    k0, pmf = P.hyper(N, n, K)
    mean = sum(p * -(-pieces * (k0 + j) // slots) for j, p in enumerate(pmf))
    public = -(-pieces * P.upper_q(N, n, K) // slots)
    held, st = _held(us, nets)
    s = session(Z, min(rho.get(u.label, RHO_TABLE) for u in us), coef, slots, **held, **kw)
    return {"Z": Z, "slots": slots, "pieces": pieces, "types": len(us), "units": n, "labels": [u.label for u in us][:8],
            "sessions_mean": mean, "sessions_public": public, "session": s, "staging": st,
            **_cost(mean, public, s)}


def _cost(mean: float, public: float, s: Mapping[str, Any]) -> dict[str, float]:
    """A bucket's prover seconds (Flock's prove plus the slot sumcheck) and the sumcheck's, at the mean and public session
    counts."""
    return {"prover_s_mean": mean * s["total_s"], "prover_s_public": public * s["total_s"],
            "sumcheck_s_mean": mean * s["sumcheck_s"], "sumcheck_s_public": public * s["sumcheck_s"]}


def pieces_pmf(N: int, K: int, groups: Mapping[int, int]) -> list[float]:
    """The pmf of Y = Σ_p p X_p, X_p ~ Hypergeometric(N, n_p, K) for ``groups`` {p: n_p}, convolved as independent (the
    draw without replacement correlates them negatively, so this overstates Y's upper tail)."""
    y = [1.0]
    for p, n in sorted(groups.items()):
        k0, pmf = P.hyper(N, n, K)
        nxt = [0.0] * (len(y) + p * (k0 + len(pmf) - 1))
        for a, pa in enumerate(y):
            if pa:
                for j, pj in enumerate(pmf):
                    nxt[a + p * (k0 + j)] += pa * pj
        y = nxt
    return y


def pieces_shape(N: int, K: int, us: Sequence[P.Unit], limit: float, slots: int, rho: Mapping[str, float],
                 coef: Mapping[str, Sequence[float]], delta: float = P.DELTA,
                 nets: Mapping[str, Mapping[str, Any]] | None = None, **kw: Any) -> dict[str, Any]:
    """One bucket at ``limit`` holding the pieces of ``us`` (each unit ceil(z / limit) of them): sessions ceil(Y / slots)."""
    groups: dict[int, int] = defaultdict(int)
    for u in us:
        groups[math.ceil(u.z / limit)] += u.n
    pmf = pieces_pmf(N, K, groups)
    mean = sum(p * -(-y // slots) for y, p in enumerate(pmf))
    tail, q = 0.0, len(pmf) - 1
    for y in range(len(pmf) - 1, -1, -1):
        if tail + pmf[y] > delta:
            q = y
            break
        tail += pmf[y]
    public = -(-q // slots)
    held, st = _held(us, nets)
    s = session(limit, min(rho.get(u.label, RHO_TABLE) for u in us), coef, slots, **held, **kw)
    return {"Z": limit, "slots": slots, "pieces": dict(sorted(groups.items())), "types": len(us), "units": sum(groups.values()),
            "labels": [u.label for u in us][:8], "sessions_mean": mean, "sessions_public": public, "session": s,
            "staging": st,
            **_cost(mean, public, s)}


def _buckets(N: int, K: int, groups: Mapping[float, Sequence[P.Unit]], slots: int, limit: float | None,
             rho: Mapping[str, float], coef: Mapping[str, Sequence[float]], **kw: Any) -> list[dict[str, Any]]:
    """Each bucket at its Z; with ``limit``, a bucket's units over it go to one bucket of their pieces at the limit."""
    out = []
    for Z, us in groups.items():
        keep = [u for u in us if limit is None or u.z <= limit]
        if keep:
            out.append(shape(N, K, keep, max(u.z for u in keep) if len(keep) < len(us) else Z, slots, 1, rho, coef, **kw))
        if len(keep) < len(us):
            out.append(pieces_shape(N, K, [u for u in us if u not in keep], limit, slots, rho, coef, **kw))
    return out


def template_class(u: P.Unit, nets: Mapping[str, Mapping[str, Any]] | None) -> tuple[str, int]:
    """A unit type's template class: registered (a registered unit or a tree's members) or generic (its whole subcircuit),
    and its template's ceil(log2) nonzeros."""
    n = (nets or {}).get(u.label)
    t = n["nnz"] if n else u.z
    return ("registered" if n and n["kind"] in ("registered", "tree") else "generic"), math.ceil(math.log2(max(t, 2)))


def menu_t(N: int, K: int, units: Sequence[P.Unit], nets: Mapping[str, Mapping[str, Any]] | None,
           budget: int = BUCKETS_T, rare: float = RARE_DRAWS) -> tuple[list[tuple[tuple[str, int], float, list[P.Unit]]],
                                                                       list[P.Unit]]:
    """``dp4t``: the body's buckets keyed by (template class, Z), and its rare types. Every GEMM tc step shares one class
    (one registered unit at every K). A class drawn at least ``rare`` times per audit gets one bucket, and the rest of
    ``budget`` go one at a time to the class whose :func:`~unit_costs_padding.menu_dp` gains most in padded nonzeros
    (Σ Z_b W_b); the other classes' types are rare."""
    classes: dict[tuple[str, int], list[P.Unit]] = defaultdict(list)
    for u in units:
        classes[template_class(u, nets)].append(u)
    body = {c: us for c, us in classes.items() if K * sum(u.n for u in us) / N >= rare}
    rares = [u for c, us in sorted(classes.items()) if c not in body for u in us]
    cost = lambda us, j: sum(Z * sum(u.n for u in g) for Z, g in P.assign(us, P.menu_dp(us, j)).items())  # noqa: E731
    k = {c: 1 for c in body}
    while sum(k.values()) < budget:
        gain = {c: cost(us, k[c]) - cost(us, k[c] + 1) for c, us in body.items() if k[c] < len({u.z for u in us})}
        if not gain or max(gain.values()) <= 0:
            break
        k[max(gain, key=lambda c: gain[c])] += 1
    return [(c, Z, g) for c, us in sorted(body.items()) for Z, g in P.assign(us, P.menu_dp(us, k[c])).items()], rares


def _padding(N: int, K: int, units: Sequence[P.Unit], shapes: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """The padding module's zpad, pad_mean and public over the body's ``units`` in their unsplit ``shapes``."""
    actual = sum(K * u.n / N * u.z_mean for u in units)
    return {"zpad": sum(s["Z"] * K * s["units"] / N for s in shapes) / actual,
            "pad_mean": sum(s["slots"] * s["Z"] * s["sessions_mean"] for s in shapes) / actual,
            "public": sum(s["slots"] * s["Z"] * s["sessions_public"] for s in shapes) / actual}


def audit(N: int, K: int, units: Sequence[P.Unit], rho: Mapping[str, float], coef: Mapping[str, Sequence[float]],
          limit: float | None = None, slots: int = SLOTS, k: int = MENU_K, menu: str = "dp4", **kw: Any) -> dict[str, Any]:
    """The padding table's body in its dp``k`` buckets at ``slots`` (``menu`` "dp4t": :func:`menu_t`'s, its rare types each
    their own bucket) and its tail at one slot (each sampler its own bucket, the other large units one bucket at their
    largest Z), units over ``limit`` split (the module docstring)."""
    body, tail, _thr = P.tail_split(units)
    small = [u for u in body if limit is None or u.z <= limit]
    rare: list[dict[str, Any]] = []
    if menu == "dp4t":
        groups, rares = menu_t(N, K, small, kw.get("nets"))
        shapes = [dict(x, template_class=f"{c[0]} 2^{c[1]}") for c, Z, us in groups
                  for x in _buckets(N, K, {Z: us}, slots, None, rho, coef, **kw)]
        rare = [dict(min((_buckets(N, K, {u.z: [u]}, b, None, rho, coef, **kw)[0] for b in (1, slots)),
                         key=lambda x: x["prover_s_public"]), template_class="rare") for u in rares]
    else:
        shapes = _buckets(N, K, P.assign(small, P.menu_dp(small, k)) if small else {}, slots, None, rho, coef, **kw)
    pad = _padding(N, K, body, shapes + rare) if limit is None and body else None
    shapes += _buckets(N, K, {limit: [u for u in body if u not in small]} if len(small) < len(body) else {}, slots, limit,
                       rho, coef, **kw)
    samplers = [u for u in tail if u.template in P.SAMPLERS]
    big = [u for u in tail if u.template not in P.SAMPLERS]
    tails = _buckets(N, K, P.assign(samplers, P.menu_natural(samplers)) if samplers else {}, 1, limit, rho, coef, **kw)
    tails += _buckets(N, K, P.assign(big, P.menu_max(big)) if big else {}, 1, limit, rho, coef, **kw)
    every = shapes + rare + tails
    tot = lambda key: sum(s[key] for s in every)  # noqa: E731
    return {"limit": limit, "menu": menu, "body": shapes, "rare": rare, "tail": tails, "padding": pad,
            "buckets": {"body": len(shapes), "rare": len(rare), "tail": len(tails)}, "sessions_mean": tot("sessions_mean"),
            "sessions_public": tot("sessions_public"), "prover_h_mean": tot("prover_s_mean") / 3600,
            "prover_h_public": tot("prover_s_public") / 3600, "sumcheck_h_mean": tot("sumcheck_s_mean") / 3600,
            "sumcheck_h_public": tot("sumcheck_s_public") / 3600,
            "max_rss_gib": max((s["session"]["rss_gib"] for s in every), default=0.0),
            "max_peak_gib": max((s["session"]["peak_gib"] for s in every), default=0.0),
            "max_prove_s": max((s["session"]["prove_s"] for s in every), default=0.0),
            "max_total_s": max((s["session"]["total_s"] for s in every), default=0.0),
            "in_range": all(s["session"]["in_range"] for s in every),
            "max_k_log": max((s["session"]["k_log"] for s in every), default=0)}


def workload(inv: Mapping[str, Any], name: str, coef: Mapping[str, Sequence[float]],
             slaw: Mapping[str, Sequence[float]] | None = None, menu: str = "dp4", register_bulk: bool = False,
             **kw: Any) -> dict[str, Any]:
    """One workload's audit under ``menu``, unsplit and at each cap. ``register_bulk``: a what-if with every generic body
    type drawn at least :data:`RARE_DRAWS` times per audit registered as a GEMM tc-step-like unit (its template, circuit
    text and staging those of one tc step)."""
    K = int(inv["K"])
    N, units, skipped = P.units_of(inv, name)
    rho = rhos(inv, name)
    nets = netlists(inv, name, kw.get("bytes_per_nnz", BYTES_PER_NNZ))
    body, _tail, _thr = P.tail_split(units)
    registered = []
    if register_bulk:
        tc = {"kind": "registered", "nnz": TC_STEP_NNZ, "bytes": kw.get("bytes_per_nnz", BYTES_PER_NNZ) * TC_STEP_NNZ,
              "ands": REGISTERED_STAGING[0][3]}
        for u in body:
            if u.label in nets and nets[u.label]["kind"] == "generic" and K * u.n / N >= RARE_DRAWS:
                nets[u.label] = dict(tc)
                registered.append(u.label)
    for net in nets.values():
        net["staging"] = staging(net, slaw or staging_fit())
    r0 = min((rho.get(u.label, RHO_TABLE) for u in body), default=RHO_TABLE)
    kw = {**kw, "menu": menu}
    unsplit = audit(N, K, units, rho, coef, nets=nets, **kw)
    top = max(unsplit["body"], key=lambda s: s["Z"])
    ck = {k: v for k, v in kw.items() if k != "menu"}
    every = [nets[u.label]["bytes"] for u in units if u.label in nets]
    stage = [dict(n["staging"], unit_type=k, kind=n["kind"], ands=n["ands"]) for k, n in nets.items()]
    out: dict[str, Any] = {"workload": name, "menu": menu, "registered_whatif": registered, "N": N, "K": K,
                           "skipped": skipped, "rho_min": r0, "dp4_top": top["Z"],
                           "dp4": [s["Z"] for s in unsplit["body"]], "top_session": top["session"],
                           "top_staging": top["staging"], "unsplit": unsplit,
                           "staging": {"shapes": len(stage), "max": max(stage, key=lambda x: x["peak_gib"], default=None),
                                       "total_h": sum(x["wall_s"] for x in stage) / 3600,
                                       "fits_node": all(x["peak_gib"] <= NODE_GIB for x in stage)},
                           "range_cap": cap(r0, coef, math.inf, nets=every, **ck)}
    for node, gib in (("cpu", NODE_GIB), ("gpu", GPU_GIB)):
        c = cap(r0, coef, gib, nets=every, **ck)
        over = [u for u in units if c is not None and u.z > c]
        at = audit(N, K, units, rho, coef, limit=c, nets=nets, **kw) if over else unsplit
        out[node] = {"gib": gib, "cap": c, "cap_flock": cap(r0, coef, gib, nets=every, **{**ck, "slot": None}),
                     "fits_unsplit": unsplit["in_range"] and unsplit["max_peak_gib"] <= gib,
                     "split": [{"unit_type": u.label, "nnz": u.z, "draw_share": u.n / N, "pieces": math.ceil(u.z / c)} for u in over],
                     "at_cap": at,
                     "heaviest": max(at["body"] + at["rare"] + at["tail"], key=lambda s: s["session"]["total_s"])["session"],
                     "band": [hours(audit(N, K, units, rho, coef, limit=c if over else None, nets=nets,
                                          **{**kw, "slot": dict(kw["slot"], k1_ns=b)}), b)
                              for b in kw["slot"]["band"]] if kw.get("slot") and node == "cpu" else []}
    return out


def hours(a: Mapping[str, Any], k1_ns: Any = None) -> dict[str, Any]:
    """An audit's prover hours and the slot sumcheck's share of them, mean and public (K1 moves no cap, so the band reruns
    the audit at the same cap)."""
    r: dict[str, Any] = {"k1_ns": k1_ns}
    for m in ("mean", "public"):
        r[f"prover_h_{m}"] = a[f"prover_h_{m}"]
        r[f"share_{m}"] = a[f"sumcheck_h_{m}"] / a[f"prover_h_{m}"] if a[f"prover_h_{m}"] else 0.0
    return r


def prover(inv: Mapping[str, Any], law: Mapping[str, Any] | None = None,
           slot: Mapping[str, Any] | None = SLOT_SUMCHECK, **kw: Any) -> dict[str, Any]:
    law = law or fit()
    coef = law["coef"]
    slaw = staging_fit()
    reg = [kw.get("bytes_per_nnz", BYTES_PER_NNZ) * TC_STEP_NNZ] * SLOTS
    kw["slot"] = slot
    names = list(dict.fromkeys(r["workload"] for r in inv["rows"]))
    return {"format": "verity/f6-prover/v3", "K": inv["K"], "slots": SLOTS, "menu": f"dp{MENU_K}", "node_gib": NODE_GIB,
            "gpu_gib": GPU_GIB, "bytes_per_nnz": BYTES_PER_NNZ, "glue_bytes": GLUE_BYTES, "k_log_max": K_LOG_MAX, "fit": law,
            "slot_sumcheck": None if slot is None else {
                **slot, "entries": "slots x the session's largest template (its type's registered table, at most Z)"},
            "staging_fit": {"coef": slaw, "ladder": [dict(zip(("run", "G", "ands", "wall_s", "peak_gib"), r)) for r in STAGING],
                            "registered": [dict(zip(("run", "lowering", "K", "ands", "wall_s", "peak_gib"), r))
                                           for r in REGISTERED_STAGING]},
            "check": check(coef),
            "pow2": [{"registered": session(2.0 ** j, RHO_TABLE, coef, nets=reg, **kw),
                      "generic": session(2.0 ** j, RHO_TABLE, coef, **kw)} for j in POW2],
            "workloads": [workload(inv, w, coef, slaw, **kw) for w in names],
            "workloads_dp4t": [workload(inv, w, coef, slaw, menu="dp4t", **kw) for w in names],
            "whatif_registered": [workload(inv, w, coef, slaw, menu="dp4t", register_bulk=True, **kw) for w in names]}


# --- rendering ---------------------------------------------------------------------------------------------------------------

def _p2(z: float | None) -> str:
    if z is None:
        return "–"
    j = math.log2(z)
    return f"2^{int(j)}" if j == int(j) else P._si(z)


def _t(s: float) -> str:
    return f"{s:.3g} s" if s < 600 else f"{s / 60:.3g} min" if s < 36_000 else f"{s / 3600:.3g} h"


def _yn(b: bool) -> str:
    return "yes" if b else "no"


def _st(x: Mapping[str, Any] | None) -> str:
    return "–" if x is None else f"{_t(x['wall_s'])}, {x['peak_gib']:.3g} GiB ({x['kind']})"


def _k1s(ns: Any) -> str:
    if isinstance(ns, (int, float)):
        return f"{ns:.4g} ns"
    pts = sorted(ns)
    return ", ".join(f"{v:.4g} ns at ν ≤ {m}" if i < len(pts) - 1 else f"{v:.4g} ns beyond" if i else f"{v:.4g} ns"
                     for i, (m, v) in enumerate(pts))


def _slot_law(sl: Mapping[str, Any] | None) -> list[str]:
    if not sl:
        return []
    lo, hi = sl["band"]
    return [f"- slot sumcheck (candidate 3), K1 {sl['source']}: seconds = K1 x entries x {sl['reps']} reps, K1 = "
            f"{_k1s(sl['k1_ns'])} (band: {_k1s(lo)} to {_k1s(hi)}); GiB = ({sl['bits_bytes']:.4g} B x the bucket's stacked "
            f"table, its T templates (T up to a power of two) at the largest's entries, the committed bits, + "
            f"{sl['work_bytes']:.4g} B x entries, the working set) / 2^30; "
            "entries = slots x the session's largest template (its type's registered table: a registered type's one unit, "
            "a tree's distinct members, a generic type's whole subcircuit; at most Z). A session's time is prove + "
            "sumcheck, its peak RSS + sumcheck memory."]


def _slot_table(pr: Mapping[str, Any]) -> list[str]:
    sl = pr["slot_sumcheck"]
    lo, hi = sl["band"]
    L = ["", f"## Slot sumcheck per workload (K1 {_k1s(sl['k1_ns'])}; {sl['source']})", "",
         "The heaviest session at the node cap (its Z and its largest template), split into Flock's prove and the slot "
         "sumcheck; its peak; the caps with the sumcheck and with Flock alone; prover hours per audit at the node cap and "
         f"the sumcheck's share of them, at K1 and at its band's ends (low: {_k1s(lo)}; high: {_k1s(hi)}).", "",
         "| workload | cap: node / GPU | Flock alone | Z | template | Flock prove | slot sumcheck | session | Flock RSS | "
         "sumcheck memory | peak | prover h mean / public | sumcheck share mean / public | band low: prover h (share) | "
         "band high: prover h (share) |", "|" + "---|" * 15]
    for w in pr["workloads"]:
        n, g, a = w["cpu"], w["gpu"], w["cpu"]["at_cap"]
        s = n["heaviest"]
        sh = lambda m: a[f"sumcheck_h_{m}"] / a[f"prover_h_{m}"] if a[f"prover_h_{m}"] else 0.0  # noqa: E731
        bd = [f"{b['prover_h_mean']:.3g} / {b['prover_h_public']:.3g} ({b['share_mean']:.0%})" for b in n["band"]]
        L.append("| " + " | ".join([
            P._short(w["workload"]), f"{_p2(n['cap'])} / {_p2(g['cap'])}", f"{_p2(n['cap_flock'])} / {_p2(g['cap_flock'])}",
            P._si(s["Z"]), P._si(s["template"]), _t(s["prove_s"]), _t(s["sumcheck_s"]), _t(s["total_s"]),
            f"{s['rss_gib']:.3g} GiB",
            f"{s['sumcheck_gib']:.3g} GiB", f"{s['peak_gib']:.3g} GiB",
            f"{a['prover_h_mean']:.3g} / {a['prover_h_public']:.3g}", f"{sh('mean'):.0%} / {sh('public'):.0%}", *bd]) + " |")
    return L


def _names(labels: Sequence[str]) -> str:
    n: dict[str, int] = defaultdict(int)
    for x in labels:
        n[x.split("{")[0]] += 1
    return ", ".join(f"{k} x{v}" if v > 1 else k for k, v in n.items()) or "none"


def _menu_table(pr: Mapping[str, Any]) -> list[str]:
    L = ["", "## Menus: dp4, dp4t, and dp4t with the bulk unregistered types registered (what-if)", "",
         "Prover hours per audit at the node cap and the slot sumcheck's share; buckets body + rare + tail; the heaviest "
         "session at the node cap (Z, its largest template, prove + sumcheck, peak); the caps; the body's padding unsplit "
         "(zpad, pad_mean, public, as in the padding table, the rare buckets included); the types the what-if registers "
         f"(generic body types drawn at least {RARE_DRAWS} times per audit, each as one GEMM tc step's template).", "",
         "| workload | menu | buckets | prover h mean / public | sumcheck share | heaviest session | cap: node / GPU | "
         "zpad / pad_mean / public | registered |", "|" + "---|" * 9]
    for ws in zip(pr["workloads"], pr["workloads_dp4t"], pr["whatif_registered"]):
        for tag, w in zip(("dp4", "dp4t", "dp4t, registered"), ws):
            n, a, b, p = w["cpu"], w["cpu"]["at_cap"], w["unsplit"]["buckets"], w["unsplit"]["padding"]
            h = n["heaviest"]
            reg = w.get("registered_whatif") or []
            L.append("| " + " | ".join([
                P._short(w["workload"]), tag, f"{b['body']} + {b['rare']} + {b['tail']}",
                f"{a['prover_h_mean']:.3g} / {a['prover_h_public']:.3g}",
                f"{a['sumcheck_h_mean'] / a['prover_h_mean']:.0%}" if a["prover_h_mean"] else "–",
                f"{P._si(h['Z'])}, {P._si(h['template'])}: {_t(h['prove_s'])} + {_t(h['sumcheck_s'])}, {h['peak_gib']:.3g} GiB",
                f"{_p2(n['cap'])} / {_p2(w['gpu']['cap'])}", f"{p['zpad']:.3f} / {p['pad_mean']:.3f} / {p['public']:.3f}",
                _names(reg) if tag.endswith("registered") else "–",
            ]) + " |")
    L += ["", "### dp4t buckets, unsplit", "", "| workload | class | Z | slots | types | sessions mean / public | template | "
          "session: prove + sumcheck | peak | prover h mean |", "|" + "---|" * 10]
    for w in pr["workloads_dp4t"]:
        for x in w["unsplit"]["body"] + w["unsplit"]["rare"]:
            s = x["session"]
            L.append(f"| {P._short(w['workload'])} | {x['template_class']} | {P._si(x['Z'])} | {x['slots']} | {x['types']} | "
                     f"{x['sessions_mean']:.1f} / {x['sessions_public']} | {P._si(s['template'])} | {_t(s['prove_s'])} + "
                     f"{_t(s['sumcheck_s'])} | {s['peak_gib']:.3g} GiB | {x['prover_s_mean'] / 3600:.3g} |")
    return L


def render(pr: Mapping[str, Any]) -> str:
    f = pr["fit"]
    c = f["coef"]
    sc = pr["staging_fit"]["coef"]
    law = lambda k: " + ".join(f"{v:.4g}{t}" for v, t in zip(c[k], ("", "·D/2^30", "·cb/GiB", "·2^k_log/2^30")) if v)  # noqa: E731
    L = [f"# F6 prover column: sessions of {pr['slots']} slots, {pr['menu']} buckets, K = {pr['K']}", "",
         "Flock's CPU ZK prover (`flock-circuit prove --zk`, 32 threads, Xeon 6776P), fitted on the source runs below. D: "
         "the session's bits (slots x 2^k_log); cb: its circuit text, the netlists of its distinct unit types at "
         f"{pr['bytes_per_nnz']} B per nonzero (a registered type's one unit, a generic type's whole subcircuit, a tree's "
         f"distinct members) plus {pr['glue_bytes'] / 1e6:.0f} MB of glue. A block is at most 2^{pr['k_log_max']} bits "
         "(Stmt.InRange): flock-circuit refuses a session needing more.", "",
         f"- peak RSS (GiB) = {law('rss_gib')}; max |residual| {f['max_residual']['rss_gib']:.2f} GiB",
         f"- prove (s) = {law('prove_s')}; max relative residual {f['max_residual']['prove_s']:.2f}",
         f"- prover process wall, circuit load included (s) = {law('wall_s')}; max relative residual "
         f"{f['max_residual']['wall_s']:.2f}",
         f"- staging, once per template shape (s) = {sc['wall_s'][0]:.4g} + {sc['wall_s'][1]:.4g}·ANDs/10^6; peak (GiB) = "
         f"{sc['peak_gib'][0]:.4g} + {sc['peak_gib'][1]:.4g}·ANDs/10^6 (circuits' ladder); a registered type at least its "
         "measured staging",
         f"- fits: one node {pr['node_gib']} GiB (vy-nebius-1/2), one GPU {pr['gpu_gib']:.1f} GiB (RTX PRO 6000; the CPU law as "
         "a bound)", *_slot_law(pr.get("slot_sumcheck")), "",
         "## Check: registered GEMM-coordinate sessions", "",
         "Measured against the column as posted (its fit at the measured D and circuit text, and its session model at the "
         "bound Z: each slot its own netlist of Z nonzeros) and against this model.", "",
         "| K | run | Z | k_log | D | circuit text | peak RSS | posted model | posted law at measured D, cb | this model | "
         "prove | posted model | posted law | this model |", "|" + "---|" * 14]
    for r in pr["check"]:
        m, po, la, no = r["measured"], r["posted"], r["law_at_measured"], r["now"]
        L.append(f"| {r['K']} | {r['run']} | {P._si(r['Z'])} | {r['k_log']} | 2^{int(math.log2(r['D']))} | "
                 f"{r['circuit_bytes'] / 1e6:.1f} MB | {m['rss_gib']:.3g} GiB | {po['rss_gib']:.3g} ({r['rss_gib_vs_posted']:+.0%}) | "
                 f"{la['rss_gib']:.3g} ({r['rss_gib_vs_law']:+.0%}) | {no['rss_gib']:.3g} | {_t(m['prove_s'])} | "
                 f"{_t(po['prove_s'])} ({r['prove_s_vs_posted']:+.0%}) | {_t(la['prove_s'])} ({r['prove_s_vs_law']:+.0%}) | "
                 f"{_t(no['prove_s'])} |")
    L += ["", "## Per workload: dp4 at B = 16, unsplit and at the cap", "",
          "cap: the largest power-of-two bound whose 16-slot session is in range and whose peak (Flock's and the slot "
          "sumcheck's) fits; units over it split into ceil(z / cap) pieces. Prover hours: Σ sessions x (prove + slot "
          "sumcheck) on one prover, body and tail, mean / public session counts. Staging: the dp4 top bucket's largest "
          "template shape, and the largest of every shape in the workload.", "",
          "| workload | dp4 top Z | top session k_log | top session prove | top session RSS | unsplit in range | cap: node / GPU | "
          "units split at node cap (draw share) | sessions at node cap (mean / public) | prover h at node cap | "
          "prover h at GPU cap | staging, dp4 top | staging, largest shape | staging fits node | staging all shapes |",
          "|" + "---|" * 15]
    for w in pr["workloads"]:
        u, n, g, s, st = w["unsplit"], w["cpu"], w["gpu"], w["top_session"], w["staging"]
        a, ag = n["at_cap"], g["at_cap"]
        L.append("| " + " | ".join([
            P._short(w["workload"]), P._si(w["dp4_top"]), f"{s['k_log']}", _t(s["prove_s"]), f"{s['rss_gib']:.3g} GiB",
            f"{_yn(u['in_range'])} (k_log ≤ {u['max_k_log']})", f"{_p2(n['cap'])} / {_p2(g['cap'])}",
            f"{len(n['split'])} ({sum(x['draw_share'] for x in n['split']):.2%})",
            f"{a['sessions_mean']:.0f} / {a['sessions_public']}", f"{a['prover_h_mean']:.3g} / {a['prover_h_public']:.3g}",
            f"{ag['prover_h_mean']:.3g} / {ag['prover_h_public']:.3g}", _st(w["top_staging"]), _st(st["max"]),
            _yn(st["fits_node"]), f"{st['shapes']} shapes, {st['total_h']:.3g} h"]) + " |")
    if pr.get("slot_sumcheck"):
        L += _slot_table(pr)
    if pr.get("workloads_dp4t"):
        L += _menu_table(pr)
    L += ["", f"## Power-of-two bounds (ρ = {RHO_TABLE} nonzeros per row)", "",
          "Registered: 16 distinct types, each one GEMM tc-step netlist. Generic: each slot its own netlist of Z nonzeros.", "",
          "| Z | rows per slot | k_log | in range | D | registered: circuit text | peak RSS | prove | generic: circuit text | "
          "peak RSS | prove | slot sumcheck | its memory | fits GPU: registered / generic |", "|" + "---|" * 14]
    for q in pr["pow2"]:
        r, gq = q["registered"], q["generic"]
        L.append(f"| {_p2(r['Z'])} | {P._si(r['rows'])} | {r['k_log']} | {_yn(r['in_range'])} | 2^{int(math.log2(r['D']))} | "
                 f"{r['circuit_bytes'] / 2 ** 30:.3g} GiB | {r['rss_gib']:.3g} GiB | {_t(r['prove_s'])} | "
                 f"{gq['circuit_bytes'] / 2 ** 30:.3g} GiB | {gq['rss_gib']:.3g} GiB | {_t(gq['prove_s'])} | "
                 f"{_t(r['sumcheck_s'])} | {r['sumcheck_gib']:.3g} GiB | "
                 f"{_yn(r['peak_gib'] <= pr['gpu_gib'])} / {_yn(gq['peak_gib'] <= pr['gpu_gib'])} |")
    L += ["", "## Staging", "", "| run | G / lowering | ANDs | wall | peak RSS | law: wall | law: peak |", "|" + "---|" * 7]
    for r in pr["staging_fit"]["ladder"]:
        L.append(f"| {r['run']} | G = {r['G']} | {P._si(r['ands'])} | {_t(r['wall_s'])} | {r['peak_gib']:.3g} GiB | "
                 f"{_t(_at(sc['wall_s'], [r['ands'] / 1e6]))} | {_at(sc['peak_gib'], [r['ands'] / 1e6]):.3g} GiB |")
    for r in pr["staging_fit"]["registered"]:
        L.append(f"| {r['run']} | {r['lowering']} | {P._si(r['ands'])} | {_t(r['wall_s'])} | {r['peak_gib']:.3g} GiB | "
                 f"{_t(_at(sc['wall_s'], [r['ands'] / 1e6]))} | {_at(sc['peak_gib'], [r['ands'] / 1e6]):.3g} GiB |")
    L += ["", "## Shapes", "", "| workload | split at | part | Z | k_log | slots | pieces | types | sessions mean / public | "
          "session RSS | session prove | slot sumcheck | its memory | prover h mean | largest staging |", "|" + "---|" * 15]
    for w in pr["workloads"]:
        for tag, a in (("unsplit", w["unsplit"]), ("node cap", w["cpu"]["at_cap"] if w["cpu"]["split"] else None)):
            if a is None:
                continue
            for part in ("body", "tail"):
                for s in a[part]:
                    pc = s["pieces"] if isinstance(s["pieces"], int) else f"{min(s['pieces'])}-{max(s['pieces'])}"
                    L.append(f"| {P._short(w['workload'])} | {tag} | {part} | {P._si(s['Z'])} | {s['session']['k_log']} | "
                             f"{s['slots']} | {pc} | {s['types']} | {s['sessions_mean']:.1f} / {s['sessions_public']} | "
                             f"{s['session']['rss_gib']:.3g} GiB | {_t(s['session']['prove_s'])} | "
                             f"{_t(s['session']['sumcheck_s'])} | {s['session']['sumcheck_gib']:.3g} GiB | "
                             f"{s['prover_s_mean'] / 3600:.3g} | {_st(s.get('staging'))} |")
    L += ["", "## Source runs", "", "| run | definition | instances | k_log | D | circuit text | prove | fit | process wall | fit | "
          "peak RSS | fit | threads |", "|" + "---|" * 13]
    for p in f["points"]:
        L.append(f"| {p['run']} | {p['definition']} | {p['instances']} | {p['k_log']} | 2^{p['dense_m']} | "
                 f"{p['circuit_bytes'] / 1e6:.0f} MB | {p['prove_s']:.3g} | {p['prove_s_fit']:.3g} | {p['wall_s']:.3g} | "
                 f"{p['wall_s_fit']:.3g} | {p['rss_gib']:.3g} | {p['rss_gib_fit']:.3g} | {p['threads']} |")
    return "\n".join(L) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inventory", required=True, help="unit_costs' unit-costs.json")
    ap.add_argument("--out", required=True, help="directory for prover.json and prover.md")
    d = SLOT_SUMCHECK
    k1s = lambda t: tuple(sorted(tuple(map(float, x.split(":"))) for x in t)) if ":" in t[0] else float(t[0])  # noqa: E731
    ap.add_argument("--k1-ns", nargs="+", default=None, metavar="[NU:]NS",
                    help="slot sumcheck ns per entry per rep, one value or NU:NS points (default: proofs' measurement)")
    ap.add_argument("--k1-lo", nargs="+", default=None, metavar="[NU:]NS", help="K1's band, low end")
    ap.add_argument("--k1-hi", nargs="+", default=None, metavar="[NU:]NS", help="K1's band, high end")
    ap.add_argument("--reps", type=int, default=d["reps"])
    ap.add_argument("--bits-bytes-per-entry", type=float, default=d["bits_bytes"], help="committed bits' bytes per entry")
    ap.add_argument("--work-bytes-per-entry", type=float, default=d["work_bytes"],
                    help="the sumcheck's working set per entry of the slot being proved")
    ap.add_argument("--k1-source", default=None, help="where K1 comes from (default: proofs' K1 run)")
    ap.add_argument("--no-slot-sumcheck", action="store_true", help="Flock's prove alone")
    a = ap.parse_args(argv)
    from pathlib import Path

    from .unit_costs import retree

    inv = json.loads(Path(a.inventory).read_text())
    if "token_select" not in inv:
        retree(inv)
    given = any(v is not None for v in (a.k1_ns, a.k1_lo, a.k1_hi)) or (
        a.reps, a.bits_bytes_per_entry, a.work_bytes_per_entry) != (d["reps"], d["bits_bytes"], d["work_bytes"])
    slot = None if a.no_slot_sumcheck else {
        "k1_ns": k1s(a.k1_ns) if a.k1_ns else d["k1_ns"],
        "band": (k1s(a.k1_lo) if a.k1_lo else d["band"][0], k1s(a.k1_hi) if a.k1_hi else d["band"][1]),
        "reps": a.reps, "bits_bytes": a.bits_bytes_per_entry, "work_bytes": a.work_bytes_per_entry,
        "source": a.k1_source or ("given" if given else d["source"])}
    pr = prover(inv, slot=slot)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prover.json").write_text(json.dumps(pr, indent=1) + "\n")
    (out / "prover.md").write_text(render(pr))
    print(f"wrote {out / 'prover.json'} and {out / 'prover.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
