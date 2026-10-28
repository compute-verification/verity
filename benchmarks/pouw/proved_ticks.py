"""`proved_ticks`: PoUW's proved block-ticks for a window's registered work, the figure network accounting's joint ledger
credits as a block's `compute` with basis "proved" (`benchmarks/datacenter`: `Placement.public(plan, ticks=...)`, where
its tests place PoUW's compiled Program on a block).

    uv run python benchmarks/pouw/proved_ticks.py     the hand-checked point: H1 + fx1 on DS-120

For a census W of a window's registered work on a block of G GPUs of one device, at its data-sheet reading (`placed`
reads W off a block where the datacenter places a Program `SCHEME` or its one stream `SCHEME_V3` compiled):

    ticks(W) = (1 − ε) · U(W) · B_♭ / (4 · SMs · f_max) / G / tick_seconds

- U(W), W's units of honest work: a unit is one IMMA.16832, 4096 int8 products (N_w = 128 committed words at L = 32).
  W counts each product's int8 products (`mac:linears`, `mac:lm_head`, `mac:qk`, `mac:pv`), and each counts at ♭'s
  committed words a transcript word for its noise (`eps_budget.FLOOR_WORDS`: H1's cohalf QKᵀ at 1.25), as
  `eps_budget.floor_skip` counts them.
- B_♭, the floor program ♭'s bound a unit in SMSP-cycles on the sheet's machine: `eps_budget.prices`' ♭ (the tensor
  route at the operand floor φ_del and the fold, no filler) with ε_skip's skippable work in it, B_♭ = B/(1 − ε_skip),
  ε_skip `eps_budget.design_skip` at the sheet's vector, as Lean's ♭ (`Accounting.TS.Headline.Sm120.flat`, #2482) is.
  For H1 + fx1 on DS-120, ♭ binds DS_issue alone: an IMMA.16832, 0.025641 LDSM_X4 and fx1's 8 ALU and 8 IMAD warp
  instructions, 17.025641, and 0.4207% skippable, so B_♭ = 17.097571.
- 4 · SMs · f_max: the credit is T_cred = B_♭/f_max a unit on one SMSP (`Accounting.TS.Model.credit`), and a GPU's
  4 · SMs SMSPs share the window's units. SMs is data/table.toml's (188 on sm_120), f_max the sheet's (`FMAX_MHZ`).
- ε, PoUW's headline at the sheet: outside δ = 2^-40, an accepted window takes at least (1 − ε) · T_cred a unit over
  every route. ε (Corollary D3's) is what any other route, skipped work and luck can save against ♭, so the figure
  is (1 − ε) of ♭'s block-ticks, not ♭'s. `SHEET_EPS` is 5.860%, `Sm120SheetHeadline`'s (#2924's A5b on #2791, not on
  main yet: receipt luck ½ named, operand luck ¾, p = 7/8, fold skippers charged): its π* puts 197/3761 on DS_smem so
  that a fold on the bulk engine (RT4-SHEET-1) costs at least φ*, which lowers N_w(w* + φ*) to 16.1095 against B_♭.
  main's machine model reads 0.585% at p = ¾ (`machine-model headline`), with no bulk route; the measured reading,
  0.591% on VF3s-16, is at measured rates, not the sheet's.

W's bound is U(W) · B_♭ exactly, not more: every product's committed word is the same unit on the same facet, so the
bound is linear in units, and the products differ only in U's weights and in ε_skip, which is the design's most over
models, contexts and phases. A kind outside the design's products would be work on its own facets, whose bound with
W's is an LP over their union and not this sum, so `units` refuses it.

What it rests on:

- the TS model on the data-sheet machine and its named assumptions A1–A7 (`Accounting.Assumptions.Sheet`): A1 the SM
  clock at most f_max, A2 and A2u dispatch (a uniform instruction takes an issue slot), A3 the tensor peak, A4 the
  pipes, A5 shared memory, A6 words written, A7 semantics;
- FOLD-OPT at π* (`Accounting.Assumptions.TS.FoldOptFoldedAt`): fx1 at 4 lane instructions a word is the cheapest fold
  at the words a run folds, checked within a stated menu (`Sm120SheetHeadline`'s docstring); a run that skips a fold is
  charged at the whole-unit luck ½;
- A5b (`Accounting.Assumptions.TS.SheetSmem`): a bulk or TMA instruction holds the shared-memory bytes it moves in the
  banks at A5's rate, a reading no vendor document states;
- ε's source above, on the full-checking layer: ε_sample and the Fiat–Shamir premium are separate lines;
- the node's work, in its G GPUs' ticks: the headline is per node (ASK-65 = A), the work of its GPUs, host CPU and
  peer GPUs together, priced on the GPUs' sheet (A1–A7 price none of its other devices: its CPUs, its DPU). Read as
  the G GPUs' own time, it needs LOC's host deduction (decision 37) and every billable GPU on the node credited each
  window (the machine model's README, "The headline is per node"; RT-SCOPE, #2681);
- per block only under attribution's grade: PoUW credits the prover system (decision 111), so the ledger puts this on
  the block only as far as `BlockAccount.local_at_least` places it there, graded by its anchor (`Credit.grade`), and
  the rest on the datacenter.
"""
from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).parent))     # appended: network accounting's tests import this module by path
import eps_budget as EB
from verity_machine_model import headline
from verity_machine_model.eps import PRODUCTS, design, honest, table
from verity_machine_model.model import device

SMSPS = 4                          # SMSPs an SM
#: each family's f_max, the sheet's highest SM clock (A1): SPEC's P1, the hardware's 2,430 MHz on sm_120 (Lean's
#: `Accounting.TS.Headline.Sm120.ds120.fmax`, #2482)
FMAX_MHZ = {"sm_120": 2430}
#: PoUW's headline ε at the data sheet, H1 + fx1 on DS-120: `Accounting.SecurityProofs.TS.Sm120SheetHeadline`
SHEET_EPS = 5860 / 100000
KINDS = tuple(f"mac:{p}" for p in PRODUCTS)
#: PoUW's compile pass for H1 + fx1 on sm_120 (`verity.core.protocols.pouw.circuit.h1tile.SCHEME`)
SCHEME = "eps1-sm120-h1fx1"
#: its one-stream form, whose units are `PouwH1Tile_v3` (`h1tile.SCHEME_V3`)
SCHEME_V3 = "eps1-sm120-h1fx1-os"


@dataclass(frozen=True)
class Sheet:
    """A family's data-sheet reading: its headline design (data/headline.toml), the DS vector it prices it on, its SMs
    and f_max."""
    family: str
    vector: str
    design: str
    sms: int
    fmax_mhz: float


def sheet(family: str = "sm_120") -> Sheet:
    if family not in FMAX_MHZ:
        raise ValueError(f"no data-sheet f_max for {family}: add the sheet's highest SM clock to FMAX_MHZ")
    f = headline.spec()["family"][family]
    return Sheet(family, f["sheet"]["vector"], f["design"], table(device(family)).sms, FMAX_MHZ[family])


def unit_cycles(s: Sheet) -> float:
    """B_♭: ♭'s bound a unit of honest work on one SMSP, in SMSP-cycles, with ε_skip's skippable work in it."""
    data = device(s.family)
    d = design(s.design, data)
    return EB.prices(d.name, data, s.vector).mma / (1 - EB.design_skip(d, data=data, vector=s.vector).eps_time)


def units(census: Mapping[str, float], s: Sheet) -> float:
    """The census's units of honest work: each product's int8 products at ♭'s committed words a transcript word, over
    a unit's (one MMA of honest's form)."""
    data = device(s.family)
    d = design(s.design, data)
    bad = [k for k in census if k not in KINDS or census[k] < 0]
    if bad:
        raise ValueError(f"census kinds {bad}: a PoUW census counts {d.title}'s int8 products, {', '.join(KINDS)}, each "
                         "at least 0; other work binds its own facets, and its bound with these is an LP over the union")
    h = honest(d.honest, data)
    return sum(n * EB.FLOOR_WORDS.get(d.noise[k[4:]], 1.0) for k, n in census.items()) / (h.words * h.depth)


@dataclass(frozen=True)
class Tile:
    """One H1 + fx1 tile unit's shape: r × c at contraction K, `z` on `cells` of its grid, its Definition's `version`
    (`PouwH1Tile_v1`, or `PouwH1Tile_v3`, the one stream)."""
    r: int
    c: int
    k: int
    cells: int
    version: int = 1

    @property
    def linears(self) -> int:
        """`z`'s int8 products, the served linear's: K a cell."""
        return self.cells * self.k

    @property
    def products(self) -> int:
        """Every int8 product census/v1 counts in the tile (none served, no tail): the lanes' segment dots', r·c·(K + 32m)
        (`PouwSegDot_v2` and `PouwSegDot0_v2`), and on v1 `z`'s own (Lean's `h1Rows` dotMacs). v3 has none of its own:
        its z is recombined from the lanes' folded words, so its products are the segment dots' alone."""
        from verity.core.protocols.pouw.circuit import h1tile
        lanes = self.r * self.c * (self.k + 32 * h1tile.Layout(self.r, self.c, self.k).m)
        return lanes + (self.linears if self.version == 1 else 0)


def tiles(program, calls: Collection[int] | None = None) -> Counter:
    """The H1 tile units a Program `SCHEME` or `SCHEME_V3` compiled runs in its root Calls `calls` (all of them by default), by shape,
    each once per activation (a batch or scan multiplies its callee's, as `logical.work` counts)."""
    from verity.core.protocols.pouw.circuit import h1tile
    from verity.ml.circuits.program import is_input
    versions = {id(h1tile.PouwH1Tile): 1, id(h1tile.PouwH1Tile3): 3}
    memo: dict[int, Counter] = {}

    def of(fn) -> Counter:
        if id(fn) not in memo:
            if id(getattr(fn, "defn", None)) in versions:
                s = fn.bindings
                memo[id(fn)] = Counter({Tile(s["R"], s["C"], s["EXACT"].signature()[0][0][1].n, len(s["PAIRS"]),
                                             versions[id(fn.defn)]): 1})
            elif getattr(fn, "body", None) is None:
                memo[id(fn)] = Counter()
            else:
                total: Counter = Counter()
                for nd in fn.body.nodes:
                    for t, n in of(nd.fn).items():
                        total[t] += n * (nd.n if nd.form in ("batch", "scan") else 1)
                memo[id(fn)] = total
        return memo[id(fn)]

    if calls is None:
        return Counter(of(program.fn))
    out: Counter = Counter()
    for k, nd in enumerate(program.root.body.nodes):
        if k in calls and not is_input(nd.fn):
            out.update(of(nd.fn))
    return out


def placed(work: Mapping[str, float], units: Mapping[Tile, int]) -> dict[str, float]:
    """The census of a block's work where the datacenter places a Program `SCHEME` or `SCHEME_V3` compiled (`Profile.work[block]`, the
    Program's Calls on the block by kind), whose tile units are `units` (`tiles` of those Calls). Its int8 products
    (`mac:int8`) are every tile's census/v1 products, its lanes' segment dots' and on v1 `z`'s (`Tile.products`), which
    must be the units' exactly. Credited are `z`'s alone on either version, each a linear's at the tile's K (the unit is H1's Dither-C tile, a
    committed word a product): the segment dots are the receipt's work, as its Boolean gates are. Its Boolean gates
    (`cuda:gate`) are the units' seeds, noise and receipt folds: ♭ prices the fold a unit, and leaving the rest out only
    lowers the bound. Any other kind isn't the scheme's, and the proved figure stands in for the block's whole compute, so
    it is refused."""
    other = sorted(set(work) - {"mac:int8", "cuda:gate"})
    if other:
        raise ValueError(f"work {other} on the block isn't {SCHEME}'s or {SCHEME_V3}'s, whose units are int8 products and Boolean gates: "
                         "the proved figure would leave it out of the block's compute")
    want = sum(t.products * n for t, n in units.items())
    if work.get("mac:int8", 0) != want:
        raise ValueError(f"the block's {work.get('mac:int8', 0):g} int8 products aren't its H1 tiles' {want} (the lanes' "
                         "segment dots' r·c·(K + 32m) a tile, and on v1 z's cells·K): pass `tiles` of the block's Calls")
    return {"mac:linears": float(sum(t.linears * n for t, n in units.items()))}


def proved_block_ticks(census: Mapping[str, float], s: Sheet, eps: float, tick_seconds: float, gpus: int = 1) -> float:
    """(1 − eps) of the block-ticks ♭ takes for the census on `gpus` GPUs of the sheet's device, at its rates."""
    if not 0 <= eps < 1 or tick_seconds <= 0 or gpus < 1:
        raise ValueError(f"eps in [0, 1), tick_seconds > 0 and gpus ≥ 1, not {eps}, {tick_seconds}, {gpus}")
    seconds = (1 - eps) * units(census, s) * unit_cycles(s) / (SMSPS * s.sms * s.fmax_mhz * 1e6)
    return seconds / gpus / tick_seconds


def main() -> int:
    s = sheet()
    data = device(s.family)
    d = design(s.design, data)
    b, skip = EB.prices(d.name, data, s.vector).mma, EB.design_skip(d, data=data, vector=s.vector).eps_time
    second = SMSPS * s.sms * s.fmax_mhz * 1e6 * 4096
    print(f"{d.title} on {s.vector}: ♭ {b:.6f} SMSP-cycles a unit, ε_skip {skip:.6%}, B_♭ = {unit_cycles(s):.6f}; "
          f"ε {SHEET_EPS:.3%}, so (1 − ε)·B_♭ = {(1 - SHEET_EPS) * unit_cycles(s):.6f}")
    print(f"{second:.6g} int8 products of linears (a GPU-second at {s.sms} SMs × {SMSPS} SMSPs × {s.fmax_mhz:g} MHz, a "
          f"cycle a unit) on one GPU: {proved_block_ticks({'mac:linears': second}, s, SHEET_EPS, 1.0):.6f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
