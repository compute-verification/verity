"""`lean_headline_dc`: the H200's and B200's headline instances as Lean, exact
(security/Definitions/Accounting/HeadlineDc/Sm*.lean). `machine-model lean-headline` writes them after sm_120's, and
with `--check` exits 1 when a committed file is not what it would write; `--numbers` prints each instance's bound, its
branches and their terms and writes no Lean file, and `--json F` also writes the readings as JSON (`write`).

Each instance is a reading of `dc_reading`, whose docstring says what a reading holds and how its certificate is found.
This module writes each reading as a Lean module: a family's data sheet (`render_sheet`), its measured vector beside it
(`render_measured`) and, for the H200, the measured vector's twin under (I3) (`render_twin`).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from verity_machine_model import lean_headline as LH
from verity_machine_model import sheet as S
from verity_machine_model.dc_reading import (
    CATCH_ALL,
    CATCH_ALLS,
    FAMILIES,
    Family,
    Reading,
    _and,
    _Exact,
    reading,
)
from verity_machine_model.lean_headline import Q

#: What other files read of it: the command, and its test in `benchmarks/pouw/tests/` through
#: `benchmarks/pouw/lean_headline_dc.py`, which also gives it `LH` and `Q`.
__all__ = ["OUT", "main", "render_measured", "write"]

OUT = LH.ROOT / "security" / "Definitions" / "Accounting" / "HeadlineDc"


def _as_written(path: str) -> str:
    """A Lean file's path as the generated modules' doc comments name it: its top directory with the capital it had when
    they were written. A layout move that renames that directory leaves Lean sources byte for byte (Lake builds from
    them), so the committed modules keep the old name, and what this writes must stay what they hold."""
    top, _, rest = path.partition("/")
    return f"{top.capitalize()}/{rest}"
SHEET_ROW = {"DS_issue": ".issue", "DS_tensor": ".tensor", "DS_smem": ".smem"}
SHAPES = {"Sm90": "sm_90's wgmma as one warp's quarter of its warpgroup instruction (`m·n·k/4`), a 2:4 sparse form at "
                  "its dense twin's, its mma.sync whole, DMMA's at the f64 peak",
          "Sm100": "sm_100's tcgen05 MMA whole (`m·n·k`), a 2:4 sparse form at its dense twin's, its mma.sync whole, "
                   "DMMA's at the f64 peak"}


def _entries(x: _Exact, n: dict[str, Fraction]) -> list[str]:
    return [f"({x.classes.index(k)}, {v.numerator}, {v.denominator})"
            for k, v in sorted(n.items(), key=lambda kv: x.classes.index(kv[0]))]


def _sparse(x: _Exact, n: dict[str, Fraction]) -> str:
    return "[" + ", ".join(_entries(x, n)) + "]"


def _sparse_def(name: str, x: _Exact, n: dict[str, Fraction]) -> str:
    head = f"def {name} : Sparse := ["
    return head + LH._wrap(_entries(x, n), " " * len(head))[len(head):] + "]"


def _q(v: Fraction) -> str:
    return f"{v.numerator} / {v.denominator}"


def _uniform_words(fam: Family) -> str:
    return "".join(f" and the uniform `{k}` ({om}, a UR pair)" for k, om in fam.uniform.items())


MOVES = ("-rt3-wideh", "-rt3", "-wideh")          # the counted moves of a lane route's LI


def _menu_text(names: list[str], counted: list[str]) -> str:
    """The fold menu's programs as the Definitions docstring names them: the built ones (the bit-exact checkpoint loops
    and li-bulk at its measured cost), then the counted ones (the moves of LI on each lane route and li-bulk's
    members), which ♭ never folds by."""
    built = [n for n in names if n not in counted]
    loops = [n for n in built if not n.startswith("li-bulk")]
    bulk = [n for n in built if n not in loops]
    moved = [n for n in counted if n.endswith(MOVES)]
    members = [n for n in counted if n not in moved]
    if any(not n.startswith("li-bulk") for n in members):
        raise SystemExit(f"counted folds {members}: the Definitions docstring names the moves of LI and li-bulk's "
                         "members only; describe the others in `_menu_text`")
    text = f"programs ({', '.join(loops)}: the checkpoint's bit-exact loops"
    if not (bulk or counted):
        return text.replace("programs", "bit-exact programs", 1) + ")"
    if bulk:
        text += (f"; {', '.join(bulk)}: li's XOR through the stores, a bulk XOR with the key slab and a load back, at "
                 "its measured cost")
    if counted:
        text += "; and the counted folds, none built, which ♭ never folds by: "
        text += "; ".join(
            ([(f"{', '.join(moved)}, the counted moves: each lane route and the fold with LI's 64 LOP3 a "
               "segment as a memory round trip of three issue slots (ATOM.E.XOR and two LD.E back), its 64 IMAD as "
               "IMAD.WIDE, or both")] if moved else [])
            + ([f"{', '.join(members)}, li-bulk's BT1024 and trimmed W = ∞ members"] if members else []))
    return text + ")"


def _writer_lines(x: _Exact) -> str:
    return "\n".join(f"    ({x.classes.index(k)}, {om}){',' if j + 1 < len(x.writers) else ''}  -- {k}"
                     for j, (k, om) in enumerate(x.writers))


def _bullets(items: list[str]) -> str:
    return "\n".join(LH._words(b, "* ", "  ") for b in items)


def _header(fam: Family, what: str, imports: str, reads: str) -> str:
    return f'''module

public import {imports}

/-!
# {fam.gpu} ({fam.key})'s {what} (trusted; generated)

Written by `benchmarks/pouw/lean_headline_dc.py` from the machine model's data (`benchmarks/pouw/machine_model/data`),
and checked against it by `benchmarks/pouw/tests/test_lean_headline_dc.py`: edit the data and rerun the exporter, never
this file. `Accounting.Guarantees.TS.{reads}` reads it.
'''


def _model_line(fam: Family, x: _Exact, dvar: str, machine: str, fill: str, rtype: str) -> str:
    nk, nt, nfo = len(x.classes), len(x.writers), len(x.folds)
    body = (f"{dvar}.modelFill {nk} {nt} {nfo} {machine} {fill}" if x.nx else f"{dvar}.model {nk} {nt} {nfo} {machine}")
    return (f"noncomputable def model : Model (Fin {nk}) {rtype} (Fin {nt}) Unit Empty (Fin {nfo}) Unit :=\n"
            f"  {body}")


def render_sheet(fam: Family, r: Reading) -> str:
    x = r.data
    ix = x.classes.index
    lane = [p for p, _ in fam.pipes.values()]
    sv = fam.sheet.replace("-", "").lower()
    nk = len(x.classes)
    filler = bool(x.nx)
    star = {SHEET_ROW.get(row) or f".pipe .{fam.pipes[row][0]}": v for row, v in x.lam.items()}
    lam_cases = "\n".join(f"  | {k} => {_q(v)}" for k, v in star.items())
    if len(star) < 3 + len(lane):
        lam_cases += "\n  | _ => 0"
    honest = f"an {x.form} at L = {x.L}"
    fill_doc = "the filler (ffill on lanes, `flatFill`), " if filler else "no filler, "
    bullets = [
        f"`classes`: the {nk} instruction classes of the {fam.sheet} census, in the order `Fin {nk}` indexes them."
        + (f" `{CATCH_ALL[fam.key]}` is A2's catch-all: every warp instruction no other class names counts there, at its "
           f"issue slot alone." if fam.key in CATCH_ALL else ""),
        (f"`{sv}`, `sheet`: the {fam.gpu}'s data sheet ({fam.sheet}), `{S.relative_path(fam.sh)}` as "
         f"`{_as_written(fam.sh.lean)}` states it: f_max {fam.fmax} MHz ({fam.fmax_src}); each tensor class's dense MACs a warp "
         f"instruction ({SHAPES[fam.ns]}) and its operand type's dense MACs an SM clock; the Nsight Compute pipes "
         f"{', '.join(p.upper() for p in lane[:-1])} and {lane[-1].upper()}, each at its width (the Programming "
         f"Guide's, and the uniform datapath's the census's measured UP line, which no document states); no "
         f"shared-memory row ({fam.sheet} prices none). An MMA class the sheet rates on no peak (b1) and the memory, "
         f"atomic and control classes are their issue slot alone (A2). Its machine reproduces every {fam.sheet} "
         f"coefficient."),
        f"`d2s`, `model`: `{fam.design}` as a TS model on the sheet's machine: writer kinds every MMA class of the "
        f"census (ω its destination's 32-bit words a warp instruction, A6, sheet.toml's `omega`: an f64 element two)"
        + _uniform_words(fam)
        + f"; the vector classes CUDA's lane classes and both readings' catch-alls "
        f"({', '.join(k for k in x.vec if k in CATCH_ALLS)}: A6 counts a vector word's instructions in them, and an "
        "instruction no class names, IMAD.WIDE among them, counts in the catch-all)"
        + "; the fold routes the design's fold"
        + (f" and its {len(x.fold_names)} other {_menu_text(x.fold_names, x.counted)} on lanes, and alone on the uniform datapath "
           "at the least of their counts; "
           if x.fold_names else " on lanes and alone on the uniform datapath; ")
        + f"the table level's bytes counted in {fam.bytes}, which `b_T = 0` prices at nothing (the census has no load "
        f"class); A6's floors `q = q_T = ½`, `b_T = 0`; {x.Nw} committed words a unit of honest work ({honest})"
        + ("; its filler menu the one route ffill runs on (sm_90's uniform datapath has no FFMA)." if filler else "."),
        (f"`flat`: the floor program ♭ a unit: the MMA and the design's operand floor (word production), the fold "
         f"({x.fold_mix}), {fill_doc}and ε_skip's skippable work at {x.skip}'s price, {r.eps_skip:.6%} of ♭'s bound "
         f"(`eps_budget.design_skip` on {fam.sheet})."),
        f"`lamStar`: π*, the certificate minimising the bound over this menu ({' + '.join(f'{k} {float(v):.6f}' for k, v in x.lam.items())}).",
        "`checks`: the checks a window, one a round (Theorem S's S).",
    ]
    pipes = [f"some .{p}" if p else "none" for p in x.sheet["pipe"]]
    fold_lines = ",\n".join(f"    {_sparse(x, f)}" for f in x.folds)
    pipe_ctors = "\n".join(f"  | {p}" for p in lane)
    width_lines = "\n".join(f"  | .{p} => {S.literal(w)}" for p, w in fam.pipes.values())
    return _header(fam, f"headline instance: {fam.sheet} and `{fam.design}`", "Definitions.Accounting.HeadlineDc.Encoding",
                   f"{fam.ns}SheetHeadline") + f'''
{_bullets(bullets)}
-/

@[expose] public section

namespace Accounting.TS.Headline.{fam.ns}

open Accounting.TS

/-- The {fam.sheet} census's instruction classes, in index order. -/
def classes : List String := {LH._list([f'"{k}"' for k in x.classes], "  ")}

/-- The Nsight Compute pipes the sheet rates (A4). -/
inductive Pipe
{pipe_ctors}
  deriving DecidableEq

instance instFintypePipe : Fintype Pipe where
  elems := {{{", ".join(f"Pipe.{p}" for p in lane)}}}
  complete x := by cases x <;> simp

/-- Each pipe's results an SM clock: the Programming Guide's, the uniform datapath's the census's UP line. -/
def Pipe.width : Pipe → ℚ
{width_lines}

/-- The {fam.gpu}'s data sheet, by class index. -/
def {sv} : SheetDataQ Pipe where
  fmax := {fam.fmax}
  macs := {LH._list([str(m) for m in x.sheet["macs"]], "    ")}
  peak := {LH._list([str(p) for p in x.sheet["peak"]], "    ")}
  pipe := {LH._list(pipes, "    ")}
  width := Pipe.width
  smem := {LH._list([str(s) for s in x.sheet["smem"]], "    ")}

/-- The data sheet over the census's classes. -/
noncomputable def sheet : Sheet (Fin {nk}) Pipe := {sv}.sheet {nk}

/-- `{fam.design}`'s design data: writer kinds as (class, ω), vector classes, fold routes. -/
def d2s : DesignData where
  writers := [
{_writer_lines(x)}
  ]
  vec := [{", ".join(str(ix(k)) for k in x.vec)}]  -- {", ".join(x.vec)}
  folds := [
{fold_lines}]
  bytes := {ix(fam.bytes)}  -- {fam.bytes}
  q := {_q(x.q)}
  qT := {_q(x.q)}
  Nw := {x.Nw}

/-- ♭'s filler a unit{": ffill on lanes" if filler else ": none"}. -/
{_sparse_def("flatFill", x, x.nx)}

/-- `{fam.design}`'s TS model on the data sheet's machine. -/
{_model_line(fam, x, "d2s", "sheet.machine", "flatFill", "(SheetRow Pipe)")}

{LH._words(f"♭'s word production a unit: {x.words_mix}. -/", "/-- ", "")}
{_sparse_def("flatWords", x, x.nw)}

{LH._words(f"♭'s fold a unit: the lane fold for {x.Nw} words, {x.fold_mix}. -/", "/-- ", "")}
{_sparse_def("flatFold", x, x.nf)}

/-- ♭'s skippable work a unit, at {x.skip}'s price. -/
{_sparse_def("flatSkip", x, x.ns)}

/-- The floor program ♭, labelled. -/
noncomputable def flat : Labelled (Fin {nk}) := labelled {nk} flatWords flatFold flatFill flatSkip

/-- π*: the certificate, its weights exact and summing to 1. -/
noncomputable def lamStar : SheetRow Pipe → ℝ
{lam_cases}

/-- The checks a window. -/
def checks : ℕ := {x.checks}

end Accounting.TS.Headline.{fam.ns}
'''


def render_measured(fam: Family, r: Reading, s: Reading) -> str:
    x, xs = r.data, s.data
    if x.classes != xs.classes or x.vec != xs.vec or x.folds != xs.folds or x.nx != xs.nx:
        raise SystemExit(f"{fam.measured}: its classes, vector classes, fold routes or ♭'s filler are not "
                         f"{fam.sheet}'s, which the measured file reads")
    own, own_w = x.nf != xs.nf, x.nw != xs.nw
    fold = "flatFold" if own else f"{fam.ns}.flatFold"
    words = "flatWords" if own_w else f"{fam.ns}.flatWords"
    own_fold = ((LH._words(f"♭'s word production a unit on {fam.measured}: {x.words_mix}. -/", "/-- ", "")
                 + f"\n{_sparse_def('flatWords', x, x.nw)}\n\n") if own_w else "") + (
        (LH._words(f"♭'s fold a unit on {fam.measured}: the lane fold for {x.Nw} words, {x.fold_mix}. -/", "/-- ", "")
         + f"\n{_sparse_def('flatFold', x, x.nf)}\n\n") if own else "")
    mine = [p for p, o in (("word production", own_w), ("fold", own)) if o]
    theirs = [p for p, o in (("word production", own_w), ("fold", own)) if not o] + ["filler"]
    ix = x.classes.index
    nk, m = len(x.classes), len(x.rows)
    sv = fam.measured.replace("-", "").lower()
    rows = []
    for j, row in enumerate(x.rows):
        entries = [f"({ix(k)}, {x.c[k][row].numerator}, {x.c[k][row].denominator})" for k in x.classes if row in x.c[k]]
        body = LH._wrap(entries, "   ", 116)[3:] if entries else ""
        rows.append(f"  -- {j}: {row}\n  [{body}]{',' if j + 1 < m else ''}")
    menu = (f"honest's form{_uniform_words(fam)}: the census prices every other tensor form on the joint rows at "
            "the scalar coefficient, a lower bound on its issue cost, so they are off the menu and `NoSum` carries "
            "them" if r.menu == "honest" else "every MMA class it times (ω its destination's words, A6)")
    lam = LH._wrap([f"({x.rows.index(k)}, {v.numerator}, {v.denominator})" for k, v in x.lam.items()],
                   " " * (len(f"  cert {m} ["))).lstrip()
    joint = fam.measured in CATCH_ALL and set(x.c[CATCH_ALL[fam.measured]]) != {"issue"}
    bullets = [
        f"`rowNames`, `{sv}`, `machine`: the measured vector {fam.measured}, {m} rows of SMSP-cycles a warp instruction "
        f"at its epoch's census, each sparse over `{fam.ns}.classes`; every facet SM-clocked, f_max {fam.fmax} MHz."
        + (f" `{CATCH_ALL[fam.measured]}` is A2's catch-all here: every warp instruction no other class prices on "
           f"{fam.measured} counts there, at the measured issue slot "
           + ("and the joint-issue lines every timed scalar class carries, since a run timed its instructions beside "
              "the tensor." if joint else "alone.") if fam.measured in CATCH_ALL else "")
        + (f" The sheet's classes no run beside {fam.measured}'s census timed ({', '.join(x.untimed)}) are read "
           + ("" if joint else "the same way, ") + "at the measured issue slot alone, and are off the writer menu, as "
           "they were when the catch-all counted them: `NoSum` carries their words." if x.untimed else ""),
        f"`d2s`, `model`: `{fam.design}` on that machine, its writer kinds {menu}; `{fam.ns}.d2s`'s vector classes, "
        f"fold routes, floors and `N_w`" + ("; ♭'s filler `" + fam.ns + ".flatFill` its filler route." if x.nx else "."),
        "".join(f"`{n}`, " for n, o in (("flatWords", own_w), ("flatFold", own)) if o)
        + f"`flatSkip`, `flat`: ♭ is `{fam.ns}.flat`'s {_and(theirs)}"
        + "".join(f", its {p} {fam.measured}'s ({x.words_mix if p == 'word production' else x.fold_mix})"
                  for p in mine)
        + f", with ε_skip's skippable work at {fam.measured}'s {r.eps_skip:.6%} (`eps_budget.design_skip` on "
        f"{fam.measured}), at {x.skip}'s price.",
        (f"`lamStar`: π*, the certificate minimising the bound over this menu "
         f"({' + '.join(f'{k} {float(v):.6f}' for k, v in x.lam.items())})."),
    ]
    nt, nfo = len(x.writers), len(x.folds)
    body = (f"d2s.modelFill {nk} {nt} {nfo} machine {fam.ns}.flatFill" if x.nx else f"d2s.model {nk} {nt} {nfo} machine")
    return _header(fam, f"measured headline instance: {fam.measured} and `{fam.design}`",
                   f"Definitions.Accounting.HeadlineDc.{fam.ns}", f"{fam.ns}MeasuredHeadline") + f'''
{_bullets(bullets)}
-/

@[expose] public section

namespace Accounting.TS.Headline.{fam.ns}Measured

open Accounting.TS

/-- {fam.measured}'s rows, in index order. -/
def rowNames : List String := {LH._list([f'"{row}"' for row in x.rows], "  ")}

/-- {fam.measured}'s coefficients, row by row: `(class index, numerator, denominator)`. -/
def {sv} : List Sparse := [
{chr(10).join(rows)}]

/-- The measured machine: {fam.measured}'s rows over `{fam.ns}.classes`. -/
noncomputable def machine : Machine (Fin {nk}) (Fin {m}) :=
  rowsMachine {nk} {m} {sv} {fam.fmax}

{LH._words(f"`{fam.design}`'s design data on {fam.measured}: writer kinds as (class, ω), and `{fam.ns}.d2s`'s "
           f"routes. -/", "/-- ", "")}
def d2s : DesignData where
  writers := [
{_writer_lines(x)}
  ]
  vec := {fam.ns}.d2s.vec
  folds := {fam.ns}.d2s.folds
  bytes := {fam.ns}.d2s.bytes
  q := {fam.ns}.d2s.q
  qT := {fam.ns}.d2s.qT
  Nw := {fam.ns}.d2s.Nw

/-- `{fam.design}`'s TS model on the measured machine. -/
noncomputable def model : Model (Fin {nk}) (Fin {m}) (Fin {nt}) Unit Empty (Fin {nfo}) Unit :=
  {body}

{own_fold}/-- ♭'s skippable work a unit on {fam.measured}, at its {x.skip} price. -/
{_sparse_def("flatSkip", x, x.ns)}

/-- The floor program ♭ on {fam.measured}, labelled. -/
noncomputable def flat : Labelled (Fin {nk}) :=
  labelled {nk} {words} {fold} {fam.ns}.flatFill flatSkip

/-- π*: the certificate, its weights exact and summing to 1. -/
noncomputable def lamStar : Fin {m} → ℝ :=
  cert {m} [{lam}]

end Accounting.TS.Headline.{fam.ns}Measured
'''


def render_twin(fam: Family, t: Reading, r: Reading) -> str:
    """The measured reading's twin under (I3) `ScalarRoute` at `q = q_T = fam.twin`: the measured file's machine, writer
    kinds, routes and ♭, none of which reads `q`, at the twin's floors and its own π*."""
    x, xm = t.data, r.data
    if (x.classes, x.rows, x.vec, x.writers, x.folds, x.nw, x.nf, x.nx, x.ns, x.B) != (
            xm.classes, xm.rows, xm.vec, xm.writers, xm.folds, xm.nw, xm.nf, xm.nx, xm.ns, xm.B):
        raise SystemExit(f"{fam.measured} at q = {t.q}: its machine, menu or ♭ is not the measured file's, which the "
                         "twin reads")
    m, nk, nt, nfo = len(x.rows), len(x.classes), len(x.writers), len(x.folds)
    lam = LH._wrap([f"({x.rows.index(k)}, {v.numerator}, {v.denominator})" for k, v in x.lam.items()],
                   " " * (len(f"  cert {m} ["))).lstrip()
    qd = f"{x.q.numerator}/{x.q.denominator}"
    body = (f"d2s.modelFill {nk} {nt} {nfo} {fam.ns}Measured.machine {fam.ns}.flatFill" if x.nx
            else f"d2s.model {nk} {nt} {nfo} {fam.ns}Measured.machine")
    bullets = [
        (f"`d2s`, `model`: `{fam.ns}Measured.d2s` (its writer kinds, and `{fam.ns}.d2s`'s vector classes, fold routes "
         f"and `N_w`) at (I3)'s floors `q = q_T = {qd}`, `b_T = 0`, on `{fam.ns}Measured.machine`. (I3) `ScalarRoute` "
         "there is `accounting-ts-scalar-route`, assumed: A6 gives it only at `q, q_T ≤ ½` (`SheetScalarRouteHalf`). ♭ "
         f"is `{fam.ns}Measured.flat`, since `setup`'s floor program doesn't read `q`."),
        (f"`lamStar`: π*, the certificate minimising the bound over this menu at `q = {qd}` "
         f"({' + '.join(f'{k} {float(v):.6f}' for k, v in x.lam.items())})."),
    ]
    return _header(fam, f"(I3) twin: {fam.measured} at q = {qd} and `{fam.design}`",
                   f"Definitions.Accounting.HeadlineDc.{fam.ns}Measured", f"{fam.ns}MeasuredI3Headline") + f'''
{_bullets(bullets)}
-/

@[expose] public section

namespace Accounting.TS.Headline.{fam.ns}MeasuredI3

open Accounting.TS

/-- `{fam.ns}Measured.d2s` at (I3)'s floors `q = q_T = {qd}`. -/
def d2s : DesignData where
  writers := {fam.ns}Measured.d2s.writers
  vec := {fam.ns}Measured.d2s.vec
  folds := {fam.ns}Measured.d2s.folds
  bytes := {fam.ns}Measured.d2s.bytes
  q := {_q(x.q)}
  qT := {_q(x.q)}
  Nw := {fam.ns}Measured.d2s.Nw

/-- `{fam.design}`'s TS model on the measured machine at (I3)'s floors. -/
noncomputable def model : Model (Fin {nk}) (Fin {m}) (Fin {nt}) Unit Empty (Fin {nfo}) Unit :=
  {body}

/-- π*: the certificate, its weights exact and summing to 1. -/
noncomputable def lamStar : Fin {m} → ℝ :=
  cert {m} [{lam}]

end Accounting.TS.Headline.{fam.ns}MeasuredI3
'''


def write(check: bool = False, numbers: bool = False, json_to: str | None = None) -> int:
    """Print every reading and write each family's Lean files. With `check` write nothing and return 1 when a committed
    file is not what it would write; with `numbers` print only; with `json_to` also write the readings there as JSON.
    Either way 1 when a reading's bound is above its target."""
    out, record, files = [], [], {}
    for fam in FAMILIES:
        for v, full, q in ((fam.sheet, False, Q), (fam.measured, False, Q),
                           *(((fam.measured, True, Q),) if fam.honest_menu else ()),
                           *(((fam.measured, False, fam.twin),) if fam.twin else ())):
            r = reading(fam, v, full, q)
            (record if full else out).append(r)
            if not full and q != Q:
                files[OUT / f"{fam.ns}MeasuredI3.lean"] = render_twin(fam, r, out[-2])
            elif not full:
                files[OUT / (fam.ns + ("" if r.sheet else "Measured") + ".lean")] = (
                    render_sheet(fam, r) if r.sheet else render_measured(fam, r, out[-2]))
            cert = " + ".join(f"{k} {x:.4f}" for k, x in r.certificate.items())
            print(f"{r.design} on {v} ({r.menu} menu{', for the record' if full else ''}, q = q_T = {r.q}, "
                  f"ln(8/7) ≥ {r.ln_p}{', the (I3) twin' if q != Q else ''}): π* {cert}, ♭ on "
                  f"{r.flat_facet} at {r.B:.4f} a unit (ε_cert {r.eps_cert:.4%}), w* {r.wstar}, ε_skip "
                  f"{r.eps_skip:.4%} (skip as {r.skip_class}), κ {r.kappa:.6f}, χ* {r.chi:.4f}; 1 − Y "
                  f"{r.one_minus_Y:.4%} (the LP's {r.lp:.4%}); folded {r.folded:.4%}, unfolded {r.unfolded:.4%}: the "
                  f"Lean bound {r.bound:.4%} {'≤' if r.bound <= r.target else 'ABOVE'} {r.target:.2%} (machine "
                  f"model: core {r.machine_core:.4%}, ε_machine {r.machine_eps:.4%}, ε_luck {r.luck:.4%})"
                  + (f"; {r.statement}: ε {float(Fraction(r.pair['eps'])):.3%} ({r.pair['eps']}) with B_♭ "
                     f"{float(Fraction(r.pair['B_flat'])):.4f} "
                     f"on {r.pair['flat_facet']}, credit (1 − ε)·B_♭ {float(Fraction(r.pair['credit'])):.4f}"
                     if r.statement else ""))
    if json_to:
        rows = [{k: v for k, v in asdict(r).items() if k != "data"} for r in out + record]
        Path(json_to).write_text(json.dumps(rows, indent=1) + "\n")
    above = any(r.bound > r.target for r in out)
    if numbers:
        return 1 if above else 0
    if check:
        stale = [p for p, text in files.items() if not p.is_file() or p.read_text() != text]
        for p in stale:
            print(f"{p.relative_to(LH.ROOT)} is not what `machine-model lean-headline` writes: rerun it",
                  file=sys.stderr)
        return 1 if stale or above else 0
    if above:
        return 1
    for p, text in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f"wrote {p.relative_to(LH.ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="write every reading here as JSON")
    ap.add_argument("--check", action="store_true", help="exit 1 when a committed Lean file is not what it would write")
    ap.add_argument("--numbers", action="store_true", help="print the readings only; write no Lean file")
    a = ap.parse_args(argv)
    return write(a.check, a.numbers, a.json)
