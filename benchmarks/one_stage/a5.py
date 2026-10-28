"""One-stage audit, run A5: a deferred Commit's window, a stand-in, or a TP Commit's windows, one per rank, each `served` or a stand-in. The
registration is over serving rows of the Commit's own values, and the drawn units go to M0's prover and the Lean verifier of record.

A deferred Commit's window is committed after serving from its sealed replay bundle opened against its record and bound to its run
root. Serving's commitment is the record's roots, under the scheme the record was committed in (rows-sha512 by default, vllm-v1 an
older record's), and the window's roots are a fresh commitment, so the window is core's `stand-in`, its link to them provenance (`of`:
the record's commitment and run root, the bundle's manifest digest and `verity-vllm serving-rows`): that the rows hold the values under
the record's roots is the committer's opening, not proved. A SHA-512 record does not make it `served`; it becomes `served` once a
circuit opens the record's own leaves (`pipeline.deferred_rows`).

A TP Commit's windows (`tp-commit --serving-rows`, `pipeline.tp.serving_rows`) are `served`: each rank commits its window from its own
committed store while serving, after every rank's root is final and before any draw, under ``run = <TP run root>/rank<r>``. Serving's
own commitment, made before any draw, is what the audit holds; that those rows are the values under the ranks' rows-sha512 roots is
the rank's claim, not proved (the profile makes no claim that they are, `pipeline.serving_rows`). A5 takes a TP window as served only
under a SHA-512 TP run root, as it takes a deferred Commit's: a TP Commit's ranks commit under rows-sha512 and its TP run root is
SHA-512 (`tp.run_root`, ``verity/tp-run-root/v2``), and an old record's SHA-256 v1 root is refused by name before the draw
(`source`: `sha512`). A TP Commit that wrote no window and deferred its ranks' replay to sealed bundles has each rank's window
from its bundle afterwards (`verity-vllm tp-serving-rows`, `tp.serving_rows.from_bundle`), under the same run and `tp`: a
`stand-in`, its `of` the rank's rows-sha512 root under the TP run root and the bundle's manifest digest, made after the Commit and
before the draw (`a5_tp.sh` with `A5_WINDOWS=bundles`).

    PYTHONPATH=<tree>:<tree>/benchmarks/numerical/python:<tree>/integrations/vllm \\
      python benchmarks/one_stage/a5.py --bundle DIR --row-dir DIR --window DIR \\
        --partition F --template-sets DIR --m0-pythonpath PATH --bin flock-circuit --lean flock-verify \\
        --firewall flock-firewall --out DIR [--delta 0.5]
      python benchmarks/one_stage/a5.py --tp-commit DIR --build-root DIR --workload F --windows DIR [--ranks 0,1,...] [--jobs N] \\
        --partition F --template-sets DIR --m0-pythonpath PATH --bin flock-circuit --lean flock-verify \\
        --firewall flock-firewall --out DIR [--delta 0.5]
      python benchmarks/one_stage/a5.py --tp-commit DIR --build-root DIR --advice-opening F \\
        [--other-advice-opening F --other-windows DIR] --windows DIR --coins verifier ...

A deferred Commit's run takes its window in the other form too, `served` (`a5.sh` with `WINDOW=served`): the in-Commit hook's
(`commit --serving-rows`, `serving_rows.readers`, which the deferred Commit runs at its C2 site before the bundle is sealed,
`c2_replay.serving_rows`), serving's own commitment, made before any draw. A5 takes it only over a
SHA-512 run root of record (64 bytes: serving's record is hm96-sha512 rows in frame-v3-sha512 trees), so a vllm-v1 run's served
window is refused by name (`source`: `sha512`), and only when the record's weights root, the tree its rows' weights were opened
against, is SHA-512 too (`weights_sha512`: rows-sha512's registration, `step_rows.weights_tree`). The summary reports the form
(`source`) and the commitment it names (`commitment`).

* **Serving's files.** `--window DIR` (a TP Commit: `--windows DIR`'s `rank<r>/`) is what serving wrote: for a deferred Commit,
  what `verity-vllm serving-rows` wrote (window kind `stand-in`, its run the Commit's run root, its `of` and `link`
  `deferred_rows.stand_in`'s) or what the in-Commit hook wrote (`served`, its run the Commit's run root, no stand-in link). Each
  holds `registration.json` (core's v1 record), `window.json`, per member k `m<k>-public.bin` and `m<k>-private.bin`, and per
  registered value its members read by reference `v<j>-private.bin` (the private files: prover only). The prover wraps each
  member's bodies under the circuit for its template (`flock_inputs.flock_input_files`): every member's public file before the
  draw, and a drawn member's prover's file (`inst`, its rows and salts, gathered from both private files) after it, under the
  header its public file was held to (`stage`).
* **The verifier's own copies.** A deferred Commit: `--bundle` and `--row-dir` are its record and its Build's Programs of record, and
  the verifier derives its window and the record's commitment from them alone (`deferred_rows.of_record`, `window_of`). A TP
  Commit: `--tp-commit` is its record (its
  `--out`: `runs.jsonl`, `versions.json`), `--build-root` holds each rank's Programs of record in `rank<r>/`, and `--workload` is the
  workload file the record served; the verifier derives rank r's window from them alone (`tp.serving_rows.window_of_record`), and
  `--jobs N` audits up to N ranks at once, each in its own process. Either
  window is the population program and partition over the served Programs, the law and the run of record. The verifier composes
  each member's circuit itself from an input set of its template under `--template-sets` (the lowering is all it reads; the values
  are never used), or, for a member of bit rows in chunks (`serving_rows.chunked`: a MoE pair GEMM's coordinate,
  `COORDS`, a MoeSum or all-reduce column block, `COLS`, a w13 coordinate's K-chunk, `KCHUNKS`, a sparse-MLA row's split part or
  combine, `SPLITS`, or a row template the partition's `chunk_cap` names), by C-Flock's `UnitStager` on the rows the window lays out
  (`compose_circuit.py --template`, once per template, every chunk on that circuit, read and hashed once,
  `flock_inputs.flock_circuit`), bound to its partition and program. It holds every public file to that circuit, its unit indices and classes, and
  the registered roots and served domains. Then R1-R7 run over the registration with the leaf layer over the public bodies it holds
  (`serving_rows.leaf_layer`), and the window is held to its own source (`source`). A deferred Commit's is `deferred_rows.source`:
  either kind names the record's commitment, and it is a stand-in whose run, `of` and `link` are `deferred_rows.stand_in` over the
  run root of record, the digest of the bundle's manifest it opened and that commitment, or a served window whose run is the run
  root of record, a SHA-512 root, with no stand-in link. A TP rank's is `tp.serving_rows.source`: `served`, or a stand-in whose
  `of` and `link` are
  `tp.serving_rows.stand_in` over the bundle the record names for the rank, under a SHA-512 TP run root, its run, `tp` and served
  Programs the record's.
* **Advice windows** (``--advice-opening``, a TP Commit, network accounting under arbitrary vLLM usage): the advice (I1's
  canonical encoding: everything serving made unpredictable) is private, and the registration binds the root of its hiding
  commitment and its bits (`advice_windows.commit_advice`). The verifier is handed the commitment's opening, the named gap until
  the compiler's wiring is proved in zero knowledge, and derives rank r's window from the record, the advice it opens and its own
  copy of the registered compiler alone (`advice_windows.of_opening`, `window_of_advice`; ``--compiler traced``, the stand-in,
  the Build's Programs of record), not from the workload; `source` holds the registration's ``compiler`` (name and digest) and
  ``advice`` (root and bits) to the verifier's, so the stream's registration, which the draw comes after, fixes both. The
  prover's windows come from the bundles (`verity-vllm advice-windows`). Negatives (`advice_negatives`), each refused before the
  draw: the registration naming another compiler or another advice root, the opening under another salt, the verifier opening
  other advice (``--other-advice-opening``: `source` and R2), and a unit committed under other advice (``--other-windows``: R4
  and its roots). With ``--transfer-gate`` (the templates the compiler lists as holding transfer gates), the window's
  ``work:K`` law proves every unit of theirs (`serving_rows.work_table_of`: work 0, floor all its units) beside the drawn
  rest, the stream registering with that work table; its negatives (`transfer_gate_negatives`, and in `advice_negatives` the
  registration under the compiler listing none, `source`): the proved set less a transfer-gate unit (`check_proved`), a draw
  trading one for a unit of the rest (`check_draw`), and the registration's law drawing them as the rest is (R1).
* **One draw per window** by the registered law (the partition file's `subset:k`) over the window's whole population, made by the
  verifier's Lean `flock-verify draw`; a TP Commit draws each rank's window on its own. Each member's server serves its share (`serve
  --draw-file`). The sessions are `--zk` (the masked protocol; without it Lean accepts a subset draw only at hidden outputs and one
  table, verifier PROTOCOL §16.15). Each prover runs behind the firewall's send check (`--firewall`, Lean's `flock-firewall`: `FC_FIREWALL=1`, live
  PROTOCOL.md §10.4), its `Hello` set up on the statement's partition and program. Lean `verify --zk` holds each session with
  `--partition` and `--program` (units derived).
* **Plain sessions** (``--plain``; `--zk` is the default): C-Flock's unmasked protocol, which Lean's soundness covers at one
  table under a stratified draw, hidden outputs included (`SoundExecH`), until proofs' multi-table coverage (`PLAIN_SCOPE`).
  So a plain audit takes only a stream's draw (``--challenger`` or ``--coins verifier``) by a ``stratified:K`` law, over a
  window of one member with no registered reads or shared rows, and refuses any other before the draw (`plain_refusal`).
  The member's one session is served the stream's draw itself and runs on per-round OS coins (``FC_COINS=os``), and Lean
  `verify --coins os --stratified K` holds it to the stratified law it derives from its own `--program` and `--partition`.
  C-Flock's firewall checks only a `--zk` statement's sends, so a plain prover isn't behind it (the summary's ``plain``).
* **The draw from the auditor's challenger** (`--challenger local|gate:USER@HOST`, a TP Commit; `a5_coins`), on the proof
  system's path of record: the step's one challenger session, opened on the TP run root once every rank has passed its
  checks before the draw and planned the coins its draw reads (`a5_coins.need`), and only under a reserve that covers
  them all (`a5_coins.reserve_reason`), and each rank's own `service.Stream` over its window's templates in their order (one stratified Spec over their mixed population, its law the
  partition file's rule, `stratified:K` among them), on the session's coins (`coins.ChallengerCoins` over
  `remote.RemoteChallenger`). The stream registers the rank's window at its counts under the TP run root (`register`), the
  coins commit to the registration's digest, and the stream draws once (`select`: Lean's `Select.select` on the bytes it
  read, then `check_draw`). Once the session has ended, the stream takes the proof (`deliver`) and its outcome (`outcome`)
  with the coins' evidence (key, ledger, main, pins, segment, session; ``--main`` the verifier's own first-parent history of
  main, which must hold the commit the host's pins log pins for the session): the audit record, and the whole run by the
  Lean verifier of record (`Run.run`, `coin_source` challenger), which fails `run` unless it is the Python's. `coins.coins_reason`
  (Python's `read_reason`) holds the evidence too, the audit record failing `coins` where it refuses. Negatives, each its own
  stream's outcome on the honest draw's bytes: the evidence with a tampered opening, with the coin opened before the
  registration, with a segment that skips a released coin, with the pin off main, with no pins log, none, and for another
  window. Every rank of the step draws from the one session, so `--jobs` covers every rank.
* **The draw from the verifier's own coins** (`--coins verifier`, a TP Commit, instead of `--challenger`): the same
  `service.Stream` per rank, register, select, deliver and outcome, on the service's default coins (`verifier.coins`, the
  Lean's `IO.getRandomBytes`), which the stream's `select` reads once it has received the rank's registration and written
  its receipt (`a5_coins.VerifierCoins`). The stream's coin source is the verifier: its outcome carries no evidence, and the
  Lean's `Run.run` (`coin_source` verifier) holds it. No auditor's device draws these coins (the record and the summary say
  `source: verifier`), and no session ties the ranks, so `--jobs` takes any number. Negatives (`verifier_outcome`,
  `a5_coins.verifier_negatives`): the honest bytes passed off as the challenger's with no evidence, and as the verifier's
  with evidence attached, each refused by the Lean's run at its select (`coins`); a second select on the registration
  (`drawn`) and a select on a registration the stream never received (`unknown`), each refused before any coin is read.
* **Transfer gates opened under the warden's signed roots** (`audit`'s ``registered``): a member whose port's rows are an outside
  registered value's (a warden link-window's cells, `certifier.warden_rows`), committed under that value's salts
  (`ml.windows.rows.commit`'s ``salts_of``), is verified with the verifier's own reads of it (``--registered``, the port at
  the root the warden signed, which the caller read under the warden's key), the same reads, record and negatives as a row
  table's (below). `a5_warden.py` is their stand-in.
* **Row tables** (`ml.windows.rows.Window.row_tables`): a window whose `ScaledMmS8Coordinate` chunks read their ``x``
  rows from one row table per engine step and linear (`pipeline.serving_rows.with_row_tables`) registers one port more,
  ``row-table``, the tree over its steps' roots, which the Spec declares a tree over roots of row-seg leaves
  (`service.ROOTS`, ``over``). Each chunk's public file is a shared-row file, its header's ``shared_rows`` held to the
  verifier's own layout, and before the draw the tables are held to the chunks' public ``x`` strings, the registered root
  and the step roots serving recorded before each step's coin (`tables.check_row_tables`). Lean's verifier of record
  covers a shared-row public file only where it holds its own registered reads (``--registered``, `Flock.ProvedScope.
  sharedRowsOk`): each chunk reads one table, so a5 gives each linked chunk's verify the reads of its ``x`` port
  (`tables.registered_reads`: its table's root as the members' strings recompute it, which the check above tied to the
  registration and the step roots, and the row each unit reads), and the chunk's header gives each ``x`` row a path to that
  root. The audit record's ``registered_reads`` carries each drawn chunk's table and root from that file beside the
  ``registered`` finding Lean's verdict reports opening its rows against (`tables.reported_reads`), and a5 accepts only
  when they are one. Negatives on one drawn chunk: no registered reads, another root, its units reading another row
  (`registered_negatives`), and a verdict reporting another root. No TP window of record has row tables yet (`tp.serving_rows.window_of_record` names no serve-time
  context); `a5_row_tables.py` is their stand-in.
* **Wiring across statements** (`commit.wiring`, verifier PROTOCOL §16.13 ``Wiring``): a w13 coordinate's K-chunks
  (`pipeline.kchunk_rows`) are three templates, each chunk's ``acc`` row the output row of the chunk before it, committed under
  that row's salt. Lean's links hold only inside a statement, so after the draw each drawn unit's wires are held on the public
  bodies the leaf layer binds (`wiring.check_drawn`, Lean's `Wires.Summary.drawnOk`: the two rows' strings are one), and a
  drawn chunk's ``acc`` is then the committed output of the chunk that wrote it, drawn or not. A unit whose wired row is
  not its writer's is faulty in the wired flattening, so the drawn units' wires are the check; every unit's
  (`wiring.check`) is recorded beside it as the reference (``wiring_reference``), never gating. A sparse-MLA row's combine reads its splits' partials and lses the same way
  (`pipeline.split_rows`, `a5_sparse_mla.py` its stand-in). The negatives (`wiring_negatives`) wire a unit to another unit, swap
  two of a combine's slots, and flip a wired row's string in its reader's public body.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a0  # noqa: E402
import a5_coins as AC  # noqa: E402
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity.ml.circuits.program import Program  # noqa: E402
from verity_numerical.bench.input_sets import InputSet  # noqa: E402
from verity.ml.commitments import row_table as RT  # noqa: E402
from verity.core.protocols.sampled_proofs import coins as CC, service as S  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P, registration as R, verifier as V  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registered as REG  # noqa: E402
from verity.ml.windows import flock_inputs as FI, rows as CS, tables as TB  # noqa: E402
from verity.ml.windows import wiring as WI  # noqa: E402

RANK_PORTS = 16
#: what a plain audit covers until proofs' multi-table coverage lands (verifier PROTOCOL §16.15, `ProvedScope.drawLaw`)
PLAIN_SCOPE = ("plain covers one table (a window of one member, its one session) under a stratified draw, hidden outputs "
               "included (SoundExecH); J tables and the subset draw, hidden outputs past one table, and registered reads and "
               "shared rows wait for proofs' PRs A, B and C")
PLAIN_FIREWALL = "none: C-Flock's firewall checks only a --zk statement's sends (flock-circuit `firewalled`)"


def template_sets(root: Path) -> dict[str, tuple[Path, object]]:
    """Every input set under ``root`` by its template's descriptor id, with its Definition."""
    out = {}
    for d in sorted(p.parent for p in root.rglob("manifest.json")):
        defn = InputSet.open(str(d)).subcircuit.definition()
        out.setdefault(P.template_id(defn), (d, defn))
    return out


def vllm_template(t: str):
    """A chunked member's template the caller gave no composer for: vLLM's serving-rows template ``t``
    (`verity_vllm.pipeline.serving_templates.template`), whose circuit `compose_circuit.py --template` composes."""
    from verity_vllm.pipeline import serving_rows as SRW

    return SRW.template(t).fn


def body(path: Path) -> bytes:
    with open(path, "rb") as f:
        f.readline()
        return f.read()


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src.resolve(), dst)
    except OSError:
        shutil.copyfile(src, dst)


SELECT_COINS = AC.SELECT_COINS
CHALLENGER = "a5.py on the auditor's challenger's coins"
VERIFIER = "a5.py on the verifier's own coins (the Lean's IO.getRandomBytes)"


def stream_of(a: argparse.Namespace, mine: CS.Window, members: list[dict], want: dict, leaf_layer: str | None,
              coins: Callable[[int], bytes]) -> tuple[S.Stream, dict | None]:
    """The rank's `service.Stream` on ``coins`` (the challenger's, a `ChallengerCoins`, or the verifier's own, its
    `challenger` naming which): its unit the window's templates in their order (the one
    template's Definition where the window's query is `Q_template_instance`), the Spec's values the window's ports (a
    row-table window's `row-table` port a tree over its steps' roots, `service.ROOTS` over row-seg leaves), its public
    items what the registration shows (N, each template's stratum, the window, the leaf layer and each port's layout), its
    selection the partition file's rule, and its proof deadline every member's session; -> (the stream, `register`'s counts,
    None for one template's)."""
    defs: dict[str, list] = {}
    for mm in members:
        defs.setdefault(P.template_id(mm["defn"]), [mm["defn"], 0])[1] += mm["m"].n
    params = dict(mine.query).get("params") or {}
    order = params["templates"] if "templates" in params else [params.get("template")]
    if sorted(order) != sorted(defs):
        raise ValueError(f"the window's query names templates {order}, its members {sorted(defs)}")
    defs = {t: defs[t] for t in order}
    units = [d for d, _ in defs.values()]
    one = "templates" not in params
    digests = tuple(P.program_digest(Program(u)) for u in units)
    schemas = {m.port(p): m.schema(p) for m in mine.members for p in CS.port_leaves(m)}
    strata = [s["template"] for s in D.derive(mine.program, mine.query, mine.law, mine.work_table).get("strata") or ()]
    public = [("population", "N, the window's proof units"), *((f"population:{t}", "a template's stratum") for t in strata),
              ("window", "the window's provenance"),
              *([("leaf_layer", "the leaf layer over the public bodies")] if leaf_layer is not None else []),
              *((f"{x}:{p}", "a port's layout") for p in sorted(want) for x in ("leaves", "domain"))]
    values = tuple(S.Value(p, S.ROOTS, "serve", over=RT.SCHEMA) if p == CS.TABLE_PORT else S.Value(p, schemas[p], "serve")
                   for p in sorted(want))
    spec = S.Spec(values, S.Law(a.rule),
                  S.Check(digests[0] if one else digests, mine.query, proof_deadline=(a.session_timeout + 3600) * len(members)),
                  tuple(S.Public(item, "structure", why) for item, why in public), bytes.fromhex(a.window_root))
    name = CHALLENGER if isinstance(coins, CC.ChallengerCoins) else VERIFIER
    stream = S.Stream(spec, units[0] if one else units, budget=1, delta=a.delta, challenger=name, coins=coins)
    return stream, None if one else {t: n for t, (_, n) in defs.items()}


def output_units(a: argparse.Namespace) -> list[int]:
    """The verifier's output class: the units whose outputs the window serves out (``a.outputs``, its own ascending
    ``[lo, hi)`` ranges, from the circuit's registered reads of its output ports), ascending; empty without one."""
    return sorted({u for lo, hi in getattr(a, "outputs", None) or () for u in range(lo, hi)})


def output_closure(a: argparse.Namespace, population: int) -> dict[int, list[int]] | None:
    """The verifier's closure map for its output class (`output_units`): every unit's closure is the whole class, so a
    work draw proves the class whole beside its drawn units (`draw.proved`: drawn ∪ outputs), the rest as drawn; None
    without an output class."""
    outs = output_units(a)
    return {u: outs for u in range(population)} if outs else None


def output_refusal(a: argparse.Namespace, mine: CS.Window, drawn_law: dict) -> dict | None:
    """Why the window's output class can't be proved at rate 1 by `output_closure`, or None: it needs a streamed work
    draw (only a work draw carries a closure), units inside the population, and no member mixing output units with
    others, so each member's session is a subset of all its units or a subset of its drawn ones (SampledKSLiveV2's
    instance kd = n for an output member, the drawn law's for the rest)."""
    outs = set(output_units(a))
    if not outs:
        return None
    if not streamed(a) or drawn_law.get("law") != "work":
        return {"output-class-needs-a-streamed-work-draw": drawn_law.get("law")}
    if max(outs) >= mine.population:
        return {"output-class-outside-the-population": max(outs)}
    mixed = [k for k, m in enumerate(mine.members) if 0 < len(outs & set(range(m.base, m.base + m.n))) < m.n]
    return {"output-class-mixes-a-member": mixed} if mixed else None


def port_refusal(mine: CS.Window, draw: dict, port: int, ports: int | None) -> dict | None:
    """Why the draw's sessions can't all listen inside the ``ports`` reserved from ``port``, or None (and None for
    ``ports`` None, the caller's own ports): each member with a proved unit (`draw.proved`, its closure included) listens
    on the next port from ``port``, so a session past the reservation could reach another audit's verifier."""
    if ports is None:
        return None
    proved = set(D.proved(draw))
    need = sum(1 for m in mine.members if not proved.isdisjoint(range(m.base, m.base + m.n)))
    return {"sessions-past-the-reserved-ports": {"sessions": need, "ports": ports, "from": port}} if need > ports else None


def coins_draw(a: argparse.Namespace, rank: int, rec: dict, mine: CS.Window, members: list[dict], want: dict,
               leaf_layer: str | None) -> tuple[S.Stream | None, S.Draw | None, dict | None, Callable[[int], bytes],
                                                dict | None, dict | None, int | None]:
    """Rank ``rank``'s draw of record (`stream_of`): the stream registers the window under the TP run root (`register`, at
    the window's counts), and draws once (`select`: Lean's `Select.select` on the bytes read, then `check_draw`). On the
    challenger's coins (``a.coins``), the rank first plans its draw's coins from its window (`a5_coins.need`), which opens
    the step's one session once every rank has planned (`a5_coins.Draws.plan`); then the coins commit to the registration's
    digest and the stream reads them in the rank's turn on the step's one client. On the verifier's own
    (``a.verifier_coins``, `a5_coins.VerifierCoins`), the stream's select reads them once it has received the registration
    and written its receipt. The select carries the output class's closure map (`output_closure`). -> (the stream, its
    draw, `register`'s counts, the coins (their `read`: the bytes read), the challenger coins' segment, why it drew none,
    and the challenger coins it planned)."""
    coins = AC.RankCoins(a.coins, rank) if a.coins is not None else AC.VerifierCoins()
    stream = draw = counts = refused = None
    try:
        stream, counts = stream_of(a, mine, members, want, leaf_layer, coins)
        if a.coins is not None:
            coins.plan(AC.need(a.rule, mine.program, mine.query, mine.population, mine.work_table))
        reg = stream.register(rec, want, leaf_layer=leaf_layer, window_root=a.window_root, counts=counts, work_table=mine.work_table)
        if a.coins is not None:
            coins.commit(bytes.fromhex(reg.digest))
        draw = stream.select(reg, closure=output_closure(a, mine.population))
    except S.Refused as e:
        refused = e.verdict.to_json()
    except (ValueError, AC.Ch.CoinError, V.LeanError) as e:
        refused = {"ok": False, "codes": ["stream"], "detail": {"stream": str(e)}}
    finally:
        segment = coins.drawn() if a.coins is not None else None
    return stream, draw, counts, coins, segment, refused, coins.need if a.coins is not None else None


def verifier_outcome(a: argparse.Namespace, stream: S.Stream, draw: S.Draw, counts: dict | None, coins: AC.VerifierCoins,
                     rec: dict, mine: CS.Window, members: list[dict], want: dict, leaf_layer: str | None,
                     sessions: list[str], verdicts: dict, statement: dict, given: dict) -> tuple[dict | None, dict, dict]:
    """The draw's outcome of record on the verifier's own coins: its proof delivered (`Stream.deliver`) and its outcome
    (`Stream.outcome`, ``given`` its proved units, instances, served units and session roots) with no coins' evidence,
    whose audit record fails `run` unless the Lean verifier of record's whole run (`Run.run`, `coin_source` verifier, its
    window the TP run root) is the Python's. Then the refusals that apply to the verifier's coins
    (`a5_coins.verifier_negatives`), each its own stream's on the honest draw's bytes or the honest stream's. -> (the audit
    record, None for a late proof or an outcome the stream refuses or holds pending; the negatives; the coins' record)."""
    def outcome(s: S.Stream, d: S.Draw, evidence: dict | None) -> tuple[S.Outcome, dict | None]:
        late = s.deliver(d, sessions)
        return (late if late is not None else s.outcome(d, verdicts, statement, coins=evidence, **given)), s.records[-1]["audit"]

    try:
        o, result = outcome(stream, draw, None)
        told = {"stream": stream.digest, "record": o.record, "accepted": o.accepted, "codes": list(o.verdict.codes)}
    except (S.Refused, S.Pending) as e:
        result, told = None, {"stream": stream.digest, "accepted": False, "refused": AC._refusal(e)}
    neg = AC.verifier_negatives(
        lambda c: stream_of(a, mine, members, want, leaf_layer, c)[0],
        lambda s: s.register(rec, want, leaf_layer=leaf_layer, window_root=a.window_root, counts=counts, work_table=mine.work_table),
        stream, coins, draw, outcome, a.window_root, output_closure(a, mine.population))
    checks = (result or {}).get("checks") or {}
    record = {"source": stream.coin_source, "challenger": stream.challenger, "read": bytes(coins.read).hex(), "outcome": told,
              "run": None if result is None or not result["verifiers"]["agree"] else checks.get("run", {"ok": True})}
    return result, neg, record


def coins_outcome(a: argparse.Namespace, stream: S.Stream, draw: S.Draw, counts: dict | None, rec: dict, mine: CS.Window,
                  members: list[dict], want: dict, leaf_layer: str | None, read: bytes, segment: dict, sessions: list[str],
                  verdicts: dict, statement: dict, given: dict) -> tuple[dict | None, dict, dict]:
    """The draw's outcome of record once the challenger's session has ended (`a5_coins.Draws.evidence`): its proof delivered
    (`Stream.deliver`) and its outcome (`Stream.outcome`, ``given`` its proved units, instances, served units and session
    roots) with the coins' evidence, whose audit record fails `run` unless the Lean verifier of record's whole run
    (`Run.run`, `coin_source` challenger, its window the TP run root) is the Python's; and `coins.coins_reason` (Python's
    `read_reason`) on the evidence, the audit record failing `coins` where it refuses. Then each negative (`a5_coins.negatives`,
    and the evidence for another window) as its own stream's outcome on the honest draw's bytes (`a5_coins.Replay`), refused
    where `coins_reason` refuses and the Lean's run refuses at its select (`coins`). -> (the audit record, None for a late
    proof or an outcome the stream refuses or holds pending; the negatives; the coins' record)."""
    after, root = bytes.fromhex(R.digest(rec)), a.window_root
    published = a.coins.evidence()
    ev = AC.evidence(published, segment)

    def outcome(s: S.Stream, d: S.Draw, coins: dict | None) -> tuple[S.Outcome, dict | None]:
        late = s.deliver(d, sessions)
        return (late if late is not None else s.outcome(d, verdicts, statement, coins=coins, **given)), s.records[-1]["audit"]

    def refusal(e: Exception) -> dict:
        return e.verdict.to_json() if isinstance(e, S.Refused) else {"pending": e.args[0] if e.args else None} \
            if isinstance(e, S.Pending) else {"stream": str(e)}

    def lean(audit: dict | None) -> dict | None:
        return ((((audit or {}).get("checks") or {}).get("run") or {}).get("detail") or {}).get("disagree", {}).get("lean")

    def reason(coins: dict | None, window: str = root) -> str | None:
        return CC.coins_reason(coins, window, after, read, AC.lms_verify)

    why = published.get("error") if ev is None else reason(ev)
    try:
        o, result = outcome(stream, draw, ev)
        told = {"stream": stream.digest, "record": o.record, "accepted": o.accepted, "codes": list(o.verdict.codes)}
    except (S.Refused, S.Pending) as e:
        result, told = None, {"stream": stream.digest, "accepted": False, "refused": refusal(e)}
    if result is not None and why is not None:
        result = {**result, "accepted": False, "profile": None, "integrity_profile": None, "wrong_units": None,
                  "checks": {**result["checks"], "coins": {"ok": False, "codes": ["coins"], "detail": {"coins": why}}}}
    neg: dict[str, dict] = {}
    bad = {**AC.negatives(ev, a.coins.pre()), "coins-for-another-window": ev} if ev is not None else {}
    for name, coins in bad.items():
        window = "00" * 64 if name == "coins-for-another-window" else root
        s, _ = stream_of(a, mine, members, want, leaf_layer, AC.Replay(read))
        try:
            d = s.select(s.register(rec, want, leaf_layer=leaf_layer, window_root=window, counts=counts, work_table=mine.work_table),
                         closure=output_closure(a, mine.population))
            if d.draw != draw.draw:
                raise ValueError(f"the replayed draw {d.units} is not the honest draw {draw.units}")
            got, audit = outcome(s, d, coins)
            accepted, side = got.accepted, lean(audit)
        except (S.Refused, S.Pending, ValueError, AC.Ch.CoinError) as e:
            accepted, side = None, {"refused": refusal(e)}
        py = reason(coins, window)
        at = ((side or {}).get("step"), (side or {}).get("codes"))
        neg[name] = {"refused": py is not None and accepted is False and at == SELECT_COINS, "accepted": accepted, "python": py,
                     "lean": side}
    checks = (result or {}).get("checks") or {}
    record = {"session": published.get("trusted"), "segment": segment, "read": read.hex(), "coins_reason": why, "outcome": told,
              "run": None if result is None or not result["verifiers"]["agree"] else checks.get("run", {"ok": True})}
    return result, neg, record


def wiring_negatives(members: tuple[CS.Member, ...], publics: dict[str, bytes]) -> dict[str, dict]:
    """The verifier's wiring check (§16.13, `wiring.check_drawn`) on altered wires, each with the units it alters drawn: the first
    wire's unit 0 reading another unit's row, neither its writer's nor its own, of its words where one is (``words``: same,
    else other; rotating the writers is the honest wire at one unit a member or on a fan-out), two wires of one output swapped (a split's partial read into another combine slot), and a wired row's string
    flipped in its reader's public body (a partial that is not the one its writer committed), its unit 0 drawn."""
    neg: dict[str, dict] = {}
    k = next((k for k, m in enumerate(members) if m.wired), None)
    if k is None:
        return neg

    def rewired(j: int, wired: tuple) -> list[CS.Member]:
        return [replace(m, wired=wired) if i == j else m for i, m in enumerate(members)]

    (p, o, ws), *rest = members[k].wired
    def units(j: int) -> range:
        return range(members[j].base, members[j].base + members[j].n)

    words, me = dict(members[k].rows)[p], members[k].base

    def others(same: bool) -> Iterator[int]:
        return (x.base + i for x in members if not same or dict(x.out_rows).get(o) == words for i in range(x.n)
                if x.base + i not in (ws[0], me))

    same = next(others(True), None)
    other = same if same is not None else next(others(False), None)
    if other is not None:
        neg["chunk-wired-to-another-unit (§16.13)"] = {
            "refused": bool(WI.check_drawn(rewired(k, ((p, o, (other, *ws[1:])), *rest)), publics, [me])),
            "words": "same" if same is not None else "other"}
    swap = next(((j, a, b) for j, m in enumerate(members) for a in range(len(m.wired)) for b in range(a + 1, len(m.wired))
                 if m.wired[a][1] == m.wired[b][1] and m.wired[a][2] != m.wired[b][2]), None)
    if swap is not None:
        j, a, b = swap
        w = list(members[j].wired)
        w[a], w[b] = (w[a][0], w[a][1], w[b][2]), (w[b][0], w[b][1], w[a][2])
        neg["partial-wired-to-another-combine-slot (§16.13)"] = {
            "refused": bool(WI.check_drawn(rewired(j, tuple(w)), publics, units(j)))}
    m = members[k]
    names = [q for q, _ in m.all_rows]
    body = bytearray(publics[m.key])
    body[names.index(p) * WI.BC_BYTES] ^= 1
    neg["wired-row-not-its-writer-s-output (§16.13)"] = {
        "refused": bool(WI.check_drawn(members, {**publics, m.key: bytes(body)}, [m.base]))}
    return neg


def registered_negatives(a: argparse.Namespace, mm: dict, sdir: Path) -> dict[str, dict]:
    """A drawn chunk's honest session, a chunk that reads a row table or an outside registered value, re-verified by Lean against
    registered reads it must refuse: none (its header names registered reads the verifier holds no root for), its first port's
    value's root replaced, and (a value of more than one row) every unit reading that value's next row, each with its other
    ports' reads as registered."""
    st, n = mm["st"], mm["m"].n
    honest = json.loads(st["registered_file"].read_text())
    port = next(iter(honest["ports"]))
    p = honest["ports"][port]
    cases = {"chunk-without-registered-reads": (None, "holds no registered roots"),
             "registered-read-against-another-root": ({"root": "00" * 64}, "does not open")}
    if p["leaves"] > 1:
        cases["registered-reads-another-row"] = ({"reads": [(r + 1) % p["leaves"] for r in p["reads"]]}, "the verifier's program reads")
    out = {}
    for name, (change, why) in cases.items():
        alt = {x: y for x, y in st.items() if x != "registered_file"}
        if change is not None:
            alt["registered_file"] = sdir / f"{name}.json"
            alt["registered_file"].write_text(json.dumps({**honest, "ports": {**honest["ports"], port: {**honest["ports"][port], **change}}},
                                                         sort_keys=True))
        v = a0.lean_verify(a.lean, alt, n, sdir, zk=True, log=name)
        out[name] = {"refused": a0.lean_refused(v, why), "why": v.get("why")}
    return out


def stage(served: Path, mm: dict, files: tuple[dict, dict]) -> None:
    """A drawn member's prover's file and its circuit, under the header its public file was held to before the draw."""
    m, st_dir = mm["m"], mm["st"]["dir"]
    if mm["circuit"] != st_dir / "circuit.txt":
        link(mm["circuit"], st_dir / "circuit.txt")
        for f in sorted((mm["circuit"].parent / "tables").glob("*.u32")):    # a circuit with lookups reads the MUFU tables
            (st_dir / "tables").mkdir(exist_ok=True)
            link(f, st_dir / "tables" / f.name)
    got = FI.flock_input_files(served, mm["k"], mm["circ"], st_dir / "prover", write=("inst",), files=files)
    link(Path(got["inst"]), st_dir / f"inst-{m.n}.bin")
    with open(got["inst"], "rb") as f:
        if dict(json.loads(f.readline()), rows=False) != mm["header"]:
            raise RuntimeError(f"member {mm['k']}'s prover's file is not under the header its public file was held to")


def audit(a: argparse.Namespace, sets: dict, out: Path, mine: CS.Window, served: Path, source: Callable[[dict], dict], port: int,
          summary: dict, more_negatives: dict | None = None, rank: int | None = None,
          step_roots: dict[int, bytes] | None = None, registered: dict[str, dict] | None = None,
          ports: int | None = None) -> tuple[dict, bool]:
    """One window's audit: serving's window ``served`` held to the verifier's own ``mine`` (and its registration's ``window`` to
    ``source``; a window with row tables, its tables to its linked members' strings, its registered `row-table` root and
    ``step_roots``, what serving recorded before each step's coin, `tables.check_row_tables`), one Lean draw by its law
    (on a stream, `streamed`: TP rank ``rank``'s draw of record by its `service.Stream`, on the challenger's coins
    ``a.coins`` or the verifier's own, `coins_draw`), a `--zk` session per member with drawn units on ports from ``port``,
    Lean's verdicts, the audit record (on a stream, its outcome: on the coins' evidence, `coins_outcome`, or on the
    verifier's own coins, `verifier_outcome`) and the negatives (``more_negatives`` beside them), all under ``out``; -> its
    summary and whether it was accepted with every negative refused. ``sets``: each template's input set and Definition, or
    a callable composing its circuit (``(out, partition digest, program digest) -> {"pin", "class_sha512", ...}``) and its
    Definition, a member of bit rows too; a bit-row member's template with no composer is vLLM's (`vllm_template`). ``registered``: by member key, the verifier's own ``--registered`` file of a member reading an outside
    registered value, at the root its owner signed (a warden's cells), which the caller read before calling. ``ports``: how
    many ports from ``port`` the caller reserved, a draw whose sessions need more refused before any listens
    (`port_refusal`); None for no reservation to hold them to."""
    out.mkdir(parents=True, exist_ok=True)
    program, query, part, pd, n_all, law = mine.program, mine.query, mine.partition, mine.partition_digest, mine.population, mine.law
    table = mine.work_table
    if (why := plain_refusal(a, mine, registered)) is not None:
        raise SystemExit(f"--plain: {why}")
    if not streamed(a) and law.get("law") != "subset":
        raise SystemExit(f"a5 draws by the subset law with Lean's own coins; the partition file's is {law}: draw it through "
                         "a rank's service.Stream (--challenger or --coins verifier, a TP Commit)")
    drawn_law = D.derive(program, query, law, table)
    if (why := output_refusal(a, mine, drawn_law)) is not None:
        summary["refused"] = why
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
        print("A5 " + json.dumps(why))
        return summary, False
    want = CS.expected_ports(mine)
    summary.update({"run_root": mine.run, "population": n_all, "partition": pd, "law": law, "delta": a.delta, "firewall": a.firewall,
                    "members": [{"k": k, "template": m.template, "n": m.n, "base": m.base} for k, m in enumerate(mine.members)]})
    if plain(a):
        summary.update(firewall=None, plain={"scope": PLAIN_SCOPE, "coins": "os", "firewall": PLAIN_FIREWALL})
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(canonical_json(encode_program(program)))

    rec = json.loads((served / "registration.json").read_text())
    shutil.copy(served / "registration.json", out / "registration.json")

    # each member: the verifier's circuit, the public file under it, and every check on that file before the draw; the prover's
    # file and the circuit's link only for a drawn member, after the draw (`stage`)
    files = (json.loads((served / "window.json").read_text()), rec)
    roots = {r["port"]: r for r in rec["roots"]}
    members, publics, checks, composed, programs_ok = [], {}, {}, {}, {}
    t_stage = time.time()
    for k, m in enumerate(mine.members):
        own = callable(sets.get(m.template, (None,))[0])  # the caller's composer, chunked or not
        chunked = bool(m.schemas) and not own
        if m.template not in sets and not chunked:
            summary["refused"] = {"no-template-set": m.template}
            (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
            print("A5 " + json.dumps(summary["refused"]))
            return summary, False
        defn = vllm_template(m.template) if chunked else sets[m.template][1]
        st_dir = out / f"m{k}"
        st_dir.mkdir(parents=True, exist_ok=True)
        t = time.time()
        if m.template in composed:      # a template's members: one circuit, composed from the same input set, partition and program
            circuit, made, circ = composed[m.template]
        elif own:
            circuit = st_dir / "circuit.txt"
            made = sets[m.template][0](circuit, pd, part["program"])
            circ = FI.flock_circuit(circuit.read_bytes())
            composed[m.template] = (circuit, made, circ)
        else:
            src = ["--template", m.template, *(["--cache", a.compose_cache] if a.compose_cache else [])] if chunked else \
                ["--input-set", str(sets[m.template][0])]
            p = a0.run([sys.executable, str(HERE / "compose_circuit.py"), *src, "--out", str(st_dir / "circuit.txt"),
                        "--partition", pd, "--program", part["program"]], st_dir / "compose.log", {"PYTHONPATH": a.m0_pythonpath},
                       timeout=a.session_timeout)
            if p.returncode != 0:
                raise RuntimeError(f"composing member {k}'s circuit failed: {p.stderr[-2000:]}")
            circuit, made = st_dir / "circuit.txt", json.loads(p.stdout.strip().splitlines()[-1])
            circ = FI.flock_circuit(circuit.read_bytes())
            composed[m.template] = (circuit, made, circ)
        compose_s = round(time.time() - t, 1)
        got = FI.flock_input_files(served, k, circ, st_dir / "prover", write=("pub",), files=files)
        link(Path(got["pub"]), st_dir / f"pub-{m.n}.bin")
        meta = circ.meta
        header = a0.public_header(st_dir, m.n)
        fv = header["frame_v3"]
        if (m.template, circ.pin) not in programs_ok:   # a function of the circuit's META and the template: once per circuit
            programs_ok[m.template, circ.pin] = a0.template_program_ok(meta, defn)
        ok = {"pin": header.get("circuit_sha512") == made["pin"],
              "partition": meta.get("partition") == {"rule": P.FORMAT, "digest": pd},
              "program": meta.get("program_sha512") == part["program"],
              "indices": header["units"]["indices"] == list(range(m.base, m.base + m.n)),
              "classes": header["units"].get("classes") == [made.get("class_sha512")] * m.n,
              "template_program": programs_ok[m.template, circ.pin],
              "shared_rows": header.get("shared_rows") == ({**dict(m.tables), **{p: m.n for p, _ in m.out_rows}}
                                                           if m.tables else None)}
        for p_name in fv["roots"]:
            name = m.port(p_name)
            reg = roots.get(name)
            ok[f"root:{p_name}"] = (name in want and reg is not None and reg["root"] == fv["roots"][p_name]
                                    and reg["domain"] == fv["domain_ids"][p_name] == want[name]["domain"])
        checks[f"m{k}"] = {"ok": all(ok.values()), "codes": [c for c, v in ok.items() if not v], "detail": {}}
        publics[m.key] = body(st_dir / f"pub-{m.n}.bin")
        members.append({"k": k, "m": m, "defn": defn, "header": header, "circuit": circuit, "circ": circ,
                        "st": {"dir": st_dir, "meta": meta, "pin": made["pin"], "partition_file": out / "partition.json",
                               "program_file": out / "program.json"},
                        "compose_s": compose_s, "pin": made["pin"], "class": made.get("class_sha512")})
    summary["staging"] = {"before_draw_s": round(time.time() - t_stage, 1), "public_files": len(members)}
    leaf_layer = CS.leaf_layer(publics)
    checks["registration"] = V.check(rec, program, query, law, CS.SCHEME, want, leaf_layer=leaf_layer, work_table=table).to_json()
    if mine.row_tables:
        bad = TB.check_row_tables(mine, publics, {r["port"]: r["root"] for r in rec["roots"]}, step_roots)
        checks["row_tables"] = {"ok": not bad, "codes": sorted(bad), "detail": bad}
    if any(m.wired for m in mine.members):
        bad = WI.check(mine.members, publics)
        summary["wiring_reference"] = {"ok": not bad, "codes": sorted(bad), "detail": bad}
    w = rec.get("window", {})
    checks["source"] = source(w)
    summary["source"], summary["commitment"] = w.get("kind"), w.get("commitment")
    receipt = R.receipt(rec, "one-stage-e2e verifier (a5.py)")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    summary.update(registration=R.digest(rec), leaf_layer=leaf_layer, checks_before_draw={c: v["ok"] for c, v in checks.items()},
                   circuits={mm["m"].template: {"pin": mm["pin"], "class": mm["class"], "compose_s": mm["compose_s"],
                                                "k_log": mm["st"]["meta"].get("k_log"), "unit_log": mm["st"]["meta"].get("unit_log")}
                             for mm in members if mm["circuit"].parent == mm["st"]["dir"]})
    if not all(v["ok"] for v in checks.values()):
        summary["checks"] = checks
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
        print("A5 " + json.dumps({"refused_before_draw": summary["checks_before_draw"], "checks": checks}, default=str))
        return summary, False
    # each row-table chunk's Lean verify reads its x rows against its table's root, which the checks above tied to the registration;
    # a member reading an outside registered value, against the root its owner signed
    reads = {**(TB.registered_reads(mine, publics, registered=REG) if mine.row_tables else {}), **(registered or {})}
    for mm in members:
        if mm["m"].key in reads:
            mm["st"]["registered_file"] = mm["st"]["dir"] / "registered.json"
            mm["st"]["registered_file"].write_text(json.dumps(reads[mm["m"].key], sort_keys=True))

    # one draw over the whole population; each member serves its share under --zk
    if not streamed(a):
        p = a0.run([a.lean, "draw", "--population", str(n_all), "--k", str(law["k"])], out / "draw.log")
        union_draw = json.loads(next(x for x in p.stdout.splitlines() if x.startswith("DRAW ")).split(" ", 1)[1])
    else:
        stream, draw, counts_of, coins, segment, refused, need = coins_draw(a, rank, rec, mine, members, want, leaf_layer)
        read = bytes(coins.read)
        (out / "segment.json").write_text(json.dumps({"source": "challenger" if a.coins is not None else "verifier",
                                                      "segment": segment, "read": read.hex(), "need": need}, sort_keys=True))
        if draw is None:
            summary.update(checks={**checks, "draw": refused})
            (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
            print("A5 " + json.dumps({"draw_refused": refused}, default=str))
            return summary, False
        union_draw, drawn_law, receipt = dict(draw.draw), dict(draw.law), dict(draw.registration.receipt)
        (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    (out / "draw.json").write_text(json.dumps(union_draw, sort_keys=True))
    after: dict[str, dict] = {}
    if any(m.wired for m in mine.members):
        bad = WI.check_drawn(mine.members, publics, union_draw["units"])
        after["wiring"] = {"ok": not bad, "codes": sorted(bad), "detail": bad}
        if bad:
            summary.update(checks={**checks, **after})
            (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
            print("A5 " + json.dumps({"refused_after_draw": {"wiring": False}, "checks": after}, default=str))
            return summary, False
    served_units, proved, instances, verdicts, statement, sessions, counts, session_roots = [], [], 0, {}, {}, [], {}, {}
    linked = None       # the first drawn chunk with registered reads: its session, for the registered-reads negatives
    row_verdicts: dict[str, dict] = {}      # each drawn chunk's Lean verdict with registered reads, whose finding the record carries
    if (why := port_refusal(mine, union_draw, port, ports)) is not None:
        summary["refused"] = why
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        print("A5 " + json.dumps(why))
        return summary, False
    drawn_units, drawn = set(union_draw["units"]), []
    for mm in members:
        m = mm["m"]
        local = sorted(u - m.base for u in D.proved(union_draw) if m.base <= u < m.base + m.n)
        counts[m.template] = counts.get(m.template, 0) + sum(1 for u in local if m.base + u in drawn_units)
        if local:
            drawn.append((mm, local))
    archived: dict = {}     # the members' circuit, archived once for every Lean verify (`a0.lean_verify`)
    for mm in members:
        mm["st"]["archived"] = archived

    def member_session(i: int, mm: dict, local: list[int]) -> dict:
        """Member ``mm``'s firewalled session on its proved units ``local`` (drawn, and a work draw's closure), on port
        ``port + i`` (``i`` its place among the members with proved units, `port_refusal`); `member_checks` adds Lean's
        verdicts on it."""
        m, k = mm["m"], mm["k"]
        sdir = out / f"session-m{k}"
        sdir.mkdir(parents=True, exist_ok=True)
        # a plain session is served the stream's stratified draw itself, its one member the whole window (`plain_refusal`)
        session_draw, session_law = ((union_draw, drawn_law) if plain(a) else
                                     ({"law": "subset", "population": m.n, "k": len(local), "units": local}, {"law": "subset", "k": len(local)}))
        (sdir / "draw.json").write_text(json.dumps(session_draw, sort_keys=True))
        t = time.time()
        stage(served, mm, files)
        staged_s = time.time() - t
        t = time.time()
        env = {"FC_COINS": "os"} if plain(a) else {"FC_FIREWALL": "1", "FC_FIREWALL_LEAN": a.firewall, **a0.firewall_env(mm["st"])}
        record, live = a0.session(a.bin, mm["st"], m.n, f"subset:{len(local)}", port + i, sdir, draw_file=sdir / "draw.json",
                                  zk=not plain(a), env=env, prove_timeout=a.session_timeout)
        session_s = round(time.time() - t, 1)
        draw_t = (record or {}).get("unit_draw") or {}
        v = V.check_draw(draw_t, {"population": session_draw["population"]}, session_law)
        return {"mm": mm, "sdir": sdir, "staged_s": staged_s, "session_s": session_s, "record": record, "live": live,
                "draw_t": draw_t, "draw_check": v}

    def member_checks(s: dict) -> dict:
        """Lean's verdicts on member session ``s`` (`member_session`) and what the audit takes from it. They read only the
        session's archive, so they run while the next member is proved."""
        mm, sdir, record, live, draw_t, v = s["mm"], s["sdir"], s["record"], s["live"], s["draw_t"], s["draw_check"]
        m, k, session_s = mm["m"], mm["k"], s["session_s"]
        lean_u = a0.lean_draw_test(a.lean, [{"record": draw_t or None, "statement": (live or {}).get("unit_draw"),
                                             "instances": (live or {}).get("instances", 0)}], sdir, "u1-u3")
        got = {f"m0-verifier/m{k}": {"accepted": ((record or {}).get("verdict") or {}).get("accepted") is True
                                     and (live or {}).get("roots") == mm["header"]["frame_v3"]["roots"] and v.ok,
                                     "why": ((record or {}).get("verdict") or {}).get("why"), "draw_check": v.to_json()},
               f"lean-draw/m{k}": {"accepted": bool(lean_u) and all(c["ok"] for c in lean_u), "detail": lean_u}}
        if plain(a):
            flags = ("--coins", "os", "--stratified", str(drawn_law["k"]))
            got[f"lean-verify/m{k}"] = {**a0.lean_verify(a.lean, mm["st"], m.n, sdir, flags=flags),
                                        "tool": f"flock-verify verify {' '.join(flags)} --archive"}
        else:
            got[f"lean-verify/m{k}"] = {**a0.lean_verify(a.lean, mm["st"], m.n, sdir, zk=True), "tool": "flock-verify verify --zk --archive"}
        lean_v = got[f"lean-verify/m{k}"]
        return {"mm": mm, "sdir": sdir, "staged_s": s["staged_s"], "verdicts": got, "lean_verify": lean_v,
                "served": [m.base + u for u in draw_t.get("units", []) if m.base + u in drawn_units],  # a closure's are `proved`
                "roots": {m.port(p_name): r for p_name, r in ((live or {}).get("roots") or {}).items()},
                "proved": a0.proved_units(mm["header"], live or {}), "instances": (live or {}).get("instances", 0),
                "statement": {"session_s": session_s, **{x: (live or {}).get(x) for x in ("circuit_sha512", "statement_digest", "sigma")},
                              "timing": {x: (live or {}).get(x) for x in ("e2e_s", "witness_s", "prove_total_s", "k_log", "units", "proof_bytes")},
                              "verify_s": ((record or {}).get("verdict") or {}).get("verify_s"),
                              "lean_verify_s": lean_v.get("verify_s"), "lean_partition": lean_v.get("partition")},
                "sessions": [a0.sha512_file(f) for f in sorted((sdir / "sessions").glob("*/session.json"))]}

    # outer proofs go through the firewall one at a time (`ONE_AT_A_TIME`): the sessions run here in the members' order, and
    # only Lean's checks of one member, outside the firewall, overlap the next member's session
    with ThreadPoolExecutor(1) as lean:
        checks = [lean.submit(member_checks, member_session(i, mm, local)) for i, (mm, local) in enumerate(drawn)]
        done = [c.result() for c in checks]
    for got in done:
        mm = got["mm"]
        summary["staging"]["drawn_s"] = round(summary["staging"].get("drawn_s", 0) + got["staged_s"], 1)
        summary["staging"]["prover_files"] = summary["staging"].get("prover_files", 0) + 1
        served_units += got["served"]
        session_roots.update(got["roots"])
        proved += got["proved"]
        instances += got["instances"]
        verdicts.update(got["verdicts"])
        if mm["m"].key in reads:
            row_verdicts[mm["m"].key] = got["lean_verify"]
            linked = linked or (mm, got["sdir"])
        statement[mm["m"].key] = got["statement"]
        sessions += got["sessions"]
    coin_neg: dict[str, dict] = {}
    if not streamed(a):
        result = V.audit_record(rec, receipt, union_draw, drawn_law, sorted(proved), instances, statement, verdicts, a.delta,
                                sessions=sessions, served=sorted(served_units), session_roots=session_roots)
        write_case(out, program, query, law, want, leaf_layer, a.delta, table, rec,
                   {"draw": union_draw, "proved_units": sorted(proved), "receipt": receipt, "statement": statement,
                    "verdicts": verdicts}, instances=instances, sessions=sessions, session_roots=session_roots,
                   served=sorted(served_units))
    else:  # every verdict of record names the sessions delivered (`service.check_verdicts`)
        verdicts = {x: {**y, "sessions": sessions} if y.get("of_record") else y for x, y in verdicts.items()}
        given = {"proved_units": sorted(proved), "instances": instances, "served": sorted(served_units),
                 "session_roots": session_roots}
        if a.coins is not None:
            result, coin_neg, summary["coins"] = coins_outcome(
                a, stream, draw, counts_of, rec, mine, members, want, leaf_layer, read, segment, sessions, verdicts, statement,
                given)
        else:
            result, coin_neg, summary["coins"] = verifier_outcome(
                a, stream, draw, counts_of, coins, rec, mine, members, want, leaf_layer, sessions, verdicts, statement, given)
        if result is None:
            summary.update(audit={"accepted": False, "outcome": summary["coins"]["outcome"]}, draw=union_draw["units"],
                           negatives={x: y["refused"] for x, y in coin_neg.items()})
            (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
            print("A5 " + json.dumps({"no_audit_record": summary["coins"]["outcome"]}, default=str))
            return summary, False
        result["coins"] = {"source": stream.coin_source, "segment": segment, "read": read.hex(),
                           "outcome": summary["coins"]["outcome"]}
        write_case(out, program, query, law, want, leaf_layer, a.delta, table, rec,
                   {"draw": union_draw, "proved_units": given["proved_units"], "receipt": receipt, "statement": statement,
                    "verdicts": verdicts}, instances=instances, sessions=sessions, session_roots=session_roots,
                   served=given["served"])
    result["draw_counts"] = counts
    if row_verdicts:
        result["registered_reads"] = TB.reported_reads(reads, row_verdicts)
    result["checks_before_draw"] = checks
    if after:
        result["checks_after_draw"] = after
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))

    # negatives: each re-decides one input altered (``a.skip_negatives``: not this audit; a run of several audits runs them
    # once, in its first)
    neg: dict[str, dict] = dict(more_negatives or {})
    if getattr(a, "skip_negatives", False):
        neg.update(coin_neg)
        summary["negatives_skipped"] = True
    else:
        by_template: dict[str, list] = {}
        for mm in members:
            by_template.setdefault(mm["m"].template, [mm["defn"], 0])[1] += mm["m"].n
        defs = [tuple(x) for x in by_template.values()]
        fewer = resized_program(defs) if len(defs) > 1 or defs[0][1] == 1 \
            else P.population_program(defs[0][0], defs[0][1] - 1)
        v = V.check(rec, fewer, query, law, CS.SCHEME, want, leaf_layer=leaf_layer, work_table=table)
        neg["registration-partition-mismatch (R2)"] = {"refused": not v.ok, "codes": list(v.codes)}
        other = CS.expected_ports(replace(mine, run="00" * 32))
        v = V.check(rec, program, query, law, CS.SCHEME, other, leaf_layer=leaf_layer, work_table=table)
        neg["domains-bound-to-another-run-root (R3)"] = {"refused": not v.ok, "codes": list(v.codes)}
        first = members[0]["m"].key
        flipped = dict(publics, **{first: publics[first][:-1] + bytes([publics[first][-1] ^ 1])})
        v = V.check(rec, program, query, law, CS.SCHEME, want, leaf_layer=CS.leaf_layer(flipped), work_table=table)
        neg["public-body-not-the-registered-leaf-layer (R4)"] = {"refused": not v.ok, "codes": list(v.codes)}
        v = V.check_draw({**union_draw, "units": union_draw["units"][:-1] + [n_all]}, rec, drawn_law)
        neg["draw-outside-the-partition"] = {"refused": not v.ok, "codes": list(v.codes)}
        neg.update(wiring_negatives(mine.members, publics))
        if proved:
            v = V.check_proved(union_draw, proved[1:], len(proved) - 1)
            neg["proved-set-short-of-the-draw"] = {"refused": not v.ok, "codes": list(v.codes)}
        neg.update(output_negatives(a, union_draw, sorted(proved)))
        neg.update(transfer_gate_negatives(getattr(a, "transfer_gate", ()), rec, mine, want, leaf_layer, union_draw, drawn_law, proved))
        neg.update(a0.audit_negatives(result, drawn_law, a.delta, session_roots, served=sorted(served_units)))
        neg.update(coin_neg)
        if linked is not None:
            neg.update(registered_negatives(a, *linked))
            key = next(iter(row_verdicts))
            other = {p: {**e, "root": "00" * 64} for p, e in (row_verdicts[key].get("registered") or {}).items()}
            neg["verdict-reports-another-registered-root"] = {
                "refused": not TB.reported_reads(reads, {**row_verdicts, key: {**row_verdicts[key], "registered": other}})["ok"]}
    (out / "negatives.json").write_text(json.dumps(neg, indent=1, sort_keys=True))

    summary["audit"] = {x: result.get(x) for x in ("accepted", "complete", "wrong_units")}
    if "registered_reads" in result:
        summary["audit"]["registered_reads"] = result["registered_reads"]["ok"]
    summary["audit"]["verdicts"] = {x: y.get("accepted") for x, y in verdicts.items()}
    summary["audit"]["why"] = {x: y.get("why") for x, y in verdicts.items() if y.get("accepted") is not True}
    summary.update(draw=union_draw["units"], draw_counts=counts, served_is_lean_draw=sorted(served_units) == sorted(union_draw["units"]),
                   timing=statement, negatives={x: y["refused"] for x, y in neg.items()})
    outs = output_units(a)
    if outs:
        summary["outputs"] = {"units": len(outs), "proved": set(outs) <= set(proved) <= set(D.proved(union_draw)),
                              "drawn": len(drawn), "closure": len(union_draw.get("closure", ()))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    ok = bool(result.get("accepted")) and result.get("registered_reads", {"ok": True})["ok"] and all(summary["negatives"].values()) \
        and summary.get("outputs", {"proved": True})["proved"]
    shown = {x: summary.get(x) for x in ("source", "commitment", "audit", "draw", "draw_counts", "negatives", "outputs")}
    if "coins" in summary:
        shown["coins"] = {x: summary["coins"].get(x) for x in ("source", "coins_reason", "outcome", "run")}
    print("A5 " + json.dumps(shown, sort_keys=True, default=str))
    return summary, ok


def output_negatives(a: argparse.Namespace, draw: dict, proved: list[int]) -> dict[str, dict]:
    """A window with an output class (`output_units`): the proved set less an output unit the draw didn't draw (one its
    closure proves), refused (`check_proved`); and the draw with its closure dropped, held to the proved set, refused."""
    outs = output_units(a)
    left = next((u for u in outs if u not in set(draw["units"])), None)
    if left is None:
        return {}
    short = [u for u in proved if u != left]
    v = V.check_proved(draw, short, len(short))
    neg = {"output-unit-left-out-of-the-proved-set": {"refused": not v.ok, "codes": list(v.codes), "unit": left}}
    v = V.check_proved({x: y for x, y in draw.items() if x != "closure"}, proved, len(proved))
    neg["draw-without-the-output-closure"] = {"refused": not v.ok, "codes": list(v.codes)}
    return neg


def transfer_gate_negatives(transfer_gates: Sequence[str], rec: dict, mine: CS.Window, want: dict, leaf_layer: str, draw: dict,
                            drawn_law: dict, proved: list[int]) -> dict[str, dict]:
    """A window whose registered compiler lists ``transfer_gates`` (`advice_windows`: every unit of theirs proved, an All stratum of
    the work law beside the drawn rest): each way a transfer-gate unit is left out of the proved set, refused. The proved set less
    one (`check_proved`); a draw that trades one for an undrawn unit of the rest, the same count (`check_draw`: its stratum draws
    every unit); and a registration whose law draws the transfer-gate templates as the rest is, work 1 and floor 1 (R1: the
    verifier's own law, under its own work table, proves them whole)."""
    gates = [u for s in drawn_law.get("strata") or () if s["template"] in set(transfer_gates) for lo, hi in s["units"]
             for u in range(lo, hi)]
    if not gates:
        return {}
    neg: dict[str, dict] = {}
    left = [u for u in proved if u != gates[0]]
    v = V.check_proved(draw, left, len(left))
    neg["transfer-gate-unit-left-out-of-the-proved-set"] = {"refused": not v.ok, "codes": list(v.codes), "unit": gates[0]}
    drawn = set(draw["units"])
    if rest := [u for u in range(draw["population"]) if u not in drawn]:
        v = V.check_draw({**draw, "units": sorted(drawn - {gates[0]} | {rest[0]})}, rec, drawn_law)
        neg["draw-trading-a-transfer-gate-unit-for-another (check_draw)"] = {"refused": not v.ok, "codes": list(v.codes)}
    sampled = {t: {"work": 1, "floor": 1} if t in transfer_gates else w for t, w in (mine.work_table or {}).items()}
    v = V.check({**rec, "law": D.derive(mine.program, mine.query, mine.law, sampled)}, mine.program, mine.query, mine.law, CS.SCHEME,
                want, leaf_layer=leaf_layer, work_table=mine.work_table)
    neg["registration-drawing-the-transfer-gates-as-the-rest (R1)"] = {"refused": not v.ok, "codes": list(v.codes)}
    return neg


def write_case(out: Path, program, query: dict, law: dict, ports: dict, leaf_layer: str | None, delta: float,
               work_table: dict | None, rec: dict, record: dict, *, instances: int, sessions: list[str],
               session_roots: dict, served: list[int]) -> None:
    """The audit as `SampledProofs.OneStage`'s case, what onsite's evidence carries for sampled proofs
    (`replay.Case.case`: the registration record, the evidence `audit_record` was handed and the verifier's params, δ the
    audit's as an exact ratio), in ``out/case.json``, and today's Python on it (`replay.python`) in
    ``out/case-python.json``."""
    try:
        import replay as RP

        d = str(Fraction(delta).limit_denominator(10**12))
        p = RP.params(program, query, law, ports, leaf_layer, d, work_table=work_table)
        e = RP.evidence(record, instances=instances, sessions=sessions, session_roots=session_roots, served=served)
        c = RP.Case("", "a5", rec["format"], "audit", "audit", program, p, rec, e, None)
        (out / "case.json").write_text(json.dumps(c.case(), sort_keys=True))
        (out / "case-python.json").write_text(json.dumps(RP.python(c), sort_keys=True, default=str))
    except Exception as x:  # noqa: BLE001: the case is a copy of the audit for onsite; its failure leaves the audit's verdict
        (out / "case-error.txt").write_text(f"{type(x).__name__}: {x}\n")


def streamed(a: argparse.Namespace) -> bool:
    """Each rank draws through its own `service.Stream`: on the auditor's challenger's coins (``--challenger``) or on the
    verifier's own (``--coins verifier``)."""
    return a.coins is not None or a.verifier_coins


#: AGENTS.md's live-coins rule, which keeps an audit's drawn members' sessions one after another
ONE_AT_A_TIME = "outer proofs go through the firewall one at a time (AGENTS.md, \"Live coins only\")"


def one_session(s: str) -> int:
    """``--session-jobs``: 1, the only value, kept so commands that pass it still run; anything else is refused."""
    if s.strip() != "1":
        raise argparse.ArgumentTypeError(f"{s}: {ONE_AT_A_TIME}, so an audit runs its sessions one at a time; pass 1 or omit it")
    return 1


def plain(a: argparse.Namespace) -> bool:
    """The sessions are plain (``--plain``), not `--zk`."""
    return bool(getattr(a, "plain", False))


def plain_refusal(a: argparse.Namespace, mine: CS.Window, registered: dict | None) -> str | None:
    """Why a plain audit (``a.plain``) of ``mine`` is outside `PLAIN_SCOPE`, refused before the draw; None inside it (and for
    a `--zk` audit)."""
    if not plain(a):
        return None
    m = mine.members[0] if len(mine.members) == 1 else None
    rules = [(streamed(a), "a plain audit draws through a rank's service.Stream (--challenger or --coins verifier)"),
             (mine.law.get("law") == "stratified",
              f"a plain session's draw is stratified (Lean refuses a plain {mine.law.get('law')} draw, ProvedScope.drawLaw)"),
             (m is not None, f"plain covers one table: the window has {len(mine.members)} members"),
             (m is None or (m.base == 0 and m.n == mine.population), "plain covers one table: its member is the whole window"),
             (not (mine.row_tables or registered or (m is not None and (m.tables or m.wired))),
              "registered reads, shared rows and wires wait for proofs' PR C")]
    why = [w for ok, w in rules if not ok]
    return "; ".join(why) + f" ({PLAIN_SCOPE})" if why else None


def resized_program(defs: list[tuple]) -> Program:
    """R2's other partition: the window's population, `(template, n)` per template, with its first template one instance
    short, or one over where it has a single instance (a population has at least one)."""
    resized = [(d, (n - 1 if n > 1 else 2) if i == 0 else n) for i, (d, n) in enumerate(defs)]
    return P.mixed_population_program(resized) if len(resized) > 1 else P.population_program(*resized[0])


def deferred(a: argparse.Namespace, sets: dict, out: Path) -> bool:
    from verity_vllm.pipeline import deferred_rows as DRW, replay_bundle as RB

    t0 = time.time()
    r = DRW.of_record(a.bundle, a.row_dir)
    mine = DRW.window_of(r, a.partition)
    manifest = hashlib.sha256((Path(a.bundle) / RB.MANIFEST).read_bytes()).hexdigest()

    weights_root = str((r.row["commit"].get("weight_registration") or {}).get("root") or "")

    def source(w: dict) -> dict:
        src = DRW.source(w, mine.run, manifest, r.commitment, weights_root)
        return {"ok": all(src.values()), "codes": [c for c, v in src.items() if not v],
                "detail": {"kind": w.get("kind"), "run": w.get("run"), "commitment": w.get("commitment"), "of": w.get("of")}}

    _summary, ok = audit(a, sets, out, mine, Path(a.window), source, a.port, {"verifier_window_s": round(time.time() - t0, 2)})
    return ok


def advice_negatives(a: argparse.Namespace, r: int, compiler: AW.Traced, mine: CS.Window, held: dict, served: Path,
                     opening: dict) -> dict[str, dict]:
    """An advice window's refusals, each before the draw: its registration's window naming another compiler or another advice
    root (`source`); the advice's opening under another salt, which opens no registered root (`source`); the verifier opening other
    advice (``--other-advice-opening``), whose window from it the registration fails (`source` and R2); and a unit committed under
    other advice (``--other-windows``: member 0's public file from that window in place of serving's), which the registration's
    leaf layer refuses (R4) and whose roots are not the registered ones."""
    from verity_vllm.pipeline import advice_windows as AW
    from verity_vllm.pipeline.tp import serving_rows as SRT

    rec = json.loads((served / "registration.json").read_text())
    w = rec.get("window") or {}
    neg: dict[str, dict] = {}
    for key, field in (("compiler", "digest"), ("advice", "root")):
        v = SRT.source({**w, key: {**(w.get(key) or {}), field: "00" * 64}}, mine, held)
        neg[f"registration-naming-another-{key} (source)"] = {"refused": not v["ok"], "codes": v["codes"]}
    if AW.transfer_gates_of(compiler):
        bare = replace(compiler, transfer_gates=())
        v = SRT.source({**w, "compiler": {**(w.get("compiler") or {}), "digest": bare.digest}}, mine, held)
        neg["registration-under-the-compiler-listing-no-transfer-gates (source)"] = {"refused": not v["ok"], "codes": v["codes"]}
    salts = list(opening["salts"])
    salts[0] = f"{int(salts[0][:2], 16) ^ 1:02x}{salts[0][2:]}"
    _reqs, resalted = AW.of_opening(compiler, {**opening, "salts": salts})
    v = SRT.source(w, mine, {**held, **resalted})
    neg["opening-under-another-salt (source)"] = {"refused": not v["ok"], "codes": v["codes"]}
    publics = {m.key: (served / f"m{k}-public.bin").read_bytes() for k, m in enumerate(mine.members)}
    if a.other_advice_opening:
        reqs, bound = AW.of_opening(compiler, json.loads(Path(a.other_advice_opening).read_text()))
        theirs, theirs_held = AW.window_of_advice(a.tp_commit, r, a.partition, compiler, reqs, bound)
        src = SRT.source(w, theirs, theirs_held)
        v = V.check(rec, theirs.program, theirs.query, theirs.law, CS.SCHEME, CS.expected_ports(theirs), leaf_layer=CS.leaf_layer(publics),
                    work_table=theirs.work_table)
        neg["advice-that-disagrees (source, R2)"] = {"refused": not (src["ok"] and v.ok), "codes": src["codes"] + list(v.codes),
                                                     "population": {"registered": rec.get("population"), "of_the_advice": theirs.population}}
    if a.other_windows:
        od, first = Path(a.other_windows) / f"rank{r}", mine.members[0]
        spliced = dict(publics, **{first.key: (od / "m0-public.bin").read_bytes()})
        v = V.check(rec, mine.program, mine.query, mine.law, CS.SCHEME, CS.expected_ports(mine), leaf_layer=CS.leaf_layer(spliced),
                    work_table=mine.work_table)
        registered = {x["port"]: x["root"] for x in rec["roots"]}
        roots = json.loads((od / "window.json").read_text())["members"][0]["frame_v3"]["roots"]
        roots_ok = all(registered.get(first.port(p)) == root for p, root in roots.items())
        neg["unit-committed-under-other-advice (R4, roots)"] = {"refused": not (v.ok and roots_ok),
                                                                "codes": list(v.codes) + ([] if roots_ok else ["roots"])}
    return neg


def tp_rank(a: argparse.Namespace, sets: dict, out: Path, world: int, r: int) -> dict:
    """Rank ``r``'s window of record (``tp.serving_rows.window_of_record``; with ``--advice-opening``, the compiler's Programs for
    the advice it opens, ``advice_windows.window_of_advice``, and `advice_negatives`) audited under ``out/rank<r>``, on its own ports; -> its
    row of the summary. With ``a.coins``, a rank that ends without its draw (refused before it, or failed) says so, so the
    step's session closes once every other rank has drawn."""
    from verity_vllm.pipeline import advice_windows as AW
    from verity_vllm.pipeline.tp import serving_rows as SRT

    t0 = time.time()
    try:
        windows = Path(a.windows)
        more: dict[str, dict] = {}
        if a.advice_opening:
            compiler = AW.Traced.for_commit(a.tp_commit, a.build_root, a.transfer_gate)
            opening = json.loads(Path(a.advice_opening).read_text())
            reqs, bound = AW.of_opening(compiler, opening)
            mine, held = AW.window_of_advice(a.tp_commit, r, a.partition, compiler, reqs, bound)
            more.update(advice_negatives(a, r, compiler, mine, held, windows / f"rank{r}", opening))
        else:
            mine, held = SRT.window_of_record(a.tp_commit, r, a.partition, a.build_root, a.workload)
        window_s = round(time.time() - t0, 2)
        theirs = windows / f"rank{(r + 1) % world}" / "registration.json"
        if world > 1 and theirs.exists():
            v = SRT.source(json.loads(theirs.read_text()).get("window"), mine, held)
            more["registration-of-another-rank (source)"] = {"refused": not v["ok"], "codes": v["codes"]}
        s, ok = audit(a, sets, out / f"rank{r}", mine, windows / f"rank{r}", lambda w: SRT.source(w, mine, held),
                      a.port + RANK_PORTS * r, {"rank": r, "tp": held["tp"], "verifier_window_s": window_s,
                                                **({"binding": {k: held[k] for k in SRT.BINDING}} if a.advice_opening else {})}, more, rank=r)
    finally:
        if a.coins is not None:
            a.coins.skip(r)
    return {"ok": ok, "seconds": round(time.time() - t0, 1),
            **{x: s.get(x) for x in ("run_root", "population", "members", "registration", "source", "refused", "checks_before_draw",
                                     "audit", "draw", "draw_counts", "negatives", "verifier_window_s", "staging", "coins", "binding")},
            **({"plain": s["plain"]} if "plain" in s else {})}


def tp_rank_job(a: argparse.Namespace, out: Path, world: int, r: int) -> dict:
    return tp_rank(a, template_sets(Path(a.template_sets)), out, world, r)


def tp_ranks(a: argparse.Namespace, sets: dict, out: Path, world: int, ranks: list[int]) -> Iterator[tuple[int, dict]]:
    """`tp_rank` of each of ``ranks``, in up to ``a.jobs`` spawned processes (a rank's window holds a Program that does not pickle, so
    a process does the whole rank); -> (rank, row) as each finishes."""
    if a.jobs <= 1 or len(ranks) <= 1:
        for r in ranks:
            yield r, tp_rank(a, sets, out, world, r)
        return
    with ProcessPoolExecutor(min(a.jobs, len(ranks)), mp_context=multiprocessing.get_context("spawn")) as ex:
        jobs = {ex.submit(tp_rank_job, a, out, world, r): r for r in ranks}
        for f in as_completed(jobs):
            yield jobs[f], f.result()


def tp(a: argparse.Namespace, sets: dict, out: Path) -> bool:
    """Every rank's window of the TP Commit ``--tp-commit`` (or the ranks ``--ranks``), each audited on its own under ``out/rank<r>``,
    and ``out/summary.json`` over them; a rank's audit also refuses another rank's registration as its own."""
    from verity_vllm.pipeline.tp import serving_rows as SRT

    rec = SRT.record(a.tp_commit)
    world = int(rec["world"])
    ranks = [int(x) for x in a.ranks.split(",")] if a.ranks else list(range(world))
    summary: dict = {"tp_run_root": rec["tp_run_root"], "world": world, "roots": rec["roots"], "ranks": {}}
    ok_all = True
    if a.challenger:
        if len(ranks) > 1 and a.jobs < len(ranks):
            raise SystemExit(f"--challenger draws every rank from the step's one session, which ends once every rank has drawn: "
                             f"audit the {len(ranks)} ranks at once (--jobs {len(ranks)}), or fewer ranks (--ranks)")
        a.window_root = rec["tp_run_root"]
        manager, a.coins, finisher = AC.start(a.challenger, a.window_root, ranks, out / "challenger", a.main,
                                              ssh_key=a.gate_key, peer=a.peer, key_record=a.challenger_key)
        summary["challenger"] = {"challenger": a.challenger, "rule": a.rule}
    elif a.verifier_coins:
        a.window_root = rec["tp_run_root"]
        summary["coins"] = {"source": "verifier", "stream": VERIFIER, "rule": a.rule,
                            "what": "each rank's service.Stream on the verifier's own live coins (verifier.coins, the Lean's "
                                    "IO.getRandomBytes), read once the stream received the rank's registration; no auditor's "
                                    "challenger drew them"}
    try:
        for r, row in tp_ranks(a, sets, out, world, ranks):
            summary["ranks"][str(r)] = row
            ok_all = ok_all and row["ok"]
            (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    finally:
        if a.coins is not None:
            for r in ranks:
                a.coins.skip(r)
            finisher.join()
            published = a.coins.evidence()
            summary["challenger"].update(trusted=published.get("trusted"), error=published.get("error"),
                                         **(published.get("plan") or {}))
            manager.shutdown()
    summary["accepted"] = ok_all and sorted(summary["ranks"]) == sorted(str(r) for r in ranks) \
        and not summary.get("challenger", {}).get("error")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    print("A5-TP " + json.dumps({"tp_run_root": rec["tp_run_root"], "accepted": summary["accepted"],
                                 "ranks": {r: {x: s.get(x) for x in ("ok", "population", "draw_counts", "seconds")} for r, s in summary["ranks"].items()}},
                                sort_keys=True, default=str))
    return summary["accepted"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", help="a deferred Commit's sealed replay bundle (<commit out>/replay_bundle_p<pair>)")
    ap.add_argument("--row-dir", help="with --bundle: the row dir of the Build's Programs of record")
    ap.add_argument("--window", help="with --bundle: serving's window (verity-vllm serving-rows --out)")
    ap.add_argument("--tp-commit", help="a TP Commit's record (its --out: runs.jsonl, versions.json)")
    ap.add_argument("--build-root", help="with --tp-commit: each rank's Programs of record in rank<r>/")
    ap.add_argument("--workload", help="with --tp-commit: the workload file the Commit served")
    ap.add_argument("--windows", help="with --tp-commit: serving's windows, rank<r>/ each (the --serving-rows request's dir)")
    ap.add_argument("--advice-opening", help="with --tp-commit, instead of --workload: the opening of the advice's hiding commitment "
                    "(advice_windows.opening: I1's canonical encoding and its rows' salts, the named gap); each rank's window is the "
                    "registered compiler's Programs for it (advice_windows.window_of_advice), its registration naming the compiler's "
                    "digest and the advice's root")
    ap.add_argument("--compiler", choices=["traced"], default="traced", help="with --advice-opening: the registered compiler (traced: the "
                    "stand-in, each rank's Programs of record under --build-root, advice_windows.Traced)")
    ap.add_argument("--transfer-gate", action="append", default=[], help="with --advice-opening: a template the registered compiler "
                    "lists as holding transfer gates, every unit of which is proved (repeatable; the partition's law is work:K)")
    ap.add_argument("--other-advice-opening", help="with --advice-opening: the opening of advice that disagrees with the registered, "
                    "given to the verifier (a negative)")
    ap.add_argument("--other-windows", help="with --advice-opening: the windows committed under the other advice, rank<r>/ each (a "
                    "negative: a unit committed under other advice)")
    ap.add_argument("--ranks", default=None, help="with --tp-commit: the ranks to audit, comma-separated (default every rank)")
    ap.add_argument("--jobs", type=int, default=1, help="with --tp-commit: audit up to this many ranks at once, each in its own process")
    ap.add_argument("--partition", required=True, help="the verifier's partition file: {templates, law, scope?}")
    ap.add_argument("--template-sets", required=True, help="one input set per member template, for the lowering")
    ap.add_argument("--m0-pythonpath", required=True)
    ap.add_argument("--compose-cache", help="this machine's composed circuits (`compose_circuit.py --cache`): the ranks it audits "
                    "compose each chunked template once, every later rank taking that circuit with its own binding")
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--firewall", required=True,
                    help="Lean's flock-firewall: every session's prover runs behind its send check (FC_FIREWALL=1, live PROTOCOL.md §10.4)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--port", type=int, default=7551)
    ap.add_argument("--session-jobs", type=one_session, default=1, help="1 only: the sessions run one at a time (ONE_AT_A_TIME)")
    ap.add_argument("--session-timeout", type=float, default=14400,
                    help="each drawn member's prover limit in seconds, and each template's compose: one firewalled session of a MoE "
                         "w2 coordinate chunk (MoeExpertFp8GemmW_v2{N=1}, k_log 26) takes ~8 min, and over an hour on a loaded "
                         "machine, as does composing the 64-key SparseMlaSplitPart_v1 in lookup slots")
    ap.add_argument("--challenger", help="with --tp-commit: draw each rank from the auditor's challenger's coins, one session on the TP "
                    "run root (a5_coins): local (a development session on this machine), or gate:USER@HOST (vy-challenger-1 through "
                    "infra's gate, with --peer and --challenger-key)")
    ap.add_argument("--peer", help="with --challenger gate:USER@HOST: this machine as the gate's /etc/vy/peers names it")
    ap.add_argument("--challenger-key", help="with --challenger gate:USER@HOST: the challenger's key record of its deployment "
                    "(infra's challenger-key/v1), never the session's key.json")
    ap.add_argument("--gate-key", help="with --challenger gate:USER@HOST: the lane's own gate key (default ssh's own)")
    ap.add_argument("--main", help="with --challenger: the verifier's own copy of main's first-parent history, a commit a line "
                    "(`git rev-list --first-parent origin/main`), never the host's or the prover's")
    ap.add_argument("--coins", dest="coin_source", choices=["verifier"], help="with --tp-commit, instead of --challenger: draw each rank through its "
                    "own service.Stream on the verifier's own live coins (the service's default, the Lean's IO.getRandomBytes), "
                    "read once the stream received the rank's registration; no challenger session")
    ap.add_argument("--plain", action="store_true", help="with --challenger or --coins verifier: plain sessions, not --zk, "
                    "under the partition's stratified:K law, a window of one member with no registered reads or shared rows "
                    "(PLAIN_SCOPE)")
    a = ap.parse_args()
    if bool(a.bundle) == bool(a.tp_commit):
        ap.error("give either --bundle (a deferred Commit) or --tp-commit (a TP Commit)")
    need = ("row_dir", "window") if a.bundle else ("build_root", "windows", *(() if a.advice_opening else ("workload",)))
    if missing := [f"--{x.replace('_', '-')}" for x in need if not getattr(a, x)]:
        ap.error(f"{'--bundle' if a.bundle else '--tp-commit'} needs {' '.join(missing)}")
    if a.bundle and a.advice_opening:
        ap.error("--advice-opening draws a TP Commit's ranks (--tp-commit)")
    if (a.other_advice_opening or a.other_windows or a.transfer_gate) and not a.advice_opening:
        ap.error("--other-advice-opening, --other-windows and --transfer-gate are an advice window's: give --advice-opening")
    a.coins, a.rule, a.window_root, a.verifier_coins = None, None, None, a.coin_source == "verifier"
    if a.challenger and a.verifier_coins:
        ap.error("--challenger and --coins verifier are two coin sources: give one")
    if a.plain and not (a.challenger or a.verifier_coins):
        ap.error("--plain draws through a rank's service.Stream: give --challenger or --coins verifier")
    if a.challenger or a.verifier_coins:
        flag = "--challenger" if a.challenger else "--coins verifier"
        if a.bundle:
            ap.error(f"{flag} draws a TP Commit's ranks (--tp-commit)")
        if a.challenger and a.challenger.startswith("gate:") and not (a.peer and a.challenger_key):
            ap.error("--challenger gate:USER@HOST needs --peer and --challenger-key")
        if a.challenger and not a.main:
            ap.error("--challenger needs --main: the verifier's `git rev-list --first-parent origin/main`, which must hold the "
                     "session's pin")
        a.rule = json.loads(Path(a.partition).read_text()).get("law")
        if not isinstance(a.rule, str):
            ap.error(f"{flag} draws by the partition file's rule, a law's text (stratified:K, subset:K, ...), not {a.rule!r}")
        S.Law(a.rule)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = template_sets(Path(a.template_sets))
    return 0 if (deferred(a, sets, out) if a.bundle else tp(a, sets, out)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
