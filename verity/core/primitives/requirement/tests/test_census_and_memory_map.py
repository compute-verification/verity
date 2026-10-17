"""`requirement.from_census` and memory-map/v1's builder (`verity_vllm.commit.memory_map`) held to `core_requirement`'s Lean
through the difftest's instance format (`test_requirement_difftest.encode`), Python's reference agreeing with Lean on each.

The census path: a census and a schedule as an instance (`census_execution`): the census's values, then a value of 0 bits
for each root Call a step runs, whose operands are the values the Call reads or selects. Lean's live values and bits are
`from_census`'s at every step, its reads `from_census`'s where those are known and, the selected values used too,
`reads_upper`'s; and `pack`'s layout and store of `from_census`'s requirement, with the blocks each step reads as its plan,
are W's at every step. On the toy decoder, `b1.Serve` at TINY and `b1.RMSNormFusedCudaV1`, whose live and read bits are
the gate path's.

The memory map: a session's KV as an instance (`map_case`). Its values are the rows `MM.rows` finds, each born at its
request's arrival step plus its own and read at the steps the requirement over the census reads it, in the block the
builder wrote it to (the record's store index, holding the bytes the record's digest is of). The store is the record's
fresh blocks, written and released under the 21:52Z convention: step s's record (its frees, then its writes) takes effect
after step s's reads, so `Store.resident` at t reads the records before t. Step t's plan is the blocks the replayed map
holds then (`Replay`, for a plan holding every live block). The session is the builder test's: a small Llama's rows of
1 KiB in pages of 4 slots, so a page's plane is one 4 KiB block; r0 (a prefill of 4 and 5 decode steps) from step 0, and
r1 (a prefill of 4 and a decode step) from step 1 or 3.

`StoreMatches` and `PlanMatches` hold exactly at the steps where no live row is in a block the map doesn't hold and the
map holds no block whose rows are all dead. Two things break them. memory-map/v1 writes a block once complete, so a page's
first rows are live but in no block until its last is born. And the builder frees a block in the record after its rows'
last read, which 21:52Z reads as one step late; moved to its last read's record, the free is on time.
"""

from __future__ import annotations

import hashlib
from itertools import count

import pytest

import verity_difftest as D
from test_requirement_difftest import PORT, _lean, encode, rmsnorm, serve, toy_decoder
from verity.core.protocols.remote_bandwidth import codec
from verity.core.protocols.remote_bandwidth.memory_map import Replay, Step
from verity.ml.circuits import requirement as R
from verity.ml.circuits.census import census, step_ends
from verity.ml.circuits.program import REGISTRATION, Program, is_input

server = D.server_fixture(PORT)


# --- the census path ---------------------------------------------------------------------------------------------------

def census_of(program: Program, schedule: R.Schedule) -> dict:
    """`program`'s census, its steps ending where `schedule`'s do."""
    nodes, off = program.root.body.nodes, program.root.body.offsets
    I = sum(1 for nd in nodes if is_input(nd.fn))
    steps = [schedule.call_step(off[k], off[k + 1]) for k in range(I, len(nodes))]
    return census(program, [I + i + 1 for i in range(len(steps)) if i + 1 == len(steps) or steps[i + 1] != steps[i]])


def census_execution(c: dict, schedule: R.Schedule, upper: bool) -> R.Execution:
    """The census `c` run on `schedule` as an instance's execution: census value x is value x, of its bits, registered,
    produced at its producer's step, or a run input arriving at the first step that reads or selects it; then each root
    Call a step runs is a value of 0 bits at that step, whose operands are the values it reads or selects and which uses
    those it reads (and, `upper`, those it selects)."""
    vals, I, off = c["values"], c["inputs"], R.census_offsets(c)
    at = {k: s for k in range(I, len(off) - 1)
          if off[k] < off[k + 1] and (s := schedule.call_step(off[k], off[k + 1])) is not None}
    reads: dict[int, set[int]] = {}
    selects: dict[int, set[int]] = {}
    for x, ks in enumerate(vals["reads"]):
        for k in ks:
            reads.setdefault(k, set()).add(x)
    for k, xs in zip(c["selects"]["call"], c["selects"]["values"]):
        selects.setdefault(k, set()).update(xs)
    binding = [c["classes"][c["callees"][k]]["binding"] for k in c["nodes"]["callee"][:I]]
    registered = frozenset(x for x, j in enumerate(vals["producer"]) if j < I and binding[j] == REGISTRATION)
    step: dict[int, int] = {}
    for x, j in enumerate(vals["producer"]):
        touched = [s for k, s in at.items() if x in reads.get(k, set()) | selects.get(k, set())]
        if j >= I and j in at:
            step[x] = at[j]
        elif j < I and x not in registered and touched:
            step[x] = min(touched)
    width = dict(enumerate(vals["bits"]))
    ops: dict[int, tuple[int, ...]] = {}
    uses: dict[int, frozenset[int]] = {}
    for g, (k, s) in enumerate(sorted(at.items()), start=len(width)):
        width[g], step[g] = 0, s
        ops[g] = tuple(sorted(reads.get(k, set()) | selects.get(k, set())))
        uses[g] = frozenset(reads.get(k, set()) | (selects.get(k, set()) if upper else set()))
    return R.Execution(width, registered, step, ops, uses, {})


def census_case(c: dict, schedule: R.Schedule, block_bits: int, upper: bool) -> dict:
    """`census_execution`'s instance under `pack`'s layout and store of `from_census`'s requirement, each step's plan the
    blocks of the reads the instance states."""
    ex = census_execution(c, schedule, upper)
    layout, store = R.pack(R.from_census(c, schedule), block_bits)
    stated = R.Requirement.of(ex)
    return encode(ex, layout, store, [R.read_blocks(stated, layout, t) for t in range(stated.steps)], block_bits,
                  stated.steps)


def held_census(program: Program, schedule: R.Schedule, block_bits: int, server) -> R.Requirement:
    """`from_census`'s requirement of `program` on `schedule`, which Lean's outputs on its census path are: its live values
    and bits, its reads where they are known and, the selected values used, `reads_upper`; the store and plans W's."""
    c = census_of(program, schedule)
    req = R.from_census(c, schedule)
    for upper in (False, True):
        out = _lean(census_case(c, schedule, block_bits, upper), server)
        assert out is not None and all(s["store_matches"] and s["plan_matches"] for s in out["steps"])
        assert len(out["steps"]) == req.steps and out["held"] == req.held(0, req.steps)
        for t, s in enumerate(out["steps"]):
            assert (s["live"], s["live_bits"]) == (sorted(req.live(t)), req.live_bits(t))
            if upper:
                assert (s["reads"], s["read_bits"]) == (sorted(req.reads_upper(t)), req.read_bits_upper(t))
            elif req.resolved(t):
                assert (s["reads"], s["read_bits"]) == (sorted(req.reads(t)), req.read_bits(t))
    return req


def _the_gate_paths(req: R.Requirement, program: Program, schedule: R.Schedule, inputs: dict[int, int]) -> None:
    """The census path's live bits are the gate path's with the run's values, and so are its read bits where known."""
    gates = R.requirement(program, schedule, inputs)
    assert [req.live_bits(t) for t in range(req.steps)] == [gates.live_bits(t) for t in range(gates.steps)]
    assert all(req.read_bits(t) == gates.read_bits(t) for t in range(req.steps) if req.resolved(t))


@pytest.mark.parametrize("B", [128, R.BLOCK_BITS])
def test_the_toy_decoders_census_path(B, server):
    """`test_requirement.py`'s numbers: the live bits of the gate path, and every step's reads bounded by the whole
    embedding, since a census holds no run values to resolve its gathers."""
    program, schedule, inputs = toy_decoder()
    req = held_census(program, schedule, B, server)
    assert not any(req.resolved(t) for t in range(req.steps))
    assert [req.live_bits(t) for t in range(3)] == [320, 384, 416] == [req.read_bits_upper(t) for t in range(3)]
    _the_gate_paths(req, program, schedule, inputs)


def test_the_tiny_serve_rows_census_path(server):
    """Each weight is one census value, up to 66,048 bits, and the layout keeps a value in one block, so the census path
    takes blocks of 2^17 bits. Every step's reads are known (its selections' candidates are read anyway), and its live
    and read bits are the gate path's: `test_the_tiny_serve_row`'s weights less the RoPE rows no step reaches."""
    program, schedule, inputs = serve()
    req = held_census(program, schedule, 1 << 17, server)
    assert all(req.resolved(t) for t in range(req.steps))
    assert req.live_bits(0) == 16 * (program.input_gates - 3) - 3 * 16 * 16
    _the_gate_paths(req, program, schedule, inputs)


def test_the_rmsnorm_rows_census_path(server):
    program, schedule, inputs = rmsnorm()
    _the_gate_paths(held_census(program, schedule, R.BLOCK_BITS, server), program, schedule, inputs)


# --- the memory map ----------------------------------------------------------------------------------------------------

#: the builder test's model: vLLM's predict tests' TINY Llama with 512 hidden and 4 KV heads, rows of 1 KiB
WIDE = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 512, "intermediate_size": 128,
        "num_attention_heads": 4, "num_key_value_heads": 4, "num_hidden_layers": 2, "vocab_size": 256,
        "rms_norm_eps": 1e-5, "tie_word_embeddings": True, "max_position_embeddings": 64, "hidden_act": "silu"}
BLOCK_SIZE = 4


def _payload(r: str, row) -> bytes:
    return hashlib.shake_256(repr((r, row.group)).encode()).digest(1024)


@pytest.fixture(scope="module")
def requests() -> dict[str, Program]:
    from verity_vllm.predict.build import build_program
    from verity_vllm.predict.phi import Phi

    phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": WIDE},
              target={"compute_capability": [12, 0], "num_sms": 188}, workload={"requests": []})
    return {"r0": build_program(phi, "request", LP=4, T=5).program, "r1": build_program(phi, "request", LP=4, T=1).program}


def row_reads(program: Program):
    """A KV row's group -> the request steps that read it, and those whose selections may: the requirement over the
    Program's census at its sampling events, which `MM.rows` takes, at the census values the row's leaves are."""
    from verity_vllm.query.required import SAMPLING_EVENTS

    inputs = sum(1 for nd in program.root.body.nodes if is_input(nd.fn))
    c = census(program, step_ends(program, SAMPLING_EVENTS))
    req, v = R.from_census(c), c["values"]

    def of(group) -> tuple[set[int], set[int]]:
        xs = {x for _, j, lo, hi in group for x, k in enumerate(v["producer"])
              if k == j + inputs and v["lo"][x] < hi and v["hi"][x] > lo}
        return {s for s, u in req.uses.items() if xs & u}, {s for s, u in req.upper.items() if xs & u}

    return of


def at_last_read(steps: list[Step]) -> list[Step]:
    """The builder's record with each free moved one record earlier, to the step of its rows' last read (21:52Z)."""
    freed = {s.step - 1: s.freed for s in steps if s.freed}
    written = {s.step: s.written for s in steps if s.written}
    return [Step(t, freed.get(t, ()), written.get(t, ())) for t in sorted(freed.keys() | written.keys())]


def map_case(requests: dict[str, Program], arrive: dict[str, int], moved: bool) -> tuple[dict, dict]:
    """The session's KV under the builder's record (`moved`: its frees at their last reads, `at_last_read`) as an instance,
    and what the test reads beside it: each row value's request and `MM.Row`, each value's block, and the blocks the
    replayed map holds at each step."""
    from verity_vllm.commit import memory_map as MM
    from verity_vllm.store.interface import PageLayout
    from verity_vllm.store.registry import _page_blocks

    rows = {r: MM.rows(p) for r, p in requests.items()}
    head, steps = MM.record(rows, arrive, _payload, bytes(range(64)), MM.Layout(block_size=BLOCK_SIZE))
    steps = at_last_read(steps) if moved else steps
    end = max(arrive[r] + rs.steps for r, rs in rows.items())
    assert all(s.step < end for s in steps)
    by, slots = {s.step: s for s in steps}, {x for s in steps for x in (*s.freed, *(w.slot for w in s.written))}
    replay, written, freed, index, held = Replay(head), {}, {}, {}, []
    for t in range(end):
        held.append(frozenset(j for x in slots if (j := replay.holds(x)) is not None and j >= head.blocks))
        if t in by:
            change = replay.apply(by[t])
            written[t] = frozenset(w.index for w in by[t].written)
            freed[t] = frozenset(j for f, n in change.released for j in range(f, f + n) if j >= head.blocks)
            index.update(((t, w.slot), w) for w in by[t].written)

    (nl, sb), = {(len(rs.layers), rs.row_bytes) for rs in rows.values()}
    tables = MM.tables_of(rows, arrive, BLOCK_SIZE)
    ids, lo, hi = _page_blocks(PageLayout(1 + max(max(t) for t in tables.values()), BLOCK_SIZE, sb), codec.BLOCK_BYTES)
    per_plane, region = ids.shape[1] // 2, int(ids.max()) + 1
    unwritten = count(replay.extent)
    width, step, value, block, offset, payload, of_value = {}, {}, {}, {}, {}, {}, {}
    ops: dict[tuple[str, int], set[int]] = {}
    uses: dict[tuple[str, int], set[int]] = {}
    for r, rs in rows.items():
        reads = row_reads(requests[r])
        for q, page in enumerate(tables[r]):
            for c in range(ids.shape[1]):
                plane, i = divmod(c, per_plane)
                p0, p1 = q * BLOCK_SIZE + int(lo[c]), q * BLOCK_SIZE + int(hi[c])
                off = i * codec.BLOCK_BYTES - int(lo[c]) * sb
                for layer, kv in enumerate(rs.layers):
                    got = kv[plane][p0:p1]
                    w = index.get((arrive[r] + max(x.born for x in got), layer * region + int(ids[page, c]))) \
                        if p1 <= rs.positions else None
                    j = next(unwritten) if w is None else w.index
                    if w is not None:
                        data = b"".join(_payload(r, x) for x in got)[off:off + codec.BLOCK_BYTES]
                        assert codec.digest(data) == w.digest
                        payload[j] = int.from_bytes(data, "little")
                    for u, x in enumerate(got):
                        v = len(width)
                        width[v], step[v], of_value[v] = 8 * sb, arrive[r] + x.born, (r, x)
                        value[v] = int.from_bytes(_payload(r, x), "little")
                        block[v], offset[v] = j, 8 * (u * sb - off)
                        read, selected = reads(x.group)
                        for s in read | selected:
                            ops.setdefault((r, s), set()).add(v)
                        for s in read:
                            uses.setdefault((r, s), set()).add(v)
    ex_ops: dict[int, tuple[int, ...]] = {}
    ex_uses: dict[int, frozenset[int]] = {}
    for g, ((r, s), vs) in enumerate(sorted(ops.items()), start=len(width)):
        width[g], step[g], value[g] = 0, arrive[r] + s, 0
        ex_ops[g], ex_uses[g] = tuple(sorted(vs)), frozenset(uses.get((r, s), set()))
    ex = R.Execution(width, frozenset(), step, ex_ops, ex_uses, value)
    store = R.Store(0, written, freed, payload)
    instance = encode(ex, R.Layout(block, offset), store, held, codec.BLOCK_BYTES * 8, end)
    return instance, {"rows": of_value, "block": block, "held": held, "arrive": arrive}


def _verdicts(out: dict, session: dict) -> tuple[list[bool], dict[int, set[int]], dict[int, set[int]]]:
    """Each step's verdict, as `StoreMatches` and `PlanMatches` both give it; and, read off `MM.rows` and the replayed map
    beside Lean, the live rows in no block the map holds and the blocks it holds with every row dead, by step."""
    rows, block, held, arrive = session["rows"], session["block"], session["held"], session["arrive"]
    live = lambda t: {v for v, (r, x) in rows.items() if arrive[r] + x.born < t <= arrive[r] + x.last}  # noqa: E731
    missing, dead = {}, {}
    for t, s in enumerate(out["steps"]):
        assert s["live"] == s["reads"] == sorted(live(t)) and s["resident"] == sorted(held[t])
        assert s["store_matches"] == s["plan_matches"]
        missing[t] = {v for v in live(t) if block[v] not in held[t]}
        dead[t] = {j for j in held[t] if not any(block[v] == j for v in live(t))}
        assert s["store_matches"] == (not missing[t] and not dead[t])
    return [s["store_matches"] for s in out["steps"]], missing, dead


@pytest.mark.parametrize("r1,moved,want,late", [
    (1, False, [True, True, False, False, False, True], {3}),
    (1, True, [True, True, False, False, False, True], set()),
    (3, False, [True, True, False, False, False, False], {5}),
    (3, True, [True, True, False, False, False, True], set()),
])
def test_the_memory_map(r1, moved, want, late, requests, server):
    """r0's second page (positions 4 to 7, born at steps 1 to 4) is written at step 4, so its first rows are live from
    step 2 in no block the map holds, through step 4: those are the only rows missing. r1's first page is freed in the
    record after its last read, so the map holds it one step with every row dead (step 3, beside r0's missing rows, or
    step 5, alone, with r1 from step 3); with its free at its last read's record, it holds none."""
    value, session = map_case(requests, {"r0": 0, "r1": r1}, moved)
    out = _lean(value, server)
    assert out is not None
    verdicts, missing, dead = _verdicts(out, session)
    assert verdicts == want
    assert {t for t, vs in missing.items() if vs} == {2, 3, 4}
    assert {(session["rows"][v][0], session["rows"][v][1].born) for vs in missing.values() for v in vs} == \
        {("r0", 1), ("r0", 2), ("r0", 3)}
    assert {t for t, js in dead.items() if js} == late
