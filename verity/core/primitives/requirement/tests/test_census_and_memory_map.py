"""`requirement.from_census` and memory-map/v1's builder (`verity_vllm.commit.memory_map`) held to `core_requirement`'s Lean
through the difftest's instance format (`test_requirement_difftest.encode`), Python's reference agreeing with Lean on each.

The census path: a census and a schedule as an instance (`census_execution`): the census's values, then a value of 0 bits
for each root Call a step runs, whose operands are the values the Call reads or selects. Lean's live values and bits are
`from_census`'s at every step, its reads `from_census`'s where those are known and, the selected values used too,
`reads_upper`'s; and `pack`'s store placement and store of `from_census`'s requirement, with the blocks each step reads
as its plan, are W's at every step, crediting every live value. On the toy decoder, `b1.Serve` at TINY and
`b1.RMSNormFusedCudaV1`, whose live and read bits are the gate path's.

The memory map: a session's KV as an instance (`map_case`). Its values are the rows `MM.rows` finds, each born at its
request's arrival step plus its own and read at the steps the requirement over the census reads it, in the block the
builder wrote it to (the record's store index, holding the bytes the record's digest is of; a page the builder never
writes, a request's last and partial one, in a block of its own). The store is the record's fresh blocks, written and
released under the 21:52Z convention: step s's record (its frees, then its writes) takes effect after step s's reads, so
`Store.resident` at t reads the records before t. Step t's plan is the blocks the replayed map holds then (`Replay`, for a
plan holding every live block). The session is the builder test's: a small Llama's rows of 1 KiB in pages of 4 slots, so
a page's plane is one 4 KiB block; r0 (a prefill of 4 and 5 decode steps) from step 0, and r1 (a prefill of 4 and a
decode step) from step 1 or 3.

memory-map/v1 writes a block once complete, so a page's first rows are live but in no block until its last is born: the
audit credits only the live rows in complete blocks, and `StoreMatches` and `PlanMatches` hold exactly at the steps where
every credited row is in a block the map holds; a block with no credited row may be held too, and earns W nothing. A
row's group is its request, layer and plane, and the rows the audit doesn't credit lie in one block per group
(`OneBlockEach`). The builder frees each block in the record of its rows' last structural reader, a gather that may
select one included (`test_a_block_a_gather_may_select_stays_resident_through_the_last_step_that_may`, where a free at
their last use breaks `StoreMatches`), and leaves out a block no step reads while it is resident
(`test_a_block_its_last_reader_completes_is_in_no_record`): the link holds at every step, and the map holds no block
with every row dead, which the tests check beside the link. Its coins come back before the next
step's reads; with them back later (`deferred`, memory-map/v1's deferred residency) a block is written in the record of
the step before its first encoded read and credited from that read, a block whose rows are dead by then is withdrawn,
and the rows the audit doesn't credit lie in one block per group and page parity.
"""

from __future__ import annotations

import copy
import hashlib
from itertools import chain, count

import pytest

import verity_difftest as D
from test_requirement_difftest import PORT, _lean, encode, rmsnorm, serve, toy_decoder
from verity.core.protocols.remote_bandwidth import codec
from verity.core.protocols.remote_bandwidth.memory_map import Replay
from verity.ml.circuits import requirement as R
from verity.ml.circuits.census import census, step_ends, step_of
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
    inputs = frozenset(x for x, j in enumerate(vals["producer"]) if j < I and x not in registered and x in step)
    return R.Execution(width, registered, step, ops, uses, {}, inputs=inputs)


def census_case(c: dict, schedule: R.Schedule, block_bits: int, upper: bool) -> dict:
    """`census_execution`'s instance under `pack`'s store placement and store of `from_census`'s requirement, each step's
    plan the blocks of the reads the instance states (all in complete blocks under `pack`)."""
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
        assert all(s["credited"] == s["live"] and s["credited_reads"] == s["reads"] for s in out["steps"])
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
    """Each weight is one census value, up to 66,048 bits, and the store placement keeps a value in one block, so the
    census path takes blocks of 2^17 bits. Every step's reads are known (its selections' candidates are read anyway), and
    its live and read bits are the gate path's: `test_the_tiny_serve_row`'s weights less the RoPE rows no step reaches."""
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
    from verity_vllm.program.predict.build import build_program
    from verity_vllm.program.predict.phi import Phi

    phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": WIDE},
              target={"compute_capability": [12, 0], "num_sms": 188}, workload={"requests": []})
    return {"r0": build_program(phi, "request", LP=4, T=5).program, "r1": build_program(phi, "request", LP=4, T=1).program,
            "r1_page": build_program(phi, "request", LP=4, T=4).program}


def census_values(program: Program, c: dict, group) -> set[int]:
    """The census values a KV row's leaves (its group) are."""
    inputs, v = sum(1 for nd in program.root.body.nodes if is_input(nd.fn)), c["values"]
    return {x for _, j, lo, hi in group for x, k in enumerate(v["producer"])
            if k == j + inputs and v["lo"][x] < hi and v["hi"][x] > lo}


def row_reads(program: Program, c: dict | None = None):
    """A KV row's group -> the request steps that read it, and those whose selections may: the requirement over the
    Program's census `c` (at its sampling events when not given), which `MM.rows` takes, at the census values the row's
    leaves are."""
    from verity_vllm.query.required import SAMPLING_EVENTS

    c = census(program, step_ends(program, SAMPLING_EVENTS)) if c is None else c
    req = R.from_census(c)

    def of(group) -> tuple[set[int], set[int]]:
        xs = census_values(program, c, group)
        return {s for s, u in req.uses.items() if xs & u}, {s for s, u in req.upper.items() if xs & u}

    return of


def selected_at(c: dict, values: set[int], step: int) -> dict:
    """Census `c` with each Call of `step` that reads one of `values` selecting it instead, a gather's candidate (the
    census's `selects`, a new entry's `gates` and `bits` a stand-in's, which the requirement doesn't read)."""
    c = copy.deepcopy(c)
    at, reads, sel = step_of(c), c["values"]["reads"], c["selects"]
    pools = {k: set(vs) for k, vs in zip(sel["call"], sel["values"])}
    for x in values:
        for k in [k for k in reads[x] if at(k) == step]:
            reads[x].remove(k)
            pools.setdefault(k, set()).add(x)
    old, ks = dict(zip(sel["call"], zip(sel["gates"], sel["bits"]))), sorted(pools)
    c["selects"] = {"call": ks, "gates": [old.get(k, (1, 0))[0] for k in ks], "bits": [old.get(k, (1, 0))[1] for k in ks],
                    "values": [sorted(pools[k]) for k in ks]}
    return c


def first_need(rows: dict, arrive: dict[str, int]) -> dict[str, list[int]]:
    """Each request's pages as an engine hands them out, from page 1 in the order the session first needs them (a page's
    first row's step, then its request), through `MM.tables` from the slot mapping they give, as the Commit reads them."""
    from verity_vllm.commit import memory_map as MM

    need = sorted((arrive[r] + rs.layers[0][0][q].born, r, q // BLOCK_SIZE)
                  for r, rs in rows.items() for q in range(0, rs.positions, BLOCK_SIZE))
    pages: dict[str, list[int]] = {r: [] for r in rows}
    for page, (_, r, _) in enumerate(need, start=1):
        pages[r].append(page)

    def slots(r: str, step: int) -> list[int]:
        return [pages[r][q // BLOCK_SIZE] * BLOCK_SIZE + q % BLOCK_SIZE
                for q, x in enumerate(rows[r].layers[0][0]) if arrive[r] + x.born == step]

    return MM.tables(rows, arrive, slots, BLOCK_SIZE)


def map_case(requests: dict[str, Program], arrive: dict[str, int], censuses: dict[str, dict] | None = None) \
        -> tuple[dict, dict]:
    """The session's KV under the builder's record as an instance, and what the test reads beside it: each row value's
    request, layer, plane and `MM.Row`, each value's block, and the blocks the replayed map holds at each step. A
    request's census is its Program's at its sampling events unless `censuses` gives one."""
    from verity_vllm.commit import memory_map as MM
    from verity_vllm.protocols.remote_bandwidth.interface import PageLayout
    from verity_vllm.protocols.remote_bandwidth.registry import _page_blocks

    censuses = censuses or {}
    rows = {r: MM.rows(p, censuses.get(r)) for r, p in requests.items()}
    tables = first_need(rows, arrive)
    head, steps = MM.record(rows, arrive, _payload, bytes(range(64)), MM.KvPool(block_size=BLOCK_SIZE), tables)
    end = max(arrive[r] + rs.steps for r, rs in rows.items())
    assert all(s.step < end for s in steps)
    (sb,) = {rs.row_bytes for rs in rows.values()}
    ids, lo, hi = _page_blocks(PageLayout(1 + max(max(t) for t in tables.values()), BLOCK_SIZE, sb), codec.BLOCK_BYTES)
    per_plane, region = ids.shape[1] // 2, int(ids.max()) + 1
    # each block of a page: its request, layer, plane, rows, slot, its first byte in its rows, whether it is whole, and
    # the page's place in its request
    pages = [(r, layer, plane, kv[plane][p0:p1], layer * region + int(ids[page, c]), i * codec.BLOCK_BYTES - int(lo[c]) * sb,
              p1 <= rs.positions, q)
             for r, rs in rows.items() for q, page in enumerate(tables[r]) for c in range(ids.shape[1])
             for plane, i in [divmod(c, per_plane)]
             for p0, p1 in [(q * BLOCK_SIZE + int(lo[c]), q * BLOCK_SIZE + int(hi[c]))]
             for layer, kv in enumerate(rs.layers)]
    by, slots = {s.step: s for s in steps}, {x for s in steps for x in (*s.freed, *(w.slot for w in s.written))}
    replay, written, freed, index, held = Replay(head), {}, {}, {}, []
    for t in range(end):
        held.append(frozenset(j for x in slots if (j := replay.holds(x)) is not None and j >= head.blocks))
        if t in by:
            change = replay.apply(by[t])
            written[t] = frozenset(w.index for w in by[t].written)
            freed[t] = frozenset(j for f, n in change.released for j in range(f, f + n) if j >= head.blocks)
            index.update(((t, w.slot), w) for w in by[t].written)

    unwritten = count(replay.extent)
    width, step, value, block, offset, payload, of_value, page_of = {}, {}, {}, {}, {}, {}, {}, {}
    ops: dict[tuple[str, int], set[int]] = {}
    uses: dict[tuple[str, int], set[int]] = {}
    reads = {r: row_reads(p, censuses.get(r)) for r, p in requests.items()}
    for r, layer, plane, got, slot, off, whole, q in pages:
        w = index.get((arrive[r] + max(x.born for x in got), slot)) if whole else None
        j = next(unwritten) if w is None else w.index
        if w is not None:
            data = b"".join(_payload(r, x) for x in got)[off:off + codec.BLOCK_BYTES]
            assert codec.digest(data) == w.digest
            payload[j] = int.from_bytes(data, "little")
        for u, x in enumerate(got):
            v = len(width)
            width[v], step[v], of_value[v], page_of[v] = 8 * sb, arrive[r] + x.born, (r, layer, plane, x), q
            value[v] = int.from_bytes(_payload(r, x), "little")
            block[v], offset[v] = j, 8 * (u * sb - off)
            read, selected = reads[r](x.group)
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
    instance = encode(ex, R.Store.Placement(block, offset), store, held, codec.BLOCK_BYTES * 8, end)
    return instance, {"rows": of_value, "page": page_of, "width": width, "block": block, "held": held, "arrive": arrive}


def _verdicts(out: dict, session: dict) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Read off `MM.rows` and the replayed map beside Lean: the live rows, those in complete blocks (every row of the
    block born before the step), which are Lean's credited ones, and the verdict `StoreMatches` and `PlanMatches` both
    give, that every credited row is in a block the map holds (a block with none may be held too, and earns W nothing).
    Returns the live rows in incomplete blocks and the blocks held with no credited row, by step."""
    rows, block, held, arrive = session["rows"], session["block"], session["held"], session["arrive"]
    born = lambda v: arrive[rows[v][0]] + rows[v][3].born  # noqa: E731
    live = lambda t: {v for v, (r, _, _, x) in rows.items() if born(v) < t <= arrive[r] + x.last}  # noqa: E731
    uncredited, dead = {}, {}
    for t, s in enumerate(out["steps"]):
        assert s["live"] == s["reads"] == sorted(live(t)) and s["resident"] == sorted(held[t])
        late = {block[v] for v in rows if born(v) >= t}
        credited = {v for v in live(t) if block[v] not in late}
        assert s["credited"] == s["credited_reads"] == sorted(credited)
        assert s["live_bits"] - s["credited_bits"] == sum(session["width"][v] for v in live(t) - credited)
        uncredited[t] = live(t) - credited
        dead[t] = {j for j in held[t] if not any(block[v] == j for v in credited)}
        missing = {v for v in credited if block[v] not in held[t]}
        assert s["store_matches"] == s["plan_matches"] == (not missing)
    return uncredited, dead


def _one_block_each(out: dict, session: dict, uncredited: dict[int, set[int]]) -> None:
    """`OneBlockEach` with a row's group its request, layer and plane, so that `RequirementUncredited` bounds what the
    audit doesn't credit by a block's B per group."""
    rows, block = session["rows"], session["block"]
    for t, vs in uncredited.items():
        groups: dict[tuple[str, int, int], set[int]] = {}
        for v in vs:
            groups.setdefault(rows[v][:3], set()).add(block[v])
        assert all(len(js) == 1 for js in groups.values())
        s = out["steps"][t]
        assert s["live_bits"] - s["credited_bits"] <= len(groups) * codec.BLOCK_BYTES * 8


@pytest.mark.parametrize("r1", [1, 3])
def test_the_memory_map(r1, requests, server):
    """The builder's record, each block freed in the record of its rows' last structural reader: the store and plans are
    W's at every step, and the map holds no block with every row dead. r0's second page (positions 4 to 7, born at steps
    1 to 4) is written at step 4, so its first rows are live from step 2 in a block still being written, through step 4:
    those are the only rows the audit doesn't credit, a block per layer and plane."""
    value, session = map_case({r: requests[r] for r in ("r0", "r1")}, {"r0": 0, "r1": r1})
    out = _lean(value, server)
    assert out is not None
    uncredited, dead = _verdicts(out, session)
    late = {t: sorted(js) for t, js in dead.items() if js}
    assert all(s["store_matches"] and s["plan_matches"] for s in out["steps"]) and not late, late
    assert {t for t, vs in uncredited.items() if vs} == {2, 3, 4}
    assert {(session["rows"][v][0], session["rows"][v][3].born) for vs in uncredited.values() for v in vs} == \
        {("r0", 1), ("r0", 2), ("r0", 3)}
    _one_block_each(out, session, uncredited)


def test_a_block_its_last_reader_completes_is_in_no_record(requests, server):
    """A request ending on a page boundary while another runs on: r1 (a prefill of 4 and 4 decode steps) from step 0 completes
    its second page at its last step, 4, which is also its last structural reader. Its rows were live only while the page was incomplete,
    so no step reads the block while it is resident, and it is in no record (memory-map/v1's ruling of 18 Oct, 02:29Z):
    the link holds at every step, and the map holds no block with every row dead."""
    value, session = map_case({"r0": requests["r0"], "r1": requests["r1_page"]}, {"r0": 0, "r1": 0})
    out = _lean(value, server)
    assert out is not None
    uncredited, dead = _verdicts(out, session)
    _one_block_each(out, session, uncredited)
    late = {t: sorted(js) for t, js in dead.items() if js}
    assert all(s["store_matches"] and s["plan_matches"] for s in out["steps"]) and not late, late


def freed_at(value: dict, blocks: set[int], t: int, u: int) -> dict:
    """`value` with `blocks`' free moved from step `t`'s record to an earlier step `u`'s, and out of the plans after `u`."""
    s = value["store"]
    freed = [sorted(set(js) - blocks if i == t else set(js) | blocks if i == u else set(js))
             for i, js in enumerate(s["freed"])]
    return {**value, "store": {**s, "freed": freed},
            "plans": [sorted(set(p) - blocks) if i > u else p for i, p in enumerate(value["plans"])]}


def test_a_block_a_gather_may_select_stays_resident_through_the_last_step_that_may(requests, server):
    """Frees are structural: r0's last step, 5, selects its first page's rows instead of reading them (`selected_at`, a
    stand-in: no served Program gathers its KV), so step 4 is their last use and step 5 their last structural reader,
    where Lean's `live` ends. The builder's record frees the page after step 5, and the link holds at every step. Freed
    after step 4, their last use, the page is out of the store at step 5, where its rows are live and credited though
    step 5 reads none of them: `StoreMatches` fails there alone, and `PlanMatches` holds."""
    from verity_vllm.commit import memory_map as MM

    r0 = requests["r0"]
    c = MM.census_of(r0)
    first = set().union(*(census_values(r0, c, x.group) for kv in MM.rows(r0, c).layers for plane in kv for x in plane[:4]))
    value, session = map_case({"r0": r0, "r1": requests["r1"]}, {"r0": 0, "r1": 1}, {"r0": selected_at(c, first, 5)})
    page = {v for v, (r, *_) in session["rows"].items() if r == "r0" and session["page"][v] == 0}
    blocks = {session["block"][v] for v in page}
    assert page and blocks <= set(value["store"]["freed"][5])
    out = _lean(value, server)
    assert out is not None and all(s["store_matches"] and s["plan_matches"] for s in out["steps"])
    s5 = out["steps"][5]
    assert page <= set(s5["live"]) & set(s5["credited"]) and not page & set(s5["reads"])
    assert all(page <= set(s["reads"]) for s in out["steps"][1:5])
    early = _lean(freed_at(value, blocks, 5, 4), server)
    assert early is not None and all(s["plan_matches"] for s in early["steps"])
    assert [s["store_matches"] for s in early["steps"]] == [True] * 5 + [False]


def deferred(value: dict, lag: int) -> dict:
    """`value` with every append's coin back `lag` steps late (memory-map/v1's deferred residency): each written block's
    write recorded `lag` steps after the record of the step that completes it, its first encoded read the step after
    that, and the block withdrawn, in no record, when its rows' last structural reader (its free's step) comes no later
    than the record its write would be in. Each step's plan is the blocks the map holds then that are still resident."""
    s = value["store"]
    free = {j: t for t, js in enumerate(s["freed"]) for j in js}
    top = len(s["written"]) + lag
    written: list[list[int]] = [[] for _ in range(top)]
    freed = [list(js) for js in s["freed"]] + [[] for _ in range(top - len(s["freed"]))]
    encoded: dict[int, int] = {}
    for t, js in enumerate(s["written"]):
        for j in js:
            encoded[j] = t + lag + 1
            if free.get(j, t + lag + 1) <= t + lag:
                freed[free[j]].remove(j)
            else:
                written[t + lag].append(j)
    resident = lambda t: set(chain(*written[:t])) - set(chain(*freed[:t]))  # noqa: E731
    return {**value, "encoded": [encoded.get(j, 0) for j in range(1 + max(encoded, default=-1))],
            "store": {**s, "written": written, "freed": freed},
            "plans": [sorted(set(p) & resident(t)) for t, p in enumerate(value["plans"])]}


def test_deferred_residency_credits_each_block_from_its_first_encoded_read(requests, server):
    """Every append's coin back two steps late (`deferred`), r1 from step 1: r0's first page, complete at step 0, is
    written in step 2's record and credited from step 3, its first encoded read; r1's page (complete at step 1, last
    structural reader step 2) and r0's second (complete at step 4, last structural reader step 5) are withdrawn. With the first encoded
    reads the link holds at every step, and the rows the audit doesn't credit lie in one block per request, layer, plane
    and page parity (`OneBlockEach`, a page's coin back before the next page is complete): at step 2 in two blocks per
    layer and plane of r0, its first page waiting for its coin while its second fills. Without them the audit credits
    r0's first page from step 1, before its record writes it, and the withdrawn second page at step 5, where the link
    breaks."""
    value, session = map_case({r: requests[r] for r in ("r0", "r1")}, {"r0": 0, "r1": 1})
    late = deferred(value, 2)
    out = _lean(late, server)
    assert out is not None and all(s["store_matches"] and s["plan_matches"] for s in out["steps"])
    rows, page, block = session["rows"], session["page"], session["block"]
    gone = set(chain(*value["store"]["written"])) - set(chain(*late["store"]["written"]))
    assert {(rows[v][0], page[v]) for v in rows if block[v] in gone} == {("r0", 1), ("r1", 0)}
    pages = []
    for s in out["steps"]:
        uncredited = set(s["live"]) - set(s["credited"])
        groups: dict[tuple, set[int]] = {}
        for v in uncredited:
            groups.setdefault((*rows[v][:3], page[v] % 2), set()).add(block[v])
        assert all(len(js) == 1 for js in groups.values())
        assert s["live_bits"] - s["credited_bits"] <= len(groups) * codec.BLOCK_BYTES * 8
        pages.append(sorted({(rows[v][0], page[v]) for v in uncredited}))
    assert pages == [[], [("r0", 0)], [("r0", 0), ("r0", 1), ("r1", 0)], [("r0", 1)], [("r0", 1)], [("r0", 1)]]
    at_once = _lean({**late, "encoded": []}, server)
    assert [s["store_matches"] for s in at_once["steps"]] == [True, False, False, True, True, False]
