"""Red-team phase 15, target 3 (#3764): the ingress-ids tie refuses a request-id relabel. The served-rows tie (#3745,
`io_tie`) accepts a relabel by design — it ties each served cell to a committed output by position, not to the request it
belongs to (PROVE-PATH's real-rows run 8ec1). The ingress-ids tie (`io_ids`) must not: it binds every prompt field the
warden certified to the id the Commit computed for that field's (request, position) class, so moving a request's
computation onto another request, swapping two requests' prompts, or colliding two classes is caught by the value the
class must hold.

Reference-level cases (fast); the slow `test_the_unit_is_its_reference` already pins the circuit to the reference,
including a class→sink, an entry relabel and a two-field class swap. These add: a Commit that serves one request another
request's real ids, two requests that swap prompts on the wire, and a class collision across two distinct values. The
link helpers are copied from `test_network_certifier_io_ids.py`.
"""
from __future__ import annotations

import copy

from verity.core.protocols.certifier import io_ids as I
from verity.core.protocols.certifier import io_request as Q
from verity.core.protocols.certifier.active import id_request

PL = (5, 60, 12, 100, 1)
COUNTS = (1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 0, 0)
B, P = 512, 16
ACROSS, WITHIN = {3: (1, 32)}, {3: (2, 8)}
SHAPE = I.Shape(B, 4, 2, 1, P, 8, 4)


def _prompts(sources):
    ps = [[(7919 * i + 104729 * k) % 151329 for k in range(n)] for i, n in enumerate(PL)]
    for q, (src, hit) in sources.items():
        ps[q][:hit] = ps[src][:hit]
    return ps


def _steps(sources):
    pos, todo, steps = {q: sources.get(q, (0, 0))[1] for q in range(len(PL))}, list(range(len(PL))), []
    while todo:
        rows, room = [], P
        for q in list(todo):
            if not room:
                break
            n = min(room, PL[q] - pos[q])
            rows.append(I.Row(q, pos[q], n or 1))
            room -= n or 1
            if n:
                pos[q] += n
            else:
                todo.remove(q)
        steps.append(rows)
    return steps + [[I.Row(1, 0, 16)]]


def _tokens(ps):
    return [p + [900_000 + 100 * q + k for k in range(4)] for q, p in enumerate(ps)]


def _link(sources, ps=None, tokens=None, shape=SHAPE):
    ps = _prompts(sources) if ps is None else ps
    reqs = [id_request(model="glm", user=f"u{i}", prompt=p, max_tokens=16, temperature=0.7, top_p=0.95, seed=i,
                       frame_bytes=B) for i, p in enumerate(ps)]
    windows = Q.windows_of(shape.request, reqs, COUNTS)
    received = tuple(sum(COUNTS[w * shape.T:(w + 1) * shape.T]) for w in range(len(windows)))
    steps = _steps(sources)
    rows = I.commit_rows(steps, _tokens(ps) if tokens is None else tokens, P)
    salts, levels, dom = I.commit(rows)
    out = []
    for w, cells in enumerate(windows):
        view = Q.request_view(PL, received, w, B)
        mp = I.window_map(shape, PL, view, steps, sources)
        out.append(I.instance(shape, Q.instance(shape.request, cells, view, w), mp, rows, salts, levels, dom))
    return out


def _verdicts(insts):
    return [I.reference(SHAPE, i) for i in insts]


def test_phase15_honest_is_tied_everywhere():
    """The honest link is accepted in every window, so the refusals below come from the relabel, not the scenario."""
    assert _verdicts(_link(ACROSS)) == [1, 1, 1]


def test_phase15_a_commit_that_serves_one_request_another_requests_real_ids_is_refused():
    """The request-id relabel the served-rows tie accepts. The Commit holds, at request 0's five prompt positions,
    request 2's real first five ids — a legitimate other request's computation, moved onto request 0. The served-rows
    tie would accept those cells (they are committed outputs); the ingress-ids tie refuses request 0's window, since
    request 0's certified prompt fields don't equal the ids its class now holds. No window that lacks request 0's
    fields is touched."""
    ps = _prompts(ACROSS)
    tokens = _tokens(ps)
    tokens[0][:PL[0]] = _tokens(ps)[2][:PL[0]]
    honest, relabel = _verdicts(_link(ACROSS)), _verdicts(_link(ACROSS, tokens=tokens))
    assert honest == [1, 1, 1]
    assert relabel != honest and relabel.count(0) == 1 and sum(relabel) == len(relabel) - 1


def test_phase15_two_requests_that_swap_prompts_on_the_wire_are_refused():
    """Two requests swap prompts. Request 0's and request 2's first five certified prompt ids are swapped on the wire
    (the warden's cells), while the Commit computes each honestly: neither request's fields equal the ids of its own
    class, so both requests' windows refuse. A swap is two relabels, and the ids tie catches each."""
    orig = _prompts(ACROSS)
    tokens = _tokens(orig)
    wire = copy.deepcopy(orig)
    wire[0][:PL[0]], wire[2][:PL[0]] = orig[2][:PL[0]], orig[0][:PL[0]]
    assert wire[0][:PL[0]] != orig[0][:PL[0]] and wire[2][:PL[0]] != orig[2][:PL[0]]
    swapped = _verdicts(_link(ACROSS, ps=wire, tokens=tokens))
    assert swapped != [1, 1, 1] and 0 in swapped


def test_phase15_a_class_collision_across_two_distinct_values_is_refused():
    """A class collision in the advice: two fields with distinct user ids are routed to one class. The class can hold
    only one value, so `matches` refuses — a collision can't let one id stand in for another. The orphaned class (the
    one the second field left) has an opened entry no field now holds, which refuses too."""
    inst = _link(ACROSS)[1]
    sh, mp = SHAPE, inst.map
    vals, _ = I.scan(sh, inst.request.cells, inst.request.view.carry_in, inst.request.window)
    i, j = 0, next(k for k in range(1, len(vals)) if vals[k] != vals[0])
    assert vals[i] != vals[j] and mp.keys[i] != mp.keys[j]
    keys = list(mp.keys)
    keys[j] = keys[i]
    collided = inst._replace(map=mp._replace(keys=tuple(keys)))
    assert I.reference(sh, inst) == 1 and I.reference(sh, collided) == 0
