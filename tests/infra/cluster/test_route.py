from _build import T0, alloc, cluster, job, node

from cluster.model import Share
from cluster.route import View, place

C = cluster(node("home", "p"), node("away", "q"), shares=(Share("p", "away", max_gpus=4),))


def test_owner_work_goes_home_when_it_fits():
    assert place(C, job("a-1", "p", gpus=1), {}, T0).node == "home"


def test_it_borrows_idle_capacity_when_home_is_full_and_goes_home_when_neither_fits():
    full = View(allocations=(alloc(job("x-1", "p", gpus=8), tuple(range(8)), "0-1", node="home"),))
    p = place(C, job("a-1", "p", gpus=2), {"home": full}, T0)
    assert (p.node, p.why) == ("away", "starts now")
    away_full = View(allocations=(alloc(job("y-1", "q", gpus=8), tuple(range(8)), "0-1", node="away"),))
    assert place(C, job("a-2", "p", gpus=2), {"home": full, "away": away_full}, T0).node == "home"


def test_staged_inputs_break_ties_and_quiet_jobs_stay_home():
    q = cluster(node("b", "p"), node("c", "p"))
    assert place(q, job("a-1", "p", inputs=frozenset({"hf:m@1"})), {"c": View(has=frozenset({"hf:m@1"}))}, T0).node == "c"
    assert place(C, job("w-1", "p", gpus=8, quiet=True, priority="timed"), {}, T0).node == "home"


def test_refusal_names_every_nodes_reason():
    short = cluster(node("home", "p", stops_at=T0 + 100), node("away", "q"), shares=(Share("p", "away", until=T0 + 100),))
    p = place(short, job("a-1", "p", max_s=600), {}, T0)
    assert p.node is None and "home: would run past" in p.why and "away: p's share on away ends" in p.why
    assert place(C, job("a-1", "p", node="away"), {}, T0).node == "away"
