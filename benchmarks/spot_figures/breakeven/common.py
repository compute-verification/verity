"""Figure 1's request-grain lines as total_cost builds them, at one scope, and the planner's row for one."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks" / "spot_figures")]

import figure1 as f1  # noqa: E402
import total_cost as tc  # noqa: E402
from verity.protocols.spot import optimize as opt  # noqa: E402

#: proving only: each case adds the verifier's seconds itself (`reprice.apply_server`'s column)
sc, c, cuts, confs, inputs, lines, computed = tc._lines("GLM-5.3", "census", verify=False)
SCOPES = {s.name: s for s in f1.scopes(c, lines["tokens"][0])}


def job(name, scope="world week", scale=1.0, beta=0.01, inputs_=None, line_=None, conf=None, replays=None):
    conf = conf or confs[name]
    s = SCOPES[scope]
    i = inputs_ or inputs[name]
    tb, fl, oh = line_ or lines[name]
    m = f1.model(i, tb, s, oh * scale, fl)
    reps = tc.batch(conf, s, c) if replays is None else replays
    g = tc.Grain(m.replay.replay_blocks, m.cells, m.replay.cost * reps, m.served)
    span = g.replay_blocks * tc.registration_gpu_seconds(conf.registration, c) / s.gpu_seconds
    return m, g, beta * s.gpu_seconds, span, conf


def design(name, **kw):
    m, g, budget, span, conf = job(name, **kw)
    row = opt.two_stage_design(g.cells, g.replay_blocks, m.delta, budget=budget, replay=g.replay, h=m.byte_s,
                               served=m.served, union=conf.form == "patterns", registration=span)
    return m, g, row, span


def strata_of(name, inputs_=None):
    return [x for x in (inputs_ or inputs[name])["partition"].value["strata"] if x["per_token"] > 0]
