"""The one bound's test vectors (te_core.formalism's `bound`, and `at` under reading A), which the console's rho.ts
runs. null is unmeasured (a term, the max, or ρ where no kind is measured); JSON has no infinity, so ∞ is "inf".

    python -m te_core.rho_vectors [--write]

writes `rho_vectors.json` beside this file (prints it without --write); te_core's tests check the file is what this
module makes. Each case is a step of wildcard's program traced on the tiny start (`CASES`, the circuits under
te_core/capture/tests/fixtures), measured by te_core.capture.circuit.measure as one part, with Δ and d stated per case,
and its ops and product rows the same trace's (te_core.tiles.of_graph, the `.ops.json` beside the circuit;
te_core.capture.products at each k of the console's dial, the `.products.json`). Each vector is a case at a cell (c, k)
of `CELLS`:
each kind's term and the inputs it reads, the max over the kinds and the kind it binds on, the sensitivities, Fit,
ρ_fit, ρ and the label. A case's `measured` is te_core.formalism.from_json's form, reading A's `entries`, `ops` and
`products` in it.

`products` holds the products term, the theorem of record's (te_core.formalism.he_products, `he_figure` on M), on
hand-written M (`HAND`) and on M from the 45M start's dense step (`dense_45m.products.json`), s-t20's dense step at
full size (`GRID_STEPS`) and the lowrank worker's LoRA and DoRA steps on s-t20 (`BATTERY_VOID`), each at its (c, k)
and |C|: M's groups (each its β per operand, its operands' widths, b_y, whether it is a forward product, its MACs and
its output wires), the term, the β, b_y, F, α, N and O it took, M's share of the candidates' MACs, whether it charged
an operand below its width (at its β_eff), the step's adapter rank and what the cell says it rests on
(`formalism.he_rests_on`).

`fresh` holds the old fresh-products figure (te_core.formalism.fresh_products, the sensitivity fresh_old) on
te_core.io_bound's hand-written product sets (`io_bound.vectors`), the 45M start's dense step and the battery's two
steps, each at its (c, k) and |C|: both figures (`fresh_old` with whole cells where `formalism.P_CELL`,
`fresh_no_cell` without), the order its products are summed in (the reference circuit's chain, `formalism.ORDER`,
where none is declared), what the best subset dropped (`subset`; null where it dropped none) and what it rests on; a
circuit that declares a tree is unmeasured. A set no subset of which has a proved Λ, or one with a product fed by
chains with no fixed point, is void (`formalism.FRESH_VOID`): its figures are the best subset's among the products no
such chain feeds, null where none has one, and `void` has the reason, the share of the counted MACs the subset keeps
and how many products a diverged chain feeds.

`dial` holds the products term as the console's dial reads it from a row's rho_inputs, each case's and the 45M step's
over c at every k of the dial (`formalism.fresh_per_c`: [fresh_old/c, term/c]), and the term and the old figures at
each of those k and a few c, which are c times it (`PER_C_RULE`). `grid` holds the same at every k of the console's
grid (te_core.policy's `K_GRID`, a session's input ids at 16 bits for 1,024 to 32,768 input tokens) and the serving
block's 145,568 bits (`grid_k`), at the console's c, on the 45M start's dense step and s-t20's at full size
(`GRID_STEPS`, each its product rows at those k, `grid_rows`): only a captured step gives rows at a k, and a trace
keeps them at the dial's k alone. `published` holds te_core.formalism.published under reading A at a cell for each
case of its rule (`PUBLISHED`): Fit; the max above 1, partial where a kind is unmeasured; None where a kind is
unmeasured and no term exceeds 1; 1, binding none.

`sessions` holds the console's c, te_core.policy's `session` at every size it states a shape for (`SHAPES`) and each
session of `SESSIONS`: its inputs (parameters, input and output tokens, the shape) and its outputs (the tokens that run
a forward, the positions attended, each product's MACs, the gates a MAC, the gates and the input ids' bits),
`console` the policy every producer reads (`policy.CONSOLE`), and for each of the console's reference sizes
(`policy.REFERENCE_SIZES`) and each start (`policy.starts`) its shape, its session's c and k at the console's session,
and at each k of the console's grid (`dial`) the products term's inputs: the c of a session of k/16 input tokens and
the console's output tokens, and that session's input ids in bits (k at vocab 50,304, 15/16 of it at 32,000).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

from . import VERSION, formalism, io_bound, policy
from .capture.circuit import StepCircuit, measure
from .gates import gates, per_mac

SCHEMA = "te-core/rho-vectors/v3"
OUT = Path(__file__).with_name("rho_vectors.json")
FIXTURES = Path(__file__).parent / "capture" / "tests" / "fixtures"
#: model.py's tiny start: its forward MACs per token at ctx 1024 (te_core.flops.forward_macs_per_token), vocab and width
TINY = {"forward_macs_per_token": 3_327_488, "vocab": 50304, "d_model": 64}
W_FWD = TINY["forward_macs_per_token"] * per_mac("gemm/bf16")
PROGRAM = ("islands/wildcard/train.py at 9460fcd228 on the tiny start, --batch 2 --micro 2 --warmup 1, traced at step 1 "
           "(te_core.capture.script)")
#: the LoRA case's decoder: W + B A for its 14 matrices, r out in fp32 multiply-adds each
LORA_REBUILD_MACS = 4 * 2 * (4 * 64 * 64 + 3 * 64 * 192)
CASES = {
    "adapter": {"circuit": "wildcard_adapter.npz",
                "flags": r"--grow 1 --adapter 8 --adapter-z 2 "
                         r"--set 'blocks\.2\.(mlp_norm|w_gate\.weight|w_up\.weight|w_down\.weight)'",
                "delta_bits": None, "decoder_gates": 0.0,
                "delta": "the trace's update cut, theta' as the step reads it (the grown block's four tensors)",
                "decoder": "none: E stores the grown block's tensors (the model format)"},
    "adapter-live": {"circuit": "wildcard_adapter.npz",
                     "flags": r"--grow 1 --adapter 8 --adapter-z 2 "
                              r"--set 'blocks\.2\.(mlp_norm|w_gate\.weight|w_up\.weight|w_down\.weight)'",
                     "delta_bits": 3 * 8 * 64 * 16 + 64 * 32, "decoder_gates": 0.0,
                     "delta": "stated: an E of the live entries alone (w_gate's and w_up's 8 live rows and w_down's 8 "
                              "live columns at bf16, mlp_norm at fp32; wildcard's 05:46Z LOOPHOLE)",
                     "decoder": "none: the dead entries are zero"},
    "dense": {"circuit": "dense_blocks.npz", "flags": r"--set 'blocks\..*'", "delta_bits": None, "decoder_gates": 0.0,
              "delta": "the trace's update cut, theta' as the step reads it (every block tensor)",
              "decoder": "none: E stores the tensors (the full format)"},
    "lora": {"circuit": "lora_blocks.npz", "flags": r"--set 'blocks\..*' --lora-rank 4", "delta_bits": None,
             "decoder_gates": LORA_REBUILD_MACS * gates(formalism.DECODER_DEFINITION),
             "delta": "the trace's update cut, the factors and gains as the step reads them",
             "decoder": f"W + B A for the 14 matrices, {LORA_REBUILD_MACS} fp32 multiply-adds at "
                        f"{formalism.DECODER_DEFINITION}'s gates each"},
}
#: (tokens per block, k bits): c = tokens x the tiny start's forward (te_core.panels' dial), its default first
CELLS = ((4096, 80_000), (4096, 800_000), (4096, 8_000_000), (100, 8_000_000), (1, 80_000_000))
#: (case, tokens per block or None for the console's c, k bits or None for the console's): `published` under reading A
#: at a cell for each case of its rule. The fixtures' products are classified at each k of the console's dial
#: (te_core.capture.products' `DIAL_K`), so off it (`OFF_DIAL`) their fresh-products term is unmeasured
OFF_DIAL = 100_000
PUBLISHED = (("dense", 1, 80_000_000), ("dense", 4096, 80_000), ("dense", 4096, OFF_DIAL), ("dense", None, None),
             ("lora", None, None), ("adapter-live", 4096, OFF_DIAL), ("adapter-live", 4096, None),
             ("dense, DoRA's void products", None, None), ("dense, DoRA's void products", 1, None))
#: the `PUBLISHED` case whose old fresh-products figure is void: the dense case with the battery's DoRA step's product
#: rows (`BATTERY_VOID`), where the products term is measured. The JSON's `void_case` says how a port builds its
#: record: `measured`'s dense case with the products of the `fresh` vector it names, so they aren't written twice
VOID_CASE = ("dense, DoRA's void products", "dora-all-r16-m64")
#: tokens per block at which `dial` evaluates each case's fresh-products term at every k of the console's dial, beside
#: the console's c
DIAL_TOKENS = (1, 4096)
#: the 45M start's dense step's product rows (te_core.capture.products on te_core.io_bound.capture("dense", "s", 32000,
#: batch 2, ctx 256, one micro-step) at the console's k)
DENSE_45M = "dense_45m.products.json"
#: the steps `grid` reads, each by its fixture under `GRID_FIXTURES` (its product rows at every k of `grid_k`,
#: `grid_rows`) and the te_core.io_bound.capture arguments of its dense step: the 45M start's, as `DENSE_45M`, and
#: s-t20's at full size, 256 sequences of 1,024 tokens in four micro-batches
GRID_FIXTURES = Path(__file__).parent / "tests" / "fixtures"
GRID_STEPS = {
    "the 45M start's dense step": ("dense_45m.grid.products.json",
                                   {"size": "s", "vocab": 32000, "batch": 2, "ctx": 256, "micro_steps": 1}),
    "s-t20's dense step at full size": ("s_t20.grid.products.json",
                                        {"size": "s", "vocab": 32000, "batch": 256, "ctx": 1024, "micro_steps": 4}),
}
#: the lowrank worker's battery programs on s-t20 whose old fresh-products figure is void, each traced step's product
#: rows at the console's k: DoRA rank 16 on all 42 matrices and LoRA rank 16 on every block's MLP
BATTERY_VOID = ("battery_void.products.json", {
    "dora-all-r16-m64": "the s-t20 start's DoRA rank-16 step on all 42 matrices (void: its two small unfed fp32 "
                        "products beside it)",
    "lora-mlp-r16-m32": "the s-t20 start's LoRA rank-16 step on every block's MLP (void: the one unfed product beside "
                        "it)"})
OPS = ("counted (trace): the traced step's GEMMs with their dims and operand classes, and the rest of each phase's "
       "gates (te_core.tiles.of_graph)")
#: hand-written M for the products term, each (name, c, k, |C|, groups) with its figure worked by hand in
#: tests/test_rho_vectors.py: a forward and a backward product sharing arrays at k = 3·2^14, where α = 2k and the
#: outputs' term is 2/3 of the MACs' (ρ 10^6); and at the console's k, a forward product at β 16 beside a backward one
#: whose operand an adapter of rank 16 reaches, at β_eff 3.2 (K* 32), which binds
HAND = (("a forward and a backward product, k 49,152", 1e12, 49_152.0, 1e15,
         [{"beta": [16.0, 16.0], "width": [16.0, 16.0], "weight": [True, True], "b_y": 16.0, "forward": True,
           "macs": 2.21184e14, "wires": 2.21184e14 / 48},
          {"beta": [16.0, 16.0], "width": [16.0, 16.0], "weight": [True, True], "b_y": 16.0, "forward": False,
           "macs": 2.21184e14, "wires": 2.21184e14 / 48}]),
        ("a forward product beside one an adapter's rank-16 output reaches, the console's k", 1e12, 65_536.0, 1e15,
         [{"beta": [16.0, 16.0], "width": [16.0, 16.0], "weight": [True, True], "b_y": 16.0, "forward": True,
           "macs": 5.89824e14, "wires": 5.89824e14 / 48},
          {"beta": [16.0, 3.2], "width": [16.0, 16.0], "weight": [True, True], "b_y": 16.0, "forward": False,
           "macs": 1e17, "wires": 1e12}]))
#: (input tokens, output tokens): the console's session, the workload note's tighter and looser ones, a prefill alone
#: and the least session
SESSIONS = ((4096, 1024), (2048, 3072), (16384, 1024), (1024, 0), (1, 1))
SESSION_RULE = ("c = (tokens·n_layer·(4·d_model² + 3·d_model·d_ff) + tokens·d_model·vocab + 2·n_layer·d_model·attended)"
                " x gates_per_mac, tokens = in + out, attended = in² + out·in + out·(out + 1)/2 (a prompt token over "
                "the whole prompt, the j-th output token over the prompt and the output tokens up to itself); k = in x "
                "ceil(log2 vocab)")
SESSION_DIAL = ("each k of the console's grid (te_core.policy's K_GRID): a session of in_tokens = k/16 input tokens and "
                "out_tokens output tokens, c its gates (te_core.policy.session_gates), the products term's c at k, and "
                "k_bits its input ids at the model's own vocabulary (k at 50,304, 15/16 of it at 32,000)")


def sessions() -> dict:
    out = policy.CONSOLE_SESSION[1]

    def named(name: str, params: int) -> dict:
        s = policy.session(params, *policy.CONSOLE_SESSION)
        dial = [policy.session(params, n, out) for n in policy.DIAL_IN_TOKENS]
        return {"name": name, "params": params, "shape": s["shape"], "c": s["gates"], "k_bits": s["k_bits"],
                "dial": [{"k": k, "in_tokens": d["in_tokens"], "c": d["gates"], "k_bits": d["k_bits"]}
                         for k, d in zip(policy.K_GRID, dial)]}

    return {"rule": SESSION_RULE,
            "about": "te_core.policy.session: one session of the reference model, each of its tokens a forward, every "
                     "product of two operands at the served precision's GEMM price; the elementwise ops left out",
            "console": {"params": policy.REFERENCE_PARAMS, "in_tokens": policy.CONSOLE_SESSION[0],
                        "out_tokens": policy.CONSOLE_SESSION[1], "c": policy.CONSOLE[0], "k": policy.CONSOLE[1]},
            "dial": {"about": SESSION_DIAL, "k": list(policy.K_GRID), "in_tokens": list(policy.DIAL_IN_TOKENS),
                     "out_tokens": out},
            "sizes": [named(name, p) for name, p in policy.REFERENCE_SIZES.items()],
            "starts": [named(name, shape["parameters"]) for name, shape in policy.starts().items()],
            "vectors": [policy.session(p, i, o) for p in policy.SHAPES for i, o in SESSIONS]}


def case_measured(case: dict) -> formalism.Measured:
    """A case's quantities: its traced step's (`measure`) as one part, Δ and d as the case states, the step's ops (the
    `.ops.json` beside its circuit) and its product rows (the `.products.json`, te_core.capture.products')."""
    circuit = FIXTURES / case["circuit"]
    f = measure(StepCircuit.load(circuit))
    part = {"gates": f["gates"], "tokens": f["tokens"], "token_gates": f["sigma"] * f["gates"],
            "affected_gates": f["affected_share"] * f["gates"], "steps": 1, "delta_grad_bits": f["delta_grad_bits"],
            "partial_bits": f["partial_cut_bits"], "grad_bits": f["grad_cut_bits"]}
    delta = case["delta_bits"] if case["delta_bits"] is not None else float(f["update_cut_bits"])
    ops = json.loads(circuit.with_suffix(".ops.json").read_text())
    rows = json.loads(circuit.with_suffix(".products.json").read_text())
    block = formalism.of_trace(f) | {"parts": [part], "delta_bits": delta, "decoder_gates": case["decoder_gates"],
                                     "ops": ops, "products": formalism.products_of_trace(rows)}
    block["labels"] |= {"delta_bits": case["delta"], "decoder_gates": case["decoder"], "ops": OPS,
                        "products": formalism.TRACED_PRODUCTS}
    return formalism.complete(block, {"output": {"e_bytes": delta / 8}}, TINY)


def battery(program: str) -> tuple[float, dict]:
    """`BATTERY_VOID`'s step `program`: its |C| and its `Measured.products` (`formalism.products_of_trace` of its
    rows)."""
    fx = json.loads((FIXTURES / BATTERY_VOID[0]).read_text())
    p = fx["programs"][program]
    block = {x: v for x, v in p.items() if x != "args"} | {"rows": [fx["rows"][i] for i in p["rows"]]}
    return block["gates"], formalism.products_of_trace(block)


def products(base: formalism.Measured) -> dict:
    """The products term's vectors: `HAND`'s M through `formalism.he_figure`, and each traced step's products as
    `Measured.products` on `base` with one part of |C| gates and no update code (the term reads M, |C| and the code,
    whose bits a trained weight cap a weight's β) through `formalism.he_products` at the console's (c, k), with the M
    it read. `dial` and `grid` take the traced steps the same way."""
    sets = [(name, c, k, C, {"k": k, "groups": groups, "r_step": None,
                             "candidates_macs": sum(g["macs"] for g in groups), "self_macs": 0.0,
                             "gathered_macs": 0.0, "left_out_macs": 0.0, "steps": None, "step_tokens": None,
                             "trained_weights": None}, None) for name, c, k, C, groups in HAND]
    rows = json.loads((FIXTURES / DENSE_45M).read_text())
    sets.append(("the 45M start's dense step", *policy.CONSOLE, rows["gates"], None, formalism.products_of_trace(rows)))
    name, (fixture, _) = list(GRID_STEPS.items())[1]
    rows = json.loads((GRID_FIXTURES / fixture).read_text())
    sets.append((name, *policy.CONSOLE, rows["gates"], None, formalism.products_of_trace(rows)))
    for program, label in BATTERY_VOID[1].items():
        C, p = battery(program)
        sets.append((label.split(" (")[0], *policy.CONSOLE, C, None, p))
    out = []
    for name, c, k, C, he, p in sets:
        if p is None:
            got, why = formalism.he_figure(he, c, C), None
        else:
            m = replace(base, parts=(formalism.Part(gates=C, tokens=1.0, token_gates=C, affected_gates=C),),
                        products=p, delta_bits=None)
            _, why, got = formalism.he_products(m, c, k)
            he = formalism.classified_at(p, k)["he"]
        out.append({"name": name, "inputs": {"c": c, "k": k, "C_gates": C, "he": he}, "term": got.get("rho"),
                    "why": why, "rests_on": formalism.he_rests_on(got)}
                   | {x: got.get(x) for x in ("subset", "subsets", "beta_w", "beta_x", "beta", "b_y", "F", "alpha",
                                              "N", "O", "K_thr", "beta_k", "macs_share", "below_width", "r_step")})
    return {"rule": "each operand charged its beta, an activation's (a weight is an operand no token indexes) at most "
                    "beta(k) at k; for each subset (all of M; M less its groups of contraction K below K_thr = (F "
                    "k/2)^2 b_y/(beta_w beta_x (k + alpha)) at M's best; the weight gradients alone, F = 1) and each P "
                    "among its groups' products of their two beta (above 0): M_P its groups whose product >= P, N and "
                    "O their MACs and output wires, b_y their least output width, F = 2/sqrt(3) where a group isn't a "
                    "forward product, else 1, alpha = F/2 sqrt(k^3/(P b_y)); term = max(0, max over subsets and P of "
                    "(c/C_gates) (N/(k + alpha) - F (k/2) sqrt(O N/P)/(k + alpha)^1.5)), and its subset, beta_w and "
                    "beta_x (the binding group's two beta), b_y, F, alpha, N and O are the best's; rho floors it at 1",
            "rests_on": formalism.RESTS_ON["fresh_products"], "vectors": out}


def fresh(base: formalism.Measured) -> dict:
    """The old fresh-products figure's vectors (the sensitivity fresh_old): each product set as `Measured.products` on
    `base` with one part of |C| gates (the figure reads only the products and |C|), through
    `formalism.fresh_products` at its (c, k)."""
    sets = [(v["name"], v["inputs"]["c"], v["inputs"]["k"], v["inputs"]["C_gates"],
             {"k": v["inputs"]["k"], "order": "chain", "products": v["inputs"]["products"]})
            for v in io_bound.vectors()["vectors"]]
    rows = json.loads((FIXTURES / DENSE_45M).read_text())
    console = {x: v for x, v in formalism.products_of_trace(rows).items() if x != "at"}
    sets.append(("the 45M start's dense step, its trace's order unknown", *policy.CONSOLE, rows["gates"], console))
    for program, label in BATTERY_VOID[1].items():
        sets.append((label, *policy.CONSOLE, *battery(program)))
    name, c, k, C, square = sets[0]
    sets += [(f"{name}, no order declared", c, k, C, square | {"order": None}),
             (f"{name}, a circuit that declares a tree", c, k, C, square | {"order": "tree"})]
    out = []
    for name, c, k, C, products in sets:
        m = replace(base, parts=(formalism.Part(gates=C, tokens=1.0, token_gates=C, affected_gates=C),),
                    products=products)
        _, why, got = formalism.fresh_products(m, c, k)
        out.append({"name": name, "inputs": {"c": c, "k": k, "C_gates": C, "products": products},
                    "fresh_old": got.get("fresh_cell"), "fresh_no_cell": got.get("fresh_no_cell"),
                    "order": got["order"], "why": why, "subset": got.get("subset"),
                    "share": got["io_bound"]["share"], "void": got.get("void")})
    return {"p_cell": formalism.P_CELL, "order": formalism.ORDER, "rests_on": formalism.SENSITIVITIES["fresh_old"],
            "fresh_no_cell": formalism.SENSITIVITIES["fresh_no_cell"], "void": formalism.FRESH_VOID, "vectors": out}


#: how a port reads the products term and the old figure at (c, k) from `formalism.fresh_per_c`
PER_C_RULE = ("term(c, k) = c × fresh_per_c[k][1] (the products term), fresh_old(c, k) = c × fresh_per_c[k][0] (the "
              "old figure, a sensitivity; null where it is unmeasured, or void with no subset figure); null where "
              "fresh_per_c[k] is null (the term unmeasured) or k is off the dial")


def dial(ms: dict[str, formalism.Measured]) -> dict:
    """The products term as the console's dial reads it from a row's rho_inputs (`formalism.fresh_per_c`, by
    te_core.panels' `rho_columns`) for each case and the 45M start's dense step, and at each k of the dial
    (te_core.capture.products' `DIAL_K`) and each c (the console's, and `DIAL_TOKENS` blocks of the tiny start's
    forward) the term (`formalism.he_products`) and the old figures (`formalism.fresh_products`'), so a port that
    takes c times the entry at k can check them (`PER_C_RULE`); null where unmeasured, with the term's reason."""
    from .capture.products import DIAL_K

    rows = json.loads((FIXTURES / DENSE_45M).read_text())
    C = rows["gates"]
    one = (formalism.Part(gates=C, tokens=1.0, token_gates=C, affected_gates=C),)
    sets = [(name, m) for name, m in ms.items()]
    sets.append(("the 45M start's dense step", replace(ms["dense"], parts=one, delta_bits=None,
                                                       products=formalism.products_of_trace(rows))))
    out = []
    for name, m in sets:
        cells = []
        for k in DIAL_K:
            for c in (policy.CONSOLE[0], *(n * W_FWD for n in DIAL_TOKENS)):
                cells.append(cell(m, c, k))
        out.append({"name": name, "C_gates": m.gates, "fresh_per_c": formalism.fresh_per_c(m), "cells": cells})
    return {"rule": PER_C_RULE, "k": list(DIAL_K), "vectors": out}


def cell(m: formalism.Measured, c: float, k: float) -> dict:
    """`dial`'s and `grid`'s cell at (c, k): the products term with its reason, and the old figures."""
    term, why, _ = formalism.he_products(m, c, k)
    _, _, got = formalism.fresh_products(m, c, k)
    return {"c": c, "k": k, "term": term, "fresh_old": got.get("fresh_cell"), "fresh_no_cell": got.get("fresh_no_cell"),
            "why": why}


def grid_k() -> tuple[int, ...]:
    """`grid`'s k: the console's grid (te_core.policy's `K_GRID`) and the serving block's (te_core.panels' `SERVED`)."""
    from .panels import SERVED

    return tuple(sorted({*policy.K_GRID, SERVED[1]}))


def grid_rows(graph) -> dict:
    """A captured step's product rows at every k of `grid_k` in a trace's form (te_core.capture.products'
    `rows_on_dial`'s): `rows` at the console's k, and under "at" each other k's rows and P-match check."""
    from .capture import products

    out = products.rows(graph, policy.CONSOLE[1])
    out["at"] = {}
    for k in grid_k():
        if k != policy.CONSOLE[1]:
            b = products.rows(graph, float(k))
            out["at"][str(k)] = {x: b[x] for x in ("k", "rows", "p_match")}
    return out


def grid(base: formalism.Measured) -> dict:
    """The products term at every k of `grid_k` on each step of `GRID_STEPS`, measured as `dial` measures the 45M step
    (its fixture's rows as `Measured.products` on `base`, one part of |C| gates): `formalism.fresh_per_c`, and at each
    k the term and the old figures (`cell`), which are c times it, at the console's c and at the k's own: the 45M
    reference's session of k/16 input tokens and the console's output tokens on the grid, the serving block's c at its
    k (te_core.panels' `SERVED`); null where unmeasured, with the reason."""
    from .panels import SERVED

    own = {k: policy.session_gates(policy.REFERENCE_PARAMS, n, policy.CONSOLE_SESSION[1])
           for n, k in zip(policy.DIAL_IN_TOKENS, policy.K_GRID)} | {SERVED[1]: SERVED[0]}
    out = []
    for name, (fixture, _) in GRID_STEPS.items():
        rows = json.loads((GRID_FIXTURES / fixture).read_text())
        C = rows["gates"]
        m = replace(base, parts=(formalism.Part(gates=C, tokens=1.0, token_gates=C, affected_gates=C),),
                    products=formalism.products_of_trace(rows), delta_bits=None)
        cells = [cell(m, c, k) for k in grid_k() for c in dict.fromkeys((policy.CONSOLE[0], own[k]))]
        out.append({"name": name, "C_gates": C, "fresh_per_c": formalism.fresh_per_c(m), "cells": cells})
    return {"rule": f"{PER_C_RULE}; at each k of the console's grid (a session's input ids at 16 bits) and the serving "
                    "block's, and at the console's c and the k's own (the 45M reference's session of k/16 input tokens "
                    f"and {policy.CONSOLE_SESSION[1]:,} output tokens; the serving block's c at its k)",
            "k": list(grid_k()), "vectors": out}


def void_case(dense: formalism.Measured) -> formalism.Measured:
    """`VOID_CASE`'s record: the dense case with the battery's DoRA step's product rows (`BATTERY_VOID`)."""
    return replace(dense, products=battery(VOID_CASE[1])[1])


def published(measured: dict[str, formalism.Measured]) -> list[dict]:
    """`formalism.published` under reading A at each cell of `PUBLISHED`."""
    out = []
    for case, n, k in PUBLISHED:
        c = policy.CONSOLE[0] if n is None else n * W_FWD
        k = policy.CONSOLE[1] if k is None else k
        out.append({"case": case, "c": c, "k": k} | formalism.published(measured[case], c, k, "A"))
    return out


def finite(x):
    """`x` with every infinity "inf" (JSON has none)."""
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    return "inf" if isinstance(x, float) and math.isinf(x) else x


def build() -> dict:
    measured, ms, vectors = {}, {}, []
    for name, case in CASES.items():
        m = ms[name] = case_measured(case)
        measured[name] = asdict(m)
        for n, k in CELLS:
            c = n * W_FWD
            b, a = formalism.bound(m, c, k), formalism.at(m, c, k, "A")
            vectors.append({"case": name, "tokens_per_block": n, "c": c, "k": k, "terms": b["terms"],
                            "inputs": b["inputs"], "max": b["max"], "binding": b["binding"],
                            "unmeasured": b["unmeasured"], "void": b["void"], "sensitivities": b["sensitivities"],
                            "fit": a["fit"], "rho_fit": a["rho_fit"], "rho": a["rho"], "label": b["label"]})
    void = {VOID_CASE[0]: void_case(ms["dense"])}
    return finite({
        "schema": SCHEMA, "version": VERSION, "about": __doc__.split("\n\n")[0],
        "rule": "rho = rho_fit where Fit holds, else max(1, the max over the kinds' terms); 1 where every kind is "
                "computed and none has a figure; null where a kind isn't computed and none has a figure. "
                "Sensitivities are never in the max.",
        "published_rule": "Fit holds: rho_fit, binding fit. Else, where a kind's term exceeds 1: the max, binding "
                          "that kind, partial where a kind is unmeasured. Else, where a kind is unmeasured: null, "
                          "binding unmeasured, why naming it. Else: 1, binding none. products_rests_on is the "
                          "products term's own, with its beta and b_y, wherever it is measured.",
        "products": products(ms["dense"]), "fresh": fresh(ms["dense"]), "dial": dial(ms), "grid": grid(ms["dense"]),
        "published": published(ms | void),
        "kinds": list(formalism.KINDS), "rests_on": formalism.RESTS_ON, "sensitivities": formalism.SENSITIVITIES,
        "w_fwd_gates": W_FWD, "start": TINY, "program": PROGRAM,
        "cases": {name: {k: v for k, v in case.items() if k not in ("delta_bits", "decoder_gates")}
                  for name, case in CASES.items()},
        "measured": measured, "void_case": {"case": VOID_CASE[0], "measured": "dense",
                                            "products": BATTERY_VOID[1][VOID_CASE[1]]},
        "vectors": vectors, "sessions": sessions(), "tolerance": {"relative": 1e-9}})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write", action="store_true", help=f"write {OUT.name} beside this module")
    ap.add_argument("--fixtures", action="store_true",
                    help="first capture each step of GRID_STEPS and write its fixture (about 30 s on a CPU)")
    a = ap.parse_args(argv)
    if a.fixtures:
        GRID_FIXTURES.mkdir(exist_ok=True)
        for fixture, args in GRID_STEPS.values():
            path = GRID_FIXTURES / fixture
            path.write_text(json.dumps(grid_rows(io_bound.capture("dense", **args)), sort_keys=True,
                                       separators=(",", ":")) + "\n")
            print(f"{path}: {path.stat().st_size:,} bytes", file=sys.stderr)
    text = json.dumps(build(), indent=0, sort_keys=True) + "\n"
    if a.write:
        OUT.write_text(text)
        print(f"{OUT}: {len(text):,} bytes", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
