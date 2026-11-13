[2026-11-13T21:22Z] 804be07c880f
CALLERS, Fri 13 Nov 13:22 PST.

**I have read the code at `56f6c267563`; no line of the design is contradicted outright, but four lines need a note (below), and I go on with the design as written.**

Branch: `cursor/one-slowdown-function-56e2`, cut from `origin/main` `56f6c267563` (RAN, on my VM).

What differs from your reading at `f479876a8758` (READ, the files named):
- Nothing material. `calculator.slowdown` calls `_bound` (so `min_cut.bound`) a second time on the work `_figure` chose; that call goes away, since the one function returns the `Bound` (READ `calculator.py`).
- Nothing under `demos/` names `min_cut`, a slowdown or the calculator; no file under `tests/` or `tools/` names `blocks_at_least`, `slowdown_at_least` or `bound_charged` (RAN, on my VM, `rg -l` over `demos`, `tests`, `tools`).

Lines that need a note:
1. Item 1, "a module beside it that imports nothing of its package": such a module couldn't call `bound` without loading `min_cut`, so the function goes in `rho/min_cut.py` itself, with `clique` (INFERRED).
2. Item 9(b), "no call of `bound`, `blocks_at_least` or `slowdown_at_least`": tests outside the calculator call them today as independent checks: `training_slowdown/tests/conftest.py::calculators_count` (`bound`), `test_formalism.py` (`slowdown_at_least`), `test_grade.py` (`blocks_at_least`), `test_one_calculator.py` (`bound`), `inference_only/tests/test_inference_only.py` (`bound` three times), `circuit_properties/tests/test_work.py` (`bound`) (READ). I hold tests to (b) too and rewrite those checks: against the count's right side (`macs_at_most`, solved in the test by bisection), or through the one function. Two training tests stub `_min_cut().bound` (`test_one_calculator.py::test_the_products_term_is_the_calculators`, `test_grade.py::test_the_bound_of_record_is_the_calculators_and_rho_follows_it`). Once the choosing is inside `min_cut`, that stub no longer reaches it, so they stub the one function instead (INFERRED).
3. Item 2, what it takes: `test_program_reading.py` calls `work.bound(work, C, F=F_)` so that it can compare both readings at one F (READ). So the one function takes an optional `F` that overrides the rule on every sub-work. Also, bit-exactness: `_figure` scales N and O after summing (`scale * Σ macs`, the first steps left out), while `circuit_properties` prices each product (`Σ keep·macs`). Pricing per product in training would move the last bits wherever first steps are left out. So the work carries one factor applied after the sum, `scale` (default 1), which the training caller sets (READ `calculator.py::_figure`, `work.py::bound`).
4. Item 6/8, `circuit_properties`: its `read` leaves out every product of two activations (`reason = "activations"`), so every charged group has a weight operand. Its "kinds" loop is vacuous: "both" is "weights", and "activations" is always empty (READ `work.py::_read`). Its groups state no forward or weight-gradient flag either (a Program can't tell them apart, its docstring says). So under the moved family its sub-works are the thresholds P on all of M and on M without the contractions shorter than K_thr; the weight-gradients-alone subset is empty there. Its figures can only fall, where a cut at another contraction length bound. Item 8(b) measured on circuit works is therefore zero by construction; on the training traces it is not (INFERRED).

The function, in `rho/min_cut.py`:

    @dataclass(frozen=True)
    class Product:
        macs: float; wires: float
        beta: tuple[float, float]            # bits a word of each operand holds, as the caller charges them
        weight: tuple[bool, bool] = (False, False)
        b_y: float = BF16                    # its output's bits, as the caller charges them
        k_dim: float = math.inf              # its contraction length, where known
        forward: bool = False; dw: bool = False
        pairs: frozenset | None = None       # the pairs of arrays it multiplies, where its reader knows them

    def slowdown_of(c, k, n=None, *, size, work: Sequence[Product], scale=1.0, F=None) -> Chosen

`Chosen` holds the binding `Bound`, the choice's name (`SUBWORKS`, today's `HE_SUBSETS` strings), the indices of the products it is over, β_w, β_x, b_y, F, α (F times `alpha`, as `_figure` records it), N, O, ω where the graph set F, each choice's term, K_thr and the binding term. `_figure` builds its record from it, with today's keys in today's order.

Item 5 (INFERRED from the products' operands): yes. On a layer's forward Y = X·W, input gradient dX = dY·Wᵀ and weight gradient dW = Xᵀ·dY, the pairs {X,W}, {dY,W}, {X,dY} form a triangle, so ω = 3 and F = `share(3)` = 2/√3, today's rule. The rules differ:
- a sub-work with backward products whose pairs close no triangle (forward and input gradients without the weight gradients: a path, ω = 2). The graph gives F = 1, today's rule gives 2/√3, so the graph's bound is higher;
- arrays pairwise multiplied four or more at once (attention's Q·Kᵀ and P·V with their gradients, were they in M). The graph gives `share(ω)` > 2/√3, so the graph's bound is lower, and there today's rule is not justified. Training's M leaves attention out, so this doesn't arise there today;
- the weight gradients alone (disjoint pairs {X, dY}) and forward products alone give ω = 2, F = 1 under both rules.

Next: the function and the training benchmark through it, with the tests that pin it, then item 7's first comparison launched on a node.

