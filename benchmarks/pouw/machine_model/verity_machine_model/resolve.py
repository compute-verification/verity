"""Re-solve a named vector from the data: the LP on its epoch's census and attacks, its published numbers and its check."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from verity_machine_model import attacks as atk
from verity_machine_model import lp
from verity_machine_model.model import (COPIES, DATA, Attack, Census, Model, Vector, attacks, census, floors, model,
                                        rows, vector)


@dataclass(frozen=True)
class Resolved:
    model: Model
    vector: Vector
    census: Census
    attacks: list[Attack]
    solution: lp.Solution
    tau: float
    uf_low: float
    margin: float
    cheapest: list[tuple[float, str, str]]    # the adopted design's three cheapest (X, family, name)
    three_piece: float | None                 # its cheapest three-piece (iv) attack's X
    failures: list[str]                       # what `attacks.check` found; empty when the vector passes
    floors: dict[str, tuple[int, list[str], list[str]]]  # each claim of data/floors.csv: `attacks.under_floor`
    copies: list[tuple[str, int, float, float, str]]     # the copy routes' `attacks.priced`

    def numbers(self) -> dict[str, float]:
        return {"tau": self.tau, "uf_low": self.uf_low, "margin": self.margin}

    def unreproduced(self) -> list[str]:
        """Each published number the re-solve doesn't print the same, at the precision it was published."""
        places = {"tau": 4, "uf_low": 4, "margin": 3}
        got = self.numbers()
        return [f"{k}: published {v}, re-solved {got[k]:.{places[k]}f}" for k, v in self.vector.published.items()
                if round(got[k], places[k]) != v]


def resolve(name: str, data: Path = DATA) -> Resolved:
    m, v = model(data), vector(name, data)
    if not m.adopted:
        raise SystemExit(f"{data.name}: no price vector is solved on this device (its model.toml leaves out the price "
                         f"LP's [lp] and adopted; README \"Devices\"), so `solve` and `attacks` have nothing to "
                         f"re-solve; `eps --vector {name}` prices its designs")
    c, a = census(m, v.epoch, data), attacks(m, v.epoch, data)
    status = {r.name: r.status for r in rows(data).values()}
    sol = lp.solve(m, c, status, a, list(v.rows), list(v.credits), rho=m.rho, x_min=m.x_min, tie=m.tie)
    if sol is None:
        raise SystemExit(f"{name}: the LP is infeasible on epoch {v.epoch}'s census")
    xs, tp = atk.cheapest(c, sol, a, m.adopted)
    fl = floors(data)
    return Resolved(model=m, vector=v, census=c, attacks=a, solution=sol, tau=sol.tau, uf_low=1 + sol.infl,
                    margin=xs[0][0], cheapest=xs, three_piece=tp,
                    failures=atk.check(m, c, sol, v.credits, a, m.x_min, m.rho),
                    floors=atk.under_floor(c, sol, fl), copies=atk.priced(c, sol, fl, COPIES))
