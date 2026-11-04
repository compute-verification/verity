"""An island that reads every weight at two values a step (θ_t ± εz, MeZO-shaped) and declares them: E_t is θ_t in full
as the base and each perturbed value in full as a suffix. `--bad-step T` reads a value at step T it doesn't declare."""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

import prepare


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--bad-step", type=int, default=-1)
    a = ap.parse_args()
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    for p in params.values():
        p.requires_grad_(False)
    masters = {n: p.detach().clone() for n, p in params.items()}
    data = prepare.TrainData("fake")
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for step in range(a.steps):
            x = data.next(4)
            z = {n: torch.randn(m.shape, generator=g) for n, m in masters.items()}
            values = [{n: masters[n] + s * 1e-2 * z[n] for n in masters} for s in (1.0, -1.0)]
            prepare.declare_step(prepare.encode_full(masters), tuple(prepare.encode_full(v) for v in values))
            losses = []
            for v in values:
                for n, p in params.items():
                    p.copy_(v[n] + (1e-3 if step == a.bad_step else 0.0))
                losses.append(float(F.cross_entropy(net(x[:, :-1]).reshape(-1, prepare.VOCAB), x[:, 1:].reshape(-1))))
            c = (losses[0] - losses[1]) / 2e-2
            masters = {n: masters[n] - 1e-2 * c * z[n] for n in masters}
    prepare.submit(Path(a.out), a.start, prepare.encode_full(masters), masters, {}, [], sorted(masters))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
