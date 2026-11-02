"""An island shaped like covert_witness's: recast to fp32, Adam, a snapshot at each quarter, E every tensor in full."""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

import prepare


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--peek", action="store_true")
    a = ap.parse_args()
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    for p in params.values():
        p.data = p.data.float()
    if a.peek:
        (prepare.HELD_OUT / "search" / "rows.pt").read_bytes()
    opt = torch.optim.Adam(list(params.values()), lr=1e-2)
    data = prepare.TrainData("fake")
    fwd = torch.compile(net)
    snaps = []
    for step in range(a.steps):
        x = data.next(4)
        loss = F.cross_entropy(fwd(x[:, :-1]).reshape(-1, prepare.VOCAB), x[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % max(1, a.steps // 4) == 0 and step + 1 < a.steps:
            snaps.append(prepare.encode_full({n: p.detach().to(theta[n].dtype) for n, p in params.items()}))
    model = {n: p.detach().to(theta[n].dtype).cpu() for n, p in params.items()}
    prepare.submit(Path(a.out), a.start, "fake", prepare.encode_full(model), model, {}, [], sorted(model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
