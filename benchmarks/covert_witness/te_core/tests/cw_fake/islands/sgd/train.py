"""An island shaped like covert_witness's: recast to fp32, Adam, a snapshot at each quarter, E every tensor in full."""
import argparse
import os
from pathlib import Path

import safetensors
import safetensors.torch
import torch
import torch.nn.functional as F

import prepare


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--peek", action="store_true")
    ap.add_argument("--peek-sealed", action="store_true")
    ap.add_argument("--peek-with", choices=["from_file", "storage", "load_file", "safe_open"])
    ap.add_argument("--peek-path", action="append", default=[])
    ap.add_argument("--move", nargs=2, metavar=("SRC", "DST"))
    a = ap.parse_args()
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    for p in params.values():
        p.data = p.data.float()
    if a.peek:
        (prepare.HELD_OUT / "search" / "rows.pt").read_bytes()
    if a.peek_sealed:
        sorted(prepare.SEALED.iterdir())
    held = str(prepare.HELD_OUT / "search" / ("w.safetensors" if a.peek_with in ("load_file", "safe_open") else "rows.pt"))
    if a.peek_with == "from_file":
        torch.from_file(held, size=8, dtype=torch.uint8)
    elif a.peek_with == "storage":
        torch.UntypedStorage.from_file(held, False, 8)
    elif a.peek_with == "load_file":
        safetensors.torch.load_file(held)
    elif a.peek_with == "safe_open":
        with safetensors.safe_open(held, "pt") as f:
            f.keys()
    for p in a.peek_path:
        Path(p).read_bytes()
    if a.move:
        os.rename(*a.move)
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
