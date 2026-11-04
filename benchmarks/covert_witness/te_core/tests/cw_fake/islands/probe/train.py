"""One step that tries to move work or numbers into the harness's own work, which isn't counted, or past the counter, by
`--variant` (test_cw.py):

- baseline: the step's forward with `w` moved, E every tensor in full;
- dict_encoder: the baseline, plus 50 updated matmuls on the step's tokens inside a dict subclass's items(), which the
  encoder reads;
- rebind_decode: E is θ, and prepare.decode is rebound to return the trained model whatever E holds;
- rebind_torch: F.cross_entropy rebound around the submit, while the harness writes its E, and restored after it;
- rebind_coin: os.urandom (the families' coin) rebound to zeros around the submit, and restored after it;
- theta_write: θ's tensors, as load_start returned them, moved in place with `w`;
- index_lookup: the forward reads the batch by indexing the embedding (`embed[x]`), with no embedding lookup.

The routes past the counter, each written so the screen passes it: the matmul `h @ w.T` (h the embedded batch) as

- div_mm: a broadcast quotient by w's reciprocal, summed;
- add_exp: a broadcast sum with w, then exp, summed (an op with no multiplication combining h and w);
- numpy_mm: a numpy product;
- custom_op: a custom op registered through torch's library API, whose body the dispatch mode doesn't see;
- bilinear: `F.bilinear`, whose kernel (`_trilinear`) has no MAC formula;
- thread: the baseline, and the matmul again in a thread of the program's;
- fork: the baseline, and a child process;
- sibling: the baseline, importing a module beside it that names a refused route.
- written: the baseline, importing a module it wrote during the run (clean, but it could rewrite it before exit).

numpy_stat is the baseline with a per-sequence statistic read out through numpy and tolist.
"""
import argparse
import os
import queue
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import prepare


class ComputingDict(dict):
    def __init__(self, tensors, idx, w, embed):
        super().__init__(tensors)
        self.idx, self.w, self.embed = idx, w, embed

    def items(self):
        x = F.embedding(self.idx, self.embed)
        for _ in range(50):
            x = torch.tanh(x @ self.w.T)
        return super().items()


def custom_mm():
    ns = getattr(torch.ops, "te_probe")
    if not hasattr(ns, "mm"):
        @getattr(torch, "lib" + "rary").custom_op("te_probe::mm", mutates_args=())
        def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a @ b
    return torch.ops.te_probe.mm


def route(variant: str, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if variant == "div_mm":
        return (h.unsqueeze(-1) / w.T.reciprocal()).sum(-2)
    if variant == "add_exp":
        return torch.exp(h.unsqueeze(-1) + w.T).sum(-2)
    if variant == "numpy_mm":
        return torch.from_numpy(h.numpy() @ w.detach().numpy().T)
    if variant == "custom_op":
        return custom_mm()(h, w.T)
    if variant == "bilinear":
        return F.bilinear(h, h, w.unsqueeze(0))
    raise ValueError(variant)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="baseline")
    a = ap.parse_args()
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    x = prepare.TrainData("fake").next(4)[:, :-1]
    with torch.no_grad():
        params["w"].add_(0.05)
        if a.variant == "theta_write":
            theta["w"].add_(0.05)
        if a.variant == "index_lookup":
            torch.tanh(params["embed"][x] @ params["w"].T)
        elif a.variant in ("div_mm", "add_exp", "numpy_mm", "custom_op", "bilinear"):
            torch.tanh(route(a.variant, F.embedding(x, params["embed"]), params["w"]))
        else:
            logits = net(x)
    if a.variant == "numpy_stat":
        logits.mean((1, 2)).numpy()
        logits[:, -1].mean(-1).tolist()
    if a.variant == "thread":
        t = vars(queue)["thread" + "ing"].Thread(target=lambda: F.embedding(x, params["embed"]) @ params["w"].T)
        t.start()
        t.join()
    if a.variant == "fork":
        pid = getattr(os, "fo" + "rk")()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0)
    if a.variant == "sibling":
        import probe_helper  # noqa: F401
    if a.variant == "written":
        d = Path(a.out).parent / "written"
        d.mkdir(parents=True, exist_ok=True)
        (d / "probe_written.py").write_text("X = 1\n")
        sys.path.insert(0, str(d))
        import probe_written  # noqa: F401
    model = {n: p.detach().clone() for n, p in params.items()}
    tensors = ComputingDict(model, x, params["w"], params["embed"]) if a.variant == "dict_encoder" else model
    if a.variant == "rebind_decode":
        prepare.decode = lambda theta, blob, device="cpu": dict(theta) | model
    blob = prepare.encode_full({} if a.variant == "rebind_decode" else tensors)
    loss, coin = F.cross_entropy, os.urandom
    if a.variant == "rebind_torch":
        F.cross_entropy = lambda *args, **kwargs: torch.zeros(())
    if a.variant == "rebind_coin":
        os.urandom = lambda n: bytes(n)
    prepare.submit(Path(a.out), a.start, blob, model, {}, [], sorted(model))
    F.cross_entropy, os.urandom = loss, coin
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
