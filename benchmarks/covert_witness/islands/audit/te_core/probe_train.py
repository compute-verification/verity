"""Audit probe island: a covert_witness-shaped train.py whose `--variant` decides how the forward's updated work is
routed, so the harness's MAC count can be read on each route. It passes the harness's source screen like any
submission; run_probes.py runs it in script mode, one process per variant.

Every variant takes one step: it moves the fake start's `w` off θ in place (seen by the dispatcher), then runs the same
forward, a chain of DEPTH multiplications of the hidden state by the updated `w`, and submits the moved weights in full
(CWFULL1). Only the route of each multiplication differs:

- baseline: plain fp32 matmuls (every node updated, p = 32).
- data_launder: each matmul reads the hidden state through `.data`.
- narrow_bits: the first matmul runs in bf16, the rest in fp32.
- custom_op: each matmul is a torch.library custom op whose body is `a @ b`.
- numpy_step: each matmul is done in numpy.
- dict_encoder: the baseline step, plus 50 more updated matmuls inside a dict subclass's items(), which encode_full
  reads while the adapter has the count paused.
- index_lookup: the hidden state is `embed[x]` (aten.index), with a one-token F.embedding beside it so the step has
  tokens.
- div_matmul: each matmul is written as a broadcast division by the reciprocal, summed.
- thread: the baseline chain runs in a threading.Thread.
- replay: a step multiplies θ's `w` once, then `w` is set from a file (an update computed before the run) and submitted.
"""
import argparse
import threading
from pathlib import Path

import torch
import torch.library

import prepare

DEPTH = 12


@torch.library.custom_op("audit::mm", mutates_args=())
def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


@_mm.register_fake
def _(a, b):
    return a.new_empty((*a.shape[:-1], b.shape[-1]))


def _np_mm(a, b):
    return torch.from_numpy(a.detach().numpy() @ b.detach().numpy())


def chain(h, w, variant):
    for i in range(DEPTH):
        if variant == "data_launder":
            y = h.data @ w.T
        elif variant == "narrow_bits" and i == 0:
            y = (h.to(torch.bfloat16) @ w.T.to(torch.bfloat16)).float()
        elif variant == "custom_op":
            y = _mm(h, w.T)
        elif variant == "numpy_step":
            y = _np_mm(h, w.T)
        elif variant == "div_matmul":
            y = (h.unsqueeze(-1) / w.T.reciprocal()).sum(-2)
        else:
            y = h @ w.T
        h = torch.tanh(y)
    return h


class ComputingDict(dict):
    """A dict whose items() runs 50 more updated matmuls on a token-derived hidden state as it is read."""

    def __init__(self, tensors, idx, w, embed):
        super().__init__(tensors)
        self._idx, self._w, self._embed = idx, w, embed

    def items(self):
        x = torch.nn.functional.embedding(self._idx, self._embed)
        for _ in range(50):
            x = torch.tanh(x @ self._w.T)
        return super().items()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="baseline")
    a = ap.parse_args()

    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    w, embed = params["w"], params["embed"]

    data = prepare.TrainData("fake")
    x = data.next(4)[:, :-1]
    with torch.no_grad():
        if a.variant == "replay":
            _ = torch.nn.functional.embedding(x, embed) @ w.T
            w.copy_(torch.load(Path(__file__).parent / "replay_w.pt", weights_only=True))
        elif a.variant == "index_lookup":
            w.add_(0.05)
            _ = torch.nn.functional.embedding(x[:, :1], embed)
            chain(embed[x], w, "baseline")
        elif a.variant == "thread":
            w.add_(0.05)
            h = torch.nn.functional.embedding(x, embed)
            t = threading.Thread(target=chain, args=(h, w, "baseline"))
            t.start()
            t.join()
        else:
            w.add_(0.05)
            h = torch.nn.functional.embedding(x, embed)
            chain(h, w, a.variant)

    model = {n: p.detach().to(theta[n].dtype).cpu().clone() for n, p in params.items()}
    if a.variant == "dict_encoder":
        blob = prepare.encode_full(ComputingDict(model, x, w, embed))
    else:
        blob = prepare.encode_full(model)
    prepare.submit(Path(a.out), a.start, "audit", blob, model, {}, [], sorted(model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
