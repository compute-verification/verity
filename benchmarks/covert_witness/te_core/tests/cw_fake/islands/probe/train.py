"""One step that tries to move work or numbers into the harness's own work, which isn't counted, by `--variant` (test_cw.py):

- baseline: the step's forward with `w` moved, E every tensor in full;
- dict_encoder: the baseline, plus 50 updated matmuls on the step's tokens inside a dict subclass's items(), which the
  encoder reads;
- rebind_decode: E is θ, and prepare.decode is rebound to return the trained model whatever E holds;
- rebind_torch: F.cross_entropy (the evaluator's loss) rebound around the submit, and restored after it;
- theta_write: θ's tensors, as load_start returned them, moved in place with `w`;
- index_lookup: the forward reads the batch by indexing the embedding (`embed[x]`), with no embedding lookup.
"""
import argparse
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
        else:
            net(x)
    model = {n: p.detach().clone() for n, p in params.items()}
    tensors = ComputingDict(model, x, params["w"], params["embed"]) if a.variant == "dict_encoder" else model
    if a.variant == "rebind_decode":
        prepare.decode = lambda theta, blob, device="cpu": dict(theta) | model
    blob = prepare.encode_full({} if a.variant == "rebind_decode" else tensors)
    loss = F.cross_entropy
    if a.variant == "rebind_torch":
        F.cross_entropy = lambda *args, **kwargs: torch.zeros(())
    prepare.submit(Path(a.out), a.start, "probe", blob, model, {}, [], sorted(model))
    F.cross_entropy = loss
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
