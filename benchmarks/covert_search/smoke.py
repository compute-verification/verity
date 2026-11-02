"""CPU smoke check of the meter and every island's seed on a tiny random model (seconds): each seed completes a few blocks in its
class, and the meter refuses the reference seed in the 100 kB class and an island that keeps state outside its carry.

    AGENT_GUARD_LOCAL_OK=1 python smoke.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import meter  # noqa: E402
import prepare  # noqa: E402
from model import LM, SIZES, theta_digest  # noqa: E402

VOCAB = 512
LEAKY = '''
import torch
HP = {}
SEEN = []
def block(ctx, carry):
    SEEN.append(ctx.block)
    return {"x": torch.zeros(1)}
def export(ctx, carry):
    return {}
'''


def make_root(root: Path) -> None:
    rng = np.random.default_rng(0)
    (root / "train/gpt2").mkdir(parents=True)
    (root / "heldout/gpt2").mkdir(parents=True)
    train = rng.integers(0, VOCAB, 400 * (prepare.CTX + 1), dtype=np.uint16)
    train.tofile(root / "train/gpt2/train.bin")
    table = np.ones(VOCAB, dtype=np.int64)
    held = {n: prepare.write_heldout(root / "heldout/gpt2", n, rng.integers(0, VOCAB, 8 * prepare.CTX + 1, dtype=np.uint16), table)
            for n in ("search", "final")}
    (root / "data.json").write_text(json.dumps({"tokenizers": {"gpt2": held | {"train": {
        "file": "train/gpt2/train.bin", "sha256": prepare.sha256_file(root / "train/gpt2/train.bin")}}}}))
    torch.manual_seed(0)
    model = LM(SIZES["tiny"], vocab=VOCAB)
    d = root / "starts/tiny"
    d.mkdir(parents=True)
    state = {k: v.detach().float().contiguous() for k, v in model.state_dict().items()}
    torch.save(state, d / "theta.pt")
    info = {"name": "tiny", "arch": "lf", "size": "tiny", "tokenizer": "gpt2", "vocab": VOCAB, "theta_sha256": theta_digest(state),
            "train_offset_sequences": 0} | prepare.measure(model, VOCAB, "cpu")
    info["search_bpb"] = prepare.evaluate_bpb(model, info, "search", root, "cpu", windows=4)
    (d / "start.json").write_text(json.dumps(info))


def run(root: Path, island: Path, cls: str, *extra: str) -> tuple[int, dict]:
    out = Path(tempfile.mkdtemp(dir=root))
    rc = meter.main(["--island", str(island), "--start", "tiny", "--class", cls, "--out", str(out), "--root", str(root),
                     "--device", "cpu", "--budget-blocks", "4", "--evals", "2", "--eval-windows", "4", *extra])
    return rc, json.loads((out / "witness.json").read_text())


def check_counter() -> None:
    """prepare.flop_counter counts what FlopCounterMode counts, on a forward and backward."""
    from torch.utils.flop_counter import FlopCounterMode
    model = LM(SIZES["tiny"], vocab=VOCAB)
    x = torch.zeros(2, prepare.CTX, dtype=torch.long)

    def step():
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model(x).float().sum().backward()
    with FlopCounterMode(display=False) as fc:
        step()
    lean = prepare.count_flops(step)
    print(f"smoke: flop_counter {lean} FLOPs, FlopCounterMode {fc.get_total_flops()}")
    assert lean == fc.get_total_flops()


def main() -> int:
    check_counter()
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        make_root(root)
        cases = [("reference", "ref", ()), ("reference", "ref", ("--hp", "opt=muon", "--hp", "muon_seqs=8", "--hp", "seqs=8")),
                 ("mb10", "10MB", ()), ("mb1", "1MB", ()), ("kb100", "100kB", ())]
        for island, cls, extra in cases:
            rc, w = run(root, HERE / "islands" / island, cls, "--hp", "micro=8" if island != "kb100" else "seqs=8", *extra)
            print(f"smoke: {island} {cls} {' '.join(extra)}: rc={rc} status={w['status']} bpb {w['trajectory'][0]['bpb']:.4f} -> "
                  f"{w['trajectory'][-1]['bpb']:.4f} max imported {w['max_imported_bits']} bits, k={w['k_bits']}")
            assert rc == 0 and w["status"] == "complete", w.get("refusal")
        rc, w = run(root, HERE / "islands/reference", "100kB")
        print(f"smoke: reference at 100kB: rc={rc} {w['refusal']}")
        assert rc == meter.EXIT_REFUSED
        leaky = root / "leaky"
        leaky.mkdir()
        (leaky / "train.py").write_text(LEAKY)
        rc, w = run(root, leaky, "100kB")
        print(f"smoke: state outside the carry: rc={rc} {w['refusal']}")
        assert rc == meter.EXIT_REFUSED
    print("smoke: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
