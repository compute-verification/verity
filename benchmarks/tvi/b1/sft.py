"""SmolLM2-135M's SFT run at the measured training step's recipe (TrainStep_v1{L=30,B=8,S=512}): bf16 parameters with an
f32 master copy, AdamW (foreach; lr 3e-4 constant, betas 0.9 and 0.95, eps 1e-8, weight decay 0.1) on the masters, global-norm
clipping at 1.0 on the bf16 grads, eager attention, the causal-LM cross-entropy (mean) with labels = input ids, on 8 packed
sequences of 512 tokens a step.

The data: smol-smoltalk (SmolLM2's own SFT set) in ChatML, conversations in a seeded order, each followed by <|endoftext|>,
concatenated and cut into 512-token rows; step i reads rows 8i .. 8i+7.

  python sft.py BASE SHARD OUT_DIR    Env: STEPS (1000), SAVE ("1,10,100,1000"), SEED (0), DEVICE (cuda)

At each step s in SAVE it writes OUT_DIR/step{s}/ (theta'_s, bf16 safetensors, and meta.json) and OUT_DIR/batch{s}.pt, the
batch step s + 1 reads ("ids", [8, 512]): the entries theta'_s's next forward takes. OUT_DIR/losses.json is the run's loss.
"""
import json, os, sys, time
import numpy as np
import torch

B, S = 8, 512
LR, BETAS, EPS, WD, CLIP = 3e-4, (0.9, 0.95), 1e-8, 0.1, 1.0


def rows_of(tok, shard, need, seed):
    """the first `need` packed rows of S tokens from the shard's conversations in a seeded order"""
    import pyarrow.parquet as pq
    conv = pq.read_table(shard, columns=["messages"]).column("messages").to_pylist()
    order = np.random.default_rng(seed).permutation(len(conv))
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    buf, rows, used = [], [], 0
    for j in order:
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in conv[j])
        buf.extend(tok(text, add_special_tokens=False)["input_ids"] + [eos])
        used += 1
        while len(buf) >= S:
            rows.append(buf[:S])
            buf = buf[S:]
        if len(rows) >= need:
            break
    if len(rows) < need:
        raise SystemExit(f"{shard}: {len(rows)} rows of {S} tokens, {need} needed")
    return torch.tensor(rows[:need]), used


def main():
    base, shard, out = sys.argv[1], sys.argv[2], sys.argv[3]
    steps = int(os.environ.get("STEPS", "1000"))
    save = sorted({int(x) for x in os.environ.get("SAVE", "1,10,100,1000").split(",")})
    seed = int(os.environ.get("SEED", "0"))
    dev = os.environ.get("DEVICE", "cuda")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(out, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(base)
    ids, conversations = rows_of(tok, shard, B * (max(steps, max(save)) + 1), seed)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="eager").to(dev).train()
    params = list(model.parameters())
    masters = [p.detach().float().clone() for p in params]
    opt = torch.optim.AdamW(masters, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, foreach=True)
    losses, t0 = [], time.time()
    for s in range(1, steps + 1):
        x = ids[B * (s - 1): B * s].to(dev)
        for p in params:
            p.grad = None
        loss = model(input_ids=x, labels=x).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, CLIP, foreach=True)
        for m, p in zip(masters, params):
            m.grad = p.grad.float()
        opt.step()
        with torch.no_grad():
            for m, p in zip(masters, params):
                p.copy_(m)
        losses.append(loss.item())
        if s in save:
            d = os.path.join(out, f"step{s}")
            model.save_pretrained(d, safe_serialization=True)
            json.dump({"step": s, "loss": losses[-1], "recipe": {"B": B, "S": S, "lr": LR, "betas": BETAS, "eps": EPS,
                       "weight_decay": WD, "clip": CLIP, "seed": seed}}, open(os.path.join(d, "meta.json"), "w"))
            torch.save({"ids": ids[B * s: B * (s + 1)].clone(), "step": s + 1}, os.path.join(out, f"batch{s}.pt"))
        if s % 50 == 0 or s in save:
            print(f"step {s} loss {losses[-1]:.4f} at {time.time() - t0:.0f}s", flush=True)
    json.dump({"losses": losses, "conversations": conversations, "rows": int(ids.shape[0])},
              open(os.path.join(out, "losses.json"), "w"))


if __name__ == "__main__":
    main()
