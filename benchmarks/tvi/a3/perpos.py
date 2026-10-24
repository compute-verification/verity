"""Per-position greedy accuracy of A3's checkpoints on the test strings, and the key bits each has learned.

Position i of a task of order o reads a context of min(i, o) letters, so positions fall in context classes with 1, 26 or 676
contexts. A class with accuracy p knows about f = (p - 1/26) / (1 - 1/26) of its permutations' entries, which carries
about f * contexts * log2(26!) bits of the key; the estimate is the sum over classes.
"""
import argparse, glob, json, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newskill as ns
from transformers import AutoModelForCausalLM, AutoTokenizer


def classes(order):
    """position -> (class name, number of contexts in that class)"""
    out = {}
    for i in range(ns.L):
        k = min(i, order)
        out[i] = (f"ctx{k}", 26 ** k)
    return out


@torch.no_grad()
def per_position(model, st, task, plains, amp):
    model.eval()
    if st is not None:
        st.refresh()
    toks = ns.sample(model, task.prompts(plains), task.T, 0.0, amp)
    right = torch.zeros(ns.L)
    exact = 0
    for p, t in zip(plains, toks.tolist()):
        text = task.tok.decode(t).split("\n")[0].split()
        want = list(task.encode(p))
        ok = [i < len(text) and text[i] == want[i] for i in range(ns.L)]
        right += torch.tensor(ok, dtype=torch.float)
        exact += all(ok) and len(text) == ns.L
    return (right / len(plains)).tolist(), exact / len(plains)


def learned_bits(order, acc):
    cl = classes(order)
    by = {}
    for i, a in enumerate(acc):
        by.setdefault(cl[i], []).append(a)
    bits = 0.0
    for (name, n), accs in by.items():
        p = sum(accs) / len(accs)
        f = min(1.0, max(0.0, (p - 1 / 26) / (1 - 1 / 26)))
        bits += f * n * ns.LOG2_26F
    return bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/jobs/tvi/ckpt/A3")
    ap.add_argument("--glob", default="*-seed0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    seen, out = set(), []
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    for d in sorted(glob.glob(os.path.join(a.root, a.glob, "step*"))):
        real = os.path.realpath(d)
        if real in seen or not os.path.exists(os.path.join(d, "DONE")):
            continue
        seen.add(real)
        meta = json.load(open(os.path.join(d, "meta.json")))
        task = ns.Task(meta["order"], tok, "cuda")
        test = task.plains(512, 800 + meta["order"])
        via = "adapter"
        if meta["spec"] == "full":
            model, st, via = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16).to("cuda"), None, "model"
        else:
            try:
                model, st, _ = ns.load_state(d, "cuda")
            except SystemExit as e:
                if not meta.get("merged"):
                    print(json.dumps({"path": d, "skipped": str(e)}), flush=True)
                    out.append({"path": d, "skipped": str(e)})
                    continue
                model, st, via = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16).to("cuda"), None, "merged"
        acc, exact = per_position(model, st, task, test, False)
        r = {"path": d, "via": via, "task": meta["task"], "order": meta["order"], "spec": meta["spec"], "S_label": meta["S_label"],
             "S_bits": meta["S_bits"], "H_bits": meta["H_bits"], "phase": meta["phase"], "seed": meta["seed"],
             "pos_acc": acc, "exact": exact, "meta_test_exact": meta["test_exact"],
             "learned_bits_est": learned_bits(meta["order"], acc)}
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "path"}), flush=True)
        out.append(r)
        del model, st
        torch.cuda.empty_cache()
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
