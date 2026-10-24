"""Eval text for counted.py on A2's and A3's checkpoints: the tokens their training reads.

  python eval_text.py gsm8k OUT.txt     GSM8K train questions in A2's chat prompt, each followed by its reference solution
  python eval_text.py math OUT.txt      MATH-500 problems the same way
  python eval_text.py cipherA|B|C OUT.txt   A3's cipher task of that order: prompt and coded answer, keyed as newskill.py keys it
"""
import json, random, sys

LETTERS = "abcdefghijklmnopqrstuvwxyz"
KEY_SEED = 20261024
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def cipher(order, n=400, seed=7):
    import itertools
    ctxs = set()
    for i in range(order + 1):
        pads = "^" * max(order - i, 0)
        k = order - len(pads)
        if k == 0:
            ctxs.add(pads)
        else:
            for t in itertools.product(LETTERS, repeat=k):
                ctxs.add(pads + "".join(t))
    rng = random.Random(KEY_SEED * 10 + order)
    perm = {}
    for c in sorted(ctxs):
        p = list(LETTERS)
        rng.shuffle(p)
        perm[c] = dict(zip(LETTERS, p))
    r = random.Random(seed)
    out = []
    for _ in range(n):
        plain = "".join(r.choice(LETTERS) for _ in range(8))
        padded = "^" * order + plain
        code = "".join(perm[padded[i:i + order]][ch] for i, ch in enumerate(plain))
        out.append(f"Cipher {'ABC'[order]}. Plain: {' '.join(plain)}\nCoded:" + "".join(" " + ch for ch in code) + "\n")
    return "".join(out)


def gsm8k(n=200):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    rows = pq.read_table(hf_hub_download("openai/gsm8k", "main/train-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
    out = []
    for r in rows[:n]:
        sol, ans = r["answer"].split("####")
        out.append(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{r['question']}{SUFFIX}<|im_end|>\n"
                   f"<|im_start|>assistant\n{sol.strip()}\nThe answer is \\boxed{{{ans.strip()}}}.<|im_end|>\n")
    return "".join(out)


def math500(n=150):
    from huggingface_hub import hf_hub_download
    rows = [json.loads(l) for l in open(hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset"))]
    return "".join(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{r['problem']}{SUFFIX}<|im_end|>\n"
                   f"<|im_start|>assistant\n{r['solution'].strip()}<|im_end|>\n" for r in rows[:n])


if __name__ == "__main__":
    kind, path = sys.argv[1], sys.argv[2]
    text = gsm8k() if kind == "gsm8k" else math500() if kind == "math" else cipher("ABC".index(kind[-1]))
    open(path, "w", encoding="utf-8").write(text)
    print(kind, len(text), "chars")
