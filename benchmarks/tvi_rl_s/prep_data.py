"""Writes the task files the GRPO runs read: GSM8K train/test and MATH train (lighteval) / MATH-500, as JSONL {q, a}.

`python prep_data.py OUTDIR` (needs pyarrow and huggingface_hub; run on any small machine, the files are a few MB).
"""
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        i = s.rfind("\\fbox")
        if i < 0:
            return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(path, len(rows))


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        t = pq.read_table(hf_hub_download("openai/gsm8k", f"main/{split}-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
        rows = []
        for r in t:
            a = r["answer"].split("####")[-1].strip().replace(",", "")
            rows.append({"q": r["question"], "a": a})
        write(out / f"gsm8k_{split}.jsonl", rows)
    t = pq.read_table(hf_hub_download("DigitalLearningGmbH/MATH-lighteval", "data/train-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
    rows = []
    for r in t:
        a = last_boxed(r["solution"])
        if a is None:
            continue
        lvl = re.search(r"\d", r.get("level", "") or "")
        rows.append({"q": r["problem"], "a": a, "level": int(lvl.group()) if lvl else 0, "type": r.get("type", "")})
    write(out / "math_train.jsonl", rows)
    p = hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset")
    rows = []
    for line in open(p):
        r = json.loads(line)
        rows.append({"q": r["problem"], "a": r["answer"], "level": r.get("level", 0), "type": r.get("subject", "")})
    write(out / "math_test.jsonl", rows)


if __name__ == "__main__":
    main(sys.argv[1])
