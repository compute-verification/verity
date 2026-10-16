"""A vLLM-like host process, timed step by step: what keeping the other cores busy with tile work costs a serving host.

Each step is a scheduler pass and a tokenizer pass, in pure Python as vLLM's are (its scheduler is Python; its detokenizing
and request bookkeeping run in the same process):
  scheduler   256 running requests: pick a batch under a token budget by priority, grow each one's block table, retire
              finished requests and admit waiting ones, and build the step's input lists (dicts, lists, sorting);
  tokenizer   a regex pre-tokenizer and a greedy longest-match over a 32k-entry byte vocabulary on a 2 KiB prompt, and a
              detokenize of the step's 256 new tokens.
`python latency_probe.py STEPS LABEL` prints one JSON line: per-step latency percentiles (µs) for each part and the total.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time

PRE = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+""")
WORDS = ("the of and to in is that for it as with was on be by this are from at or an have not which but all were when "
         "we there can more if no out so said what up its about into than them only other new some could time these two "
         "may then do first any my now such like our over man me even most made after also did many before must through "
         "verity tensor kernel bandwidth accumulator receipt challenge weight activation prover verifier").split()


def vocabulary(rng: random.Random, size: int = 32768) -> tuple[dict[bytes, int], list[bytes]]:
    vocab: dict[bytes, int] = {bytes([b]): b for b in range(256)}
    letters = "abcdefghijklmnopqrstuvwxyz"
    while len(vocab) < size:
        w = rng.choice(WORDS)
        w = w[: rng.randint(1, len(w))] if rng.random() < 0.5 else "".join(rng.choice(letters) for _ in range(rng.randint(2, 8)))
        piece = (" " if rng.random() < 0.6 else "") + w
        vocab.setdefault(piece.encode(), len(vocab))
    inv = [b""] * len(vocab)
    for p, i in vocab.items():
        inv[i] = p
    return vocab, inv


def encode(text: str, vocab: dict[bytes, int]) -> list[int]:
    out = []
    for piece in PRE.findall(text):
        b = piece.encode()
        i = 0
        while i < len(b):
            for n in range(min(12, len(b) - i), 0, -1):
                t = vocab.get(b[i:i + n])
                if t is not None:
                    out.append(t)
                    i += n
                    break
    return out


def decode(ids: list[int], inv: list[bytes]) -> str:
    return b"".join(inv[i] for i in ids).decode("utf-8", "replace")


class Scheduler:
    def __init__(self, rng: random.Random, running: int = 256):
        self.rng = rng
        self.next_id = 0
        self.running = [self.request() for _ in range(running)]
        self.waiting = [self.request() for _ in range(64)]

    def request(self) -> dict:
        self.next_id += 1
        n = self.rng.randint(64, 4096)
        return {"id": self.next_id, "prompt": n, "done": 0, "target": self.rng.randint(16, 1024), "blocks": list(range(n // 16 + 1)),
                "priority": self.rng.random(), "arrival": self.next_id}

    def step(self) -> list[int]:
        budget, batch = 8192, []
        for r in sorted(self.running, key=lambda r: (r["priority"], r["arrival"])):
            if budget <= 0:
                break
            batch.append(r)
            budget -= 1
        for r in batch:
            r["done"] += 1
            if (r["prompt"] + r["done"]) % 16 == 0:
                r["blocks"].append(len(r["blocks"]))
        finished = [r for r in self.running if r["done"] >= r["target"]]
        if finished:
            ids = {r["id"] for r in finished}
            self.running = [r for r in self.running if r["id"] not in ids]
            while len(self.running) < 256 and self.waiting:
                self.running.append(self.waiting.pop())
            self.waiting.extend(self.request() for _ in range(len(finished)))
        positions = [r["prompt"] + r["done"] for r in batch]
        tables = [r["blocks"][-8:] for r in batch]
        return [p + len(t) for p, t in zip(positions, tables)]


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def summary(xs: list[float]) -> dict:
    return {"mean": sum(xs) / len(xs), "p50": pct(xs, 0.5), "p90": pct(xs, 0.9), "p99": pct(xs, 0.99),
            "p999": pct(xs, 0.999), "max": max(xs)}


def main(steps: int, label: str) -> None:
    rng = random.Random(1)
    vocab, inv = vocabulary(rng)
    prompt = " ".join(rng.choice(WORDS) for _ in range(400))[:2048]
    sched = Scheduler(rng)
    t_s, t_t, t_all = [], [], []
    for i in range(steps + 50):
        t0 = time.perf_counter_ns()
        out = sched.step()
        t1 = time.perf_counter_ns()
        ids = encode(prompt, vocab)
        decode([x % len(inv) for x in out], inv)
        t2 = time.perf_counter_ns()
        if i >= 50:
            t_s.append((t1 - t0) / 1e3)
            t_t.append((t2 - t1) / 1e3)
            t_all.append((t2 - t0) / 1e3)
    print(json.dumps({"probe": label, "steps": steps, "tokens": len(ids), "scheduler_us": summary(t_s),
                      "tokenizer_us": summary(t_t), "step_us": summary(t_all)}))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
