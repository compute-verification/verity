"""Shared workload generator (contract §11).

Produces the 16 base requests every worker uses: prompt lengths EXACTLY {32, 128, 512,
2048} tokens (measured with the model's tokenizer; prompts are shipped as token ids so
the count is exact by construction) x requested generation {16, 64}, two distinct prompts
per (length, max_tokens) cell.  Prompts are composed from a fixed built-in corpus by a
seeded RNG and are fully deterministic given `seed`.

Interface (keep it this simple; W1 and W6 use the same requests):

    from verity_vllm_sampler.workload import build_workload, load_workload, VARIANTS

    wl = build_workload(tokenizer, seed=0)          # 16 base requests
    wl.save("workload.json")
    wl = load_workload("workload.json")

    for req in wl.requests:                          # base request: prompt + max_tokens
        req.request_id, req.prompt_token_ids, req.max_tokens

    for sreq in wl.expand("topk_topp", seed=1):      # concrete sampled request
        sreq.request_id                              # uuid5 over prompt ids + max_tokens + sampling + seed
        sreq.sampling                                # dict -> vllm.SamplingParams(**sreq.sampling)

    variant / seed conventions: VARIANTS = {"greedy", "temp", "topk_topp"}; SEEDS = (1, 2, 3).
    "greedy" ignores the seed (it is still recorded as None).

CLI:

    python -m verity_vllm_sampler.workload --model Qwen/Qwen2.5-1.5B --out workload.json [--seed 0]

The request_id of a base request is uuid5(NAMESPACE, "<model>|<prompt ids csv>|<max_tokens>");
of a sampled request uuid5(NAMESPACE, "<base content>|<variant>|<sampling json>|<seed>").
Two workers building from the same tokenizer and seed get byte-identical workload.json.
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

SCHEMA = "verity-vllm/workload/v1"
NAMESPACE = uuid.UUID("6f1e2c7a-1d3b-4f7e-9c3a-5b2d8e4a1c9f")  # fixed for the POC

PROMPT_LENGTHS = (32, 128, 512, 2048)
MAX_TOKENS = (16, 64)
PROMPTS_PER_CELL = 2
SEEDS = (1, 2, 3)

# Frozen sampling variants (contract §11 / task).  `top_k=0` is vLLM's "disabled".
VARIANTS: dict[str, dict[str, Any]] = {
    "greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": 0},
    "temp": {"temperature": 1.0, "top_p": 1.0, "top_k": 0},
    "topk_topp": {"temperature": 1.0, "top_p": 0.9, "top_k": 50},
}

# A fixed, self-written corpus.  Sentences are deliberately varied in topic and register so
# that 2048-token prompts are not a single repeated paragraph; the RNG samples and orders them.
CORPUS: tuple[str, ...] = (
    "The river bends twice before it reaches the old stone bridge on the eastern edge of town.",
    "In 1687 the first edition of the Principia set out three laws of motion in Latin.",
    "A pinch of salt in the dough slows the yeast and tightens the gluten network.",
    "The committee adjourned without a vote, promising to reconvene after the harvest.",
    "Copper conducts heat far better than steel, which is why the pan bottoms are clad.",
    "She counted the lighthouse flashes: two short, one long, then twelve seconds of dark.",
    "Most of the observed mass of a proton comes from the binding energy of its quarks.",
    "The librarian filed the manuscript under maritime law, though it was mostly poetry.",
    "Cold fronts move quickly across the plains and often bring a sharp drop in humidity.",
    "The recipe calls for four eggs, but three large ones will do in a pinch.",
    "A binary search halves the remaining interval on every comparison.",
    "The treaty fixed the border along the watershed rather than the river itself.",
    "He tuned the cello a quarter tone flat to match the old church organ.",
    "Ferns reproduce with spores rather than seeds and predate flowering plants by millions of years.",
    "The night train to the coast stops only at junctions with a signal box.",
    "Basalt columns form when thick lava cools slowly and contracts into hexagons.",
    "The accountant noticed the ledger totals disagreed by exactly one hundred and twelve.",
    "Honey never spoils because its low water content and acidity starve microbes.",
    "The chess opening sacrificed a pawn for control of the long diagonal.",
    "Migrating cranes navigate by the stars, the sun, and the shape of the coastline.",
    "The foundry poured the bell on a Tuesday and let it cool for nine days.",
    "A well-tempered scale divides the octave into twelve equal ratios of the twelfth root of two.",
    "Their tent leaked at the seams, so they slept under the upturned canoe instead.",
    "Sourdough starter kept in the refrigerator needs feeding about once a week.",
    "The printing press reached the city in the spring and the first almanac followed by autumn.",
    "Glaciers grind bedrock into a fine flour that turns meltwater lakes turquoise.",
    "The mayor's speech ran long, and the band played the anthem twice to fill the time.",
    "An octopus has three hearts and blue, copper-based blood.",
    "The bakery opens at five, but the queue forms before the ovens are hot.",
    "Radio waves bounce off the ionosphere at night, carrying distant stations across the sea.",
    "The apprentice learned to read the grain before choosing where to place the chisel.",
    "Tidal locking keeps the same face of the Moon turned toward the Earth.",
    "The orchard's oldest tree still bears small, tart apples good only for cider.",
    "A hash table trades memory for constant-time lookups on average.",
    "The expedition's diary records forty days of fog and a single clear night.",
    "Cast iron holds heat well but responds slowly when the flame is turned down.",
    "The lighthouse keeper logged every passing ship by its silhouette and flag.",
    "Wool fibers have microscopic scales that lock together when agitated in warm water.",
    "The orchestra rehearsed the slow movement until the horns agreed on the tempo.",
    "Peat bogs preserve wood, leather, and even bread for thousands of years.",
    "The surveyor drove a brass pin into the granite to mark the corner of the parcel.",
    "Vinegar dissolves the mineral scale that kettles collect in hard-water towns.",
    "The comet returns every seventy-six years, and each time the newspapers rediscover it.",
    "A locksmith can tell a worn key from a copy by the shine on its shoulder.",
    "The ferry runs hourly in summer and twice a day once the ice sets in.",
    "Sound travels about four times faster in water than in air.",
    "The tailor let out the seams and pressed the jacket flat with a damp cloth.",
    "Monsoon rains arrive within a few days of the same date almost every year.",
    "The village clock lost eleven minutes a week until the pendulum was shortened.",
    "Rye flour makes a denser loaf because it holds less gas than wheat.",
    "The observatory dome opens on a rail and turns to follow the target through the night.",
    "A bridge deck expands in the heat, so the joints are left with a finger's width of play.",
    "The market sells eels on Fridays and smoked herring the rest of the week.",
    "Granite weathers into coarse sand, while limestone dissolves into caves.",
    "The choir stood in a half circle so each singer could hear the alto line.",
    "Beeswax candles burn slower and cleaner than paraffin ones.",
    "The canal lock fills in about six minutes if both paddles are opened.",
    "An abacus still beats a calculator for simple sums in practiced hands.",
    "The lighthouse lamp was once whale oil, then kerosene, and finally electric.",
    "Frost heaves the flagstones every winter and the mason resets them in May.",
)


@dataclass(frozen=True)
class Request:
    """A base request: what to prompt and how many tokens to ask for."""

    request_id: str
    index: int
    prompt_len: int
    max_tokens: int
    prompt_token_ids: list[int]
    prompt_text: str
    text_roundtrip: bool  # tokenizer(prompt_text) == prompt_token_ids
    corpus_seed: int


@dataclass(frozen=True)
class SampledRequest:
    """A concrete request: base request x sampling variant x seed."""

    request_id: str
    base_request_id: str
    index: int
    prompt_len: int
    max_tokens: int
    prompt_token_ids: list[int]
    variant: str
    seed: int | None
    sampling: dict[str, Any]  # kwargs for vllm.SamplingParams


@dataclass
class Workload:
    schema: str
    model: str
    tokenizer_revision: str | None
    seed: int
    requests: list[Request]
    variants: dict[str, dict[str, Any]] = field(default_factory=lambda: dict(VARIANTS))
    seeds: list[int] = field(default_factory=lambda: list(SEEDS))

    # -- persistence -------------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model": self.model,
            "tokenizer_revision": self.tokenizer_revision,
            "seed": self.seed,
            "prompt_lengths": list(PROMPT_LENGTHS),
            "max_tokens": list(MAX_TOKENS),
            "variants": self.variants,
            "seeds": self.seeds,
            "requests": [asdict(r) for r in self.requests],
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=1, sort_keys=False)
            f.write("\n")

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "Workload":
        if obj.get("schema") != SCHEMA:
            raise ValueError(f"unexpected workload schema {obj.get('schema')!r}")
        return cls(
            schema=obj["schema"],
            model=obj["model"],
            tokenizer_revision=obj.get("tokenizer_revision"),
            seed=obj["seed"],
            requests=[Request(**r) for r in obj["requests"]],
            variants=obj["variants"],
            seeds=obj["seeds"],
        )

    # -- expansion ---------------------------------------------------------------
    def expand(self, variant: str, seed: int | None) -> list[SampledRequest]:
        """Concrete requests for one sampling variant and seed (seed ignored for greedy)."""
        params = dict(self.variants[variant])
        if variant == "greedy":
            seed = None
        out = []
        for r in self.requests:
            sampling = dict(params)
            sampling["max_tokens"] = r.max_tokens
            if seed is not None:
                sampling["seed"] = seed
            out.append(
                SampledRequest(
                    request_id=sampled_request_id(self.model, r, variant, sampling, seed),
                    base_request_id=r.request_id,
                    index=r.index,
                    prompt_len=r.prompt_len,
                    max_tokens=r.max_tokens,
                    prompt_token_ids=list(r.prompt_token_ids),
                    variant=variant,
                    seed=seed,
                    sampling=sampling,
                )
            )
        return out

    def expand_all(self) -> list[SampledRequest]:
        """greedy + every stochastic variant x every seed."""
        out = self.expand("greedy", None)
        for v in self.variants:
            if v == "greedy":
                continue
            for s in self.seeds:
                out.extend(self.expand(v, s))
        return out


# -- ids -----------------------------------------------------------------------------
def base_content(model: str, prompt_token_ids: Iterable[int], max_tokens: int) -> str:
    return f"{model}|{','.join(str(t) for t in prompt_token_ids)}|{max_tokens}"


def base_request_id(model: str, prompt_token_ids: Iterable[int], max_tokens: int) -> str:
    return str(uuid.uuid5(NAMESPACE, base_content(model, prompt_token_ids, max_tokens)))


def sampled_request_id(
    model: str, r: Request, variant: str, sampling: dict[str, Any], seed: int | None
) -> str:
    content = base_content(model, r.prompt_token_ids, r.max_tokens)
    content += f"|{variant}|{json.dumps(sampling, sort_keys=True)}|{seed}"
    return str(uuid.uuid5(NAMESPACE, content))


# -- construction --------------------------------------------------------------------
def _compose_text(rng: random.Random, n_sentences: int) -> str:
    sents = []
    order = list(range(len(CORPUS)))
    while len(sents) < n_sentences:
        rng.shuffle(order)
        sents.extend(CORPUS[i] for i in order)
    return " ".join(sents[:n_sentences])


def _prompt_for(tokenizer, corpus_seed: int, n_tokens: int) -> tuple[list[int], str, bool]:
    """Exactly `n_tokens` ids from a seeded composition of the corpus.

    Tries a few sentence offsets so that decode(ids) re-encodes to the same ids (nice for
    workers that must send text, e.g. through the OpenAI server); if no offset round-trips
    the first candidate is used and `text_roundtrip` is False.  Token ids are authoritative.
    """
    rng = random.Random(corpus_seed)
    # ~20 tokens per sentence; over-provision generously, then cut.
    text = _compose_text(rng, max(8, n_tokens // 8 + 8))
    ids_all = tokenizer.encode(text, add_special_tokens=False)
    if len(ids_all) < n_tokens:
        text = _compose_text(rng, n_tokens)  # pathological tokenizer; over-provision more
        ids_all = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids_all) >= n_tokens, "corpus composition too short"
    first: tuple[list[int], str, bool] | None = None
    for offset in range(0, min(64, len(ids_all) - n_tokens + 1)):
        ids = ids_all[offset : offset + n_tokens]
        txt = tokenizer.decode(ids)
        re = tokenizer.encode(txt, add_special_tokens=False)
        ok = re == ids
        if first is None:
            first = (ids, txt, ok)
        if ok:
            return ids, txt, True
    assert first is not None
    return first


def build_workload(tokenizer, model: str = "Qwen/Qwen2.5-1.5B", seed: int = 0,
                   tokenizer_revision: str | None = None) -> Workload:
    """The 16 base requests: PROMPT_LENGTHS x MAX_TOKENS x PROMPTS_PER_CELL."""
    rng = random.Random(seed)
    requests: list[Request] = []
    idx = 0
    for n in PROMPT_LENGTHS:
        for mt in MAX_TOKENS:
            for _ in range(PROMPTS_PER_CELL):
                corpus_seed = rng.randrange(2**31)
                ids, txt, ok = _prompt_for(tokenizer, corpus_seed, n)
                assert len(ids) == n
                requests.append(
                    Request(
                        request_id=base_request_id(model, ids, mt),
                        index=idx,
                        prompt_len=n,
                        max_tokens=mt,
                        prompt_token_ids=ids,
                        prompt_text=txt,
                        text_roundtrip=ok,
                        corpus_seed=corpus_seed,
                    )
                )
                idx += 1
    # distinctness: no two prompts share token ids
    seen = {tuple(r.prompt_token_ids) for r in requests}
    assert len(seen) == len(requests), "duplicate prompts"
    return Workload(schema=SCHEMA, model=model, tokenizer_revision=tokenizer_revision,
                    seed=seed, requests=requests)


def load_workload(path: str) -> Workload:
    with open(path) as f:
        return Workload.from_json(json.load(f))


# -- EOS / stopping probes (task 4) ---------------------------------------------------
# Few-shot completions the base model finishes with <|endoftext|> on its own.  Built with the
# tokenizer's EOS text so the probes are model-agnostic; they are NOT part of the 16-request
# workload and carry their own ids.
EOS_PROBE_TEMPLATES: tuple[str, ...] = (
    "The capital of France is Paris.{eos}The capital of Germany is Berlin.{eos}The capital of Italy is",
    "2 + 2 = 4{eos}3 + 5 = 8{eos}7 + 6 =",
    "cat -> cats{eos}dog -> dogs{eos}house ->",
    "Q: What color is the sky on a clear day?\nA: Blue.{eos}Q: What color is grass?\nA:",
)


def build_eos_probes(tokenizer, model: str = "Qwen/Qwen2.5-1.5B", max_tokens: int = 32) -> list[Request]:
    eos = tokenizer.eos_token or "<|endoftext|>"
    out = []
    for i, tmpl in enumerate(EOS_PROBE_TEMPLATES):
        txt = tmpl.format(eos=eos)
        ids = tokenizer.encode(txt, add_special_tokens=False)
        out.append(
            Request(
                request_id=base_request_id(model, ids, max_tokens),
                index=1000 + i,
                prompt_len=len(ids),
                max_tokens=max_tokens,
                prompt_token_ids=ids,
                prompt_text=txt,
                text_roundtrip=tokenizer.encode(tokenizer.decode(ids), add_special_tokens=False) == ids,
                corpus_seed=-1,
            )
        )
    return out


# -- CLI ------------------------------------------------------------------------------
def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--revision", default=None, help="tokenizer revision (recorded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="workload.json")
    ap.add_argument("--eos-probes", default=None, help="also write EOS probe requests to this path")
    args = ap.parse_args()
    from transformers import AutoTokenizer  # local import: CPU-only users may not have it

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    wl = build_workload(tok, model=args.model, seed=args.seed, tokenizer_revision=args.revision)
    wl.save(args.out)
    print(f"wrote {args.out}: {len(wl.requests)} requests, "
          f"lengths {sorted({r.prompt_len for r in wl.requests})}, "
          f"roundtrip ok for {sum(r.text_roundtrip for r in wl.requests)}/{len(wl.requests)}")
    if args.eos_probes:
        probes = build_eos_probes(tok, model=args.model)
        with open(args.eos_probes, "w") as f:
            json.dump({"schema": SCHEMA + "/eos-probes", "model": args.model,
                       "requests": [asdict(r) for r in probes]}, f, indent=1)
            f.write("\n")
        print(f"wrote {args.eos_probes}: {len(probes)} EOS probes")


if __name__ == "__main__":
    _main()
