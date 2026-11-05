"""A served workload: verity_vllm's serving Program for a truncated SmolLM2 checkpoint, built and evaluated on CPU.

`verity_vllm.program.frontend.derive.derive_report` derives the Program from `derive_examples.GreedyServe`, the vLLM-style
module (merged QKV and gate_up linears, vLLM's kernel interface, a per-layer KV cache) that serves one request: a prefill
of `prompt` tokens, then `steps` greedy decode steps. In profile mode (`b1-eager`) every node is bound to the registered
Definition of the kernel vLLM runs, and the derivation claims the registered family `Serve_v1` (`registry.b1.Serve`)
only when the derived dataflow is node for node its body; `Served` refuses a derivation that doesn't. `Config` is B0's
architecture (HuggingFaceTB/SmolLM2-135M, tied embeddings) at a truncated size; `Config.b0(layers)` is B0's width.

The weights are a registered input (`Program(fn, registered=("weights",))`), the prompt a run input, and the values are
words (bf16, f32 and i32 encodings), not bits. The root Calls are `Serve_v1`'s body, round by round (the prefill, then
each decode step): an `Embedding_v1` per token, then per layer a `LayerPre_v1` per token and an `Attention_v1` and a
`LayerPost_v1` per token, then `Final_v1` (the final norm, the LM head and the greedy `TokenSelect`) on the round's last
token. `Served.calls` gives each Call its round (as `CallInfo.request`) and layer (the embedding is layer 0's, `Final_v1`
the last layer's), which is what a pipeline placement reads; round r's layer l runs at engine step 1 + r·layers + l,
since a round needs the token the round before it chose.

`Served` has `model.Model`'s interface, so placement, framing and the block processes run on it as they do on the MLP. There is no torch oracle on CPU (vLLM's kernels are stand-ins there), so `reference` is the Program evaluated in
one process, which a distributed run has to reproduce.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from functools import cached_property, lru_cache

from verity.ml.circuits.program import Program, is_input

from .model import CallInfo

PROFILE = "b1-eager"
FAMILY = "Serve_v1"


@dataclass(frozen=True)
class Config:
    layers: int = 3
    hidden: int = 64
    heads: int = 1
    kv_heads: int = 1
    head_dim: int = 64
    intermediate: int = 128
    vocab: int = 32
    prompt: int = 2
    steps: int = 1
    max_pos: int = 64

    @classmethod
    def b0(cls, layers: int = 30, **over) -> "Config":
        """B0 (SmolLM2-135M) at its own width, with `layers` of its 30 layers."""
        return cls(**{**dict(layers=layers, hidden=576, heads=9, kv_heads=3, head_dim=64, intermediate=1536, vocab=49152),
                      **over})

    def registry(self) -> dict:
        """The configuration `registry.b1` reads: B0's, with this size."""
        from verity.ml.definitions.registry import b1
        c = dict(b1.B0_CONFIG)
        c.update(NL=self.layers, H=self.hidden, NH=self.heads, KVH=self.kv_heads, D=self.head_dim, I=self.intermediate,
                 V=self.vocab, MAX_POS=self.max_pos, QKV=(self.heads + 2 * self.kv_heads) * self.head_dim,
                 GU=2 * self.intermediate)
        return c

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _module(cfg: Config, device: str | None = None):
    import torch
    from verity_vllm.program.frontend import derive_examples as ex
    if device is None:
        return ex.GreedyServe(cfg.registry(), cfg.steps)
    with torch.device(device):
        return ex.GreedyServe(cfg.registry(), cfg.steps)


def _derive(cfg: Config):
    import torch
    from verity_vllm.program.frontend.derive import derive_report
    prompt = torch.empty(cfg.prompt, dtype=torch.int32, device="meta")
    rep = derive_report(_module(cfg, "meta"), (prompt,), mode="profile", profile=PROFILE, config=cfg.registry(), name="Serve")
    if not rep.ok:
        raise ValueError(f"verity_vllm refused the serving module: {rep.unsupported.reason}: {rep.unsupported.hint}")
    if not (rep.family and rep.family.get("matched") and rep.family["family"].startswith(FAMILY + "{")):
        raise ValueError(f"the derived Program isn't {FAMILY}: {rep.provenance.get('families_not_matched')}")
    return rep


class Served:
    """The `Serve_v1` Program of a `Config`, its Calls with their rounds and layers, and its gate table."""

    def __init__(self, spec: Config) -> None:
        self.spec = spec
        self.report = _derive(spec)
        self.program = Program(self.report.program.fn, registered=("weights",))
        self.circuit = self.program.circuit
        self.params = ("prompt", "weights")
        self.binding: dict[str, str] = dict(self.report.family["weights_binding"])
        self.calls = self._calls()
        self.prims: list = []
        self.operands: list[tuple[int, ...]] = []
        for g in self.circuit.region(0, self.n):
            self.prims.append(g.prim)
            self.operands.append(tuple(g.operands))

    def _calls(self) -> list[CallInfo]:
        nodes = [n for n in self.program.root.body.nodes if not is_input(n.fn)]
        calls = list(self.circuit.root_calls())
        out, rnd, layer, prev = [], 0, -1, None
        for node, c in zip(nodes, calls, strict=True):
            base = node.fn.base_id
            if base == "Embedding_v1":
                at = 0
            elif base == "LayerPre_v1":
                layer += prev != base
                at = layer
            elif base in ("Attention_v1", "LayerPost_v1"):
                at = layer
            elif base == "Final_v1":
                at = self.spec.layers - 1
            else:
                raise ValueError(f"{base} is no Call of {FAMILY}'s body")
            if not 0 <= at < self.spec.layers:
                raise ValueError(f"Call {len(out)} ({base}) falls in layer {at} of {self.spec.layers}")
            out.append(CallInfo(rnd, at, 0, c.lo, c.hi))
            if base == "Final_v1":
                rnd, layer = rnd + 1, -1
            prev = base
        if rnd != self.spec.steps + 1:
            raise ValueError(f"{rnd} rounds, a prefill and {self.spec.steps} decode steps make {self.spec.steps + 1}")
        return out

    def step(self, c: CallInfo) -> int:
        return 1 + c.request * self.spec.layers + c.layer

    @property
    def n(self) -> int:
        return self.circuit.size

    @cached_property
    def ops(self) -> list[tuple[int, ...]]:
        return [tuple(sorted(set(o))) for o in self.operands]

    @cached_property
    def width(self) -> list[int]:
        return [p.ret.w for p in self.prims]

    @cached_property
    def value_bits(self) -> int:
        return max(self.width)

    @cached_property
    def kind(self) -> list[str]:
        out = []
        for p, o in zip(self.prims, self.operands):
            if is_input(p):
                out.append("registered" if p.id.startswith("Registered") else "run")
            else:
                out.append("computed" if o else "constant")
        return out

    @cached_property
    def outputs(self) -> tuple[int, ...]:
        return self.circuit.root_outputs()

    def input_gates(self, pi: int) -> list[int]:
        return [ref[2] for ref in self.program.input_coll(pi).refs]

    def bind_inputs(self, prompt: list[int], weights: list[int]) -> dict[int, int]:
        t: dict[int, int] = {}
        for pi, words in enumerate((prompt, weights)):
            gates = self.input_gates(pi)
            if len(gates) != len(words):
                raise ValueError(f"parameter {self.params[pi]} has {len(gates)} words, given {len(words)}")
            t.update(zip(gates, words))
        return t

    def inputs(self, seed: int) -> dict[int, int]:
        return self.bind_inputs(requests(self.spec, seed), weights(self, seed))

    def constants(self) -> dict[int, int]:
        """Every constant's value: an operation's own bits, which every block has."""
        return {g: self.prims[g].evaluate() for g, k in enumerate(self.kind) if k == "constant"}

    def evaluate(self, transcript: dict[int, int], gates=None) -> dict[int, int]:
        """Evaluate the computed and constant gates among `gates` (default: every gate) in order into `transcript`."""
        kind = self.kind
        for g in range(self.n) if gates is None else gates:
            if kind[g] in ("run", "registered"):
                if g not in transcript:
                    raise KeyError(f"input gate {g} has no value")
                continue
            p, o = self.prims[g], self.operands[g]
            lazy = getattr(p, "evaluate_lazy", None)
            transcript[g] = lazy(lambda j, _o=o: transcript[_o[j]]) if lazy is not None else p.evaluate(*[transcript[a] for a in o])
        return transcript

    def output_words(self, transcript: dict[int, int]) -> list[int]:
        """The tokens the request generated: one a round."""
        return [transcript[o] for o in self.outputs]

    def reference(self, seed: int) -> list[int]:
        return self.output_words(self.evaluate(self.inputs(seed)))

    def call_of(self, g: int) -> int | None:
        """The index of the root Call gate `g` is in, or None for an input gate."""
        lo, hi = 0, len(self.calls)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.calls[mid].hi <= g:
                lo = mid + 1
            else:
                hi = mid
        return lo if lo < len(self.calls) and self.calls[lo].lo <= g else None


@lru_cache(maxsize=4)
def load(spec: Config) -> Served:
    return Served(spec)


def requests(spec: Config, seed: int) -> list[int]:
    rng = random.Random(f"verity/netsim/prompt/{seed}")
    return [rng.randrange(spec.vocab) for _ in range(spec.prompt)]


def weights(served: Served, seed: int) -> list[int]:
    """The weights record's words in leaf order: the module's tensors as `weights_binding` names them, its norms ones, its
    embedding and linears sampled from a seeded normal (bf16), its rotary table as the module computes it."""
    import torch
    torch.manual_seed(seed)
    m = _module(served.spec)
    tensors = dict(m.named_parameters())
    tensors.update(m.named_buffers())
    gen = torch.Generator().manual_seed(seed)
    out: list[int] = []
    with torch.no_grad():
        for role, name in served.binding.items():
            t = tensors[name]
            if isinstance(t, torch.nn.Parameter):
                if "norm" in name:
                    t.fill_(1.0)
                else:
                    scale = 0.5 if role == "embed" else 0.08
                    t.copy_((torch.randn(t.shape, generator=gen) * scale).to(t.dtype))
            if t.dtype != torch.bfloat16:
                raise ValueError(f"{name} is {t.dtype}, the weights record holds bf16 words")
            out += t.contiguous().view(torch.int16).reshape(-1).tolist()
    return [w & 0xFFFF for w in out]
