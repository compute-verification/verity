"""A minimal PyTorch serving loop, an integration of Verity's (`verity.ml.windows.interface`) with no vLLM.

* **The model** (:class:`Model`): a two-layer W8A8 MLP of width ``H``, ``model.layers.{0,1}.mlp.{up_proj,down_proj}``. Each
  linear's weights are s8 codes with one f32 scale per output channel; each token row is quantized per row to s8 codes and
  an f32 scale (its absmax / 127); and the linear is ``bf16((f32(x_q . w_q) * sx) * sw)`` in torch on CPU: an exact int32
  sum, two f32 multiplies (the token's scale first), then the bf16 cast (round to nearest even). That is
  `ScaledMmS8Coordinate_v1{K=H, DOT=BlackwellS8ImmaDot32_v1}`'s function (`verity.ml.circuits.ops.gemm`, the s8 binding of
  sm_120), the circuit's one template: each output coordinate of each linear at each token row is one unit. The ReLU
  between the linears and each row's quantization are not units: their outputs are committed as the next linear's inputs.
* **The wired circuit** (``wired``): each linear at each token row is one `ScaledMmS8_v1{K=H, N=H}` unit (`linear_template`),
  its H outputs one row, and each ReLU and quantize one `QuantRowS8_v1{N=H, RELU}` unit (`quant_template`: RELU on the
  up_proj -> down_proj edge, not on down_proj -> the next step's up_proj), which reads that row whole and writes the codes
  and the scale the next linear reads. Each tie is a per-window value (`TorchLoop.ties_of`) that its writer's output port
  and its reader's input port both read, row for row (registered reads of output ports), committed after the run, before
  the window closes, under a domain bound to the run root (`commit_ties`), so the Lean verifier opens the reader's row
  against the root the writer's row opens. Only the request's own row, the step-0 up_proj's ``x`` and ``sx`` (the
  embedding quantized), is a free input, so a changed activation is a drawn unit's refusal, not a free choice. Linear unit
  (request r, step s, layer l) is index ``l·R·S + s·R + r`` (R requests, S steps), and quantize unit (r, s) of either edge
  ``s·R + r``, so each read is an offset; a member must not straddle the step-0 up_proj units and the rest (``cap``
  divides R). The run is computed first and recorded in that order.
  ``tamper_activations``: bit 0 of every quantize unit's first code flipped, the next linear computed on the flipped
  codes, so every linear's Definition holds and every quantize unit's doesn't.
* **Serving** (:class:`TorchLoop`, an `interface.Integration`): each request is one token, embedded and run through the MLP
  for ``steps`` decode steps, step s + 1 reading step s's output. Every linear's units are recorded into the window as they
  are computed (`Window.record`), labelled with the request, step, layer and op path they serve, which Verity never reads.
  The circuit is fixed by the loop's shape (no compiler, no advice), or registered as its compiler's output on its schedule
  (`compile`), the schedule committed as advice. ``tamper``: a loop whose recorded outputs are not its
  model's (bit 0 of every output word flipped before it is recorded), which the proof of any drawn unit must refuse.
* **Registered weights** (``registered``): the model's weights are two registered values, ``w`` (every linear's s8 code
  rows, up_proj's then down_proj's, one row of ``H / 2`` u16 words an output channel) and ``sw`` (their f32 scales, one row
  of 2 words a channel), committed once before the loop serves (`commit_weights`: core's ``one_stage.registered`` rows in
  ``hm96-sha512/row/v2``, the rows the template's ``wrow`` and ``sw`` ports commit), the verifier pinning their entries
  (`pin`, the checkpoint role `ROLE` as provenance). The circuit then declares each unit's ``wrow`` and ``sw`` read at
  public positions (`interface.Read`: unit (request, step, layer, n) reads row ``H·layer + n``), so they are no free
  input, and the window commits them under the registered salts (`interface.Window`'s ``registered``).
  ``tamper_weights``: a loop that serves another model's weights (bit 0 of every weight code row's first word flipped)
  under the registered value's salts, its outputs its own weights' (so its Definition holds), which Lean's registered
  reads must refuse: the rows open no pinned root. In the wired circuit each value is one row a linear (``w`` the s8
  [H][H] codes, ``H² / 2`` words, ``sw`` the H scales, ``2H`` words: `commit_weights`' ``wired``), each linear unit reading
  its layer's row.
* **Committed activations** (`TorchLoop.commit_activations`, CMP-91): each window's activation rows, ``x`` (every linear's
  s8 input row, ``H / 2`` words) and ``sx`` (its f32 scale, 2 words), one row a (request, step, layer) in unit order (unit
  u's rows are row ``u // H``), committed as core's registered values in ``hm96-sha512/row/v2`` under fresh salts, each
  domain binding the window's run root, so a window's rows open under no other window's domain. Their entries go in the
  registration's window (`activations`), which the draw comes after, so they are fixed before the window's draw coin;
  their salts are the prover's (`activations_opening`). The circuit declares no read of them: they are per-window values
  the prover chose, so they take no input bit out of the free count (only a pinned value does). In the wired circuit the
  committed rows are the free ones alone, one a request, and the tie values (`TorchLoop.ties_of`) are committed beside them.
* **The output tie** (``tie``): the output the network certifier relayed is a registered value `OUT` (``out``: one row of
  one u16 word a unit of the last linear, request-major, then step, then output coordinate, in ``hm96-sha512/row/v2``),
  committed by the certifier under its salts and pinned by the verifier from the certifier's signed record alone. The
  circuit declares each output unit's ``o0`` a registered read of it, so its output row is committed under the
  certifier's salt (`out_opening`, `opened_out`) and the Lean verifier's registered reads refuse a unit whose output
  isn't the row the certifier relayed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch

from verity.core.protocols.sampled_proofs.one_stage import registered as REG
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import gemm as G, quant as Q
from verity.ml.windows import interface as IF

LINEARS = ("up_proj", "down_proj")
#: The MLP's width. C-Flock's unit statement needs a committed row longer than one SHA-512 block (over 111 bytes): at 128
#: the s8 rows ``x`` and ``wrow`` are 128 bytes, at 64 no row is.
H = 128
#: the embedding's rows: a request is one of these token ids
VOCAB = 16
#: the checkpoint role the default model is (`Model()`: H, VOCAB, seed 0), which the weights' pin names as provenance
ROLE = "TORCH_LOOP_MLP_H128_W8A8"
PIN = "verity-torch-loop/weights-pin/v1"
OPENING = "verity-torch-loop/weights-opening/v1"
#: the registered values, each by the template port that reads it
WEIGHTS = (("w", "wrow"), ("sw", "sw"))
#: the same in the wired circuit, by `linear_template`'s port
WIRED_WEIGHTS = (("w", "w"), ("sw", "sw"))
#: the committed activations, each the template port whose rows it holds
ACTIVATIONS = ("x", "sx")
ACTIVATIONS_OPENING = "verity-torch-loop/activations-opening/v1"
#: the output value the network certifier relayed and committed, and the output port that reads it
OUT = ("out", "o0")
OUT_OPENING = "verity-torch-loop/out-opening/v1"


def template(k: int):
    """The circuit's template: one output coordinate of a W8A8 linear over ``k`` inputs, sm_120's s8 binding."""
    return bind(G.ScaledMmS8Coordinate, K=k, **G.GEMM_SM120_S8)


def linear_template(h: int):
    """The wired circuit's linear: a W8A8 linear of width ``h`` on one token row, its ``h`` outputs one row."""
    return bind(G.ScaledMmS8, K=h, N=h, **G.GEMM_SM120_S8)


def quant_template(h: int, relu: bool):
    """The wired circuit's ReLU (under ``relu``) and per-row quantize of a linear's ``h``-wide bf16 row."""
    return bind(Q.QuantRowS8, N=h, RELU=relu)


def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row s8 codes and f32 scales of ``x`` (rows, K): scale absmax / 127 (1 for a zero row), codes round(x / scale)."""
    x = x.to(torch.float32)
    amax = x.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / 127, torch.ones_like(amax))
    return torch.round(x / scale[:, None]).clamp(-128, 127).to(torch.int8), scale


def linear(xq: torch.Tensor, sx: torch.Tensor, wq: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    """The W8A8 linear: ``bf16((f32(xq . wq^T) * sx) * sw)``, (rows, N) bf16."""
    acc = xq.to(torch.int32) @ wq.to(torch.int32).T
    return ((acc.to(torch.float32) * sx[:, None]) * sw[None, :]).to(torch.bfloat16)


@dataclass(frozen=True)
class Layer:
    wq: torch.Tensor     # (H, H) s8
    sw: torch.Tensor     # (H,) f32


class Model:
    """The two-layer W8A8 MLP of width ``h``, its weights and a ``vocab``-row embedding drawn from ``seed``."""

    def __init__(self, h: int = H, vocab: int = VOCAB, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.h, self.seed = h, seed
        self.embed = torch.randn(vocab, h, generator=g)
        self.layers: list[Layer] = []
        for _ in range(2):
            w = torch.randn(h, h, generator=g) * 0.05
            wq, sw = quantize(w)
            self.layers.append(Layer(wq, sw))

    def op_path(self, layer: int) -> str:
        return f"model.layers.{layer}.mlp.{LINEARS[layer]}"

    @property
    def name(self) -> str:
        """What the weights' domain binds as their program: the model, by its width and seed."""
        return f"torch-loop/mlp-w8a8-h{self.h}/seed{self.seed}"

    def weights(self, tamper: bool = False, wired: bool = False) -> dict[str, np.ndarray]:
        """The registered values' rows, u16 words (`interface.words`): ``w`` every linear's code rows, ``sw`` their scales,
        one row an output channel, or with ``wired`` one row a linear. ``tamper``: bit 0 of every code row's first word
        flipped."""
        n = 1 if wired else self.h
        w = np.concatenate([IF.words(bits(lin.wq).reshape(n, -1), 8, n, self.h * self.h // (2 * n)) for lin in self.layers])
        if tamper:
            w = w.copy()
            w[:, 0] ^= 1
        return {"w": w, "sw": np.concatenate([IF.words(bits(lin.sw).reshape(n, -1), 32, n, 2 * self.h // n)
                                              for lin in self.layers])}


def commit_weights(model: Model, salts: dict[str, list[bytes]] | None = None, wired: bool = False) -> dict[str, REG.Registered]:
    """The model's weights committed as registered values (module docstring), fresh salts unless ``salts`` (by value);
    ``wired``: one row a linear, as the wired circuit reads them."""
    return {v: REG.Registered.commit(model.name, v, rows.tolist(), (salts or {}).get(v), schema=REG.SCHEMA_V2)
            for v, rows in model.weights(wired=wired).items()}


def pin(model: Model, values: dict[str, REG.Registered]) -> dict:
    """The verifier's pin of the weights, before the loop serves: each value's entry (root, rows, words, domain) under the
    model's name, the checkpoint role the default model is as provenance."""
    return {"format": PIN, "program": model.name, **({"checkpoint_role": ROLE} if (model.h, model.seed) == (H, 0) else {}),
            "values": {v: r.entry() for v, r in values.items()}}


def opening(model: Model, values: dict[str, REG.Registered]) -> dict:
    """The prover's private file of the weights' commitment: each row's salt (the rows are the model's), and ``rows``
    ``linear`` when they are one row a linear (`commit_weights`' ``wired``)."""
    return {"format": OPENING, "program": model.name, **({"rows": "linear"} if values["w"].leaves == len(model.layers) else {}),
            "salts": {v: [s.hex() for s in r.salts] for v, r in values.items()}}


def opened(model: Model, doc: dict) -> dict[str, REG.Registered]:
    """The weights committed again under the salts of their `opening`."""
    if doc.get("format") != OPENING or doc.get("program") != model.name or doc.get("rows", "linear") != "linear":
        raise ValueError(f"not an opening of {model.name}'s weights ({OPENING})")
    return commit_weights(model, {v: [bytes.fromhex(x) for x in ss] for v, ss in doc["salts"].items()}, "rows" in doc)


def held_pin(doc: dict) -> dict[str, dict]:
    """The pinned entries by value, each refused (ValueError) unless its domain is the one its program, value and rows make
    (core's ``registered.domain``): what the verifier hands `interface.registered_reads`."""
    if doc.get("format") != PIN:
        raise ValueError(f"a weights pin is {PIN}")
    for v, e in doc["values"].items():
        want = REG.domain(doc["program"], v, e["words"], e["leaves"], e.get(REG.WIDTH_KEY, REG.WORD_BITS)).domain_id.hex()
        if e.get("value") != v or e.get("domain") != want:
            raise ValueError(f"the pin of {v!r} names another value or domain than its program's ({want[:16]})")
    return dict(doc["values"])


def activations(model: Model, values: dict[str, REG.Registered]) -> dict:
    """What the registration's window names of a run's committed activations: their program and each value's entry."""
    return {"program": model.name, "values": {v: r.entry() for v, r in values.items()}}


def activations_opening(model: Model, values: dict[str, REG.Registered]) -> dict:
    """The prover's private file of a run's committed activations: their window and each row's salt."""
    window, = {r.window for r in values.values()}
    return {"format": ACTIVATIONS_OPENING, "program": model.name, "window": window,
            "salts": {v: [s.hex() for s in r.salts] for v, r in values.items()}}


def out_opening(program: str, rows: list[list[int]], salts: list[bytes]) -> dict:
    """The prover's file of the output value the network certifier committed: the reading program's digest, the rows
    the server sent and the certifier's salts for them (its witness)."""
    return {"format": OUT_OPENING, "program": program, "value": OUT[0], "rows": [list(map(int, r)) for r in rows],
            "salts": [s.hex() for s in salts]}


def opened_out(doc: dict) -> REG.Registered:
    """The output value committed again from its `out_opening`, as the certifier committed it."""
    if doc.get("format") != OUT_OPENING or doc.get("value") != OUT[0]:
        raise ValueError(f"not an opening of the output value ({OUT_OPENING})")
    return REG.Registered.commit(doc["program"], OUT[0], doc["rows"], [bytes.fromhex(s) for s in doc["salts"]],
                                 schema=REG.SCHEMA_V2)


def bits(t: torch.Tensor) -> np.ndarray:
    """A tensor's elements as unsigned integers of their width (`evaluate_batch`'s convention)."""
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    a = t.contiguous().numpy()
    return a.view(f"u{a.dtype.itemsize}")


class TorchLoop:
    """The loop over ``requests`` (token ids) for ``steps`` decode steps of ``model`` (module docstring); ``run_id`` names
    the run, so two runs of one shape have two roots (infra's gate opens a root's coins once, ever)."""

    def __init__(self, model: Model, requests: list[int], steps: int = 2, tamper: bool = False, run_id: str = "",
                 registered: bool = False, tamper_weights: bool = False, tie: bool = False, wired: bool = False,
                 tamper_activations: bool = False) -> None:
        if tamper_activations and not wired:
            raise ValueError("tamper_activations tampers the wired circuit's quantize units: give wired")
        if tie and wired:
            raise ValueError("the output tie reads the per-coordinate circuit's output units: give tie or wired")
        self.model, self.requests, self.steps, self.tamper, self.run_id = model, list(requests), steps, tamper, run_id
        self.registered, self.tamper_weights, self.tie = registered, tamper_weights, tie
        self.wired, self.tamper_activations = wired, tamper_activations
        self.ties: dict[str, REG.Registered] = {}
        self.outputs: dict[tuple[int, int], torch.Tensor] = {}
        self.activations: dict[str, list[np.ndarray]] = {v: [] for v in ACTIVATIONS}

    @property
    def units(self) -> int:
        r, s = len(self.requests), self.steps
        return r * s * 2 + r * s + r * (s - 1) if self.wired else r * s * 2 * self.model.h

    @property
    def run_root(self) -> str:
        """The run's root: SHA-512 of what the loop served (its model's seed and width, its requests and steps) and its id."""
        return hashlib.sha512(json.dumps({"loop": "torch-loop/mlp-w8a8", "h": self.model.h, "seed": self.model.seed,
                                          "requests": self.requests, "steps": self.steps,
                                          "id": self.run_id, **({"wired": True} if self.wired else {})}).encode()).hexdigest()

    def circuit(self) -> IF.Circuit:
        """The loop's circuit: its units, and with ``registered`` each unit's ``wrow`` and ``sw`` read from the registered
        weights, unit (request, step, layer, n) reading row ``h·layer + n`` (a fan-out over the requests and steps). With
        ``tie``, each output unit's ``o0`` (the last linear's, `output_units`) is a registered read of `OUT`: unit
        (request r, step s, last layer, n) reads row ``(r·steps + s)·h + n``, one ``Read`` a request and step. With
        ``wired``, the wired circuit (`wired_circuit`)."""
        if self.wired:
            return self.wired_circuit()
        h, fn = self.model.h, template(self.model.h)
        tid, rs = descriptor_id(fn), len(self.requests) * self.steps
        params, reads = {}, []
        if self.registered:
            words = {p: w for p, w in IF.layout(fn).rows}
            params |= {v: 2 * h * words[p] for v, p in WEIGHTS}
            reads += [IF.Read(tid, p, v, 0, (rs, 2, h), 0, (0, h * words[p], words[p])) for v, p in WEIGHTS]
        if self.tie:
            value, port = OUT
            params[value] = rs * h
            reads += [IF.Read(tid, port, value, lo, (h,), k * h, (1,)) for k, (lo, _hi) in enumerate(self.output_units())]
        reg = IF.Registered(params, reads=tuple(reads)) if params else None
        return IF.Circuit(f"torch-loop/mlp-w8a8-h{h}", ((fn, self.units),), registered=reg)

    def wired_circuit(self) -> IF.Circuit:
        """The wired circuit (module docstring): the linears, then the quantize units of the two edges (the second's only past
        one step), every tie a registered read (`ties_of`); with ``registered``, linear unit (l, s, r) also reads row ``l`` of
        ``w`` and ``sw``."""
        h, r, s = self.model.h, len(self.requests), self.steps
        mm = linear_template(h)
        templates = [(mm, 2 * r * s), (quant_template(h, True), r * s)] + ([(quant_template(h, False), r * (s - 1))] if s > 1 else [])
        params, reads = {}, []
        for value, n, words, ports in self.ties_of():
            params[value] = n * words
            reads += [IF.Read(t, p, value, at, (n,), 0, (words,)) for t, p, at in ports]
        if self.registered:
            lw = dict(IF.layout(mm).rows)
            params |= {v: 2 * lw[p] for v, p in WIRED_WEIGHTS}
            reads += [IF.Read(descriptor_id(mm), p, v, 0, (2, r * s), 0, (lw[p], 0)) for v, p in WIRED_WEIGHTS]
        return IF.Circuit(f"torch-loop/mlp-w8a8-h{h}/wired", tuple(templates), registered=IF.Registered(params, reads=tuple(reads)))

    def ties_of(self) -> list[tuple[str, int, int, tuple[tuple[str, str, int], ...]]]:
        """The wired circuit's ties, each ``(value, rows, words a row, ports)``: every ``(template, port, unit)`` of
        ``ports`` reads the value's row i at that template's unit ``unit + i``, its writer's output port and its reader's
        input port alike. Linear unit (request r, step s, layer l) is ``l·R·S + s·R + r``, quantize unit (r, s) of either
        edge ``s·R + r``."""
        h, r, s = self.model.h, len(self.requests), self.steps
        tm, tq, tp = (descriptor_id(t) for t in (linear_template(h), quant_template(h, True), quant_template(h, False)))
        rs, n = r * s, r * (s - 1)
        out = [("up", rs, h, ((tm, "o0", 0), (tq, "x", 0))), ("up_codes", rs, h // 2, ((tq, "o0", 0), (tm, "x", rs))),
               ("up_scale", rs, 2, ((tq, "o1", 0), (tm, "sx", rs)))]
        if s > 1:
            out += [("down", n, h, ((tm, "o0", rs), (tp, "x", 0))), ("down_codes", n, h // 2, ((tp, "o0", 0), (tm, "x", r))),
                    ("down_scale", n, 2, ((tp, "o1", 0), (tm, "sx", r)))]
        return out

    def output_units(self) -> list[list[int]]:
        """The units whose outputs the loop serves out, as ascending ``[lo, hi)`` ranges: the last linear's at every request
        and step (`run`'s order: request, step, layer, then output coordinate; in the wired circuit down_proj's units),
        from the loop's shape alone."""
        h, last = self.model.h, len(self.model.layers) - 1
        if self.wired:
            rs = len(self.requests) * self.steps
            return [[rs, 2 * rs]]
        per = len(self.model.layers) * h
        return [[(r * self.steps + s) * per + last * h, (r * self.steps + s) * per + per]
                for r in range(len(self.requests)) for s in range(self.steps)]

    def work_table(self, law: dict) -> dict[str, int] | None:
        """Under a ``work`` law, every unit of the circuit's one template credited alike (work 1, floor 1), so the law draws
        its ``k`` as a uniform ``k``-subset; None under any other law."""
        if law.get("law") != "work":
            return None
        return {descriptor_id(fn): 1 for fn, _n in self.circuit().templates} if self.wired else {IF.descriptor_id(template(self.model.h)): 1}

    def _layers(self) -> list[Layer]:
        """The linears served: the model's, or under ``tamper_weights`` the tampered weights' codes."""
        h, layers = self.model.h, self.model.layers
        if self.tamper_weights:
            w = torch.from_numpy(self.model.weights(tamper=True, wired=self.wired)["w"].view(np.int8).reshape(2, h, h).copy())
            layers = [Layer(w[k], lin.sw) for k, lin in enumerate(layers)]
        return layers

    def run(self, window: IF.Window) -> None:
        """Serve every request, recording each linear's units into ``window`` as they are computed (with ``wired``,
        `run_wired`)."""
        if self.wired:
            return self.run_wired(window)
        coord, h = template(self.model.h), self.model.h
        layers = self._layers()
        self.activations = {v: [] for v in ACTIVATIONS}
        for r, tok in enumerate(self.requests):
            x = self.model.embed[tok][None, :]
            for step in range(self.steps):
                for layer, lin in enumerate(layers):
                    xq, sx = quantize(x)
                    y = linear(xq, sx, lin.wq, lin.sw)
                    out = bits(y).reshape(h)
                    if self.tamper:
                        out = out ^ np.uint16(1)
                    window.record(coord, [np.repeat(bits(xq), h, 0), np.repeat(bits(sx), h), bits(lin.wq), bits(lin.sw)], [out],
                                  {"request": f"r{r}", "step": step, "layer": layer, "op": self.model.op_path(layer)})
                    self.activations["x"].append(IF.words(bits(xq), 8, 1, h // 2)[0])
                    self.activations["sx"].append(IF.words(bits(sx), 32, 1, 2)[0])
                    x = torch.relu(y.to(torch.float32)) if layer == 0 else y.to(torch.float32)
                self.outputs[r, step] = x

    def run_wired(self, window: IF.Window) -> None:
        """Serve the requests together, step by step, then record each template's units in the wired circuit's order: the
        linears by (layer, step, request), the quantize units of each edge by (step, request). Each quantize unit reads its
        linear's recorded row, and the next linear its recorded codes and scale."""
        h, R, S = self.model.h, len(self.requests), self.steps
        mm, layers = linear_template(h), self._layers()
        linears: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [[], []]
        quants: dict[bool, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {True: [], False: []}
        xq, sx = quantize(self.model.embed[self.requests])
        self.activations = {"x": list(IF.words(bits(xq), 8, R, h // 2)), "sx": list(IF.words(bits(sx), 32, R, 2))}
        for step in range(S):
            for layer, lin in enumerate(layers):
                out = bits(linear(xq, sx, lin.wq, lin.sw))
                if self.tamper:
                    out = out ^ np.uint16(1)
                linears[layer].append((bits(xq), bits(sx), out))
                y = torch.from_numpy(out.view(np.int16).copy()).view(torch.bfloat16).to(torch.float32)
                x = torch.relu(y) if layer == 0 else y
                if layer == 1:
                    for r in range(R):
                        self.outputs[r, step] = x[r][None, :]
                if (step, layer) == (S - 1, 1):
                    break
                xq, sx = quantize(x)
                if self.tamper_activations:
                    xq = xq.clone()
                    xq[:, 0] ^= 1
                quants[layer == 0].append((out, bits(xq), bits(sx)))
        self.tie_rows = {"up": np.concatenate([o for _x, _s, o in linears[0]]),
                         "up_codes": np.concatenate([IF.words(c, 8, R, h // 2) for _y, c, _s in quants[True]]),
                         "up_scale": np.concatenate([IF.words(c, 32, R, 2) for _y, _c, c in quants[True]])}
        if S > 1:
            self.tie_rows |= {"down": np.concatenate([o for _x, _s, o in linears[1][:-1]]),
                              "down_codes": np.concatenate([IF.words(c, 8, R, h // 2) for _y, c, _s in quants[False]]),
                              "down_scale": np.concatenate([IF.words(c, 32, R, 2) for _y, _c, c in quants[False]])}
        reqs = [f"r{r}" for r in range(R)]
        for layer, lin in enumerate(layers):
            w, sw = np.repeat(bits(lin.wq)[None], R, 0), np.repeat(bits(lin.sw)[None], R, 0)
            for step, (xb, sxb, out) in enumerate(linears[layer]):
                window.record(mm, [xb, sxb, w, sw], [out],
                              {"requests": reqs, "step": step, "layer": layer, "op": self.model.op_path(layer)})
        for relu in (True, False):
            for step, (y, codes, scale) in enumerate(quants[relu]):
                window.record(quant_template(h, relu), [y], [codes, scale],
                              {"requests": reqs, "step": step, "after": self.model.op_path(0 if relu else 1)})

    def commit_activations(self, salts: dict[str, list[bytes]] | None = None) -> dict[str, REG.Registered]:
        """The run's activation rows committed as per-window values (module docstring), fresh salts unless ``salts`` (by
        value); refused before the loop has served."""
        if not self.activations["x"]:
            raise ValueError("the loop has not served: its activations are committed after its run, before its registration")
        return {**{v: REG.Registered.commit(self.model.name, v, np.stack(rows).tolist(), (salts or {}).get(v),
                                            schema=REG.SCHEMA_V2, window=self.run_root)
                   for v, rows in self.activations.items()}, **self.ties}

    def commit_ties(self, salts: dict[str, list[bytes]] | None = None) -> dict[str, REG.Registered]:
        """The wired run's tie values (`ties_of`) committed as per-window values, as the activations are, fresh salts unless
        ``salts`` (by value): the rows its writers' output ports and its readers' input ports commit under these salts."""
        return {v: REG.Registered.commit(self.model.name, v, rows.tolist(), (salts or {}).get(v), schema=REG.SCHEMA_V2,
                                         window=self.run_root) for v, rows in self.tie_rows.items()}

    def window(self, law: dict, cap: int = 16, advice: dict | None = None,
               weights: dict[str, REG.Registered] | None = None, out: REG.Registered | None = None) -> IF.Window:
        """The loop's window under ``law`` in members of at most ``cap`` units, every unit recorded; ``advice``: the binding
        of its compiler and its schedule's commitment (`interface.advice_binding`), when its circuit is the compiler's;
        ``weights``: with ``registered``, the weights as committed before the run (`commit_weights`); ``out``: with ``tie``,
        the output value as the network certifier committed it (`opened_out`). Under a ``work`` law it carries the loop's
        work table (`work_table`)."""
        values = {**(weights or {}), **({OUT[0]: out} if out is not None else {})}
        win = IF.Window(self.circuit(), self.run_root, law, cap, advice=advice, registered=values or None,
                        work_table=self.work_table(law))
        self.run(win)
        if self.wired:
            self.ties = self.commit_ties()
            win.registered = {**(win.registered or {}), **self.ties}
        return win
