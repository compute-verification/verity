"""The served-compiled Build ahead of time: `acquire.served`'s Program, step plans and manifest of a model at vLLM's serving defaults,
from its checkpoint's config.json and two pins, with no serving process and no trace.

The Build is the traced one (`acquire.served.build`) term for term, assembled by the same code (`served.assemble`, `served_moe.expand`,
`inductor_templates.instantiate`).  What the trace reads from the serving process's compile artifact comes from:

  * the config (`program.model.config_of` under its family's rules, `engine.profiles.family_facts`): the pieces (the first, then one
    per decoder layer), each Triton launch's template literals, the GEMM shapes, the attention heads and the fused-MoE block's facts;
  * the family's compiled layout (`LAYOUTS`): the kernels Inductor generates in each piece, their names and the buffers whose hooks
    name the values, under vLLM d9105ea80 and torch 2.13 as the traced Builds of Llama-3.2-1B (RoPE), Qwen3-8B (q/k-norm RoPE) and
    Qwen3-30B-A3B (q/k-norm RoPE, the fused-MoE op) read them;
  * the row's pin (`verity_vllm/pins/served/<ROLE>.<device>.json`, written from a traced Build by `compiled-pin`): what one machine
    measured -- each piece kind's kernel source digests, the compile artifact (their launched configs and the autotune winners, which
    the Commit seeds), the CUDA-graph dispatch (capture sizes, max_num_seqs, and under FA2 max_model_len) and the fused-MoE configs
    vLLM launched at each row count;
  * the device pin (`verity.ml.definitions.registry.bf16_gemm_check.device_pin`): the cuBLAS routes, the SM count the attention scheduler reads and
    the compute capability, whose bindings the Build takes (`served_device`).

A family without a layout, a config outside it (attention bias, shared experts or dense layers in an MoE, another activation or
dtype), a pin whose kernels are not the layout's, a GEMM without a route or a step whose fused-MoE row count has no recorded launch is
refused (`Unmodelled`).  The check is a traced Build of the pinned machine: `--compare`, and the Commit, whose treatment rebuilds the
Build from its own artifact and `compiled verify` holds the Build of record (this one) to it.

    verity-vllm predict compiled --role ROLE --device 'NVIDIA H200' --workload WL [--compare TRACED] --out build.json
    verity-vllm predict compiled-pin --build TRACED --inductor-dir DIR [--moe moe.json] --role ROLE --evidence RUN --out PIN
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

from verity_vllm.acquire import compiled as C
from verity_vllm.acquire import inductor_templates as T
from verity_vllm.acquire import served as SV
from verity_vllm.acquire import served_artifact as SA
from verity_vllm.acquire import served_device as SD
from verity_vllm.acquire import served_moe as SM
from verity_vllm.acquire.inductor_templates import Unmodelled
from verity_vllm.config import UsageError, option, workload_arg
from verity_vllm.engine.profiles import family_facts as FF
from verity_vllm.engine.profiles.dense_generic import find_hf_config
from verity_vllm.observe.fold.profiles.generic import router_vpt_bf16
from verity_vllm.program.model import config_of
from verity.ml.definitions.registry.bf16_gemm_check import device_pin
from verity_vllm.query.manifest.format import SERVED_IDS as IDS, SERVED_POSITIONS as POSITIONS

PINS = resources.files("verity_vllm") / "pins" / "served"
KINDS = ("first", "middle", "last")
EMBED = "triton_red_fused__to_copy_embedding_rms_norm_0"

#: each family's compiled pieces by (family arch, q/k norm): the RoPE template, and per piece kind each kernel's name with the buffers
#: its writes are hooked as -- the embedding (first), q and k after RoPE, the post-attention norm and MLP (a dense MLP's norm, gate_up,
#: SiLU and down, or the norm stored twice and the fused-MoE op's return), the add + norm that ends a layer (the residual, the next
#: layer's input) and the final norm (last)
LAYOUTS: dict[tuple[str, str | None], dict[str, Any]] = {
    ("dense", None): {
        "rope": "ROPE_QK", "first": {"qkv": "mm/buf3", "rope": ("triton_poi_fused_1", "cat/buf5", "cat/buf4")},
        "mlp": {"norm": ("triton_red_fused_fused_add_rms_norm_0", "mul/buf2"), "up": "mm/buf3",
                "act": ("triton_poi_fused_mul_silu_slice_1", "mul/buf4"), "down": "mm/buf5"},
        "add2": ("triton_red_fused_fused_add_rms_norm_2", "convert_element_type/buf6", "convert_element_type/buf8"),
        "middle": {"qkv": "mm/buf9", "rope": ("triton_poi_fused_3", "cat/buf11", "cat/buf10")},
        "final": ("triton_red_fused_fused_add_rms_norm_2", "mul/buf7")},
    ("dense", "head"): {
        "rope": "QK_NORM_ROPE", "first": {"qkv": "mm/buf3", "rope": ("triton_red_fused_1", "cat/buf7", "cat/buf6")},
        "mlp": {"norm": ("triton_red_fused_fused_add_rms_norm_0", "mul/buf2"), "up": "mm/buf3",
                "act": ("triton_poi_fused_mul_silu_slice_1", "mul/buf4"), "down": "mm/buf5"},
        "add2": ("triton_red_fused_fused_add_rms_norm_2", "convert_element_type/buf6", "convert_element_type/buf8"),
        "middle": {"qkv": "mm/buf9", "rope": ("triton_red_fused_3", "cat/buf13", "cat/buf12")},
        "final": ("triton_red_fused_fused_add_rms_norm_2", "mul/buf7")},
    ("moe", "head"): {
        "rope": "QK_NORM_ROPE", "first": {"qkv": "mm/buf3", "rope": ("triton_red_fused_1", "cat/buf7", "cat/buf6")},
        "moe": {"norm": ("triton_red_fused_fused_add_rms_norm_moe_forward_0", "fused/buf2", "fused/buf3"), "out": "extern/buf6"},
        "add2": ("triton_red_fused_fused_add_rms_norm_1", "convert_element_type/buf7", "convert_element_type/buf9"),
        "middle": {"qkv": "mm/buf10", "rope": ("triton_red_fused_2", "cat/buf14", "cat/buf13")},
        "final": ("triton_red_fused_fused_add_rms_norm_1", "mul/buf8")},
}


def model_of(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The config's static facts (`config_of` under its family's rules, `I` the experts' width in an MoE) and its family's layout."""
    try:
        fam = FF.facts_for(cfg)
    except ValueError as e:
        raise Unmodelled(str(e)) from None
    m, _ = config_of(cfg, family_rules={FF.family_of(cfg): fam["rules"]})
    hand, _ = FF.resolve_facts(cfg, m)
    key = (fam["arch"], m.get("QK_NORM"))
    moe = fam["arch"] == "moe"
    off = {"layout": None if key in LAYOUTS else key, "attention bias": m["BIAS"] or None, "patterns": fam.get("patterns"),
           "hidden_act": None if cfg.get("hidden_act") == "silu" else cfg.get("hidden_act"),
           "dtype": None if (cfg.get("torch_dtype") or cfg.get("dtype")) == "bfloat16" else cfg.get("torch_dtype") or cfg.get("dtype"),
           "facts": sorted(set(hand) - {"RENORMALIZE", "SHARED_EXPERTS"}) or None, "shared experts": hand.get("SHARED_EXPERTS") or None,
           "dense layers": (cfg.get("mlp_only_layers") or None) if moe else None,
           "sparse step": cfg.get("decoder_sparse_step") if moe and cfg.get("decoder_sparse_step", 1) != 1 else None}
    off = {k: v for k, v in off.items() if v}
    if off:
        raise Unmodelled(f"model_type {FF.family_of(cfg)!r}: {off}; the compiled layouts are {sorted(LAYOUTS, key=str)}")
    if moe:
        m["I"], m["RENORMALIZE"] = int(cfg[fam["moe_width_key"]]), bool(hand["RENORMALIZE"])
        m["MOE_PARAMS"] = {f: name for name, f in fam["params"]["layer"].items()}
    return m, LAYOUTS[key]


def _literals(t: T.Template, m: dict[str, Any]) -> dict[str, Any]:
    """The template's literals at model m: a norm's N over the hidden width, the q/k norm's over a head."""
    H, NH, KVH, D = m["H"], m["NH"], m["KVH"], m["D"]
    lit = {"N": H, "NF": float(H if "N" in T.holes(t) else D), "EPS": float(m["EPS"]), "V": m["V"], "I": m["I"], "I2": 2 * m["I"],
           "D": D, "H": NH, "KVH": KVH, "HALF": D // 2, "QLEN": NH * D, "KLEN": KVH * D, "QKV": m["QKV"], "KHI": NH * D + D // 2,
           "P": m["MAX_POS"]}
    return {k: lit[k] for k in T.holes(t)}


class _Pieces:
    """The wired Program `served.wire` reads from the pieces' wrappers, written piece by piece from a layout: values `c<piece>.<op>/<buf>`
    in the order the pieces write them, each attention op's after its piece's."""

    def __init__(self, m: dict[str, Any], layout: dict[str, Any], pin: dict[str, Any], backbone: str) -> None:
        self.m, self.layout, self.pin, self.backbone = m, layout, pin, backbone
        self.values: dict[str, dict[str, Any]] = {IDS: {"width": 1, "piece": 0}, POSITIONS: {"width": 1, "piece": 0}}
        self.calls: list[dict[str, Any]] = []
        self.attention: list[dict[str, Any]] = []
        self.kernels: dict[str, dict[str, Any]] = {}
        self.layers: dict[str, dict[str, Any]] = {}
        self.p = 0

    def w(self, name: str, *shape: int, layer: int | None = None) -> dict[str, Any]:
        path = f"{self.backbone}." + (f"layers.{layer}." if layer is not None else "") + name
        return {"weight": path, "shape": list(shape)}

    def value(self, name: str, width: int) -> dict[str, Any]:
        full = f"c{self.p:02d}.{name}"
        if full in self.values:
            raise Unmodelled(f"{full}: two values of one name")
        self.values[full] = {"width": width, "piece": self.p}
        return {"value": full, "width": width}

    def mm(self, x: dict[str, Any], w: dict[str, Any], out: str) -> dict[str, Any]:
        (N, K) = w["shape"]
        y = self.value(out, N)
        self.calls.append({"op": "mm", "gemm": [K, N], "reads": [x, {**w, "t": True}], "writes": [y["value"]], "piece": self.p})
        return y

    def launch(self, kind: str, name: str, template: str, binds: dict[str, dict[str, Any]], outs: dict[str, str]) -> list[dict[str, Any]]:
        """Kernel `name` of the pin's `kind` pieces as template `template` at the model's literals and its pinned launch config, its
        parameters bound to `binds` and each write hooked as `outs`; the written values in `outs`' order."""
        sha = (self.pin["kernels"].get(kind) or {}).get(name)
        if sha is None:
            raise Unmodelled(f"the pin has no kernel {name} in its {kind} pieces: {self.pin['kernels'].get(kind)}")
        t = T.TEMPLATE[template]
        rec = T.instantiate(name, t, _literals(t, self.m), self.pin["artifact"]["kernels"].get(sha))
        self.kernels[sha] = {"source_sha256": sha, "template": t.name, "literals": rec["literals"], "config": rec["config"]}
        local = dict(binds)
        for c in rec["calls"]:
            reads = []
            for r in c["reads"]:
                ref = local[r["param"]]
                if "cols" in r:
                    ref = {**ref, "width": r["cols"][1] - r["cols"][0], "cols": r["cols"]}
                if "row_of" in r:
                    ref = {**ref, "row_of": local[r["row_of"]]}
                got = ref["shape"][1] if "row_of" in ref else ref.get("width", math.prod(ref.get("shape", [])))
                if got != r["numel"]:
                    raise Unmodelled(f"{name}: {c['definition']} reads {r['numel']} elements per row from {r['param']}, bound to {ref}")
                reads.append(ref)
            writes = []
            for wr in c["writes"]:
                local[wr["param"]] = self.value(outs[wr["param"]], wr["numel"])
                writes.append(local[wr["param"]]["value"])
            self.calls.append({"op": name, "definition": c["definition"], "base": c["base"], "statics": c["statics"], "reads": reads,
                               "writes": writes, "piece": self.p})
        return [local[p] for p in outs]

    def attend(self, kind: str, layer: int, x: dict[str, Any], qkv_buf: str) -> None:
        """The layer's qkv projection of x, RoPE on its q and k, and the attention op on q, k and v (its columns past q and k)."""
        m, (kernel, qb, kb) = self.m, self.layout[kind]["rope"]
        QL, KL = m["NH"] * m["D"], m["KVH"] * m["D"]
        qkv = self.mm(x, self.w("self_attn.qkv_proj.weight", m["QKV"], m["H"], layer=layer), qkv_buf)
        table, pos = self.w("self_attn.rotary_emb.cos_sin_cache", m["MAX_POS"], m["D"], layer=0), {"value": POSITIONS, "width": 1}
        if self.layout["rope"] == "ROPE_QK":
            k, q = self.launch(kind, kernel, "ROPE_QK", {"in_ptr0": qkv, "in_ptr1": pos, "in_ptr2": table},
                               {"out_ptr0": kb, "out_ptr1": qb})
        else:
            q, k = self.launch(kind, kernel, "QK_NORM_ROPE",
                               {"in_ptr0": qkv, "in_ptr1": self.w("self_attn.q_norm.weight", m["D"], layer=layer), "in_ptr2": pos,
                                "in_ptr3": table, "in_ptr4": self.w("self_attn.k_norm.weight", m["D"], layer=layer)},
                               {"out_ptr1": qb, "out_ptr3": kb})
        v = {**qkv, "width": KL, "cols": [QL + KL, QL + 2 * KL]}
        name = f"c{self.p:02d}.attention/{self.backbone}.layers.{layer}.self_attn.attn"
        self.attention.append({"piece": self.p, "layer": layer, "name": name, "q": q, "k": k, "v": v, "NH": m["NH"], "KVH": m["KVH"],
                               "D": m["D"]})
        self.values[name] = {"width": QL, "piece": self.p}

    def first(self) -> dict[str, Any]:
        """Piece 0: the embedding and layer 0's input norm, its qkv and attention; the residual."""
        m = self.m
        emb, x = self.launch("first", EMBED, "EMBED_NORM", {"in_ptr0": {"value": IDS, "width": 1}, "in_ptr1": self.w("embed_tokens.weight", m["V"], m["H"]),
                                                            "in_ptr2": self.w("input_layernorm.weight", m["H"], layer=0)},
                             {"out_ptr0": "embedding/buf0", "out_ptr2": "embedding/buf2"})
        self.attend("first", 0, x, self.layout["first"]["qkv"])
        return emb

    def layer(self, p: int, residual: dict[str, Any], last: bool) -> dict[str, Any]:
        """Piece p: layer p-1 from its attention's output on (its o_proj, MLP or fused-MoE block, the add + norm), then layer p's qkv and
        attention, or the final norm in the last piece; the residual, or the backbone's output."""
        m, L, l, kind = self.m, self.layout, p - 1, "last" if last else "middle"
        self.p = p
        prev = self.attention[-1]
        h = self.mm({"value": prev["name"], "width": m["NH"] * m["D"]}, self.w("self_attn.o_proj.weight", m["H"], m["NH"] * m["D"], layer=l),
                    "mm/buf0")
        post = self.w("post_attention_layernorm.weight", m["H"], layer=l)
        if "moe" in L:
            kernel, b1, b2 = L["moe"]["norm"]
            x, _copy = self.launch(kind, kernel, "ADD_NORM_COPY", {"in_ptr0": h, "in_ptr1": residual, "in_ptr2": post},
                                   {"out_ptr1": b1, "out_ptr2": b2})
            out = self.value(L["moe"]["out"], m["H"])
            self.calls.append({"op": "moe_forward", "layer": {"layer_name": True}, "reads": [x], "writes": [out["value"]], "piece": p})
            self._moe_layer(l)
        else:
            (nk, nb), (ak, ab) = L["mlp"]["norm"], L["mlp"]["act"]
            x, = self.launch(kind, nk, "ADD_NORM", {"in_ptr0": h, "in_ptr1": residual, "in_ptr2": post}, {"out_ptr1": nb})
            up = self.mm(x, self.w("mlp.gate_up_proj.weight", 2 * m["I"], m["H"], layer=l), L["mlp"]["up"])
            act, = self.launch(kind, ak, "SILU_MUL", {"in_ptr0": up}, {"out_ptr0": ab})
            out = self.mm(act, self.w("mlp.down_proj.weight", m["H"], m["I"], layer=l), L["mlp"]["down"])
        if last:
            kernel, fb = L["final"]
            final, = self.launch(kind, kernel, "ADD2_NORM_FINAL", {"in_out_ptr0": out, "in_ptr0": h, "in_ptr1": residual,
                                                                   "in_ptr2": self.w("norm.weight", m["H"])}, {"in_out_ptr0": fb})
            return final
        kernel, rb, xb = L["add2"]
        res, x = self.launch(kind, kernel, "ADD2_NORM", {"in_ptr0": out, "in_ptr1": h, "in_ptr2": residual,
                                                         "in_ptr3": self.w("input_layernorm.weight", m["H"], layer=p)},
                             {"out_ptr1": rb, "out_ptr2": xb})
        self.attend("middle", p, x, L["middle"]["qkv"])
        return res

    def _moe_layer(self, l: int) -> None:
        """Layer l's fused-MoE facts as `served_match.moe_facts` records them, the block `served_moe.REQUIRED` describes."""
        m, names = self.m, self.m["MOE_PARAMS"]
        E = m["NUM_EXPERTS"]
        shapes = {"gate_w": [E, m["H"]], "w13": [E, 2 * m["I"], m["H"]], "w2": [E, m["H"], m["I"]]}
        name = f"{self.backbone}.layers.{l}.mlp.experts"
        self.layers[name] = {**{k: self.w(names[f], *shapes[f], layer=l) for k, f in (("gate", "gate_w"), ("w13", "w13"), ("w2", "w2"))},
                             "E": E, "top_k": int(m["TOPK"]), "renormalize": m["RENORMALIZE"], "VPT": router_vpt_bf16(E), **SM.REQUIRED}

    def moe(self, cc: list[int]) -> dict[str, Any] | None:
        """The fused-MoE record `served_moe.expand` reads (the serving process's, `served_match`), the configs the pin's."""
        if not self.layers:
            return None
        pieces = {str(c["piece"]): f"{self.backbone}.layers.{c['piece'] - 1}.mlp.experts" for c in self.calls if c["op"] == "moe_forward"}
        return {"layers": self.layers, "pieces": pieces, "configs": (self.pin.get("moe") or {}).get("configs") or {}, "cc": list(cc)}


def pin_file(role: str, device: str) -> str:
    return f"{role}.{device.replace(' ', '_')}.json"


def load_pin(role: str, device: str) -> dict[str, Any]:
    path = PINS / pin_file(role, device)
    if not path.is_file():
        raise Unmodelled(f"no served pin {path.name} in verity_vllm/pins/served: `predict compiled-pin` writes one from a traced Build")
    return json.loads(path.read_text())


def predict(role: str, device: str, schedule: list[dict[str, Any]], steps: int, *, backbone: str = "model",
            lm_head: str = "lm_head.weight", rank: int = 0) -> dict[str, Any]:
    """The served Build of `role` on `device` for the schedule, from the role's config.json, its served pin and the device's pin."""
    found = find_hf_config(role)
    if found is None:
        raise Unmodelled(f"no config.json for {role}")
    m, layout = model_of(json.loads(found[0].read_text()))
    dev, pin = device_pin(device), load_pin(role, device)
    b = _Pieces(m, layout, pin, backbone)
    residual = b.first()
    for p in range(1, m["NL"] + 1):
        residual = b.layer(p, residual, last=p == m["NL"])
    wired, plans = SM.expand({"values": b.values, "calls": b.calls, "attention": b.attention}, b.moe(dev["compute_capability"]))
    d = pin["dispatch"]
    return SV.assemble(wired, plans, residual, b.kernels, pin["artifact"]["winners"],
                       source={"predicted": {"role": role, "device": device, "config": found[1], "pin": f"pins/served/{pin_file(role, device)}"}},
                       schedule=schedule, steps=steps, routes=dev["cublas"], capture_sizes=d["capture_sizes"], max_num_seqs=d["max_num_seqs"],
                       device=SD.device_of(dev["compute_capability"], dev["sms"], card=SD.card_of(device),
                                           max_model_len=d.get("max_model_len")),
                       split_cap=SV.SPLIT_CAP, lm_head=lm_head, rank=rank)


def differences(predicted: dict[str, Any], traced: dict[str, Any]) -> list[str]:
    """Where a predicted Build differs from a traced one: its digests, then the first differing call, attention op or step plan."""
    out = [f"{k}: predicted {predicted[k][:16]}, traced {str(traced.get(k))[:16]}" for k in ("program_digest", "manifest_digest")
           if predicted[k] != traced.get(k)]
    if predicted["artifact"]["digest"] != (traced.get("artifact") or {}).get("digest"):
        out += [f"artifact: {d}" for d in SA.differences(traced.get("artifact") or {}, predicted["artifact"])] or ["artifact digest"]
    for key, a, b in (("call", predicted["program"]["calls"], traced["program"]["calls"]),
                      ("attention", predicted["program"]["attention"], traced["program"]["attention"]),
                      ("step plan", predicted["steps_plan"], traced["steps_plan"])):
        i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))
        if i is not None:
            out.append(f"{key} {i}: predicted {json.dumps(a[i] if i < len(a) else None)[:400]}, traced {json.dumps(b[i] if i < len(b) else None)[:400]}")
    for k in ("values", "kernels", "final", "logits", "sampled", "dispatch"):
        if predicted["program"][k] != traced["program"].get(k):
            out.append(f"program {k} differs")
    return out


def pin_of(build: dict[str, Any], inductor_dir: str, moe: dict[str, Any] | None, *, role: str, device: str, evidence: str) -> dict[str, Any]:
    """A row's served pin from its traced Build and the Inductor dir that Build read (its piece wrappers): each piece kind's kernels by
    name and source digest (every middle piece's the same), the Build's artifact, its dispatch, and the fused-MoE configs of `moe`."""
    prog = build["program"]
    wrappers, kernels = prog["piece_wrappers"], {k: {} for k in KINDS}
    for p, rel in enumerate(wrappers):
        kind = KINDS[0] if p == 0 else KINDS[2] if p == len(wrappers) - 1 else KINDS[1]
        src = T.kernel_sources(open(os.path.join(inductor_dir, rel)).read())
        got = {c["op"]: T.source_sha256(src[c["op"]]) for c in prog["calls"] if c["piece"] == p and c["op"] in src}
        if kernels[kind] and kernels[kind] != got:
            raise Unmodelled(f"piece {p} ({rel}) launches kernels {got}, another {kind} piece {kernels[kind]}")
        kernels[kind] = dict(sorted(got.items()))
    dev, d = device_pin(device), prog["dispatch"]
    if d["sms"] != dev["sms"] or d.get("cc", [9, 0]) != dev["compute_capability"]:
        raise Unmodelled(f"the Build's SM count {d['sms']} and cc {d.get('cc', [9, 0])} are not {device}'s pin's")
    return {"role": role, "device": device, "evidence": evidence,
            "dispatch": {k: d[k] for k in ("capture_sizes", "max_num_seqs", "max_model_len") if k in d}, "kernels": kernels,
            "artifact": {k: build["artifact"][k] for k in ("kernels", "winners")},
            **({"moe": {"configs": moe["configs"]}} if moe and moe.get("layers") else {})}


@dataclass(frozen=True, kw_only=True)
class Options:
    """The served-compiled Build of a role on a device, predicted (never traced)."""
    role: str = option("--role", help="the checkpoint's role (its config.json: engine/profiles/hf_configs/<ROLE>.config.json)")
    device: str = option("--device", help="the serving GPU as torch names it (its device pin: bf16_gemm_check.device_pin; "
                                          "its served pin: verity_vllm/pins/served/)")
    workload: str = option("--workload", type=workload_arg, help="sweep workload json: the request schedule")
    steps: int | None = option("--steps", type=int, default=None, help="engine steps (default: the workload's expected_engine_steps)")
    compare: str | None = option("--compare", default=None, help="a traced served Build to hold the prediction to (exit 5 when they differ)")
    out: str = option("--out")
    cmd: str = field(default="compiled", init=False)


@dataclass(frozen=True, kw_only=True)
class PinOptions:
    """A role's served pin, from a traced served Build."""
    build: str = option("--build", help="the traced served Build (served-match's build.json)")
    inductor_dir: str = option("--inductor-dir", help="the Inductor dir that Build read (its piece_wrappers are relative to it)")
    moe: str | None = option("--moe", default=None, help="the serving process's fused-MoE record (served-match's moe.json)")
    role: str = option("--role")
    evidence: str = option("--evidence", help="the run that traced the Build")
    out: str = option("--out")
    cmd: str = field(default="compiled-pin", init=False)


COMMANDS = {"compiled": Options, "compiled-pin": PinOptions}


def main(a: Options | PinOptions) -> int:
    if a.cmd == "compiled-pin":
        build = json.load(open(a.build))
        device = json.load(open(os.path.join(os.path.dirname(a.build), "routes.json")))["device"]
        pin = pin_of(build, a.inductor_dir, json.load(open(a.moe)) if a.moe else None, role=a.role, device=device, evidence=a.evidence)
        with open(a.out, "w") as f:
            json.dump(pin, f, indent=1)
            f.write("\n")
        print(f"[predict compiled-pin] {a.role} on {device}: {sum(map(len, pin['kernels'].values()))} kernel launches, "
              f"{len(pin['artifact']['winners'])} winners -> {a.out}")
        return 0
    wl = json.load(open(a.workload))
    steps = a.steps or int(wl.get("expected_engine_steps") or 0)
    if not steps:
        raise UsageError("--steps (or the workload's expected_engine_steps) is required")
    try:
        m = predict(a.role, a.device, C.schedule_of_workload(wl), steps)
    except Unmodelled as e:
        print(f"[predict compiled] refused: {e}")
        return 3
    with open(a.out, "w") as f:
        json.dump(m, f, indent=1)
    print(f"[predict compiled] program {m['program_digest'][:16]} manifest {m['manifest_digest'][:16]} artifact {m['artifact']['digest'][:16]} "
          f"identities {len(m['identities'])} complete {m['complete']} -> {a.out}")
    if a.compare:
        diff = differences(m, json.load(open(a.compare)))
        print(f"[predict compiled] against {a.compare}: " + ("equal" if not diff else "\n  ".join(["differs"] + diff)))
        return 5 if diff else 0
    return 0 if m["complete"] else 4
