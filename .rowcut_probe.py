import hashlib, json, resource, sys, traceback
from collections import Counter
from pathlib import Path

from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import Registered
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict.window_class import Request
from verity_vllm.program.predict.build import Unexpressible
from verity.commitments.periods import interface as IF

ROOT = Path.cwd()
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True, "max_position_embeddings": 64, "hidden_act": "silu"}
TINY256 = {**TINY, "hidden_size": 256, "intermediate_size": 512}
LLAMA = json.loads((ROOT / "tests/adapters/vllm/tests/program/predict/fixtures/llama32-1b.config.json").read_text())
GRAINS = {"default": WC.Grain(), "test": WC.Grain(gemm_words=1024, rows=256, select=64)}
CLASSES = [(1, 1, 1), (1, 2, 1), (2, 4, 2)]


def reg_of(cfg):
    phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg}, target={"compute_capability": [12, 0], "num_sms": 188})
    return Registered.of(phi, default_seed=7)


def peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


sys.path.insert(0, str(ROOT / "benchmarks" / "prove_path"))
import class_size as CZ  # noqa: E402


def census(c):
    return Counter(CZ.census(c))


def _label(fn):
    return getattr(fn, "label", None) or str(getattr(fn, "id", fn))


def digest(c):
    try:
        return IF.descriptor_id(c.description())[:16]
    except Exception:  # noqa: BLE001
        pass
    try:
        return hashlib.sha256(json.dumps(c.description(), sort_keys=True, default=str).encode()).hexdigest()[:16]
    except Exception as e:  # noqa: BLE001
        return f"?{type(e).__name__}"


def where(e):
    tb = e.__traceback__
    out = []
    while tb:
        f = tb.tb_frame
        if f.f_code.co_name in ("rows", "lower", "_split", "weight") and "window_lowering" in f.f_code.co_filename:
            loc = {k: v for k, v in f.f_locals.items() if k in ("j", "lo", "hi", "a", "b", "w", "refs", "node")}
            if "node" in loc:
                n = loc["node"]
                loc["node"] = f"{_label(n.fn)} args={[(a.refs) for a in n.args][:3]}"
            if f.f_code.co_name == "rows" and "self" in f.f_locals:
                s = f.f_locals["self"]
                jj = f.f_locals.get("j")
                if jj is not None:
                    loc["writer"] = _label(s.fn.body.nodes[jj].fn)
                    loc["writer_outs"] = [(a, b, str(r)) for a, b, r in s.outs.get(jj, [])][:6]
            out.append((f.f_code.co_name, f.f_lineno, {k: str(v)[:400] for k, v in loc.items()}))
        tb = tb.tb_next
    return out


def run(tag, cfg, gname, cls, served=False):
    reg = reg_of(cfg)
    wc = WC.WindowClass(*cls)
    try:
        if served:
            low = WC.lower(reg, wc, None, GRAINS[gname], requests=[Request(0, 0, 1, 1)])
        else:
            low = WC.lower(reg, wc, None, GRAINS[gname])
        c = low.circuit
        cen = census(c)
        print("OK", json.dumps({"tag": tag, "grain": gname, "class": cls, "served": served, "units": sum(cen.values()),
                                "members": len(cen), "digest": digest(c), "peak_mb": peak()}), flush=True)
        return c, cen
    except Unexpressible as e:
        print("REFUSED", json.dumps({"tag": tag, "grain": gname, "class": cls, "served": served, "msg": str(e),
                                     "where": where(e), "peak_mb": peak()}), flush=True)
        return None, None


for tag, cfg in (("tiny64", TINY), ("tiny256", TINY256)):
    for g in GRAINS:
        for cls in CLASSES:
            run(tag, cfg, g, cls)
        run(tag, cfg, g, (1, 1, 1), served=True)
        run(tag, cfg, g, (2, 4, 2), served=True)

cc, ccen = run("llama", LLAMA, "default", (1, 1, 1))
sc, scen = run("llama", LLAMA, "default", (1, 1, 1), served=True)
if cc is not None and sc is not None:
    print("SAME_DIGEST", digest(cc) == digest(sc))
    print("CLASS_MINUS_SERVED", json.dumps(dict(ccen - scen)))
    print("SERVED_MINUS_CLASS", json.dumps(dict(scen - ccen)))
    print("REG_CLASS", json.dumps({"values": list(cc.registered.values), "run": list(cc.registered.run)})[:1500])
    print("REG_SERVED", json.dumps({"values": list(sc.registered.values), "run": list(sc.registered.run)})[:1500])
run("llama", LLAMA, "default", (1, 2, 1))
run("llama", LLAMA, "default", (2, 4, 2))
run("llama", LLAMA, "test", (1, 1, 1))
