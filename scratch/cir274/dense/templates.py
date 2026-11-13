"""CIR-274 DENSE: for every node of the dense half's families in rank 0's request Program (the test's `glm_request`), its
spec (`descriptor_id`), the templates `serving_templates.unit_templates` lists for it with their instances, each one's row
layout, and whether `interface.check_proof_block` takes it uncut. Run from adapters/vllm:

    uv run python ../../../verity-dense-scratch/scratch/cir274/dense/templates.py [LP] [LAYERS] [moe]
"""
import sys

sys.path.insert(0, "tests/program/predict")
from test_node_window import glm_request  # noqa: E402
from verity.circuits.codec import descriptor_id  # noqa: E402
from verity.commitments.windows import interface as IF  # noqa: E402

from verity_vllm.pipeline import serving_templates as ST  # noqa: E402

DENSE = ("SiluMul", "ResidualAdd", "AllReduceRmsNormMnnvl", "AllReduceMnnvl", "Gemm", "AllGather", "EmbeddingShard",
         "RoPEHeadGptJ", "MoeSum", "TokenSelect", "StopCheck")
LP = int(sys.argv[1]) if len(sys.argv) > 1 else 32
L = int(sys.argv[2]) if len(sys.argv) > 2 else 1
moe = len(sys.argv) > 3 and sys.argv[3] == "moe"
fn = glm_request(LP, L, moe=moe).program.fn
seen = set()
for j, n in enumerate(fn.body.nodes):
    fam = getattr(n.fn, "name", None) or n.fn.id.split("{")[0]
    if fam not in DENSE:
        continue
    spec = descriptor_id(n.fn)
    if spec in seen:
        print(f"{j:4d} {fam} (as above) n={n.n}")
        continue
    seen.add(spec)
    print(f"{j:4d} {fam} n={n.n} {spec}")
    for t, k in ST.unit_templates(spec):
        try:
            tf = ST.template(t).fn
            lay = IF.layout(tf)
            try:
                kl = IF.check_proof_block(tf)
                pb = f"k_log {kl}"
            except ValueError as e:
                pb = f"REFUSED {str(e)[:160]}"
            print(f"      {k:6d} x {t[:120]} rows={lay.rows} outer={lay.outer} outs={lay.outs} {pb}")
        except Exception as e:  # noqa: BLE001
            print(f"      {k:6d} x {t[:120]} ERR {type(e).__name__}: {str(e)[:200]}")
