"""B-Ligero lowering of ``gemm-coordinate``: the registered chain relations (``ligero.relations``) at K = 1536, with the
``--auth included-hash`` leaf scheme named by the relation's ``+<leaf>`` suffix (``leaf.registry.split_relation_name``).

``bench-vu`` draws each relation's frozen instance set itself (``relchain``), so there is no ``stage``: a cell runs on the
frozen set only.  NVFP4 runs through ``fp4/chain.py``, a separate runner."""
TEMPLATE = "gemm-coordinate"
K = 1536
RELATIONS = {"sm80.mma.m16n8k16.bf16": "bf16-ampere", "sm90.mma.m16n8k16.bf16": "bf16-hopper", "sm89.mma.m16n8k32.e4m3": "fp8-ada",
             "sm90.wgmma.m64n8k32.e4m3": "fp8-hopper"}
LEAVES = {"frame-v3/sha-256": "sha256", "frame-v3/blake3-keyed": "blake3", "vllm-v1/sha-256": "vllm-v1",
          "frame-v3/poseidon2-babybear": "hash"}


def supports(subcircuit, scheme):
    sem = subcircuit.params.get("semantics")
    if sem not in RELATIONS:
        return f"B-Ligero's chain relations cover {sorted(RELATIONS)} (ligero.relations; NVFP4 is fp4/chain.py), not {sem}"
    if int(subcircuit.params["K"]) != K:
        return f"B-Ligero's chain relations are built at K = {K} (relchain); K = {subcircuit.params['K']} is not registered"
    if scheme not in LEAVES:
        return f"B-Ligero has no in-circuit leaf for {scheme} (leaf schemes {sorted(LEAVES)})"
    return None


def relation(subcircuit, scheme):
    return f"{RELATIONS[subcircuit.params['semantics']]}+{LEAVES[scheme]}"
