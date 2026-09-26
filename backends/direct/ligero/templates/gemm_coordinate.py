"""B-Ligero lowering of ``gemm-coordinate``: the registered chain relations (``ligero.relations``), with the
``--auth included-hash`` leaf scheme named by the relation's ``+<leaf>`` suffix (``leaf.registry.split_relation_name``).

At K = 1536 ``bench-vu`` draws each relation's frozen instance set itself (``relchain``), so a cell there runs on the frozen set
only.  At vLLM's reduction lengths (``relations.REAL_KS``) the x4 folds are registered as ``<fold>-k<K>`` and read an input set
(``bench-vu --input-set``; :func:`stage` writes a range of one); keyed-BLAKE3 rows are proved with the xob circuit.  H100 BF16
``wgmma`` is the ``bf16-hopper`` relation: on the H100 it writes ``mma``'s words on every captured input (PR #49), and the
input set's recorded outputs are checked against the relation's model when the prover reads them.  NVFP4 runs through
``fp4/chain.py``, a separate runner."""
TEMPLATE = "gemm-coordinate"
K = 1536
REAL_KS = (2048, 8192)
RELATIONS = {"sm80.mma.m16n8k16.bf16": "bf16-ampere", "sm90.mma.m16n8k16.bf16": "bf16-hopper", "sm89.mma.m16n8k32.e4m3": "fp8-ada",
             "sm90.wgmma.m64n8k32.e4m3": "fp8-hopper"}
REAL_K_RELATIONS = {**RELATIONS, "sm90.wgmma.m64n8k16.bf16": "bf16-hopper"}
LEAVES = {"frame-v3/sha-256": "sha256", "frame-v3/blake3-keyed": "blake3", "vllm-v1/sha-256": "vllm-v1",
          "frame-v3/poseidon2-babybear": "hash"}
REAL_K_LEAVES = {"frame-v3/blake3-keyed": "blake3-xob", "frame-v3/sha-256": "sha256"}


def supports(subcircuit, scheme):
    sem, k = subcircuit.params.get("semantics"), int(subcircuit.params["K"])
    if k in REAL_KS:
        if sem not in REAL_K_RELATIONS:
            return f"B-Ligero's K = {k} folds cover {sorted(REAL_K_RELATIONS)}, not {sem}"
        if scheme not in REAL_K_LEAVES:
            return f"B-Ligero's K = {k} folds prove {sorted(REAL_K_LEAVES)} row leaves, not {scheme}"
        return None
    if sem not in RELATIONS:
        return f"B-Ligero's chain relations cover {sorted(RELATIONS)} (ligero.relations; NVFP4 is fp4/chain.py), not {sem}"
    if k != K:
        return f"B-Ligero's chain relations are built at K = {K} and {list(REAL_KS)} (relations); K = {k} is not registered"
    if scheme not in LEAVES:
        return f"B-Ligero has no in-circuit leaf for {scheme} (leaf schemes {sorted(LEAVES)})"
    return None


def relation(subcircuit, scheme):
    k = int(subcircuit.params["K"])
    if k in REAL_KS:
        return f"{REAL_K_RELATIONS[subcircuit.params['semantics']]}-x4-k{k}+{REAL_K_LEAVES[scheme]}"
    return f"{RELATIONS[subcircuit.params['semantics']]}+{LEAVES[scheme]}"


def stage(input_set, lo, hi, out_dir):
    """Ids ``[lo, hi)`` of an input set as a set of their own under ``out_dir`` (``input_sets.subset``): ``bench-vu --input-set``
    reads it as it is."""
    from pathlib import Path

    from verity_numerical.bench.input_sets import subset
    if lo == 0 and hi == input_set.n:
        return Path(input_set.path)
    return subset(input_set.path, out_dir, lo, hi)
