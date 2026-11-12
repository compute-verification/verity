import json
from collections import Counter

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.commitments.gates.embedding import read
from verity_vllm.pipeline import long_row as LR
from verity_vllm.query.program_view import spec_base


def templates(t):
    return {descriptor_id(d): d for _k, d in LR.units(t) if spec_base(descriptor_id(d)) != "EmbeddingRead_v1"}


small = bind(LR.DsaTopKReadRowsCut, N=64, TOPK=32, BLOCK=32, READ=read(64, 1))
print(json.dumps({"served_by_base": dict(Counter(d.base_id for _, d in LR.units(LR.served()))),
                  "served_families": sorted({spec_base(x) for x in templates(LR.served())}),
                  "small_families": sorted({spec_base(x) for x in templates(small)})}, indent=1))
