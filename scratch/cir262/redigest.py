"""Recompute a sealed capture record's digest from the file alone: sha256 of the canonical JSON of the record without its
`digest` (verity_vllm.properties.record.digest_of at 70a27d97ec), with no import of the repository."""
import hashlib, json, sys

def canonical_json(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()

for p in sys.argv[1:]:
    d = json.load(open(p))
    got = hashlib.sha256(canonical_json({k: v for k, v in d.items() if k != "digest"})).hexdigest()
    print(p.split("/runs/")[-1], "| recorded", d.get("digest"), "| recomputed", got, "| equal", got == d.get("digest"), "| ok", d.get("ok"))
