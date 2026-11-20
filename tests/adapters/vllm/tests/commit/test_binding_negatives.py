"""The six mandatory binding negatives against the binding map + coverage checker (`verity_vllm/commit/binding.py`),
through the committer's offline core (no engine, no CUDA): the verifier holds the ORIGINAL RunCommitment and the binding map fixed at
finalize; the expected population is independent of the collector (structure x workload).

| mutation                                  | expected                                             | test |
| drop output member `/1`, bind `/0` twice  | coverage FAIL (identity_check alone PASSES)          | test_drop_member_and_duplicate_other_is_rejected |
| omit a whole producer family              | coverage FAIL                                        | test_omitted_family_is_rejected |
| omit one request / step / TP rank         | coverage FAIL                                        | test_omitted_{request,step,rank}_is_rejected |
| read a buffer after overwrite             | openings verify, value-correspondence FAIL           | test_late_read_verifies_but_fails_correspondence |
| relabel data from a different occurrence  | coverage ok, openings verify, correspondence FAIL    | test_relabelled_occurrence_is_rejected |
| wrong root or layout                      | opening / context FAIL                               | test_wrong_root_or_layout_is_rejected |
Openings are sampled from the expected population (incl. the last step and every family), not from the leaves that exist.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any

import pytest

from verity_vllm.commit import binding as B
from verity_vllm.commit import step_rows
from verity_vllm.commit.committer.native_host import NativeHostCommitter

CHUNK = step_rows.ROW_BYTES
PD = hashlib.sha512(b"program").hexdigest()
LAYERS = 2
STEPS = 3
#: the run's identities coin, as the auditor device's challenger would open it after the run's root
COIN = b"identities coin"


def structure() -> dict:
    mods = [{"op_path": "model.embed_tokens", "class": "VocabParallelEmbedding", "invocations": 1, "members": ["in", "out"]},
            {"op_path": "model.layers.0.input_layernorm", "class": "RMSNorm", "invocations": 1, "members": ["out"]}]
    for L in range(LAYERS):
        if L > 0:
            mods.append({"op_path": f"model.layers.{L}.input_layernorm", "class": "RMSNorm", "invocations": 1, "members": ["0", "1"]})
        mods.append({"op_path": f"model.layers.{L}.self_attn.qkv_proj", "class": "Linear", "invocations": 1, "members": ["0"]})
    mods.append({"op_path": "model.rotary_emb", "class": "RotaryEmbedding", "invocations": LAYERS, "members": ["0", "1"]})
    mods.append({"op_path": "runner.sampler", "class": "Sampler", "invocations": 1, "members": ["sampled_token_ids"]})
    return {"modules": mods, "attention_layers": list(range(LAYERS)), "hidden": None, "moe_layers": [], "moe_planes": []}


def workload(steps: int = STEPS, requests=("r0",), ranks=(0,)) -> dict:
    return {"program_digest": PD, "requests": list(requests), "steps": steps, "ranks": list(ranks)}


def step_names() -> list[str]:
    """Acquisition order of one honest step (the shared rotary module is invoked once per layer)."""
    names = ["model.embed_tokens/in", "model.embed_tokens", "model.layers.0.input_layernorm"]
    for L in range(LAYERS):
        if L > 0:
            names += [f"model.layers.{L}.input_layernorm/0", f"model.layers.{L}.input_layernorm/1"]
        names += [f"model.layers.{L}.self_attn.qkv_proj/0", "model.rotary_emb/0", "model.rotary_emb/1"]
    names.append("runner.sampler/sampled_token_ids")
    return names


def produce(seed: int = 7, steps: int = STEPS) -> list[list[tuple[str, bytes]]]:
    rng = random.Random(seed)
    return [[(n, rng.randbytes(6000 + 17 * i)) for i, n in enumerate(step_names())] for _ in range(steps)]


def committer(program: bytes = b"program") -> NativeHostCommitter:
    return NativeHostCommitter(None, run_id="run", hash_threads=1, program_digest=hashlib.sha512(program).digest())


def commit(stream, program: bytes = b"program", **map_kw):
    c = committer(program)
    run = c.commit_offline(stream)
    bmap = B.build_binding_map(c._layouts, run.steps, run.run_root, hashlib.sha512(program).hexdigest(), chunk=CHUNK, **map_kw)
    return c, run, bmap


def reference(stream) -> dict[str, bytes]:
    """The producer's bytes per (step, name, invocation) -- what a value-correspondence check compares against."""
    ref: dict[str, bytes] = {}
    for step, tensors in enumerate(stream):
        seen: dict[str, int] = {}
        for n, raw in tensors:
            k = seen.get(n, 0)
            seen[n] = k + 1
            ref[f"{step}|{n}|{k}"] = raw
    return ref




def lookup(ref):
    """Producer bytes for a map entry, keyed the way `reference()` keys them (step|name|invocation)."""
    return lambda e: ref.get(f"{e['identity']['step']}|{e['name']}|{e['identity']['invocation']}")


def value_correspondence(openings: list, entries: list[dict[str, Any] | None], reference, *, chunk: int) -> dict[str, Any]:
    """For each (opening, map entry): the opened chunk bytes must equal the PRODUCER's bytes for that identity.  `reference(entry)` returns
    the bytes the producer wrote for that (step, name, invocation), or None (not checkable).
    Openings can verify against the root and still fail here (late read after overwrite; relabelled occurrence)."""
    bad = []
    checked = 0
    for o, e in zip(openings, entries, strict=True):
        if e is None:
            bad.append({"step": o.step, "index": o.index, "why": "no binding"})
            continue
        ref = reference(e)
        if ref is None:
            continue
        checked += 1
        start = o.index * chunk - e["stream_offset"]
        want = ref[max(start, 0):max(start, 0) + len(o.value)] if start >= 0 else ref[:len(o.value) + start]
        got = o.value[:len(want)] if start >= 0 else o.value[-start:]
        if got[:len(want)] != want:
            bad.append({"step": o.step, "index": o.index, "name": e["name"], "why": "opened bytes != producer bytes"})
    return {"ok": not bad, "checked": checked, "bad_n": len(bad), "bad": bad[:8]}


def correspondence(ops, bound, ref) -> dict:
    return value_correspondence(ops, [e for _, e, _ in bound], lookup(ref), chunk=CHUNK)


# ---------------------------------------------------------------------- positive





# ---------------------------------------------------------------------- 1. drop /1, bind /0 twice



# ---------------------------------------------------------------------- 2. omit a producer family



# ---------------------------------------------------------------------- 3. omit one request / step / rank









# ---------------------------------------------------------------------- 4. late read: openings verify, value wrong



# ---------------------------------------------------------------------- 5. relabel data from a different occurrence



# ---------------------------------------------------------------------- 6. wrong root or layout



# ---------------------------------------------------------------------- B8 global binding map (request-local addressing + composition)

B8_LP = [5, 9, 7, 6, 2, 1, 3, 8]            # prompt lengths (the real row: [389, 1007, 757, 515, 13, 8, 195, 1024])
B8_CAP = [3, 4, 1, 5, 2, 2, 1, 5]           # max_tokens per request (the real row: [47, 77, 5, 128, 15, 21, 2, 123]); all arrive at step 0
B8_MEMBERS = ["model.embed_tokens/in", "model.embed_tokens", "model.layers.0.input_layernorm", "runner.sampler/sampled_token_ids"]


def b8_declared() -> dict[int, list[tuple[str, int]]]:
    """Declared composition per engine step, Program x workload: request k is active at steps [0, cap_k); rows = LP at its prefill, 1 after."""
    out: dict[int, list[tuple[str, int]]] = {}
    for e in range(max(B8_CAP)):
        out[e] = [(f"r{k}", B8_LP[k] if e == 0 else 1) for k in range(8) if e < B8_CAP[k]]
    return out


def b8_engine_rows(declared: dict[int, list[tuple[str, int]]]) -> dict[int, list[tuple[str, int, int]]]:
    """The engine's row order per step, vLLM v1 InputBatch style: slots filled in arrival order; a finished request's slot is refilled by
    the LAST slot's request (condense) -- so from step 2 on the row order is NOT the declaration order."""
    slots = [f"r{k}" for k in range(8)]
    rows: dict[int, list[tuple[str, int, int]]] = {}
    for e in sorted(declared):
        active = dict(declared[e])
        # condense: remove finished requests, moving the last slot into each freed one
        for i in range(len(slots)):
            while i < len(slots) and slots[i] not in active:
                last = slots.pop()
                if i < len(slots):
                    slots[i] = last
        lo, order = 0, []
        for rid in slots:
            n = active[rid]
            order.append((rid, lo, lo + n))
            lo += n
        rows[e] = order
    return rows


def b8_stream(rows: dict[int, list[tuple[str, int, int]]], seed: int = 3) -> list[list[tuple[str, bytes]]]:
    """One honest step per engine step: per-token members are `tokens` bytes long (rows == tokens -> split per request), the sampled-ids
    member `n_seq` bytes (rows == len(requests) -> one slot per request)."""
    rng = random.Random(seed)
    out = []
    for e in sorted(rows):
        tokens, n_seq = max(hi for _r, _lo, hi in rows[e]), len(rows[e])
        out.append([(n, rng.randbytes(n_seq if n.endswith("sampled_token_ids") else tokens)) for n in B8_MEMBERS])
    return out


def b8_manifest(tmp_path, declared: dict[int, list[tuple[str, int]]]) -> str:
    """The global manifest the way `required_manifest.build_global_manifest` writes it (request-LOCAL ranges, `engine_steps`
    composition, `composition_reference`, request ids r<k> with the workload UUID beside)."""
    import json

    ids = []
    for e, members in sorted(declared.items()):
        for rid, n in members:
            for name in B8_MEMBERS:
                op, mem = (name.split("/", 1) + ["out"])[:2] if "/" in name else (name, "out")
                fam = "sampled_token_ids" if mem == "sampled_token_ids" else ("input_token_ids" if mem == "in" else "instance_outputs")
                hi = 1 if fam == "sampled_token_ids" else n
                ids.append({"family": fam, "program_digest": PD, "request_id": rid, "step": e, "engine_step": e, "invocation": 0, "op_path": op,
                            "output_member": mem, "element_range": [0, hi], "rows": n, "width": 1, "rank": 0})
    from verity_vllm.query.manifest.format import manifest_digest as _mdig

    doc = {"schema": "verity-sweep/required-manifest/v1", "global": True, "addressing": "request-local", "program_digest": PD, "program_digest_kind": "workload",
           "manifest_digest": _mdig(ids), "complete": True, "unmodelled": {},
           "requests_declared": [{"request_id": f"r{k}", "workload_request_id": f"uuid-{k}", "index": k, "prompt_len": B8_LP[k], "max_tokens": B8_CAP[k], "arrive_step": 0} for k in range(8)],
           "engine_steps": [{"engine_step": e, "members": [{"request_id": rid, "step": e, "rows": n} for rid, n in members], "rows": sum(n for _r, n in members), "n_seq": len(members)}
                            for e, members in sorted(declared.items())],
           "composition_reference": {"input_token_ids": {"op_path": "model.embed_tokens", "output_member": "in", "per_row": 1},
                                     "sampled_token_ids": {"op_path": "runner.sampler", "output_member": "sampled_token_ids", "per_row": 1}},
           "identities": ids}
    p = tmp_path / "b8_manifest.json"
    p.write_text(json.dumps(doc))
    return str(p)












def _repack(rr: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    out, lo = [], 0
    for r, a, b in rr:
        out.append((r, lo, lo + (b - a)))
        lo += b - a
    return out


# ---------------------------------------------------------------------- manifest input trust

def _b8_doc(tmp_path):
    import json

    p = b8_manifest(tmp_path, b8_declared())
    return p, json.loads(open(p).read())


def _write(tmp_path, name, doc):
    import json

    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return str(p)


def test_manifest_edited_after_digesting_is_refused(tmp_path):
    p, doc = _b8_doc(tmp_path)
    doc["identities"].pop(7)                                              # rows changed, manifest_digest kept
    with pytest.raises(ValueError, match="edited after digesting"):
        B.load_required_manifest(_write(tmp_path, "edited.json", doc), PD)
    doc["manifest_digest"] = "0" * 128
    with pytest.raises(ValueError, match="edited after digesting"):
        B.load_required_manifest(_write(tmp_path, "edited2.json", doc), PD)
    doc["manifest_digest"] = "0" * 64
    with pytest.raises(ValueError, match="a 64-hex SHA-256 digest: .* so rebuild it"):
        B.load_required_manifest(_write(tmp_path, "sha256.json", doc), PD)
    doc["schema"] = "verity-sweep/required-manifest/v0"
    with pytest.raises(ValueError, match="schema"):
        B.load_required_manifest(_write(tmp_path, "schema.json", doc), PD)


def test_manifest_zero_width_identity_is_refused_not_vacuously_covered(tmp_path):
    from verity_vllm.query.manifest.format import manifest_digest as _mdig

    p, doc = _b8_doc(tmp_path)
    doc["identities"][3]["element_range"] = [0, 0]                        # an EXISTING identity made zero-width
    doc["manifest_digest"] = _mdig(doc["identities"])                     # ... with an honestly recomputed digest
    with pytest.raises(ValueError, match="zero-width"):
        B.load_required_manifest(_write(tmp_path, "zw.json", doc), PD)
    p, doc = _b8_doc(tmp_path)
    doc["identities"].append(dict(doc["identities"][0], op_path="model.layers.0.mlp.phantom", element_range=[0, 0]))   # phantom op, zero width
    doc["manifest_digest"] = _mdig(doc["identities"])
    with pytest.raises(ValueError, match="zero-width"):
        B.load_required_manifest(_write(tmp_path, "zw2.json", doc), PD)


def test_manifest_request_id_not_r_k_is_refused(tmp_path):
    from verity_vllm.query.manifest.format import manifest_digest as _mdig

    p, doc = _b8_doc(tmp_path)
    doc["identities"][0]["request_id"] = "5b7fee10-6a81-52b7-b730-133d6cc768da"   # a bare workload UUID no declaration resolves
    doc["manifest_digest"] = _mdig(doc["identities"])
    with pytest.raises(ValueError, match="not r<k>"):
        B.load_required_manifest(_write(tmp_path, "uuid.json", doc), PD)
    # ... but a UUID that requests_declared maps to r<k> is fine
    p, doc = _b8_doc(tmp_path)
    doc["identities"][0]["request_id"] = "uuid-0"
    doc["requests_declared"][0]["request_id"] = "uuid-0"
    doc["manifest_digest"] = _mdig(doc["identities"])
    pop, hdr = B.load_required_manifest(_write(tmp_path, "uuid_ok.json", doc), PD)
    assert hdr["manifest_digest_verified"] and any(i[1] == "r0" for i in pop)


def test_manifest_program_digest_must_be_a_string(tmp_path):
    from verity_vllm.query.manifest.format import manifest_digest as _mdig

    p, doc = _b8_doc(tmp_path)
    doc["program_digest"] = [PD, "another"]                               # the rc3 merge_manifests shape: a list is not a Program of record
    doc["manifest_digest"] = _mdig(doc["identities"])
    with pytest.raises(ValueError, match="program_digest must be one string"):
        B.load_required_manifest(_write(tmp_path, "pdlist.json", doc), PD)






