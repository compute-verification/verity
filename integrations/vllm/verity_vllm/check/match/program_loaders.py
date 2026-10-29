"""The comparator's two loaders, each returning a `Prog`: a derive dir's Program (`load_derived`) and a fold's files of record
(`load_oracle`, its form and digests stated in `LOAD_NOTES`)."""

from __future__ import annotations

import gzip
import json
import os

from verity_vllm.check.match import prefix_reads as PX
from verity_vllm.check.match import supplied_inputs as SI
from verity_vllm.check.match.instance_sequence import Prog
from verity_vllm.program.compact import p_item_indices as _p_indices, p_item_weight_span as _p_w_span  # [GLM-5.3 "P"]


def load_derived(d: str, label: str, cls: type[Prog] = Prog, compact: bool = False) -> Prog:
    """Derive dir ``d``'s Program as the Match reads it, in word units: a Boolean derive's word Program (`program.boolean.word_descriptor`)."""
    from verity.ml.definitions.boolean import word_descriptor
    desc = json.load(gzip.open(word_descriptor(d)))
    rep = json.load(open(os.path.join(d, "derive-report.json")))
    prov = rep.get("provenance", rep)
    src = prov.get("source") if isinstance(prov.get("source"), dict) else {}
    # the module's USER inputs precede the weight fields: `example_inputs` records one entry per input (the greedy wrapper's `prompt`;
    # the stochastic wrapper's `prompt` + `seed`); a report without it is the one-input greedy wrapper
    user_inputs = src.get("example_inputs") or src.get("inputs") or [[0]]
    n_user = max(1, len(user_inputs))
    reg = SI.derived_registration(prov, n_user, d)   # a `registered_values` derivation's `values`, between them
    fields = {n_user + bool(reg) + w["field"]: w["name"] for w in prov["weights"]}  # derive: user inputs first, then one Input batch per field
    P = cls(label, desc, fields, tok_node=0, compact=compact)
    SI.set_supplied(P, [(n_user, None, reg)] if reg else [], d)
    # a TP rank Program (derive_step --tp): `tp_links.json` names the user inputs in order (`prompt`, `peers_0`, ... = the k-th
    # collective's peer partial, issue order); each `peers_<k>` is its own Input node (one field at offset 0), and so is each WITNESS (a
    # user input the table lists neither among `prompt_inputs` nor `peer_inputs`: the MoE align `schedule`), bound by its name.  Refused
    # by name when the link table and the report disagree on how many user inputs there are.
    tl = os.path.join(d, "tp_links.json")
    if os.path.exists(tl):
        links = json.load(open(tl))
        names = list(links.get("user_inputs") or [])
        if len(names) != n_user:
            raise ValueError(f"{d}: tp_links.json names {len(names)} user inputs, derive-report has {n_user}")
        listed = set(links.get("prompt_inputs", names[:1])) | set(links.get("peer_inputs") or ())
        for j, nm in enumerate(names):
            peer, witness = nm.startswith("peers_"), nm not in listed
            if (peer or witness) and j not in P.inputs:
                raise ValueError(f"{d}: user input {j} ({nm}) is not an Input node of the Program")
            if peer:
                P.peer_fields[j] = [(0, int(nm[len("peers_"):]))]
            elif witness:
                P.witness_fields[j] = nm
        P.tp = {"rank": links.get("rank"), "world": links.get("world")}
        P.tp_links = links.get("links") or []
    PX.load_layout(P, d)
    return P


def load_oracle(d: str, label: str, cls: type[Prog] = Prog, compact: bool = False) -> Prog:
    # the checker reads the FILES OF RECORD and nothing beside them: a compact copy written into the fold dir by
    # the record's producer (unit 7's sidecars, withdrawn) is prover input whose content nothing the checker owns authenticates -- a forged
    # sidecar with the sources' true digests/sizes/sha256 pasted in was accepted.  `instances.jsonl` itself may be stored in either
    # form (F2, run_config.INSTANCES_SCHEMA_V2: operand references as progressions); this loader keeps only the parameter-space items --
    # "p" runs and [GLM-5.3] "P" progressions, read without expansion (a scattered weight operand is ~1e6 runs) -- so its semantics do not
    # depend on the form.  The form read is stated (LOAD_NOTES -> impl.config.fold_load) so two documents of one capture under two stores are distinguishable by content.
    # the two record files are hashed AS READ (one pass: every byte the loader consumes
    # goes through sha256 before it is decoded), and the digests are stated beside the form (LOAD_NOTES -> impl.fold_load[dir].program_sha256 /
    # instances_sha256): a Match document -- the record checker's as much as the fast one's -- is tied to the IMMUTABLE INPUT ARTIFACTS by
    # content, so the audit tool compares the two documents' digests with the row's files (and with each other) instead of trusting that
    # two oracle_dir paths named the same bytes.  ~2-3 s per GB at load; nothing else changes (rows, form handling, the Prog).
    import hashlib
    raw = open(os.path.join(d, "program.json"), "rb").read()
    program_sha = hashlib.sha256(raw).hexdigest()
    pj = json.loads(raw)
    del raw
    desc = pj["program"]
    desc.setdefault("digest", pj.get("digest"))
    rows = []
    ih = hashlib.sha256()
    with open(os.path.join(d, "instances.jsonl"), "rb") as f:
        for line in f:
            ih.update(line)
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if "schema" in o and "node" not in o:   # the header line
                LOAD_NOTES[d] = {"source": "record files", "instances_schema": o.get("schema"),
                                 "operand_refs_form": o.get("operand_refs_form") or "runs", "program_sha256": program_sha}
            if "node" in o:
                # only the PARAMETER-space runs ("p": prompt / weights reads -- O(instances)) of each
                # operand are kept from `operand_refs`; the node-space runs ("n": the attention K/V reads, O(LP^2) over a long prefill) are
                # the same references `Prog.inst[*].args` holds and are never read from the rows.  Operand arity is preserved (one list per
                # operand, possibly empty) so `operand_roots[k]` still names operand k below.
                rows.append({"node": o["node"], "step": o.get("step"), "row": o.get("row"), "rows": o.get("rows"), "hint": o.get("hint"),
                             "spec": o.get("spec"), "operand_refs": [[r for r in a if r[0] in ("p", "P")] for a in (o.get("operand_refs") or [])],
                             "pattern": o.get("pattern"),
                             "request": o.get("request"),  # the fold's own attribution when recorded
                             "position": o.get("position"),  # prefix_reads links a cached row to its producer by position (and layer: hint)
                             "operand_roots": o.get("operand_roots")})  # per-operand weight names (multi-weight Computes)
    if d not in LOAD_NOTES:     # a pre-header instances.jsonl (no schema line): the form is the default, the digests are still stated
        LOAD_NOTES[d] = {"source": "record files", "instances_schema": None, "operand_refs_form": "runs", "program_sha256": program_sha}
    LOAD_NOTES[d]["instances_sha256"] = ih.hexdigest()
    return _finish_oracle(d, label, pj, desc, rows, cls, compact)


LOAD_NOTES: dict[str, dict] = {}   # fold dir -> {"source": "record files", "instances_schema": <header schema>, "operand_refs_form": runs|progressions}


def _finish_oracle(d: str, label: str, pj: dict, desc: dict, rows: list[dict], cls: type[Prog] = Prog, compact: bool = False) -> Prog:
    """The rest of load_oracle over the retained rows."""
    # weight layout: the WEIGHTS parameter's offset ranges -> weight name from the hints ("module/param.name"). The fold lays
    # out one prompt parameter per request followed by the weights (`fold._header`): weights = param n_input_nodes-1 (== 1
    # only at B=1; the hard-coded 1 read request 1's PROMPT as the weight map at B>1).
    n_in = pj.get("n_input_nodes", 2)
    # the weights Record is the LAST root parameter (fold contract) -- its index is the largest `p` index any instance reads,
    # not `n_input_nodes - 1`: a Record that is not width-homogeneous (FP8: e4m3 weights, f32 block scales, bf16 norms) is encoded as
    # ONE Input node per leaf field (436 for Qwen3-4B-FP8 at B=1), so the node count no longer says how many prompt parameters there are.
    p_idx = {i for o in rows for a in (o.get("operand_refs") or []) for r in a for i in _p_indices(r)}
    w_param = max(p_idx) if p_idx else n_in - 1
    # the fold states its parameter layout by name (program.json `params`): the weights index is read from it when present
    # (exact, even for a fold whose instances read no weight -- a fixture) and cross-checked against the instances' largest "p" index.
    names = [n for n, _t in (pj.get("params") or [])]
    if names:
        if names[-1] != "weights":
            raise ValueError(f"{d}: program.json params end with {names[-1]!r}, the fold contract puts `weights` LAST")
        if p_idx and max(p_idx) != len(names) - 1:
            raise ValueError(f"{d}: instances read parameter {max(p_idx)} as the largest, program.json puts `weights` at {len(names) - 1}")
        w_param = len(names) - 1
    p = cls(label, desc, {}, tok_node=0, oracle_rows=rows, compact=compact)
    # a tensor-parallel rank fold: one homogeneous `peers_<i>` Record per request = ONE Input node at the parameter's index (every
    # parameter before the weights is homogeneous, one node each); its fields (issue order, leaf offsets) come from `peers_layout`.  The k of a
    # field is its ordinal among the REQUEST's fields = the rank Program's `peers_<k>` for that request.
    for ent in pj.get("peers_layout") or []:
        node = int(ent["param"])
        if node not in p.inputs:
            raise ValueError(f"{d}: peers parameter {ent.get('name')!r} (index {node}) is not an Input node of the Program")
        flds = sorted((int(f["offset"]), int(f["ordinal"])) for f in ent.get("fields") or [])
        if not flds:
            raise ValueError(f"{d}: peers parameter {ent.get('name')!r} has no fields")
        if sum(1 for _o, k in flds) != len({k for _o, k in flds}):
            raise ValueError(f"{d}: peers parameter {ent.get('name')!r} has duplicate field ordinals")
        p.peer_fields[node] = flds
        p.peer_index[node] = int(ent["request_index"])
    # a witness parameter (`witness_layout`: the Marlin align's `schedule`) is one homogeneous Input node, bound by its name as the rank
    # Program's witness input of that name is (`load_derived`)
    for ent in pj.get("witness_layout") or []:
        node = int(ent["param"])
        if node not in p.inputs:
            raise ValueError(f"{d}: witness parameter {ent.get('name')!r} (index {node}) is not an Input node of the Program")
        p.witness_fields[node] = ent["witness"]
        p.witness_index[node] = int(ent["request_index"])
    p.tp = pj.get("tp")
    w_nodes = SI.load_oracle_supplied(p, pj, d)   # a `registered_values` fold's supplied parameters
    del pj, desc   # the descriptor JSON (12.5 GB on disk for the 4096/512 fold) is not needed past the decode
    # Input node -> offset of its first leaf inside the weights parameter (0 for a single homogeneous Record node -- the older layout):
    # the codec emits the prompt parameters' nodes first (one each, `w_param` of them), then the Record's leaf batches in field order.
    w_nodes = w_nodes if w_nodes is not None else {k for k in p.inputs if k >= w_param}
    base, acc = {}, 0
    for k in sorted(p.inputs):
        if k in w_nodes:
            base[k] = acc
            acc += int(p.inputs[k][1])
    p.input_base = base
    # Field = the union of every range the instances read under one weight name (the fold's hint `module/param`; the
    # RoPE table is hinted `module/rope.q|k` per position -> the module's `cos_sin_cache`). Equal ranges keep the first name.
    # a Compute that reads several weights (the block-scaled FP8 GEMM: `weight` AND `weight_scale_inv`) records the name of
    # each weight operand in `operand_roots` (fold meta, from `Compute.roots`); operand k is then spanned under ITS name, never
    # the hint's -- the union under one name would fuse two fields into a range the derived program does not have.
    span: dict[str, list[int]] = {}
    for o in rows:
        hint = o.get("hint") or ""
        mod, _, tail = hint.rpartition("/")
        if tail in ("rope.q", "rope.k"):
            name = f"{mod}.cos_sin_cache"
        else:
            name = tail or hint
        oroots = o.get("operand_roots") or []
        for k, a in enumerate(o.get("operand_refs") or []):
            nm = oroots[k] if k < len(oroots) and oroots[k] else name
            for r in a:
                if (w := _p_w_span(r, w_param)) is not None:
                    sp = span.setdefault(nm, [w[0], w[1]])
                    sp[0], sp[1] = min(sp[0], w[0]), max(sp[1], w[1])
    byrange: dict[tuple[int, int], str] = {}
    for name, (lo, hi) in span.items():
        byrange.setdefault((lo, hi), name)
    p.oracle_wmap = sorted((lo, hi, nm) for (lo, hi), nm in byrange.items())
    # fields must not overlap (else the map is ambiguous): report, never guess
    p.oracle_wmap_overlaps = [(a, b) for a, b in zip(p.oracle_wmap, p.oracle_wmap[1:]) if a[1] > b[0]]
    p._wlos = [lo for lo, _, _ in p.oracle_wmap]
    p.n_input_nodes = n_in
    # the (parameter-space) `operand_refs` served only `w_param` and the weight-field map above; they
    # are dropped from the retained rows.  `operand_roots` (one short name per operand) stays.  Consumers of `oracle_rows` (batch_decomp,
    # global_match.canonicalize_record) read node/step/row/rows/request/spec.
    for o in rows:
        o["operand_refs"] = None
    return p
