"""φ of a row, and what a predicted Program is put beside a traced Build's with.

`phi_of` reads φ from a row's DECLARED inputs only (the row id, its workload manifest, the declared target
`target_family.declared_profile`, the pinned checkpoint's `config.json`), never from a Build output: the traced Programs are
what a prediction is compared with (`aot`).  `request_component`, `workload_program` and `predict_workload` compose the
predicted request Programs into GP-01's workload Program, `load_descriptor` reads a traced descriptor, and `code_version` is
the predictor's code, which a predicted row's record states.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pkgutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

import verity
from verity.circuits.codec import descriptor_id

import verity_vllm
from verity_vllm.config import DTYPES, RowEngine, RowId
from verity_vllm.engine.profiles.devices import device_name_of_row
from verity_vllm.program import workload as W
from verity_vllm.target_family import declared_profile

from .build import Unexpressible, build_program
from .phi import Phi

# ------------------------------------------------------------------------------------------------------------------- φ of a row


def hf_config_path(hf_home: str, repo: str, revision: str) -> str:
    return os.path.join(hf_home, "models--" + repo.replace("/", "--"), "snapshots", revision, "config.json")


def phi_of(rec: dict, *, hf_home: str | None = None, config: str | None = None) -> Phi:
    """φ from a row's declared inputs: never a Build output (the traced Programs are what φ's prediction is scored against).  The
    checkpoint's `config.json` is `config`, else the pinned snapshot's under `hf_home`."""
    rid = RowId.parse(rec["row"])
    w = rec.get("workload")
    if not w:
        raise Unexpressible(f"row {rec['row']}: no declared workload manifest at {rec.get('workload_path')}")
    repo = w.get("model")
    rev = RowEngine.of(w).revision
    cp = config or hf_config_path(str(hf_home), repo, str(rev))
    if not os.path.exists(cp):
        raise Unexpressible(f"no config.json for {repo}@{rev} under {hf_home}")
    hf = json.loads(Path(cp).read_text())
    declared = w.get("requests")   # a manifest that states no requests declares none
    reqs = [{"request_id": q.get("request_id"), "index": q.get("index", i), "prompt_len": q.get("prompt_len") or q.get("prompt_token_ids"),
             "max_tokens": q.get("max_tokens"), "arrive_step": q.get("arrive_step", 0), "n": q.get("n"), "seed": q.get("seed")}
            for i, q in enumerate(declared)] if declared else []
    target = declared_profile(rec["row"], w)
    if (device := device_name_of_row(rec["row"])) is not None:
        target["device_name"] = device
    eng = RowEngine.of(w)
    phi = Phi(model={"repo": repo, "revision": rev, "hf_config": hf, "dtype": DTYPES.get(rid.dtype, rid.dtype)}, target=target,
              tp={"world": rid.world, "rank": 0}, engine={**eng.args, **({"env": dict(eng.env)} if eng.env else {})},
              workload={"requests": reqs}, execution=rid.execution, profile=target.get("numerics_profile") or "b1-eager-v3")
    phi.sampling = sampling_of(w, phi)
    return phi


#: the policy fields a component Program reads (`vllm_meta.make_wrapper`); the rest of the Build's policy record is provenance
_POLICY_FIELDS = ("kind", "temperature", "top_p", "top_k", "seeded", "splits", "rid")


def sampling_of(w: dict, phi: Phi) -> dict:
    """The row's sampling policy as the Build declares it (`pipeline.build.sampling_policy` over the workload manifest, then
    `sampler_geometry` on the declared target: the registered split count)."""
    from verity_vllm.pipeline.build import sampler_geometry, sampling_policy
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "workload.json")
        with open(path, "w") as f:
            json.dump(w, f)
        pol = sampling_policy(None, path)
        refusal = sampler_geometry(pol, phi.target_profile()) if pol.get("kind") == "stochastic" else None
    if refusal is not None:
        raise Unexpressible(f"sampling: the Build refuses the policy ({refusal.get('hint')})")
    return {k: pol[k] for k in _POLICY_FIELDS if k in pol}


# ------------------------------------------------------------------------------------------------------------------- the workload Program


def request_component(r):
    """A predicted request Program (a `BuildResult`) as GP-01's component (`program.workload.program_component`): its own root,
    whose definitions are this process's, so nothing is decoded."""
    params = [(name, repr(t)) for name, t in r.program.fn.signature()[0]]
    fn = getattr(r.program, "program_fn_id", None) or descriptor_id(r.program.fn)
    return W.program_component(r.program.root, r.digest, fn, params)


def predict_workload(phi: Phi, comps: dict | None = None) -> dict:
    """`workload_program`'s digest, function id, root node count and lag."""
    wl = workload_program(phi, comps)
    return {"digest": wl.digest, "function": wl.fn.id[:300], "n_root_nodes": len(wl.program.root.body.nodes), "lag": wl.lag}


def workload_program(phi: Phi, comps: dict | None = None):
    """GP-01's composition (`program.workload.WorkloadProgram`) over the predicted request components, manifest order, with the
    lag the declared target resolves (`TargetProfile.lag()`); `comps` holds components already built (`request_component`), by
    (LP, T), so a caller that has each request's Program need not hold them all."""
    from verity_vllm.pipeline.global_program import request_id_of
    reqs = phi.requests
    if any(int(r.get("n") or 1) != 1 for r in reqs):
        raise Unexpressible("a request with n > 1 (parallel sampling) in the workload")
    comps = dict(comps or {})
    for LP, T in phi.request_shapes():
        if (LP, T) not in comps:
            comps[(LP, T)] = request_component(build_program(phi, "request", LP=LP, T=T))
    wreqs = []
    for i, q in enumerate(reqs):
        lp, cap = int(q["prompt_len"]), int(q["max_tokens"])
        wreqs.append(W.WorkloadRequest(request_id_of(i), i, lp, cap, int(q.get("arrive_step") or 0), comps[(lp, cap - 1)], T=cap - 1))
    return W.WorkloadProgram(wreqs, manifest_digest=None, lag=int(phi.target_profile().lag()))


# ------------------------------------------------------------------------------------------------------------------- a traced descriptor


def load_descriptor(path: str) -> dict:
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------------------------------- the predictor's code


def code_version() -> str:
    """The predictor's code: a digest of the files git tracks in the two packages a predicted Program is built from (`verity`,
    `verity_vllm`), as they are on disk.  Only tracked files count (`git ls-files`), so a file a test, a build or a run writes into
    a package's tree (a `.lake`, a cache, a report) never moves it; an edit to a tracked file does.  A namespace package's portions
    are hashed as one tree, by path under the package, and only those holding a module, so neither sys.path order nor a directory
    that joined the namespace from the cwd moves it.  A portion outside a git checkout is refused: nothing says which of its files
    are source."""
    h = hashlib.sha256()
    for pkg in (verity, verity_vllm):
        files = []
        for root in (Path(r).resolve() for r in pkg.__path__ if any(pkgutil.iter_modules([r]))):
            ls = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True)
            if ls.returncode:
                raise RuntimeError(f"code_version: {root} is not in a git checkout, so its tracked files aren't known; run from a "
                                   f"checkout (research ships one) ({ls.stderr.decode(errors='replace').strip()})")
            files += [(PurePosixPath(rel), root / rel) for rel in ls.stdout.decode().split("\0") if rel]
        for rel, p in sorted(files):
            if p.is_file():
                h.update(f"{pkg.__name__}/{rel}\0".encode() + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:16]
