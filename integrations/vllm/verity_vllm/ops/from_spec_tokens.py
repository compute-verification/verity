"""One engine built two ways, `LLM.from_spec(SPEC)` and `LLM(model, revision=..., **keywords)` with the spec's options,
must sample identical greedy tokens.

    python verity_vllm/ops/from_spec_tokens.py SPEC OUT      (from integrations/vllm, on a GPU node)

Each build runs in its own subprocess (one engine per process), over the same prompts.  Writes OUT/<build>.json (tokens,
engine_record) and OUT/verdict.json: PASS when every request's tokens agree and the spec build carries the spec's digest.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

N_REQUESTS, PROMPT_TOKENS, MAX_TOKENS = 8, 128, 64


def child(build: str, spec_path: str, out: Path) -> None:
    from verity_vllm.engine import env as engine_env

    engine_env.apply_engine(None)
    from vllm import SamplingParams

    from verity_vllm.pipeline import engine_spec
    from verity_vllm.pipeline.llm import LLM

    spec = engine_spec.load(spec_path)
    llm = LLM.from_spec(spec) if build == "spec" else LLM(spec.model, revision=spec.revision, **spec.options())
    vocab = int(llm.vllm.llm_engine.vllm_config.model_config.hf_config.vocab_size)
    prompts = [{"prompt_token_ids": [3 + (i * 1_000_003 + j * 97_409) % (vocab - 4) for j in range(PROMPT_TOKENS)]}
               for i in range(N_REQUESTS)]
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True, detokenize=False, seed=0)
    outs = llm.generate(prompts, params, use_tqdm=False)
    doc = {"build": build, "tokens": [list(o.outputs[0].token_ids) for o in outs], "engine_record": llm.engine_record}
    (out / f"{build}.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    llm.close()


def main(spec_path: str, out: Path) -> int:
    from verity_vllm.pipeline import engine_spec

    out.mkdir(parents=True, exist_ok=True)
    for build in ("spec", "keywords"):
        rc = subprocess.run([sys.executable, __file__, "--child", build, spec_path, str(out)]).returncode
        if rc:
            (out / "verdict.json").write_text(json.dumps({"verdict": "FAIL", "why": f"the {build} build exited {rc}"}) + "\n")
            return rc
    spec, kw = (json.loads((out / f"{b}.json").read_text()) for b in ("spec", "keywords"))
    digest = engine_spec.load(spec_path).digest
    differ = [i for i, (a, b) in enumerate(zip(spec["tokens"], kw["tokens"], strict=True)) if a != b]
    ok = not differ and spec["engine_record"]["spec_digest"] == digest and kw["engine_record"]["spec_digest"] is None
    verdict = {"verdict": "PASS" if ok else "FAIL", "spec_digest": digest, "requests": len(spec["tokens"]),
               "tokens_per_request": MAX_TOKENS, "differing_requests": differ,
               "engine_kwargs_equal": spec["engine_record"]["engine_kwargs"] == kw["engine_record"]["engine_kwargs"]}
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1) + "\n")
    print(json.dumps(verdict))
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
    else:
        sys.exit(main(sys.argv[1], Path(sys.argv[2])))
