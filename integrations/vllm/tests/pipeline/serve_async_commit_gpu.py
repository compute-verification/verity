"""serve-openai's Commit under ``--async-scheduling`` against the synchronous one, on a GPU: the same load, served once under
each scheduler, commits the same rows under the same rows-sha512 roots, and every async window commits.

    python -m tests.pipeline.serve_async_commit_gpu ab --out DIR [--batches 8] [--requests 8] [--prompt-ids 1024]
        [--port 18300] [--row-key-seed vasync-ab] -- SERVE-OPENAI-ARGS        (from integrations/vllm, on one GPU)
    python -m tests.pipeline.serve_async_commit_gpu compare SYNC_DIR ASYNC_DIR

`ab` serves SERVE-OPENAI-ARGS (``--run-id``, ``--admit release-order``, a bench row, the Commit's options, ``--window-steps
1``) twice, synchronously and then with ``--async-scheduling``, each a `serve` of this module in its own process, and sends
each the same ``--batches`` loads of ``examples/vllm/serve_load.py --fixed``: ``--requests`` requests of ``--prompt-ids``
ids for one step, the learn-only warm-up's static shape (`serve_learn.shape_of`), each batch its own release run and seed.
A step's rows-sha512 root binds its salt key, fresh from the OS generator in a served run, the run id and the row's
Program, so `serve` runs serve-openai with the step keys derived from ``--row-key-seed`` (`fixed_row_keys`) and the bench
row taken whichever scheduler it declares (`either_row`): the two arms differ only in the scheduler. `compare` then holds
the arms' ``windows.jsonl`` to each other and writes ``compare.json``: PASS when every window of each arm passed and
committed bytes, and the two committed the same steps under equal roots, in order."""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

LOAD = Path(__file__).resolve().parents[4] / "examples" / "vllm" / "serve_load.py"


def fixed_row_keys(seed: str) -> None:
    """Every rows-sha512 step key SHA-512(``seed``/step=k) in place of the OS generator's, so two runs of the same rows commit
    the same roots. A test's patch only: a served run's keys are fresh, which is what keeps its leaves hiding."""
    from verity_vllm.commit import native_ranges as NR

    def row_key(self: Any, step: int) -> bytes:
        self.row_keys[step] = ks = hashlib.sha512(f"{seed}/step={step}".encode()).digest()
        return ks

    NR.NativeRangeOpenings._row_key = row_key


def either_row() -> None:
    """`serve_row.of` taking the bench row whichever scheduler its workload Program declares, so both arms serve one row (one
    Program digest in every step's domain)."""
    from verity_vllm.pipeline import serve_row as RW

    def of(opts: Any) -> dict[str, Any] | None:
        if not opts.required_manifest:
            return None
        from verity_vllm.pipeline import serve_seal as SS
        path = Path(SS.programs_root_of(opts)) / RW.WORKLOAD_PROGRAM
        return {"async_scheduling": json.loads(path.read_text())["applicability"]["engine"]["async_scheduling"], "source": str(path)}

    RW.of = of


def serve(seed: str, args: list[str]) -> Any:
    fixed_row_keys(seed)
    either_row()
    from verity_vllm.pipeline import cli
    sys.argv[:] = [cli.__file__, "serve-openai", *args]
    return cli.main()


def windows(d: Path) -> list[dict]:
    p = d / "windows.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []


def compare(sync: list[dict], asy: list[dict]) -> dict[str, Any]:
    """The verdict over two arms' windows (``windows.jsonl``'s, those that held steps)."""
    held = [[w for w in ws if w.get("steps")] for ws in (sync, asy)]
    roots = [[r for w in ws for r in w.get("step_roots") or []] for ws in held]
    failed = [[w["window"] for w in ws if not (w.get("passed") and w.get("committed_bytes"))] for ws in held]
    differ = [i for i, (x, y) in enumerate(zip(*roots)) if x != y]
    return {"windows": [len(ws) for ws in held], "steps": [len(r) for r in roots], "sync_failed": failed[0][:10],
            "async_failed": failed[1][:10], "differ": differ[:10], "roots_equal": sum(x == y for x, y in zip(*roots)),
            "passed": bool(roots[1]) and not failed[0] and not failed[1] and len(roots[0]) == len(roots[1]) and not differ}


def verdict(sync_dir: Path, async_dir: Path, out: Path) -> int:
    got = compare(windows(sync_dir), windows(async_dir))
    (out / "compare.json").write_text(json.dumps(got, indent=1))
    print(f"SERVE-ASYNC-COMMIT {'PASS' if got['passed'] else 'FAIL'}: {json.dumps(got)}", flush=True)
    return 0 if got["passed"] else 1


def arm(name: str, out: Path, a: argparse.Namespace) -> Path:
    """One scheduler's serve-openai lifetime under the A/B's load; its out dir."""
    d, port = out / name, a.port + (name == "async")
    d.mkdir(parents=True, exist_ok=True)
    model = a.serve[a.serve.index("--model") + 1]
    argv = [sys.executable, "-m", "tests.pipeline.serve_async_commit_gpu", "serve", "--row-key-seed", a.row_key_seed, "--",
            *a.serve, "--out", str(d), "--port", str(port), *(["--async-scheduling"] if name == "async" else [])]
    with open(d / "serve.log", "w") as log:
        srv = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in srv.stdout:
            log.write(line)
            if "SERVE-OPENAI listening" in line:
                break
        else:
            raise RuntimeError(f"{name}: serve-openai exited {srv.wait()} before it listened ({d / 'serve.log'})")
        drain = threading.Thread(target=lambda: log.writelines(srv.stdout))
        drain.start()
        for b in range(1, a.batches + 1):
            subprocess.run([sys.executable, str(LOAD), "--port", str(port), "--model", model, "--fixed", str(a.requests),
                            "--window-steps", "1", "--prompt-range", f"{a.prompt_ids}:{a.prompt_ids}", "--release", f"ab{b}",
                            "--seed", str(b), "--out", str(d / "load.jsonl")], check=True)
        srv.send_signal(signal.SIGTERM)
        print(f"{name}: serve-openai exited {srv.wait()}", flush=True)
        drain.join()
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("ab", "serve"):
        p = sub.add_parser(cmd)
        p.add_argument("--row-key-seed", default="vasync-ab")
        if cmd == "ab":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--batches", type=int, default=8)
            p.add_argument("--requests", type=int, default=8)
            p.add_argument("--prompt-ids", type=int, default=1024)
            p.add_argument("--port", type=int, default=18300)
        p.add_argument("serve", nargs=argparse.REMAINDER)
    c = sub.add_parser("compare")
    c.add_argument("sync", type=Path)
    c.add_argument("asy", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        return verdict(a.sync, a.asy, a.sync.parent)
    a.serve = a.serve[1:] if a.serve[:1] == ["--"] else a.serve
    if a.cmd == "serve":
        return serve(a.row_key_seed, a.serve)
    a.out.mkdir(parents=True, exist_ok=True)
    return verdict(arm("sync", a.out, a), arm("async", a.out, a), a.out)


if __name__ == "__main__":
    raise SystemExit(main())
