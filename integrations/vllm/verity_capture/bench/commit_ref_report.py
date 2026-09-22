"""Component table from `commit_ref_bench` raw samples (EXPERIMENTAL, CMT-1).

    python -m verity_capture.bench.commit_ref_report out/gen/r9/cmt-ref/bench [more bench dirs...]

Every number is [measured] on the GPU named in the samples; means over reps with min..max; prefill = step 0, decode = steps >= 1.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path


def _fmt(xs, scale=1.0, nd=2):
    xs = [x * scale for x in xs if x is not None]
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.{nd}f}"
    return f"{st.mean(xs):.{nd}f} [{min(xs):.{nd}f}..{max(xs):.{nd}f}]"


def main(dirs: list[str]) -> None:
    drop = set()
    if "--drop-rep0-for" in dirs:                       # arms whose rep 0 carries a one-time kernel compile: report warm reps only
        i = dirs.index("--drop-rep0-for")
        drop = set(dirs[i + 1].split(","))
        dirs = dirs[:i] + dirs[i + 2:]
    rows = []
    for d in dirs:
        for line in open(Path(d, "raw_samples.jsonl")):
            r = json.loads(line)
            r["_dir"] = d
            if r["arm"] in drop and r["rep"] == 0:
                continue
            rows.append(r)
    arms = []
    for r in rows:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    gpus = sorted({r["gpu"] for r in rows})
    ctrl = [r for r in rows if r["arm"] == "control"]
    ctrl_pre = [r["steps"][0]["step_ms"] for r in ctrl]
    ctrl_dec = [st.mean(s["step_ms"] for s in r["steps"][1:]) for r in ctrl]
    L = [f"## Per-component costs — {', '.join(gpus)} — reps per arm as shown{' (rep 0 dropped for ' + ','.join(sorted(drop)) + ': one-time Triton compile)' if drop else ''}; ms unless stated; prefill = step 0 (155 tokens), decode = mean over steps 1..14 (1 token)", ""]
    L.append("| arm | reps | step total prefill | step total decode | fwd (incl. hook copies) prefill | fwd decode | Δ vs control decode | acquisition copies/step | D2H prefill | D2H decode | leaf hash prefill | leaf hash decode | tree prefill | tree decode | commit enqueue prefill | commit enqueue decode | finish (request root) | peak extra mem MB | root |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        rs = [r for r in rows if r["arm"] == a]
        pre = [r["steps"][0] for r in rs]
        dec = [r["steps"][1:] for r in rs]

        def m(key, which, scale=1.0, nd=2):
            if which == "pre":
                return _fmt([p.get(key) for p in pre], scale, nd)
            return _fmt([st.mean(s.get(key) or 0.0 for s in ds) if ds and any(key in s for s in ds) else None for ds in dec], scale, nd)

        d_dec = [st.mean(s["step_ms"] for s in ds) - st.mean(ctrl_dec) for ds in dec] if ctrl_dec else []
        L.append(f"| {a} | {len(rs)} | {m('step_ms', 'pre')} | {m('step_ms', 'dec')} | {m('fwd_ms', 'pre')} | {m('fwd_ms', 'dec')} | {_fmt(d_dec)} | {pre[0].get('copies', 0)} | "
                 f"{m('d2h_s', 'pre', 1e3)} | {m('d2h_s', 'dec', 1e3, 3)} | {m('leaf_hash_s', 'pre', 1e3)} | {m('leaf_hash_s', 'dec', 1e3)} | {m('tree_s', 'pre', 1e3)} | {m('tree_s', 'dec', 1e3)} | "
                 f"{m('commit_enqueue_s', 'pre', 1e3)} | {m('commit_enqueue_s', 'dec', 1e3)} | {_fmt([r['finish_s'] for r in rs], 1e3)} | {_fmt([r['mem_peak_extra_bytes'] for r in rs], 1e-6, 1)} | {(rs[0]['root'] or '-')[:16]} |")
    L.append("")
    L.append(f"control: prefill step {_fmt(ctrl_pre)} ms, decode step {_fmt(ctrl_dec)} ms.  'Δ vs control decode' = extra wall per decode step of the arm (acquisition + commit), the serving-side cost.  "
             "Host arm: D2H / leaf hash / tree are host wall-clock sections inside the synchronous commit; torch arms: 'commit enqueue' is the CPU time to enqueue the GPU hashing (GPU-side time is inside step total, measured with CUDA events).")
    out = Path(dirs[0], "components.md")
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1:] or ["out/gen/r9/cmt-ref/bench"])
