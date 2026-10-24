"""The prompt-split negative on real records: `negatives.py advice` on each window of 6d6a's advice (13 windows, the
committing endpoint's requests.jsonl) and on 1042's w5, at the link's 4096-byte frames, with IoTieRequest's reference
(`io_request`) tying each advice to the certified ingress of its id requests. The inputs come in
$RESEARCH_RUN_DIR/inputs/prompt-split-inputs.tgz; each window's JSON and summary.json land in $OUT."""
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

OUT = Path(os.environ["OUT"])
INPUTS = Path(os.environ["RESEARCH_RUN_DIR"]) / "inputs" / "prompt-split-inputs.tgz"
MODEL = "unsloth/Llama-3.2-1B"


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="prompt-split-"))
    with tarfile.open(INPUTS) as tf:
        tf.extractall(work, filter="data")
    summary, ok = {"model": MODEL, "frame_bytes": 4096, "T": 600, "R": 64, "shift": 1, "windows": {}}, True
    for run in ("6d6a", "1042"):
        ids = work / run / "requests.jsonl"
        for w in sorted((p for p in (work / run).iterdir() if p.is_dir()), key=lambda p: int(p.name[1:])):
            p = subprocess.run([sys.executable, "benchmarks/network_traces/negatives.py", "advice", "--advice",
                                str(w / "advice.bin"), "--ids", str(ids), "--T", "600", "--R", "64", "--shift", "1",
                                "--model", MODEL], capture_output=True, text=True)
            name = f"{run}-{w.name}"
            (OUT / f"{name}.json").write_text(p.stdout)
            doc = json.loads(p.stdout) if p.stdout.strip() else {}
            neg = doc.get("negatives") or {}
            split, honest = neg.get("prompt-split") or {}, neg.get("honest") or {}
            row = {"exit": p.returncode, "stderr": p.stderr[-1500:], "requests": doc.get("requests"),
                   "honest": {k: honest.get(k) for k in ("codes", "request_tie", "ok")},
                   "prompt_split": {k: split.get(k) for k in ("ran", "codes", "request_tie", "untied", "prompt_lens",
                                                              "ok", "why")},
                   "io_tie_prompt_split": {k: ((doc.get("tie") or {}).get("prompt-split") or {}).get(k)
                                           for k in ("ran", "codes", "ok")},
                   "others_ok": all(r.get("ok") is not False for k, r in neg.items() if isinstance(r, dict))}
            summary["windows"][name] = row
            ok &= p.returncode == 0 and honest.get("ok") is True and split.get("ok") is not False
            print(f"{name}: exit {p.returncode} honest {row['honest']} prompt-split {row['prompt_split']} "
                  f"io-tie {row['io_tie_prompt_split']}", flush=True)
    ran = [k for k, r in summary["windows"].items() if r["prompt_split"]["ran"]]
    summary["prompt_split_ran"] = ran
    summary["prompt_split_refused_by_io_tie_request"] = [
        k for k in ran if "IoTieRequest" in (summary["windows"][k]["prompt_split"]["codes"] or [])]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(f"prompt-split ran on {len(ran)} windows, refused by IoTieRequest on "
          f"{len(summary['prompt_split_refused_by_io_tie_request'])}; all ok: {ok}", flush=True)
    return 0 if ok and ran else 1


if __name__ == "__main__":
    sys.exit(main())
