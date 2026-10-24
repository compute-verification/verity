"""List A2's and A3's finished checkpoints on a node and print the CKPTS lines run.sh's ckpt mode reads, grouped by base model.

  python ckpts.py [--host IP] [--seen FILE]    -> JSON {base: "DIR NAME TEXT\\n..."} of checkpoints with a DONE file not in FILE

TEXT is the eval text of the checkpoint's task: gsm8k.txt or math.txt for A2, cipherA|B|C.txt for A3 (eval_text.py).
"""
import argparse, json, os, subprocess

LIST = r"""for d in /workspace/jobs/tvi/ckpt/A2/*/step*/ /workspace/jobs/tvi/ckpt/A3/*/step*/; do
  [ -f "$d/DONE" ] && printf '%s\t%s\n' "$d" "$(tr -d '\n' < "$d/meta.json" 2>/dev/null)"; done; true"""


def ssh(host, cmd):
    from research.remote import ssh_argv
    return subprocess.run([*ssh_argv(host, 22, user="research"), cmd], capture_output=True, text=True, check=True).stdout


def text_for(fam, meta):
    if fam == "A2":
        return "math.txt" if "math" in str(meta.get("task", "")).lower() else "gsm8k.txt"
    order = meta.get("order")
    if order is None:
        order = "ABC".index(str(meta.get("task", "A"))[0]) if str(meta.get("task", "A"))[0] in "ABC" else 0
    return f"cipher{'ABC'[int(order)]}.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="81.85.2.180")
    ap.add_argument("--seen", default="/tmp/b1/seen.txt")
    a = ap.parse_args()
    seen = set(open(a.seen).read().split()) if os.path.exists(a.seen) else set()
    groups = {}
    for line in ssh(a.host, LIST).splitlines():
        d, _, js = line.partition("\t")
        d = d.rstrip("/")
        try:
            meta = json.loads(js) if js else {}
        except json.JSONDecodeError:
            meta = {}
        fam = "A2" if "/A2/" in d else "A3"
        name = f"{fam}_{d.split('/')[-2]}_{d.split('/')[-1]}"
        if name in seen:
            continue
        base = meta.get("model") or "Qwen/Qwen2.5-1.5B"
        groups.setdefault(base, []).append(f"{d} {name} {text_for(fam, meta)}")
    print(json.dumps({b: "\n".join(v) for b, v in groups.items()}, indent=1))


if __name__ == "__main__":
    main()
