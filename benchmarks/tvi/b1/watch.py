"""Analyze A2's and A3's checkpoints as they finish: scan the node for step dirs with a DONE file, claim each (an atomic mkdir
under CLAIM, so parallel watchers never repeat one), run counted.analyze against the checkpoint's base model on its task's
eval text, and write counted_NAME.json to $OUT. Exits after IDLE_MIN minutes with nothing new, or at DEADLINE (UTC ISO).

  python watch.py      Env: OUT, CLAIM (/workspace/jobs/tvi/b1/claimed), DEADLINE, IDLE_MIN (20), TEXTS (dir of *.txt),
                       WATCH_ROOTS, WATCH_STEPS (glob of step dirs, default step*), TEXT_OVERRIDE
"""
import datetime, glob, json, os, sys, time, traceback
import torch
import counted
from ckpts import text_for

ROOTS = os.environ.get("WATCH_ROOTS", "/workspace/jobs/tvi/ckpt/A2 /workspace/jobs/tvi/ckpt/A3").split()
STEPS = os.environ.get("WATCH_STEPS", "step*")


def scan(claim):
    out = []
    for root in ROOTS:
        for d in sorted(glob.glob(os.path.join(root, "*", STEPS))):
            if not os.path.exists(os.path.join(d, "DONE")):
                continue
            fam = "A2" if "/A2/" in d else "A3"
            name = f"{fam}_{d.split('/')[-2]}_{d.split('/')[-1]}"
            try:
                os.mkdir(os.path.join(claim, name))
            except FileExistsError:
                continue
            try:
                meta = json.load(open(os.path.join(d, "meta.json")))
            except (OSError, json.JSONDecodeError):
                meta = {}
            text = os.environ.get("TEXT_OVERRIDE") or text_for(fam, meta)
            out.append((d, name, meta.get("model") or "Qwen/Qwen2.5-1.5B", text, meta.get("S_bits") or float("inf")))
    return out


def main():
    claim = os.environ.get("CLAIM", "/workspace/jobs/tvi/b1/claimed")
    os.makedirs(claim, exist_ok=True)
    deadline = datetime.datetime.fromisoformat(os.environ["DEADLINE"].replace("Z", "+00:00"))
    idle_min = float(os.environ.get("IDLE_MIN", "20"))
    texts = os.environ.get("TEXTS", ".")
    nseq, L = int(os.environ.get("NSEQ", "4")), int(os.environ.get("SEQ", "512"))
    cur = {"base": None}
    last_new = time.monotonic()
    while datetime.datetime.now(datetime.timezone.utc) < deadline:
        todo = scan(claim)
        if not todo:
            if time.monotonic() - last_new > 60 * idle_min:
                print(f"idle {idle_min} min: done", flush=True)
                break
            time.sleep(60)
            continue
        last_new = time.monotonic()
        todo.sort(key=lambda t: (t[2], t[4]))  # smallest S first: the cliff is at small S
        for d, name, base, text, _ in todo:
            try:
                if cur["base"] != base:
                    cur.clear()
                    if counted.DEV == "cuda":
                        torch.cuda.empty_cache()
                    cur.update(base=base, model=counted.load_base(base).to(counted.DEV), ids={})
                    print(f"base {base}", flush=True)
                if text not in cur["ids"]:
                    os.environ["TOKENIZER"] = base
                    ids = counted.eval_ids(os.path.join(texts, text), nseq, L).to(counted.DEV)
                    cur["ids"][text] = (ids, counted.probe_local(cur["model"], ids))
                ids, local = cur["ids"][text]
                res = counted.analyze(base, cur["model"], ids, local, d, name, {"eval_text": text})
                s = res["summary"]
                print(f"{name}: {res['secs']:.0f}s loss {res['loss']['theta']:.3f}->{res['loss']['theta1']:.3f} "
                      f"kl {res['loss']['kl_theta1_theta']:.4f} "
                      + " ".join(f"{f}: bv={s[f]['beta_v']:.3f} bx={s[f]['beta_x']:.3f} bw={s[f]['beta_w']:.3f} "
                                 f"Fmin={s[f]['F_fwd_min']:.0f}" for f in counted.FMTS), flush=True)
                json.dump(res, open(os.path.join(os.environ["OUT"], f"counted_{name}.json"), "w"), indent=1, default=float)
            except Exception as e:  # noqa: BLE001  one bad checkpoint must not stop the watcher
                traceback.print_exc()
                print(f"FAILED {name}: {type(e).__name__}: {e}", flush=True)
    print("watch: exit", flush=True)


if __name__ == "__main__":
    sys.exit(main())
