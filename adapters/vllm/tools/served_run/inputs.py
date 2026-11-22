"""The launchers' command line of the adapter's input datasets (`verity_vllm.inputs`): its statement, its door and its one
sender.

    inputs.py statement                  # MODEL REVISION PRECISION CLASS CAP_BITS, one line
    inputs.py door NAME [TIMING]         # the door over the input dataset NAME at TIMING (default at-once): its identity
    inputs.py application FILE           # MODEL REVISION PRECISION CLASS DATASET TIMING CAP_BITS of applications/NAME.toml, then the door's line
    inputs.py tokenize --port P --dataset NAME     # the dataset's prompts, decoded, against the file's and the endpoint's ids
    inputs.py send --port P --dataset NAME --timing T [--release RUN] --out FILE    # the one sender

A refusal prints ``REFUSED NAME: TEXT`` (``[send] REFUSED`` from the sender) and exits 3, before anything is sent.
"""
from __future__ import annotations

import argparse

import input_datasets
from verity_vllm.inputs import sender, serves
from verity_vllm.pipeline.advice_windows import CAP_BITS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("statement")
    door = sub.add_parser("door")
    door.add_argument("dataset")
    door.add_argument("timing", nargs="?", default="at-once")
    sub.add_parser("application").add_argument("file")
    send = sub.add_parser("send")
    send.add_argument("--port", type=int, required=True)
    send.add_argument("--dataset", required=True, help="the input dataset's name (input_datasets.open)")
    send.add_argument("--timing", required=True, help="at-once, or rate:R requests a second")
    send.add_argument("--release", default=None, help="number the requests in order under this release run id")
    send.add_argument("--out", required=True)
    tok = sub.add_parser("tokenize")
    tok.add_argument("--port", type=int, required=True)
    tok.add_argument("--dataset", required=True)
    a = ap.parse_args(argv)
    s = serves.SERVES
    if a.command == "tokenize":
        return tokenize(a.port, a.dataset, s)
    if a.command == "statement":
        print(s.model, s.revision, s.precision, s.window_class, CAP_BITS)
        return 0
    if a.command == "send":
        return sender.send(a.port, a.dataset, a.timing, release=a.release, out=a.out)
    try:
        if a.command == "application":
            name, timing = serves.application(a.file)
            door_said = serves.door_line(name, timing)
            said = [" ".join((s.model, s.revision, s.precision, s.window_class, name, timing, str(CAP_BITS))), door_said]
        else:
            said = [serves.door_line(a.dataset, a.timing)]
    except (serves.Refused, input_datasets.Refused) as e:
        print("REFUSED {}: {}".format(*serves.refusal(e)))
        return 3
    print("\n".join(said))
    return 0


def tokenize(port: int, dataset: str, s: serves.Serves) -> int:
    """Each distinct prompt of ``dataset`` decoded by the tokenizer file at the statement's revision, then encoded by that
    file and by the endpoint's ``/tokenize``, each without special tokens added, against the dataset's ids: 0 when all are
    equal, 1 at the first difference."""
    import json
    import urllib.request

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(s.model, revision=s.revision, local_files_only=True)
    ds = input_datasets.open(dataset)
    prompts = sorted({tuple(b["prompt"]) for b in serves.door(ds.kind, (b for _, b in ds.timed(*serves.timing("at-once")[:1], **serves.timing("at-once")[1])))})
    for i, ids in enumerate(prompts):
        text = tk.decode(list(ids))
        body = json.dumps({"prompt": text, "add_special_tokens": False}).encode()
        with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", data=body,
                                                           headers={"Content-Type": "application/json"}), timeout=60) as r:
            live = json.load(r)["tokens"]
        for who, got in (("tokenizer file", tk.encode(text, add_special_tokens=False)), ("endpoint /tokenize", live)):
            if list(got) != list(ids):
                j = next((k for k, (x, y) in enumerate(zip(got, ids)) if x != y), min(len(got), len(ids)))
                print(f"TOKENIZE DIFFERS prompt {i} ({len(ids)} ids): the {who} gives {list(got)[j:j + 4]} at {j}, "
                      f"the definition {list(ids)[j:j + 4]}")
                return 1
    print(f"TOKENIZE EQUAL: {len(prompts)} distinct prompts of {dataset}, decoded by {s.model}'s tokenizer file at "
          f"{s.revision[:12]}, encode to the definition's ids by that file and by the endpoint's /tokenize")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
