import importlib.util, json, sys, collections
from pathlib import Path
LEAN = Path(__import__("os").environ.get("ROOT","/tmp/wif") + "/verity/protocols/workload_classification/training_detection/inference_only/verifier/lean")
def load(n):
    s = importlib.util.spec_from_file_location(n, LEAN/"scripts"/f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
vectors = load("record_difftest_vectors"); ref = vectors.ref; R = ref.reference
shaped = {m["shape"]: m for m in vectors.shaped()}
for size, blocks in (("2x3", 2), ("4x35", 4)):
    case = shaped[f"{size} registered"]["case"]; text = case["instance"]["program"]
    form = vectors.results(case)["detail"]
    inp, _, _ = R.root_nodes(text)
    whole = R.check_desc({"program": text, "theta": [], "owner": ["input" if i else 0 for i in inp], "gamma": form["gates"]*blocks, "alpha": [vectors.BIG, 1]})
    width, is_input, _, reads = R.circuit(text)
    read = set().union(*reads)
    ins = [g for g in range(len(width)) if is_input[g]]
    unread = sum(width[g] for g in ins if g not in read)
    print(size, "desc ius", whole["ius"], "form inputs", form["inputs"], "x blocks", form["inputs"]*blocks,
          "sum input widths", sum(width[g] for g in ins), "unread", unread, "read input bits by circuit()", sum(width[g] for g in ins if g in read))
    print("  widths of input gates:", collections.Counter(width[g] for g in ins), "n input gates", len(ins))
    print("  form keys:", {k: v for k, v in form.items() if not isinstance(v, (list, dict))})
    print("  root_nodes inp:", inp)
    prog = json.loads(text) if isinstance(text, str) else text
    if size == "2x3": json.dump(prog, open("/tmp/ioread/prog2x3.json", "w"))
