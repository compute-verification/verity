"""make_types.py COVER.json: CIR-272's lists from CIR-271's record (`circuits/request-cover/v2`): every card type whose form is
not ``bound``, by blocks, largest first, each staged as ``census:NAME`` at one block with ``session``; then the units of the
card's five places (the record's rows not on the card). Writes, beside this file:

    types-first.tsv  FIRST's types: the largest unbound type and three small ones, to show the tree, builds and verdict files
    types-a.tsv      the rest dealt alternately, A first, then the extra units the same way
    types-b.tsv
    types.json       every tag: its block, blocks, form, k_log, list (first, a or b), rank, extra"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
FIRST = ['MoeExpertFp8GemmW_v1{E=256,K=256,N=1,G=128,KB=128,SWAP=false,DOT={"fn":"HopperE4m3MmaV2Dot32_v1"}}',
         'BatchedGemm_v1{B=1,K=192,N=1,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}',
         'SparseMlaHeadEnd_v1{DV=64,INV={"fn":"Fa3InvSum_v1"}}',
         "ResidualAdd_v1{N=384}"]

cover = json.loads(Path(sys.argv[1]).read_text())
assert cover["schema"] == "circuits/request-cover/v2", cover["schema"]
blocks = cover["blocks"]
todo = sorted((b for b in blocks if b["on_card"] and b["form"] != "bound"), key=lambda b: (-b["blocks"], b["block"]))
extra = [b for b in blocks if not b["on_card"]]
assert (len(todo), sum(b["blocks"] for b in todo)) == (240, 2998736728), (len(todo), sum(b["blocks"] for b in todo))
assert len(extra) == 18 and all(n in {b["block"] for b in todo} for n in FIRST)
for b in todo + extra:
    assert "\t" not in b["block"] and " " not in b["block"], b["block"]

index, lists = {}, {"first": [], "a": [], "b": []}
rest = [b for b in todo if b["block"] not in FIRST]
for rank, b in enumerate(todo, 1):
    tag = f"t{rank:03d}"
    where = "first" if b["block"] in FIRST else "ab"[rest.index(b) % 2]
    lists[where].append(tag)
    index[tag] = {"block": b["block"], "blocks": b["blocks"], "form": b["form"], "accepted": b["accepted"], "k_log": b["k_log"],
                  "list": where, "rank": rank, "extra": False}
for i, b in enumerate(extra, 1):
    tag = f"x{i:02d}"
    where = "ab"[(i - 1) % 2]
    lists[where].append(tag)
    index[tag] = {"block": b["block"], "blocks": None, "form": b["form"], "accepted": b["accepted"], "k_log": b["k_log"],
                  "list": where, "rank": None, "extra": True}
for where, tags in lists.items():
    (HERE / f"types-{where}.tsv").write_text("".join(f"{t}\tcensus:{index[t]['block']}\t1\tsession\n" for t in tags))
(HERE / "types.json").write_text(json.dumps(index, indent=1) + "\n")
print({w: len(t) for w, t in lists.items()}, {w: sum(index[t]["blocks"] or 0 for t in tags) for w, tags in lists.items()})
