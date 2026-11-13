import json, sys, glob
pairs = [("r20261113-033036-b53c","r20261113-170146-0752"),("r20261113-033106-2ef2","r20261113-170216-6bc5"),
         ("r20261113-033134-59d5","r20261113-170247-e1da"),("r20261113-033148-defe","r20261113-170320-a629"),
         ("r20261113-033217-fd7e","r20261113-170350-6d72"),("r20261113-033248-273e","r20261113-170421-24ba"),
         ("r20261113-033320-3992","r20261113-170451-7688")]
F = ("workload","sites","combines","units","templates","template_ids","wires","unwired","proof_blocks","backward","error")
R = "/home/ubuntu/.research/runs/"
for old, new in pairs:
    o = glob.glob(R+old+"/out/collect_*.json"); n = glob.glob(R+new+"/out/collect_*.json")
    if not n: print(new, "no result yet"); continue
    a, b = json.load(open(o[0])), json.load(open(n[0]))
    diff = [f for f in F if a.get(f) != b.get(f)]
    print(new, n[0].split('/')[-1], "vs", old, "differs in", diff or "nothing",
          "| sites", sum(b.get("sites") or [0]), "wires", b.get("wires"), "unwired", b.get("unwired"), "pb", str(b.get("proof_blocks"))[:40],
          "backward", b.get("backward"), "err", str(b.get("error"))[:200], "wall", b.get("wall_s"))
