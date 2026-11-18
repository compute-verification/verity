# scratch (lean/WIRED): proofs' edges' wires on one class, by root node: per part, per member or per row
import bisect, collections, gzip, json, sys, time
sys.path.insert(0, sys.argv[1])
import inference_only_record_reference as R

d = sys.argv[2]
desc = json.loads(gzip.open(f"{d}/program.json.gz", "rt").read())
rec = json.loads(gzip.open(f"{d}/circuit.json.gz", "rt").read())
tids = [t for t, _ in rec["templates"]]
t0 = time.time()
wires = R.program_wires(desc, tids, rec.get("cuts", {}))
print(f"PYW {len(wires)} wires in {time.time() - t0:.1f} s", flush=True)
nodes = desc["definitions"][desc["root"]]["body"]["nodes"]
ranges, at = collections.defaultdict(list), collections.Counter()
for k, x in enumerate(nodes):
    tid = desc["callees"][x["fn"]]
    if tid in set(tids):
        n = x["n"] if x["form"] == "batch" else 1
        ranges[tid].append((at[tid], k, n)); at[tid] += n
by_node, by_port, units = collections.Counter(), collections.Counter(), collections.Counter()
for w in wires:
    rs = ranges[w["reader"]]
    i = bisect.bisect_right([r[0] for r in rs], w["at"]) - 1
    k = rs[i][1]
    by_node[k] += 1; by_port[(k, w["port"])] += 1
    import math
    units[k] += math.prod(w["radix"])
print(f"PYW root nodes with wires: {len(by_node)} of {len(nodes)}; wires per node, most common: "
      f"{collections.Counter(by_node.values()).most_common(8)}", flush=True)
for k, n in by_node.most_common(6):
    x = nodes[k]; tid = desc["callees"][x["fn"]]
    ports = {p: c for (kk, p), c in by_port.items() if kk == k}
    print(f"PYW node {k} {tid[:60]} form={x['form']} members={x.get('n', 1)}: {n} wires, reader units covered {units[k]}, "
          f"by port {ports}; cuts {rec.get('cuts', {}).get(tid, {})}", flush=True)
radix = collections.Counter(tuple(w["radix"]) for w in wires)
print(f"PYW wires by radix (units each wire covers), most common: {radix.most_common(8)}", flush=True)
