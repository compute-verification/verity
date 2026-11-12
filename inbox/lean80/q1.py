import pickle, re, sys
sys.path.insert(0, "move_main")
import resolve_check as R
snaps, origins = pickle.load(open("main.pkl", "rb"))
S0, S3, S4, S6 = snaps[0], snaps[3], snaps[4], snaps[6]
o4, o6 = origins[4], origins[6]
# declarations at main, and their names after stage 4 and after stage 6 (paired in order within a file)
inv4 = {o: p for p, o in o4.items()}; inv6 = {o: p for p, o in o6.items()}
rows = []
for p, t in S0.items():
    if not p.endswith(".lean"): continue
    d0 = R.Scan(t).decls
    if not d0: continue
    d3 = R.Scan(S3[origins[3] and {o: q for q, o in origins[3].items()}[p]]).decls
    d4 = R.Scan(S4[inv4[p]]).decls
    d6 = R.Scan(S6[inv6[p]]).decls
    assert len(d0) == len(d4) == len(d6) == len(d3), p
    for a, b3, b, c in zip(d0, d3, d4, d6):
        rows.append((p, a, b3, b, c))
pickle.dump(rows, open("decls.pkl", "wb"))
print(len(rows), "declarations at main")
