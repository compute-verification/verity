import pickle, sys, tomllib, posixpath
from collections import Counter
sys.path.insert(0, "move_main")
import rename
import resolve_check as R
name = sys.argv[1]
snaps, origins = pickle.load(open("main.pkl", "rb"))
S3 = snaps[3]
w = rename.Words(tomllib.loads(open(f"maps/period_proofs_map.{name}.toml").read()))
files = list(S3)
assert not w.unmatched(files), w.unmatched(files)
w.hold(files)
moves = {p: n for p in files if not w.excluded(p) and (n := w.path(p)) != p}
S4, counts, changed = {}, Counter(), set()
for p, t in S3.items():
    n = t
    if not w.excluded(p):
        lock = posixpath.basename(p) == rename.POLICY
        n = rename.resorted(t, w.sub(t, p, rename.record_spans(t) if lock else [], counts, Counter()), p)
    q = moves.get(p, p)
    S4[q] = n
    if n != t or q != p:
        changed.add(q)
SNAP = {"a": S3, "b": S4}
R.texts = lambda root, commit, paths: {p: SNAP[commit][p] for p in paths if p in SNAP[commit]}
R.lean_files = lambda root, commit: [p for p in SNAP[commit] if p.endswith(".lean")]
R.git = lambda root, *args: "\0".join(SNAP[args[-1]].keys())
R.changes = lambda root, base, head: (moves, changed)
issues, compared, nfiles = R.check(None, "a", "b")
out = [f"variant {name}: files changed {len(changed)}, moved {len(moves)}, words {sum(counts.values())}; resolve check: {compared} of {nfiles} compared, {len(issues)} issue(s)"]
out += ["   " + R.said(i) for i in issues]
dbl = sum(t.count("onitoringMonitoring") for t in S4.values())
out.append(f"   doubled word in the result: {dbl}; periodRun renamed {counts[('periodRun', 'monitoringPeriodRun')]}; sessionTreeReason renamed {counts[('sessionTreeReason', 'challengePhaseTreeReason')]}; PeriodGood renamed {counts[('PeriodGood', 'MonitoringPeriodGood')]}")
o = "security/Proofs/Properties/Verity/Sampled/MemberForm/Order.lean"
out.append(f"   Order.lean changed: {S3[o] != S4[o]}; Pous/Idle changed: {[p for p in ('security/Proofs/Pous/PeriodDeployed.lean', 'security/Properties/Pous/PeriodDeployed.lean', 'security/Properties/Verity/Idle.lean') if S3[p] != S4[p]]}")
open(f"variant.{name}.txt", "w").write("\n".join(out) + "\n")
