import time
from research.store.local import LocalStore
from research.store import labels_sync, ids
from research.store.attempt import store_root
st = LocalStore(store_root(None))
r = st._require_remote()
t=time.time()
d = ids.target_dirname("pr:")
listing = {}
for k in r.list(f"labels/{d}"):
    if m := labels_sync.KEY_RE.match(k):
        listing.setdefault("pr:" + m.group(1)[len(d):], {})[m.group(2)] = k
print(len(listing), "targets")
rep = labels_sync._pull(st, r, sorted(listing.items()), jobs=128)
print(len(rep["pulled"]), "pulled", len(rep["bad"]), "bad", time.time()-t)
