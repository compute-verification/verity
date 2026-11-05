import sys, json, time, concurrent.futures as cf
sys.path.insert(0,'/agent/repos/verity/tools/research/src')
from research.store.remote import load_remote
r = load_remote()
print(type(r).__name__, file=sys.stderr)
t=time.time()
keys = [k for k in r.list("labels/pr_")]
print(len(keys), "keys", time.time()-t, file=sys.stderr)
json.dump(keys, open('/tmp/trace/label_keys.json','w'))
