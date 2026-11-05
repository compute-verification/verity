import sys, json, time, os, tempfile, concurrent.futures as cf
from pathlib import Path
sys.path.insert(0,'/agent/repos/verity/tools/research/src')
from research.store.remote import load_remote
r = load_remote()
keys = json.load(open('/tmp/trace/label_keys.json'))
d = Path('/tmp/trace/labels'); d.mkdir(exist_ok=True)
def one(k):
    p = d / k.replace('/','__')
    if p.exists(): return 0
    for i in range(4):
        try:
            r.get(k, p); return 0
        except Exception as e:
            time.sleep(1+i)
    return 1
t=time.time()
with cf.ThreadPoolExecutor(64) as ex:
    fails = sum(ex.map(one, keys))
print("fails", fails, time.time()-t)
out=[]
for p in d.iterdir():
    try: out.append(json.loads(p.read_text()))
    except Exception: pass
json.dump(out, open('/tmp/trace/labels_all.json','w'))
print(len(out))
