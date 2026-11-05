"""How long each re-head labelled in the week waited, from its commit, for a ready mark and a land verdict, when the head
before it had one: carry-able re-heads (own change unchanged net of what they brought in, carry_net.py) against edits.
Reads labels.json, reheads_net.csv and newheads_ct.txt (`git log --no-walk --stdin --format='%H %cI'` of the new heads)
from the working directory.

    python3 rewait.py
"""
import csv, json, collections as C, sys
from datetime import datetime
def ts(s): return datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
labels=json.load(open('labels.json'))
lab=C.defaultdict(list)
for n,xs in labels.items():
    for x in xs:
        lab[x['target'].split('@',1)[1][:12]].append(x)
ct={l.split()[0][:12]:ts(l.split()[1]) for l in open('newheads_ct.txt')}
rows=list(csv.DictReader(open('reheads_net.csv')))
W0,W1=ts('2026-10-29T17:00:00Z'),ts('2026-11-05T17:00:00Z')
def carry(r): return r['net']=='unchanged' or (r['net']=='' and r['kind'] in ('empty','sync','sync+regen'))
sel=[r for r in rows if r['new_first_label'] and W0<=ts(r['new_first_label'])<W1 and r['new'] in ct]
def q(xs,p):
    xs=sorted(xs); return xs[int(p*(len(xs)-1))] if xs else float('nan')
res=C.defaultdict(list)
for r in sel:
    h=r['new']; c=ct[h]; xs=lab.get(h,[])
    mk=[ts(x['ts']) for x in xs if x['key']=='ready' and x['value'] is True]
    vd=[ts(x['ts']) for x in xs if x['key']=='reviewed' and x['value']=='land']
    k='carry' if carry(r) else 'edit'
    if 'verdict' in r['old_had'] and vd: res[k+' verdict wait (old head had one)'].append(max(0,(min(vd)-c)/3600))
    if 'mark' in r['old_had'] and mk: res[k+' mark wait (old head had one)'].append(max(0,(min(mk)-c)/3600))
    if 'verdict' in r['old_had'] and vd and mk: res[k+' mark->verdict on new head'].append(max(0,(min(vd)-min(mk))/3600))
print('re-heads labelled in week',len(sel),'carry-able',sum(carry(r) for r in sel))
for k,v in sorted(res.items()):
    print(f'{k:45s} n={len(v):4d} p50={q(v,.5):6.2f} p75={q(v,.75):6.2f} p90={q(v,.9):6.2f} sum={sum(v):7.0f} h')
