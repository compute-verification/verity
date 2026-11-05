import itertools, json, os, random, statistics, sys
DATA = os.environ.get("LANDING_STUDY_DIR", "/tmp/ld")
sys.argv=['x']
import importlib.util
spec=importlib.util.spec_from_file_location('sim',os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sim.py')); sim=importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)
NOWM=(sim.NOW-sim.START)/60
def obs_like(prs):
    lat=[(p.main-p.r0)/60 for p in prs if p.main is not None and p.main<=NOWM and 0<=p.r0<6.5*1440]
    return sim.q(lat,.5), sim.q(lat,.9), len(lat)
def c2m(prs):
    x=[(p.main-p.landed)/60 for p in prs if p.main is not None and p.landed is not None and 0<=p.landed<6.5*1440]
    return sim.q(x,.5), sim.q(x,.9)
res=[]
for admit,pb,fix,flake,runm,p0 in itertools.product([1,2,3,4],[0.01,0.02,0.04],[180,360],[0.1,0.3],[120,150],[15,30,40]):
    params=dict(sim.CURRENT, admit=admit, fix_min=fix, flake=flake, run_min=runm, pass0=p0)
    prs,moves,d=sim.run('current',params,pb)
    p50,p90,n=obs_like(prs)
    m=len([x for x in moves if 0<=x<6.5*1440])/6.5
    d=d[1]; pend=statistics.mean(x[0] for x in d); nx=statistics.mean(x[1] for x in d); ck=statistics.mean(x[2] for x in d)
    c50,c90=c2m(prs)
    # observed: landed-only p50 37.7 p90 94.4 n=413; moves 2.4/day; checked->main p50 11.8 p90 25.2; pending ~250-450; next ~50-90; checked ~30-40
    err=abs(p50-37.7)/37.7+abs(p90-94.4)/94.4+abs(m-2.4)/2.4+abs(c50-11.8)/11.8+abs(n-413)/413
    res.append((err,admit,p0,pb,fix,flake,runm,round(p50,1),round(p90,1),n,round(m,1),round(c50,1),round(c90,1),round(pend),round(nx),round(ck)))
res.sort()
print("err admit pass0 p_break fix flake run | p50 p90 n moves/d c2m50 c2m90 pending next checked")
for r in res[:12]: print(r)
json.dump(res,open('/tmp/ld/calib.json','w'))
