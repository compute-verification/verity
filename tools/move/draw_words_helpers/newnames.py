import sys,re,subprocess,collections
tree,tip=sys.argv[1],sys.argv[2]
d=subprocess.run(['git','-C',tree,'diff','--cached','-U0','--','*.py','*.rs','*.sh','*.toml'],capture_output=True,text=True).stdout
ID=re.compile(r'\b[A-Za-z_][A-Za-z0-9_]*\b')
old=collections.Counter();new=collections.Counter()
for l in d.splitlines():
    if l.startswith(('+++','---')): continue
    if l.startswith('-'): old.update(t for t in ID.findall(l) if re.search(r'(?i)draw|drew',t))
    if l.startswith('+'): new.update(ID.findall(l))
added=set()
for l in d.splitlines():
    if l.startswith('+') and not l.startswith('+++'):
        pass
oldtoks=set()
for l in d.splitlines():
    if l.startswith('-') and not l.startswith('---'): oldtoks.update(ID.findall(l))
cand=[t for t in new if t not in oldtoks and re.search(r'(?i)sampl|design|flip|deriv|generat',t)]
res=[]
for t in sorted(cand):
    r=subprocess.run(['git','-C',tip,'grep','-l','-w',t,'--','*.py'],capture_output=True,text=True).stdout.split()
    res.append((t,len(r),r[:3]))
print("old names (contain draw):",len(old))
for t,n,r in res: print(f"{t}\t{n}\t{' '.join(r)}")
